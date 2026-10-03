"""Trading backtest: would buying the lots a model calls cheap make money?

Setup (the result of seven iterations of the original research, five of which
gave wrong answers; the pitfalls are listed in experiments/07_trading_backtest):

* A model trained on sales before the test period predicts every test sale.
  A *buy signal* fires when the sale price is below the prediction by a
  margin (10%, 15%): we pretend to be the buyer at that price.
* CSFloat purchases are trade-locked for 7 days. The resale value is the
  median of comparable sales in days 7..14 after the purchase: the same item
  name, float within +-0.01 (a wider tolerance mixes physically different
  items and flips the sign of the result), at least 3 comparables.
* Crafts cannot be compared one to one. Their resale is a "clean equivalent"
  (comparable sales with their own sticker value removed) plus
  ``beta(r) * addon`` of the bought item, where ``r = addon / anchor`` and
  ``beta`` - the share of sticker value that reaches the price - is fitted on
  the training period only.
* Profit = resale * (1 - 2% seller fee) - price. A control buys the same
  number of random lots from the same pool.
* Time on market (listing -> sale) matters: lots sold within minutes are
  taken by bots and are not available to a human buyer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TOM_EDGES = [0, 1 / 1440, 10 / 1440, 60 / 1440, 0.25, 1, 7, 1e9]
TOM_LABELS = ["<1 min", "1-10 min", "10-60 min", "1-6 h", "6-24 h", "1-7 d", ">7 d"]
ACCESS_MIN_DAYS = 10 / 1440


@dataclass(frozen=True)
class BacktestConfig:
    hold_days: int = 7  # trade ban
    resale_window_days: int = 7
    float_tolerance: float = 0.01
    min_comparables: int = 3
    fee: float = 0.02
    curve_bins: int = 40
    max_curve_share: float = 0.25  # crafts whose resale is mostly the curve are not measurable


def time_on_market_days(frame: pd.DataFrame) -> pd.Series:
    return (frame["event_time"] - frame["listed_at"]).dt.total_seconds() / 86400.0


def addon_cents(frame: pd.DataFrame) -> pd.Series:
    """Reference value of stickers and charms attached to the item."""
    stickers = np.expm1(frame["log_sticker_sum_ref_price"]).fillna(0)
    charms = np.expm1(frame["log_keychain_ref_price"]).fillna(0)
    return stickers + charms


def segment(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(frame["sticker_count"] > 0, "craft", np.where(frame["keychain_count"] > 0, "charm", "clean")),
        index=frame.index,
    )


def passthrough_curve(train: pd.DataFrame, bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """beta(r): median share of sticker value that shows up in the sale price."""
    addon = addon_cents(train)
    anchor = train["anchor_cents"]
    r = addon / anchor
    ok = (addon > 0) & (r > 0)
    beta = (train["price_cents"] - anchor)[ok] / addon[ok]
    lr = np.log10(r[ok])
    edges = np.unique(np.quantile(lr, np.linspace(0, 1, bins + 1)))
    groups = pd.DataFrame({"lr": lr, "beta": beta}).groupby(pd.cut(lr, edges, include_lowest=True), observed=True)
    curve = groups.median()
    return curve["lr"].to_numpy(), np.clip(curve["beta"].to_numpy(), 0.0, 1.0)


def apply_curve(r: np.ndarray, xs: np.ndarray, betas: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    out = np.zeros(r.shape)
    ok = np.isfinite(r) & (r > 0)
    out[ok] = np.interp(np.log10(r[ok]), xs, betas)
    return out


def resale_after_ban(test: pd.DataFrame, candidates: np.ndarray, config: BacktestConfig) -> pd.DataFrame:
    """Median / 25th percentile of comparable clean-equivalent sales after the trade ban."""
    test = test.reset_index(drop=True)
    n = len(test)
    median, p25, comps, curve_median = (np.full(n, np.nan) for _ in range(4))
    lo = np.timedelta64(config.hold_days, "D")
    hi = np.timedelta64(config.hold_days + config.resale_window_days, "D")
    for _, group in test.groupby("name", sort=False):
        idx = group.index.to_numpy()
        t = group["event_time"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
        order = np.argsort(t, kind="stable")
        idx, t = idx[order], t[order]
        f = group["float_value"].to_numpy(dtype=float)[order]
        base = group["base_equiv"].to_numpy(dtype=float)[order]
        curve = group["curve_part"].to_numpy(dtype=float)[order]
        for j in np.flatnonzero(candidates[idx]):
            a = np.searchsorted(t, t[j] + lo, "left")
            b = np.searchsorted(t, t[j] + hi, "right")
            if b - a < config.min_comparables:
                continue
            sel = np.abs(f[a:b] - f[j]) <= config.float_tolerance
            if sel.sum() < config.min_comparables:
                continue
            values = base[a:b][sel]
            median[idx[j]] = np.median(values)
            p25[idx[j]] = np.percentile(values, 25)
            comps[idx[j]] = sel.sum()
            curve_median[idx[j]] = np.median(curve[a:b][sel])
    out = test.copy()
    own = out["curve_part"].to_numpy()
    out["resale_med"] = median + own
    out["resale_p25"] = p25 + own
    out["n_comparables"] = comps
    # how much of the resale estimate comes from the curve rather than comparables
    out["curve_share"] = (own - curve_median) / pd.Series(median + own).replace(0, np.nan).to_numpy()
    return out


def _bootstrap(cost: np.ndarray, profit: np.ndarray, draws: int = 2000) -> tuple[float, float]:
    if len(cost) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(7)
    i = rng.integers(0, len(cost), size=(draws, len(cost)))
    roi = 100 * profit[i].sum(axis=1) / cost[i].sum(axis=1)
    return float(np.percentile(roi, 5)), float(np.percentile(roi, 95))


def strategy_row(trades: pd.DataFrame, pool: pd.DataFrame, resale: str, fee: float, days: float) -> dict:
    cost = trades["price_cents"].to_numpy(dtype=float)
    profit = trades[resale].to_numpy(dtype=float) * (1 - fee) - cost
    random_roi = np.nan
    if len(pool) >= 30 and len(trades) >= 5:
        rng = np.random.default_rng(42)
        rois = []
        for _ in range(20):
            pick = pool.iloc[rng.choice(len(pool), size=min(len(trades), len(pool)), replace=False)]
            c = pick["price_cents"].to_numpy(dtype=float)
            rois.append(100 * (pick[resale].to_numpy(dtype=float) * (1 - fee) - c).sum() / c.sum())
        random_roi = float(np.mean(rois))
    roi = 100 * profit.sum() / cost.sum() if len(cost) else np.nan
    low, high = _bootstrap(cost, profit)
    per_month = 30.0 / days
    return {
        "trades": int(len(trades)),
        "capital_usd": round(cost.sum() / 100),
        "profit_usd": round(profit.sum() / 100),
        "roi_pct": round(roi, 2),
        "roi_ci90": f"{low:.2f} .. {high:.2f}",
        "random_roi_pct": round(random_roi, 2),
        "edge_pp": round(roi - random_roi, 1) if np.isfinite(random_roi) else np.nan,
        "profitable_pct": round(100 * float((profit > 0).mean()), 1) if len(profit) else np.nan,
        "avg_ticket_usd": round(float(cost.mean()) / 100, 2) if len(cost) else np.nan,
        "profit_per_month_usd": round(profit.sum() / 100 * per_month),
    }


def signal_trades(pool: pd.DataFrame, signal: str, margin: float) -> pd.DataFrame:
    return pool[pool["price_cents"] < pool[signal] * (1 - margin)]
