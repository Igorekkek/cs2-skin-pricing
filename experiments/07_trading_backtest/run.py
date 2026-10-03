"""Does an accurate price model find profitable trades?

Re-implementation of the original 7-iteration trading study on the new
pipeline. A dedicated model is trained on everything before the last 21 days;
purchases happen in the first 7 days of that window, resale in days 7..14
after each purchase (after the trade ban). See src/skinprice/research/trading.py.

Usage: python experiments/07_trading_backtest/run.py [--force]
"""

import numpy as np
import pandas as pd

from skinprice.research import Experiment
from skinprice.research.experiment import to_markdown
from skinprice.research.trading import (
    ACCESS_MIN_DAYS,
    TOM_EDGES,
    TOM_LABELS,
    BacktestConfig,
    addon_cents,
    apply_curve,
    passthrough_curve,
    resale_after_ban,
    segment,
    signal_trades,
    strategy_row,
    time_on_market_days,
)
from skinprice.split import SplitConfig, assign_split

TEST_DAYS, BUY_DAYS, VAL_DAYS = 21, 7, 3
MARGINS = (0.10, 0.15)

exp = Experiment("07_trading_backtest")
features = exp.features()
end = features["event_time"].max()
test_start = end - pd.Timedelta(days=TEST_DAYS)
features["split"] = assign_split(
    features["event_time"],
    SplitConfig(test_start=str(test_start), val_start=str(test_start - pd.Timedelta(days=VAL_DAYS))),
)
exp.variant("model_trading_split", "premium_catboost", features, note="trained before the 21-day trading window")

config = BacktestConfig()
train = features[features["split"] == "train"]
test = features[features["split"] == "test"].merge(
    exp.predictions("model_trading_split")[["listing_id", "pred_cents"]], on="listing_id"
)
test = test.sort_values("event_time").reset_index(drop=True)

xs, betas = passthrough_curve(train, config.curve_bins)
addon = addon_cents(test)
test["curve_part"] = apply_curve((addon / test["anchor_cents"]).to_numpy(), xs, betas) * addon
test["base_equiv"] = test["price_cents"] - test["curve_part"]
candidates = (test["event_time"] < test_start + pd.Timedelta(days=BUY_DAYS)).to_numpy()


def build_pool(tolerance: float) -> pd.DataFrame:
    cfg = BacktestConfig(float_tolerance=tolerance)
    scored = resale_after_ban(test, candidates, cfg)
    pool = scored[candidates & scored["resale_med"].notna()].copy()
    pool["segment"] = segment(pool)
    pool["tom_days"] = time_on_market_days(pool)
    pool["tom_bucket"] = pd.cut(pool["tom_days"], TOM_EDGES, labels=TOM_LABELS, include_lowest=True)
    pool["reference_cents"] = pool["anchor_cents"]
    return scored, pool


scored, pool = build_pool(config.float_tolerance)
measurable = pool[(pool["segment"] == "clean") | ((pool["segment"] == "craft") & (pool["curve_share"] < config.max_curve_share))]

# 1. Is the resale estimator itself unbiased? (estimate / actual price on candidate lots)
bias_rows = []
cand = scored[candidates & scored["resale_med"].notna()]
for seg, group in cand.groupby(segment(cand)):
    ratio = group["resale_med"] / group["price_cents"]
    bias_rows.append({"segment": seg, "lots": len(group), "coverage_pct": round(100 * len(group) / candidates.sum(), 1),
                      "median_ratio": round(float(ratio.median()), 4), "mean_ratio": round(float(ratio.mean()), 4)})
bias = pd.DataFrame(bias_rows)

# 2. Money: segment x accessibility x signal x margin
rows = []
for seg in ("clean", "craft"):
    seg_pool = measurable[measurable["segment"] == seg]
    for scope, scope_pool in (("all lots", seg_pool), ("on sale >10 min", seg_pool[seg_pool["tom_days"] > ACCESS_MIN_DAYS])):
        for signal, label in (("pred_cents", "model"), ("reference_cents", "CSFloat reference")):
            for margin in MARGINS:
                trades = signal_trades(scope_pool, signal, margin)
                if len(trades) < 5:
                    continue
                rows.append({"segment": seg, "scope": scope, "signal": f"{label} @{margin:.0%}",
                             **strategy_row(trades, scope_pool, "resale_med", config.fee, BUY_DAYS)})
money = pd.DataFrame(rows)

# 3. Quick resale at the 25th percentile instead of the median (clean, model)
quick = []
clean = measurable[measurable["segment"] == "clean"]
for margin in MARGINS:
    trades = signal_trades(clean, "pred_cents", margin)
    quick.append({"signal": f"model @{margin:.0%}", **strategy_row(trades, clean, "resale_p25", config.fee, BUY_DAYS)})
quick = pd.DataFrame(quick)

# 4. Time on market: who buys the cheap lots?
tom_rows = []
for bucket, group in clean.groupby("tom_bucket", observed=True):
    trades = signal_trades(group, "pred_cents", 0.15)
    row = {"time_on_market": bucket, "lots": len(group), "share_pct": round(100 * len(group) / len(clean), 1),
           "median_price_to_model": round(float((group["price_cents"] / group["pred_cents"]).median()), 3)}
    if len(trades) >= 5:
        r = strategy_row(trades, group, "resale_med", config.fee, BUY_DAYS)
        row.update({k: r[k] for k in ("trades", "roi_pct", "random_roi_pct", "edge_pp")})
    tom_rows.append(row)
tom = pd.DataFrame(tom_rows)

# 5. Sensitivity: float tolerance 0.05 instead of 0.01
_, wide_pool = build_pool(0.05)
wide_clean = wide_pool[(wide_pool["segment"] == "clean") & (wide_pool["tom_days"] > ACCESS_MIN_DAYS)]
narrow_clean = clean[clean["tom_days"] > ACCESS_MIN_DAYS]
tolerance = pd.DataFrame([
    {"float_tolerance": tol, **strategy_row(signal_trades(p, "pred_cents", 0.10), p, "resale_med", config.fee, BUY_DAYS)}
    for tol, p in ((0.01, narrow_clean), (0.05, wide_clean))
])

tables = {"resale_estimator_bias": bias, "money": money, "quick_resale_p25": quick,
          "time_on_market_clean_model15": tom, "float_tolerance_clean_access_model10": tolerance}
exp.record("tables", {name: t.replace({np.nan: None}).to_dict("records") for name, t in tables.items()})
titles = {
    "resale_estimator_bias": "Проверка оценки перепродажи (оценка / фактическая цена)",
    "money": "Деньги",
    "quick_resale_p25": "Быстрая перепродажа по 25-му перцентилю сопоставимых продаж (обычные предметы)",
    "time_on_market_clean_model15": "Доходность в зависимости от того, сколько лот висел (обычные предметы, модель, скидка ≥15%)",
    "float_tolerance_clean_access_model10": "Допуск по float (обычные предметы, висели >10 мин, модель, скидка ≥10%)",
}
extra = "\n\n".join(f"#### {titles[name]}\n\n{to_markdown(t)}" for name, t in tables.items())
exp.report(extra=f"Тестовое окно {test_start:%Y-%m-%d} .. {end:%Y-%m-%d}; покупки в первые {BUY_DAYS} дней.\n\n" + extra)
