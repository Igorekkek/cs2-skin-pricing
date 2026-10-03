"""Market history features, strictly from earlier sales.

For a training row sold at time ``t`` every feature here uses only sales with
a timestamp strictly before ``t`` (sales sharing the same timestamp do not see
each other). At inference the same quantities come from a ``MarketSnapshot``:
the final state of the same accumulators over all history. ``tests/
test_history_parity.py`` checks that both paths give identical numbers.

Features:

* ``wt_recent_premium``    median log-premium of the last 200 sales of the skin
                            (weapon_type), global median as a fallback;
* ``log_wt_prior_sales``   how many times the skin sold before (liquidity);
* ``log_wt_sales_7d``      sales of the skin during the previous 7 days;
* ``seed_past_premium``    mean premium of earlier sales of the same skin+seed:
                            the only way to learn valuable patterns (blue gems,
                            fade %, Doppler phases) without an external catalog;
* ``sticker_past_premium_*``  what the same sticker ids added to the price before
                            (CSFloat's sticker reference price is often stale);
* ``name_recent_premium``  median price of the last 20 sales of the exact item
                            (with wear and StatTrak), relative to the anchor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from skinprice.features.static import MAX_STICKER_SLOTS

SKIN_WINDOW = 200
SKIN_MIN_HISTORY = 10
NAME_WINDOW = 20
LIQUIDITY_WINDOW = pd.Timedelta(days=7)

HISTORY_COLUMNS = [
    "wt_recent_premium",
    "log_wt_prior_sales",
    "log_wt_sales_7d",
    "seed_past_premium",
    "log_seed_past_seen",
    "sticker_past_premium_mean",
    "sticker_past_premium_max",
    "log_sticker_past_seen_min",
    "name_recent_premium",
    "log_name_prior_sales",
]


def to_ns(times: pd.Series) -> np.ndarray:
    """UTC timestamps as int64 nanoseconds (independent of pandas' datetime unit)."""
    return pd.to_datetime(times, utc=True).dt.as_unit("ns").astype("int64").to_numpy()


class Timeline:
    """Rows grouped by key and ordered by time, with 'strictly earlier' lookups."""

    def __init__(self, keys: np.ndarray, times_ns: np.ndarray):
        self.n = len(keys)
        codes, _ = pd.factorize(pd.Series(keys), sort=False)
        self.order = np.lexsort((np.arange(self.n), times_ns, codes))
        self.codes = codes[self.order]
        self.times = times_ns[self.order]
        boundaries = np.flatnonzero(np.diff(self.codes)) + 1
        self.starts = np.concatenate(([0], boundaries))
        self.ends = np.concatenate((boundaries, [self.n]))
        # prior[i]: number of rows of the same key strictly earlier than sorted row i.
        self.prior = np.empty(self.n, dtype=np.int64)
        self.group_start = np.empty(self.n, dtype=np.int64)
        for a, b in zip(self.starts, self.ends, strict=True):
            t = self.times[a:b]
            self.prior[a:b] = np.searchsorted(t, t, side="left")
            self.group_start[a:b] = a

    def _unsort(self, sorted_values: np.ndarray) -> np.ndarray:
        out = np.empty_like(sorted_values)
        out[self.order] = sorted_values
        return out

    def prior_count(self) -> np.ndarray:
        return self._unsort(self.prior.astype(float))

    def _lookup(self, per_row_inclusive: np.ndarray) -> np.ndarray:
        """Value of an inclusive running statistic at the last strictly earlier row."""
        idx = self.group_start + self.prior - 1
        result = np.where(self.prior > 0, per_row_inclusive[np.maximum(idx, 0)], np.nan)
        return self._unsort(result)

    def prior_sum(self, values: np.ndarray) -> np.ndarray:
        v = values[self.order]
        cumulative = pd.Series(v).groupby(self.codes).cumsum().to_numpy()
        out = self._lookup(cumulative)
        return np.where(np.isnan(out), 0.0, out)

    def prior_rolling_median(self, values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
        v = pd.Series(values[self.order])
        rolled = (
            v.groupby(self.codes)
            .rolling(window, min_periods=min_periods)
            .median()
            .reset_index(level=0, drop=True)
            .sort_index()
            .to_numpy()
        )
        return self._lookup(rolled)

    def prior_expanding_median(self, values: np.ndarray, min_periods: int) -> np.ndarray:
        v = pd.Series(values[self.order])
        rolled = (
            v.groupby(self.codes)
            .expanding(min_periods=min_periods)
            .median()
            .reset_index(level=0, drop=True)
            .sort_index()
            .to_numpy()
        )
        return self._lookup(rolled)

    def prior_window_count(self, window: pd.Timedelta) -> np.ndarray:
        width = window.value
        counts = np.empty(self.n, dtype=float)
        for a, b in zip(self.starts, self.ends, strict=True):
            t = self.times[a:b]
            counts[a:b] = np.searchsorted(t, t, side="left") - np.searchsorted(
                t, t - width, side="left"
            )
        return self._unsort(counts)


def _sticker_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Unique (row, sticker_id) pairs from the canonical sticker slots."""
    parts = []
    for i in range(MAX_STICKER_SLOTS):
        ids = pd.to_numeric(frame[f"sticker_{i}_id_cat"], errors="coerce")
        parts.append(pd.DataFrame({"_row": np.arange(len(frame)), "sticker_id": ids.to_numpy()}))
    pairs = pd.concat(parts, ignore_index=True)
    pairs = pairs[pairs["sticker_id"] >= 0].drop_duplicates()
    return pairs.sort_values("_row", kind="stable").reset_index(drop=True)


def history_features(frame: pd.DataFrame, usable: pd.Series) -> pd.DataFrame:
    """History features for every row of the static feature frame.

    Only ``usable`` rows (positive price and an anchor) feed the history and
    receive values; other rows get NaN.
    """
    out = pd.DataFrame(np.nan, index=frame.index, columns=HISTORY_COLUMNS)
    h = frame.loc[usable.to_numpy()].reset_index(drop=True)
    if h.empty:
        return out
    times = to_ns(h["event_time"])
    premium = h["log_premium"].to_numpy(dtype=float)
    result = pd.DataFrame(index=h.index)

    skin = Timeline(h["weapon_type"].to_numpy(), times)
    everyone = Timeline(np.zeros(len(h), dtype=np.int64), times)
    global_median = everyone.prior_expanding_median(premium, SKIN_MIN_HISTORY)
    skin_median = skin.prior_rolling_median(premium, SKIN_WINDOW, SKIN_MIN_HISTORY)
    result["wt_recent_premium"] = pd.Series(skin_median).fillna(pd.Series(global_median)).fillna(0.0)
    result["log_wt_prior_sales"] = np.log1p(skin.prior_count())
    result["log_wt_sales_7d"] = np.log1p(skin.prior_window_count(LIQUIDITY_WINDOW))

    seed = Timeline(h["skin_seed_key"].to_numpy(), times)
    seed_count = seed.prior_count()
    seed_sum = seed.prior_sum(premium)
    result["seed_past_premium"] = np.where(seed_count > 0, seed_sum / np.maximum(seed_count, 1), np.nan)
    result["log_seed_past_seen"] = np.log1p(seed_count)

    name = Timeline(h["name"].to_numpy(), times)
    name_median = name.prior_rolling_median(h["log_price"].to_numpy(dtype=float), NAME_WINDOW, 1)
    result["name_recent_premium"] = name_median - h["log_anchor"].to_numpy()
    result["log_name_prior_sales"] = np.log1p(name.prior_count())

    pairs = _sticker_pairs(h)
    if len(pairs):
        rows = pairs["_row"].to_numpy()
        sticker = Timeline(pairs["sticker_id"].to_numpy(), times[rows])
        count = sticker.prior_count()
        total = sticker.prior_sum(premium[rows])
        pairs["mean"] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        pairs["count"] = count
        agg = pairs.groupby("_row").agg(
            sticker_past_premium_mean=("mean", "mean"),
            sticker_past_premium_max=("mean", "max"),
            seen_min=("count", "min"),
        )
        agg = agg.reindex(range(len(h)))
        result["sticker_past_premium_mean"] = agg["sticker_past_premium_mean"].to_numpy()
        result["sticker_past_premium_max"] = agg["sticker_past_premium_max"].to_numpy()
        result["log_sticker_past_seen_min"] = np.log1p(agg["seen_min"]).to_numpy()

    out.loc[usable.to_numpy(), HISTORY_COLUMNS] = result[HISTORY_COLUMNS].to_numpy()
    return out


@dataclass
class MarketSnapshot:
    """Final state of the history accumulators, used to featurize live listings."""

    as_of: pd.Timestamp
    skins: pd.DataFrame  # weapon_type -> recent_premium, prior_sales, sales_7d
    seeds: pd.DataFrame  # skin_seed_key -> premium_sum, count
    stickers: pd.DataFrame  # sticker_id -> premium_sum, count
    names: pd.DataFrame  # name -> recent_log_price, count, recent_anchor
    global_premium: float

    @classmethod
    def build(cls, frame: pd.DataFrame, usable: pd.Series, as_of: pd.Timestamp | None = None) -> MarketSnapshot:
        h = frame.loc[usable.to_numpy()].copy()
        if as_of is None:
            as_of = h["event_time"].max() + pd.Timedelta(microseconds=1)
        h = h[h["event_time"] < as_of].sort_values(["event_time"], kind="stable")

        def tail_median(values: pd.Series, window: int, min_periods: int) -> float:
            tail = values.dropna().tail(window)
            return float(tail.median()) if len(tail) >= min_periods else np.nan

        skins = h.groupby("weapon_type", sort=False).agg(
            recent_premium=("log_premium", lambda s: tail_median(s, SKIN_WINDOW, SKIN_MIN_HISTORY)),
            prior_sales=("log_premium", "size"),
        )
        recent = h[h["event_time"] >= as_of - LIQUIDITY_WINDOW]
        skins["sales_7d"] = recent.groupby("weapon_type").size().reindex(skins.index).fillna(0)

        seeds = h.groupby("skin_seed_key", sort=False)["log_premium"].agg(premium_sum="sum", count="size")

        pairs = _sticker_pairs(h.reset_index(drop=True))
        pairs["premium"] = h["log_premium"].to_numpy()[pairs["_row"].to_numpy()]
        stickers = pairs.groupby("sticker_id")["premium"].agg(premium_sum="sum", count="size")

        names = h.groupby("name", sort=False).agg(
            recent_log_price=("log_price", lambda s: tail_median(s, NAME_WINDOW, 1)),
            count=("log_price", "size"),
            recent_log_anchor=("log_anchor", lambda s: tail_median(s, NAME_WINDOW, 1)),
        )
        global_premium = float(h["log_premium"].median()) if len(h) >= SKIN_MIN_HISTORY else 0.0
        return cls(as_of, skins, seeds, stickers, names, global_premium)

    def features_for(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        skin = self.skins.reindex(frame["weapon_type"])
        out["wt_recent_premium"] = skin["recent_premium"].fillna(self.global_premium).to_numpy()
        out["log_wt_prior_sales"] = np.log1p(skin["prior_sales"].fillna(0)).to_numpy()
        out["log_wt_sales_7d"] = np.log1p(skin["sales_7d"].fillna(0)).to_numpy()

        seed = self.seeds.reindex(frame["skin_seed_key"])
        seed_count = seed["count"].fillna(0).to_numpy()
        out["seed_past_premium"] = np.where(
            seed_count > 0, seed["premium_sum"].to_numpy() / np.maximum(seed_count, 1), np.nan
        )
        out["log_seed_past_seen"] = np.log1p(seed_count)

        pairs = _sticker_pairs(frame.reset_index(drop=True))
        if len(pairs):
            stats = self.stickers.reindex(pairs["sticker_id"])
            count = stats["count"].fillna(0).to_numpy()
            pairs["mean"] = np.where(count > 0, stats["premium_sum"].to_numpy() / np.maximum(count, 1), np.nan)
            pairs["count"] = count
            agg = pairs.groupby("_row").agg(
                mean=("mean", "mean"), max=("mean", "max"), seen_min=("count", "min")
            ).reindex(range(len(frame)))
            out["sticker_past_premium_mean"] = agg["mean"].to_numpy()
            out["sticker_past_premium_max"] = agg["max"].to_numpy()
            out["log_sticker_past_seen_min"] = np.log1p(agg["seen_min"]).to_numpy()
        else:
            out["sticker_past_premium_mean"] = np.nan
            out["sticker_past_premium_max"] = np.nan
            out["log_sticker_past_seen_min"] = np.nan

        names = self.names.reindex(frame["name"])
        out["name_recent_premium"] = (names["recent_log_price"].to_numpy() - frame["log_anchor"].to_numpy())
        out["log_name_prior_sales"] = np.log1p(names["count"].fillna(0)).to_numpy()
        return out[HISTORY_COLUMNS]

    def fallback_anchor(self, name: str) -> float | None:
        """Typical anchor for an item when the listing has no CSFloat reference."""
        if name in self.names.index:
            value = self.names.at[name, "recent_log_anchor"]
            if pd.notna(value):
                return float(np.expm1(value))
        return None

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for key in ("skins", "seeds", "stickers", "names"):
            getattr(self, key).reset_index().to_parquet(directory / f"{key}.parquet", index=False)
        meta = {"as_of": self.as_of.isoformat(), "global_premium": self.global_premium}
        (directory / "snapshot.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> MarketSnapshot:
        directory = Path(directory)
        meta = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
        index_columns = {"skins": "weapon_type", "seeds": "skin_seed_key", "stickers": "sticker_id", "names": "name"}
        parts = {
            key: pd.read_parquet(directory / f"{key}.parquet").set_index(column)
            for key, column in index_columns.items()
        }
        return cls(pd.Timestamp(meta["as_of"]), global_premium=meta["global_premium"], **parts)
