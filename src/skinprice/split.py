"""Chronological train / validation / test split.

The newest sales form the test set, the ones just before it the validation
set (early stopping and calibration only). Sales sharing a timestamp always
land in the same part. The test set is never used to choose anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    test_fraction: float = 0.20
    val_fraction: float = 0.10
    # Fixed boundaries (ISO dates) override the fractions when given.
    test_start: str | None = None
    val_start: str | None = None


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _boundary(times: pd.Series, fraction_from_end: float) -> pd.Timestamp:
    position = int(len(times) * (1 - fraction_from_end))
    position = min(max(position, 0), len(times) - 1)
    return times.iloc[position]


def assign_split(event_time: pd.Series, config: SplitConfig = SplitConfig()) -> pd.Series:
    times = pd.to_datetime(event_time, utc=True)
    ordered = times.sort_values(ignore_index=True)
    test_start = _boundary(ordered, config.test_fraction)
    val_start = _boundary(ordered, config.test_fraction + config.val_fraction)
    if config.test_start:
        test_start = _utc(config.test_start)
    if config.val_start:
        val_start = _utc(config.val_start)
    split = np.where(times >= test_start, "test", np.where(times >= val_start, "val", "train"))
    return pd.Series(split, index=event_time.index, dtype="category")


def parts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        frame[frame["split"] == "train"],
        frame[frame["split"] == "val"],
        frame[frame["split"] == "test"],
    )
