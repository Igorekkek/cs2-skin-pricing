"""Reference points every learned model has to beat."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skinprice.models.base import PriceModel


class CSFloatReference(PriceModel):
    """CSFloat's own estimate, shifted by the median premium on validation.

    The interval comes from the empirical spread of premiums on validation.
    """

    name = "csfloat_reference"
    summary = "CSFloat reference price (calibrated), no learning"

    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> None:
        pass

    def _predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(frame), 3))


class RecentMedian(PriceModel):
    """Median price of the last 20 sales of the exact same item.

    Falls back to the CSFloat reference for items never seen before.
    """

    name = "recent_median"
    summary = "Median of the last 20 sales of the same item, reference as fallback"

    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> None:
        pass

    def _predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        center = frame["name_recent_premium"].fillna(0.0).to_numpy(dtype=float)
        return np.column_stack([center, center, center])
