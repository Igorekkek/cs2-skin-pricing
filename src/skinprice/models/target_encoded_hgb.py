"""scikit-learn histogram gradient boosting with K-fold target encoding.

Categorical features are replaced by out-of-fold mean targets
(``sklearn.preprocessing.TargetEncoder``), numeric ones go in as is (missing
values are handled natively). Three boosters: absolute error for the median
and two quantile losses. A deliberately standard stack: it shows how much of
CatBoost's advantage comes from its categorical handling.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import TargetEncoder

from skinprice.models.base import HIGH_Q, LOW_Q, PriceModel


class TargetEncodedHGB(PriceModel):
    name = "target_encoded_hgb"
    summary = "sklearn HistGradientBoosting with out-of-fold target encoding of categories"
    default_params = {
        "max_iter": 800,
        "learning_rate": 0.06,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 40,
        "l2_regularization": 0.1,
        "random_state": 42,
    }

    def __init__(self, params=None, spec=None):
        super().__init__(params, spec)
        self.encoder: TargetEncoder | None = None
        self.models: dict[str, HistGradientBoostingRegressor] = {}

    def _matrix(self, frame: pd.DataFrame, fit: bool = False) -> np.ndarray:
        cats = frame[self.spec.categorical].astype(str)
        if fit:
            self.encoder = TargetEncoder(
                target_type="continuous", smooth="auto", cv=5, random_state=self.params["random_state"]
            )
            encoded = self.encoder.fit_transform(cats, self.target(frame))
        else:
            encoded = self.encoder.transform(cats)
        numeric = frame[self.spec.numeric].to_numpy(dtype=float)
        return np.hstack([encoded, numeric])

    def _booster(self, loss: str, quantile: float | None = None) -> HistGradientBoostingRegressor:
        p = self.params
        kwargs = dict(
            loss=loss,
            max_iter=p["max_iter"],
            learning_rate=p["learning_rate"],
            max_leaf_nodes=p["max_leaf_nodes"],
            min_samples_leaf=p["min_samples_leaf"],
            l2_regularization=p["l2_regularization"],
            early_stopping=False,  # a random validation slice would break the time order
            random_state=p["random_state"],
        )
        if quantile is not None:
            kwargs["quantile"] = quantile
        return HistGradientBoostingRegressor(**kwargs)

    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> None:
        x = self._matrix(train, fit=True)
        y = self.target(train)
        x_valid = self._matrix(valid)
        y_valid = self.target(valid)
        for key, loss, q in (
            ("point", "absolute_error", None),
            ("low", "quantile", LOW_Q),
            ("high", "quantile", HIGH_Q),
        ):
            model = self._booster(loss, q).fit(x, y)
            self.models[key] = model
            if key == "point":
                curve = [float(np.mean(np.abs(p - y_valid))) for p in model.staged_predict(x_valid)]
                self.info["valid_mae_curve_every_50"] = curve[::50]
                self.info["best_iteration"] = int(np.argmin(curve)) + 1

    def _refit(self, full: pd.DataFrame) -> None:
        x, y = self._matrix(full, fit=True), self.target(full)
        for key, loss, q in (("point", "absolute_error", None), ("low", "quantile", LOW_Q), ("high", "quantile", HIGH_Q)):
            self.models[key] = self._booster(loss, q).fit(x, y)

    def _predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        x = self._matrix(frame)
        return np.column_stack([self.models[k].predict(x) for k in ("low", "point", "high")])

    def _save(self, directory: Path) -> None:
        joblib.dump({"encoder": self.encoder, "models": self.models}, directory / "model.joblib", compress=3)

    def _load(self, directory: Path) -> None:
        payload = joblib.load(directory / "model.joblib")
        self.encoder, self.models = payload["encoder"], payload["models"]
