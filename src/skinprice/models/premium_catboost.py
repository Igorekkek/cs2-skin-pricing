"""CatBoost on the log-premium over CSFloat's reference price.

CatBoost handles the high-cardinality categories (skin + seed, sticker ids,
collections) natively with ordered target statistics, so no manual encoding
is needed. Two ways to get the interval:

* ``quantile_mode = "separate"``: three models (MAE for the median, two
  quantile models), as in the original research;
* ``quantile_mode = "multi"``: one ``MultiQuantile`` model with three outputs
  sharing the same trees, three times smaller and faster.

Model size is dominated by CTR tables for combinations of categorical
features; ``max_ctr_complexity`` limits how many categories may be combined
(CatBoost's default is 4 on CPU, which produced a 9 GB model).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from skinprice.models.base import HIGH_Q, LOW_Q, PriceModel


class PremiumCatBoost(PriceModel):
    name = "premium_catboost"
    summary = "CatBoost on log-premium over the CSFloat reference, native categorical handling"
    default_params = {
        "quantile_mode": "multi",
        "iterations": 3000,
        "learning_rate": 0.06,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "max_ctr_complexity": 1,
        # True: target statistics of categories follow the row order (chronological here)
        # instead of random permutations
        "has_time": False,
        "early_stopping_rounds": 150,
        "thread_count": -1,
        "random_seed": 42,
        "max_train_rows": None,
    }

    def __init__(self, params=None, spec=None):
        super().__init__(params, spec)
        self.models: dict[str, CatBoostRegressor] = {}

    def _pool(self, frame: pd.DataFrame, with_label: bool = True) -> Pool:
        data = frame[self.spec.columns].copy()
        for column in self.spec.categorical:
            data[column] = data[column].astype(str)
        label = self.target(frame) if with_label else None
        return Pool(data, label=label, cat_features=self.spec.categorical)

    def _regressor(self, loss: str, iterations: int | None = None) -> CatBoostRegressor:
        p = self.params
        return CatBoostRegressor(
            loss_function=loss,
            eval_metric=loss,
            iterations=iterations or p["iterations"],
            learning_rate=p["learning_rate"],
            depth=p["depth"],
            l2_leaf_reg=p["l2_leaf_reg"],
            max_ctr_complexity=p["max_ctr_complexity"],
            has_time=p["has_time"],
            early_stopping_rounds=None if iterations else p["early_stopping_rounds"],
            thread_count=p["thread_count"],
            random_seed=p["random_seed"],
            allow_writing_files=False,
            verbose=250,
        )

    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> None:
        if self.params["max_train_rows"]:
            train = train.tail(int(self.params["max_train_rows"]))
        train_pool, valid_pool = self._pool(train), self._pool(valid)
        if self.params["quantile_mode"] == "multi":
            losses = {"multi": f"MultiQuantile:alpha={LOW_Q},0.5,{HIGH_Q}"}
        else:
            losses = {
                "point": "MAE",
                "low": f"Quantile:alpha={LOW_Q}",
                "high": f"Quantile:alpha={HIGH_Q}",
            }
        for key, loss in losses.items():
            model = self._regressor(loss)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            self.models[key] = model
            self.info[f"{key}_best_iteration"] = int(model.get_best_iteration())

    def _refit(self, full: pd.DataFrame) -> None:
        if self.params["max_train_rows"]:
            full = full.tail(int(self.params["max_train_rows"]))
        pool = self._pool(full)
        for key, model in list(self.models.items()):
            refit = self._regressor(model.get_params()["loss_function"], self.info[f"{key}_best_iteration"] + 1)
            refit.fit(pool)
            self.models[key] = refit

    def _predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        pool = self._pool(frame, with_label=False)
        if "multi" in self.models:
            return np.asarray(self.models["multi"].predict(pool)).reshape(len(frame), 3)
        return np.column_stack(
            [self.models[k].predict(pool) for k in ("low", "point", "high")]
        )

    def feature_importance(self) -> pd.Series:
        model = self.models.get("multi") or self.models["point"]
        return pd.Series(model.get_feature_importance(), index=model.feature_names_).sort_values(ascending=False)

    def _save(self, directory: Path) -> None:
        for key, model in self.models.items():
            model.save_model(str(directory / f"{key}.cbm"))

    def _load(self, directory: Path) -> None:
        for path in sorted(directory.glob("*.cbm")):
            model = CatBoostRegressor()
            model.load_model(str(path))
            self.models[path.stem] = model
