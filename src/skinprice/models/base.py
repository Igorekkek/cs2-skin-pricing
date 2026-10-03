"""Common interface of all price models.

Every model predicts three numbers per listing in log-premium space
(see ``features.static``): the 15% quantile, the median and the 85% quantile.
After fitting, a calibration step on the validation split shifts each of the
three outputs so that the median is unbiased and the [q15, q85] interval
covers 70% of validation sales. Prices are then ``expm1(premium + log_anchor)``.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from skinprice.features.spec import FeatureSpec

LOW_Q, HIGH_Q = 0.15, 0.85
MAX_LOG_PRICE = 22.0  # expm1(22) cents ~ $36B, a guard against overflow


@dataclass
class Calibration:
    low: float = 0.0
    center: float = 0.0
    high: float = 0.0

    @classmethod
    def fit(cls, raw: np.ndarray, target: np.ndarray) -> Calibration:
        return cls(
            low=float(np.quantile(target - raw[:, 0], LOW_Q)),
            center=float(np.median(target - raw[:, 1])),
            high=float(np.quantile(target - raw[:, 2], HIGH_Q)),
        )

    def apply(self, raw: np.ndarray) -> np.ndarray:
        out = raw + np.array([self.low, self.center, self.high])
        out[:, 0] = np.minimum(out[:, 0], out[:, 1])
        out[:, 2] = np.maximum(out[:, 2], out[:, 1])
        return out


@dataclass
class Prediction:
    premium: np.ndarray  # (n, 3): low, center, high in log-premium space
    log_anchor: np.ndarray  # zeros when the model predicts log-price directly

    def _price(self, column: int) -> np.ndarray:
        return np.expm1(np.clip(self.premium[:, column] + self.log_anchor, 0, MAX_LOG_PRICE))

    @property
    def price(self) -> np.ndarray:
        return self._price(1)

    @property
    def low(self) -> np.ndarray:
        return self._price(0)

    @property
    def high(self) -> np.ndarray:
        return self._price(2)

    def to_frame(self, index: pd.Index | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            {"pred_cents": self.price, "low_cents": self.low, "high_cents": self.high},
            index=index,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class PriceModel(ABC):
    name: ClassVar[str]
    summary: ClassVar[str]
    default_params: ClassVar[dict[str, Any]] = {}

    def __init__(self, params: dict[str, Any] | None = None, spec: FeatureSpec | None = None):
        self.params = {**self.default_params, **(params or {})}
        self.spec = spec or FeatureSpec()
        self.calibration = Calibration()
        self.info: dict[str, Any] = {}

    # --- implemented by each model -------------------------------------------------
    @abstractmethod
    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> None: ...

    @abstractmethod
    def _predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        """Uncalibrated (n, 3) log-premium quantiles."""

    def _refit(self, full: pd.DataFrame) -> None:  # noqa: B027 - optional hook
        """Retrain on train+validation with the budget (trees, epochs) chosen on validation."""

    def _save(self, directory: Path) -> None:  # noqa: B027 - optional hook
        pass

    def _load(self, directory: Path) -> None:  # noqa: B027 - optional hook
        pass

    # --- shared ----------------------------------------------------------------------
    @property
    def predicts_premium(self) -> bool:
        """False when the experiment option ``target = "price"`` is set: the model then
        learns log1p(price) directly, without the CSFloat anchor."""
        return self.params.get("target", "premium") == "premium"

    def target(self, frame: pd.DataFrame) -> np.ndarray:
        column = "log_premium" if self.predicts_premium else "log_price"
        return frame[column].to_numpy(dtype=float)

    def offset(self, frame: pd.DataFrame) -> np.ndarray:
        if self.predicts_premium:
            return frame["log_anchor"].to_numpy(dtype=float)
        return np.zeros(len(frame))

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> PriceModel:
        started = time.time()
        self._fit(train, valid)
        raw = self._predict_raw(valid)
        self.calibration = Calibration.fit(raw, self.target(valid))
        if self.params.get("refit_on_validation"):
            # The newest sales before the test period sit in the validation split. Once the
            # number of trees / epochs is chosen there, retrain on train+validation with that
            # budget. Calibration offsets stay from the validation-scored model: residuals of
            # the refitted model on data it has seen would make the interval too narrow.
            self._refit(pd.concat([train, valid]))
            self.info["refit_on_validation"] = True
        self.info["fit_seconds"] = round(time.time() - started, 1)
        return self

    def predict(self, frame: pd.DataFrame) -> Prediction:
        raw = self._predict_raw(frame)
        return Prediction(self.calibration.apply(raw), self.offset(frame))

    def save(self, directory: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._save(directory)
        files = {
            p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)}
            for p in sorted(directory.iterdir())
            if p.is_file() and p.name not in {"manifest.json", "test_predictions.parquet"}
        }
        manifest = {
            "model": self.name,
            "summary": self.summary,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "params": self.params,
            "features": self.spec.to_dict(),
            "calibration": asdict(self.calibration),
            "info": self.info,
            "files": files,
            "size_mb": round(sum(f["bytes"] for f in files.values()) / 2**20, 2),
            **(extra or {}),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return manifest

    @classmethod
    def from_manifest(cls, directory: Path, manifest: dict[str, Any]) -> PriceModel:
        model = cls(manifest["params"], FeatureSpec.from_dict(manifest["features"]))
        model.calibration = Calibration(**manifest["calibration"])
        model.info = manifest.get("info", {})
        model._load(Path(directory))
        return model


def read_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
