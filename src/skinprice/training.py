"""Train a model on the feature table, evaluate it and store it with its metrics.

Layout of a stored model::

    <workspace>/models/<model>/<version>/
        manifest.json             params, features, calibration, metrics, sizes
        <model files>             .cbm / .joblib / .pt
        test_predictions.parquet  listing_id + predicted price and interval
    <workspace>/models/<model>/LATEST   name of the newest version
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from skinprice import models, paths
from skinprice.evaluation.metrics import price_metrics
from skinprice.features.spec import FeatureSpec
from skinprice.models.base import PriceModel
from skinprice.split import parts


@dataclass
class TrainResult:
    model: PriceModel
    metrics: dict[str, dict[str, float]]
    test_predictions: pd.DataFrame
    directory: Path | None


def model_config(name: str) -> dict[str, Any]:
    path = paths.CONFIG_DIR / "models" / f"{name}.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def evaluate(model: PriceModel, frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    prediction = model.predict(frame)
    table = prediction.to_frame(frame.index)
    metrics = price_metrics(frame["price_cents"].to_numpy(), table["pred_cents"], table["low_cents"], table["high_cents"])
    return metrics, table


def train_and_evaluate(
    name: str,
    features: pd.DataFrame,
    params: dict[str, Any] | None = None,
    spec: FeatureSpec | None = None,
    version: str | None = None,
    save: bool = True,
    root: Path | None = None,
    directory: Path | None = None,
) -> TrainResult:
    """Fit on train, calibrate on validation, score validation and test, store.

    The model is saved to ``directory`` when given (experiments use this), otherwise to
    ``<root or workspace/models>/<name>/<version>`` and marked as the latest version.
    """
    config = model_config(name)
    params = {**config.get("params", {}), **(params or {})}
    if spec is None and "features" in config:
        spec = FeatureSpec.from_dict(config["features"])
    train, valid, test = parts(features)
    model = models.create(name, params, spec).fit(train, valid)

    valid_metrics, _ = evaluate(model, valid)
    test_metrics, test_table = evaluate(model, test)
    test_table.insert(0, "listing_id", test["listing_id"].to_numpy())
    metrics = {"val": valid_metrics, "test": test_metrics}

    if save:
        version = version or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        mark_latest = directory is None
        if directory is None:
            root = Path(root) if root else paths.workspace().models_dir
            directory = root / name / version
        directory = Path(directory)
        model.save(
            directory,
            extra={
                "version": version,
                "metrics": metrics,
                "data": {
                    split: {"rows": int(len(f)), "from": str(f["event_time"].min()), "to": str(f["event_time"].max())}
                    for split, f in (("train", train), ("val", valid), ("test", test))
                },
            },
        )
        test_table.to_parquet(directory / "test_predictions.parquet", index=False)
        if mark_latest:
            (directory.parent / "LATEST").write_text(version, encoding="utf-8")
    return TrainResult(model, metrics, test_table, directory)


def latest_dir(name: str, root: Path | None = None) -> Path:
    root = Path(root) if root else paths.workspace().models_dir
    pointer = root / name / "LATEST"
    if not pointer.exists():
        raise FileNotFoundError(f"No trained {name!r} in {root}. Run `skinprice train {name}`.")
    return root / name / pointer.read_text(encoding="utf-8").strip()
