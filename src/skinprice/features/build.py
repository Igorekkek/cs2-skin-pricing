"""Assemble the training table: static + history features, target, split."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from skinprice import paths
from skinprice.data.io import load_manifest, load_tables
from skinprice.features.history import MarketSnapshot, history_features
from skinprice.features.static import static_features
from skinprice.split import SplitConfig, assign_split

FEATURE_VERSION = 1


@dataclass(frozen=True)
class FeatureConfig:
    # None keeps every sticker reference price as collected (original setup);
    # a number masks prices CSFloat updated more than N days after the sale.
    max_sticker_reference_lag_days: float | None = None
    split: SplitConfig = SplitConfig()

    def to_dict(self) -> dict:
        return {**asdict(self), "split": asdict(self.split)}


def usable_rows(frame: pd.DataFrame) -> pd.Series:
    return (frame["price_cents"] > 0) & (frame["anchor_cents"] > 0)


def build_feature_table(tables: dict[str, pd.DataFrame], config: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    frame = static_features(tables, max_sticker_reference_lag_days=config.max_sticker_reference_lag_days)
    usable = usable_rows(frame)
    frame = frame.join(history_features(frame, usable))
    frame = frame[usable].reset_index(drop=True)
    frame["split"] = assign_split(frame["event_time"], config.split)
    return frame


def feature_path(name: str = "features") -> Path:
    return paths.workspace().features_dir / f"{name}.parquet"


def build_and_save(
    dataset_dir: Path | None = None,
    name: str = "features",
    config: FeatureConfig = FeatureConfig(),
    out_dir: Path | None = None,
) -> dict:
    started = time.time()
    dataset_dir = Path(dataset_dir) if dataset_dir else paths.workspace().dataset_dir
    tables = load_tables(dataset_dir)
    frame = build_feature_table(tables, config)
    out_dir = Path(out_dir) if out_dir else paths.workspace().features_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / f"{name}.parquet", index=False, compression="zstd")
    bounds = frame.groupby("split")["event_time"].agg(["min", "max", "size"])
    manifest = {
        "feature_version": FEATURE_VERSION,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {"dir": str(dataset_dir), **load_manifest(dataset_dir)},
        "config": config.to_dict(),
        "rows": int(len(frame)),
        "splits": {
            split: {"rows": int(row["size"]), "from": row["min"].isoformat(), "to": row["max"].isoformat()}
            for split, row in bounds.iterrows()
        },
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def load_features(name_or_path: str | Path = "features", columns: list[str] | None = None) -> pd.DataFrame:
    """Feature table; ``columns`` reads only those columns (the full table needs several GB)."""
    path = Path(name_or_path)
    if path.suffix != ".parquet":
        path = feature_path(str(name_or_path))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `skinprice features build` first.")
    return pd.read_parquet(path, columns=columns)


def build_snapshot(frame: pd.DataFrame) -> MarketSnapshot:
    """Snapshot over the whole feature table (every row is usable by construction)."""
    return MarketSnapshot.build(frame, pd.Series(True, index=frame.index))
