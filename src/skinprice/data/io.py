"""Load the parquet dataset (full one from the workspace or the bundled sample)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from skinprice import paths
from skinprice.data.schema import TABLES


def load_tables(directory: Path | None = None) -> dict[str, pd.DataFrame]:
    directory = Path(directory) if directory else paths.workspace().dataset_dir
    missing = [t for t in TABLES if not (directory / f"{t}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset tables {missing} not found in {directory}. "
            "Run `skinprice data build` or pass --dataset data/sample."
        )
    return {t: pd.read_parquet(directory / f"{t}.parquet") for t in TABLES}


def load_manifest(directory: Path) -> dict:
    path = Path(directory) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
