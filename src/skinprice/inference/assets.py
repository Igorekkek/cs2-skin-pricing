"""Build the serving assets: market snapshot + sticker/charm/item catalog."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from skinprice import paths
from skinprice.data.io import load_tables
from skinprice.features.build import load_features
from skinprice.features.history import MarketSnapshot
from skinprice.inference.catalog import Catalog


def build_serving_assets(
    features: str | Path = "features",
    dataset_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    out_dir = Path(out_dir) if out_dir else paths.workspace().snapshot_dir
    frame = load_features(features)
    snapshot = MarketSnapshot.build(frame, pd.Series(True, index=frame.index))
    snapshot.save(out_dir)
    catalog = Catalog.build(load_tables(dataset_dir))
    catalog.save(out_dir)
    return {
        "as_of": snapshot.as_of.isoformat(),
        "skins": len(snapshot.skins),
        "items": len(catalog.items),
        "stickers": len(catalog.stickers),
        "dir": str(out_dir),
    }
