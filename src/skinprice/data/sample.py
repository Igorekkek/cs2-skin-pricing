"""Small public sample of the dataset.

The sample keeps *all* sales of a deterministic subset of skins, so per-skin
history features behave the same way as on the full data.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import pandas as pd

from skinprice.data.build import write_tables
from skinprice.features.identity import parse_names


def _bucket(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) % 1000


def sample_tables(tables: dict[str, pd.DataFrame], per_mille: int = 50) -> dict[str, pd.DataFrame]:
    sales = tables["sales"]
    weapon_type = parse_names(sales["name"])["weapon_type"]
    keep = weapon_type.map(_bucket) < per_mille
    sales = sales[keep.to_numpy()].reset_index(drop=True)
    ids = set(sales["listing_id"])
    return {
        "sales": sales,
        "stickers": tables["stickers"][tables["stickers"]["listing_id"].isin(ids)].reset_index(drop=True),
        "keychains": tables["keychains"][tables["keychains"]["listing_id"].isin(ids)].reset_index(drop=True),
    }


def write_sample(tables: dict[str, pd.DataFrame], out_dir: Path, per_mille: int = 50) -> dict:
    sample = sample_tables(tables, per_mille)
    return write_tables(
        sample,
        Path(out_dir),
        {"source": "sample of the full dataset", "skins_per_mille": per_mille},
    )
