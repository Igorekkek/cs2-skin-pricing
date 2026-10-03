"""Catalog for hand-typed listings: stickers, charms and recent sales per item.

A real CSFloat listing carries the sticker ids and prices, CSFloat's reference
block and item metadata. When a listing is typed in by hand, these come from
here: sticker ids and prices by name, and the reference block and metadata of
the most similar recent sale of the same item (closest float).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RECENT_SALES_PER_ITEM = 20
RECENT_COLUMNS = [
    "name", "float_value", "sold_at", "paint_index", "def_index", "rarity", "rarity_name", "collection",
    "reference_base_price", "reference_predicted_price", "reference_float_factor", "reference_quantity",
    "reference_last_updated",
]


@dataclass
class Catalog:
    stickers: pd.DataFrame  # sticker_id, name, reference_price_cents, reference_quantity, updated_at
    keychains: pd.DataFrame  # keychain_id, name, reference_price_cents, reference_quantity, updated_at
    items: list[str]
    recent_sales: pd.DataFrame  # last sales per item name: float, reference block, metadata

    @classmethod
    def build(cls, tables: dict[str, pd.DataFrame]) -> Catalog:
        def latest(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
            frame = frame.dropna(subset=[id_column, "name"]).sort_values("reference_updated_at", na_position="first")
            frame = frame.drop_duplicates(id_column, keep="last")
            return frame[[id_column, "name", "reference_price_cents", "reference_quantity", "reference_updated_at"]].rename(
                columns={"reference_updated_at": "updated_at"}
            ).reset_index(drop=True)

        sales = tables["sales"]
        items = sorted(sales["name"].dropna().astype(str).unique().tolist())
        usable = sales[sales["reference_predicted_price"] > 0].sort_values("sold_at")
        recent = usable.groupby("name", sort=False).tail(RECENT_SALES_PER_ITEM)[RECENT_COLUMNS].reset_index(drop=True)
        return cls(latest(tables["stickers"], "sticker_id"), latest(tables["keychains"], "keychain_id"), items, recent)

    def similar_sale(self, name: str, float_value: float | None) -> pd.Series | None:
        """The recent sale of ``name`` with the closest float (or the latest one)."""
        sales = self.recent_sales[self.recent_sales["name"] == name]
        if sales.empty:
            return None
        if float_value is None or sales["float_value"].isna().all():
            return sales.iloc[-1]
        return sales.loc[(sales["float_value"] - float_value).abs().idxmin()]

    def find_sticker(self, text: str) -> pd.Series | None:
        return _find(self.stickers, text, "sticker_id")

    def find_keychain(self, text: str) -> pd.Series | None:
        return _find(self.keychains, text, "keychain_id")

    def search(self, kind: str, query: str, limit: int = 20) -> list[str]:
        names = {
            "stickers": self.stickers["name"].astype(str),
            "keychains": self.keychains["name"].astype(str),
            "items": pd.Series(self.items),
        }[kind]
        words = query.lower().split()
        mask = pd.Series(True, index=names.index)
        for word in words:
            mask &= names.str.lower().str.contains(word, regex=False)
        return names[mask].head(limit).tolist()

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.stickers.to_parquet(directory / "catalog_stickers.parquet", index=False)
        self.keychains.to_parquet(directory / "catalog_keychains.parquet", index=False)
        (directory / "catalog_items.json").write_text(json.dumps(self.items), encoding="utf-8")
        self.recent_sales.to_parquet(directory / "catalog_recent_sales.parquet", index=False)

    @classmethod
    def load(cls, directory: Path) -> Catalog:
        directory = Path(directory)
        return cls(
            pd.read_parquet(directory / "catalog_stickers.parquet"),
            pd.read_parquet(directory / "catalog_keychains.parquet"),
            json.loads((directory / "catalog_items.json").read_text(encoding="utf-8")),
            pd.read_parquet(directory / "catalog_recent_sales.parquet"),
        )


def _find(frame: pd.DataFrame, text: str, id_column: str) -> pd.Series | None:
    text = str(text).strip()
    if text.isdigit():
        hit = frame[frame[id_column] == float(text)]
    else:
        names = frame["name"].astype(str)
        hit = frame[names.str.lower() == text.lower()]
        if hit.empty:
            hit = frame[names.str.lower().str.contains(text.lower(), regex=False)]
    return None if hit.empty else hit.iloc[0]
