"""Column layout of the three dataset tables.

The dataset is stored as three parquet files with one row per sale, per
applied sticker and per attached charm (keychain). The same layout is
produced from the collector's SQLite database (``data.build``) and from a
single CSFloat listing JSON at inference time (``data.normalize``), so
training and serving share one feature pipeline.
"""

from __future__ import annotations

SALES_TIME_COLUMNS = ["created_at", "sold_at", "reference_last_updated"]

SALES_COLUMNS = [
    "listing_id",
    "name",
    "price_cents",
    "reference_base_price",
    "reference_predicted_price",
    "reference_float_factor",
    "reference_quantity",
    "reference_keychain_price",
    "reference_last_updated",
    "float_value",
    "paint_seed",
    "paint_index",
    "def_index",
    "rarity",
    "rarity_name",
    "collection",
    "sale_type",
    "is_seller",
    "is_watchlisted",
    "watchers",
    "created_at",
    "sold_at",
    # Fields CSFloat returns inside the item JSON that the original parser
    # did not keep as columns: float rank, Case Hardened blue %, Fade %.
    "low_rank",
    "high_rank",
    "blue_playside",
    "blue_backside",
    "gold_playside",
    "gold_backside",
    "fade_percentage",
    "fade_rank",
]

# ``position`` is the index in the listing's sticker array. Several stickers
# may share one ``slot`` (CS2 allows free placement), so rows are unique by
# (listing_id, position), not by slot.
STICKER_COLUMNS = [
    "listing_id",
    "position",
    "slot",
    "sticker_id",
    "name",
    "wear",
    "reference_price_cents",
    "reference_quantity",
    "reference_updated_at",
]

KEYCHAIN_COLUMNS = [
    "listing_id",
    "position",
    "slot",
    "keychain_id",
    "name",
    "pattern",
    "reference_price_cents",
    "reference_quantity",
    "reference_updated_at",
]

TABLES = ("sales", "stickers", "keychains")
