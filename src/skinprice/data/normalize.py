"""Turn CSFloat listing JSON into dataset rows.

CSFloat returns the same object shape for a sold item in
``/api/v1/history/{name}/sales`` and for an active listing in
``/api/v1/listings/{id}``. These functions are used by the collector, by the
dataset builder (for fields kept only in raw JSON) and by the inference
service.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from skinprice.data.schema import KEYCHAIN_COLUMNS, SALES_COLUMNS, STICKER_COLUMNS


def _num(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: Any) -> int | None:
    number = _num(value)
    return int(number) if number is not None else None


def _flag(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def item_extras(item: dict[str, Any]) -> dict[str, Any]:
    """Pattern and rank fields that CSFloat sends only for some items."""
    blue = item.get("blue_gem") or {}
    fade = item.get("fade") or {}
    return {
        "low_rank": _int(item.get("low_rank")),
        "high_rank": _int(item.get("high_rank")),
        "blue_playside": _num(blue.get("playside_blue")),
        "blue_backside": _num(blue.get("backside_blue")),
        "gold_playside": _num(blue.get("playside_gold")),
        "gold_backside": _num(blue.get("backside_gold")),
        "fade_percentage": _num(fade.get("percentage")),
        "fade_rank": _int(fade.get("rank")),
    }


def sale_row(listing: dict[str, Any]) -> dict[str, Any]:
    item = listing.get("item") or {}
    reference = listing.get("reference") or {}
    row = {
        "listing_id": str(listing.get("id") or ""),
        "name": item.get("market_hash_name"),
        "price_cents": _int(listing.get("price")),
        "reference_base_price": _num(reference.get("base_price")),
        "reference_predicted_price": _num(reference.get("predicted_price")),
        "reference_float_factor": _num(reference.get("float_factor")),
        "reference_quantity": _num(reference.get("quantity")),
        "reference_keychain_price": _num(reference.get("keychain_price")),
        "reference_last_updated": reference.get("last_updated"),
        "float_value": _num(item.get("float_value")),
        "paint_seed": _int(item.get("paint_seed")),
        "paint_index": _int(item.get("paint_index")),
        "def_index": _int(item.get("def_index")),
        "rarity": _int(item.get("rarity")),
        "rarity_name": item.get("rarity_name"),
        "collection": item.get("collection"),
        "sale_type": listing.get("type"),
        "is_seller": _flag(listing.get("is_seller")),
        "is_watchlisted": _flag(listing.get("is_watchlisted")),
        "watchers": _int(listing.get("watchers")),
        "created_at": listing.get("created_at"),
        "sold_at": listing.get("sold_at"),
    }
    row.update(item_extras(item))
    return row


def _child_rows(listing: dict[str, Any], key: str, id_column: str) -> list[dict[str, Any]]:
    item = listing.get("item") or {}
    rows = []
    for position, child in enumerate(item.get(key) or []):
        if not isinstance(child, dict):
            continue
        reference = child.get("reference") or {}
        row = {
            "listing_id": str(listing.get("id") or ""),
            "position": position,
            "slot": _int(child.get("slot")) if child.get("slot") is not None else position,
            id_column: _int(child.get("stickerId")),
            "name": child.get("name"),
            "reference_price_cents": _num(reference.get("price")),
            "reference_quantity": _num(reference.get("quantity")),
            "reference_updated_at": reference.get("updated_at"),
        }
        if key == "stickers":
            row["wear"] = _num(child.get("wear"))
        else:
            row["pattern"] = _int(child.get("pattern"))
        rows.append(row)
    return rows


def sticker_rows(listing: dict[str, Any]) -> list[dict[str, Any]]:
    return _child_rows(listing, "stickers", "sticker_id")


def keychain_rows(listing: dict[str, Any]) -> list[dict[str, Any]]:
    return _child_rows(listing, "keychains", "keychain_id")


def to_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True, format="ISO8601")


def listings_to_tables(listings: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Normalize a batch of listing objects into the three dataset tables."""
    sales = pd.DataFrame([sale_row(x) for x in listings], columns=SALES_COLUMNS)
    stickers = pd.DataFrame(
        [r for x in listings for r in sticker_rows(x)], columns=STICKER_COLUMNS
    )
    keychains = pd.DataFrame(
        [r for x in listings for r in keychain_rows(x)], columns=KEYCHAIN_COLUMNS
    )
    return coerce_tables({"sales": sales, "stickers": stickers, "keychains": keychains})


def coerce_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply the canonical dtypes shared by every source of rows."""
    sales = tables["sales"].copy()
    sales["listing_id"] = sales["listing_id"].astype(str)
    for column in ("created_at", "sold_at", "reference_last_updated"):
        sales[column] = to_utc(sales[column])
    for column in ("name", "rarity_name", "collection", "sale_type"):
        sales[column] = sales[column].astype("string")
    numeric = [
        c
        for c in SALES_COLUMNS
        if c not in {"listing_id", "name", "rarity_name", "collection", "sale_type"}
        and c not in {"created_at", "sold_at", "reference_last_updated"}
    ]
    for column in numeric:
        sales[column] = pd.to_numeric(sales[column], errors="coerce").astype("float64")

    children = {}
    for key, id_column, extra in (
        ("stickers", "sticker_id", "wear"),
        ("keychains", "keychain_id", "pattern"),
    ):
        frame = tables[key].copy()
        frame["listing_id"] = frame["listing_id"].astype(str)
        frame["reference_updated_at"] = to_utc(frame["reference_updated_at"])
        frame["name"] = frame["name"].astype("string")
        for column in ("position", "slot", id_column, extra, "reference_price_cents", "reference_quantity"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        children[key] = frame
    return {"sales": sales, **children}
