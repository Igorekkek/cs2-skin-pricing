"""Accepted inputs and their conversion to a CSFloat-shaped listing.

Three ways to describe an item:

1. CSFloat listing JSON (what ``/api/v1/listings/{id}`` returns) - exact;
2. a CSFloat link or listing id - fetched from the API (needs network,
   an API key is used when ``CSFLOAT_API_KEY`` is set);
3. manual fields: name, float, seed, sticker names with wear, optional price.
   Sticker ids and prices come from the catalog; without CSFloat's reference
   price the anchor is estimated from recent sales of the same item.

Everything is converted to the CSFloat JSON shape, so the same normalizer
and feature code as in training is used.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

import pandas as pd

from skinprice import paths
from skinprice.inference.catalog import Catalog

LISTING_API = "https://csfloat.com/api/v1/listings/{id}"
_ID_PATTERN = re.compile(r"(?:csfloat\.com/item/)?(\d{12,25})")


def listing_id_from_text(text: str) -> str | None:
    match = _ID_PATTERN.search(text.strip())
    return match.group(1) if match else None


def fetch_listing(listing_id: str, timeout: float = 15.0) -> dict[str, Any]:
    request = urllib.request.Request(LISTING_API.format(id=listing_id), headers={"Accept": "application/json"})
    key = paths.setting("CSFLOAT_API_KEY")
    if key:
        request.add_header("Authorization", key)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https host
        return json.loads(response.read().decode("utf-8"))


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(str(value).replace(",", "."))


def manual_listing(fields: dict[str, Any], catalog: Catalog) -> tuple[dict[str, Any], list[str]]:
    """Build a CSFloat-shaped listing from hand-typed fields. Returns (listing, warnings)."""
    warnings: list[str] = []
    name = str(fields.get("name") or "").strip()
    if not name:
        raise ValueError("Item name is required, e.g. 'AK-47 | Redline (Field-Tested)'")
    stickers = []
    for slot, entry in enumerate(fields.get("stickers") or []):
        if isinstance(entry, str):
            entry = {"name": entry}
        found = catalog.find_sticker(entry.get("name") or entry.get("id") or "")
        if found is None:
            warnings.append(f"Unknown sticker {entry.get('name')!r}, ignored")
            continue
        sticker = {
            "stickerId": int(found["sticker_id"]),
            "slot": slot,
            "name": found["name"],
            "reference": {
                "price": found["reference_price_cents"],
                "quantity": found["reference_quantity"],
                "updated_at": str(found["updated_at"]) if found["updated_at"] is not None else None,
            },
        }
        wear = _to_float(entry.get("wear"))
        if wear:
            sticker["wear"] = wear
        stickers.append(sticker)
    keychains = []
    for slot, entry in enumerate(fields.get("keychains") or []):
        if isinstance(entry, str):
            entry = {"name": entry}
        found = catalog.find_keychain(entry.get("name") or entry.get("id") or "")
        if found is None:
            warnings.append(f"Unknown charm {entry.get('name')!r}, ignored")
            continue
        keychains.append(
            {
                "stickerId": int(found["keychain_id"]),
                "slot": slot,
                "name": found["name"],
                "pattern": entry.get("pattern"),
                "reference": {"price": found["reference_price_cents"], "quantity": found["reference_quantity"]},
            }
        )
    price = _to_float(fields.get("price_usd"))
    float_value = _to_float(fields.get("float_value"))
    item: dict[str, Any] = {
        "market_hash_name": name,
        "float_value": float_value,
        "paint_seed": fields.get("paint_seed"),
        "paint_index": fields.get("paint_index"),
        "stickers": stickers,
        "keychains": keychains,
    }
    reference: dict[str, Any] = {}
    source = None
    # CSFloat's reference block and item metadata of the closest recent sale of the item,
    # so a typed-in listing looks like the real listings the models were trained on.
    similar = catalog.similar_sale(name, float_value)
    if similar is not None:
        updated = similar["reference_last_updated"]
        reference = {
            "base_price": _plain(similar["reference_base_price"]),
            "predicted_price": _plain(similar["reference_predicted_price"]),
            "float_factor": _plain(similar["reference_float_factor"]),
            "quantity": _plain(similar["reference_quantity"]),
            "last_updated": None if pd.isna(updated) else pd.Timestamp(updated).isoformat(),
        }
        for key in ("paint_index", "def_index", "rarity", "rarity_name", "collection"):
            if item.get(key) is None:
                item[key] = _plain(similar[key])
        similar_float = "" if pd.isna(similar["float_value"]) else f" (float {similar['float_value']:.4f})"
        source = f"CSFloat reference of a similar recent sale{similar_float}"
    listing = {
        "id": "manual",
        "type": "buy_now",
        "price": round(price * 100) if price else None,
        "reference": reference,
        "item": item,
        "_reference_source": source,
    }
    return listing, warnings


def _plain(value: Any) -> Any:
    """numpy/pandas scalars -> plain Python, missing -> None."""
    if value is None or value is pd.NA:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def resolve(payload: dict[str, Any], catalog: Catalog) -> tuple[dict[str, Any], list[str]]:
    """Turn any accepted request body into a CSFloat-shaped listing."""
    if "listing" in payload:
        listing = payload["listing"]
        if isinstance(listing, str):
            listing = json.loads(listing)
        return listing, []
    if "url" in payload:
        listing_id = listing_id_from_text(str(payload["url"]))
        if not listing_id:
            raise ValueError("Could not find a CSFloat listing id in the link")
        return fetch_listing(listing_id), []
    if "manual" in payload:
        return manual_listing(payload["manual"], catalog)
    raise ValueError("Expected one of: 'listing' (CSFloat JSON), 'url', 'manual'")
