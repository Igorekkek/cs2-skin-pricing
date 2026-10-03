"""Build the parquet dataset from a SQLite database of sales.

Two database layouts are supported:

* ``raw``    - written by ``skinprice collect``: one table ``listings`` with
               the untouched CSFloat JSON of every sale. Normalization happens
               here, through the same code the inference service uses.
* ``legacy`` - the original project's parser: ``sales`` with parsed columns,
               ``sale_stickers``, ``sale_keychains``. Fields that only exist
               in ``sales.raw_json`` (float rank, blue gem, fade) are parsed out.

Only completed sales with a positive price are exported.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from skinprice.data import normalize
from skinprice.data.schema import KEYCHAIN_COLUMNS, SALES_COLUMNS, STICKER_COLUMNS

BUILDER_VERSION = 2
CHUNK_ROWS = 50_000
_EXTRA_MARKERS = ("low_rank", "high_rank", "blue_gem", '"fade"')


def detect_layout(conn: sqlite3.Connection) -> str:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "listings" in tables:
        return "raw"
    if {"sales", "sale_stickers"} <= tables:
        return "legacy"
    raise ValueError(f"Unrecognized database layout, tables: {sorted(tables)}")


def _extras_from_raw(raw: object) -> dict:
    if not isinstance(raw, str) or not any(marker in raw for marker in _EXTRA_MARKERS):
        return {}
    try:
        item = json.loads(raw).get("item") or {}
    except (ValueError, AttributeError):
        return {}
    return normalize.item_extras(item)


_LEGACY_SALES_SQL = """
SELECT listing_id, name, price_cents,
       reference_base_price, reference_predicted_price, reference_float_factor,
       reference_quantity, reference_keychain_price, reference_last_updated,
       float_value, paint_seed, paint_index,
       item_def_index AS def_index, item_rarity AS rarity, item_rarity_name AS rarity_name,
       COALESCE(item_collection_name, item_collection) AS collection,
       sale_type, is_seller, is_watchlisted, watchers, created_at, sold_at, raw_json
FROM sales
WHERE state = 'sold' AND price_cents > 0 AND name IS NOT NULL
"""

_LEGACY_STICKERS_SQL = """
SELECT st.listing_id, st.sticker_index AS position, st.slot, st.sticker_id, st.name, st.wear,
       COALESCE(st.reference_price_cents, st.price_cents) AS reference_price_cents,
       st.reference_quantity, st.reference_updated_at
FROM sale_stickers st JOIN sales s ON s.listing_id = st.listing_id
WHERE s.state = 'sold' AND s.price_cents > 0
"""

_LEGACY_KEYCHAINS_SQL = """
SELECT k.listing_id, k.keychain_index AS position, k.slot, k.keychain_id, k.name, k.pattern,
       k.price_cents AS reference_price_cents, k.reference_quantity, k.reference_updated_at
FROM sale_keychains k JOIN sales s ON s.listing_id = k.listing_id
WHERE s.state = 'sold' AND s.price_cents > 0
"""


def _read_legacy(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    chunks = []
    for chunk in pd.read_sql_query(_LEGACY_SALES_SQL, conn, chunksize=CHUNK_ROWS):
        extras = pd.DataFrame([_extras_from_raw(raw) for raw in chunk["raw_json"]], index=chunk.index)
        chunk = chunk.drop(columns=["raw_json"])
        for column in extras.columns:
            chunk[column] = extras[column]
        chunks.append(chunk.reindex(columns=SALES_COLUMNS))
    sales = pd.concat(chunks, ignore_index=True)

    tables = {"sales": sales}
    known = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    tables["stickers"] = pd.read_sql_query(_LEGACY_STICKERS_SQL, conn).reindex(columns=STICKER_COLUMNS)
    if "sale_keychains" in known:
        tables["keychains"] = pd.read_sql_query(_LEGACY_KEYCHAINS_SQL, conn).reindex(
            columns=KEYCHAIN_COLUMNS
        )
    else:
        tables["keychains"] = pd.DataFrame(columns=KEYCHAIN_COLUMNS)
    return tables


def _read_raw(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {"sales": [], "stickers": [], "keychains": []}
    query = "SELECT raw_json FROM listings ORDER BY sold_at"
    for chunk in pd.read_sql_query(query, conn, chunksize=CHUNK_ROWS):
        listings = [json.loads(raw) for raw in chunk["raw_json"]]
        tables = normalize.listings_to_tables([x for x in listings if x.get("state") == "sold"])
        for key, frame in tables.items():
            parts[key].append(frame)
    return {key: pd.concat(frames, ignore_index=True) for key, frames in parts.items() if frames}


def _clean(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    tables = normalize.coerce_tables(tables)
    sales = tables["sales"]
    sales = sales[sales["price_cents"] > 0].drop_duplicates("listing_id", keep="last")
    sales = sales.sort_values(["sold_at", "listing_id"]).reset_index(drop=True)
    known_ids = set(sales["listing_id"])

    stickers = tables["stickers"]
    stickers = stickers[stickers["listing_id"].isin(known_ids)]
    # A handful of rows report wear in percent instead of a 0..1 fraction.
    wear = stickers["wear"]
    stickers = stickers.assign(wear=wear.where(wear <= 1, wear / 100.0).clip(0, 1))
    stickers = stickers.drop_duplicates(["listing_id", "position"])

    keychains = tables["keychains"]
    keychains = keychains[keychains["listing_id"].isin(known_ids)]
    keychains = keychains.drop_duplicates(["listing_id", "position"])
    return {
        "sales": sales,
        "stickers": stickers.reset_index(drop=True),
        "keychains": keychains.reset_index(drop=True),
    }


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path, manifest: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, frame in tables.items():
        frame.to_parquet(out_dir / f"{key}.parquet", index=False, compression="zstd")
    sales = tables["sales"]
    manifest = {
        **manifest,
        "builder_version": BUILDER_VERSION,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": {key: int(len(frame)) for key, frame in tables.items()},
        "sold_at_min": sales["sold_at"].min().isoformat() if len(sales) else None,
        "sold_at_max": sales["sold_at"].max().isoformat() if len(sales) else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_dataset(db_path: Path, out_dir: Path) -> dict:
    started = time.time()
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        layout = detect_layout(conn)
        tables = _read_legacy(conn) if layout == "legacy" else _read_raw(conn)
    finally:
        conn.close()
    tables = _clean(tables)
    manifest = write_tables(
        tables,
        Path(out_dir),
        {"source": Path(db_path).name, "source_layout": layout},
    )
    manifest["seconds"] = round(time.time() - started, 1)
    return manifest
