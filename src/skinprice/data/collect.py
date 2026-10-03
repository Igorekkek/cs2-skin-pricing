"""Collect sales history from CSFloat into SQLite.

For every market hash name the collector requests
``GET https://csfloat.com/api/v1/history/{name}/sales`` (the latest sales of
that item) and stores each sale's JSON untouched in the ``listings`` table.
Running it periodically accumulates history; already known sales are skipped.
Parsing is left to ``skinprice data build``, which uses the same normalizer as
the inference service.

An API key from your CSFloat profile is required (``CSFLOAT_API_KEY``).
Keep the request rate low; on HTTP 429 the collector waits and retries.
"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from skinprice import paths

HISTORY_API = "https://csfloat.com/api/v1/history/{name}/sales"

SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    listing_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    sold_at    TEXT,
    fetched_at TEXT NOT NULL,
    raw_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS listings_sold_at ON listings (sold_at);
"""


def fetch_sales(name: str, api_key: str, timeout: float = 20.0) -> list[dict]:
    url = HISTORY_API.format(name=urllib.parse.quote(name, safe=""))
    request = urllib.request.Request(url, headers={"Authorization": api_key, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https host
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("sales") or []
    return [sale for sale in payload if isinstance(sale, dict) and sale.get("id")]


def store(conn: sqlite3.Connection, sales: list[dict], fetched_at: str) -> int:
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO listings (listing_id, name, sold_at, fetched_at, raw_json) VALUES (?, ?, ?, ?, ?)",
        [
            (
                str(sale["id"]),
                (sale.get("item") or {}).get("market_hash_name") or "",
                sale.get("sold_at"),
                fetched_at,
                json.dumps(sale, separators=(",", ":")),
            )
            for sale in sales
        ],
    )
    conn.commit()
    return conn.total_changes - before


def collect(names: list[str], db_path: Path, delay_seconds: float = 3.0, api_key: str | None = None) -> dict:
    api_key = api_key or paths.setting("CSFLOAT_API_KEY")
    if not api_key:
        raise SystemExit("Set CSFLOAT_API_KEY (environment or .env) to collect data")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    stats = {"items": 0, "new_sales": 0, "errors": 0}
    try:
        for position, name in enumerate(names, 1):
            for attempt in range(4):
                try:
                    sales = fetch_sales(name, api_key)
                    break
                except urllib.error.HTTPError as error:
                    if error.code == 429 and attempt < 3:
                        wait = float(error.headers.get("Retry-After") or 60 * (attempt + 1))
                        print(f"rate limited, waiting {wait:.0f}s", flush=True)
                        time.sleep(wait)
                        continue
                    print(f"[{position}/{len(names)}] {name}: HTTP {error.code}", flush=True)
                    sales = None
                    break
                except (urllib.error.URLError, TimeoutError, ValueError) as error:
                    print(f"[{position}/{len(names)}] {name}: {error}", flush=True)
                    sales = None
                    break
            if sales is None:
                stats["errors"] += 1
            else:
                added = store(conn, sales, datetime.now(UTC).isoformat())
                stats["items"] += 1
                stats["new_sales"] += added
                print(f"[{position}/{len(names)}] {name}: {len(sales)} sales, {added} new", flush=True)
            time.sleep(delay_seconds)
    finally:
        conn.close()
    return stats
