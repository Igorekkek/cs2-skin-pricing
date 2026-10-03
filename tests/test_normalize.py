"""The JSON path used at inference must agree with the database columns used for training."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skinprice.data.normalize import listings_to_tables

FIXTURES = Path(__file__).parent / "fixtures" / "listings.json"


@pytest.fixture(scope="module")
def listings():
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def test_fixture_covers_interesting_cases(listings):
    items = [x["item"] for x in listings]
    assert any(i.get("stickers") for i in items)
    assert any(i.get("keychains") for i in items)
    assert any(i.get("low_rank") or i.get("high_rank") for i in items)


def test_sales_rows_match_dataset(listings, sample_tables):
    from_json = listings_to_tables(listings)["sales"].set_index("listing_id")
    from_db = sample_tables["sales"].set_index("listing_id").loc[from_json.index]
    numeric = [
        "price_cents", "reference_predicted_price", "reference_base_price", "float_value",
        "paint_seed", "paint_index", "low_rank", "high_rank", "blue_playside", "fade_percentage",
    ]
    for column in numeric:
        np.testing.assert_allclose(
            from_json[column].to_numpy(dtype=float), from_db[column].to_numpy(dtype=float),
            equal_nan=True, err_msg=column,
        )
    assert (from_json["name"] == from_db["name"]).all()
    assert (from_json["sold_at"] == from_db["sold_at"]).all()


def test_sticker_rows_match_dataset(listings, sample_tables):
    from_json = listings_to_tables(listings)["stickers"]
    ids = set(from_json["listing_id"])
    from_db = sample_tables["stickers"][sample_tables["stickers"]["listing_id"].isin(ids)]
    key = ["listing_id", "position"]
    merged = from_json.merge(from_db, on=key, suffixes=("_json", "_db"))
    assert len(merged) == len(from_json) == len(from_db)
    np.testing.assert_allclose(merged["sticker_id_json"], merged["sticker_id_db"])
    np.testing.assert_allclose(merged["slot_json"], merged["slot_db"])
    np.testing.assert_allclose(merged["reference_price_cents_json"], merged["reference_price_cents_db"])
    pd.testing.assert_series_equal(
        merged["wear_json"].fillna(-1), merged["wear_db"].fillna(-1), check_names=False, atol=1e-6
    )
