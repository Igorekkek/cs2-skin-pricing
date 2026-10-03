"""End to end: train tiny models on the sample, serve them, call the API."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from skinprice import models
from skinprice.features.history import HISTORY_COLUMNS, MarketSnapshot
from skinprice.inference.catalog import Catalog
from skinprice.inference.predictor import Predictor
from skinprice.server.app import create_app
from skinprice.split import parts

FIXTURES = Path(__file__).parent / "fixtures" / "listings.json"


@pytest.fixture(scope="module")
def predictor(sample_tables, sample_features):
    train, valid, _ = parts(sample_features)
    trained = {
        "premium_catboost": models.create("premium_catboost", {"iterations": 60, "thread_count": 4}).fit(train, valid),
        "csfloat_reference": models.create("csfloat_reference").fit(train, valid),
    }
    snapshot = MarketSnapshot.build(sample_features, pd.Series(True, index=sample_features.index))
    return Predictor(trained, snapshot, Catalog.build(sample_tables))


@pytest.fixture(scope="module")
def listing():
    return next(x for x in json.loads(FIXTURES.read_text(encoding="utf-8")) if x["item"].get("stickers"))


def test_json_listing_gets_training_features(predictor, listing, sample_features):
    """Static features from JSON equal the training row (time-dependent ones aside)."""
    frame, source = predictor.featurize(listing, now=pd.Timestamp(listing["sold_at"]))
    assert source == "CSFloat reference price"
    row = sample_features.set_index("listing_id").loc[listing["id"]]
    assert frame.iloc[0]["listed_at"] == row["listed_at"]
    skip = set(HISTORY_COLUMNS) | {"listing_id", "event_time", "listed_at", "split"}
    for column in frame.columns:
        if column in skip or column not in row.index:
            continue
        expected, actual = row[column], frame.iloc[0][column]
        if isinstance(expected, str) or isinstance(actual, str):
            assert str(actual) == str(expected), column
        else:
            np.testing.assert_allclose(float(actual), float(expected), rtol=1e-6, equal_nan=True, err_msg=column)


def test_predict_json(predictor, listing):
    result = predictor.predict({"listing": listing})
    assert set(result["models"]) == {"premium_catboost", "csfloat_reference"}
    estimate = result["ensemble"]
    assert estimate["low_usd"] <= estimate["price_usd"] <= estimate["high_usd"]
    assert "asking" in result  # the fixture has a sale price


def test_predict_manual_uses_closest_recent_sale(predictor, sample_tables):
    sales = sample_tables["sales"]
    name = sales["name"].value_counts().index[0]
    sticker = sample_tables["stickers"]["name"].dropna().iloc[0]
    result = predictor.predict({"manual": {"name": name, "float_value": 0.2, "stickers": [{"name": sticker}]}})
    assert result["anchor"]["source"].startswith("CSFloat reference of a similar recent sale")
    assert result["item"]["stickers"][0]["name"] == sticker
    estimate = result["ensemble"]
    assert 0 < estimate["low_usd"] < estimate["high_usd"]


def test_listing_without_reference_falls_back_to_recent_sales(predictor, listing):
    bare = {**listing, "reference": {}}
    result = predictor.predict({"listing": bare})
    assert result["anchor"]["source"].startswith("estimated")
    assert result["ensemble"]["price_usd"] > 0


def test_unknown_item_is_a_clear_error(predictor):
    with pytest.raises(ValueError, match="never sold"):
        predictor.predict({"manual": {"name": "Nonexistent | Skin (Factory New)"}})


def test_api(predictor, listing):
    client = TestClient(create_app(predictor))
    assert client.get("/api/health").json() == {"status": "ok"}
    assert "<title>CS2 Skin Pricer</title>" in client.get("/").text
    response = client.post("/api/predict", json={"listing": listing})
    assert response.status_code == 200
    assert response.json()["ensemble"]["price_usd"] > 0
    assert client.post("/api/predict", json={"nothing": 1}).status_code == 422
    assert client.get("/api/search", params={"kind": "stickers", "q": "sticker"}).status_code == 200
