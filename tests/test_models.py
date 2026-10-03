import numpy as np
import pytest

from skinprice import models
from skinprice.split import parts

FAST = {
    "premium_catboost": {"iterations": 60, "thread_count": 4},
    "target_encoded_hgb": {"max_iter": 40},
    "quantile_mlp": {"epochs": 1, "width": 32, "blocks": 1, "batch_size": 1024},
}


@pytest.mark.parametrize("name", sorted(models.REGISTRY))
def test_fit_predict_save_load(name, sample_features, tmp_path):
    train, valid, test = parts(sample_features)
    model = models.create(name, FAST.get(name)).fit(train, valid)
    prediction = model.predict(test)

    assert prediction.premium.shape == (len(test), 3)
    assert np.isfinite(prediction.price).all()
    assert (prediction.low <= prediction.price + 1e-6).all()
    assert (prediction.high >= prediction.price - 1e-6).all()

    model.save(tmp_path)
    restored = models.load(tmp_path)
    np.testing.assert_allclose(restored.predict(test).price, prediction.price, rtol=1e-5)


def test_calibration_hits_validation_coverage(sample_features):
    train, valid, _ = parts(sample_features)
    model = models.create("csfloat_reference").fit(train, valid)
    p = model.predict(valid)
    true = valid["price_cents"].to_numpy()
    coverage = np.mean((true >= p.low) & (true <= p.high))
    assert 0.65 <= coverage <= 0.75


def test_split_is_chronological(sample_features):
    train, valid, test = parts(sample_features)
    assert train["event_time"].max() < valid["event_time"].min()
    assert valid["event_time"].max() < test["event_time"].min()
