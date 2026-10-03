"""Model registry: name -> class, creation and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skinprice.features.spec import FeatureSpec
from skinprice.models.base import Prediction, PriceModel, read_manifest
from skinprice.models.baselines import CSFloatReference, RecentMedian
from skinprice.models.premium_catboost import PremiumCatBoost
from skinprice.models.target_encoded_hgb import TargetEncodedHGB

REGISTRY: dict[str, type[PriceModel]] = {
    cls.name: cls for cls in (CSFloatReference, RecentMedian, PremiumCatBoost, TargetEncodedHGB)
}

try:
    from skinprice.models.quantile_mlp import QuantileMLP

    REGISTRY[QuantileMLP.name] = QuantileMLP
except ImportError:  # pragma: no cover
    pass


def create(name: str, params: dict[str, Any] | None = None, spec: FeatureSpec | None = None) -> PriceModel:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](params, spec)


def load(directory: Path) -> PriceModel:
    manifest = read_manifest(directory)
    return REGISTRY[manifest["model"]].from_manifest(Path(directory), manifest)


__all__ = ["REGISTRY", "PriceModel", "Prediction", "create", "load"]
