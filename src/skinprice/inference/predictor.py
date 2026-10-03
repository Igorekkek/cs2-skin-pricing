"""Price a listing with every trained model.

``Predictor`` loads the latest version of each model, the market snapshot
and the catalog from the workspace. ``predict`` runs the listing through the
same normalizer and feature code as training and returns each model's
estimate, an ensemble estimate and, when the listing has an asking price,
how that price compares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skinprice import models, paths
from skinprice.data.normalize import listings_to_tables
from skinprice.features.history import MarketSnapshot
from skinprice.features.static import static_features
from skinprice.inference.catalog import Catalog
from skinprice.inference.listing import resolve
from skinprice.models.base import PriceModel
from skinprice.training import latest_dir

DEFAULT_MODELS = ("premium_catboost", "target_encoded_hgb", "quantile_mlp", "csfloat_reference")
BASELINES = {"csfloat_reference", "recent_median"}


def _usd(cents: float | None) -> float | None:
    return None if cents is None or not np.isfinite(cents) else round(float(cents) / 100, 2)


@dataclass
class Predictor:
    models: dict[str, PriceModel]
    snapshot: MarketSnapshot
    catalog: Catalog
    versions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_workspace(cls, names: tuple[str, ...] | list[str] | None = None, snapshot_dir: Path | None = None) -> Predictor:
        snapshot_dir = Path(snapshot_dir) if snapshot_dir else paths.workspace().snapshot_dir
        loaded, versions = {}, {}
        for name in names or DEFAULT_MODELS:
            try:
                directory = latest_dir(name)
            except FileNotFoundError:
                continue
            loaded[name] = models.load(directory)
            versions[name] = directory.name
        if not loaded:
            raise FileNotFoundError("No trained models found. Run `skinprice train <model>` first.")
        return cls(loaded, MarketSnapshot.load(snapshot_dir), Catalog.load(snapshot_dir), versions)

    def featurize(self, listing: dict[str, Any], now: datetime | None = None) -> tuple[pd.DataFrame, str]:
        tables = listings_to_tables([listing])
        sales = tables["sales"]
        name = str(sales.at[0, "name"])
        reference_source = listing.get("_reference_source") or "CSFloat reference price"
        has_reference = (sales.at[0, "reference_predicted_price"] or 0) > 0 or (sales.at[0, "reference_base_price"] or 0) > 0
        if not has_reference:
            fallback = self.snapshot.fallback_anchor(name)
            if fallback is None:
                raise ValueError(
                    f"{name!r}: the listing has no CSFloat reference price and the item never sold in our data"
                )
            sales.loc[0, "reference_predicted_price"] = fallback
            reference_source = "estimated from recent sales of this item"
        # The moment the listing is priced: the sale time for a historical sale, the
        # snapshot time for a typed-in listing (its reference comes from that market
        # state), otherwise now.
        if now is None:
            if pd.notna(sales.at[0, "sold_at"]):
                now = sales.at[0, "sold_at"]
            elif listing.get("id") == "manual":
                now = self.snapshot.as_of
            else:
                now = datetime.now(UTC)
        event_time = pd.Series([pd.Timestamp(now)])
        frame = static_features(tables, event_time=event_time)
        frame = frame.join(self.snapshot.features_for(frame))
        return frame, reference_source

    def predict(self, payload: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
        listing, warnings = resolve(payload, self.catalog)
        frame, reference_source = self.featurize(listing, now)
        row = frame.iloc[0]

        estimates = {}
        for name, model in self.models.items():
            p = model.predict(frame)
            estimates[name] = {
                "price_usd": _usd(p.price[0]),
                "low_usd": _usd(p.low[0]),
                "high_usd": _usd(p.high[0]),
                "summary": model.summary,
                "version": self.versions.get(name),
            }
        learned = [v for k, v in estimates.items() if k not in BASELINES] or list(estimates.values())
        ensemble = {
            key: round(float(np.exp(np.mean([np.log(v[key]) for v in learned]))), 2)
            for key in ("price_usd", "low_usd", "high_usd")
        }

        stickers = [
            {"name": s.get("name"), "wear": s.get("wear"), "reference_usd": _usd((s.get("reference") or {}).get("price"))}
            for s in (listing.get("item") or {}).get("stickers") or []
        ]
        result: dict[str, Any] = {
            "item": {
                "name": row["name"],
                "skin": row["weapon_type"],
                "wear": row["wear_name"],
                "float": None if pd.isna(row["float_value"]) else float(row["float_value"]),
                "paint_seed": None if pd.isna(row["paint_seed"]) else int(row["paint_seed"]),
                "stickers": stickers,
                "charms": [k.get("name") for k in (listing.get("item") or {}).get("keychains") or []],
            },
            "anchor": {"price_usd": _usd(row["anchor_cents"]), "source": reference_source},
            "history": {
                "sales_of_this_skin": int(round(np.expm1(row["log_wt_prior_sales"]))),
                "snapshot_as_of": self.snapshot.as_of.isoformat(),
            },
            "models": estimates,
            "ensemble": ensemble,
            "warnings": warnings,
        }
        asking = row["price_cents"]
        if pd.notna(asking) and asking > 0:
            asking_usd = float(asking) / 100
            result["asking"] = {
                "price_usd": round(asking_usd, 2),
                "discount_to_estimate_pct": round((1 - asking_usd / ensemble["price_usd"]) * 100, 1),
                "below_interval": asking_usd < ensemble["low_usd"],
                "above_interval": asking_usd > ensemble["high_usd"],
            }
        return result
