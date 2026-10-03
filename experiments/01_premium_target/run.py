"""Predict the price, or the premium over CSFloat's own estimate?

Three CatBoost models, same data and split:
  * price_no_reference   - target log1p(price), no CSFloat reference features
                           (how the very first model of the project worked);
  * price_with_reference - target log1p(price), reference given as a feature;
  * premium              - target log1p(price) - log1p(reference): the model
                           only explains what the reference misses.

Usage: python experiments/01_premium_target/run.py [--force]
"""

from skinprice.features.spec import DEFAULT_GROUPS, FeatureSpec
from skinprice.research import Experiment

exp = Experiment("01_premium_target")
features = exp.features()

no_reference = FeatureSpec(
    groups=tuple(g for g in DEFAULT_GROUPS if g not in ("reference", "market_history")),
    drop=("skin_seed_key", "sticker_to_anchor_ratio", "log_reference_keychain_price"),
)
exp.variant("price_no_reference", "premium_catboost", features, params={"target": "price"}, spec=no_reference)
exp.variant("price_with_reference", "premium_catboost", features, params={"target": "price"})
exp.reuse("premium", "02_model_comparison", "premium_catboost")

by_sticker = {
    label: exp.segments(label, features, "sticker_value")[["sticker_value", "median_ape", "median_ratio"]]
    for label in ("price_no_reference", "price_with_reference", "premium")
}
from skinprice.research.experiment import to_markdown  # noqa: E402

extra = "\n\n".join(f"#### {label}: по стоимости наклеек\n\n{to_markdown(t)}" for label, t in by_sticker.items())
exp.report(extra=extra)
