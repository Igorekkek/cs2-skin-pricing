"""Which feature groups matter? Drop one group at a time, add two suspicious ones.

Trained on the latest 200k training rows: the original research found that
decisions about the *feature set* transfer from a subsample to the full data
(decisions about the training regime do not), and this keeps 9 runs cheap.

New groups tested here for the first time:
  * float_rank      - CSFloat's global float rank (low_rank / high_rank),
  * pattern_extras  - Case Hardened blue/gold share, Fade percentage;
both were in the raw API response but dropped by the original parser.
And two groups that should NOT be used:
  * lot_behavior    - watchers counted after the sale (look-ahead),
  * calendar        - hour and weekday of the sale.

Usage: python experiments/04_feature_ablation/run.py [--force]
"""

import pandas as pd

from skinprice.features.spec import DEFAULT_GROUPS, FeatureSpec
from skinprice.research import Experiment
from skinprice.research.experiment import to_markdown

BASE = {"max_train_rows": 200_000}
exp = Experiment("04_feature_ablation")
features = exp.features()

exp.variant("all_default", "premium_catboost", features, params=BASE)
for group in ("pattern_seed", "float_rank", "pattern_extras", "stickers", "keychains", "market_history", "reference"):
    spec = FeatureSpec(groups=tuple(g for g in DEFAULT_GROUPS if g != group))
    exp.variant(f"minus_{group}", "premium_catboost", features, params=BASE, spec=spec)
for group in ("lot_behavior", "calendar"):
    exp.variant(f"plus_{group}", "premium_catboost", features, params=BASE, spec=FeatureSpec(DEFAULT_GROUPS + (group,)))

# Noise band: the same configuration with two other random seeds.
for seed in (7, 123):
    exp.variant(f"all_default_seed{seed}", "premium_catboost", features, params={**BASE, "random_seed": seed})

# Removing market_history *improved* the subsample model. Too surprising to trust a
# single subsample run: re-check on the full training data with two different learners.
no_history = FeatureSpec(groups=tuple(g for g in DEFAULT_GROUPS if g != "market_history"))
for model in ("premium_catboost", "target_encoded_hgb"):
    exp.reuse(f"full_{model}", "02_model_comparison", model)
    exp.variant(f"full_{model}_minus_market_history", model, features, spec=no_history)

# The new groups only concern a few percent of rows: look at those rows.
test = features[features["split"] == "test"]
subsets = {
    "has float rank": test["log_low_rank"].notna() | test["log_high_rank"].notna(),
    "blue gem / fade data": test["blue_playside"].notna() | test["fade_percentage"].notna(),
}
rows = []
from skinprice.evaluation.metrics import price_metrics  # noqa: E402

for label in ("all_default", "minus_float_rank", "minus_pattern_extras"):
    pred = exp.predictions(label).set_index("listing_id").reindex(test["listing_id"])
    for subset, mask in subsets.items():
        m = price_metrics(test["price_cents"][mask], pred["pred_cents"].to_numpy()[mask.to_numpy()])
        rows.append({"variant": label, "subset": subset, "rows": m["rows"], "median_ape": m["median_ape"],
                     "mean_ape": m["mean_ape"], "mae_usd": m["mae_usd"]})
exp.report(extra="#### Только продажи, у которых есть новые поля\n\n" + to_markdown(pd.DataFrame(rows)))
