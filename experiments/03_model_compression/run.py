"""Can the 9 GB CatBoost model be made small without losing accuracy?

Two levers:
  * max_ctr_complexity - how many categorical features CatBoost may combine
    into one CTR feature (default 4 on CPU). Each combination stores a table
    of statistics for every observed value tuple: with skin+seed keys and
    sticker ids this is where almost all of the 9 GB came from.
  * quantile_mode - three separate models (MAE + two quantiles, the original
    setup) versus one MultiQuantile model with three outputs.

Usage: python experiments/03_model_compression/run.py [--force] [--slow]
"""

import sys

from skinprice.features.spec import FeatureSpec
from skinprice.research import Experiment

exp = Experiment("03_model_compression")
features = exp.features()
# skin_seed_key (skin + pattern seed) has 429k distinct values in 575k training rows: an
# almost unique id whose combinations blow up the CTR tables. Pattern information is
# still available through the time-safe seed_past_premium feature.
with_seed_key = FeatureSpec(drop=())
without_seed_key = FeatureSpec(drop=("skin_seed_key",))  # the project default since this experiment

for label, params, spec in [
    ("multi_ctr1", {"quantile_mode": "multi", "max_ctr_complexity": 1}, with_seed_key),
    ("multi_ctr2", {"quantile_mode": "multi", "max_ctr_complexity": 2}, with_seed_key),
    ("multi_ctr2_no_seed_key", {"quantile_mode": "multi", "max_ctr_complexity": 2}, without_seed_key),
    ("multi_ctr1_no_seed_key", {"quantile_mode": "multi", "max_ctr_complexity": 1}, without_seed_key),
    ("separate_ctr1_no_seed_key", {"quantile_mode": "separate", "max_ctr_complexity": 1}, without_seed_key),
]:
    exp.variant(label, "premium_catboost", features, params=params, spec=spec)

# The original configuration (three models, CatBoost's default CTR complexity): ~3 hours
# and ~10 GB on disk, so only with --slow.
if "--slow" in sys.argv or "separate_ctr4" in exp.results:
    exp.variant("separate_ctr4", "premium_catboost", features,
                params={"quantile_mode": "separate", "max_ctr_complexity": 4}, spec=with_seed_key)

exp.report()
