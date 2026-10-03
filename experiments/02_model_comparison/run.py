"""All models on the same features and the same chronological split.

Usage: python experiments/02_model_comparison/run.py [--force]
"""

import pandas as pd

from skinprice.research import Experiment
from skinprice.research.experiment import ensemble_metrics, to_markdown
from skinprice.research.labels import column

MODELS = ["csfloat_reference", "recent_median", "target_encoded_hgb", "quantile_mlp", "premium_catboost"]
LEARNED = ["premium_catboost", "target_encoded_hgb", "quantile_mlp"]

exp = Experiment("02_model_comparison")
features = exp.features()
for name in MODELS:
    exp.variant(name, name, features)

predictions = [exp.predictions(name) for name in LEARNED]
exp.record("ensemble_all3", {"model": "geometric mean of 3", "metrics": {"test": ensemble_metrics(features, predictions)}})
exp.record(
    "ensemble_catboost_mlp",
    {"model": "geometric mean of 2", "metrics": {"test": ensemble_metrics(features, [predictions[0], predictions[2]])}},
)

sections = []
for by in ("price_band", "sticker_value", "has_charm", "pattern_info"):
    pivot = pd.concat(
        {name: exp.segments(name, features, by).set_index(by)["median_ape"] for name in MODELS}, axis=1
    ).reset_index()
    rows = exp.segments("premium_catboost", features, by).set_index(by)[["rows", "share_of_rows", "share_of_error"]]
    pivot = pivot.join(rows, on=by)
    sections.append(f"#### Медианная ошибка (%) по сегментам: {column(by)}\n\n{to_markdown(pivot)}")
exp.report(extra="\n\n".join(sections))
