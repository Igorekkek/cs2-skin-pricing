"""Do sticker prices observed after the sale inflate the accuracy on crafts?

The collector stores CSFloat's sticker reference price as it was at fetch
time. For a sale fetched weeks later that price is from the future: the
model may learn sticker values it could not have known. At serving time the
prices are live, so the question is only whether the *evaluation* is
optimistic.

Variants (same model, features rebuilt with sticker prices masked when
updated more than N days after the sale):
  * lag_all - no masking (original setup)
  * lag_30  - mask prices updated >30 days after the sale
  * lag_3   - mask prices updated >3 days after the sale

All variants are also scored on the "fresh" test crafts, whose sticker prices
were all updated at most 3 days after the sale: on those rows the features
are identical across variants, so differences there are pure model effects.

Usage: python experiments/05_sticker_reference_leakage/run.py [--force]
"""

import numpy as np
import pandas as pd

from skinprice import paths
from skinprice.data.io import load_tables
from skinprice.evaluation.metrics import price_metrics
from skinprice.features.build import FeatureConfig, build_and_save, feature_path, load_features
from skinprice.research import Experiment
from skinprice.research.experiment import to_markdown

exp = Experiment("05_sticker_reference_leakage")
variants = {"lag_all": "features", "lag_30": "features_lag30", "lag_3": "features_lag3"}
for label, name in variants.items():
    if label != "lag_all" and not feature_path(name).exists():
        build_and_save(name=name, config=FeatureConfig(max_sticker_reference_lag_days=float(label.split("_")[1])))

exp.reuse("lag_all", "02_model_comparison", "premium_catboost")
for label in ("lag_30", "lag_3"):
    exp.variant(label, "premium_catboost", load_features(variants[label]))

# Per listing: the largest delay between the sale and a sticker price update.
tables = load_tables(paths.workspace().dataset_dir)
stickers = tables["stickers"].merge(tables["sales"][["listing_id", "sold_at"]], on="listing_id")
lag = (stickers["reference_updated_at"] - stickers["sold_at"]).dt.total_seconds() / 86400
max_lag = lag.fillna(np.inf).groupby(stickers["listing_id"]).max()

features = load_features("features")
test = features[features["split"] == "test"].set_index("listing_id")
crafts = test[test["sticker_count"] > 0]
lags = max_lag.reindex(crafts.index)
subsets = {
    "all test": test.index,
    "crafts": crafts.index,
    "fresh crafts (all sticker prices <=3d)": crafts.index[lags <= 3],
    "crafts with a price 3-30d late": crafts.index[(lags > 3) & (lags <= 30)],
    "stale crafts (some price >30d late)": crafts.index[lags > 30],
}
rows = []
for label in variants:
    pred = exp.predictions(label).set_index("listing_id")
    for subset, index in subsets.items():
        if len(index) == 0:  # e.g. no test sale can be fetched >30 days after it happened
            rows.append({"variant": label, "subset": subset, "rows": 0})
            continue
        m = price_metrics(test.loc[index, "price_cents"], pred.loc[index, "pred_cents"])
        rows.append({"variant": label, "subset": subset, "rows": m["rows"], "median_ape": m["median_ape"],
                     "mean_ape": m["mean_ape"], "mae_usd": m["mae_usd"], "median_ratio": m["median_ratio"]})
train_crafts = features[(features["split"] == "train") & (features["sticker_count"] > 0)]
train_lags = max_lag.reindex(train_crafts["listing_id"])
share = pd.DataFrame(
    {"sticker price lag": ["<=3 days", "3-30 days", ">30 days or unknown"],
     "share of train crafts %": [100 * np.mean(train_lags <= 3), 100 * np.mean((train_lags > 3) & (train_lags <= 30)),
                                 100 * np.mean(train_lags > 30)],
     "share of test crafts %": [100 * np.mean(lags <= 3), 100 * np.mean((lags > 3) & (lags <= 30)), 100 * np.mean(lags > 30)]}
)
table = pd.DataFrame(rows)
exp.record("subsets", {"rows": table.to_dict("records"), "lag_share": share.to_dict("records")})
exp.report(extra=f"#### По подвыборкам\n\n{to_markdown(table)}\n\n#### Крафты по задержке цены наклеек (максимум по наклейкам лота)\n\n{to_markdown(share)}")
