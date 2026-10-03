"""Where does the remaining error live, and how far is the model from the noise floor?

1. Error budget: share of test rows vs share of absolute dollar error by
   segment (sticker value, price band, charms).
2. Extrapolation: crafts whose exact sticker set never appeared in training
   versus seen sets.
3. Noise floor: identical clean items (same name, float within 0.005) sold
   on the same day still differ in price. The median deviation from the
   group median is the irreducible error for such items; compare the model
   on exactly those rows.

Uses the predictions of 02_model_comparison/premium_catboost.

Usage: python experiments/06_error_budget/run.py
"""

import numpy as np
import pandas as pd

from skinprice.evaluation.metrics import price_metrics
from skinprice.research import Experiment
from skinprice.research.experiment import GENERATED, to_markdown
from skinprice.research.labels import column

exp = Experiment("06_error_budget")
source = Experiment("02_model_comparison", force=False)
features = exp.features()
test = features[features["split"] == "test"].copy()
pred = source.predictions("premium_catboost").set_index("listing_id").reindex(test["listing_id"])
test["pred_cents"] = pred["pred_cents"].to_numpy()

budget = {
    by: source.segments("premium_catboost", features, by)[[by, "rows", "share_of_rows", "share_of_error", "median_ape", "mae_usd"]]
    for by in ("sticker_value", "price_band", "has_charm")
}

# 2. Seen vs novel sticker sets
slots = [f"sticker_{i}_id_cat" for i in range(5)]


def sticker_set(frame: pd.DataFrame) -> pd.Series:
    return frame[slots].apply(lambda row: "|".join(sorted(v for v in row if v != "-1")), axis=1)


train = features[features["split"] == "train"]
seen_sets = set(sticker_set(train[train["sticker_count"] > 0]))
crafts = test[test["sticker_count"] > 0].copy()
crafts["set_seen"] = sticker_set(crafts).isin(seen_sets).map({True: "seen in train", False: "novel"})
crafts["sticker_value"] = pd.cut(
    np.expm1(crafts["log_sticker_sum_ref_price"]), [0, 100, 500, 2_000, 10_000, 50_000, np.inf],
    labels=["<$1", "$1-5", "$5-20", "$20-100", "$100-500", ">$500"], include_lowest=True,
)
rows = []
for (value, seen), group in crafts.groupby(["sticker_value", "set_seen"], observed=True):
    m = price_metrics(group["price_cents"], group["pred_cents"])
    rows.append({"sticker_value": value, "set": seen, "rows": m["rows"], "median_ape": m["median_ape"]})
extrapolation = pd.DataFrame(rows).pivot(index="sticker_value", columns="set", values=["rows", "median_ape"])
extrapolation.columns = [f"{a} ({b})" for a, b in extrapolation.columns]
extrapolation = extrapolation.reset_index()
novel_share = float((crafts["set_seen"] == "novel").mean() * 100)

# 3. Noise floor on clean items: each sale against the median of the OTHER identical
# sales of that day (leave-one-out). This is an optimistic floor - it uses same-day
# prices the model cannot know - but it shows how much a perfect, contemporaneous
# comparable would still miss.
clean = test[(test["sticker_count"] == 0) & (test["keychain_count"] == 0) & test["float_value"].notna()].copy()
clean["day"] = clean["event_time"].dt.floor("D")
clean["float_bin"] = (clean["float_value"] / 0.005).astype(int)
keys = ["name", "day", "float_bin"]
clean["group_size"] = clean.groupby(keys)["price_cents"].transform("size")
twins = clean[clean["group_size"] >= 3].sort_values(keys).copy()


def loo_median(values: pd.Series) -> pd.Series:
    v = values.to_numpy(dtype=float)
    return pd.Series([np.median(np.delete(v, i)) for i in range(len(v))], index=values.index)


twins["others_median"] = twins.groupby(keys, group_keys=False)["price_cents"].apply(loo_median)
twins["price_band"] = pd.cut(twins["price_cents"], [0, 100, 500, 2_000, 10_000, np.inf],
                             labels=["<$1", "$1-5", "$5-20", "$20-100", ">$100"], right=False)
floor_rows = []
for band, group in [("all", twins)] + list(twins.groupby("price_band", observed=True)):
    floor_rows.append({
        "price_band": band, "rows": len(group),
        "floor_mdape_same_day_comparables": price_metrics(group["price_cents"], group["others_median"])["median_ape"],
        "model_mdape": price_metrics(group["price_cents"], group["pred_cents"])["median_ape"],
    })
floor = pd.DataFrame(floor_rows)

exp.record(
    "analysis",
    {
        "budget": {k: v.to_dict("records") for k, v in budget.items()},
        "novel_sticker_set_share_pct": novel_share,
        "extrapolation": extrapolation.to_dict("records"),
        "noise_floor": floor.to_dict("records"),
    },
)
text = "\n\n".join(f"#### Бюджет ошибки по сегментам: {column(by)}\n\n{to_markdown(t)}" for by, t in budget.items())
text += f"\n\n#### Новые наборы наклеек: у {novel_share:.1f}% тестовых крафтов такого набора не было в обучении\n\n"
text += to_markdown(extrapolation)
text += "\n\n#### Шум рынка: каждая продажа против медианы других таких же продаж того же дня\n\n" + to_markdown(floor)
(exp.dir / "results.md").write_text(f"{GENERATED}\n\n" + text + "\n", encoding="utf-8")
print(text)
