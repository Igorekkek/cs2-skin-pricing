"""Per-listing features that need nothing but the listing itself.

Everything here is computable from one CSFloat listing JSON, so the same
function serves both the training table and live inference.

Core idea of the whole project: CSFloat publishes its own estimate for every
item, ``reference.predicted_price``, which accounts for wear and float but
not for stickers, charms or rare patterns. For a plain item it is already a
very good price. The models therefore predict the log-premium over this
anchor rather than the raw price:

    log_premium = log1p(price) - log1p(anchor)
    price       = expm1(prediction + log1p(anchor))

The anchor is the predicted reference minus the charm component that CSFloat
folds into it (charms are modelled separately).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skinprice.features.identity import WEAR_BANDS, parse_names

MAX_STICKER_SLOTS = 5
EXTREME_LOW_FLOAT = 5e-4
PREMIUM_FINISH_PATTERN = r"\((?:Holo|Foil|Gold|Glitter|Lenticular)\)"


def _log1p(values: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(values, errors="coerce").clip(lower=0))


def _signed_log1p(values: pd.Series) -> pd.Series:
    return np.sign(values) * np.log1p(values.abs())


def _int_category(values: pd.Series) -> pd.Series:
    """Integer ids as strings, ``-1`` for missing (stable across float/int dtypes)."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.fillna(-1).round().astype("int64").astype(str)


def _text_category(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    return text.where(text.notna() & text.ne(""), "unknown").astype(str)


def reference_features(sales: pd.DataFrame, event_time: pd.Series) -> pd.DataFrame:
    predicted = pd.to_numeric(sales["reference_predicted_price"], errors="coerce")
    base = pd.to_numeric(sales["reference_base_price"], errors="coerce")
    predicted = predicted.where(predicted > 0)
    base = base.where(base > 0)

    # CSFloat's predicted price includes the attached charm; keep only a
    # coherent component (strictly between 0 and the reference itself).
    charm = pd.to_numeric(sales["reference_keychain_price"], errors="coerce").fillna(0)
    charm = charm.where((charm > 0) & predicted.notna() & (charm < predicted), 0.0)

    anchor = (predicted - charm).fillna(base)
    lag_days = (event_time - sales["reference_last_updated"]).dt.total_seconds() / 86400.0
    return pd.DataFrame(
        {
            "anchor_cents": anchor,
            "log_anchor": np.log1p(anchor),
            "has_reference_predicted": predicted.notna().astype("int8"),
            "log_reference_base_price": np.log1p(base.fillna(anchor)),
            "reference_float_factor": pd.to_numeric(sales["reference_float_factor"], errors="coerce"),
            "log_reference_quantity": _log1p(sales["reference_quantity"]),
            "log_reference_keychain_price": np.log1p(charm),
            "reference_lag_days": _signed_log1p(lag_days.clip(-365, 365)),
        },
        index=sales.index,
    )


def float_features(sales: pd.DataFrame, wear_name: pd.Series) -> pd.DataFrame:
    """Float on a log scale and its position inside the wear band.

    Collectors pay for the edges of a band and for extreme low floats; on a
    linear scale 0.001 and 0.00001 are neighbours for a tree.
    """
    fv = pd.to_numeric(sales["float_value"], errors="coerce")
    lo = wear_name.map(lambda w: WEAR_BANDS.get(w, (0.0, 1.0))[0]).astype(float)
    hi = wear_name.map(lambda w: WEAR_BANDS.get(w, (0.0, 1.0))[1]).astype(float)
    return pd.DataFrame(
        {
            "float_value": fv,
            "log_float": np.log10(fv.clip(lower=1e-7)),
            "float_band_position": ((fv - lo) / (hi - lo)).clip(0, 1),
            "is_extreme_low_float": (fv < EXTREME_LOW_FLOAT).astype("int8"),
            "log_low_rank": np.log1p(sales["low_rank"]),
            "log_high_rank": np.log1p(sales["high_rank"]),
        },
        index=sales.index,
    )


def pattern_features(sales: pd.DataFrame) -> pd.DataFrame:
    """Case Hardened blue/gold share and Fade percentage (when CSFloat sends them)."""
    return pd.DataFrame(
        {
            "blue_playside": sales["blue_playside"],
            "blue_backside": sales["blue_backside"],
            "gold_playside": sales["gold_playside"],
            "gold_backside": sales["gold_backside"],
            "fade_percentage": sales["fade_percentage"],
            "log_fade_rank": np.log1p(sales["fade_rank"]),
        },
        index=sales.index,
    )


def sticker_features(
    sales: pd.DataFrame,
    stickers: pd.DataFrame,
    event_time: pd.Series,
    *,
    max_reference_lag_days: float | None = None,
) -> pd.DataFrame:
    """Aggregates over all stickers plus the five most valuable ones by slot.

    Slots are ordered by sticker price, not by their position on the gun:
    a craft is a set, and two identical sets placed differently must look the
    same to the model.

    ``max_reference_lag_days`` masks sticker prices that CSFloat updated more
    than that many days after the sale (the collector stores the price seen at
    fetch time, which for backfilled sales can be far in the future).
    """
    out = pd.DataFrame(index=sales.index)
    keyed = pd.DataFrame({"listing_id": sales["listing_id"].values, "_row": np.arange(len(sales))})
    s = stickers.merge(keyed, on="listing_id", how="inner")

    price = s["reference_price_cents"].astype(float)
    if max_reference_lag_days is not None and len(s):
        sold = event_time.iloc[s["_row"].to_numpy()].reset_index(drop=True)
        lag = (s["reference_updated_at"].reset_index(drop=True) - sold).dt.total_seconds() / 86400
        price = price.where(~(lag > max_reference_lag_days).to_numpy())
    s = s.assign(
        price=price.values,
        # CSFloat omits wear for an unscraped sticker.
        wear=s["wear"].fillna(0.0),
        name=s["name"].fillna(""),
    )
    s["pristine"] = (s["wear"] <= 0).astype(int)
    s["pristine_price"] = s["price"].where(s["pristine"] == 1, 0.0)
    s["katowice_2014"] = s["name"].str.contains("Katowice 2014", regex=False).astype(int)
    s["katowice_2015"] = s["name"].str.contains("Katowice 2015", regex=False).astype(int)
    s["premium_finish"] = s["name"].str.contains(PREMIUM_FINISH_PATTERN, regex=True).astype(int)

    grouped = s.groupby("_row")
    agg = grouped.agg(
        sticker_count=("sticker_id", "size"),
        n_distinct_stickers=("sticker_id", "nunique"),
        n_unpriced_stickers=("price", lambda x: int(x.isna().sum())),
        sum_ref_price=("price", "sum"),
        max_ref_price=("price", "max"),
        min_ref_price=("price", "min"),
        sum_pristine_price=("pristine_price", "sum"),
        mean_sticker_wear=("wear", "mean"),
        n_pristine_stickers=("pristine", "sum"),
        n_katowice_2014=("katowice_2014", "sum"),
        n_katowice_2015=("katowice_2015", "sum"),
        n_premium_finish=("premium_finish", "sum"),
        min_ref_quantity=("reference_quantity", "min"),
        max_slot=("slot", "max"),
    )
    agg = agg.reindex(range(len(sales)))
    count = agg["sticker_count"].fillna(0)
    out["sticker_count"] = count.values
    out["n_distinct_stickers"] = agg["n_distinct_stickers"].fillna(0).values
    out["n_unpriced_stickers"] = agg["n_unpriced_stickers"].fillna(0).values
    for column in ("sum_ref_price", "max_ref_price", "min_ref_price", "sum_pristine_price"):
        out[f"log_sticker_{column}"] = np.log1p(agg[column].fillna(0).clip(lower=0)).values
    out["log_sticker_mean_ref_price"] = np.log1p(
        (agg["sum_ref_price"] / count.where(count > 0)).fillna(0)
    ).values
    out["mean_sticker_wear"] = agg["mean_sticker_wear"].values
    out["pristine_ratio"] = (agg["n_pristine_stickers"] / count.where(count > 0)).values
    out["n_katowice_2014"] = agg["n_katowice_2014"].fillna(0).values
    out["n_katowice_2015"] = agg["n_katowice_2015"].fillna(0).values
    out["n_premium_finish"] = agg["n_premium_finish"].fillna(0).values
    out["log_sticker_min_quantity"] = np.log1p(agg["min_ref_quantity"]).values
    out["has_slot_ge5"] = (agg["max_slot"] >= 5).astype("int8").values
    out["craft_fill_ratio"] = (count / MAX_STICKER_SLOTS).clip(upper=1).values
    out["is_uniform_craft"] = ((count > 1) & (agg["n_distinct_stickers"] == 1)).astype("int8").values

    # Canonical slots: most valuable sticker first.
    s = s.sort_values(["_row", "price", "sticker_id"], ascending=[True, False, True], na_position="last")
    s["rank"] = s.groupby("_row").cumcount()
    top = s[s["rank"] < MAX_STICKER_SLOTS]
    for i in range(MAX_STICKER_SLOTS):
        one = top[top["rank"] == i].set_index("_row").reindex(range(len(sales)))
        out[f"sticker_{i}_id_cat"] = _int_category(one["sticker_id"]).values
        out[f"sticker_{i}_wear"] = one["wear"].values
        out[f"log_sticker_{i}_ref_price"] = np.log1p(one["price"]).values
    first4 = [out[f"sticker_{i}_id_cat"] for i in range(4)]
    out["is_quad_craft"] = (
        (first4[0] != "-1") & (first4[0] == first4[1]) & (first4[0] == first4[2]) & (first4[0] == first4[3])
    ).astype("int8")
    return out


def keychain_features(sales: pd.DataFrame, keychains: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=sales.index)
    keyed = pd.DataFrame({"listing_id": sales["listing_id"].values, "_row": np.arange(len(sales))})
    k = keychains.merge(keyed, on="listing_id", how="inner")
    k = k.sort_values(["_row", "reference_price_cents"], ascending=[True, False], na_position="last")
    agg = k.groupby("_row").agg(
        keychain_count=("keychain_id", "size"),
        max_price=("reference_price_cents", "max"),
        min_quantity=("reference_quantity", "min"),
    ).reindex(range(len(sales)))
    first = k.drop_duplicates("_row").set_index("_row").reindex(range(len(sales)))
    out["keychain_count"] = agg["keychain_count"].fillna(0).values
    out["log_keychain_ref_price"] = np.log1p(agg["max_price"].fillna(0).clip(lower=0)).values
    out["log_keychain_min_quantity"] = np.log1p(agg["min_quantity"]).values
    out["keychain_0_id_cat"] = _int_category(first["keychain_id"]).values
    out["keychain_0_pattern"] = first["pattern"].values
    return out


def static_features(
    tables: dict[str, pd.DataFrame],
    *,
    event_time: pd.Series | None = None,
    max_sticker_reference_lag_days: float | None = None,
) -> pd.DataFrame:
    """One row per sale. ``event_time`` defaults to ``sold_at`` (training)."""
    sales = tables["sales"].reset_index(drop=True)
    if event_time is None:
        event_time = sales["sold_at"]
    event_time = pd.to_datetime(pd.Series(event_time).reset_index(drop=True), utc=True)
    event_time.index = sales.index

    names = parse_names(sales["name"])
    frame = pd.DataFrame(
        {
            "listing_id": sales["listing_id"].astype(str),
            "name": sales["name"].astype(str),
            "event_time": event_time,
            # When the lot was put up for sale; not a feature, used by the
            # trading analysis (time on market).
            "listed_at": pd.to_datetime(sales["created_at"], utc=True),
            "price_cents": pd.to_numeric(sales["price_cents"], errors="coerce"),
        }
    )
    frame = frame.join(names)
    frame["rarity_name"] = _text_category(sales["rarity_name"])
    frame["collection"] = _text_category(sales["collection"])
    frame["sale_type"] = _text_category(sales["sale_type"])
    frame["paint_index_cat"] = _int_category(sales["paint_index"])
    frame["paint_seed"] = pd.to_numeric(sales["paint_seed"], errors="coerce")
    frame["skin_seed_key"] = frame["weapon_type"] + "_" + _int_category(sales["paint_seed"])

    frame = frame.join(float_features(sales, frame["wear_name"]))
    frame = frame.join(pattern_features(sales))
    frame = frame.join(reference_features(sales, event_time))
    frame = frame.join(
        sticker_features(
            sales,
            tables["stickers"],
            event_time,
            max_reference_lag_days=max_sticker_reference_lag_days,
        )
    )
    frame = frame.join(keychain_features(sales, tables["keychains"]))
    frame["sticker_to_anchor_ratio"] = np.log1p(
        np.expm1(frame["log_sticker_sum_ref_price"]) / frame["anchor_cents"]
    )

    # Features describing the seller's lot rather than the item. Kept for the
    # ablation study only: watchers are counted after the sale.
    frame["watchers"] = pd.to_numeric(sales["watchers"], errors="coerce")
    frame["is_watchlisted"] = pd.to_numeric(sales["is_watchlisted"], errors="coerce")
    frame["listing_hour"] = event_time.dt.hour
    frame["listing_dow"] = event_time.dt.dayofweek

    frame["log_price"] = np.log1p(frame["price_cents"])
    frame["log_premium"] = frame["log_price"] - frame["log_anchor"]
    return frame
