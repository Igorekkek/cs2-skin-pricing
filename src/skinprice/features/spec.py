"""Named feature groups. Models and ablation experiments select groups by name."""

from __future__ import annotations

from dataclasses import dataclass

from skinprice.features.history import HISTORY_COLUMNS
from skinprice.features.static import MAX_STICKER_SLOTS

_STICKER_SLOT_CATS = [f"sticker_{i}_id_cat" for i in range(MAX_STICKER_SLOTS)]
_STICKER_SLOT_NUMS = [f"sticker_{i}_wear" for i in range(MAX_STICKER_SLOTS)] + [
    f"log_sticker_{i}_ref_price" for i in range(MAX_STICKER_SLOTS)
]

# group -> (categorical columns, numeric columns)
GROUPS: dict[str, tuple[list[str], list[str]]] = {
    "identity": (
        ["weapon_type", "weapon", "wear_name", "rarity_name", "collection", "sale_type"],
        ["is_stattrak", "is_souvenir"],
    ),
    "pattern_seed": (["paint_index_cat", "skin_seed_key"], ["paint_seed"]),
    "float": ([], ["float_value", "log_float", "float_band_position", "is_extreme_low_float"]),
    "float_rank": ([], ["log_low_rank", "log_high_rank"]),
    "pattern_extras": (
        [],
        ["blue_playside", "blue_backside", "gold_playside", "gold_backside", "fade_percentage", "log_fade_rank"],
    ),
    "reference": (
        [],
        [
            "log_anchor",
            "has_reference_predicted",
            "log_reference_base_price",
            "reference_float_factor",
            "log_reference_quantity",
            "reference_lag_days",
        ],
    ),
    "stickers": (
        _STICKER_SLOT_CATS,
        [
            "sticker_count",
            "n_distinct_stickers",
            "n_unpriced_stickers",
            "log_sticker_sum_ref_price",
            "log_sticker_max_ref_price",
            "log_sticker_min_ref_price",
            "log_sticker_mean_ref_price",
            "log_sticker_sum_pristine_price",
            "mean_sticker_wear",
            "pristine_ratio",
            "n_katowice_2014",
            "n_katowice_2015",
            "n_premium_finish",
            "log_sticker_min_quantity",
            "has_slot_ge5",
            "craft_fill_ratio",
            "is_uniform_craft",
            "is_quad_craft",
            "sticker_to_anchor_ratio",
        ]
        + _STICKER_SLOT_NUMS,
    ),
    "keychains": (
        ["keychain_0_id_cat"],
        ["keychain_count", "log_keychain_ref_price", "log_keychain_min_quantity", "keychain_0_pattern",
         "log_reference_keychain_price"],
    ),
    "market_history": ([], list(HISTORY_COLUMNS)),
    # Off by default. Watchers are counted after the sale (look-ahead), and the
    # calendar showed <1% effect in the original experiments.
    "lot_behavior": ([], ["watchers", "is_watchlisted"]),
    "calendar": ([], ["listing_hour", "listing_dow"]),
}

DEFAULT_GROUPS = (
    "identity",
    "pattern_seed",
    "float",
    "float_rank",
    "pattern_extras",
    "reference",
    "stickers",
    "keychains",
    "market_history",
)


# skin_seed_key (skin + pattern seed) is nearly unique per sale (429k values in 575k
# training rows). As a categorical feature it inflates CatBoost's CTR tables ~6x and
# slightly hurts accuracy (experiment 03); pattern information still reaches the models
# through the time-safe seed_past_premium feature.
DEFAULT_DROP = ("skin_seed_key",)


@dataclass(frozen=True)
class FeatureSpec:
    groups: tuple[str, ...] = DEFAULT_GROUPS
    drop: tuple[str, ...] = DEFAULT_DROP

    def __post_init__(self) -> None:
        unknown = set(self.groups) - set(GROUPS)
        if unknown:
            raise ValueError(f"Unknown feature groups: {sorted(unknown)}")

    @property
    def categorical(self) -> list[str]:
        return [c for g in self.groups for c in GROUPS[g][0] if c not in self.drop]

    @property
    def numeric(self) -> list[str]:
        return [c for g in self.groups for c in GROUPS[g][1] if c not in self.drop]

    @property
    def columns(self) -> list[str]:
        return self.categorical + self.numeric

    def to_dict(self) -> dict:
        return {"groups": list(self.groups), "drop": list(self.drop)}

    @classmethod
    def from_dict(cls, data: dict) -> FeatureSpec:
        return cls(groups=tuple(data["groups"]), drop=tuple(data.get("drop", ())))
