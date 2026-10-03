"""Parse CS2 market hash names: weapon, finish, wear, StatTrak/Souvenir."""

from __future__ import annotations

import pandas as pd

STATTRAK_PREFIX = "StatTrak™ "
SOUVENIR_PREFIX = "Souvenir "

WEAR_BANDS = {
    "Factory New": (0.00, 0.07),
    "Minimal Wear": (0.07, 0.15),
    "Field-Tested": (0.15, 0.38),
    "Well-Worn": (0.38, 0.45),
    "Battle-Scarred": (0.45, 1.00),
}


def parse_names(names: pd.Series) -> pd.DataFrame:
    """Split ``StatTrak™ AK-47 | Redline (Field-Tested)`` into its parts.

    ``weapon_type`` is the skin without wear and without the StatTrak/Souvenir
    prefix (``AK-47 | Redline``): the unit CSFloat's reference price and most
    history features are keyed on.
    """
    name = names.astype("string").fillna("").str.strip()
    is_stattrak = name.str.startswith(STATTRAK_PREFIX)
    is_souvenir = name.str.startswith(SOUVENIR_PREFIX)
    bare = name.str.removeprefix(STATTRAK_PREFIX).str.removeprefix(SOUVENIR_PREFIX)

    has_wear = bare.str.endswith(")") & bare.str.contains("(", regex=False)
    parts = bare.str.rsplit("(", n=1)
    weapon_type = bare.where(~has_wear, parts.str[0].str.strip())
    wear = parts.str[1].str[:-1].where(has_wear)
    weapon = weapon_type.str.split(" | ", n=1, regex=False).str[0]
    finish = weapon_type.str.split(" | ", n=1, regex=False).str[1]
    return pd.DataFrame(
        {
            "weapon_type": weapon_type.astype(str),
            "weapon": weapon.fillna("unknown").astype(str),
            "finish": finish.fillna("unknown").astype(str),
            "wear_name": wear.where(wear.isin(list(WEAR_BANDS)), "unknown").fillna("unknown").astype(str),
            "is_stattrak": is_stattrak.astype("int8"),
            "is_souvenir": is_souvenir.astype("int8"),
        },
        index=names.index,
    )
