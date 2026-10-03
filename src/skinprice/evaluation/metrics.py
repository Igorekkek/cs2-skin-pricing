"""Price accuracy metrics and segment breakdowns."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_BANDS = [0, 100, 500, 2_000, 10_000, 50_000, np.inf]
PRICE_LABELS = ["<$1", "$1-5", "$5-20", "$20-100", "$100-500", ">$500"]
STICKER_BANDS = [-np.inf, 0, 100, 500, 2_000, 10_000, 50_000, np.inf]
STICKER_LABELS = ["none", "<$1", "$1-5", "$5-20", "$20-100", "$100-500", ">$500"]


def price_metrics(
    true_cents: np.ndarray,
    pred_cents: np.ndarray,
    low_cents: np.ndarray | None = None,
    high_cents: np.ndarray | None = None,
) -> dict[str, float]:
    true = np.asarray(true_cents, dtype=float)
    pred = np.maximum(np.asarray(pred_cents, dtype=float), 1.0)
    ape = np.abs(pred - true) / true
    log_error = np.log1p(pred) - np.log1p(true)
    result = {
        "rows": int(len(true)),
        "mae_usd": float(np.mean(np.abs(pred - true)) / 100),
        "median_ape": float(np.median(ape) * 100),
        "mean_ape": float(np.mean(ape) * 100),
        "p90_ape": float(np.quantile(ape, 0.9) * 100),
        "within_10pct": float(np.mean(ape <= 0.10) * 100),
        "within_20pct": float(np.mean(ape <= 0.20) * 100),
        "rmse_log": float(np.sqrt(np.mean(log_error**2))),
        "median_ratio": float(np.median(pred / true)),
    }
    if low_cents is not None and high_cents is not None:
        low = np.asarray(low_cents, dtype=float)
        high = np.asarray(high_cents, dtype=float)
        result["interval_coverage"] = float(np.mean((true >= low) & (true <= high)) * 100)
        result["interval_width"] = float(np.median((high - low) / pred) * 100)
    return result


def segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Segment labels for every row of a feature table."""
    sticker_value = np.expm1(frame["log_sticker_sum_ref_price"]).where(frame["sticker_count"] > 0, -1)
    return pd.DataFrame(
        {
            "price_band": pd.cut(frame["price_cents"], PRICE_BANDS, labels=PRICE_LABELS, right=False),
            "sticker_value": pd.cut(sticker_value, STICKER_BANDS, labels=STICKER_LABELS),
            "has_charm": np.where(frame["keychain_count"] > 0, "charm", "no charm"),
            "pattern_info": np.where(
                frame["blue_playside"].notna() | frame["fade_percentage"].notna(),
                "blue gem / fade",
                "other",
            ),
        },
        index=frame.index,
    )


def segment_table(
    frame: pd.DataFrame,
    pred_cents: np.ndarray,
    by: str,
    low_cents: np.ndarray | None = None,
    high_cents: np.ndarray | None = None,
) -> pd.DataFrame:
    labels = segments(frame)[by]
    rows = []
    for label in labels.cat.categories if hasattr(labels, "cat") else sorted(labels.unique()):
        mask = (labels == label).to_numpy()
        if not mask.any():
            continue
        m = price_metrics(
            frame["price_cents"].to_numpy()[mask],
            np.asarray(pred_cents)[mask],
            None if low_cents is None else np.asarray(low_cents)[mask],
            None if high_cents is None else np.asarray(high_cents)[mask],
        )
        error_usd = np.abs(np.asarray(pred_cents)[mask] - frame["price_cents"].to_numpy()[mask]).sum() / 100
        rows.append({by: label, **m, "abs_error_usd_total": float(error_usd)})
    table = pd.DataFrame(rows)
    if len(table):
        table["share_of_rows"] = table["rows"] / table["rows"].sum() * 100
        table["share_of_error"] = table["abs_error_usd_total"] / table["abs_error_usd_total"].sum() * 100
    return table
