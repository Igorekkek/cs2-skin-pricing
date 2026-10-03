"""Experiment runner with resumable, version-controlled results."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skinprice import paths
from skinprice.evaluation.metrics import price_metrics, segment_table
from skinprice.features.build import load_features
from skinprice.features.spec import FeatureSpec
from skinprice.research.labels import russian
from skinprice.training import train_and_evaluate

SPLIT_NAMES = {"test": "тестовой", "val": "валидационной", "train": "обучающей"}
GENERATED = "<!-- сгенерировано run.py -->"

SUMMARY_COLUMNS = [
    "median_ape",
    "mean_ape",
    "p90_ape",
    "mae_usd",
    "within_10pct",
    "interval_coverage",
    "size_mb",
    "fit_minutes",
]


class Experiment:
    def __init__(self, slug: str, *, force: bool | None = None):
        self.slug = slug
        self.dir = paths.EXPERIMENTS_DIR / slug
        self.dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.dir / "results.json"
        self.results: dict[str, dict[str, Any]] = (
            json.loads(self.results_path.read_text(encoding="utf-8")) if self.results_path.exists() else {}
        )
        self.force = ("--force" in sys.argv) if force is None else force
        # --only a,b  trains just these variants now (others are reported from cache if present)
        self.only: set[str] | None = None
        if "--only" in sys.argv:
            self.only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
        self.runs_dir = paths.workspace().runs_dir / slug

    # --- data --------------------------------------------------------------------------
    @staticmethod
    def features(name: str = "features") -> pd.DataFrame:
        return load_features(name)

    # --- variants ------------------------------------------------------------------------
    def variant(
        self,
        label: str,
        model: str,
        features: pd.DataFrame,
        *,
        params: dict[str, Any] | None = None,
        spec: FeatureSpec | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        if label in self.results and not self.force:
            print(f"[{self.slug}] {label}: cached")
            return self.results[label]
        if self.only is not None and label not in self.only:
            print(f"[{self.slug}] {label}: skipped (--only)")
            return {}
        print(f"[{self.slug}] {label}: training {model} ...", flush=True)
        result = train_and_evaluate(model, features, params=params, spec=spec, version=label, directory=self.runs_dir / label)
        manifest = json.loads((result.directory / "manifest.json").read_text(encoding="utf-8"))
        record = {
            "model": model,
            "note": note,
            "params": manifest["params"],
            "features": manifest["features"],
            "metrics": result.metrics,
            "size_mb": manifest["size_mb"],
            "fit_minutes": round(manifest["info"].get("fit_seconds", 0) / 60, 1),
            "info": {k: v for k, v in manifest["info"].items() if k != "history"},
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self.record(label, record)
        m = result.metrics["test"]
        print(
            f"[{self.slug}] {label}: MdAPE {m['median_ape']:.2f}%  MAE ${m['mae_usd']:.2f}  "
            f"coverage {m.get('interval_coverage', float('nan')):.1f}%  size {record['size_mb']} MB",
            flush=True,
        )
        return record

    def reuse(self, label: str, other_slug: str, other_label: str | None = None) -> dict[str, Any]:
        """Copy a variant already trained by another experiment (same features, same split)."""
        other = Experiment(other_slug, force=False)
        other_label = other_label or label
        if other_label not in other.results:
            raise KeyError(f"Run experiments/{other_slug} first: it has no variant {other_label!r}")
        target = self.runs_dir / label
        target.mkdir(parents=True, exist_ok=True)
        source = other.runs_dir / other_label / "test_predictions.parquet"
        (target / "test_predictions.parquet").write_bytes(source.read_bytes())
        record = {**other.results[other_label], "note": f"reused from {other_slug}/{other_label}"}
        self.record(label, record)
        return record

    def record(self, label: str, record: dict[str, Any]) -> None:
        """Store any result (also non-model analyses) under ``label``."""
        self.results[label] = record
        self.results_path.write_text(json.dumps(self.results, indent=2, default=str), encoding="utf-8")

    def predictions(self, label: str) -> pd.DataFrame:
        return pd.read_parquet(self.runs_dir / label / "test_predictions.parquet")

    # --- reporting -----------------------------------------------------------------------
    def summary(self, split: str = "test") -> pd.DataFrame:
        rows = []
        for label, r in self.results.items():
            if "metrics" not in r:
                continue
            m = r["metrics"][split]
            rows.append(
                {
                    "variant": label,
                    "model": r["model"],
                    **{c: m.get(c) for c in SUMMARY_COLUMNS if c in m},
                    "size_mb": r.get("size_mb"),
                    "fit_minutes": r.get("fit_minutes"),
                }
            )
        return pd.DataFrame(rows)

    def report(self, split: str = "test", extra: str = "") -> str:
        table = self.summary(split)
        text = f"{GENERATED}\n\n### Результаты на {SPLIT_NAMES.get(split, split)} выборке\n\n"
        text += to_markdown(table) + "\n"
        if extra:
            text += "\n" + extra + "\n"
        (self.dir / "results.md").write_text(text, encoding="utf-8")
        print(text)
        return text

    def segments(self, label: str, features: pd.DataFrame, by: str) -> pd.DataFrame:
        test = features[features["split"] == "test"]
        pred = self.predictions(label).set_index("listing_id").reindex(test["listing_id"])
        return segment_table(test, pred["pred_cents"].to_numpy(), by, pred["low_cents"].to_numpy(), pred["high_cents"].to_numpy())


def to_markdown(table: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """Small dependency-free markdown table, with Russian column names and labels."""
    if table.empty:
        return "_результатов пока нет_"
    table = russian(table)
    cols = list(table.columns)

    def fmt(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if isinstance(value, float | np.floating):
            return format(float(value), floatfmt)
        return str(value)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


def ensemble_metrics(frame: pd.DataFrame, predictions: list[pd.DataFrame], weights: list[float] | None = None) -> dict:
    """Metrics of a geometric-mean blend of several models' test predictions."""
    test = frame[frame["split"] == "test"].set_index("listing_id")
    weights = weights or [1 / len(predictions)] * len(predictions)
    logs = {
        key: sum(w * np.log(p.set_index("listing_id").reindex(test.index)[key].to_numpy()) for w, p in zip(weights, predictions, strict=True))
        for key in ("pred_cents", "low_cents", "high_cents")
    }
    return price_metrics(test["price_cents"].to_numpy(), *(np.exp(logs[k]) for k in ("pred_cents", "low_cents", "high_cents")))


def write_json(path: Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
