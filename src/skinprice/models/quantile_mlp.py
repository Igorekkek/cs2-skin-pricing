"""Residual MLP with categorical embeddings and a non-crossing quantile head.

* every categorical feature gets an embedding (size grows with cardinality,
  rare values share an "unknown" slot);
* numeric features are robust-scaled (median / IQR), missing values get the
  median plus a missingness indicator;
* a stem layer and pre-norm residual blocks;
* the head predicts the median and two softplus gaps, so q15 <= q50 <= q85
  holds by construction;
* loss: pinball loss for the three quantiles plus a smooth-L1 term on the
  median; AdamW, reduce-on-plateau and early stopping on validation MAE.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from skinprice.models.base import HIGH_Q, LOW_Q, PriceModel

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - torch is an optional dependency
    torch = None


def embedding_dim(cardinality: int) -> int:
    return int(min(32, max(4, round(1.6 * cardinality**0.3))))


if torch is not None:

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, dropout: float):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.body = nn.Sequential(
                nn.Linear(width, width * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(width * 2, width),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return x + self.body(self.norm(x))

    class QuantileNet(nn.Module):
        def __init__(self, cardinalities, numeric_count, width=384, blocks=4, dropout=0.1):
            super().__init__()
            dims = [embedding_dim(c) for c in cardinalities]
            self.embeddings = nn.ModuleList(nn.Embedding(c, d) for c, d in zip(cardinalities, dims, strict=True))
            self.stem = nn.Sequential(
                nn.Linear(sum(dims) + numeric_count, width), nn.LayerNorm(width), nn.SiLU(), nn.Dropout(dropout)
            )
            self.blocks = nn.Sequential(*(ResidualBlock(width, dropout) for _ in range(blocks)))
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, 3))
            for embedding in self.embeddings:
                nn.init.normal_(embedding.weight, std=0.02)

        def forward(self, cats, nums):
            x = torch.cat([emb(cats[:, i]) for i, emb in enumerate(self.embeddings)] + [nums], dim=1)
            raw = self.head(self.blocks(self.stem(x)))
            center = raw[:, 1]
            return torch.stack([center - F.softplus(raw[:, 0]), center, center + F.softplus(raw[:, 2])], dim=1)


def _pinball(pred, target, q):
    error = target - pred
    return torch.maximum(q * error, (q - 1) * error)


class QuantileMLP(PriceModel):
    name = "quantile_mlp"
    summary = "Residual MLP with categorical embeddings and a non-crossing q15/q50/q85 head"
    default_params = {
        "width": 384,
        "blocks": 4,
        "dropout": 0.1,
        "epochs": 25,
        "batch_size": 4096,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 4,
        "min_category_count": 2,
        "max_vocab": 100_000,
        "threads": 0,
        "seed": 42,
    }

    def __init__(self, params=None, spec=None):
        super().__init__(params, spec)
        if torch is None:
            raise ImportError("quantile_mlp needs PyTorch: pip install 'skinprice[nn]'")
        self.vocab: dict[str, dict[str, int]] = {}
        self.medians: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        self.missing_columns: list[str] = []
        self.net: QuantileNet | None = None

    # --- preprocessing ---------------------------------------------------------------
    def _fit_preprocessing(self, train: pd.DataFrame) -> None:
        p = self.params
        for column in self.spec.categorical:
            counts = train[column].astype(str).value_counts()
            kept = counts[counts >= p["min_category_count"]].head(p["max_vocab"]).index
            self.vocab[column] = {value: i + 1 for i, value in enumerate(kept)}  # 0 = unknown
        for column in self.spec.numeric:
            values = pd.to_numeric(train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = float(values.median()) if values.notna().any() else 0.0
            iqr = float(values.quantile(0.75) - values.quantile(0.25)) if values.notna().any() else 1.0
            if not math.isfinite(iqr) or iqr < 1e-6:
                std = float(values.std()) if values.notna().sum() > 1 else 1.0
                iqr = std if math.isfinite(std) and std > 1e-6 else 1.0
            self.medians[column], self.scales[column] = median, iqr
        self.missing_columns = [c for c in self.spec.numeric if train[c].isna().any()]

    def _arrays(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cats = np.zeros((len(frame), len(self.spec.categorical)), dtype=np.int64)
        for i, column in enumerate(self.spec.categorical):
            cats[:, i] = frame[column].astype(str).map(self.vocab[column]).fillna(0).to_numpy(dtype=np.int64)
        nums = []
        for column in self.spec.numeric:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            scaled = (values.fillna(self.medians[column]) - self.medians[column]) / self.scales[column]
            nums.append(scaled.clip(-10, 10).to_numpy(dtype=np.float32))
        for column in self.missing_columns:
            nums.append(frame[column].isna().to_numpy(dtype=np.float32))
        return cats, np.column_stack(nums).astype(np.float32)

    def _build_net(self) -> QuantileNet:
        p = self.params
        cardinalities = [len(self.vocab[c]) + 1 for c in self.spec.categorical]
        numeric_count = len(self.spec.numeric) + len(self.missing_columns)
        return QuantileNet(cardinalities, numeric_count, p["width"], p["blocks"], p["dropout"])

    # --- training --------------------------------------------------------------------
    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> None:
        p = self.params
        torch.manual_seed(p["seed"])
        np.random.seed(p["seed"])
        if p["threads"]:
            torch.set_num_threads(int(p["threads"]))
        self._fit_preprocessing(train)
        cats, nums = self._arrays(train)
        target = np.clip(self.target(train).astype(np.float32), -6, 9)
        valid_cats, valid_nums = self._arrays(valid)
        valid_target = self.target(valid).astype(np.float32)

        self.net = self._build_net()
        self.info["parameters"] = int(sum(t.numel() for t in self.net.parameters()))
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
        cats_t, nums_t, target_t = map(torch.from_numpy, (cats, nums, target))

        best, best_state, stale, history = math.inf, None, 0, []
        for epoch in range(1, p["epochs"] + 1):
            started = time.time()
            lr = optimizer.param_groups[0]["lr"]
            train_loss = self._run_epoch(optimizer, cats_t, nums_t, target_t)
            valid_pred = self._forward(valid_cats, valid_nums)
            valid_mae = float(np.mean(np.abs(valid_pred[:, 1] - valid_target)))
            scheduler.step(valid_mae)
            history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss, "valid_mae": valid_mae,
                            "seconds": round(time.time() - started, 1)})
            print(f"epoch {epoch:02d} valid_mae={valid_mae:.5f} ({history[-1]['seconds']}s)", flush=True)
            if valid_mae < best - 1e-5:
                best, stale = valid_mae, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= p["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.info["history"] = history
        self.info["best_valid_mae"] = best
        self.info["best_epoch"] = int(np.argmin([h["valid_mae"] for h in history])) + 1

    def _run_epoch(self, optimizer, cats_t, nums_t, target_t) -> float:
        p = self.params
        self.net.train()
        order = torch.randperm(len(target_t))
        total = 0.0
        for start in range(0, len(order) - p["batch_size"] + 1, p["batch_size"]):
            idx = order[start : start + p["batch_size"]]
            out = self.net(cats_t[idx], nums_t[idx])
            y = target_t[idx]
            loss = (
                0.5 * _pinball(out[:, 0], y, LOW_Q)
                + 2.0 * _pinball(out[:, 1], y, 0.5)
                + 0.5 * _pinball(out[:, 2], y, HIGH_Q)
                + F.smooth_l1_loss(out[:, 1], y, reduction="none", beta=0.15)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)
            optimizer.step()
            total += loss.item() * len(idx)
        return total / max(len(order), 1)

    def _refit(self, full: pd.DataFrame) -> None:
        """Retrain from scratch on train+validation, replaying the chosen epochs and learning rates."""
        p = self.params
        torch.manual_seed(p["seed"])
        self._fit_preprocessing(full)
        cats, nums = self._arrays(full)
        target = np.clip(self.target(full).astype(np.float32), -6, 9)
        tensors = tuple(map(torch.from_numpy, (cats, nums, target)))
        self.net = self._build_net()
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
        for epoch in self.info["history"][: self.info["best_epoch"]]:
            for group in optimizer.param_groups:
                group["lr"] = epoch["lr"]
            self._run_epoch(optimizer, *tensors)

    @torch.inference_mode() if torch is not None else (lambda f: f)
    def _forward(self, cats: np.ndarray, nums: np.ndarray) -> np.ndarray:
        self.net.eval()
        out = []
        for start in range(0, len(cats), 65536):
            out.append(self.net(torch.from_numpy(cats[start : start + 65536]),
                                torch.from_numpy(nums[start : start + 65536])).numpy())
        return np.concatenate(out) if out else np.zeros((0, 3), dtype=np.float32)

    def _predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        return self._forward(*self._arrays(frame)).astype(float)

    # --- persistence -----------------------------------------------------------------
    def _save(self, directory: Path) -> None:
        torch.save(self.net.state_dict(), directory / "model.pt")
        state = {"vocab": self.vocab, "medians": self.medians, "scales": self.scales, "missing_columns": self.missing_columns}
        (directory / "preprocessing.json").write_text(json.dumps(state), encoding="utf-8")

    def _load(self, directory: Path) -> None:
        state = json.loads((directory / "preprocessing.json").read_text(encoding="utf-8"))
        self.vocab, self.medians, self.scales = state["vocab"], state["medians"], state["scales"]
        self.missing_columns = state["missing_columns"]
        self.net = self._build_net()
        self.net.load_state_dict(torch.load(directory / "model.pt", weights_only=True))
        self.net.eval()
