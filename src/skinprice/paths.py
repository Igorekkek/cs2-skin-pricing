"""Filesystem layout and local settings.

Heavy files (datasets, feature tables, trained models, snapshots) live in a
workspace directory outside git. Its location is read from the
``SKINPRICE_WORKSPACE`` setting: an environment variable or a line in the
repository's ``.env`` file. Without it the workspace is ``./workspace``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
CONFIG_DIR = REPO_ROOT / "configs"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def setting(name: str, default: str | None = None) -> str | None:
    """Environment variable first, then ``.env`` in the repo root, then default."""
    value = os.environ.get(name)
    if value:
        return value
    value = _read_dotenv(REPO_ROOT / ".env").get(name)
    return value if value else default


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def snapshot_dir(self) -> Path:
        return self.root / "snapshot"

    @property
    def runs_dir(self) -> Path:
        """Scratch output of experiments (predictions, large intermediate files)."""
        return self.root / "runs"

    def model_dir(self, model: str, version: str) -> Path:
        return self.models_dir / model / version


def workspace() -> Workspace:
    return Workspace(_resolve(setting("SKINPRICE_WORKSPACE", "./workspace")))


def source_db() -> Path:
    default = str(workspace().root / "csfloat_history.db")
    return _resolve(setting("SKINPRICE_SOURCE_DB", default))
