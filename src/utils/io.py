"""Reading and writing experiment artefacts, with provenance.

Layout:

* ``results/raw/`` -- one row per episode, written before any aggregation.
* ``results/processed/`` -- the aggregated tables that figures and the README
  are built from.
* ``results/figures/`` -- generated plots.

Every raw file is accompanied by a ``*.meta.json`` recording the configuration,
the git commit, the library versions and the wall-clock time. Without that, a
CSV six months later is just numbers; with it, the run can be reproduced.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import REPOSITORY_ROOT, Config

RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"
RAW_DIRECTORY = RESULTS_DIRECTORY / "raw"
PROCESSED_DIRECTORY = RESULTS_DIRECTORY / "processed"
FIGURES_DIRECTORY = RESULTS_DIRECTORY / "figures"


def ensure_directories() -> None:
    """Create the results directories if they do not already exist."""
    for directory in (RAW_DIRECTORY, PROCESSED_DIRECTORY, FIGURES_DIRECTORY):
        directory.mkdir(parents=True, exist_ok=True)


def git_commit() -> str:
    """Return the current commit hash, or ``"unknown"`` outside a repository."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def library_versions() -> dict[str, str]:
    """Versions of the libraries whose behaviour could change a result."""
    versions = {"python": platform.python_version()}
    for module_name in ("numpy", "pandas", "scipy", "torch", "sklearn", "gymnasium"):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        versions[module_name] = getattr(module, "__version__", "unknown")
    return versions


def save_raw(frame: pd.DataFrame, name: str, config: Config | None = None, **metadata: Any) -> Path:
    """Write per-episode rows to ``results/raw/<name>.csv`` with provenance.

    Args:
        frame: The rows to write.
        name: File stem, without extension.
        config: Configuration to record alongside the data.
        **metadata: Any additional facts worth recording about the run.

    Returns:
        The path written.
    """
    ensure_directories()
    path = RAW_DIRECTORY / f"{name}.csv"
    frame.to_csv(path, index=False)

    provenance: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "rows": int(len(frame)),
        "libraries": library_versions(),
        **metadata,
    }
    if config is not None:
        provenance["config"] = config.to_dict()
    path.with_suffix(".meta.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8"
    )
    return path


def save_processed(frame: pd.DataFrame, name: str) -> Path:
    """Write an aggregated table to ``results/processed/<name>.csv``."""
    ensure_directories()
    path = PROCESSED_DIRECTORY / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def load_raw(name: str) -> pd.DataFrame:
    """Read ``results/raw/<name>.csv``.

    Raises:
        FileNotFoundError: With a hint about which command regenerates the file.
    """
    path = RAW_DIRECTORY / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run the experiment that produces it "
            f"(see `make experiments`) before rebuilding figures."
        )
    return pd.read_csv(path)


def figure_path(name: str) -> Path:
    """Path of a figure file, creating the directory if needed."""
    ensure_directories()
    return FIGURES_DIRECTORY / f"{name}.png"


__all__ = [
    "FIGURES_DIRECTORY",
    "PROCESSED_DIRECTORY",
    "RAW_DIRECTORY",
    "RESULTS_DIRECTORY",
    "ensure_directories",
    "figure_path",
    "git_commit",
    "library_versions",
    "load_raw",
    "save_processed",
    "save_raw",
]
