"""Deterministic seeding helpers.

Reproducibility rules used throughout this repository:

* Every stochastic component receives an explicit seed; nothing relies on a
  global default.
* Scenario seeds and learning seeds are kept separate, so the same evaluation
  scenarios can be replayed against a differently-seeded agent.
* :func:`derive_seed` turns a base seed plus a label into a stable child seed,
  which lets independent components (workload, devices, network) draw from
  independent streams without accidental correlation.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np

MAX_SEED = 2**31 - 1


def derive_seed(base_seed: int, label: str) -> int:
    """Derive a stable child seed from a base seed and a label.

    The mapping is a pure function of its inputs (no process-level randomness),
    so a given ``(base_seed, label)`` pair always yields the same child seed,
    across runs and across machines.

    Args:
        base_seed: The parent seed.
        label: A short name identifying the component, e.g. ``"workload"``.

    Returns:
        A non-negative seed below :data:`MAX_SEED`.
    """
    digest = hashlib.sha256(f"{int(base_seed)}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % MAX_SEED


def make_rng(seed: int, label: str | None = None) -> np.random.Generator:
    """Create a NumPy generator, optionally from a derived sub-stream.

    Args:
        seed: Base seed.
        label: Optional component label; when given, the generator is seeded
            with ``derive_seed(seed, label)`` instead of ``seed`` itself.

    Returns:
        A freshly seeded ``numpy.random.Generator``.
    """
    effective_seed = derive_seed(seed, label) if label is not None else int(seed)
    return np.random.default_rng(effective_seed)


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for a training run.

    Args:
        seed: The base seed.
        deterministic_torch: Whether to ask PyTorch for deterministic kernels.
            Slightly slower, but makes repeated runs bit-comparable on CPU.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    torch = _import_torch()
    if torch is None:
        return
    torch.manual_seed(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _import_torch() -> Any | None:
    """Import PyTorch lazily so that non-RL scripts do not pay for it."""
    try:
        import torch  # noqa: PLC0415 - deliberately lazy
    except ImportError:  # pragma: no cover - torch is a hard requirement in practice
        return None
    return torch


__all__ = ["MAX_SEED", "derive_seed", "make_rng", "seed_everything"]
