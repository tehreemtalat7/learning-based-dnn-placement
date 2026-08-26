"""Common agent interface.

Every placement method in this repository -- heuristic, supervised, tabular or
deep -- is driven through the same two calls, so the evaluation loop is
identical for all of them and no method gets accidental extra information.

An agent sees the observation vector and the ``info`` mapping published by the
environment. ``info`` contains the action mask and the per-device cost of the
pending decision, which is exactly the information encoded (in normalised form)
in the observation. The greedy heuristics read the raw numbers from ``info``;
the learning agents read the observation. Neither sees anything the other could
not.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Agent(Protocol):
    """Protocol implemented by every placement method."""

    name: str

    def reset(self) -> None:
        """Prepare for a new episode. Stateless agents may do nothing."""

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> int:
        """Choose a device for the layer awaiting placement.

        Args:
            observation: The environment's state vector.
            info: The environment's info mapping, including ``"action_mask"``.

        Returns:
            The index of the chosen device. Implementations must respect the
            action mask whenever masking is enabled.
        """
        ...


def action_mask_from_info(info: Mapping[str, Any], num_actions: int) -> np.ndarray:
    """Extract the action mask, defaulting to "everything allowed".

    Args:
        info: The environment's info mapping.
        num_actions: Size of the action space.

    Returns:
        A boolean array of length ``num_actions``. If the mask forbids every
        action -- which the workload generator is designed to make impossible --
        a permissive mask is returned so that the caller fails loudly downstream
        rather than silently indexing an empty array.
    """
    mask = info.get("action_mask")
    if mask is None:
        return np.ones(num_actions, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.ones(num_actions, dtype=bool)
    return mask


def masked_argmin(values: np.ndarray, mask: np.ndarray) -> int:
    """Index of the smallest value among the allowed actions.

    Ties are broken by the lowest index, which keeps every heuristic
    deterministic given a scenario.
    """
    candidates = np.where(mask, values, np.inf)
    return int(np.argmin(candidates))


__all__ = ["Agent", "action_mask_from_info", "masked_argmin"]
