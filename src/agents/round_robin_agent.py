"""Round-robin placement.

Cycles through the devices, skipping any that cannot host the current layer.
It ignores cost entirely, so it deliberately incurs a device switch at almost
every layer boundary -- which makes it a clean illustration of how expensive
communication is in this cost model.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.agents.base import action_mask_from_info


class RoundRobinAgent:
    """Assigns layers to devices in cyclic order, skipping infeasible ones."""

    def __init__(self, num_actions: int, name: str = "round_robin") -> None:
        """Create the agent.

        Args:
            num_actions: Size of the action space.
            name: Label used in results tables and figures.
        """
        self.num_actions = num_actions
        self.name = name
        self._cursor = 0

    def reset(self) -> None:
        """Restart the cycle so every episode begins at device 0."""
        self._cursor = 0

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> int:
        """Return the next feasible device in cyclic order."""
        mask = action_mask_from_info(info, self.num_actions)
        for offset in range(self.num_actions):
            candidate = (self._cursor + offset) % self.num_actions
            if mask[candidate]:
                self._cursor = candidate + 1
                return int(candidate)
        return int(np.flatnonzero(mask)[0])  # unreachable while the mask is non-empty
