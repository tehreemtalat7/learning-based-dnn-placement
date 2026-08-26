"""Uniformly random placement over the feasible devices.

The weakest baseline, and the one the objective's normalisation is calibrated
against: because the cost references are the expected cost of a uniformly random
placement, this agent should score close to 1.0. That makes it a useful sanity
check on the whole cost model as well as a comparison point.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.agents.base import action_mask_from_info


class RandomAgent:
    """Chooses uniformly among the devices that can host the current layer."""

    def __init__(self, num_actions: int, seed: int = 0, name: str = "random") -> None:
        """Create the agent.

        Args:
            num_actions: Size of the action space.
            seed: Seed for the agent's own generator, kept separate from the
                scenario seeds so that re-running an evaluation with a different
                agent seed does not change which problems are solved.
            name: Label used in results tables and figures.
        """
        self.num_actions = num_actions
        self.name = name
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """No per-episode state to clear."""

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> int:
        """Pick a feasible device uniformly at random."""
        mask = action_mask_from_info(info, self.num_actions)
        feasible = np.flatnonzero(mask)
        return int(self._rng.choice(feasible))
