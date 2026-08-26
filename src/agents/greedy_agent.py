"""Greedy placement heuristics.

Three heuristics of increasing sophistication, mirroring the ones in the
previous project so that the two studies stay comparable:

``fastest_device``
    Minimise the execution time of the current layer. Ignores communication
    entirely, and therefore tends to scatter consecutive layers across devices.

``communication_aware``
    Minimise execution time *plus* the cost of moving the incoming activation to
    the candidate device. Captures the coupling between neighbouring layers.

``objective_aware``
    Minimise the weighted objective contribution of this layer, i.e. latency,
    energy and the communication penalty together. This was the strongest
    heuristic in the previous project and is the primary comparison point here.

All three are *myopic*: each looks only at the layer it is placing. None of them
accounts for the effect of the current choice on later layers -- neither the
communication it will cause at the next boundary, nor the memory and utilisation
it consumes. That is precisely the gap the reinforcement learning agent is asked
to close, and the reason these heuristics are the right comparison.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.agents.base import action_mask_from_info, masked_argmin

CRITERIA = ("fastest_device", "communication_aware", "objective_aware")


class GreedyAgent:
    """Myopic placement using one of the criteria in :data:`CRITERIA`."""

    def __init__(self, num_actions: int, criterion: str = "objective_aware", name: str | None = None) -> None:
        """Create the agent.

        Args:
            num_actions: Size of the action space.
            criterion: One of :data:`CRITERIA`.
            name: Label used in results tables; defaults to the criterion name.

        Raises:
            ValueError: If ``criterion`` is not recognised.
        """
        if criterion not in CRITERIA:
            raise ValueError(f"unknown greedy criterion {criterion!r}; expected one of {CRITERIA}")
        self.num_actions = num_actions
        self.criterion = criterion
        self.name = name if name is not None else f"greedy_{criterion}"

    def reset(self) -> None:
        """No per-episode state to clear."""

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> int:
        """Choose the device minimising this heuristic's immediate cost.

        Raises:
            KeyError: If the environment did not publish the candidate costs,
                which happens only if ``act`` is called on a terminal step.
        """
        mask = action_mask_from_info(info, self.num_actions)
        return masked_argmin(self._costs(info), mask)

    def _costs(self, info: Mapping[str, Any]) -> np.ndarray:
        """Per-device immediate cost according to the selected criterion."""
        if self.criterion == "fastest_device":
            return np.asarray(info["candidate_execution_ms"], dtype=np.float64)
        if self.criterion == "communication_aware":
            return np.asarray(info["candidate_execution_ms"], dtype=np.float64) + np.asarray(
                info["candidate_communication_ms"], dtype=np.float64
            )
        return np.asarray(info["candidate_cost"], dtype=np.float64)


def fastest_device_agent(num_actions: int) -> GreedyAgent:
    """Greedy agent minimising execution time only."""
    return GreedyAgent(num_actions, "fastest_device")


def communication_aware_agent(num_actions: int) -> GreedyAgent:
    """Greedy agent minimising execution plus communication time."""
    return GreedyAgent(num_actions, "communication_aware")


def objective_aware_agent(num_actions: int) -> GreedyAgent:
    """Greedy agent minimising the immediate weighted objective."""
    return GreedyAgent(num_actions, "objective_aware")


__all__ = [
    "CRITERIA",
    "GreedyAgent",
    "communication_aware_agent",
    "fastest_device_agent",
    "objective_aware_agent",
]
