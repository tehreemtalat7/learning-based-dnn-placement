"""Placement agents.

Every agent implements the :class:`~src.agents.base.Agent` protocol, so the
evaluation harness drives heuristics, supervised models and learned policies
through exactly the same loop.
"""

from __future__ import annotations

from src.agents.base import Agent, action_mask_from_info, masked_argmin
from src.agents.greedy_agent import CRITERIA, GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.round_robin_agent import RoundRobinAgent


def build_heuristic_agents(config, seed: int | None = None) -> list[Agent]:
    """Instantiate every non-learning agent, in canonical reporting order.

    Args:
        config: The loaded configuration.
        seed: Seed for the random agent; defaults to ``config.seed``.

    Returns:
        Random, round-robin and the three greedy heuristics.
    """
    num_actions = config.num_devices
    return [
        RandomAgent(num_actions, seed=config.seed if seed is None else seed),
        RoundRobinAgent(num_actions),
        *(GreedyAgent(num_actions, criterion) for criterion in CRITERIA),
    ]


__all__ = [
    "CRITERIA",
    "Agent",
    "GreedyAgent",
    "RandomAgent",
    "RoundRobinAgent",
    "action_mask_from_info",
    "build_heuristic_agents",
    "masked_argmin",
]
