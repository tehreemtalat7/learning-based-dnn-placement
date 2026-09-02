"""Tabular Q-learning over a discrete abstraction of the placement state.

Included for two reasons: as the simplest possible learner to compare against,
and as a concrete demonstration of *why* the deep agent is necessary rather than
an assertion that it is.

**The abstraction.** A table needs a finite state space, so the continuous state
is reduced to

    (layer index, device holding the activation, memory-occupancy bucket per device)

which for four devices, four buckets and a ten-layer DNN is 10 x 4 x 4^4 = 10 240
states. Everything else -- device speeds, energy rates, link latencies,
bandwidths, layer sizes -- is discarded, because those are continuous quantities
that differ in every scenario and cannot be tabulated.

**What that costs.** On a *single fixed scenario* the discarded quantities are
constants, so the abstraction loses nothing and the table converges to the
optimal policy for that problem. Trained across randomly sampled scenarios
(``q_learning.mode: pooled``) the same abstraction is severely aliased: two
states that look identical to the table may have a fast cloud link in one
scenario and a congested one in the other, and the single table has to average
over both. The training script measures both cases rather than asserting the
outcome.

This is exactly the gap function approximation exists to close, and it is the
argument for the deep agent that follows.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.agents.base import action_mask_from_info

DEFAULT_TABLE_PATH = Path("checkpoints") / "q_table.joblib"

StateKey = tuple[int, int, tuple[int, ...]]


@dataclass(frozen=True)
class QLearningHyperParameters:
    """Learning-rate and exploration settings for the tabular agent."""

    learning_rate: float
    discount: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_fraction: float
    memory_buckets: int


class TabularQAgent:
    """Q-learning over the discrete abstraction described in the module docstring."""

    def __init__(
        self,
        num_actions: int,
        hyper_parameters: QLearningHyperParameters,
        seed: int = 0,
        name: str = "tabular_q",
    ) -> None:
        """Create an agent with an empty table.

        Args:
            num_actions: Size of the action space.
            hyper_parameters: Learning and exploration settings.
            seed: Seed for exploration and tie-breaking.
            name: Label used in results tables.
        """
        self.num_actions = num_actions
        self.hyper_parameters = hyper_parameters
        self.name = name
        self._rng = np.random.default_rng(seed)
        self.table: dict[StateKey, np.ndarray] = defaultdict(
            lambda: np.zeros(num_actions, dtype=np.float64)
        )
        self.epsilon = hyper_parameters.epsilon_start

    # ------------------------------------------------------------ abstraction

    def state_key(self, info: Mapping[str, Any]) -> StateKey:
        """Map an environment info mapping to its discrete state.

        Args:
            info: The environment's info mapping.

        Returns:
            A hashable key. Memory occupancy is bucketed so that "this device is
            nearly full" is representable without tabulating a continuous value.
        """
        buckets = self.hyper_parameters.memory_buckets
        free_ratio = np.asarray(
            info.get("free_memory_ratio", np.ones(self.num_actions)), dtype=np.float64
        )
        occupancy = np.clip((free_ratio * buckets).astype(int), 0, buckets - 1)
        return (
            int(info.get("layer_index", 0)),
            int(info.get("previous_device", 0)),
            tuple(int(value) for value in occupancy),
        )

    # ----------------------------------------------------------------- acting

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> int:
        """Choose greedily among the feasible devices (used at evaluation time)."""
        mask = action_mask_from_info(info, self.num_actions)
        return self._greedy(self.state_key(info), mask)

    def act_exploring(self, info: Mapping[str, Any]) -> int:
        """Choose epsilon-greedily among the feasible devices (used while training)."""
        mask = action_mask_from_info(info, self.num_actions)
        if self._rng.random() < self.epsilon:
            return int(self._rng.choice(np.flatnonzero(mask)))
        return self._greedy(self.state_key(info), mask)

    def _greedy(self, key: StateKey, mask: np.ndarray) -> int:
        """Best known action among those the mask allows."""
        values = np.where(mask, self.table[key], -np.inf)
        best = float(np.max(values))
        # Break ties randomly so that an all-zero row does not always pick device 0.
        candidates = np.flatnonzero(values == best)
        return int(self._rng.choice(candidates))

    def reset(self) -> None:
        """No per-episode state to clear."""

    # --------------------------------------------------------------- learning

    def update(
        self,
        key: StateKey,
        action: int,
        reward: float,
        next_key: StateKey | None,
        next_mask: np.ndarray | None,
    ) -> float:
        """Apply one Q-learning update and return the temporal-difference error.

        Args:
            key: State the action was taken in.
            action: Action taken.
            reward: Reward received.
            next_key: Successor state, or ``None`` if the episode ended.
            next_mask: Feasible actions in the successor state. Bootstrapping is
                restricted to feasible actions, exactly as the behaviour policy
                is, so the table never learns a value for a placement the
                environment would refuse.

        Returns:
            The temporal-difference error, useful for monitoring convergence.
        """
        bootstrap = 0.0
        if next_key is not None:
            next_values = self.table[next_key]
            if next_mask is not None:
                next_values = np.where(next_mask, next_values, -np.inf)
            best = float(np.max(next_values))
            bootstrap = 0.0 if not np.isfinite(best) else best

        target = reward + self.hyper_parameters.discount * bootstrap
        error = target - self.table[key][action]
        self.table[key][action] += self.hyper_parameters.learning_rate * error
        return float(error)

    def set_exploration(self, progress: float) -> None:
        """Anneal epsilon linearly over the first ``epsilon_decay_fraction`` of training."""
        fraction = min(1.0, progress / self.hyper_parameters.epsilon_decay_fraction)
        start = self.hyper_parameters.epsilon_start
        end = self.hyper_parameters.epsilon_end
        self.epsilon = start + (end - start) * fraction

    # -------------------------------------------------------------- artefacts

    @property
    def states_visited(self) -> int:
        """How many distinct discrete states the table holds."""
        return len(self.table)

    def save(self, path: Path = DEFAULT_TABLE_PATH) -> Path:
        """Persist the table."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "table": {key: values.tolist() for key, values in self.table.items()},
                "num_actions": self.num_actions,
                "hyper_parameters": self.hyper_parameters,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_TABLE_PATH, seed: int = 0) -> "TabularQAgent":
        """Load a persisted table.

        Raises:
            FileNotFoundError: With a hint about the command that creates it.
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Train the tabular agent first: "
                "python -m src.training.train_q_learning"
            )
        payload = joblib.load(path)
        agent = cls(payload["num_actions"], payload["hyper_parameters"], seed=seed)
        for key, values in payload["table"].items():
            agent.table[key] = np.asarray(values, dtype=np.float64)
        agent.epsilon = 0.0
        return agent


def hyper_parameters_from_config(config) -> QLearningHyperParameters:
    """Build the hyper-parameter bundle from configuration."""
    section = config.q_learning
    return QLearningHyperParameters(
        learning_rate=section.learning_rate,
        discount=section.discount,
        epsilon_start=section.epsilon_start,
        epsilon_end=section.epsilon_end,
        epsilon_decay_fraction=section.epsilon_decay_fraction,
        memory_buckets=section.memory_buckets,
    )


__all__ = [
    "DEFAULT_TABLE_PATH",
    "QLearningHyperParameters",
    "StateKey",
    "TabularQAgent",
    "hyper_parameters_from_config",
]
