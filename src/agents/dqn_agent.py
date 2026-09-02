"""Deep Q-network for sequential DNN layer placement.

Written from scratch in PyTorch rather than pulled from a library, so that every
component can be explained and inspected. It is a standard DQN with three
additions that matter for this problem:

**Masked action selection.** Devices without enough free memory are removed from
the action space, so the agent never has to learn a constraint the simulator can
state exactly, and every placement it produces is feasible by construction.

**Masked bootstrapping.** The same mask is applied inside the target
computation. This is the subtle half: if only the behaviour policy is masked,
the network is still trained towards ``max Q(s', a')`` over *all* actions, so it
learns values for placements the environment would refuse and propagates them
backwards through the episode. :mod:`tests.test_dqn` asserts the target ignores
masked actions.

**An undiscounted return.** The horizon is fixed and short, and the objective is
the undiscounted sum of per-layer costs, so ``discount = 1.0`` makes the quantity
the agent maximises exactly the quantity the experiments report.

Double DQN is used by default: the online network chooses the successor action
and the target network scores it, which reduces the systematic overestimation
that a single ``max`` over a noisy network produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from src.agents.base import action_mask_from_info
from src.config import DQNConfig

DEFAULT_CHECKPOINT_PATH = Path("checkpoints") / "dqn.pt"

# Masked actions are pushed to a large negative value rather than -inf, because
# -inf propagates NaNs through the loss if a whole row is ever masked.
MASKED_VALUE = -1e9


class QNetwork(nn.Module):
    """Multi-layer perceptron mapping a state to one Q-value per device."""

    def __init__(self, observation_size: int, num_actions: int, hidden_sizes: tuple[int, ...]):
        """Build the network.

        Args:
            observation_size: Width of the state vector.
            num_actions: Number of devices.
            hidden_sizes: Widths of the hidden layers.
        """
        super().__init__()
        layers: list[nn.Module] = []
        previous = observation_size
        for width in hidden_sizes:
            layers.append(nn.Linear(previous, width))
            layers.append(nn.ReLU())
            previous = width
        layers.append(nn.Linear(previous, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return Q-values for a batch of states."""
        return self.network(observations)


@dataclass
class Transition:
    """One environment step, as stored in the replay buffer."""

    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    next_action_mask: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity circular buffer of transitions.

    Successor action masks are stored alongside the transitions because the
    target computation needs them: without the mask, a sampled transition cannot
    be told which successor actions were legal.
    """

    def __init__(self, capacity: int, observation_size: int, num_actions: int, seed: int = 0):
        """Allocate the buffer.

        Args:
            capacity: Maximum number of transitions retained.
            observation_size: Width of the state vector.
            num_actions: Number of devices.
            seed: Seed for sampling.
        """
        self.capacity = int(capacity)
        self.observations = np.zeros((self.capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, observation_size), dtype=np.float32)
        self.next_action_masks = np.zeros((self.capacity, num_actions), dtype=bool)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._cursor = 0
        self._size = 0

    def __len__(self) -> int:
        """Number of transitions currently stored."""
        return self._size

    def add(self, transition: Transition) -> None:
        """Insert a transition, overwriting the oldest once full."""
        index = self._cursor
        self.observations[index] = transition.observation
        self.actions[index] = transition.action
        self.rewards[index] = transition.reward
        self.next_observations[index] = transition.next_observation
        self.next_action_masks[index] = transition.next_action_mask
        self.dones[index] = float(transition.done)
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Draw a uniform batch of transitions as tensors.

        Raises:
            ValueError: If the buffer holds fewer transitions than requested.
        """
        if self._size < batch_size:
            raise ValueError(
                f"cannot sample {batch_size} transitions from a buffer holding {self._size}"
            )
        indices = self._rng.integers(0, self._size, size=batch_size)
        return {
            "observations": torch.from_numpy(self.observations[indices]),
            "actions": torch.from_numpy(self.actions[indices]),
            "rewards": torch.from_numpy(self.rewards[indices]),
            "next_observations": torch.from_numpy(self.next_observations[indices]),
            "next_action_masks": torch.from_numpy(self.next_action_masks[indices]),
            "dones": torch.from_numpy(self.dones[indices]),
        }


class DQNAgent:
    """A masked, Double-DQN agent implementing the shared agent protocol."""

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        settings: DQNConfig,
        seed: int = 0,
        name: str = "dqn",
        device: str = "cpu",
    ) -> None:
        """Build the networks, the optimiser and the replay buffer.

        Args:
            observation_size: Width of the state vector.
            num_actions: Number of devices.
            settings: Hyper-parameters from configuration.
            seed: Seed for parameter initialisation, exploration and replay.
            name: Label used in results tables.
            device: Torch device. Every experiment here runs on CPU.
        """
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.settings = settings
        self.name = name
        self.torch_device = torch.device(device)

        torch.manual_seed(seed)
        self.online = QNetwork(observation_size, num_actions, settings.hidden_sizes).to(
            self.torch_device
        )
        self.target = QNetwork(observation_size, num_actions, settings.hidden_sizes).to(
            self.torch_device
        )
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimiser = torch.optim.Adam(self.online.parameters(), lr=settings.learning_rate)
        self.buffer = ReplayBuffer(settings.buffer_capacity, observation_size, num_actions, seed)
        self._rng = np.random.default_rng(seed)
        self.epsilon = settings.epsilon_start
        self.training_steps = 0

    # ----------------------------------------------------------------- acting

    def reset(self) -> None:
        """No per-episode state to clear."""

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        """Q-values for one state, as a NumPy array."""
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
            return self.online(tensor.to(self.torch_device)).cpu().numpy()[0]

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> int:
        """Greedily choose the best feasible device (used at evaluation time)."""
        mask = action_mask_from_info(info, self.num_actions)
        values = np.where(mask, self.q_values(observation), MASKED_VALUE)
        return int(np.argmax(values))

    def act_exploring(self, observation: np.ndarray, info: Mapping[str, Any]) -> int:
        """Choose epsilon-greedily among the feasible devices (used while training)."""
        mask = action_mask_from_info(info, self.num_actions)
        if self._rng.random() < self.epsilon:
            return int(self._rng.choice(np.flatnonzero(mask)))
        values = np.where(mask, self.q_values(observation), MASKED_VALUE)
        return int(np.argmax(values))

    def set_exploration(self, progress: float) -> None:
        """Anneal epsilon linearly over the first ``epsilon_decay_fraction`` of training."""
        fraction = min(1.0, progress / self.settings.epsilon_decay_fraction)
        start, end = self.settings.epsilon_start, self.settings.epsilon_end
        self.epsilon = start + (end - start) * fraction

    # --------------------------------------------------------------- learning

    def remember(self, transition: Transition) -> None:
        """Store a transition for replay."""
        self.buffer.add(transition)

    def learn(self) -> float | None:
        """Take one gradient step, or return ``None`` if the buffer is too small."""
        if len(self.buffer) < max(self.settings.learning_starts, self.settings.batch_size):
            return None

        batch = self.buffer.sample(self.settings.batch_size)
        observations = batch["observations"].to(self.torch_device)
        actions = batch["actions"].to(self.torch_device)
        rewards = batch["rewards"].to(self.torch_device)
        next_observations = batch["next_observations"].to(self.torch_device)
        next_masks = batch["next_action_masks"].to(self.torch_device)
        dones = batch["dones"].to(self.torch_device)

        predicted = self.online(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = self.compute_targets(rewards, next_observations, next_masks, dones)
        loss = nn.functional.smooth_l1_loss(predicted, targets)
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        if self.settings.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.settings.grad_clip_norm)
        self.optimiser.step()

        self.training_steps += 1
        if self.training_steps % self.settings.target_update_interval == 0:
            self.synchronise_target()
        return float(loss.item())

    def compute_targets(
        self,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_action_masks: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the temporal-difference targets for a batch.

        Kept as its own method because this is where masking is easy to get
        wrong: infeasible successor actions must be excluded from the bootstrap
        exactly as they are excluded from the behaviour policy. Otherwise the
        network is trained towards the value of placements the environment would
        refuse, and those values propagate backwards through the episode.

        Args:
            rewards: Rewards received, shape ``(batch,)``.
            next_observations: Successor states, shape ``(batch, observation_size)``.
            next_action_masks: Boolean feasibility of each successor action.
            dones: 1.0 where the episode terminated, else 0.0.

        Returns:
            The targets, shape ``(batch,)``.
        """
        with torch.no_grad():
            target_values = self.target(next_observations).masked_fill(
                ~next_action_masks, MASKED_VALUE
            )
            if self.settings.double_dqn:
                # The online network picks the successor action; the target
                # network scores it. Both see the same mask.
                online_next = self.online(next_observations).masked_fill(
                    ~next_action_masks, MASKED_VALUE
                )
                best_actions = online_next.argmax(dim=1, keepdim=True)
                bootstrap = target_values.gather(1, best_actions).squeeze(1)
            else:
                bootstrap = target_values.max(dim=1).values

            # A successor with no feasible action cannot arise in this environment
            # (the workload generator guarantees every layer fits somewhere), but
            # treating it as terminal keeps a malformed mask from poisoning the
            # batch with a huge target.
            has_feasible = next_action_masks.any(dim=1)
            bootstrap = torch.where(has_feasible, bootstrap, torch.zeros_like(bootstrap))

            return rewards + self.settings.discount * (1.0 - dones) * bootstrap

    def synchronise_target(self) -> None:
        """Copy the online parameters into the target network."""
        self.target.load_state_dict(self.online.state_dict())

    # -------------------------------------------------------------- artefacts

    def save(self, path: Path = DEFAULT_CHECKPOINT_PATH) -> Path:
        """Persist the online network and enough metadata to rebuild the agent."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.online.state_dict(),
                "observation_size": self.observation_size,
                "num_actions": self.num_actions,
                "hidden_sizes": list(self.settings.hidden_sizes),
                "training_steps": self.training_steps,
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls,
        path: Path = DEFAULT_CHECKPOINT_PATH,
        settings: DQNConfig | None = None,
        name: str = "dqn",
    ) -> "DQNAgent":
        """Rebuild an agent from a checkpoint, ready for greedy evaluation.

        Raises:
            FileNotFoundError: With a hint about the command that creates it.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Train the deep agent first: "
                "python -m src.training.train_dqn"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)

        if settings is None:
            from src.config import load_config

            settings = load_config().dqn
        import dataclasses

        settings = dataclasses.replace(settings, hidden_sizes=tuple(payload["hidden_sizes"]))

        agent = cls(payload["observation_size"], payload["num_actions"], settings, name=name)
        agent.online.load_state_dict(payload["state_dict"])
        agent.synchronise_target()
        agent.online.eval()
        agent.epsilon = 0.0
        agent.training_steps = int(payload.get("training_steps", 0))
        return agent


__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "MASKED_VALUE",
    "DQNAgent",
    "QNetwork",
    "ReplayBuffer",
    "Transition",
]
