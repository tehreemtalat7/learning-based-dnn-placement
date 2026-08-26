"""The Gymnasium environment for sequential DNN layer placement.

One episode places one DNN: the agent is shown the current layer and the state
of every device, chooses a device, and receives the negative of that step's
weighted cost as reward. The episode ends when the last layer has been placed.

**Return equals the negative objective.** The weighted objective is a linear
function of latency, energy and communication, so summing the per-step costs
reproduces the objective of the complete placement exactly. Training signal and
evaluation metric are therefore the same quantity, and an undiscounted episode
return of ``-0.8`` means "this placement scored 0.8 on the objective", i.e. 20 %
cheaper than a uniformly random placement.

**Infeasible devices.** With ``environment.invalid_action_mode: mask`` (the
default) devices without enough free memory are removed from the action space
via the mask published in ``info["action_mask"]``; choosing one is a programming
error and raises. With ``penalty`` the choice is allowed, the device is
over-subscribed, a memory violation is recorded and a fixed penalty is charged --
which reproduces the failure mode of the previous project's independent
per-layer predictions. Experiment 5 compares the two.
"""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import Config
from src.environment.devices import DeviceFleet
from src.environment.observation import ObservationBuilder, observation_size
from src.environment.reward import (
    PlacementResult,
    step_costs,
    weighted_cost,
)
from src.environment.scenario import Scenario, sample_scenario, training_seeds
from src.utils.seed import make_rng


class InvalidActionError(ValueError):
    """Raised when an agent selects a device that action masking forbids."""


class DNNPlacementEnv(gym.Env):
    """Sequential placement of DNN layers onto heterogeneous devices."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Config,
        *,
        scenario_seeds: Sequence[int] | None = None,
        num_layers: int | Sequence[int] | None = None,
        seed: int | None = None,
    ) -> None:
        """Create an environment over a distribution of scenarios.

        Args:
            config: The loaded configuration.
            scenario_seeds: Explicit scenario seeds, consumed in order and then
                wrapped around. Used for evaluation, where every method must see
                exactly the same problems. When ``None``, each episode draws a
                fresh seed from the training pool.
            num_layers: DNN depth. An integer fixes it; a sequence samples one
                depth per episode (used to train a single policy across sizes);
                ``None`` uses ``workload.num_layers`` from the configuration.
            seed: Seed for the environment's own sampling of scenario seeds and
                depths. Independent of the seeds of the scenarios themselves.
        """
        super().__init__()
        self.config = config
        self.scenario_seeds = list(scenario_seeds) if scenario_seeds is not None else None
        self.num_layers_option = num_layers
        self._sampler_rng = make_rng(seed if seed is not None else config.seed, "env_sampler")
        self._seed_cursor = 0

        self.observation_builder = ObservationBuilder(config)
        self.action_space = spaces.Discrete(config.num_devices)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size(config.num_devices),),
            dtype=np.float32,
        )

        self.scenario: Scenario | None = None
        self.fleet: DeviceFleet | None = None
        self._layer_index = 0
        self._previous_device = 0
        self._placement: list[int] = []
        self._compute_latency_ms = 0.0
        self._communication_ms = 0.0
        self._energy = 0.0
        self._memory_violations = 0
        self._invalid_attempts = 0
        self._terminated = True

    # --------------------------------------------------------------- episode

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode on a new scenario.

        Args:
            seed: Reseeds the environment's scenario sampler (not the scenario).
            options: May contain ``"scenario"`` (a ready-made
                :class:`~src.environment.scenario.Scenario`), ``"scenario_seed"``
                (an integer) and/or ``"num_layers"``.

        Returns:
            The first observation and an info mapping containing the action mask
            and the per-device cost of the pending decision.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._sampler_rng = make_rng(seed, "env_sampler")
            self._seed_cursor = 0

        options = options or {}
        scenario = options.get("scenario")
        if scenario is None:
            scenario_seed = options.get("scenario_seed", self._next_scenario_seed())
            depth = options.get("num_layers", self._next_num_layers())
            scenario = sample_scenario(self.config, int(scenario_seed), num_layers=depth)

        self.scenario = scenario
        self.fleet = DeviceFleet(scenario.devices, self.config.environment)
        self._layer_index = 0
        self._previous_device = scenario.input_source_index
        self._placement = []
        self._compute_latency_ms = 0.0
        self._communication_ms = 0.0
        self._energy = 0.0
        self._memory_violations = 0
        self._invalid_attempts = 0
        self._terminated = False

        return self._observe()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Place the current layer on the chosen device.

        Args:
            action: Index of the device to use.

        Returns:
            ``(observation, reward, terminated, truncated, info)``. The reward is
            the negative weighted cost of this layer; on the final step ``info``
            additionally contains ``"result"``, a
            :class:`~src.environment.reward.PlacementResult`.

        Raises:
            RuntimeError: If called before :meth:`reset` or after termination.
            InvalidActionError: If action masking is enabled and the chosen
                device cannot host the layer.
        """
        if self._terminated or self.scenario is None or self.fleet is None:
            raise RuntimeError("step() called on a finished environment; call reset() first")

        device_index = int(action)
        if not 0 <= device_index < self.config.num_devices:
            raise InvalidActionError(
                f"action {device_index} is outside 0..{self.config.num_devices - 1}"
            )

        scenario = self.scenario
        fleet = self.fleet
        layer_index = self._layer_index
        layer = scenario.workload[layer_index]
        feasibility = fleet.feasibility_mask(layer.memory_gb)

        penalty = 0.0
        if not feasibility[device_index]:
            self._invalid_attempts += 1
            if self.config.environment.uses_action_masking:
                raise InvalidActionError(
                    f"device {scenario.device_names[device_index]!r} cannot host layer "
                    f"{layer.name!r} ({layer.memory_gb:.3f} GB needed, "
                    f"{fleet.free_memory_gb()[device_index]:.3f} GB free). "
                    "Respect info['action_mask'] or set environment.invalid_action_mode=penalty."
                )
            self._memory_violations += 1
            penalty = self.config.environment.invalid_action_penalty

        execution_ms, communication_ms, energy = self._candidate_costs(layer_index)
        step_execution = float(execution_ms[device_index])
        step_communication = float(communication_ms[device_index])
        step_energy = float(energy[device_index])

        step_cost = weighted_cost(
            latency_ms=step_execution + step_communication,
            communication_ms=step_communication,
            energy=step_energy,
            references=scenario.references,
            weights=scenario.weights,
            comm_double_count=self.config.objective.comm_double_count,
        )
        reward = -(step_cost + penalty)

        self._compute_latency_ms += step_execution
        self._communication_ms += step_communication
        self._energy += step_energy
        fleet.assign(device_index, layer.compute_cost, layer.memory_gb)
        self._placement.append(device_index)
        self._previous_device = device_index
        self._layer_index += 1
        self._terminated = self._layer_index >= scenario.num_layers

        observation, info = self._observe()
        info["step_execution_ms"] = step_execution
        info["step_communication_ms"] = step_communication
        info["step_energy"] = step_energy
        info["step_penalty"] = penalty
        if self._terminated:
            info["result"] = self.result()
        return observation, float(reward), self._terminated, False, info

    # ------------------------------------------------------------ inspection

    def action_mask(self) -> np.ndarray:
        """Boolean mask of devices that can host the layer awaiting placement."""
        if self.scenario is None or self.fleet is None:
            raise RuntimeError("action_mask() called before reset()")
        if self._terminated:
            return np.ones(self.config.num_devices, dtype=bool)
        layer = self.scenario.workload[self._layer_index]
        return self.fleet.feasibility_mask(layer.memory_gb)

    def result(self) -> PlacementResult:
        """Summarise the finished episode.

        Raises:
            RuntimeError: If the episode has not finished yet.
        """
        if self.scenario is None or not self._terminated:
            raise RuntimeError("result() is only available once the episode has terminated")
        scenario = self.scenario
        total_latency_ms = self._compute_latency_ms + self._communication_ms
        objective = weighted_cost(
            latency_ms=total_latency_ms,
            communication_ms=self._communication_ms,
            energy=self._energy,
            references=scenario.references,
            weights=scenario.weights,
            comm_double_count=self.config.objective.comm_double_count,
        )
        return PlacementResult(
            placement=tuple(self._placement),
            compute_latency_ms=self._compute_latency_ms,
            communication_latency_ms=self._communication_ms,
            total_latency_ms=total_latency_ms,
            energy=self._energy,
            objective=objective,
            normalised_latency=total_latency_ms / scenario.references.latency_ms,
            normalised_energy=self._energy / scenario.references.energy,
            normalised_communication=self._communication_ms / scenario.references.communication_ms,
            memory_violations=self._memory_violations,
            references=scenario.references,
            weights=scenario.weights,
        )

    @property
    def invalid_action_attempts(self) -> int:
        """How many times an infeasible device was requested this episode."""
        return self._invalid_attempts

    def render(self) -> None:
        """Print the placement decided so far."""
        if self.scenario is None:
            print("environment has not been reset")
            return
        names = self.scenario.device_names
        for index, device_index in enumerate(self._placement):
            print(f"  {self.scenario.workload[index].name:>12} -> {names[device_index]}")

    # --------------------------------------------------------------- private

    def _next_scenario_seed(self) -> int:
        """Return the next scenario seed, from the fixed list or the training pool."""
        if self.scenario_seeds is not None:
            seed = self.scenario_seeds[self._seed_cursor % len(self.scenario_seeds)]
            self._seed_cursor += 1
            return int(seed)
        return training_seeds(self.config, 1, self._sampler_rng)[0]

    def _next_num_layers(self) -> int | None:
        """Return the depth for the next episode."""
        option = self.num_layers_option
        if option is None or isinstance(option, int):
            return option
        choices = list(option)
        return int(choices[self._sampler_rng.integers(0, len(choices))])

    def _candidate_costs(self, layer_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-device execution time, communication time and energy for a layer."""
        assert self.scenario is not None and self.fleet is not None
        scenario = self.scenario
        layer = scenario.workload[layer_index]
        return step_costs(
            fleet=self.fleet,
            network=scenario.network,
            layer_index=layer_index,
            layer_compute_cost=layer.compute_cost,
            payload_mb=scenario.payload_before(layer_index),
            previous_device_index=self._previous_device,
            environment=self.config.environment,
        )

    def _observe(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Build the observation and info mapping for the pending decision."""
        assert self.scenario is not None and self.fleet is not None
        scenario = self.scenario

        if self._terminated:
            # Terminal observation: no decision is pending, so the vector is
            # zeroed and the mask is permissive. Bootstrapping must not use it.
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info: dict[str, Any] = {
                "action_mask": np.ones(self.config.num_devices, dtype=bool),
                "layer_index": self._layer_index,
                "scenario_seed": scenario.seed,
                "num_layers": scenario.num_layers,
                "terminal": True,
            }
            return observation, info

        layer_index = self._layer_index
        layer = scenario.workload[layer_index]
        execution_ms, communication_ms, energy = self._candidate_costs(layer_index)
        feasibility = self.fleet.feasibility_mask(layer.memory_gb)

        observation = self.observation_builder.build(
            scenario=scenario,
            fleet=self.fleet,
            layer_index=layer_index,
            previous_device_index=self._previous_device,
            execution_ms=execution_ms,
            communication_ms=communication_ms,
            energy=energy,
            feasibility=feasibility,
        )

        candidate_cost = np.array(
            [
                weighted_cost(
                    latency_ms=float(execution_ms[index] + communication_ms[index]),
                    communication_ms=float(communication_ms[index]),
                    energy=float(energy[index]),
                    references=scenario.references,
                    weights=scenario.weights,
                    comm_double_count=self.config.objective.comm_double_count,
                )
                for index in range(self.config.num_devices)
            ],
            dtype=np.float64,
        )

        info = {
            "action_mask": feasibility,
            "layer_index": layer_index,
            "scenario_seed": scenario.seed,
            "num_layers": scenario.num_layers,
            "previous_device": self._previous_device,
            "candidate_execution_ms": execution_ms,
            "candidate_communication_ms": communication_ms,
            "candidate_energy": energy,
            "candidate_cost": candidate_cost,
            "terminal": False,
        }
        return observation, info


__all__ = ["DNNPlacementEnv", "InvalidActionError"]
