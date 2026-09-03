"""Cost model, normalisation references and the weighted placement objective.

This module is the single source of truth for "what does a placement cost".
The Gymnasium environment, the greedy heuristics, the dynamic-programming
optimum and the exhaustive search all score placements through the functions
defined here, so no two methods can silently disagree about the arithmetic.
A unit test asserts that rolling an episode through the environment reproduces
:func:`evaluate_placement` exactly.

**Cost of placing layer `i` on device `d`, coming from device `p`:**

```
execution_ms     = compute_cost_i / effective_speed_d * 1000
communication_ms = 0 if d == p else latency_pd + payload * 8 / bandwidth_pd * 1000
energy           = compute_cost_i * energy_per_compute_d  (+ optional transmission energy)
```

where `payload` is the DNN input for the first layer and the previous layer's
activation afterwards.

**Weighted objective (minimised by every method):**

```
objective = alpha * latency / latency_reference
          + beta  * energy  / energy_reference
          + gamma * communication / communication_reference
```

**Why the references are what they are.** They are the expected cost of placing
the DNN *uniformly at random*: total compute times the mean *reciprocal* device
speed, total energy at the mean energy rate, and the communication implied by a
device switch occurring with probability `(D-1)/D` at every layer boundary.
(Reciprocals, because execution and transfer times are inversely proportional to
speed and bandwidth; with device speeds spanning an order of magnitude, using
the reciprocal of the mean instead would misplace the scale by a factor of
several.) Two useful
properties follow. An objective near `1.0` means "no better than random", and
values stay on the same scale for 5-layer and 30-layer DNNs, which is what makes
the scaling experiment interpretable. This replaces the fixed constants
(10 000 ms, 500 energy units) used in the previous project, which did not scale
with depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.config import Config, ObjectiveConfig
from src.environment.devices import DeviceFleet, DeviceProfile
from src.environment.network import NetworkModel
from src.environment.workload import Workload

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from src.environment.scenario import Scenario

# References are clamped away from zero so that normalisation can never divide
# by a degenerate value.
MINIMUM_REFERENCE = 1e-9


@dataclass(frozen=True)
class ObjectiveWeights:
    """The three weights of the placement objective."""

    alpha: float
    beta: float
    gamma: float

    @staticmethod
    def from_config(objective: ObjectiveConfig) -> "ObjectiveWeights":
        """Build fixed weights from configuration."""
        return ObjectiveWeights(objective.alpha, objective.beta, objective.gamma)

    @staticmethod
    def sample(objective: ObjectiveConfig, rng: np.random.Generator) -> "ObjectiveWeights":
        """Draw weights for one scenario, or return the configured ones.

        When ``objective.randomise_weights`` is enabled the weights are drawn
        from a symmetric Dirichlet distribution, so they always sum to one.
        """
        if not objective.randomise_weights:
            return ObjectiveWeights.from_config(objective)
        alpha, beta, gamma = rng.dirichlet(np.ones(3))
        return ObjectiveWeights(float(alpha), float(beta), float(gamma))

    def as_array(self) -> np.ndarray:
        """Return the weights as a length-3 array, for the observation vector."""
        return np.array([self.alpha, self.beta, self.gamma], dtype=np.float64)


@dataclass(frozen=True)
class CostReferences:
    """Per-scenario normalisation constants (see the module docstring)."""

    latency_ms: float
    energy: float
    communication_ms: float

    def __post_init__(self) -> None:
        for name in ("latency_ms", "energy", "communication_ms"):
            if getattr(self, name) < MINIMUM_REFERENCE:
                object.__setattr__(self, name, MINIMUM_REFERENCE)


@dataclass(frozen=True)
class StepCost:
    """Cost of placing one layer on one device."""

    execution_ms: float
    communication_ms: float
    energy: float

    @property
    def latency_ms(self) -> float:
        """Execution plus communication time for this layer."""
        return self.execution_ms + self.communication_ms


@dataclass(frozen=True)
class PlacementResult:
    """Everything measured about one complete placement of one DNN."""

    placement: tuple[int, ...]
    compute_latency_ms: float
    communication_latency_ms: float
    total_latency_ms: float
    energy: float
    objective: float
    normalised_latency: float
    normalised_energy: float
    normalised_communication: float
    memory_violations: int
    references: CostReferences
    weights: ObjectiveWeights

    @property
    def num_layers(self) -> int:
        """Number of layers placed."""
        return len(self.placement)

    @property
    def is_feasible(self) -> bool:
        """Whether the placement respects every memory constraint."""
        return self.memory_violations == 0

    @property
    def device_switches(self) -> int:
        """Number of times consecutive layers were placed on different devices."""
        return sum(
            1
            for first, second in zip(self.placement, self.placement[1:], strict=False)
            if first != second
        )


def compute_references(
    workload: Workload,
    devices: tuple[DeviceProfile, ...],
    network: NetworkModel,
    input_size_mb: float,
) -> CostReferences:
    """Compute the expected cost of a uniformly random placement.

    Args:
        workload: The DNN being placed.
        devices: The sampled devices.
        network: Link characteristics; the *uncongested* conditions are used so
            that a congestion event shows up as a genuinely higher objective
            rather than being normalised away.
        input_size_mb: Size of the DNN input held by the source device.

    Returns:
        The three normalisation constants.
    """
    # Execution time is inversely proportional to speed, and transfer time is
    # inversely proportional to bandwidth, so the expectation over a uniformly
    # chosen device (or link) uses the mean of the reciprocal, not the
    # reciprocal of the mean. Device speeds here span more than an order of
    # magnitude, so the difference is large and getting it wrong would leave the
    # objective mis-calibrated by a factor of several.
    mean_inverse_speed = float(
        np.mean(
            [
                1.0 / max(device.compute_capacity * (1.0 - device.base_utilisation), MINIMUM_REFERENCE)
                for device in devices
            ]
        )
    )
    mean_energy_rate = float(np.mean([device.energy_per_compute for device in devices]))
    # Deliberately the uncongested link characteristics: see
    # NetworkModel.base_mean_link_latency_ms for why.
    mean_latency_ms = network.base_mean_link_latency_ms()
    mean_inverse_bandwidth = network.base_mean_inverse_bandwidth()

    compute_latency_ms = workload.total_compute * mean_inverse_speed * 1000.0

    # Payload transferred before each layer: the DNN input, then activations.
    payloads = np.concatenate(([input_size_mb], workload.output_sizes_mb[:-1]))
    per_transfer_ms = mean_latency_ms + payloads * 8.0 * mean_inverse_bandwidth * 1000.0
    switch_probability = (len(devices) - 1) / len(devices)
    communication_ms = float(switch_probability * per_transfer_ms.sum())

    return CostReferences(
        latency_ms=compute_latency_ms + communication_ms,
        energy=workload.total_compute * mean_energy_rate,
        communication_ms=communication_ms,
    )


def weighted_cost(
    latency_ms: float,
    communication_ms: float,
    energy: float,
    references: CostReferences,
    weights: ObjectiveWeights,
    comm_double_count: bool,
) -> float:
    """Score a (partial or complete) placement with the weighted objective.

    Args:
        latency_ms: End-to-end latency, i.e. execution plus communication.
        communication_ms: The communication part of that latency.
        energy: Energy consumed.
        references: Per-scenario normalisation constants.
        weights: Objective weights.
        comm_double_count: When true, communication is charged inside the latency
            term *and* again under ``gamma`` as an explicit congestion penalty.
            When false, the latency term covers computation only, so the three
            terms are disjoint. Experiment 5 reports the difference.

    Returns:
        The weighted, normalised cost. Lower is better.
    """
    latency_component = latency_ms if comm_double_count else latency_ms - communication_ms
    return (
        weights.alpha * latency_component / references.latency_ms
        + weights.beta * energy / references.energy
        + weights.gamma * communication_ms / references.communication_ms
    )


def step_costs(
    fleet: DeviceFleet,
    network: NetworkModel,
    layer_index: int,
    layer_compute_cost: float,
    payload_mb: float,
    previous_device_index: int,
    environment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cost of placing the current layer on *every* device.

    Vectorised so that the environment, the greedy agents and the observation
    builder all read the same numbers.

    Args:
        fleet: Devices with their current per-episode state.
        network: Link characteristics, including any congestion in effect.
        layer_index: Index of the layer being placed (selects network conditions).
        layer_compute_cost: Compute cost of the layer.
        payload_mb: Data that must arrive before the layer can run.
        previous_device_index: Device holding that data.
        environment: The :class:`~src.config.EnvironmentConfig` in force.

    Returns:
        A tuple ``(execution_ms, communication_ms, energy)``, each of length
        ``len(fleet)``.
    """
    execution_ms = fleet.execution_time_ms(layer_compute_cost)
    communication_ms = network.transfer_times_ms(previous_device_index, payload_mb, layer_index)
    energy = fleet.execution_energy(layer_compute_cost)
    if environment.include_transmission_energy:
        transferred = np.where(communication_ms > 0.0, payload_mb, 0.0)
        energy = energy + transferred * environment.transmission_energy_per_mb
    return execution_ms, communication_ms, energy


def evaluate_placement(
    scenario: "Scenario", placement: tuple[int, ...] | list[int], config: Config
) -> PlacementResult:
    """Score a complete placement without going through the Gymnasium API.

    Used by the exhaustive search, the dynamic-programming optimum and the unit
    tests. Replays the same accumulation dynamics the environment applies.

    Args:
        scenario: The scenario the placement belongs to.
        placement: One device index per layer, in layer order.
        config: The configuration in force.

    Returns:
        A :class:`PlacementResult`.

    Raises:
        ValueError: If the placement has the wrong length or an out-of-range index.
    """
    workload = scenario.workload
    if len(placement) != len(workload):
        raise ValueError(
            f"placement has {len(placement)} entries but the DNN has {len(workload)} layers"
        )

    fleet = DeviceFleet(scenario.devices, config.environment)
    previous_device = scenario.input_source_index
    compute_latency_ms = 0.0
    communication_ms = 0.0
    energy = 0.0
    memory_violations = 0

    for layer_index, device_index in enumerate(placement):
        if not 0 <= device_index < len(fleet):
            raise ValueError(f"invalid device index {device_index} at layer {layer_index}")
        layer = workload[layer_index]
        payload_mb = (
            scenario.input_size_mb if layer_index == 0 else workload[layer_index - 1].output_size_mb
        )
        if not fleet.can_host(device_index, layer.memory_gb):
            memory_violations += 1

        execution, communication, step_energy = step_costs(
            fleet=fleet,
            network=scenario.network,
            layer_index=layer_index,
            layer_compute_cost=layer.compute_cost,
            payload_mb=payload_mb,
            previous_device_index=previous_device,
            environment=config.environment,
        )
        compute_latency_ms += float(execution[device_index])
        communication_ms += float(communication[device_index])
        energy += float(step_energy[device_index])

        fleet.assign(device_index, layer.compute_cost, layer.memory_gb)
        previous_device = device_index

    total_latency_ms = compute_latency_ms + communication_ms
    references = scenario.references
    objective = weighted_cost(
        latency_ms=total_latency_ms,
        communication_ms=communication_ms,
        energy=energy,
        references=references,
        weights=scenario.weights,
        comm_double_count=config.objective.comm_double_count,
    )
    return PlacementResult(
        placement=tuple(int(index) for index in placement),
        compute_latency_ms=compute_latency_ms,
        communication_latency_ms=communication_ms,
        total_latency_ms=total_latency_ms,
        energy=energy,
        objective=objective,
        normalised_latency=total_latency_ms / references.latency_ms,
        normalised_energy=energy / references.energy,
        normalised_communication=communication_ms / references.communication_ms,
        memory_violations=memory_violations,
        references=references,
        weights=scenario.weights,
    )


__all__ = [
    "CostReferences",
    "MINIMUM_REFERENCE",
    "ObjectiveWeights",
    "PlacementResult",
    "StepCost",
    "compute_references",
    "evaluate_placement",
    "step_costs",
    "weighted_cost",
]
