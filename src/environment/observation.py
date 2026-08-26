"""Construction of the fixed-length state vector.

The observation has a fixed width regardless of how many layers the DNN has, so
one trained policy can place 5-layer and 30-layer networks. It is organised as a
global block describing the decision context, followed by one block per device.

Normalisation follows two different rules on purpose:

* **Layer features** are divided by the *scenario's own* means. They describe the
  shape of the workload -- "this layer is twice as expensive as the average
  layer" -- which is what generalises across DNN depths.
* **Device and link features** are divided by *configuration midpoints*, which
  are the same for every scenario. This is deliberate: if they were normalised
  per scenario, a network-wide congestion event would scale every link equally
  and become invisible. With fixed references, congestion appears as a latency
  feature rising to three or four times its usual value, which is exactly the
  signal Experiment 3 asks the agent to react to.
* **Estimated costs** are divided by the scenario references over the number of
  layers, so they hover around one for a typical decision.

``docs/STATE_SPEC.md`` is generated from :func:`feature_names`, so the
documentation cannot drift away from the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import Config
from src.environment.devices import DeviceFleet
from src.environment.reward import ObjectiveWeights, weighted_cost
from src.environment.scenario import Scenario

GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "progress_fraction",
    "remaining_fraction",
    "inverse_depth",
    "layer_compute_rel",
    "layer_memory_rel",
    "layer_output_rel",
    "next_layer_compute_rel",
    "next_layer_memory_rel",
    "next_layer_output_rel",
    "remaining_compute_fraction",
    "remaining_output_fraction",
    "weight_alpha",
    "weight_beta",
    "weight_gamma",
)

DEVICE_FEATURE_NAMES: tuple[str, ...] = (
    "effective_speed_rel",
    "free_memory_ratio",
    "utilisation",
    "energy_per_compute_rel",
    "is_previous_device",
    "link_latency_rel",
    "link_bandwidth_rel",
    "is_feasible",
    "estimated_execution_rel",
    "estimated_communication_rel",
    "estimated_immediate_cost",
)


@dataclass(frozen=True)
class ObservationNormalisers:
    """Scenario-independent scales taken from the configuration midpoints."""

    compute_capacity: float
    device_memory_gb: float
    energy_per_compute: float
    link_latency_ms: float
    link_bandwidth_mbps: float

    @staticmethod
    def from_config(config: Config) -> "ObservationNormalisers":
        """Derive the fixed normalisers from the configured sampling ranges."""
        return ObservationNormalisers(
            compute_capacity=float(
                np.mean([device.compute_capacity.midpoint for device in config.devices])
            ),
            device_memory_gb=float(
                np.mean([device.memory_gb.midpoint for device in config.devices])
            ),
            energy_per_compute=float(
                np.mean([device.energy_per_compute.midpoint for device in config.devices])
            ),
            link_latency_ms=float(
                np.mean([link.latency_ms.midpoint for link in config.network.links])
            ),
            link_bandwidth_mbps=float(
                np.mean([link.bandwidth_mbps.midpoint for link in config.network.links])
            ),
        )


def feature_names(num_devices: int, device_names: tuple[str, ...] | None = None) -> list[str]:
    """Return the name of every entry of the observation vector, in order."""
    names = list(GLOBAL_FEATURE_NAMES)
    for index in range(num_devices):
        label = device_names[index] if device_names is not None else f"device{index}"
        names.extend(f"{label}__{feature}" for feature in DEVICE_FEATURE_NAMES)
    return names


def observation_size(num_devices: int) -> int:
    """Width of the observation vector for a given number of devices."""
    return len(GLOBAL_FEATURE_NAMES) + num_devices * len(DEVICE_FEATURE_NAMES)


class ObservationBuilder:
    """Builds observations for one configuration."""

    def __init__(self, config: Config) -> None:
        """Store the configuration and precompute the fixed normalisers."""
        self.config = config
        self.normalisers = ObservationNormalisers.from_config(config)
        self.size = observation_size(config.num_devices)

    def build(
        self,
        scenario: Scenario,
        fleet: DeviceFleet,
        layer_index: int,
        previous_device_index: int,
        execution_ms: np.ndarray,
        communication_ms: np.ndarray,
        energy: np.ndarray,
        feasibility: np.ndarray,
    ) -> np.ndarray:
        """Assemble the observation for the decision about ``layer_index``.

        Args:
            scenario: The scenario being solved.
            fleet: Devices with their current per-episode state.
            layer_index: Index of the layer about to be placed.
            previous_device_index: Device holding the incoming activation.
            execution_ms: Per-device execution time of the current layer.
            communication_ms: Per-device communication time of the current layer.
            energy: Per-device energy for the current layer.
            feasibility: Per-device boolean feasibility mask.

        Returns:
            A ``float32`` vector of length :attr:`size`.
        """
        workload = scenario.workload
        depth = len(workload)
        layer = workload[layer_index]

        compute_costs = workload.compute_costs
        memory_requirements = workload.memory_requirements_gb
        output_sizes = workload.output_sizes_mb
        mean_compute = max(float(compute_costs.mean()), 1e-12)
        mean_memory = max(float(memory_requirements.mean()), 1e-12)
        mean_output = max(float(output_sizes.mean()), 1e-12)

        has_next = layer_index + 1 < depth
        next_layer = workload[layer_index + 1] if has_next else None

        remaining_compute = float(compute_costs[layer_index:].sum()) / max(
            float(compute_costs.sum()), 1e-12
        )
        remaining_output = float(output_sizes[layer_index:].sum()) / max(
            float(output_sizes.sum()), 1e-12
        )

        weights = scenario.weights
        global_block = np.array(
            [
                layer_index / depth,
                (depth - layer_index) / depth,
                1.0 / depth,
                layer.compute_cost / mean_compute,
                layer.memory_gb / mean_memory,
                layer.output_size_mb / mean_output,
                (next_layer.compute_cost / mean_compute) if next_layer else 0.0,
                (next_layer.memory_gb / mean_memory) if next_layer else 0.0,
                (next_layer.output_size_mb / mean_output) if next_layer else 0.0,
                remaining_compute,
                remaining_output,
                weights.alpha,
                weights.beta,
                weights.gamma,
            ],
            dtype=np.float64,
        )

        device_block = self._device_block(
            scenario=scenario,
            fleet=fleet,
            depth=depth,
            layer_index=layer_index,
            previous_device_index=previous_device_index,
            execution_ms=execution_ms,
            communication_ms=communication_ms,
            energy=energy,
            feasibility=feasibility,
            weights=weights,
        )
        return np.concatenate([global_block, device_block]).astype(np.float32)

    def _device_block(
        self,
        scenario: Scenario,
        fleet: DeviceFleet,
        depth: int,
        layer_index: int,
        previous_device_index: int,
        execution_ms: np.ndarray,
        communication_ms: np.ndarray,
        energy: np.ndarray,
        feasibility: np.ndarray,
        weights: ObjectiveWeights,
    ) -> np.ndarray:
        """Build the per-device part of the observation."""
        normalisers = self.normalisers
        references = scenario.references
        latency_scale = max(references.latency_ms / depth, 1e-12)

        effective_speed = fleet.effective_speed() / normalisers.compute_capacity
        free_memory_ratio = fleet.free_memory_gb() / fleet.total_memory_gb
        utilisation = fleet.utilisation()
        energy_rate = fleet.energy_per_compute / normalisers.energy_per_compute

        is_previous = np.zeros(len(fleet), dtype=np.float64)
        is_previous[previous_device_index] = 1.0

        latency_ms, bandwidth_mbps = scenario.network.conditions_at(layer_index)
        # Link characteristics are reported relative to the device that currently
        # holds the activation, which is what the pending transfer would use.
        link_latency = latency_ms[previous_device_index] / normalisers.link_latency_ms
        link_bandwidth = bandwidth_mbps[previous_device_index] / normalisers.link_bandwidth_mbps
        link_latency = link_latency.copy()
        link_bandwidth = link_bandwidth.copy()
        link_latency[previous_device_index] = 0.0
        link_bandwidth[previous_device_index] = 0.0

        immediate_cost = np.array(
            [
                weighted_cost(
                    latency_ms=float(execution_ms[index] + communication_ms[index]),
                    communication_ms=float(communication_ms[index]),
                    energy=float(energy[index]),
                    references=references,
                    weights=weights,
                    comm_double_count=self.config.objective.comm_double_count,
                )
                for index in range(len(fleet))
            ],
            dtype=np.float64,
        )

        stacked = np.stack(
            [
                effective_speed,
                free_memory_ratio,
                utilisation,
                energy_rate,
                is_previous,
                link_latency,
                link_bandwidth,
                feasibility.astype(np.float64),
                execution_ms / latency_scale,
                communication_ms / latency_scale,
                immediate_cost * depth,
            ],
            axis=1,
        )
        return stacked.reshape(-1)


def describe_state_vector(config: Config) -> str:
    """Render the observation layout as Markdown.

    ``docs/STATE_SPEC.md`` is produced by this function (``make docs``), so the
    documentation of the state vector cannot drift away from the code that
    builds it.
    """
    names = feature_names(config.num_devices, config.device_names)
    lines = [
        "# State vector specification",
        "",
        "*Generated by `python -m src.environment.observation` -- do not edit by hand.*",
        "",
        f"The observation is a `float32` vector of **{len(names)}** entries: "
        f"{len(GLOBAL_FEATURE_NAMES)} global features describing the decision, followed by "
        f"{len(DEVICE_FEATURE_NAMES)} features for each of the {config.num_devices} devices.",
        "",
        "Its width does not depend on the number of layers, so one policy places DNNs of any depth.",
        "",
        "## Global block",
        "",
        "| # | Feature | Meaning | Normalisation |",
        "|--:|---|---|---|",
    ]
    global_documentation = {
        "progress_fraction": ("Index of the layer being placed", "divided by the DNN depth"),
        "remaining_fraction": ("Layers still to place, including this one", "divided by the DNN depth"),
        "inverse_depth": ("Cue for how deep the DNN is", "1 / depth"),
        "layer_compute_rel": ("Compute cost of the current layer", "divided by the mean layer compute cost of this DNN"),
        "layer_memory_rel": ("Memory footprint of the current layer", "divided by the mean layer memory of this DNN"),
        "layer_output_rel": ("Activation this layer will emit", "divided by the mean activation size of this DNN"),
        "next_layer_compute_rel": ("One-step lookahead: compute of the next layer", "as above; zero on the final layer"),
        "next_layer_memory_rel": ("One-step lookahead: memory of the next layer", "as above; zero on the final layer"),
        "next_layer_output_rel": ("One-step lookahead: activation of the next layer", "as above; zero on the final layer"),
        "remaining_compute_fraction": ("Compute still to be placed", "fraction of the DNN's total compute"),
        "remaining_output_fraction": ("Activation volume still to be produced", "fraction of the DNN's total activation volume"),
        "weight_alpha": ("Latency weight of the objective", "already in [0, 1]"),
        "weight_beta": ("Energy weight of the objective", "already in [0, 1]"),
        "weight_gamma": ("Communication weight of the objective", "already in [0, 1]"),
    }
    for index, feature in enumerate(GLOBAL_FEATURE_NAMES):
        meaning, normalisation = global_documentation[feature]
        lines.append(f"| {index} | `{feature}` | {meaning} | {normalisation} |")

    device_documentation = {
        "effective_speed_rel": ("Compute the device can currently deliver, after background and accumulated load", "divided by the mean configured device capacity"),
        "free_memory_ratio": ("Memory still free on the device", "fraction of that device's total memory"),
        "utilisation": ("Current load of the device", "already in [0, 1]"),
        "energy_per_compute_rel": ("Energy cost of running work here", "divided by the mean configured energy rate"),
        "is_previous_device": ("Whether the previous layer's activation already sits here", "0 or 1"),
        "link_latency_rel": ("Latency of the link carrying the pending transfer", "divided by the mean configured link latency; **fixed** reference, so congestion is visible"),
        "link_bandwidth_rel": ("Bandwidth of that link", "divided by the mean configured link bandwidth; fixed reference"),
        "is_feasible": ("Whether this device can host the current layer", "0 or 1; mirrors the action mask"),
        "estimated_execution_rel": ("Execution time of the current layer here", "divided by the scenario's latency reference per layer"),
        "estimated_communication_rel": ("Time to move the pending activation here", "as above"),
        "estimated_immediate_cost": ("Weighted objective contribution of choosing this device now", "multiplied by the depth, so it is around 1 for a typical decision"),
    }
    lines.extend(
        [
            "",
            "## Per-device block",
            "",
            f"Repeated once per device, in action-index order: {', '.join(f'`{name}`' for name in config.device_names)}.",
            "",
            "| Offset | Feature | Meaning | Normalisation |",
            "|--:|---|---|---|",
        ]
    )
    for offset, feature in enumerate(DEVICE_FEATURE_NAMES):
        meaning, normalisation = device_documentation[feature]
        lines.append(f"| {offset} | `{feature}` | {meaning} | {normalisation} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "* **Layer features** are normalised by this scenario's own means, so they describe the",
            "  *shape* of the workload and transfer across DNN depths.",
            "* **Device and link features** are normalised by configuration midpoints, which are the",
            "  same in every scenario. This is deliberate: normalising per scenario would rescale",
            "  every link equally during a network-wide congestion event and hide it from the agent.",
            "* **`estimated_immediate_cost`** is exactly the quantity the objective-aware greedy",
            "  heuristic minimises. Giving it to the learning agent means the comparison measures",
            "  decision quality rather than information asymmetry: any advantage the agent shows must",
            "  come from looking beyond the current layer.",
            "* The full ordered list of entry names is produced by `feature_names()` in",
            "  `src/environment/observation.py`.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    from src.config import load_config

    print(describe_state_vector(load_config()))


__all__ = [
    "DEVICE_FEATURE_NAMES",
    "GLOBAL_FEATURE_NAMES",
    "ObservationBuilder",
    "ObservationNormalisers",
    "describe_state_vector",
    "feature_names",
    "observation_size",
]
