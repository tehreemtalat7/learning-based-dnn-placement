"""Scenario sampling: one placement problem instance.

A :class:`Scenario` bundles everything that is fixed for the duration of one
episode -- the sampled devices, the network, the DNN, the objective weights and
the normalisation references -- and is fully determined by its integer seed.

Seeds come from two disjoint pools declared in ``configs/default.yaml``:
training scenarios from one, evaluation scenarios from the other. Every method
is evaluated on the *same* evaluation seeds, which makes the comparisons paired
and lets the analysis use paired statistics rather than treating the methods as
independent samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import Config
from src.environment.devices import DeviceProfile, sample_device_profiles
from src.environment.network import NetworkModel, sample_network
from src.environment.reward import CostReferences, ObjectiveWeights, compute_references
from src.environment.workload import Workload, generate_workload
from src.utils.seed import make_rng


@dataclass(frozen=True)
class Scenario:
    """One fully specified placement problem.

    Attributes:
        seed: The seed that generated this scenario; replaying it reproduces the
            instance exactly.
        devices: Sampled devices in action-index order.
        network: Link characteristics, possibly including a congestion event.
        workload: The DNN to place.
        weights: Objective weights in force.
        references: Normalisation constants derived from the above.
        input_size_mb: Size of the DNN input.
        input_source_index: Device that holds the input before layer 0 runs.
    """

    seed: int
    devices: tuple[DeviceProfile, ...]
    network: NetworkModel
    workload: Workload
    weights: ObjectiveWeights
    references: CostReferences
    input_size_mb: float
    input_source_index: int

    @property
    def num_layers(self) -> int:
        """Number of layers to place."""
        return len(self.workload)

    @property
    def num_devices(self) -> int:
        """Number of devices, which equals the size of the action space."""
        return len(self.devices)

    @property
    def device_names(self) -> tuple[str, ...]:
        """Device names in action-index order."""
        return tuple(device.name for device in self.devices)

    @property
    def has_congestion(self) -> bool:
        """Whether a congestion event occurs at any point in the episode."""
        return self.network.congestion is not None

    @property
    def congestion_start_layer(self) -> int | None:
        """Layer index at which congestion begins, or ``None`` if there is none."""
        return None if self.network.congestion is None else self.network.congestion.start_layer

    def payload_before(self, layer_index: int) -> float:
        """Data that must reach the device hosting ``layer_index``, in megabytes."""
        if layer_index == 0:
            return self.input_size_mb
        return self.workload[layer_index - 1].output_size_mb

    def describe(self) -> str:
        """Return a one-line summary, used in validation output and logs."""
        congestion = "none"
        if self.network.congestion is not None:
            event = self.network.congestion
            congestion = (
                f"from layer {event.start_layer} "
                f"(latency x{event.latency_scale:.2f}, bandwidth x{event.bandwidth_scale:.2f})"
            )
        return (
            f"scenario {self.seed}: {self.num_layers} layers, {self.num_devices} devices, "
            f"input {self.input_size_mb:.2f} MB, congestion {congestion}"
        )


def sample_scenario(config: Config, seed: int, num_layers: int | None = None) -> Scenario:
    """Sample one scenario from the configured distribution.

    Independent random streams are derived from ``seed`` for the devices, the
    network and the workload, so changing one part of the configuration does not
    shift the others' draws.

    Args:
        config: The loaded configuration.
        seed: Scenario seed; the same seed always yields the same scenario.
        num_layers: Optional override of the configured DNN depth, used by the
            scaling experiment.

    Returns:
        A :class:`Scenario`.
    """
    depth = int(num_layers if num_layers is not None else config.workload.num_layers)

    devices = sample_device_profiles(config, make_rng(seed, "devices"))
    network = sample_network(config, make_rng(seed, "network"), depth)

    workload_rng = make_rng(seed, "workload")
    largest_memory_gb = max(device.memory_gb for device in devices)
    workload = generate_workload(
        config.workload,
        workload_rng,
        num_layers=depth,
        max_layer_memory_gb=largest_memory_gb,
    )
    input_size_mb = config.workload.input_size_mb.sample(workload_rng)

    weights = ObjectiveWeights.sample(config.objective, make_rng(seed, "weights"))
    references = compute_references(workload, devices, network, input_size_mb)

    return Scenario(
        seed=int(seed),
        devices=devices,
        network=network,
        workload=workload,
        weights=weights,
        references=references,
        input_size_mb=float(input_size_mb),
        input_source_index=config.device_index(config.environment.input_source_device),
    )


def training_seeds(config: Config, count: int, rng: np.random.Generator) -> list[int]:
    """Draw ``count`` scenario seeds from the training pool.

    Args:
        config: The loaded configuration.
        count: How many seeds to draw.
        rng: Generator used for the draw.

    Returns:
        A list of seeds guaranteed to lie inside the training pool, and therefore
        disjoint from the held-out evaluation seeds.
    """
    experiment = config.experiment
    offsets = rng.integers(0, experiment.train_seed_count, size=count)
    return [int(experiment.train_seed_start + offset) for offset in offsets]


def evaluation_seeds(config: Config, count: int | None = None) -> list[int]:
    """Return the held-out evaluation seeds, identical for every method."""
    if count is None:
        return config.experiment.eval_seeds()
    return [config.experiment.eval_seed_start + index for index in range(count)]


def summarise_scenarios(scenarios: list[Scenario]) -> dict[str, float]:
    """Aggregate a scenario collection, for reporting the experimental setup.

    Returns:
        Mean layer compute cost, memory footprint, activation size, the fraction
        of (layer, device) pairs that are memory-infeasible, and the fraction of
        scenarios containing a congestion event.
    """
    if not scenarios:
        return {}
    compute, memory, output, infeasible_fraction = [], [], [], []
    for scenario in scenarios:
        compute.append(scenario.workload.compute_costs.mean())
        memory.append(scenario.workload.memory_requirements_gb.mean())
        output.append(scenario.workload.output_sizes_mb.mean())
        device_memory = np.array([device.memory_gb for device in scenario.devices])
        fits = scenario.workload.memory_requirements_gb[:, None] <= device_memory[None, :]
        infeasible_fraction.append(1.0 - fits.mean())
    return {
        "num_scenarios": float(len(scenarios)),
        "mean_layer_compute": float(np.mean(compute)),
        "mean_layer_memory_gb": float(np.mean(memory)),
        "mean_layer_output_mb": float(np.mean(output)),
        "mean_infeasible_pair_fraction": float(np.mean(infeasible_fraction)),
        "congestion_fraction": float(
            np.mean([1.0 if scenario.has_congestion else 0.0 for scenario in scenarios])
        ),
    }


__all__ = [
    "Scenario",
    "evaluation_seeds",
    "sample_scenario",
    "summarise_scenarios",
    "training_seeds",
]
