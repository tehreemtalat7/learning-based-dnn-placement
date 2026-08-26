"""Synthetic DNN workload generation.

The workloads here are *synthetic but structured*: they are not profiled from
real networks, and the README says so plainly. What they do reproduce is the
qualitative shape that makes placement interesting, and which any convolutional
classifier exhibits:

* A **feature-extraction stage** whose activation tensors are large early on and
  shrink as spatial resolution is pooled away, while per-layer compute stays
  roughly flat or grows as channel counts increase. Splitting the network here
  is expensive, because a lot of data has to cross the link.
* A **dense head** whose parameter memory is large but whose activations are
  tiny. Splitting here is cheap in communication but the memory footprint can
  exclude small devices altogether.

That tension -- cheap-to-move layers that need a lot of memory versus
expensive-to-move layers that fit anywhere -- is what makes the placement
decision non-trivial and path dependent.

Every quantity is multiplied by log-normal jitter so that two scenarios with the
same depth still differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import WorkloadConfig


@dataclass(frozen=True)
class LayerSpec:
    """One DNN layer.

    Attributes:
        name: Human-readable identifier, e.g. ``"conv_3"``.
        compute_cost: Compute units required; divided by a device's effective
            speed to obtain execution time in seconds.
        memory_gb: Memory that must be free on the hosting device.
        output_size_mb: Activation size handed to the next layer, which drives
            the communication cost when the next layer sits elsewhere.
    """

    name: str
    compute_cost: float
    memory_gb: float
    output_size_mb: float

    def __post_init__(self) -> None:
        if self.compute_cost <= 0:
            raise ValueError(f"{self.name}: compute_cost must be positive")
        if self.memory_gb <= 0:
            raise ValueError(f"{self.name}: memory_gb must be positive")
        if self.output_size_mb < 0:
            raise ValueError(f"{self.name}: output_size_mb must be non-negative")


@dataclass(frozen=True)
class Workload:
    """An ordered chain of layers forming one DNN."""

    layers: tuple[LayerSpec, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("a workload needs at least one layer")

    def __len__(self) -> int:
        """Number of layers."""
        return len(self.layers)

    def __getitem__(self, index: int) -> LayerSpec:
        """Return the layer at ``index``."""
        return self.layers[index]

    @property
    def compute_costs(self) -> np.ndarray:
        """Compute cost of every layer."""
        return np.array([layer.compute_cost for layer in self.layers], dtype=np.float64)

    @property
    def memory_requirements_gb(self) -> np.ndarray:
        """Memory footprint of every layer."""
        return np.array([layer.memory_gb for layer in self.layers], dtype=np.float64)

    @property
    def output_sizes_mb(self) -> np.ndarray:
        """Activation size produced by every layer."""
        return np.array([layer.output_size_mb for layer in self.layers], dtype=np.float64)

    @property
    def total_compute(self) -> float:
        """Sum of all layer compute costs."""
        return float(self.compute_costs.sum())


def _jitter(rng: np.random.Generator, sigma: float, size: int | None = None):
    """Multiplicative log-normal noise with unit median."""
    if sigma <= 0:
        return 1.0 if size is None else np.ones(size)
    return rng.lognormal(mean=0.0, sigma=sigma, size=size)


def generate_workload(
    workload_config: WorkloadConfig,
    rng: np.random.Generator,
    num_layers: int | None = None,
    max_layer_memory_gb: float | None = None,
) -> Workload:
    """Generate one synthetic DNN.

    Args:
        workload_config: Generator parameters.
        rng: Generator used for every draw.
        num_layers: Override for ``workload_config.num_layers``.
        max_layer_memory_gb: If given, layer memory footprints are capped just
            below this value so that every layer fits on at least one device.
            Scenario sampling passes the largest device's memory here, which
            guarantees that no episode can reach a state with no legal action.

    Returns:
        A :class:`Workload` of ``num_layers`` layers.
    """
    depth = int(num_layers if num_layers is not None else workload_config.num_layers)
    if depth < 1:
        raise ValueError("num_layers must be at least 1")

    sigma = workload_config.jitter_sigma
    head_count = int(round(depth * workload_config.head_fraction))
    head_count = min(max(head_count, 1 if depth >= 2 else 0), depth - 1 if depth >= 2 else 0)
    feature_count = depth - head_count

    base_compute = workload_config.base_compute.sample(rng)
    compute_growth = workload_config.compute_growth.sample(rng)
    activation_mb = workload_config.base_activation_mb.sample(rng)
    base_activation_mb = activation_mb
    activation_decay = workload_config.activation_decay.sample(rng)
    feature_memory_base = workload_config.feature_memory_gb.sample(rng)

    layers: list[LayerSpec] = []

    for index in range(feature_count):
        block = index // workload_config.pool_every
        compute = base_compute * (compute_growth**block) * _jitter(rng, sigma)
        # Activation shrinks steadily and halves whenever a pooling stage occurs.
        activation_mb *= activation_decay
        if index > 0 and index % workload_config.pool_every == 0:
            activation_mb *= 0.5
        # Feature-layer memory tracks activation size: bigger tensors need more
        # working memory, but the dependence is sub-linear.
        activation_ratio = activation_mb / base_activation_mb
        memory = feature_memory_base * (0.5 + 0.5 * np.sqrt(activation_ratio)) * _jitter(rng, sigma)
        layers.append(
            LayerSpec(
                name=f"conv_{index}",
                compute_cost=float(compute),
                memory_gb=float(memory),
                output_size_mb=float(activation_mb * _jitter(rng, sigma)),
            )
        )

    head_compute_scale = workload_config.head_compute_scale.sample(rng)
    head_memory_base = workload_config.head_memory_gb.sample(rng)
    last_feature_compute = layers[-1].compute_cost if layers else base_compute

    for index in range(head_count):
        is_final = index == head_count - 1
        compute = last_feature_compute * head_compute_scale * _jitter(rng, sigma)
        # Dense layers hold large weight matrices; the classifier at the very end
        # is much smaller than the hidden dense layers.
        memory_scale = 0.25 if is_final else 1.0
        memory = head_memory_base * memory_scale * _jitter(rng, sigma)
        output = workload_config.head_output_mb.sample(rng) * _jitter(rng, sigma)
        if is_final:
            output *= 0.1  # class scores are negligible in size
        layers.append(
            LayerSpec(
                name=("classifier" if is_final else f"dense_{index}"),
                compute_cost=float(compute),
                memory_gb=float(memory),
                output_size_mb=float(output),
            )
        )

    if max_layer_memory_gb is not None:
        layers = _cap_layer_memory(layers, max_layer_memory_gb)

    return Workload(tuple(layers))


def _cap_layer_memory(layers: list[LayerSpec], max_layer_memory_gb: float) -> list[LayerSpec]:
    """Cap layer memory so that every layer fits on at least one device.

    Without this, a scenario could contain a layer that fits nowhere, leaving the
    environment with an empty action mask. Capping at generation time keeps the
    guarantee "every reachable state has at least one legal action" a property of
    the simulator rather than something the agent has to cope with.
    """
    ceiling = max_layer_memory_gb * 0.95
    return [
        layer
        if layer.memory_gb <= ceiling
        else LayerSpec(layer.name, layer.compute_cost, ceiling, layer.output_size_mb)
        for layer in layers
    ]


__all__ = ["LayerSpec", "Workload", "generate_workload"]
