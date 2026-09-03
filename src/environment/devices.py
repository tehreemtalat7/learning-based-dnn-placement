"""Heterogeneous compute devices and their per-episode state.

A :class:`DeviceProfile` is one concrete device sampled from a configured
archetype and never changes during an episode. A :class:`DeviceFleet` holds the
mutable state that placement decisions accumulate: how much memory each device
currently holds and how much extra load has been assigned to it.

The accumulation is what makes placement path dependent. Both accumulation
effects can be switched off (``environment.memory_accumulates`` and
``environment.utilisation_accumulates``), which reduces the problem to a pure
shortest-path over consecutive device pairs -- the setting in which the
dynamic-programming optimum in :mod:`src.baselines.dp_optimal` is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import EnvironmentConfig

# Memory comparisons use a small tolerance so that a layer needing exactly the
# free memory of a device is treated as fitting.
MEMORY_TOLERANCE_GB = 1e-9


@dataclass(frozen=True)
class DeviceProfile:
    """One concrete device sampled from a configured archetype.

    Attributes:
        name: Human-readable identifier, e.g. ``"gpu_server"``.
        compute_capacity: Nominal compute units executed per second at zero load.
        memory_gb: Total memory available to DNN layers.
        energy_per_compute: Energy units consumed per unit of compute.
        base_utilisation: Background load in ``[0, 1)`` present before any layer
            of this DNN is placed.
    """

    name: str
    compute_capacity: float
    memory_gb: float
    energy_per_compute: float
    base_utilisation: float

    def __post_init__(self) -> None:
        if self.compute_capacity <= 0:
            raise ValueError(f"{self.name}: compute_capacity must be positive")
        if self.memory_gb <= 0:
            raise ValueError(f"{self.name}: memory_gb must be positive")
        if self.energy_per_compute <= 0:
            raise ValueError(f"{self.name}: energy_per_compute must be positive")
        if not 0.0 <= self.base_utilisation < 1.0:
            raise ValueError(f"{self.name}: base_utilisation must lie in [0, 1)")


class DeviceFleet:
    """Mutable per-episode state of every device.

    The fleet is the single place where the cost of executing a layer on a
    device is computed, so the Gymnasium environment, the exhaustive search and
    the dynamic-programming baseline all share exactly the same arithmetic.
    """

    def __init__(self, profiles: tuple[DeviceProfile, ...], environment: EnvironmentConfig) -> None:
        """Create a fleet and reset it to its initial state.

        Args:
            profiles: Sampled devices, in action-index order.
            environment: Environment configuration controlling accumulation and
                utilisation limits.
        """
        if not profiles:
            raise ValueError("a fleet needs at least one device")
        self.profiles = profiles
        self.environment = environment

        self.capacity = np.array([device.compute_capacity for device in profiles], dtype=np.float64)
        self.total_memory_gb = np.array([device.memory_gb for device in profiles], dtype=np.float64)
        self.energy_per_compute = np.array(
            [device.energy_per_compute for device in profiles], dtype=np.float64
        )
        self.base_utilisation = np.array(
            [device.base_utilisation for device in profiles], dtype=np.float64
        )

        self.used_memory_gb = np.zeros(len(profiles), dtype=np.float64)
        self.assigned_compute = np.zeros(len(profiles), dtype=np.float64)
        self.reset()

    def __len__(self) -> int:
        """Number of devices, which equals the size of the action space."""
        return len(self.profiles)

    @property
    def names(self) -> tuple[str, ...]:
        """Device names in action-index order."""
        return tuple(device.name for device in self.profiles)

    def reset(self) -> None:
        """Clear all memory and load accumulated during an episode."""
        self.used_memory_gb[:] = 0.0
        self.assigned_compute[:] = 0.0

    def copy(self) -> "DeviceFleet":
        """Return an independent fleet with the same profiles and current state."""
        clone = DeviceFleet(self.profiles, self.environment)
        clone.used_memory_gb[:] = self.used_memory_gb
        clone.assigned_compute[:] = self.assigned_compute
        return clone

    # ------------------------------------------------------------------ state

    def free_memory_gb(self) -> np.ndarray:
        """Memory still available on each device, never below zero.

        Under ``invalid_action_mode: penalty`` a device can be over-subscribed;
        the reported free memory is clipped at zero so observations stay in a
        sane range while the violation is counted separately.
        """
        return np.maximum(self.total_memory_gb - self.used_memory_gb, 0.0)

    def utilisation(self) -> np.ndarray:
        """Current utilisation of each device in ``[0, utilisation_max]``.

        Utilisation is the configured background load plus the load induced by
        the layers already assigned in this episode::

            extra = assigned_compute / (compute_capacity * utilisation_load_scale)

        so that a fixed amount of work loads a slow device far more than a fast
        one.
        """
        if not self.environment.utilisation_accumulates:
            return np.minimum(self.base_utilisation, self.environment.utilisation_max)
        extra = self.assigned_compute / (self.capacity * self.environment.utilisation_load_scale)
        return np.minimum(self.base_utilisation + extra, self.environment.utilisation_max)

    def effective_speed(self) -> np.ndarray:
        """Compute units per second currently deliverable by each device."""
        available = self.capacity * (1.0 - self.utilisation())
        floor = self.capacity * self.environment.effective_speed_floor
        return np.maximum(available, floor)

    def feasibility_mask(self, layer_memory_gb: float) -> np.ndarray:
        """Boolean mask of devices with enough free memory for a layer."""
        return self.free_memory_gb() + MEMORY_TOLERANCE_GB >= layer_memory_gb

    def can_host(self, device_index: int, layer_memory_gb: float) -> bool:
        """Whether one device can currently host a layer of the given size."""
        return bool(self.feasibility_mask(layer_memory_gb)[device_index])

    # ------------------------------------------------------------------- cost

    def execution_time_ms(self, layer_compute_cost: float) -> np.ndarray:
        """Execution time of a layer on every device, in milliseconds."""
        return layer_compute_cost / self.effective_speed() * 1000.0

    def execution_energy(self, layer_compute_cost: float) -> np.ndarray:
        """Compute energy of a layer on every device, in energy units."""
        return layer_compute_cost * self.energy_per_compute

    # ------------------------------------------------------------------ apply

    def assign(self, device_index: int, layer_compute_cost: float, layer_memory_gb: float) -> None:
        """Record that a layer has been placed on a device.

        Args:
            device_index: Index of the chosen device.
            layer_compute_cost: Compute cost of the placed layer.
            layer_memory_gb: Memory footprint of the placed layer.

        Raises:
            IndexError: If ``device_index`` is out of range.
        """
        if not 0 <= device_index < len(self.profiles):
            raise IndexError(f"device index {device_index} is outside 0..{len(self.profiles) - 1}")
        if self.environment.memory_accumulates:
            self.used_memory_gb[device_index] += layer_memory_gb
        if self.environment.utilisation_accumulates:
            self.assigned_compute[device_index] += layer_compute_cost


def sample_device_profiles(config, rng: np.random.Generator) -> tuple[DeviceProfile, ...]:
    """Draw one concrete device per configured archetype.

    ``environment.device_load_override`` replaces the sampled background load of
    named devices. Every other quantity is still drawn from the same stream, so
    sweeping one device's load leaves the rest of the scenario distribution --
    including the workload and the network -- untouched. That is what makes the
    device-load experiment a controlled comparison rather than a different
    experiment at every load level.

    Args:
        config: The loaded :class:`~src.config.Config`.
        rng: Generator used for all device draws.

    Returns:
        Sampled devices in the order the archetypes are configured, which is
        also the action-index order.
    """
    overrides = config.environment.device_load_override
    profiles = []
    for archetype in config.devices:
        # Draw every value regardless, so that overriding one device cannot
        # shift the random stream seen by the others.
        capacity = archetype.compute_capacity.sample(rng)
        memory = archetype.memory_gb.sample(rng)
        energy = archetype.energy_per_compute.sample(rng)
        load = archetype.base_utilisation.sample(rng)
        if archetype.name in overrides:
            load = overrides[archetype.name].sample(rng)
        profiles.append(
            DeviceProfile(
                name=archetype.name,
                compute_capacity=capacity,
                memory_gb=memory,
                energy_per_compute=energy,
                base_utilisation=load,
            )
        )
    return tuple(profiles)


__all__ = ["MEMORY_TOLERANCE_GB", "DeviceFleet", "DeviceProfile", "sample_device_profiles"]
