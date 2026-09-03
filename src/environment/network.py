"""Device-to-device network model, including congestion events.

Communication cost follows the same formula as the previous project, so results
from the two remain comparable::

    communication_time_ms = latency_ms + (data_size_mb * 8 / bandwidth_mbps) * 1000

Transfers inside a single device are free. Links are symmetric.

A congestion event multiplies latency up and bandwidth down from a given layer
index onwards. Events that start at layer 0 model an episode that begins in a
congested network; events with a later start index model conditions changing
*during* placement, which is what Experiment 3 measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import Config, NetworkProfileConfig


@dataclass(frozen=True)
class CongestionEvent:
    """A multiplicative degradation of every link from ``start_layer`` onwards.

    Attributes:
        start_layer: First layer index at which the degraded conditions apply.
            Zero means the whole episode is congested.
        latency_scale: Factor applied to every link latency (>= 1 degrades).
        bandwidth_scale: Factor applied to every link bandwidth (<= 1 degrades).
    """

    start_layer: int
    latency_scale: float
    bandwidth_scale: float

    def __post_init__(self) -> None:
        if self.start_layer < 0:
            raise ValueError("congestion start_layer must be non-negative")
        if self.latency_scale <= 0 or self.bandwidth_scale <= 0:
            raise ValueError("congestion scales must be positive")


class NetworkModel:
    """Symmetric link characteristics with optional mid-episode congestion."""

    def __init__(
        self,
        latency_ms: np.ndarray,
        bandwidth_mbps: np.ndarray,
        congestion: CongestionEvent | None = None,
    ) -> None:
        """Create a network from square, symmetric link matrices.

        Args:
            latency_ms: ``(D, D)`` matrix of one-way latencies; the diagonal is ignored.
            bandwidth_mbps: ``(D, D)`` matrix of link bandwidths; the diagonal is ignored.
            congestion: Optional congestion event affecting part or all of the episode.

        Raises:
            ValueError: If the matrices are not square, not symmetric, or contain
                non-positive bandwidths off the diagonal.
        """
        if latency_ms.shape != bandwidth_mbps.shape:
            raise ValueError("latency and bandwidth matrices must have the same shape")
        if latency_ms.ndim != 2 or latency_ms.shape[0] != latency_ms.shape[1]:
            raise ValueError("link matrices must be square")
        if not np.allclose(latency_ms, latency_ms.T) or not np.allclose(
            bandwidth_mbps, bandwidth_mbps.T
        ):
            raise ValueError("link matrices must be symmetric")

        off_diagonal = ~np.eye(latency_ms.shape[0], dtype=bool)
        if np.any(bandwidth_mbps[off_diagonal] <= 0):
            raise ValueError("all off-diagonal bandwidths must be positive")
        if np.any(latency_ms[off_diagonal] < 0):
            raise ValueError("link latencies must be non-negative")

        self.base_latency_ms = latency_ms.astype(np.float64, copy=True)
        self.base_bandwidth_mbps = bandwidth_mbps.astype(np.float64, copy=True)
        self.congestion = congestion

    @property
    def num_devices(self) -> int:
        """Number of devices the topology connects."""
        return self.base_latency_ms.shape[0]

    def is_congested_at(self, layer_index: int) -> bool:
        """Whether degraded conditions apply when placing ``layer_index``."""
        return self.congestion is not None and layer_index >= self.congestion.start_layer

    def conditions_at(self, layer_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(latency_ms, bandwidth_mbps)`` matrices seen at a layer."""
        if not self.is_congested_at(layer_index):
            return self.base_latency_ms, self.base_bandwidth_mbps
        assert self.congestion is not None  # narrowed by is_congested_at
        return (
            self.base_latency_ms * self.congestion.latency_scale,
            self.base_bandwidth_mbps * self.congestion.bandwidth_scale,
        )

    def transfer_times_ms(
        self, source_index: int, data_size_mb: float, layer_index: int
    ) -> np.ndarray:
        """Time to move data from one device to every device, in milliseconds.

        Args:
            source_index: Device currently holding the data.
            data_size_mb: Payload size in megabytes.
            layer_index: Layer being placed, which selects the network conditions.

        Returns:
            A vector of length ``num_devices``; the entry for ``source_index`` is
            zero because no transfer is needed.
        """
        latency_ms, bandwidth_mbps = self.conditions_at(layer_index)
        times = latency_ms[source_index] + (data_size_mb * 8.0 / bandwidth_mbps[source_index]) * 1000.0
        times = times.copy()
        times[source_index] = 0.0
        return times

    def transfer_time_ms(
        self, source_index: int, destination_index: int, data_size_mb: float, layer_index: int
    ) -> float:
        """Time to move data between two specific devices, in milliseconds."""
        if source_index == destination_index:
            return 0.0
        return float(self.transfer_times_ms(source_index, data_size_mb, layer_index)[destination_index])

    def mean_link_latency_ms(self, layer_index: int = 0) -> float:
        """Mean latency over all distinct links under the conditions at a layer."""
        latency_ms, _ = self.conditions_at(layer_index)
        off_diagonal = ~np.eye(self.num_devices, dtype=bool)
        return float(latency_ms[off_diagonal].mean())

    def base_mean_link_latency_ms(self) -> float:
        """Mean link latency ignoring any congestion event.

        Cost references are built from the *uncongested* network on purpose. If
        they used the degraded values instead, a congestion event would inflate
        the normaliser by exactly as much as it inflates the measurement, and the
        objective would report congestion as costing nothing.
        """
        off_diagonal = ~np.eye(self.num_devices, dtype=bool)
        return float(self.base_latency_ms[off_diagonal].mean())

    def base_mean_inverse_bandwidth(self) -> float:
        """Mean of ``1 / bandwidth`` ignoring any congestion event."""
        off_diagonal = ~np.eye(self.num_devices, dtype=bool)
        return float((1.0 / self.base_bandwidth_mbps[off_diagonal]).mean())

    def mean_link_bandwidth_mbps(self, layer_index: int = 0) -> float:
        """Mean bandwidth over all distinct links, used for reporting."""
        _, bandwidth_mbps = self.conditions_at(layer_index)
        off_diagonal = ~np.eye(self.num_devices, dtype=bool)
        return float(bandwidth_mbps[off_diagonal].mean())

    def mean_inverse_bandwidth(self, layer_index: int = 0) -> float:
        """Mean of ``1 / bandwidth`` over all distinct links.

        Transfer time is inversely proportional to bandwidth, so the *expected*
        transfer time over a uniformly chosen link uses this mean rather than
        the mean bandwidth. With links spanning an order of magnitude the two
        differ substantially, and using the wrong one would leave the objective
        mis-calibrated.
        """
        _, bandwidth_mbps = self.conditions_at(layer_index)
        off_diagonal = ~np.eye(self.num_devices, dtype=bool)
        return float((1.0 / bandwidth_mbps[off_diagonal]).mean())


def sample_network(
    config: Config, rng: np.random.Generator, num_layers: int
) -> NetworkModel:
    """Sample link characteristics and, if the profile allows, a congestion event.

    Args:
        config: The loaded configuration.
        rng: Generator used for link and congestion draws.
        num_layers: Episode length, used to place a mid-episode event.

    Returns:
        A :class:`NetworkModel` for one scenario.
    """
    device_names = config.device_names
    size = len(device_names)
    latency_ms = np.zeros((size, size), dtype=np.float64)
    bandwidth_mbps = np.ones((size, size), dtype=np.float64)

    for link in config.network.links:
        first = device_names.index(link.endpoints[0])
        second = device_names.index(link.endpoints[1])
        latency = link.latency_ms.sample(rng)
        bandwidth = link.bandwidth_mbps.sample(rng)
        latency_ms[first, second] = latency_ms[second, first] = latency
        bandwidth_mbps[first, second] = bandwidth_mbps[second, first] = bandwidth

    congestion = _sample_congestion(config.network.active_profile, rng, num_layers)
    return NetworkModel(latency_ms, bandwidth_mbps, congestion)


def _sample_congestion(
    profile: NetworkProfileConfig, rng: np.random.Generator, num_layers: int
) -> CongestionEvent | None:
    """Draw a congestion event for one scenario, or ``None`` if none occurs."""
    if profile.event_probability <= 0.0 or rng.random() >= profile.event_probability:
        return None

    start_layer = 0
    mid_episode = num_layers > 1 and rng.random() < profile.mid_episode_probability
    if mid_episode:
        # Start strictly inside the episode so that conditions genuinely change
        # while the agent is placing layers.
        start_layer = int(rng.integers(1, num_layers))

    return CongestionEvent(
        start_layer=start_layer,
        latency_scale=profile.latency_scale.sample(rng),
        bandwidth_scale=profile.bandwidth_scale.sample(rng),
    )


__all__ = ["CongestionEvent", "NetworkModel", "sample_network"]
