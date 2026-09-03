"""Tests for the device, network and objective arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.environment.devices import DeviceFleet, DeviceProfile
from src.environment.network import CongestionEvent, NetworkModel
from src.environment.reward import (
    CostReferences,
    ObjectiveWeights,
    compute_references,
    evaluate_placement,
    weighted_cost,
)
from src.environment.scenario import sample_scenario


@pytest.fixture
def config():
    return load_config()


def make_fleet(environment, base_utilisation=0.0):
    profiles = (
        DeviceProfile("slow", compute_capacity=1.0, memory_gb=4.0, energy_per_compute=0.5,
                      base_utilisation=base_utilisation),
        DeviceProfile("fast", compute_capacity=10.0, memory_gb=32.0, energy_per_compute=3.0,
                      base_utilisation=0.0),
    )
    return DeviceFleet(profiles, environment)


class TestDeviceFleet:
    def test_execution_time_is_compute_over_speed(self, config):
        fleet = make_fleet(config.environment)
        # 2 compute units on a device delivering 1 unit/s takes 2 s = 2000 ms.
        assert fleet.execution_time_ms(2.0)[0] == pytest.approx(2000.0)
        assert fleet.execution_time_ms(2.0)[1] == pytest.approx(200.0)

    def test_background_load_slows_a_device(self, config):
        fleet = make_fleet(config.environment, base_utilisation=0.5)
        # Half the capacity is already taken, so the layer takes twice as long.
        assert fleet.execution_time_ms(1.0)[0] == pytest.approx(2000.0)

    def test_energy_is_compute_times_rate(self, config):
        fleet = make_fleet(config.environment)
        np.testing.assert_allclose(fleet.execution_energy(2.0), [1.0, 6.0])

    def test_assignment_consumes_memory_when_accumulation_is_on(self, config):
        fleet = make_fleet(config.environment)
        assert config.environment.memory_accumulates
        fleet.assign(0, layer_compute_cost=1.0, layer_memory_gb=3.0)
        assert fleet.free_memory_gb()[0] == pytest.approx(1.0)
        assert fleet.can_host(0, 0.9)
        assert not fleet.can_host(0, 1.5)

    def test_memory_does_not_accumulate_when_disabled(self):
        config = load_config(overrides=["environment.memory_accumulates=false"])
        fleet = make_fleet(config.environment)
        fleet.assign(0, layer_compute_cost=1.0, layer_memory_gb=3.0)
        assert fleet.free_memory_gb()[0] == pytest.approx(4.0)

    def test_assigned_load_raises_utilisation(self, config):
        fleet = make_fleet(config.environment)
        before = fleet.effective_speed()[0]
        fleet.assign(0, layer_compute_cost=4.0, layer_memory_gb=0.1)
        assert fleet.effective_speed()[0] < before

    def test_effective_speed_never_falls_below_the_floor(self, config):
        fleet = make_fleet(config.environment)
        fleet.assign(0, layer_compute_cost=1_000.0, layer_memory_gb=0.1)
        floor = 1.0 * config.environment.effective_speed_floor
        assert fleet.effective_speed()[0] >= floor

    def test_feasibility_mask_reflects_free_memory(self, config):
        fleet = make_fleet(config.environment)
        np.testing.assert_array_equal(fleet.feasibility_mask(5.0), [False, True])
        np.testing.assert_array_equal(fleet.feasibility_mask(1.0), [True, True])

    def test_reset_clears_accumulated_state(self, config):
        fleet = make_fleet(config.environment)
        fleet.assign(0, 1.0, 2.0)
        fleet.reset()
        assert fleet.free_memory_gb()[0] == pytest.approx(4.0)

    def test_copy_is_independent(self, config):
        fleet = make_fleet(config.environment)
        clone = fleet.copy()
        clone.assign(0, 1.0, 2.0)
        assert fleet.free_memory_gb()[0] == pytest.approx(4.0)

    def test_out_of_range_assignment_raises(self, config):
        fleet = make_fleet(config.environment)
        with pytest.raises(IndexError):
            fleet.assign(5, 1.0, 1.0)


class TestNetworkModel:
    def build(self, congestion=None):
        latency = np.array([[0.0, 10.0], [10.0, 0.0]])
        bandwidth = np.array([[1.0, 100.0], [100.0, 1.0]])
        return NetworkModel(latency, bandwidth, congestion)

    def test_transfer_time_matches_the_documented_formula(self):
        network = self.build()
        # 10 ms latency + 5 MB * 8 bits / 100 Mbps = 10 + 0.4 s = 410 ms
        assert network.transfer_time_ms(0, 1, 5.0, layer_index=0) == pytest.approx(410.0)

    def test_transfers_within_a_device_are_free(self):
        network = self.build()
        assert network.transfer_time_ms(0, 0, 100.0, layer_index=0) == 0.0
        assert network.transfer_times_ms(1, 100.0, layer_index=0)[1] == 0.0

    def test_congestion_from_layer_zero_applies_everywhere(self):
        network = self.build(CongestionEvent(0, latency_scale=2.0, bandwidth_scale=0.5))
        # Latency doubles, bandwidth halves so the payload term doubles.
        assert network.transfer_time_ms(0, 1, 5.0, layer_index=0) == pytest.approx(20.0 + 800.0)

    def test_mid_episode_congestion_only_affects_later_layers(self):
        network = self.build(CongestionEvent(3, latency_scale=2.0, bandwidth_scale=0.5))
        assert not network.is_congested_at(2)
        assert network.is_congested_at(3)
        assert network.transfer_time_ms(0, 1, 5.0, layer_index=2) == pytest.approx(410.0)
        assert network.transfer_time_ms(0, 1, 5.0, layer_index=3) == pytest.approx(820.0)

    def test_asymmetric_matrices_are_rejected(self):
        latency = np.array([[0.0, 10.0], [4.0, 0.0]])
        bandwidth = np.array([[1.0, 100.0], [100.0, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            NetworkModel(latency, bandwidth)

    def test_non_positive_bandwidth_is_rejected(self):
        latency = np.array([[0.0, 10.0], [10.0, 0.0]])
        bandwidth = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="bandwidths must be positive"):
            NetworkModel(latency, bandwidth)


class TestObjective:
    def test_weighted_cost_is_linear_in_its_arguments(self):
        references = CostReferences(latency_ms=100.0, energy=10.0, communication_ms=20.0)
        weights = ObjectiveWeights(0.5, 0.3, 0.2)
        whole = weighted_cost(90.0, 30.0, 6.0, references, weights, comm_double_count=True)
        first = weighted_cost(40.0, 10.0, 2.0, references, weights, comm_double_count=True)
        second = weighted_cost(50.0, 20.0, 4.0, references, weights, comm_double_count=True)
        assert whole == pytest.approx(first + second)

    def test_weighted_cost_matches_a_hand_computation(self):
        references = CostReferences(latency_ms=100.0, energy=10.0, communication_ms=20.0)
        weights = ObjectiveWeights(0.5, 0.3, 0.2)
        # 0.5 * 90/100 + 0.3 * 6/10 + 0.2 * 30/20 = 0.45 + 0.18 + 0.30
        cost = weighted_cost(90.0, 30.0, 6.0, references, weights, comm_double_count=True)
        assert cost == pytest.approx(0.93)

    def test_disabling_double_counting_removes_communication_from_latency(self):
        references = CostReferences(latency_ms=100.0, energy=10.0, communication_ms=20.0)
        weights = ObjectiveWeights(0.5, 0.3, 0.2)
        # Latency term now covers computation only: 0.5 * 60/100 + 0.18 + 0.30
        cost = weighted_cost(90.0, 30.0, 6.0, references, weights, comm_double_count=False)
        assert cost == pytest.approx(0.78)

    def test_weights_must_sum_to_one_in_configuration(self, config):
        weights = ObjectiveWeights.from_config(config.objective)
        assert weights.alpha + weights.beta + weights.gamma == pytest.approx(1.0)

    def test_references_are_positive_and_scale_with_depth(self, config):
        small = sample_scenario(config, 9_000_000, num_layers=5)
        large = sample_scenario(config, 9_000_000, num_layers=20)
        for scenario in (small, large):
            assert scenario.references.latency_ms > 0
            assert scenario.references.energy > 0
            assert scenario.references.communication_ms > 0
        assert large.references.energy > small.references.energy

    def test_congestion_raises_the_objective_rather_than_being_normalised_away(self, config):
        """The property, not a proxy for it.

        References are built from the uncongested network. Building them from the
        degraded network instead would inflate the normaliser by as much as the
        measurement, and congestion would appear to cost nothing -- which is
        exactly the bug this asserts against.
        """
        congested_config = load_config(overrides=["network.profile=congested"])
        worse = 0
        for seed in range(9_000_100, 9_000_140):
            calm = sample_scenario(config, seed)
            congested = sample_scenario(congested_config, seed)
            assert congested.has_congestion
            placement = tuple(index % calm.num_devices for index in range(calm.num_layers))

            calm_result = evaluate_placement(calm, placement, config)
            congested_result = evaluate_placement(congested, placement, congested_config)
            assert congested_result.communication_latency_ms > calm_result.communication_latency_ms
            if congested_result.objective > calm_result.objective:
                worse += 1
        assert worse == 40, "congestion must make an identical placement score worse"

    def test_references_ignore_the_congestion_event(self, config):
        congested_config = load_config(overrides=["network.profile=congested"])
        calm = sample_scenario(config, 9_000_123)
        congested = sample_scenario(congested_config, 9_000_123)
        assert congested.has_congestion
        # Same seed, same links before congestion is applied, so identical references.
        assert congested.references.communication_ms == pytest.approx(
            calm.references.communication_ms, rel=1e-9
        )

class TestEvaluatePlacement:
    def test_single_device_placement_has_no_communication_after_the_input(self, config):
        scenario = sample_scenario(config, 9_000_001)
        source = scenario.input_source_index
        placement = tuple([source] * scenario.num_layers)
        result = evaluate_placement(scenario, placement, config)
        # Input already lives on the source device and nothing ever moves.
        assert result.communication_latency_ms == pytest.approx(0.0)
        assert result.device_switches == 0

    def test_wrong_length_placement_is_rejected(self, config):
        scenario = sample_scenario(config, 9_000_002)
        with pytest.raises(ValueError, match="entries but the DNN has"):
            evaluate_placement(scenario, (0, 1), config)

    def test_out_of_range_device_is_rejected(self, config):
        scenario = sample_scenario(config, 9_000_003)
        placement = tuple([99] * scenario.num_layers)
        with pytest.raises(ValueError, match="invalid device index"):
            evaluate_placement(scenario, placement, config)

    def test_totals_decompose_into_computation_and_communication(self, config):
        scenario = sample_scenario(config, 9_000_004)
        placement = tuple(index % scenario.num_devices for index in range(scenario.num_layers))
        result = evaluate_placement(scenario, placement, config)
        assert result.total_latency_ms == pytest.approx(
            result.compute_latency_ms + result.communication_latency_ms
        )
