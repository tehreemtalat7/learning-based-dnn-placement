"""Tests for the Phase 5 experiment machinery.

These check the parts an experiment could get quietly wrong: whether a sweep
really holds everything else fixed, whether every method is measured on the same
scenarios, and whether a missing checkpoint degrades gracefully instead of
producing a table that looks complete.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments._shared import compare_methods, optional_agents, relative_to
from src.config import ConfigError, load_config
from src.environment.dnn_environment import DNNPlacementEnv
from src.environment.scenario import evaluation_seeds, sample_scenario


@pytest.fixture
def config():
    return load_config()


class TestDeviceLoadOverride:
    def test_overrides_only_the_named_device(self, config):
        loaded = load_config(overrides=["environment.device_load_override.gpu_server=[0.8, 0.8]"])
        base_scenario = sample_scenario(config, 9_000_000)
        loaded_scenario = sample_scenario(loaded, 9_000_000)

        index = config.device_index("gpu_server")
        assert loaded_scenario.devices[index].base_utilisation == pytest.approx(0.8)
        for position, (before, after) in enumerate(
            zip(base_scenario.devices, loaded_scenario.devices, strict=True)
        ):
            # Capacity, memory and energy are untouched everywhere, and the load
            # of the other devices is untouched too.
            assert before.compute_capacity == pytest.approx(after.compute_capacity)
            assert before.memory_gb == pytest.approx(after.memory_gb)
            assert before.energy_per_compute == pytest.approx(after.energy_per_compute)
            if position != index:
                assert before.base_utilisation == pytest.approx(after.base_utilisation)

    def test_workload_and_network_are_unaffected(self, config):
        loaded = load_config(overrides=["environment.device_load_override.edge_device=[0.9, 0.9]"])
        base_scenario = sample_scenario(config, 9_000_007)
        loaded_scenario = sample_scenario(loaded, 9_000_007)
        np.testing.assert_array_equal(
            base_scenario.workload.compute_costs, loaded_scenario.workload.compute_costs
        )
        np.testing.assert_array_equal(
            base_scenario.network.base_latency_ms, loaded_scenario.network.base_latency_ms
        )

    def test_a_loaded_device_is_slower(self, config):
        loaded = load_config(overrides=["environment.device_load_override.gpu_server=[0.8, 0.8]"])
        index = config.device_index("gpu_server")
        quiet_env, busy_env = DNNPlacementEnv(config), DNNPlacementEnv(loaded)
        _o, quiet_info = quiet_env.reset(options={"scenario_seed": 9_000_001})
        _o, busy_info = busy_env.reset(options={"scenario_seed": 9_000_001})
        assert (
            busy_info["candidate_execution_ms"][index]
            > quiet_info["candidate_execution_ms"][index]
        )

    def test_unknown_device_name_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown device"):
            load_config(overrides=["environment.device_load_override.tpu_pod=[0.5, 0.5]"])


class TestComparisonHarness:
    def test_every_method_is_measured_on_every_scenario(self, config):
        seeds = evaluation_seeds(config, 6)
        frame = compare_methods(config, seeds, num_layers=5, include_exhaustive=False)
        counts = frame.groupby("method")["scenario_seed"].nunique()
        assert set(counts) == {len(seeds)}, "methods were not measured on identical scenarios"
        assert set(frame["scenario_seed"]) == set(seeds)

    def test_gap_against_the_best_placement_is_present_and_non_negative(self, config):
        seeds = evaluation_seeds(config, 6)
        frame = compare_methods(config, seeds, num_layers=5)
        assert "gap_vs_best_pct" in frame.columns
        assert (frame["gap_vs_best_pct"] >= -1e-9).all()
        assert frame["gap_vs_best_pct"].min() == pytest.approx(0.0, abs=1e-9)

    def test_exhaustive_search_is_included_when_asked(self, config):
        seeds = evaluation_seeds(config, 3)
        frame = compare_methods(config, seeds, num_layers=5, include_exhaustive=True)
        assert "exhaustive" in set(frame["method"])
        # Nothing may beat the true optimum.
        best = frame[frame["method"] == "exhaustive"].set_index("scenario_seed")["objective"]
        for method in set(frame["method"]):
            objectives = frame[frame["method"] == method].set_index("scenario_seed")["objective"]
            assert (objectives >= best - 1e-9).all()

    def test_depth_override_is_honoured(self, config):
        frame = compare_methods(config, evaluation_seeds(config, 3), num_layers=7)
        assert set(frame["num_layers"]) == {7}

    def test_missing_checkpoints_are_skipped_not_fatal(self, config, tmp_path, capsys):
        agents = optional_agents(
            config,
            dqn_checkpoints={"dqn_absent": tmp_path / "nope.pt"},
            include_supervised=False,
            include_tabular=False,
        )
        assert agents == []
        assert "no checkpoint" in capsys.readouterr().out

    def test_relative_to_computes_per_scenario_percentages(self):
        frame = pd.DataFrame(
            {
                "method": ["a", "b", "a", "b"],
                "scenario_seed": [1, 1, 2, 2],
                "objective": [1.0, 1.5, 2.0, 1.0],
            }
        )
        result = relative_to(frame, "a")
        b_rows = result[result["method"] == "b"].sort_values("scenario_seed")
        assert list(np.round(b_rows["vs_a_pct"], 6)) == [50.0, -50.0]


class TestMixedDepthTraining:
    def test_the_environment_samples_from_a_list_of_depths(self, config):
        env = DNNPlacementEnv(config, num_layers=[5, 20], seed=1)
        depths = set()
        for _ in range(30):
            _observation, info = env.reset()
            depths.add(info["num_layers"])
        assert depths == {5, 20}

    def test_the_observation_width_is_the_same_at_every_depth(self, config):
        env = DNNPlacementEnv(config, num_layers=[5, 30], seed=2)
        widths = set()
        for _ in range(10):
            observation, _info = env.reset()
            widths.add(observation.shape)
        assert len(widths) == 1, "a fixed-width state is what lets one policy span depths"
