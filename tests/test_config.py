"""Tests for configuration loading, merging and validation."""

from __future__ import annotations

import pytest

from src.config import (
    Config,
    ConfigError,
    Range,
    build_config,
    load_config,
    load_yaml,
    parse_overrides,
)


def test_default_config_loads_and_validates():
    config = load_config()
    assert isinstance(config, Config)
    assert config.num_devices == 4
    assert config.device_names[0] == "edge_device"
    assert config.workload.num_layers == 10
    assert config.network.profile == "normal"


def test_yaml_underscored_integers_parse_as_numbers():
    """`1_000_000` in YAML must arrive as an int, not a string."""
    config = load_config()
    assert isinstance(config.experiment.train_seed_start, int)
    assert config.experiment.train_seed_start == 1_000_000
    assert config.dqn.buffer_capacity == 100_000


def test_objective_weights_must_sum_to_one():
    with pytest.raises(ConfigError, match="sum to 1"):
        load_config(overrides=["objective.alpha=0.9"])


def test_negative_objective_weight_is_rejected():
    with pytest.raises(ConfigError, match="non-negative"):
        load_config(overrides=["objective.alpha=-0.2", "objective.beta=1.0", "objective.gamma=0.2"])


def test_overrides_are_applied_and_typed():
    config = load_config(overrides=["workload.num_layers=20", "environment.memory_accumulates=false"])
    assert config.workload.num_layers == 20
    assert config.environment.memory_accumulates is False


def test_parse_overrides_builds_nested_mapping():
    assert parse_overrides(["a.b.c=3", "a.d=true"]) == {"a": {"b": {"c": 3}, "d": True}}


def test_parse_overrides_rejects_malformed_entries():
    with pytest.raises(ConfigError, match="section.key=value"):
        parse_overrides(["not-an-override"])


def test_experiment_file_merges_on_top_of_defaults(tmp_path):
    experiment_file = tmp_path / "experiment.yaml"
    experiment_file.write_text("workload:\n  num_layers: 5\n", encoding="utf-8")
    config = load_config(experiment_file)
    # Overridden key changes...
    assert config.workload.num_layers == 5
    # ...while untouched keys keep their default values.
    assert config.workload.pool_every == 2
    assert config.num_devices == 4


def test_unknown_network_profile_is_rejected():
    with pytest.raises(ConfigError, match="not defined in network.profiles"):
        load_config(overrides=["network.profile=hurricane"])


def test_invalid_action_mode_is_validated():
    with pytest.raises(ConfigError, match="invalid_action_mode"):
        load_config(overrides=["environment.invalid_action_mode=explode"])


def test_train_and_eval_seed_pools_must_not_overlap():
    with pytest.raises(ConfigError, match="overlap"):
        load_config(overrides=["experiment.eval_seed_start=1000010"])


def test_incomplete_network_topology_is_rejected():
    data = load_yaml("default.yaml")
    data["network"]["links"] = data["network"]["links"][:-1]
    with pytest.raises(ConfigError, match="fully connected"):
        build_config(data)


def test_duplicate_device_names_are_rejected():
    data = load_yaml("default.yaml")
    data["devices"][1]["name"] = data["devices"][0]["name"]
    with pytest.raises(ConfigError, match="unique"):
        build_config(data)


def test_device_index_lookup():
    config = load_config()
    assert config.device_index("gpu_server") == 2
    with pytest.raises(ConfigError, match="unknown device"):
        config.device_index("quantum_server")


class TestRange:
    def test_scalar_becomes_fixed_range(self):
        parsed = Range.parse(3.5, field_name="x")
        assert parsed.is_fixed
        assert parsed.midpoint == 3.5

    def test_pair_becomes_interval(self):
        parsed = Range.parse([1.0, 3.0], field_name="x")
        assert (parsed.low, parsed.high) == (1.0, 3.0)
        assert parsed.midpoint == 2.0

    def test_sample_stays_inside_the_interval(self):
        import numpy as np

        rng = np.random.default_rng(0)
        parsed = Range.parse([2.0, 4.0], field_name="x")
        samples = [parsed.sample(rng) for _ in range(200)]
        assert all(2.0 <= value <= 4.0 for value in samples)

    def test_inverted_range_is_rejected(self):
        with pytest.raises(ConfigError, match="below lower bound"):
            Range.parse([5.0, 1.0], field_name="x")

    def test_wrong_length_is_rejected(self):
        with pytest.raises(ConfigError, match="exactly two entries"):
            Range.parse([1.0, 2.0, 3.0], field_name="x")
