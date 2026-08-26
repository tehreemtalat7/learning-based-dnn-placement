"""Tests for the synthetic DNN workload generator."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.environment.workload import generate_workload
from src.utils.seed import make_rng


@pytest.fixture
def config():
    return load_config()


@pytest.mark.parametrize("depth", [1, 2, 5, 10, 20, 30, 50])
def test_generated_depth_matches_the_request(config, depth):
    workload = generate_workload(config.workload, make_rng(1, "workload"), num_layers=depth)
    assert len(workload) == depth


def test_all_quantities_are_positive(config):
    workload = generate_workload(config.workload, make_rng(2, "workload"), num_layers=20)
    assert (workload.compute_costs > 0).all()
    assert (workload.memory_requirements_gb > 0).all()
    assert (workload.output_sizes_mb >= 0).all()


def test_generation_is_reproducible(config):
    first = generate_workload(config.workload, make_rng(3, "workload"), num_layers=12)
    second = generate_workload(config.workload, make_rng(3, "workload"), num_layers=12)
    np.testing.assert_array_equal(first.compute_costs, second.compute_costs)
    np.testing.assert_array_equal(first.output_sizes_mb, second.output_sizes_mb)


def test_different_seeds_give_different_workloads(config):
    first = generate_workload(config.workload, make_rng(4, "workload"), num_layers=12)
    second = generate_workload(config.workload, make_rng(5, "workload"), num_layers=12)
    assert not np.allclose(first.compute_costs, second.compute_costs)


def test_activations_shrink_through_the_feature_stage(config):
    """The early layers must be the expensive ones to move, as in a real CNN."""
    workload = generate_workload(config.workload, make_rng(6, "workload"), num_layers=20)
    feature_outputs = workload.output_sizes_mb[:10]
    assert feature_outputs[0] > feature_outputs[-1]


def test_the_dense_head_needs_more_memory_than_feature_layers(config):
    workload = generate_workload(config.workload, make_rng(7, "workload"), num_layers=20)
    head_count = max(1, round(20 * config.workload.head_fraction))
    feature_memory = workload.memory_requirements_gb[:-head_count]
    head_memory = workload.memory_requirements_gb[-head_count:]
    assert head_memory.max() > feature_memory.max()


def test_the_final_classifier_emits_a_tiny_activation(config):
    workload = generate_workload(config.workload, make_rng(8, "workload"), num_layers=15)
    assert workload.layers[-1].name == "classifier"
    assert workload.output_sizes_mb[-1] < workload.output_sizes_mb[0]


def test_memory_cap_keeps_every_layer_placeable(config):
    """No scenario may contain a layer that fits on no device."""
    cap = 2.0
    workload = generate_workload(
        config.workload, make_rng(9, "workload"), num_layers=30, max_layer_memory_gb=cap
    )
    assert workload.memory_requirements_gb.max() <= cap


def test_zero_jitter_makes_generation_smooth(config):
    quiet = load_config(overrides=["workload.jitter_sigma=0"])
    workload = generate_workload(quiet.workload, make_rng(10, "workload"), num_layers=10)
    head_count = max(1, round(10 * quiet.workload.head_fraction))
    outputs = workload.output_sizes_mb[: 10 - head_count]
    assert all(later <= earlier for earlier, later in zip(outputs, outputs[1:]))


def test_invalid_depth_is_rejected(config):
    with pytest.raises(ValueError, match="at least 1"):
        generate_workload(config.workload, make_rng(11, "workload"), num_layers=0)
