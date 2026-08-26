"""Tests for deterministic seeding helpers."""

from __future__ import annotations

import numpy as np

from src.utils.seed import MAX_SEED, derive_seed, make_rng, seed_everything


def test_derive_seed_is_deterministic():
    assert derive_seed(42, "workload") == derive_seed(42, "workload")


def test_derive_seed_separates_labels_and_bases():
    assert derive_seed(42, "workload") != derive_seed(42, "network")
    assert derive_seed(42, "workload") != derive_seed(43, "workload")


def test_derive_seed_stays_in_range():
    for base in range(50):
        seed = derive_seed(base, "devices")
        assert 0 <= seed < MAX_SEED


def test_make_rng_reproduces_the_same_stream():
    first = make_rng(7, "workload").random(10)
    second = make_rng(7, "workload").random(10)
    np.testing.assert_array_equal(first, second)


def test_make_rng_streams_are_independent_across_labels():
    workload = make_rng(7, "workload").random(10)
    network = make_rng(7, "network").random(10)
    assert not np.allclose(workload, network)


def test_seed_everything_makes_numpy_and_random_reproducible():
    import random

    seed_everything(123)
    first = (random.random(), np.random.rand())
    seed_everything(123)
    second = (random.random(), np.random.rand())
    assert first == second
