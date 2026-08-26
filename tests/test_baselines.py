"""Tests for the exact placement baselines."""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines import dp_optimal, exhaustive_search
from src.config import load_config
from src.environment.reward import evaluate_placement
from src.environment.scenario import evaluation_seeds, sample_scenario

# Accumulation disabled: the setting in which the dynamic programme is exact.
STATIC_OVERRIDES = [
    "environment.memory_accumulates=false",
    "environment.utilisation_accumulates=false",
]


@pytest.fixture
def static_config():
    return load_config(overrides=STATIC_OVERRIDES)


@pytest.fixture
def accumulating_config():
    return load_config()


class TestDynamicProgramming:
    @pytest.mark.parametrize("depth", [1, 2, 5, 7])
    def test_matches_exhaustive_search_when_exact(self, static_config, depth):
        """The core correctness check: DP must equal brute force, exactly."""
        assert dp_optimal.is_exact_for(static_config)
        for seed in evaluation_seeds(static_config, 25):
            scenario = sample_scenario(static_config, seed, num_layers=depth)
            dp_solution = dp_optimal.solve(scenario, static_config)
            brute = exhaustive_search.solve(scenario, static_config)
            assert dp_solution.objective == pytest.approx(brute.objective, rel=1e-9)

    def test_reported_objective_matches_replaying_the_placement(self, static_config):
        for seed in evaluation_seeds(static_config, 20):
            scenario = sample_scenario(static_config, seed, num_layers=6)
            solution = dp_optimal.solve(scenario, static_config)
            replayed = evaluate_placement(scenario, solution.placement, static_config)
            assert replayed.objective == pytest.approx(solution.objective, rel=1e-9)
            assert replayed.memory_violations == 0

    def test_flags_itself_as_exact_only_when_it_is(self, static_config, accumulating_config):
        assert dp_optimal.is_exact_for(static_config)
        assert not dp_optimal.is_exact_for(accumulating_config)
        scenario = sample_scenario(static_config, 9_000_000, num_layers=5)
        assert dp_optimal.solve(scenario, static_config).is_exact
        assert dp_optimal.solve(scenario, accumulating_config).is_exact is False

    def test_is_a_lower_bound_under_accumulation(self, accumulating_config):
        """The relaxation may never claim a cost above the true optimum."""
        assert not dp_optimal.is_exact_for(accumulating_config)
        for seed in evaluation_seeds(accumulating_config, 25):
            scenario = sample_scenario(accumulating_config, seed, num_layers=6)
            bound = dp_optimal.solve(scenario, accumulating_config).objective
            true_optimum = exhaustive_search.solve(scenario, accumulating_config).objective
            assert bound <= true_optimum + 1e-9

    def test_beats_every_heuristic_it_is_compared_against(self, static_config):
        from src.agents.greedy_agent import GreedyAgent
        from src.training.evaluate import evaluate_agent

        seeds = evaluation_seeds(static_config, 40)
        greedy = evaluate_agent(
            static_config, GreedyAgent(static_config.num_devices, "objective_aware"), seeds
        )
        for record in greedy:
            scenario = sample_scenario(static_config, record.scenario_seed)
            optimum = dp_optimal.solve(scenario, static_config).objective
            assert optimum <= record.objective + 1e-9

    def test_solves_deep_networks_the_brute_force_cannot(self, static_config):
        scenario = sample_scenario(static_config, 9_000_000, num_layers=50)
        solution = dp_optimal.solve(scenario, static_config)
        assert len(solution.placement) == 50
        assert np.isfinite(solution.objective)

    def test_single_layer_networks_are_handled(self, static_config):
        scenario = sample_scenario(static_config, 9_000_001, num_layers=1)
        solution = dp_optimal.solve(scenario, static_config)
        brute = exhaustive_search.solve(scenario, static_config)
        assert solution.objective == pytest.approx(brute.objective, rel=1e-9)


class TestExhaustiveSearch:
    def test_counts_the_search_space_correctly(self):
        assert exhaustive_search.count_placements(5, 4) == 1024
        assert exhaustive_search.count_placements(10, 4) == 1_048_576

    def test_refuses_searches_beyond_the_budget(self, static_config):
        scenario = sample_scenario(static_config, 9_000_000, num_layers=20)
        with pytest.raises(exhaustive_search.SearchTooLargeError, match="exceeds the budget"):
            exhaustive_search.solve(scenario, static_config)

    def test_affordability_helper_matches_the_budget(self, static_config):
        budget = static_config.experiment.max_exhaustive_combinations
        assert exhaustive_search.is_affordable(5, 4, budget)
        assert not exhaustive_search.is_affordable(20, 4, budget)

    def test_returned_placement_is_feasible_and_optimal(self, accumulating_config):
        scenario = sample_scenario(accumulating_config, 9_000_002, num_layers=5)
        solution = exhaustive_search.solve(scenario, accumulating_config)
        replayed = evaluate_placement(scenario, solution.placement, accumulating_config)
        assert replayed.memory_violations == 0
        assert replayed.objective == pytest.approx(solution.objective, rel=1e-9)
        assert solution.feasible_found <= solution.evaluated

    def test_no_feasible_placement_is_reported_clearly(self, accumulating_config):
        """A memory-impossible scenario must raise rather than return nonsense."""
        import dataclasses

        scenario = sample_scenario(accumulating_config, 9_000_003, num_layers=3)
        huge = dataclasses.replace(scenario.workload.layers[0], memory_gb=10_000.0)
        broken_workload = dataclasses.replace(
            scenario.workload, layers=(huge,) + scenario.workload.layers[1:]
        )
        broken = dataclasses.replace(scenario, workload=broken_workload)
        with pytest.raises(ValueError, match="no feasible placement"):
            exhaustive_search.solve(broken, accumulating_config)
