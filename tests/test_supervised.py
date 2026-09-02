"""Tests for the Random Forest imitation baseline."""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines import dp_optimal, exhaustive_search
from src.baselines.supervised_ml import (
    SupervisedAgent,
    collect_demonstrations,
    evaluate_imitation_accuracy,
    load_model,
    save_model,
    teacher_placement,
    train_random_forest,
)
from src.config import load_config
from src.environment.dnn_environment import DNNPlacementEnv
from src.environment.observation import observation_size
from src.environment.reward import evaluate_placement
from src.environment.scenario import evaluation_seeds, sample_scenario, training_seeds
from src.training.evaluate import evaluate_agent
from src.utils.seed import make_rng


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def small_config():
    """Five layers, where the teacher is the true optimum."""
    return load_config(overrides=["workload.num_layers=5"])


class TestTeacher:
    def test_auto_uses_exhaustive_search_at_small_depths(self, small_config):
        env = DNNPlacementEnv(small_config)
        scenario = sample_scenario(small_config, 9_000_000, num_layers=5)
        placement, source = teacher_placement(scenario, small_config, env)
        assert source == "exhaustive"
        assert placement == exhaustive_search.solve(scenario, small_config).placement

    def test_auto_falls_back_to_best_known_at_larger_depths(self, config):
        env = DNNPlacementEnv(config)
        scenario = sample_scenario(config, 9_000_000)
        _placement, source = teacher_placement(scenario, config, env)
        assert source in {"dp", "greedy_objective_aware", "greedy_communication_aware"}

    def test_teacher_is_never_worse_than_the_dynamic_programme(self, config):
        """The whole point of the best-known teacher: it may not be handicapped."""
        env = DNNPlacementEnv(config)
        for seed in evaluation_seeds(config, 15):
            scenario = sample_scenario(config, seed)
            placement, _source = teacher_placement(scenario, config, env)
            teacher_objective = evaluate_placement(scenario, placement, config).objective
            dp_placement = dp_optimal.solve(scenario, config).placement
            dp_objective = evaluate_placement(scenario, dp_placement, config).objective
            assert teacher_objective <= dp_objective + 1e-9

    def test_teachers_produce_feasible_placements(self, config):
        env = DNNPlacementEnv(config)
        for seed in evaluation_seeds(config, 15):
            scenario = sample_scenario(config, seed)
            placement, _source = teacher_placement(scenario, config, env)
            assert evaluate_placement(scenario, placement, config).memory_violations == 0

    def test_explicit_dp_teacher_is_respected(self, config):
        forced = load_config(overrides=["supervised.teacher=dp"])
        env = DNNPlacementEnv(forced)
        scenario = sample_scenario(forced, 9_000_001)
        placement, source = teacher_placement(scenario, forced, env)
        assert source == "dp"
        assert placement == dp_optimal.solve(scenario, forced).placement

    def test_exhaustive_teacher_refuses_depths_it_cannot_reach(self, config):
        forced = load_config(
            overrides=["supervised.teacher=exhaustive", "supervised.max_exhaustive_layers=6"]
        )
        env = DNNPlacementEnv(forced)
        scenario = sample_scenario(forced, 9_000_002, num_layers=10)
        with pytest.raises(ValueError, match="max_exhaustive_layers"):
            teacher_placement(scenario, forced, env)


class TestDemonstrations:
    def test_shapes_and_labels_line_up(self, config):
        seeds = training_seeds(config, 5, make_rng(0, "test"))
        data = collect_demonstrations(config, seeds, progress_every=0)
        assert data.observations.shape == (5 * config.workload.num_layers,
                                           observation_size(config.num_devices))
        assert data.actions.shape[0] == data.observations.shape[0]
        assert len(data.feature_names) == data.observations.shape[1]
        assert set(np.unique(data.actions)).issubset(set(range(config.num_devices)))

    def test_labels_reproduce_the_teacher_placement(self, config):
        seeds = training_seeds(config, 3, make_rng(1, "test"))
        data = collect_demonstrations(config, seeds, progress_every=0)
        env = DNNPlacementEnv(config)
        for seed in seeds:
            rows = data.scenario_seeds == seed
            recorded = tuple(data.actions[rows])
            scenario = sample_scenario(config, seed)
            expected, _source = teacher_placement(scenario, config, env)
            assert recorded == expected

    def test_observations_are_collected_along_the_teacher_trajectory(self, config):
        """States must come from following the teacher, not from an arbitrary rollout."""
        seeds = training_seeds(config, 2, make_rng(2, "test"))
        data = collect_demonstrations(config, seeds, progress_every=0)
        # The progress feature is the first entry and must step through the DNN.
        depth = config.workload.num_layers
        first_episode = data.observations[:depth, 0]
        np.testing.assert_allclose(
            first_episode, np.arange(depth) / depth, rtol=1e-5
        )


class TestSupervisedAgent:
    @pytest.fixture
    def trained(self, config):
        seeds = training_seeds(config, 40, make_rng(3, "test"))
        data = collect_demonstrations(config, seeds, progress_every=0)
        quick = load_config(overrides=["supervised.n_estimators=20"])
        return train_random_forest(data, quick, seed=0), data

    def test_fits_its_training_data(self, trained):
        model, data = trained
        accuracy = evaluate_imitation_accuracy(model, data)
        assert accuracy["per_layer_accuracy"] > 0.95

    def test_rollout_respects_the_action_mask(self, config, trained):
        model, _data = trained
        agent = SupervisedAgent(model, config.num_devices)
        records = evaluate_agent(config, agent, evaluation_seeds(config, 40))
        assert all(record.memory_violations == 0 for record in records)

    def test_beats_random_placement(self, config, trained):
        model, _data = trained
        seeds = evaluation_seeds(config, 40)
        from src.agents.random_agent import RandomAgent

        supervised = evaluate_agent(config, SupervisedAgent(model, config.num_devices), seeds)
        random_agent = evaluate_agent(config, RandomAgent(config.num_devices, seed=0), seeds)
        assert np.mean([r.objective for r in supervised]) < np.mean(
            [r.objective for r in random_agent]
        )

    def test_masking_overrides_a_confident_but_infeasible_prediction(self, config, trained):
        """Even a certain prediction must yield to the feasibility mask."""
        model, _data = trained
        agent = SupervisedAgent(model, config.num_devices)
        observation = np.zeros(observation_size(config.num_devices), dtype=np.float32)
        mask = np.zeros(config.num_devices, dtype=bool)
        mask[config.num_devices - 1] = True
        chosen = agent.act(observation, {"action_mask": mask})
        assert chosen == config.num_devices - 1

    def test_model_round_trips_through_disk(self, tmp_path, config, trained):
        model, data = trained
        path = save_model(model, tmp_path / "model.joblib")
        reloaded = load_model(path)
        np.testing.assert_array_equal(
            model.predict(data.observations[:20]), reloaded.predict(data.observations[:20])
        )

    def test_missing_model_reports_how_to_train_it(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train_supervised"):
            load_model(tmp_path / "absent.joblib")
