"""Tests for the tabular Q-learning agent and its abstraction."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.q_learning_agent import (
    QLearningHyperParameters,
    TabularQAgent,
    hyper_parameters_from_config,
)
from src.baselines import dp_optimal
from src.config import load_config
from src.environment.dnn_environment import DNNPlacementEnv
from src.environment.reward import evaluate_placement
from src.environment.scenario import evaluation_seeds, sample_scenario
from src.training.evaluate import evaluate_agent, run_episode
from src.training.train_q_learning import train


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def agent(config):
    return TabularQAgent(config.num_devices, hyper_parameters_from_config(config), seed=0)


class TestAbstraction:
    def test_state_key_is_hashable_and_reproducible(self, agent):
        info = {
            "layer_index": 3,
            "previous_device": 2,
            "free_memory_ratio": np.array([0.1, 0.5, 0.9, 1.0]),
        }
        first = agent.state_key(info)
        assert first == agent.state_key(info)
        assert isinstance(hash(first), int)
        assert first[0] == 3 and first[1] == 2

    def test_memory_is_bucketed_not_stored_continuously(self, config):
        parameters = QLearningHyperParameters(0.1, 0.99, 1.0, 0.02, 0.6, memory_buckets=4)
        agent = TabularQAgent(config.num_devices, parameters, seed=0)
        base = {"layer_index": 0, "previous_device": 0}
        near = agent.state_key({**base, "free_memory_ratio": np.array([0.80, 1.0, 1.0, 1.0])})
        also_near = agent.state_key({**base, "free_memory_ratio": np.array([0.99, 1.0, 1.0, 1.0])})
        far = agent.state_key({**base, "free_memory_ratio": np.array([0.10, 1.0, 1.0, 1.0])})
        assert near == also_near  # the whole point of a bucket
        assert near != far

    def test_full_memory_does_not_overflow_the_top_bucket(self, agent):
        key = agent.state_key(
            {"layer_index": 0, "previous_device": 0, "free_memory_ratio": np.ones(4)}
        )
        buckets = agent.hyper_parameters.memory_buckets
        assert all(0 <= value < buckets for value in key[2])

    def test_the_abstraction_discards_device_characteristics(self, agent, config):
        """Two scenarios with very different devices can share a state key.

        This is the aliasing the training script measures, stated as a test so
        that the limitation is documented in executable form.
        """
        environment = DNNPlacementEnv(config)
        keys = set()
        for seed in evaluation_seeds(config, 25):
            _observation, info = environment.reset(options={"scenario_seed": seed})
            keys.add(agent.state_key(info))
        # Every scenario starts at layer 0 on the input device with empty memory.
        assert len(keys) == 1


class TestActingAndLearning:
    def test_acting_respects_the_mask(self, agent, config):
        mask = np.zeros(config.num_devices, dtype=bool)
        mask[2] = True
        info = {
            "layer_index": 0,
            "previous_device": 0,
            "action_mask": mask,
            "free_memory_ratio": np.ones(config.num_devices),
        }
        assert agent.act(np.zeros(1), info) == 2

    def test_update_moves_the_value_towards_the_target(self, agent):
        key = (0, 0, (3, 3, 3, 3))
        error = agent.update(key, action=1, reward=-1.0, next_key=None, next_mask=None)
        assert error == pytest.approx(-1.0)
        assert agent.table[key][1] == pytest.approx(-0.1)  # learning rate 0.1

    def test_bootstrapping_ignores_infeasible_successors(self, agent):
        key, next_key = (0, 0, (3, 3, 3, 3)), (1, 0, (3, 3, 3, 3))
        agent.table[next_key] = np.array([100.0, -5.0, -5.0, -5.0])
        mask = np.array([False, True, True, True])
        agent.update(key, action=0, reward=0.0, next_key=next_key, next_mask=mask)
        # The 100.0 sits on a masked action and must not be bootstrapped from.
        assert agent.table[key][0] < 0.0

    def test_exploration_anneals_from_start_to_end(self, agent):
        agent.set_exploration(0.0)
        assert agent.epsilon == pytest.approx(agent.hyper_parameters.epsilon_start)
        agent.set_exploration(1.0)
        assert agent.epsilon == pytest.approx(agent.hyper_parameters.epsilon_end)

    def test_greedy_action_follows_the_learned_values(self, agent, config):
        info = {
            "layer_index": 0,
            "previous_device": 0,
            "action_mask": np.ones(config.num_devices, dtype=bool),
            "free_memory_ratio": np.ones(config.num_devices),
        }
        key = agent.state_key(info)
        agent.table[key] = np.array([-5.0, -5.0, -1.0, -5.0])
        assert agent.act(np.zeros(1), info) == 2


class TestTraining:
    def test_single_scenario_training_approaches_the_exact_solution(self, config):
        """On one fixed problem the abstraction is lossless, so the table should converge."""
        seed = config.experiment.train_seed_start
        agent = TabularQAgent(config.num_devices, hyper_parameters_from_config(config), seed=0)
        train(config, agent, episodes=3_000, fixed_seed=seed, log_every=1_000)

        scenario = sample_scenario(config, seed)
        environment = DNNPlacementEnv(config, scenario_seeds=[seed])
        learned = run_episode(environment, agent, scenario=scenario)
        dp_placement = dp_optimal.solve(scenario, config).placement
        dp_objective = evaluate_placement(scenario, dp_placement, config).objective
        assert learned.objective <= dp_objective * 1.05

    def test_training_produces_a_curve_with_the_expected_columns(self, config):
        agent = TabularQAgent(config.num_devices, hyper_parameters_from_config(config), seed=0)
        curve = train(config, agent, episodes=200, fixed_seed=None, log_every=50)
        assert list(curve.columns) == [
            "episode",
            "episode_return",
            "moving_average_return",
            "epsilon",
            "states_visited",
            "elapsed_s",
        ]
        assert len(curve) == 4
        assert curve["epsilon"].is_monotonic_decreasing

    def test_a_trained_agent_never_violates_memory(self, config):
        agent = TabularQAgent(config.num_devices, hyper_parameters_from_config(config), seed=0)
        train(config, agent, episodes=300, fixed_seed=None, log_every=300)
        records = evaluate_agent(config, agent, evaluation_seeds(config, 30))
        assert all(record.memory_violations == 0 for record in records)

    def test_table_round_trips_through_disk(self, tmp_path, config):
        agent = TabularQAgent(config.num_devices, hyper_parameters_from_config(config), seed=0)
        train(config, agent, episodes=200, fixed_seed=None, log_every=200)
        path = agent.save(tmp_path / "table.joblib")
        reloaded = TabularQAgent.load(path)
        assert reloaded.states_visited == agent.states_visited
        for key, values in agent.table.items():
            np.testing.assert_allclose(reloaded.table[key], values)

    def test_missing_table_reports_how_to_train_it(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train_q_learning"):
            TabularQAgent.load(tmp_path / "absent.joblib")
