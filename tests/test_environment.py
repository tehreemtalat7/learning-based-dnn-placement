"""Tests for the Gymnasium placement environment."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.round_robin_agent import RoundRobinAgent
from src.config import load_config
from src.environment.dnn_environment import DNNPlacementEnv, InvalidActionError
from src.environment.observation import feature_names, observation_size
from src.environment.reward import evaluate_placement
from src.environment.scenario import evaluation_seeds, sample_scenario
from src.training.evaluate import evaluate_agent, run_episode

TOLERANCE = 1e-6


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def env(config):
    return DNNPlacementEnv(config)


class TestApi:
    def test_spaces_match_the_configuration(self, config, env):
        assert env.action_space.n == config.num_devices
        assert env.observation_space.shape == (observation_size(config.num_devices),)

    def test_reset_returns_a_finite_observation_and_a_mask(self, env, config):
        observation, info = env.reset(options={"scenario_seed": 9_000_000})
        assert observation.shape == env.observation_space.shape
        assert observation.dtype == np.float32
        assert np.isfinite(observation).all()
        assert info["action_mask"].shape == (config.num_devices,)
        assert info["action_mask"].any()

    def test_feature_names_cover_the_observation(self, config):
        names = feature_names(config.num_devices, config.device_names)
        assert len(names) == observation_size(config.num_devices)
        assert len(set(names)) == len(names)

    def test_episode_length_equals_the_number_of_layers(self, env, config):
        _observation, info = env.reset(options={"scenario_seed": 9_000_001})
        steps = 0
        terminated = False
        while not terminated:
            action = int(np.flatnonzero(info["action_mask"])[0])
            _observation, _reward, terminated, truncated, info = env.step(action)
            assert truncated is False
            steps += 1
        assert steps == config.workload.num_layers

    def test_stepping_after_termination_raises(self, env):
        _observation, info = env.reset(options={"scenario_seed": 9_000_002})
        terminated = False
        while not terminated:
            action = int(np.flatnonzero(info["action_mask"])[0])
            _observation, _reward, terminated, _truncated, info = env.step(action)
        with pytest.raises(RuntimeError, match="call reset"):
            env.step(0)

    def test_out_of_range_action_raises(self, env, config):
        env.reset(options={"scenario_seed": 9_000_003})
        with pytest.raises(InvalidActionError, match="outside"):
            env.step(config.num_devices)

    def test_depth_override_changes_the_episode_length(self, env):
        _observation, info = env.reset(options={"scenario_seed": 9_000_004, "num_layers": 5})
        assert info["num_layers"] == 5


class TestRewardSemantics:
    @pytest.mark.parametrize("seed", [9_000_010, 9_000_011, 9_000_012])
    def test_return_equals_negative_objective(self, config, env, seed):
        agent = GreedyAgent(config.num_devices, "objective_aware")
        record = run_episode(env, agent, scenario_seed=seed)
        assert record.episode_return == pytest.approx(-record.objective, abs=TOLERANCE)

    @pytest.mark.parametrize("seed", [9_000_020, 9_000_021])
    def test_environment_agrees_with_the_offline_evaluator(self, config, env, seed):
        agent = RandomAgent(config.num_devices, seed=1)
        record = run_episode(env, agent, scenario_seed=seed)
        scenario = sample_scenario(config, seed)
        offline = evaluate_placement(scenario, record.placement, config)
        assert record.total_latency_ms == pytest.approx(offline.total_latency_ms, abs=TOLERANCE)
        assert record.energy == pytest.approx(offline.energy, abs=TOLERANCE)
        assert record.objective == pytest.approx(offline.objective, abs=TOLERANCE)

    def test_random_placement_scores_near_one(self, config):
        """The references are calibrated so that random placement costs about 1."""
        seeds = evaluation_seeds(config, 100)
        records = evaluate_agent(config, RandomAgent(config.num_devices, seed=0), seeds)
        mean_objective = float(np.mean([record.objective for record in records]))
        assert 0.5 <= mean_objective <= 1.8

    def test_objective_aware_greedy_beats_the_weaker_heuristics(self, config):
        seeds = evaluation_seeds(config, 60)
        scores = {}
        for agent in (
            RandomAgent(config.num_devices, seed=0),
            RoundRobinAgent(config.num_devices),
            GreedyAgent(config.num_devices, "fastest_device"),
            GreedyAgent(config.num_devices, "objective_aware"),
        ):
            records = evaluate_agent(config, agent, seeds)
            scores[agent.name] = float(np.mean([record.objective for record in records]))
        assert scores["greedy_objective_aware"] < scores["random"]
        assert scores["greedy_objective_aware"] < scores["round_robin"]


class TestActionMasking:
    def test_mask_matches_free_memory(self, config, env):
        _observation, info = env.reset(options={"scenario_seed": 9_000_030})
        scenario = env.scenario
        assert scenario is not None
        layer = scenario.workload[0]
        expected = [device.memory_gb >= layer.memory_gb for device in scenario.devices]
        np.testing.assert_array_equal(info["action_mask"], expected)

    def test_masking_forbids_infeasible_devices(self, config):
        """Find a masked device and confirm that selecting it raises."""
        env = DNNPlacementEnv(config)
        for seed in evaluation_seeds(config, 200):
            _observation, info = env.reset(options={"scenario_seed": seed})
            terminated = False
            while not terminated:
                mask = info["action_mask"]
                if not mask.all():
                    blocked = int(np.flatnonzero(~mask)[0])
                    with pytest.raises(InvalidActionError, match="cannot host"):
                        env.step(blocked)
                    return
                action = int(np.flatnonzero(mask)[0])
                _observation, _reward, terminated, _truncated, info = env.step(action)
        pytest.skip("no infeasible device occurred in the sampled scenarios")

    def test_masked_agents_never_violate_memory(self, config):
        seeds = evaluation_seeds(config, 100)
        for agent in (
            RandomAgent(config.num_devices, seed=3),
            RoundRobinAgent(config.num_devices),
            GreedyAgent(config.num_devices, "fastest_device"),
        ):
            records = evaluate_agent(config, agent, seeds)
            assert all(record.memory_violations == 0 for record in records)

    def test_penalty_mode_allows_and_charges_infeasible_choices(self):
        config = load_config(
            overrides=[
                "environment.invalid_action_mode=penalty",
                "environment.invalid_action_penalty=1.0",
            ]
        )
        env = DNNPlacementEnv(config)
        for seed in evaluation_seeds(config, 200):
            observation, info = env.reset(options={"scenario_seed": seed})
            terminated = False
            while not terminated:
                mask = info["action_mask"]
                if not mask.all():
                    blocked = int(np.flatnonzero(~mask)[0])
                    attempts_before = env.invalid_action_attempts
                    _observation, reward, _terminated, _truncated, info = env.step(blocked)
                    assert info["step_penalty"] == pytest.approx(1.0)
                    assert reward < -1.0
                    assert env.invalid_action_attempts == attempts_before + 1
                    return
                action = int(np.flatnonzero(mask)[0])
                observation, _reward, terminated, _truncated, info = env.step(action)
        pytest.skip("no infeasible device occurred in the sampled scenarios")


class TestDynamics:
    def test_memory_accumulation_can_close_devices_during_an_episode(self, config):
        """Placing several large layers on one device must eventually exclude it."""
        env = DNNPlacementEnv(config)
        for seed in evaluation_seeds(config, 300):
            _observation, info = env.reset(options={"scenario_seed": seed})
            initial_mask = info["action_mask"].copy()
            terminated = False
            saw_closure = False
            while not terminated:
                mask = info["action_mask"]
                if (initial_mask & ~mask).any():
                    saw_closure = True
                action = int(np.flatnonzero(mask)[-1])  # keep loading the last device
                _observation, _reward, terminated, _truncated, info = env.step(action)
            if saw_closure:
                return
        pytest.skip("no device filled up in the sampled scenarios")

    def test_utilisation_accumulation_slows_a_loaded_device(self, config):
        env = DNNPlacementEnv(config)
        _observation, info = env.reset(options={"scenario_seed": 9_000_040})
        first = info["candidate_execution_ms"][0]
        env.step(0)
        _observation, _reward, _terminated, _truncated, info = env.step(0)
        assert info["candidate_execution_ms"][0] > first

    def test_congestion_raises_communication_cost(self):
        calm = load_config(overrides=["network.profile=normal"])
        rough = load_config(overrides=["network.profile=congested"])
        seeds = evaluation_seeds(calm, 40)
        agent_calm = RoundRobinAgent(calm.num_devices)
        agent_rough = RoundRobinAgent(rough.num_devices)
        calm_records = evaluate_agent(calm, agent_calm, seeds)
        rough_records = evaluate_agent(rough, agent_rough, seeds)
        calm_comm = np.mean([record.communication_latency_ms for record in calm_records])
        rough_comm = np.mean([record.communication_latency_ms for record in rough_records])
        assert rough_comm > calm_comm


class TestDeterminism:
    def test_same_seed_reproduces_the_same_episode(self, config, env):
        agent = GreedyAgent(config.num_devices, "objective_aware")
        first = run_episode(env, agent, scenario_seed=9_000_050)
        second = run_episode(env, agent, scenario_seed=9_000_050)
        assert first.placement == second.placement
        assert first.objective == pytest.approx(second.objective)

    def test_scenario_seeds_are_independent_of_agent_seeds(self, config):
        env = DNNPlacementEnv(config)
        one = run_episode(env, RandomAgent(config.num_devices, seed=0), scenario_seed=9_000_051)
        two = run_episode(env, RandomAgent(config.num_devices, seed=99), scenario_seed=9_000_051)
        assert one.num_layers == two.num_layers
        assert one.scenario_seed == two.scenario_seed
