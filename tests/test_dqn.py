"""Tests for the deep Q-network agent.

The important ones here are about masking. A DQN that masks only its behaviour
policy still trains towards values for actions the environment would refuse, and
propagates them backwards through the episode -- a bug that produces a plausible
learning curve and a quietly worse policy, so it is asserted rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.dqn_agent import (
    MASKED_VALUE,
    DQNAgent,
    QNetwork,
    ReplayBuffer,
    Transition,
)
from src.config import load_config
from src.environment.observation import observation_size
from src.environment.scenario import evaluation_seeds
from src.training.evaluate import evaluate_agent
from src.training.train_dqn import train_one_seed


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def agent(config):
    return DQNAgent(observation_size(config.num_devices), config.num_devices, config.dqn, seed=0)


def _plant_final_layer(target, biases: list[float]) -> None:
    """Force a network to output fixed Q-values regardless of its input.

    Accepts either a :class:`DQNAgent` (both networks are planted) or a single
    :class:`QNetwork`, so a test can make the online and target networks disagree.
    """
    networks = (
        [target.online, target.target] if isinstance(target, DQNAgent) else [target]
    )
    with torch.no_grad():
        for network in networks:
            final = network.network[-1]
            final.weight.zero_()
            final.bias.copy_(torch.tensor(biases, dtype=final.bias.dtype))


def make_transition(observation_size_: int, num_actions: int, mask: np.ndarray, done: bool = False):
    return Transition(
        observation=np.zeros(observation_size_, dtype=np.float32),
        action=0,
        reward=-1.0,
        next_observation=np.ones(observation_size_, dtype=np.float32),
        next_action_mask=mask,
        done=done,
    )


class TestNetwork:
    def test_output_width_matches_the_action_space(self, config):
        network = QNetwork(observation_size(config.num_devices), config.num_devices, (16, 16))
        values = network(torch.zeros(5, observation_size(config.num_devices)))
        assert values.shape == (5, config.num_devices)

    def test_hidden_sizes_are_respected(self, config):
        network = QNetwork(8, 3, (7, 5))
        widths = [layer.out_features for layer in network.network if hasattr(layer, "out_features")]
        assert widths == [7, 5, 3]


class TestReplayBuffer:
    def test_grows_then_saturates_at_capacity(self, config):
        buffer = ReplayBuffer(4, 6, config.num_devices, seed=0)
        mask = np.ones(config.num_devices, dtype=bool)
        for _ in range(10):
            buffer.add(make_transition(6, config.num_devices, mask))
        assert len(buffer) == 4

    def test_overwrites_oldest_entries_first(self, config):
        buffer = ReplayBuffer(2, 3, config.num_devices, seed=0)
        mask = np.ones(config.num_devices, dtype=bool)
        for reward in (-1.0, -2.0, -3.0):
            transition = make_transition(3, config.num_devices, mask)
            transition.reward = reward
            buffer.add(transition)
        assert set(np.round(buffer.rewards, 3)) == {-3.0, -2.0}

    def test_sampling_returns_aligned_tensors(self, config):
        buffer = ReplayBuffer(32, 6, config.num_devices, seed=0)
        mask = np.ones(config.num_devices, dtype=bool)
        for _ in range(32):
            buffer.add(make_transition(6, config.num_devices, mask))
        batch = buffer.sample(8)
        assert batch["observations"].shape == (8, 6)
        assert batch["next_action_masks"].shape == (8, config.num_devices)
        assert batch["actions"].dtype == torch.int64

    def test_sampling_more_than_stored_raises(self, config):
        buffer = ReplayBuffer(8, 6, config.num_devices, seed=0)
        with pytest.raises(ValueError, match="cannot sample"):
            buffer.sample(4)


class TestMasking:
    def test_action_selection_never_returns_a_masked_device(self, agent, config):
        observation = np.random.default_rng(0).normal(size=observation_size(config.num_devices))
        for blocked in range(config.num_devices):
            mask = np.ones(config.num_devices, dtype=bool)
            mask[blocked] = False
            chosen = agent.act(observation.astype(np.float32), {"action_mask": mask})
            assert chosen != blocked

    def test_exploration_also_respects_the_mask(self, agent, config):
        agent.epsilon = 1.0  # always explore
        mask = np.zeros(config.num_devices, dtype=bool)
        mask[1] = True
        observation = np.zeros(observation_size(config.num_devices), dtype=np.float32)
        for _ in range(30):
            assert agent.act_exploring(observation, {"action_mask": mask}) == 1

    def test_bootstrapping_ignores_masked_successor_actions(self, config):
        """The subtle half of masking: the target must not use a forbidden action.

        A large value is planted on action 0 of the target network. With action 0
        feasible the target must bootstrap from it; with action 0 masked out the
        target must fall back to the best feasible action instead.
        """
        width = observation_size(config.num_devices)
        agent = DQNAgent(width, config.num_devices, config.dqn, seed=0)
        planted, alternative = 100.0, -50.0
        _plant_final_layer(agent, [planted] + [alternative] * (config.num_devices - 1))

        rewards = torch.tensor([-1.0])
        successors = torch.zeros(1, width)
        dones = torch.tensor([0.0])

        allowed = torch.ones(1, config.num_devices, dtype=torch.bool)
        target_with = agent.compute_targets(rewards, successors, allowed, dones)
        assert target_with.item() == pytest.approx(-1.0 + config.dqn.discount * planted, abs=1e-3)

        blocked = allowed.clone()
        blocked[0, 0] = False
        target_without = agent.compute_targets(rewards, successors, blocked, dones)
        assert target_without.item() == pytest.approx(
            -1.0 + config.dqn.discount * alternative, abs=1e-3
        )

    def test_terminal_transitions_are_not_bootstrapped(self, config):
        width = observation_size(config.num_devices)
        agent = DQNAgent(width, config.num_devices, config.dqn, seed=0)
        _plant_final_layer(agent, [100.0] * config.num_devices)

        targets = agent.compute_targets(
            torch.tensor([-2.0]),
            torch.zeros(1, width),
            torch.ones(1, config.num_devices, dtype=torch.bool),
            torch.tensor([1.0]),
        )
        assert targets.item() == pytest.approx(-2.0, abs=1e-4)

    def test_double_dqn_scores_the_action_the_online_network_chooses(self, config):
        """Double DQN: online picks, target scores. The two must be able to disagree."""
        import dataclasses

        width = observation_size(config.num_devices)
        settings = dataclasses.replace(config.dqn, double_dqn=True)
        agent = DQNAgent(width, config.num_devices, settings, seed=0)
        # The online network prefers action 1; the target network values it at 7.
        _plant_final_layer(agent.online, [0.0, 10.0, 0.0, 0.0][: config.num_devices])
        _plant_final_layer(agent.target, [99.0, 7.0, 0.0, 0.0][: config.num_devices])

        targets = agent.compute_targets(
            torch.tensor([0.0]),
            torch.zeros(1, width),
            torch.ones(1, config.num_devices, dtype=torch.bool),
            torch.tensor([0.0]),
        )
        # A plain max would have taken 99.0 from the target network.
        assert targets.item() == pytest.approx(7.0, abs=1e-3)

    def test_a_successor_with_no_feasible_action_is_treated_as_terminal(self, config):
        """Defensive: a malformed mask must not inject a huge target into the batch."""
        width = observation_size(config.num_devices)
        agent = DQNAgent(width, config.num_devices, config.dqn, seed=0)
        _plant_final_layer(agent, [100.0] * config.num_devices)

        targets = agent.compute_targets(
            torch.tensor([-1.0]),
            torch.zeros(1, width),
            torch.zeros(1, config.num_devices, dtype=torch.bool),
            torch.tensor([0.0]),
        )
        assert targets.item() == pytest.approx(-1.0, abs=1e-4)
        assert MASKED_VALUE < 0

    def test_learning_stays_finite_with_a_degenerate_mask(self, config):
        width = observation_size(config.num_devices)
        agent = DQNAgent(width, config.num_devices, config.dqn, seed=0)
        mask = np.zeros(config.num_devices, dtype=bool)
        for _ in range(config.dqn.learning_starts + config.dqn.batch_size):
            agent.remember(make_transition(width, config.num_devices, mask))
        loss = agent.learn()
        assert loss is not None and np.isfinite(loss)


class TestLearning:
    def test_no_gradient_step_before_the_buffer_fills(self, agent):
        assert agent.learn() is None

    def test_target_network_synchronises_on_schedule(self, config):
        width = observation_size(config.num_devices)
        import dataclasses

        settings = dataclasses.replace(config.dqn, target_update_interval=1, learning_starts=128)
        agent = DQNAgent(width, config.num_devices, settings, seed=0)
        mask = np.ones(config.num_devices, dtype=bool)
        for _ in range(settings.learning_starts + settings.batch_size):
            agent.remember(make_transition(width, config.num_devices, mask))
        agent.learn()
        for online, target in zip(
            agent.online.parameters(), agent.target.parameters(), strict=True
        ):
            assert torch.allclose(online, target)

    def test_exploration_anneals_between_the_configured_bounds(self, agent, config):
        agent.set_exploration(0.0)
        assert agent.epsilon == pytest.approx(config.dqn.epsilon_start)
        agent.set_exploration(1.0)
        assert agent.epsilon == pytest.approx(config.dqn.epsilon_end)

    def test_short_training_run_improves_on_random_placement(self, config):
        """A smoke test on learning itself, not just on the plumbing."""
        import dataclasses
        import tempfile
        from pathlib import Path

        from src.agents.random_agent import RandomAgent

        quick = dataclasses.replace(
            config,
            dqn=dataclasses.replace(
                config.dqn,
                total_steps=6_000,
                eval_interval=3_000,
                eval_episodes=20,
                hidden_sizes=(64, 64),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            agent, curve = train_one_seed(
                quick, 0, checkpoint_path=Path(directory) / "dqn.pt", quiet=True
            )
            seeds = evaluation_seeds(quick, 40)
            learned = evaluate_agent(quick, agent, seeds)
            random_agent = evaluate_agent(quick, RandomAgent(quick.num_devices, seed=0), seeds)
            assert np.mean([r.objective for r in learned]) < np.mean(
                [r.objective for r in random_agent]
            )
            assert len(curve) == 2
            assert all(record.memory_violations == 0 for record in learned)


class TestCheckpoints:
    def test_round_trips_through_disk(self, tmp_path, agent, config):
        observation = np.random.default_rng(1).normal(
            size=observation_size(config.num_devices)
        ).astype(np.float32)
        before = agent.q_values(observation)
        path = agent.save(tmp_path / "dqn.pt")
        reloaded = DQNAgent.load(path, config.dqn)
        np.testing.assert_allclose(before, reloaded.q_values(observation), rtol=1e-6)
        assert reloaded.epsilon == 0.0

    def test_missing_checkpoint_reports_how_to_train_it(self, tmp_path, config):
        with pytest.raises(FileNotFoundError, match="train_dqn"):
            DQNAgent.load(tmp_path / "absent.pt", config.dqn)

    def test_checkpoint_carries_its_own_architecture(self, tmp_path, config):
        import dataclasses

        settings = dataclasses.replace(config.dqn, hidden_sizes=(32,))
        agent = DQNAgent(observation_size(config.num_devices), config.num_devices, settings, seed=0)
        path = agent.save(tmp_path / "narrow.pt")
        # Loading with the default (128, 128) settings must still rebuild (32,).
        reloaded = DQNAgent.load(path, config.dqn)
        assert reloaded.settings.hidden_sizes == (32,)
