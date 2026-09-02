"""Train the tabular Q-learning agent, in both of its modes.

``single_scenario``
    One fixed problem. The discrete abstraction discards only quantities that
    are constant within that problem, so the table can represent the optimal
    policy exactly. The run reports how close it gets to the dynamic-programming
    solution for that scenario, and then evaluates the same table on held-out
    scenarios to show what a memorised policy is worth on problems it has never
    seen.

``pooled``
    The same table trained across randomly sampled scenarios. Here the
    abstraction is aliased: device speeds, energy rates and link characteristics
    all differ between scenarios and none of them are part of the state, so a
    single table entry has to average over problems that call for different
    answers.

The comparison is the argument for function approximation, made with measurements
rather than assertion.

Run it with::

    python -m src.training.train_q_learning
    python -m src.training.train_q_learning --set q_learning.mode=pooled
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.agents.greedy_agent import GreedyAgent
from src.agents.q_learning_agent import (
    DEFAULT_TABLE_PATH,
    TabularQAgent,
    hyper_parameters_from_config,
)
from src.baselines import dp_optimal
from src.config import Config, config_summary, load_config
from src.environment.dnn_environment import DNNPlacementEnv
from src.environment.scenario import evaluation_seeds, sample_scenario
from src.training.evaluate import evaluate_agent
from src.utils import visualization
from src.utils.io import save_processed, save_raw


def train(
    config: Config,
    agent: TabularQAgent,
    *,
    episodes: int,
    fixed_seed: int | None,
    log_every: int = 1_000,
) -> pd.DataFrame:
    """Run Q-learning and return its training curve.

    Args:
        config: The configuration in force.
        agent: The agent to train, modified in place.
        episodes: Number of episodes to run.
        fixed_seed: Scenario seed to train on repeatedly, or ``None`` to sample
            a fresh training scenario each episode.
        log_every: How often to record a row of the training curve.

    Returns:
        A frame with one row per logged episode.
    """
    env = DNNPlacementEnv(
        config,
        scenario_seeds=[fixed_seed] if fixed_seed is not None else None,
        seed=config.seed,
    )

    returns: list[float] = []
    rows: list[dict[str, float]] = []
    started = time.perf_counter()

    for episode in range(episodes):
        agent.set_exploration(episode / max(episodes - 1, 1))
        _observation, info = env.reset()
        episode_return = 0.0
        terminated = False

        while not terminated:
            key = agent.state_key(info)
            action = agent.act_exploring(info)
            _observation, reward, terminated, _truncated, info = env.step(action)
            next_key = None if terminated else agent.state_key(info)
            next_mask = None if terminated else info["action_mask"]
            agent.update(key, action, reward, next_key, next_mask)
            episode_return += reward

        returns.append(episode_return)
        if (episode + 1) % log_every == 0 or episode == episodes - 1:
            window = returns[-log_every:]
            rows.append(
                {
                    "episode": episode + 1,
                    "episode_return": returns[-1],
                    "moving_average_return": float(np.mean(window)),
                    "epsilon": agent.epsilon,
                    "states_visited": agent.states_visited,
                    "elapsed_s": time.perf_counter() - started,
                }
            )

    return pd.DataFrame(rows)


def report_single_scenario(config: Config, agent: TabularQAgent, seed: int) -> dict[str, float]:
    """Compare the learned policy against the exact solution for its own problem."""
    env = DNNPlacementEnv(config, scenario_seeds=[seed])
    scenario = sample_scenario(config, seed)

    from src.training.evaluate import run_episode

    learned = run_episode(env, agent, scenario=scenario)
    greedy = run_episode(env, GreedyAgent(config.num_devices, "objective_aware"), scenario=scenario)
    dp_solution = dp_optimal.solve(scenario, config)
    from src.environment.reward import evaluate_placement

    dp_actual = evaluate_placement(scenario, dp_solution.placement, config)

    return {
        "scenario_seed": float(seed),
        "tabular_objective": learned.objective,
        "greedy_objective": greedy.objective,
        "dp_objective": dp_actual.objective,
        "dp_lower_bound": dp_solution.objective,
        "gap_vs_dp_placement_pct": (learned.objective - dp_actual.objective)
        / dp_actual.objective
        * 100.0,
    }


def main() -> int:
    """Train the tabular agent and measure how far it generalises."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="static_environment.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="agent seed")
    parser.add_argument(
        "--train-scenario", type=int, default=None, help="fixed scenario seed for single mode"
    )
    parser.add_argument("--output", default=str(DEFAULT_TABLE_PATH))
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config, arguments.overrides)
    episodes = arguments.episodes or config.q_learning.episodes
    agent_seed = arguments.seed if arguments.seed is not None else config.q_learning.seeds[0]
    mode = config.q_learning.mode

    # The training scenario is drawn from the training pool, never the held-out one.
    fixed_seed = (
        (arguments.train_scenario or config.experiment.train_seed_start)
        if mode == "single_scenario"
        else None
    )

    print("=" * 78)
    print("TRAINING TABULAR Q-LEARNING")
    print("=" * 78)
    print(config_summary(config))
    print(f"mode: {mode}, episodes: {episodes:,}, agent seed: {agent_seed}")
    if fixed_seed is not None:
        print(f"training scenario seed: {fixed_seed}")

    agent = TabularQAgent(config.num_devices, hyper_parameters_from_config(config), seed=agent_seed)
    curve = train(config, agent, episodes=episodes, fixed_seed=fixed_seed)

    final = curve.iloc[-1]
    print(
        f"\ntrained in {final['elapsed_s']:.1f}s, "
        f"{int(final['states_visited']):,} distinct states visited"
    )
    print(f"final moving-average return: {final['moving_average_return']:.4f}")

    curve["mode"] = mode
    curve["agent_seed"] = agent_seed
    save_raw(curve, f"e1_tabular_curve_{mode}", config, mode=mode, agent_seed=agent_seed)

    reference = None
    reference_label = ""
    if fixed_seed is not None:
        scenario = sample_scenario(config, fixed_seed)
        from src.environment.reward import evaluate_placement

        dp_placement = dp_optimal.solve(scenario, config).placement
        reference = -evaluate_placement(scenario, dp_placement, config).objective
        reference_label = "return of the dynamic-programming placement"
    figure = visualization.training_curve(
        curve,
        f"fig11_tabular_training_{mode}",
        title=f"Tabular Q-learning: {mode.replace('_', ' ')}",
        subtitle="Return equals the negative weighted objective, so higher is better",
        reference_value=reference,
        reference_label=reference_label,
    )
    print(f"training curve written to {figure}")

    if fixed_seed is not None:
        comparison = report_single_scenario(config, agent, fixed_seed)
        print("\nOn the single scenario it was trained on")
        print(f"  tabular Q-learning        {comparison['tabular_objective']:.4f}")
        print(f"  objective-aware greedy    {comparison['greedy_objective']:.4f}")
        print(f"  DP placement              {comparison['dp_objective']:.4f}")
        print(f"  gap vs DP placement       {comparison['gap_vs_dp_placement_pct']:+.2f}%")
        save_processed(pd.DataFrame([comparison]), f"e1_tabular_single_scenario_{mode}")

    held_out = evaluation_seeds(config, min(config.experiment.n_eval_scenarios, 150))
    records = evaluate_agent(config, agent, held_out)
    greedy_records = evaluate_agent(
        config, GreedyAgent(config.num_devices, "objective_aware"), held_out
    )
    print(f"\nOn {len(held_out)} held-out scenarios it has never seen")
    print(f"  tabular Q-learning        {np.mean([r.objective for r in records]):.4f}")
    print(f"  objective-aware greedy    {np.mean([r.objective for r in greedy_records]):.4f}")
    print(
        f"  states seen at evaluation that were never visited in training: "
        f"{_unseen_state_fraction(config, agent, held_out):.1%}"
    )

    path = agent.save(__import__("pathlib").Path(arguments.output))
    print(f"\ntable written to {path}")
    return 0


def _unseen_state_fraction(config: Config, agent: TabularQAgent, seeds: list[int]) -> float:
    """Fraction of evaluation decisions taken in states the table never learned.

    A high value means the agent is acting on an all-zero row -- that is, guessing.
    """
    env = DNNPlacementEnv(config)
    unseen = 0
    total = 0
    known = set(agent.table)
    for seed in seeds:
        _observation, info = env.reset(options={"scenario_seed": seed})
        terminated = False
        while not terminated:
            total += 1
            if agent.state_key(info) not in known:
                unseen += 1
            action = agent.act(_observation, info)
            _observation, _reward, terminated, _truncated, info = env.step(action)
    return unseen / max(total, 1)


if __name__ == "__main__":
    raise SystemExit(main())
