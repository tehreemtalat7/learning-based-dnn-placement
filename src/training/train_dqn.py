"""Train the deep Q-network.

One run trains the agent for ``dqn.total_steps`` environment steps on scenarios
drawn from the training pool, checking progress periodically on the *validation*
pool. The checkpoint kept is the one with the best validation return, never the
last one and never the best on the test scenarios -- no decision about a model is
made using data its results are reported on.

Everything is repeated for each seed in ``dqn.seeds``, because a single
reinforcement learning run says very little: seed variance in this setting is
comparable to the differences between methods, and reporting one run would
invite reading noise as a result.

Run it with::

    python -m src.training.train_dqn
    python -m src.training.train_dqn --seeds 0 --set dqn.total_steps=20000   # quick
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.dqn_agent import DEFAULT_CHECKPOINT_PATH, DQNAgent, Transition
from src.agents.greedy_agent import GreedyAgent
from src.config import Config, config_summary, load_config
from src.environment.dnn_environment import DNNPlacementEnv
from src.environment.observation import observation_size
from src.environment.scenario import validation_seeds
from src.training.evaluate import evaluate_agent
from src.utils import visualization
from src.utils.io import save_processed, save_raw
from src.utils.seed import seed_everything

MOVING_AVERAGE_WINDOW = 100


def evaluate_on(config: Config, agent: DQNAgent, seeds: list[int]) -> float:
    """Mean greedy return of the current policy over a fixed set of scenarios."""
    records = evaluate_agent(config, agent, seeds)
    return float(np.mean([record.episode_return for record in records]))


def train_one_seed(
    config: Config,
    seed: int,
    *,
    checkpoint_path: Path,
    quiet: bool = False,
) -> tuple[DQNAgent, pd.DataFrame]:
    """Train a single agent and return it alongside its training curve.

    Args:
        config: The configuration in force.
        seed: Seed for parameters, exploration, replay and scenario sampling.
        checkpoint_path: Where to write the best-validation checkpoint.
        quiet: Suppress per-evaluation logging.

    Returns:
        ``(agent, curve)`` where the agent holds the best-validation weights.
    """
    settings = config.dqn
    seed_everything(seed)

    env = DNNPlacementEnv(config, seed=seed)
    agent = DQNAgent(
        observation_size(config.num_devices), config.num_devices, settings, seed=seed
    )
    validation = validation_seeds(config, min(config.experiment.n_valid_scenarios,
                                              settings.eval_episodes))

    rows: list[dict[str, float]] = []
    returns: list[float] = []
    losses: list[float] = []
    best_validation = -np.inf
    started = time.perf_counter()

    observation, info = env.reset()
    episode_return = 0.0

    for step in range(1, settings.total_steps + 1):
        agent.set_exploration(step / settings.total_steps)
        action = agent.act_exploring(observation, info)
        next_observation, reward, terminated, _truncated, next_info = env.step(action)

        agent.remember(
            Transition(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                next_action_mask=np.asarray(next_info["action_mask"], dtype=bool),
                done=terminated,
            )
        )
        episode_return += reward

        if step % settings.train_frequency == 0:
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

        if terminated:
            returns.append(episode_return)
            episode_return = 0.0
            observation, info = env.reset()
        else:
            observation, info = next_observation, next_info

        if step % settings.eval_interval == 0 or step == settings.total_steps:
            validation_return = evaluate_on(config, agent, validation)
            window = returns[-MOVING_AVERAGE_WINDOW:] or [float("nan")]
            rows.append(
                {
                    "step": step,
                    "episodes": len(returns),
                    "episode_return": returns[-1] if returns else float("nan"),
                    "moving_average_return": float(np.mean(window)),
                    "validation_return": validation_return,
                    "epsilon": agent.epsilon,
                    "loss": float(np.mean(losses[-1000:])) if losses else float("nan"),
                    "elapsed_s": time.perf_counter() - started,
                }
            )
            if validation_return > best_validation:
                best_validation = validation_return
                agent.save(checkpoint_path)
            if not quiet:
                print(
                    f"  step {step:>7,}  moving avg {rows[-1]['moving_average_return']:+.4f}  "
                    f"validation {validation_return:+.4f}  eps {agent.epsilon:.3f}  "
                    f"loss {rows[-1]['loss']:.5f}"
                )

    # Return the best checkpoint rather than the final weights.
    best_agent = DQNAgent.load(checkpoint_path, settings)
    curve = pd.DataFrame(rows)
    curve["agent_seed"] = seed
    curve["best_validation_return"] = best_validation
    return best_agent, curve


def main() -> int:
    """Train one agent per configured seed and report the outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="static_environment.yaml")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--output", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--tag", default="static", help="suffix for result files")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config, arguments.overrides)
    seeds = arguments.seeds if arguments.seeds else list(config.dqn.seeds)
    output = Path(arguments.output)

    print("=" * 78)
    print("TRAINING THE DEEP Q-NETWORK")
    print("=" * 78)
    print(config_summary(config))
    print(
        f"steps per seed: {config.dqn.total_steps:,}, seeds: {seeds}, "
        f"double DQN: {config.dqn.double_dqn}, discount: {config.dqn.discount}"
    )

    curves = []
    summaries = []
    for seed in seeds:
        print(f"\nseed {seed}")
        path = output.with_name(f"{output.stem}_seed{seed}{output.suffix}")
        agent, curve = train_one_seed(config, seed, checkpoint_path=path)
        curves.append(curve)
        best = float(curve["best_validation_return"].iloc[0])
        summaries.append(
            {
                "agent_seed": seed,
                "best_validation_return": best,
                "final_moving_average": float(curve["moving_average_return"].iloc[-1]),
                "training_seconds": float(curve["elapsed_s"].iloc[-1]),
                "checkpoint": str(path),
            }
        )
        print(f"  best validation return {best:+.4f}, checkpoint {path}")

    curve_frame = pd.concat(curves, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    save_raw(curve_frame, f"e1_dqn_curves_{arguments.tag}", config, seeds=seeds)
    save_processed(summary_frame, f"e1_dqn_training_summary_{arguments.tag}")

    # Keep the seed that validated best as the checkpoint the experiments load.
    best_row = summary_frame.loc[summary_frame["best_validation_return"].idxmax()]
    import shutil

    shutil.copyfile(best_row["checkpoint"], output)
    print(f"\nbest seed: {int(best_row['agent_seed'])} -> {output}")

    greedy_reference = -np.mean(
        [
            record.objective
            for record in evaluate_agent(
                config,
                GreedyAgent(config.num_devices, "objective_aware"),
                validation_seeds(config, config.dqn.eval_episodes),
            )
        ]
    )
    figures = [
        visualization.training_curve(
            curve_frame,
            f"fig01_dqn_training_return_{arguments.tag}",
            title="DQN training return",
            subtitle="Return equals the negative weighted objective, so higher is better. "
            "Faint lines are single episodes",
            x_column="step",
            value_column="episode_return",
            smoothed_column="moving_average_return",
            reference_value=float(greedy_reference),
            reference_label="objective-aware greedy",
            x_label="Environment steps",
        ),
        visualization.training_curve(
            curve_frame,
            f"fig02_dqn_validation_return_{arguments.tag}",
            title="DQN return on held-out validation scenarios",
            subtitle="Measured during training on scenarios never used for gradient updates",
            x_column="step",
            value_column="validation_return",
            smoothed_column="validation_return",
            reference_value=float(greedy_reference),
            reference_label="objective-aware greedy",
            x_label="Environment steps",
            y_label="Validation return",
        ),
    ]
    print("\nFigures written:")
    for path in figures:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
