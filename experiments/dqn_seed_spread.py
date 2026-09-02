"""Report the spread across DQN training seeds.

A single reinforcement learning run is weak evidence: two seeds of the same
algorithm on the same problem can differ by more than two different algorithms
do. The experiments load one checkpoint -- the seed that validated best -- so this
script exists to show what that selection hides, by evaluating *every* trained
seed on the held-out scenarios.

Run it with::

    python -m experiments.dqn_seed_spread
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.dqn_agent import DQNAgent
from src.agents.greedy_agent import GreedyAgent
from src.config import load_config
from src.environment.scenario import evaluation_seeds
from src.training.evaluate import evaluate_agent
from src.utils.io import save_processed, save_raw
from src.utils.metrics import records_to_frame
from src.utils.stats import mean_interval, paired_comparison


def main() -> int:
    """Evaluate every DQN seed checkpoint on the held-out scenarios."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="static_environment.yaml")
    parser.add_argument("--scenarios", type=int, default=None)
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config, arguments.overrides)
    seeds = evaluation_seeds(config, arguments.scenarios)
    paths = sorted(Path(arguments.checkpoints).glob("dqn_seed*.pt"))
    if not paths:
        print(
            f"no per-seed checkpoints in {arguments.checkpoints}; "
            "train them with `python -m src.training.train_dqn`"
        )
        return 1

    print("=" * 78)
    print("DQN SEED SPREAD")
    print("=" * 78)
    print(f"{len(paths)} checkpoints, {len(seeds)} held-out scenarios")

    greedy_records = evaluate_agent(
        config, GreedyAgent(config.num_devices, "objective_aware"), seeds
    )
    greedy_objectives = [record.objective for record in greedy_records]
    greedy_mean = float(np.mean(greedy_objectives))

    all_records = []
    rows = []
    for path in paths:
        seed_label = path.stem.replace("dqn_seed", "")
        agent = DQNAgent.load(path, config.dqn, name=f"dqn_seed{seed_label}")
        records = evaluate_agent(config, agent, seeds)
        all_records.extend(records)
        objectives = [record.objective for record in records]
        interval = mean_interval(objectives)
        comparison = paired_comparison(
            objectives, greedy_objectives, f"dqn_seed{seed_label}", "greedy_objective_aware"
        )
        rows.append(
            {
                "agent_seed": int(seed_label),
                "objective_mean": interval.estimate,
                "objective_ci_low": interval.low,
                "objective_ci_high": interval.high,
                "relative_vs_greedy_pct": comparison.relative_difference_pct,
                "win_rate_vs_greedy": comparison.win_rate,
                "p_value_vs_greedy": comparison.p_value,
                "memory_violations": sum(record.memory_violations for record in records),
            }
        )
        print(
            f"  seed {seed_label}: objective {interval.format(4)}  "
            f"{comparison.relative_difference_pct:+.2f}% vs greedy  "
            f"wins {comparison.win_rate:.0%}"
        )

    frame = pd.DataFrame(rows)
    spread = frame["objective_mean"].max() - frame["objective_mean"].min()
    print(f"\nobjective-aware greedy: {greedy_mean:.4f}")
    print(
        f"spread across seeds: {spread:.4f} "
        f"({spread / greedy_mean * 100:.2f}% of the greedy objective)"
    )

    save_raw(records_to_frame(all_records), "e1_dqn_seed_spread", config)
    save_processed(frame, "e1_dqn_seed_spread_summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
