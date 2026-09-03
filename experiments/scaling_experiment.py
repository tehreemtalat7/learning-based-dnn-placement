"""Experiment 2: how methods behave as the DNN gets deeper.

Depth is the axis on which the previous project could say least: exhaustive
search costs ``D**L``, so optimality gaps stopped at five layers. Here the
dynamic programme supplies a reference at every depth, and the fixed-width state
vector lets one policy place networks of any size.

Two learned policies are compared, which is the point of the experiment:

* ``dqn`` was trained only on 10-layer DNNs. Evaluating it at 5, 20 and 30
  layers measures **transfer** to depths it never saw.
* ``dqn_mixed`` was trained on a mixture of depths. Comparing the two separates
  "the state representation generalises" from "the policy happened to be trained
  on the right size".

Run it with::

    python -m experiments.scaling_experiment
    python -m experiments.scaling_experiment --depths 5 10 --scenarios 50
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from experiments._shared import CHECKPOINTS, compare_methods, optional_agents
from src.baselines import exhaustive_search
from src.config import config_summary, load_config
from src.environment.scenario import evaluation_seeds
from src.utils import visualization
from src.utils.io import save_processed, save_raw
from src.utils.metrics import format_summary_table, summarise_by_method

DEFAULT_DEPTHS = (5, 10, 20, 30)

SCALING_METRICS = (
    "objective",
    "total_latency_ms",
    "communication_latency_ms",
    "energy",
    "device_switches",
    "gap_vs_best_pct",
    "decision_runtime_us_per_layer",
    "decision_runtime_s",
)

# Lines are capped at a readable number; these are the methods worth tracking
# across depth, and each keeps its colour from every other figure.
PLOTTED_METHODS = (
    "random",
    "greedy_communication_aware",
    "greedy_objective_aware",
    "supervised_rf",
    "dqn",
    "dqn_mixed",
    "dp_relaxed",
)


def main() -> int:
    """Run the scaling comparison and write its figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scaling.yaml")
    parser.add_argument("--depths", type=int, nargs="*", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--scenarios", type=int, default=None)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config, arguments.overrides)
    seeds = evaluation_seeds(config, arguments.scenarios)

    print("=" * 78)
    print("EXPERIMENT 2: SCALING WITH DNN DEPTH")
    print("=" * 78)
    print(config_summary(config))
    print(f"depths: {arguments.depths}, scenarios per depth: {len(seeds)}")

    agents = optional_agents(
        config,
        dqn_checkpoints={
            "dqn": CHECKPOINTS / "dqn.pt",
            "dqn_mixed": CHECKPOINTS / "dqn_mixed.pt",
        },
    )

    frames = []
    for depth in arguments.depths:
        affordable = exhaustive_search.is_affordable(
            depth, config.num_devices, config.experiment.max_exhaustive_combinations
        )
        started = time.perf_counter()
        frame = compare_methods(
            config,
            seeds,
            num_layers=depth,
            extra_agents=agents,
            include_exhaustive=affordable,
        )
        frame["num_layers"] = depth
        frames.append(frame)

        note = "with exhaustive search" if affordable else "dynamic programme only"
        print(f"\n[{depth} layers] {note} ({time.perf_counter() - started:.1f}s)")
        print(
            format_summary_table(
                summarise_by_method(frame, SCALING_METRICS),
                ["objective", "total_latency_ms", "energy", "gap_vs_best_pct"],
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    summary = summarise_by_method(
        combined, SCALING_METRICS, group_columns=("num_layers",)
    )
    save_raw(combined, "e2_scaling", config, depths=arguments.depths)
    save_processed(summary, "e2_scaling_summary")

    figures = [
        visualization.lines_by_x(
            summary,
            "num_layers",
            "objective",
            "fig07_objective_vs_depth",
            title="Placement quality as the DNN gets deeper",
            subtitle="Weighted objective on held-out scenarios, log scale; "
            "1.0 is the cost of random placement",
            x_label="Number of DNN layers",
            methods=PLOTTED_METHODS,
            log_y=True,
        ),
        visualization.lines_by_x(
            summary,
            "num_layers",
            "gap_vs_best_pct",
            "fig07b_gap_vs_depth",
            title="Distance from the best placement found, by depth",
            subtitle="Log scale; the best-known placement is the per-scenario minimum",
            x_label="Number of DNN layers",
            methods=[m for m in PLOTTED_METHODS if m != "random"],
            log_y=True,
        ),
        visualization.lines_by_x(
            summary,
            "num_layers",
            "decision_runtime_s",
            "fig07c_runtime_vs_depth",
            title="Time to place a whole DNN",
            subtitle="Log scale. Exhaustive search is absent beyond 5 layers because it is "
            "no longer affordable",
            x_label="Number of DNN layers",
            methods=(*PLOTTED_METHODS, "exhaustive"),
            log_y=True,
        ),
    ]
    print("\nFigures written:")
    for path in figures:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
