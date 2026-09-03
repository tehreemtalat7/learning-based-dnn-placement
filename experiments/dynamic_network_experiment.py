"""Experiment 3: behaviour when network conditions change.

Three test regimes, all on the same held-out scenario seeds:

``normal``
    Low latency, high bandwidth throughout.

``congested``
    Latency multiplied and bandwidth divided from the first layer, so the whole
    episode runs in a degraded network.

``dynamic``
    Congestion may begin part way through an episode, so conditions change while
    the agent is still placing layers.

Two learned policies are compared:

* ``dqn`` trained only on the normal network -- tested under **distribution
  shift** it never saw during training.
* ``dqn_dynamic`` trained on the dynamic distribution.

The comparison to keep in mind is that the greedy heuristics are **already
reactive**: they recompute their decision at every layer from the current link
state, which the observation exposes to every method. So a learned policy has no
information advantage here. If it wins, it must be by *anticipating* -- for
instance, by keeping consecutive layers together when a transfer is about to
become expensive. If it does not win, that is the honest result and is reported
as such.

Run it with::

    python -m experiments.dynamic_network_experiment
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from experiments._shared import CHECKPOINTS, compare_methods, optional_agents
from src.config import config_summary, load_config
from src.environment.scenario import evaluation_seeds, sample_scenario
from src.utils import visualization
from src.utils.io import save_processed, save_raw
from src.utils.metrics import format_summary_table, summarise_by_method
from src.utils.stats import paired_comparison

REGIMES = ("normal", "congested", "dynamic")

REPORTED_METRICS = (
    "objective",
    "total_latency_ms",
    "communication_latency_ms",
    "energy",
    "device_switches",
    "gap_vs_best_pct",
)

PLOTTED_METHODS = (
    "greedy_communication_aware",
    "greedy_objective_aware",
    "supervised_rf",
    "dqn",
    "dqn_dynamic",
    "dp_relaxed",
)


def describe_regime(config, seeds: list[int]) -> str:
    """Summarise how often congestion actually occurs in a regime."""
    scenarios = [sample_scenario(config, seed) for seed in seeds]
    congested = [scenario for scenario in scenarios if scenario.has_congestion]
    if not congested:
        return "no congestion events"
    mid_episode = sum(
        1 for scenario in congested if (scenario.congestion_start_layer or 0) > 0
    )
    return (
        f"{len(congested)}/{len(scenarios)} scenarios congested, "
        f"{mid_episode} of them starting mid-episode"
    )


def main() -> int:
    """Compare every method across the three network regimes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=None)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    arguments = parser.parse_args()

    print("=" * 78)
    print("EXPERIMENT 3: DYNAMIC NETWORK CONDITIONS")
    print("=" * 78)

    frames = []
    for regime in REGIMES:
        config = load_config(
            "static_environment.yaml", [*arguments.overrides, f"network.profile={regime}"]
        )
        seeds = evaluation_seeds(config, arguments.scenarios)
        agents = optional_agents(
            config,
            dqn_checkpoints={
                "dqn": CHECKPOINTS / "dqn.pt",
                "dqn_dynamic": CHECKPOINTS / "dqn_dynamic.pt",
            },
        )

        started = time.perf_counter()
        frame = compare_methods(config, seeds, extra_agents=agents)
        frame["regime"] = regime
        frames.append(frame)

        print(f"\n[{regime}] {describe_regime(config, seeds)} "
              f"({time.perf_counter() - started:.1f}s)")
        print(config_summary(config))
        print(
            format_summary_table(
                summarise_by_method(frame, REPORTED_METRICS),
                ["objective", "total_latency_ms", "communication_latency_ms", "gap_vs_best_pct"],
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    summary = summarise_by_method(combined, REPORTED_METRICS, group_columns=("regime",))
    save_raw(combined, "e3_dynamic_network", config, regimes=list(REGIMES))
    save_processed(summary, "e3_dynamic_network_summary")

    print("\nDegradation from the normal network (weighted objective)")
    print("-" * 78)
    rows = []
    for method in sorted(set(combined["method"])):
        baseline = combined[(combined["method"] == method) & (combined["regime"] == "normal")]
        baseline_mean = float(baseline["objective"].mean())
        entry: dict[str, object] = {"method": method, "normal": baseline_mean}
        for regime in REGIMES[1:]:
            shifted = combined[
                (combined["method"] == method) & (combined["regime"] == regime)
            ]
            entry[regime] = float(shifted["objective"].mean())
            entry[f"{regime}_increase_pct"] = (entry[regime] - baseline_mean) / baseline_mean * 100
        rows.append(entry)
        print(
            f"  {method:<28} normal {entry['normal']:.3f}  "
            f"congested {entry['congested']:.3f} ({entry['congested_increase_pct']:+.1f}%)  "
            f"dynamic {entry['dynamic']:.3f} ({entry['dynamic_increase_pct']:+.1f}%)"
        )
    degradation = pd.DataFrame(rows)
    save_processed(degradation, "e3_dynamic_network_degradation")

    print("\nTrained-on-normal vs trained-on-dynamic, per regime")
    print("-" * 78)
    comparisons = []
    for regime in REGIMES:
        subset = combined[combined["regime"] == regime]
        pivot = subset.pivot_table(index="scenario_seed", columns="method", values="objective")
        for method in ("dqn", "dqn_dynamic", "supervised_rf"):
            if method not in pivot.columns or "greedy_objective_aware" not in pivot.columns:
                continue
            aligned = pivot[[method, "greedy_objective_aware"]].dropna()
            result = paired_comparison(
                aligned[method], aligned["greedy_objective_aware"], method, "greedy_objective_aware"
            )
            comparisons.append({"regime": regime, **result.__dict__ | {
                "difference": result.difference.estimate,
                "ci_low": result.difference.low,
                "ci_high": result.difference.high,
            }})
            print(f"  [{regime:<9}] {result.summary()}")
    save_processed(
        pd.DataFrame(comparisons).drop(columns=["name_a", "name_b"], errors="ignore"),
        "e3_dynamic_network_paired_tests",
    )

    figure = visualization.grouped_bars(
        summary,
        "regime",
        "objective",
        "fig08_network_conditions",
        title="Placement quality under three network regimes",
        subtitle="Weighted objective on identical held-out scenarios; lower is better",
        x_label="Network regime",
        methods=PLOTTED_METHODS,
        group_order=REGIMES,
        value_format="{:,.3f}",
    )
    print(f"\nFigure written:\n  {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
