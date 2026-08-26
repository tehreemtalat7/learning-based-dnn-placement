"""Experiment 1: static comparison of every placement method.

Compares all methods on the same held-out scenarios under stable device
performance and network conditions, and reports how far each lands from optimal.

The experiment runs three panels, because the optimal reference is affordable in
different ways at different depths:

``main``
    The default depth (10 layers by default) over the full evaluation set. The
    optimality reference is the dynamic-programming placement, which is exact
    when accumulation is disabled and, when it is enabled, was measured on the
    ``bound`` panel to sit about 1 % from the true optimum. Exhaustive search is
    not affordable here: with four devices, ten layers is 1 048 576 candidate
    placements, roughly 84 seconds *per scenario*.

``small``
    Five layers, where exhaustive search costs only 1 024 candidates per
    scenario. This panel carries the genuine, exactly-computed optimality gaps.

``bound``
    Eight layers on a small subset, running exhaustive search and dynamic
    programming side by side. This is what licenses the claim made in the
    ``main`` panel: it measures how far the dynamic programme actually sits from
    the true optimum instead of assuming it.

Run it with::

    python -m experiments.static_experiment
    python -m experiments.static_experiment --scenarios 50 --skip-bound   # quick
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.agents import build_heuristic_agents
from src.baselines import dp_optimal, exhaustive_search
from src.config import Config, config_summary, load_config
from src.environment.scenario import evaluation_seeds, sample_scenario
from src.training.evaluate import evaluate_agent, evaluate_placement_function
from src.utils import visualization
from src.utils.io import save_processed, save_raw
from src.utils.metrics import (
    add_gap_vs_best,
    add_optimality_gap,
    format_summary_table,
    records_to_frame,
    summarise_by_method,
)
from src.utils.stats import paired_comparison

REPORTED_METRICS = (
    "objective",
    "total_latency_ms",
    "compute_latency_ms",
    "communication_latency_ms",
    "energy",
    "device_switches",
    "memory_violations",
    "decision_runtime_us_per_layer",
    "optimality_gap_pct",
    "gap_vs_best_pct",
    "gap_vs_dp_bound_pct",
)

SMALL_PANEL_LAYERS = 5
BOUND_PANEL_LAYERS = 8
BOUND_PANEL_SCENARIOS = 40


def collect_records(
    config: Config,
    seeds: list[int],
    num_layers: int,
    *,
    include_exhaustive: bool,
) -> pd.DataFrame:
    """Run every available method over the given scenarios.

    Args:
        config: The configuration in force.
        seeds: Held-out scenario seeds, identical for every method.
        num_layers: DNN depth for this panel.
        include_exhaustive: Whether to add the brute-force optimum.

    Returns:
        A tidy frame with one row per (method, scenario).
    """
    records = []
    for agent in build_heuristic_agents(config):
        records.extend(evaluate_agent(config, agent, seeds, num_layers=num_layers))

    # The dynamic programme is exact only when neither memory nor utilisation
    # accumulates; otherwise it optimises a relaxation of the real problem, and
    # is reported under a name that says so.
    dp_method = "dp_exact" if dp_optimal.is_exact_for(config) else "dp_relaxed"
    records.extend(
        evaluate_placement_function(
            config,
            lambda scenario: dp_optimal.solve(scenario, config).placement,
            seeds,
            dp_method,
            num_layers=num_layers,
        )
    )

    if include_exhaustive:
        records.extend(
            evaluate_placement_function(
                config,
                lambda scenario: exhaustive_search.solve(scenario, config).placement,
                seeds,
                "exhaustive",
                num_layers=num_layers,
            )
        )

    return records_to_frame(records)


def dynamic_programming_bounds(config: Config, seeds: list[int], num_layers: int) -> pd.DataFrame:
    """Per-scenario dynamic-programming objective, recorded as a lower bound."""
    rows = []
    for seed in seeds:
        scenario = sample_scenario(config, seed, num_layers=num_layers)
        solution = dp_optimal.solve(scenario, config)
        rows.append(
            {
                "scenario_seed": seed,
                "num_layers": num_layers,
                "dp_objective": solution.objective,
                "is_exact": solution.is_exact,
            }
        )
    return pd.DataFrame(rows)


def run_panel(
    config: Config,
    seeds: list[int],
    num_layers: int,
    *,
    include_exhaustive: bool,
    panel: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run one panel and return its raw rows, its summary and the gap reference."""
    started = time.perf_counter()
    frame = collect_records(config, seeds, num_layers, include_exhaustive=include_exhaustive)
    elapsed = time.perf_counter() - started

    # Three references, each honest about what it is:
    #   * the true optimum, where exhaustive search is affordable;
    #   * the best placement any method found, everywhere else;
    #   * the dynamic-programming lower bound, which is certified but loose.
    if include_exhaustive:
        reference = "exhaustive"
        frame = add_optimality_gap(frame, reference)
    else:
        reference = "best_known"
        frame["optimality_gap_pct"] = float("nan")
    frame = add_gap_vs_best(frame)
    frame["panel"] = panel

    bounds = dynamic_programming_bounds(config, seeds, num_layers)
    frame = frame.merge(bounds[["scenario_seed", "dp_objective"]], on="scenario_seed", how="left")
    frame["gap_vs_dp_bound_pct"] = (
        (frame["objective"] - frame["dp_objective"]) / frame["dp_objective"] * 100.0
    )

    summary = summarise_by_method(frame, REPORTED_METRICS)
    gap_column = "optimality_gap_pct" if include_exhaustive else "gap_vs_best_pct"
    print(
        f"\n[{panel}] {len(seeds)} scenarios x {num_layers} layers, "
        f"reference = {reference} ({elapsed:.1f}s)"
    )
    print(
        format_summary_table(
            summary,
            ["objective", "total_latency_ms", "energy", gap_column, "gap_vs_dp_bound_pct"],
        )
    )
    return frame, summary, reference


def report_bound_tightness(config: Config, seeds: list[int]) -> pd.DataFrame:
    """Measure how far the dynamic-programming value sits from the true optimum.

    When accumulation is enabled the dynamic programme solves a relaxation, so
    its value is a lower bound rather than the optimum. This panel quantifies the
    slack by running exhaustive search alongside it at a depth where brute force
    is still affordable.
    """
    rows = []
    for seed in seeds:
        scenario = sample_scenario(config, seed, num_layers=BOUND_PANEL_LAYERS)
        dp_solution = dp_optimal.solve(scenario, config)
        brute = exhaustive_search.solve(scenario, config, budget=10**9)
        from src.environment.reward import evaluate_placement

        dp_actual = evaluate_placement(scenario, dp_solution.placement, config)
        rows.append(
            {
                "scenario_seed": seed,
                "num_layers": BOUND_PANEL_LAYERS,
                "dp_bound": dp_solution.objective,
                "dp_placement_objective": dp_actual.objective,
                "dp_placement_feasible": dp_actual.memory_violations == 0,
                "true_optimum": brute.objective,
                "bound_slack_pct": (brute.objective - dp_solution.objective)
                / brute.objective
                * 100.0,
                "dp_placement_gap_pct": (dp_actual.objective - brute.objective)
                / brute.objective
                * 100.0,
                "candidates_evaluated": brute.evaluated,
            }
        )
    frame = pd.DataFrame(rows)

    print(f"\n[bound] {len(frame)} scenarios x {BOUND_PANEL_LAYERS} layers")
    print(
        f"  DP lower bound sits {frame['bound_slack_pct'].mean():.2f}% below the true optimum "
        f"(max {frame['bound_slack_pct'].max():.2f}%)"
    )
    print(
        f"  DP placement is {frame['dp_placement_gap_pct'].mean():.2f}% above the true optimum "
        f"(max {frame['dp_placement_gap_pct'].max():.2f}%)"
    )
    print(
        f"  DP placement feasible under accumulation in "
        f"{frame['dp_placement_feasible'].mean():.0%} of scenarios"
    )
    return frame


def report_paired_tests(frame: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Compare every method against the strongest heuristic and the reference."""
    pivot = frame.pivot_table(index="scenario_seed", columns="method", values="objective")
    comparisons = []
    baselines = ["greedy_objective_aware", reference if reference in pivot.columns else "dp_relaxed"]
    for baseline in dict.fromkeys(baselines):
        if baseline not in pivot.columns:
            continue
        for method in pivot.columns:
            if method == baseline:
                continue
            aligned = pivot[[method, baseline]].dropna()
            if aligned.empty:
                continue
            result = paired_comparison(
                aligned[method], aligned[baseline], name_a=method, name_b=baseline
            )
            comparisons.append(
                {
                    "method": result.name_a,
                    "baseline": result.name_b,
                    "mean_difference": result.difference.estimate,
                    "ci_low": result.difference.low,
                    "ci_high": result.difference.high,
                    "relative_difference_pct": result.relative_difference_pct,
                    "win_rate": result.win_rate,
                    "p_value": result.p_value,
                    "n": result.n,
                }
            )
    return pd.DataFrame(comparisons)


def make_figures(main_summary: pd.DataFrame, small_summary: pd.DataFrame) -> list[str]:
    """Generate the static-comparison figures."""
    written = [
        visualization.bars_by_method(
            main_summary,
            "total_latency_ms",
            "fig03_latency_by_method",
            title="End-to-end latency by placement method",
            subtitle="Mean over held-out scenarios; bars show 95% confidence intervals",
            value_format="{:,.0f}",
        ),
        visualization.bars_by_method(
            main_summary,
            "energy",
            "fig04_energy_by_method",
            title="Energy consumption by placement method",
            subtitle="Mean over held-out scenarios; bars show 95% confidence intervals",
            value_format="{:,.1f}",
        ),
        visualization.bars_by_method(
            main_summary,
            "objective",
            "fig05_objective_by_method",
            title="Weighted objective by placement method",
            subtitle="1.0 is the expected cost of uniformly random placement; lower is better",
        ),
        visualization.dots_by_method(
            main_summary,
            "decision_runtime_us_per_layer",
            "fig06_runtime_by_method",
            title="Placement decision time per layer",
            subtitle="Log scale; time the method spends deciding, excluding simulation. "
            "Lines show 95% confidence intervals",
            value_format="{:,.1f} us",
        ),
        visualization.bars_by_method(
            small_summary,
            "optimality_gap_pct",
            "fig10_optimality_gap",
            title="Optimality gap against exhaustive search",
            subtitle=f"{SMALL_PANEL_LAYERS}-layer DNNs, where the true optimum is affordable",
            value_format="{:,.2f}%",
        ),
    ]
    return [str(path) for path in written]


def main() -> int:
    """Run the static comparison experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="static_environment.yaml")
    parser.add_argument("--scenarios", type=int, default=None)
    parser.add_argument("--skip-bound", action="store_true", help="skip the slow bound panel")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config, arguments.overrides)
    seeds = evaluation_seeds(config, arguments.scenarios)

    print("=" * 78)
    print("EXPERIMENT 1: STATIC COMPARISON")
    print("=" * 78)
    print(config_summary(config))
    print(f"dynamic programming is exact for this configuration: {dp_optimal.is_exact_for(config)}")

    main_frame, main_summary, main_reference = run_panel(
        config,
        seeds,
        config.workload.num_layers,
        include_exhaustive=False,
        panel="main",
    )
    small_frame, small_summary, small_reference = run_panel(
        config,
        seeds,
        SMALL_PANEL_LAYERS,
        include_exhaustive=True,
        panel="small",
    )

    save_raw(main_frame, "e1_static_main", config, panel="main", reference=main_reference)
    save_raw(small_frame, "e1_static_small", config, panel="small", reference=small_reference)
    save_processed(main_summary, "e1_static_main_summary")
    save_processed(small_summary, "e1_static_small_summary")

    tests = report_paired_tests(main_frame, main_reference)
    save_processed(tests, "e1_static_main_paired_tests")
    print("\nPaired comparisons against the strongest heuristic (objective)")
    for _, row in tests[tests["baseline"] == "greedy_objective_aware"].iterrows():
        direction = "cheaper" if row["mean_difference"] < 0 else "more expensive"
        print(
            f"  {row['method']:<28} {abs(row['relative_difference_pct']):6.2f}% {direction}, "
            f"wins {row['win_rate']:.0%}, p={row['p_value']:.2e}"
        )

    if not arguments.skip_bound:
        bound_seeds = evaluation_seeds(config, BOUND_PANEL_SCENARIOS)
        bound_frame = report_bound_tightness(config, bound_seeds)
        save_raw(bound_frame, "e1_static_bound_tightness", config, panel="bound")

    figures = make_figures(main_summary, small_summary)
    print("\nFigures written:")
    for path in figures:
        print(f"  {path}")

    if np.any(main_frame["memory_violations"] > 0):
        offenders = main_frame.loc[main_frame["memory_violations"] > 0, "method"].unique()
        print(f"\nNote: memory violations recorded for {list(offenders)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
