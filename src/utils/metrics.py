"""Turning episode records into tables, gaps and summaries.

Raw per-episode rows are written to ``results/raw/`` before any aggregation, so
every number that appears in a figure or in the README can be traced back to the
individual scenarios that produced it.

The optimality gap follows the convention used in the previous project, adapted
for minimisation::

    gap = (method_objective - reference_objective) / reference_objective * 100

computed **per scenario** and then averaged, not computed from the averages. The
two differ, and per-scenario is the honest one: it answers "how much worse is
this method on a typical problem" rather than "how do the aggregates compare".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from src.training.evaluate import EpisodeRecord
from src.utils.stats import Interval, bootstrap_interval, mean_interval

# Display names and a stable ordering, so every table and figure lists methods
# in the same sequence: weak heuristics, strong heuristics, learned, exact.
METHOD_LABELS: dict[str, str] = {
    "random": "Random",
    "round_robin": "Round robin",
    "greedy_fastest_device": "Greedy (fastest device)",
    "greedy_communication_aware": "Greedy (communication-aware)",
    "greedy_objective_aware": "Greedy (objective-aware)",
    "supervised_rf": "Random Forest (supervised)",
    "tabular_q": "Tabular Q (single scenario)",
    "tabular_q_pooled": "Tabular Q (pooled)",
    "dqn": "DQN",
    "dp_relaxed": "DP (relaxed problem)",
    "dp_exact": "Dynamic programming (optimal)",
    "dp_lower_bound": "DP lower bound",
    "exhaustive": "Exhaustive search",
}

METHOD_ORDER: tuple[str, ...] = tuple(METHOD_LABELS)

# Metrics reported for every method, with the label used on figure axes.
METRIC_LABELS: dict[str, str] = {
    "objective": "Weighted objective (lower is better)",
    "total_latency_ms": "End-to-end latency (ms)",
    "compute_latency_ms": "Computation latency (ms)",
    "communication_latency_ms": "Communication latency (ms)",
    "energy": "Energy (units)",
    "device_switches": "Device switches per DNN",
    "memory_violations": "Memory violations per DNN",
    "invalid_action_attempts": "Invalid action attempts per DNN",
    "decision_runtime_us_per_layer": "Placement runtime per layer (us)",
    "optimality_gap_pct": "Optimality gap (%)",
    "gap_vs_best_pct": "Gap vs best known placement (%)",
    "gap_vs_dp_bound_pct": "Gap vs DP lower bound (%)",
}


def label_for(method: str) -> str:
    """Human-readable name for a method key."""
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def sort_methods(methods: Iterable[str]) -> list[str]:
    """Order method keys canonically, keeping unknown keys at the end."""
    known = [method for method in METHOD_ORDER if method in set(methods)]
    unknown = sorted(set(methods) - set(known))
    return known + unknown


def records_to_frame(records: Iterable[EpisodeRecord]) -> pd.DataFrame:
    """Convert episode records into a tidy data frame, one row per episode."""
    frame = pd.DataFrame([record.to_row() for record in records])
    if frame.empty:
        return frame
    return frame.sort_values(["method", "scenario_seed"]).reset_index(drop=True)


def add_optimality_gap(
    frame: pd.DataFrame,
    reference_method: str,
    *,
    objective_column: str = "objective",
    gap_column: str = "optimality_gap_pct",
    group_columns: Sequence[str] = ("scenario_seed", "num_layers"),
) -> pd.DataFrame:
    """Add a per-scenario gap relative to a reference method.

    Args:
        frame: Tidy frame of episode rows.
        reference_method: Method whose objective is the reference, typically
            ``"exhaustive"`` where brute force is affordable.
        objective_column: Column holding the objective.
        gap_column: Name of the column to add.
        group_columns: Columns identifying one problem instance.

    Returns:
        A copy of ``frame`` with the gap column added. Scenarios for which the
        reference method has no row receive ``NaN`` rather than being dropped.

    Raises:
        ValueError: If the reference method is absent from the frame.
    """
    if reference_method not in set(frame["method"]):
        raise ValueError(
            f"reference method {reference_method!r} is not present; "
            f"available methods are {sorted(set(frame['method']))}"
        )

    keys = list(group_columns)
    reference = (
        frame.loc[frame["method"] == reference_method, keys + [objective_column]]
        .rename(columns={objective_column: "_reference_objective"})
        .drop_duplicates(subset=keys)
    )
    merged = frame.merge(reference, on=keys, how="left")
    merged[gap_column] = (
        (merged[objective_column] - merged["_reference_objective"])
        / merged["_reference_objective"]
        * 100.0
    )
    return merged.drop(columns=["_reference_objective"])


def add_gap_vs_best(
    frame: pd.DataFrame,
    *,
    objective_column: str = "objective",
    gap_column: str = "gap_vs_best_pct",
    group_columns: Sequence[str] = ("scenario_seed", "num_layers"),
    candidates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Add a per-scenario gap relative to the best placement any method found.

    This is the honest reference wherever the true optimum is out of reach. It is
    non-negative by construction, and it does not pretend that a relaxed
    dynamic-programming solution is optimal -- which matters here, because once
    memory and utilisation accumulate, a state-aware greedy heuristic can and
    does beat the relaxation.

    Args:
        frame: Tidy frame of episode rows.
        objective_column: Column holding the objective.
        gap_column: Name of the column to add.
        group_columns: Columns identifying one problem instance.
        candidates: Methods eligible to define the best-known value; defaults to
            every method present. Infeasible placements are excluded either way.

    Returns:
        A copy of ``frame`` with the gap column added.
    """
    keys = list(group_columns)
    eligible = frame
    if candidates is not None:
        eligible = eligible[eligible["method"].isin(list(candidates))]
    if "memory_violations" in eligible.columns:
        eligible = eligible[eligible["memory_violations"] == 0]

    best = (
        eligible.groupby(keys, dropna=False)[objective_column]
        .min()
        .rename("_best_objective")
        .reset_index()
    )
    merged = frame.merge(best, on=keys, how="left")
    merged[gap_column] = (
        (merged[objective_column] - merged["_best_objective"]) / merged["_best_objective"] * 100.0
    )
    return merged.drop(columns=["_best_objective"])


def summarise_by_method(
    frame: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    *,
    group_columns: Sequence[str] = (),
    bootstrap_metrics: Sequence[str] = ("optimality_gap_pct",),
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Aggregate episode rows into one row per method (and optional grouping).

    Args:
        frame: Tidy frame of episode rows.
        metrics: Columns to summarise; defaults to every known metric present.
        group_columns: Extra grouping columns, e.g. ``("num_layers",)`` for the
            scaling experiment.
        bootstrap_metrics: Metrics whose confidence interval is computed by
            bootstrap rather than the t distribution, for skewed quantities.
        confidence: Coverage of the intervals.

    Returns:
        A frame with ``mean``, ``ci_low``, ``ci_high``, ``median`` and ``n``
        columns for each requested metric.
    """
    if frame.empty:
        return frame

    chosen = [
        metric
        for metric in (metrics if metrics is not None else METRIC_LABELS)
        if metric in frame.columns
    ]
    keys = ["method", *group_columns]

    rows = []
    for key_values, group in frame.groupby(keys, dropna=False, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row: dict[str, object] = dict(zip(keys, key_values, strict=True))
        row["episodes"] = len(group)
        for metric in chosen:
            values = group[metric].dropna()
            if values.empty:
                continue
            interval: Interval = (
                bootstrap_interval(values, confidence=confidence)
                if metric in bootstrap_metrics
                else mean_interval(values, confidence=confidence)
            )
            row[f"{metric}_mean"] = interval.estimate
            row[f"{metric}_ci_low"] = interval.low
            row[f"{metric}_ci_high"] = interval.high
            row[f"{metric}_median"] = float(values.median())
        rows.append(row)

    summary = pd.DataFrame(rows)
    ordering = {method: index for index, method in enumerate(sort_methods(summary["method"]))}
    summary["_order"] = summary["method"].map(ordering)
    sort_by = ["_order", *group_columns]
    return summary.sort_values(sort_by).drop(columns="_order").reset_index(drop=True)


def format_summary_table(
    summary: pd.DataFrame,
    metrics: Sequence[str],
    *,
    decimals: int = 3,
) -> str:
    """Render a summary frame as a fixed-width table for terminal output."""
    if summary.empty:
        return "(no results)"

    headers = ["method", *metrics]
    widths = {"method": max(len(label_for(m)) for m in summary["method"]) + 2}
    for metric in metrics:
        widths[metric] = max(len(metric), 12) + 2

    lines = ["".join(header.rjust(widths[header]) for header in headers)]
    for _, row in summary.iterrows():
        cells = [label_for(row["method"]).rjust(widths["method"])]
        for metric in metrics:
            column = f"{metric}_mean"
            value = row.get(column)
            text = "-" if pd.isna(value) else f"{value:,.{decimals}f}"
            cells.append(text.rjust(widths[metric]))
        lines.append("".join(cells))
    return "\n".join(lines)


__all__ = [
    "METHOD_LABELS",
    "METHOD_ORDER",
    "METRIC_LABELS",
    "add_gap_vs_best",
    "add_optimality_gap",
    "format_summary_table",
    "label_for",
    "records_to_frame",
    "sort_methods",
    "summarise_by_method",
]
