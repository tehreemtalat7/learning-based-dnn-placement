"""Smoke tests for figure generation.

These assert that every chart function runs and writes a non-trivial PNG. They
cannot judge whether a figure looks right -- that is done by opening it -- but
they do catch the failures that break an experiment run at the very last step.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils import io as results_io
from src.utils import visualization
from src.utils.metrics import summarise_by_method


@pytest.fixture
def figures_directory(tmp_path, monkeypatch):
    directory = tmp_path / "figures"
    monkeypatch.setattr(results_io, "FIGURES_DIRECTORY", directory)
    monkeypatch.setattr(results_io, "RAW_DIRECTORY", tmp_path / "raw")
    monkeypatch.setattr(results_io, "PROCESSED_DIRECTORY", tmp_path / "processed")
    return directory


@pytest.fixture
def summary() -> pd.DataFrame:
    rows = []
    for method, objective, runtime in (
        ("random", 1.2, 7.0),
        ("greedy_objective_aware", 0.45, 3.0),
        ("dp_relaxed", 0.46, 34.0),
    ):
        for seed in range(6):
            rows.append(
                {
                    "method": method,
                    "scenario_seed": seed,
                    "num_layers": 10,
                    "objective": objective + seed * 0.01,
                    "energy": objective * 20,
                    "total_latency_ms": objective * 1000,
                    "decision_runtime_us_per_layer": runtime + seed * 0.1,
                }
            )
    return summarise_by_method(pd.DataFrame(rows))


def test_bar_chart_is_written(figures_directory, summary):
    path = visualization.bars_by_method(
        summary, "objective", "unit_bars", title="Objective", subtitle="test"
    )
    assert path.exists() and path.stat().st_size > 5_000


def test_dot_plot_is_written(figures_directory, summary):
    path = visualization.dots_by_method(
        summary, "decision_runtime_us_per_layer", "unit_dots", title="Runtime"
    )
    assert path.exists() and path.stat().st_size > 5_000


def test_line_chart_is_written(figures_directory):
    rows = []
    for method in ("dqn", "greedy_objective_aware"):
        for layers in (5, 10, 20):
            for seed in range(4):
                rows.append(
                    {
                        "method": method,
                        "scenario_seed": seed,
                        "num_layers": layers,
                        "objective": 0.4 + layers * 0.01 + seed * 0.001,
                    }
                )
    summary = summarise_by_method(
        pd.DataFrame(rows), ["objective"], group_columns=("num_layers",)
    )
    path = visualization.lines_by_x(
        summary, "num_layers", "objective", "unit_lines", title="Scaling"
    )
    assert path.exists() and path.stat().st_size > 5_000


def test_grouped_bars_are_written(figures_directory):
    rows = []
    for method in ("dqn", "greedy_objective_aware", "random"):
        for condition in ("normal", "congested"):
            for seed in range(4):
                rows.append(
                    {
                        "method": method,
                        "scenario_seed": seed,
                        "num_layers": 10,
                        "condition": condition,
                        "objective": 0.5 + (condition == "congested") * 0.2 + seed * 0.01,
                    }
                )
    summary = summarise_by_method(
        pd.DataFrame(rows), ["objective"], group_columns=("condition",)
    )
    path = visualization.grouped_bars(
        summary, "condition", "objective", "unit_groups", title="Conditions"
    )
    assert path.exists() and path.stat().st_size > 5_000


def test_colour_and_marker_assignment_is_stable_per_method():
    """A figure that drops a method must not repaint the others."""
    assert visualization.colour_for("dqn") == visualization.SLOT_COLOURS[0]
    assert visualization.colour_for("greedy_objective_aware") == visualization.SLOT_COLOURS[1]
    assert visualization.colour_for("exhaustive") == visualization.TEXT_SECONDARY
    assert visualization.marker_for("dqn") != visualization.marker_for("greedy_objective_aware")


def test_no_figure_plots_two_methods_in_the_same_colour():
    """Colour identifies the method, so a collision inside one figure is a bug.

    Several methods deliberately share a slot because they never appear
    together; this checks that assumption against the actual method lists the
    experiment scripts plot.
    """
    from experiments.dynamic_device_experiment import PLOTTED_METHODS as device_methods
    from experiments.dynamic_network_experiment import PLOTTED_METHODS as network_methods
    from experiments.scaling_experiment import PLOTTED_METHODS as scaling_methods

    for name, methods in (
        ("scaling", scaling_methods),
        ("dynamic network", network_methods),
        ("device load", device_methods),
    ):
        colours = {}
        for method in methods:
            if method in visualization.REFERENCE_METHODS:
                continue  # references share the neutral ink by design
            colour = visualization.colour_for(method)
            assert colour not in colours, (
                f"{name} figure gives {method!r} and {colours[colour]!r} the same colour"
            )
            colours[colour] = method


def test_reference_methods_are_drawn_in_neutral_ink():
    for method in ("exhaustive", "dp_relaxed", "dp_exact"):
        assert visualization.colour_for(method) == visualization.TEXT_SECONDARY
