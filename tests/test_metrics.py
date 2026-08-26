"""Tests for aggregation, gaps and paired statistics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.utils import io as results_io
from src.utils.metrics import (
    add_gap_vs_best,
    add_optimality_gap,
    format_summary_table,
    label_for,
    sort_methods,
    summarise_by_method,
)
from src.utils.stats import bootstrap_interval, mean_interval, paired_comparison


def make_frame() -> pd.DataFrame:
    rows = []
    # Two scenarios; the reference is cheapest, greedy close behind, random far off.
    for seed, base in ((1, 1.0), (2, 2.0)):
        rows.extend(
            [
                {"method": "exhaustive", "scenario_seed": seed, "num_layers": 5,
                 "objective": base, "energy": 10.0, "memory_violations": 0},
                {"method": "greedy_objective_aware", "scenario_seed": seed, "num_layers": 5,
                 "objective": base * 1.1, "energy": 11.0, "memory_violations": 0},
                {"method": "random", "scenario_seed": seed, "num_layers": 5,
                 "objective": base * 2.0, "energy": 20.0, "memory_violations": 0},
            ]
        )
    return pd.DataFrame(rows)


class TestGaps:
    def test_optimality_gap_is_computed_per_scenario(self):
        frame = add_optimality_gap(make_frame(), "exhaustive")
        greedy = frame[frame["method"] == "greedy_objective_aware"]["optimality_gap_pct"]
        assert np.allclose(greedy, 10.0)
        reference = frame[frame["method"] == "exhaustive"]["optimality_gap_pct"]
        assert np.allclose(reference, 0.0)

    def test_missing_reference_method_raises(self):
        with pytest.raises(ValueError, match="not present"):
            add_optimality_gap(make_frame(), "dqn")

    def test_gap_vs_best_is_non_negative_and_zero_for_the_winner(self):
        frame = add_gap_vs_best(make_frame())
        assert (frame["gap_vs_best_pct"] >= -1e-9).all()
        assert np.allclose(frame[frame["method"] == "exhaustive"]["gap_vs_best_pct"], 0.0)

    def test_gap_vs_best_ignores_infeasible_placements(self):
        frame = make_frame()
        # An infeasible placement is cheap but must not define the reference.
        frame.loc[len(frame)] = {
            "method": "dqn", "scenario_seed": 1, "num_layers": 5,
            "objective": 0.1, "energy": 1.0, "memory_violations": 3,
        }
        result = add_gap_vs_best(frame)
        winner = result[(result["scenario_seed"] == 1) & (result["method"] == "exhaustive")]
        assert winner["gap_vs_best_pct"].iloc[0] == pytest.approx(0.0)

    def test_per_scenario_averaging_differs_from_averaging_the_aggregates(self):
        """The documented convention: average the gaps, do not gap the averages."""
        frame = add_optimality_gap(make_frame(), "exhaustive")
        per_scenario = frame[frame["method"] == "random"]["optimality_gap_pct"].mean()
        assert per_scenario == pytest.approx(100.0)


class TestSummary:
    def test_summary_reports_mean_interval_and_median(self):
        summary = summarise_by_method(add_optimality_gap(make_frame(), "exhaustive"))
        row = summary[summary["method"] == "random"].iloc[0]
        # random costs 2.0 and 4.0 on the two scenarios
        assert row["objective_mean"] == pytest.approx(3.0)
        assert row["objective_ci_low"] <= row["objective_mean"] <= row["objective_ci_high"]
        assert row["objective_median"] == pytest.approx(3.0)
        assert row["episodes"] == 2

    def test_methods_are_reported_in_canonical_order(self):
        summary = summarise_by_method(make_frame())
        assert list(summary["method"]) == ["random", "greedy_objective_aware", "exhaustive"]

    def test_grouping_adds_one_row_per_group(self):
        frame = make_frame()
        frame.loc[frame["scenario_seed"] == 2, "num_layers"] = 10
        summary = summarise_by_method(frame, ["objective"], group_columns=("num_layers",))
        assert len(summary) == 6
        assert set(summary["num_layers"]) == {5, 10}

    def test_empty_input_returns_empty_output(self):
        assert summarise_by_method(pd.DataFrame()).empty

    def test_table_formatting_is_readable(self):
        summary = summarise_by_method(make_frame(), ["objective"])
        text = format_summary_table(summary, ["objective"])
        assert "Random" in text
        assert "3.000" in text

    def test_labels_and_ordering_helpers(self):
        assert label_for("greedy_objective_aware") == "Greedy (objective-aware)"
        assert label_for("something_new") == "Something New"
        assert sort_methods(["exhaustive", "random"]) == ["random", "exhaustive"]


class TestStatistics:
    def test_mean_interval_brackets_the_mean(self):
        interval = mean_interval([1.0, 2.0, 3.0, 4.0])
        assert interval.estimate == pytest.approx(2.5)
        assert interval.low < interval.estimate < interval.high
        assert interval.n == 4

    def test_degenerate_samples_do_not_raise(self):
        assert mean_interval([]).n == 0
        single = mean_interval([3.0])
        assert single.low == single.high == 3.0
        constant = mean_interval([2.0, 2.0, 2.0])
        assert constant.low == constant.high == 2.0

    def test_bootstrap_interval_is_reproducible(self):
        values = list(np.random.default_rng(0).lognormal(size=200))
        first = bootstrap_interval(values, seed=7, resamples=500)
        second = bootstrap_interval(values, seed=7, resamples=500)
        assert first == second
        assert first.low < first.estimate < first.high

    def test_paired_comparison_detects_a_consistent_improvement(self):
        better = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        worse = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]
        result = paired_comparison(better, worse, "better", "worse")
        assert result.difference.estimate < 0
        assert result.win_rate == 1.0
        assert result.p_value < 0.05
        assert "better" in result.summary()

    def test_identical_samples_are_not_significant(self):
        values = [1.0, 2.0, 3.0]
        result = paired_comparison(values, values)
        assert result.p_value == 1.0
        assert result.win_rate == 0.0

    def test_misaligned_samples_raise(self):
        with pytest.raises(ValueError, match="aligned"):
            paired_comparison([1.0, 2.0], [1.0])


class TestResultsIO:
    def test_saving_raw_writes_data_and_provenance(self, tmp_path, monkeypatch):
        from src.config import load_config

        monkeypatch.setattr(results_io, "RAW_DIRECTORY", tmp_path / "raw")
        monkeypatch.setattr(results_io, "PROCESSED_DIRECTORY", tmp_path / "processed")
        monkeypatch.setattr(results_io, "FIGURES_DIRECTORY", tmp_path / "figures")

        path = results_io.save_raw(make_frame(), "unit_test", load_config(), panel="test")
        assert path.exists()
        provenance = json.loads(path.with_suffix(".meta.json").read_text())
        assert provenance["rows"] == 6
        assert provenance["panel"] == "test"
        assert "numpy" in provenance["libraries"]
        assert provenance["config"]["workload"]["num_layers"] == 10

    def test_loading_a_missing_file_explains_how_to_create_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(results_io, "RAW_DIRECTORY", tmp_path / "raw")
        with pytest.raises(FileNotFoundError, match="make experiments"):
            results_io.load_raw("does_not_exist")
