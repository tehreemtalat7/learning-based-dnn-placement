"""Run every experiment in order, or rebuild the figures from existing CSVs.

``make experiments`` runs this; ``make figures`` runs it with ``--figures-only``,
which regenerates every plot from the raw CSVs without re-running any
computation. That split is what makes the claim "no figure is hand-authored"
checkable: delete ``results/figures/`` and rebuild it from the data alone.

Experiments that need a trained agent report which checkpoint is missing rather
than failing, so a fresh checkout produces a partial but clearly-labelled set of
results.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from src.config import REPOSITORY_ROOT
from src.utils import visualization
from src.utils.io import load_raw, save_processed
from src.utils.metrics import summarise_by_method

# Experiments run in this order; each is a module executed with `python -m`.
EXPERIMENTS: tuple[tuple[str, list[str]], ...] = (
    ("experiments.static_experiment", []),
    ("experiments.dqn_seed_spread", []),
    ("experiments.scaling_experiment", []),
    ("experiments.dynamic_network_experiment", []),
    ("experiments.dynamic_device_experiment", []),
)


def run_module(module: str, arguments: list[str]) -> bool:
    """Run one experiment module and report whether it succeeded."""
    print("\n" + "=" * 78)
    print(f"RUNNING {module}")
    print("=" * 78)
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", module, *arguments], cwd=REPOSITORY_ROOT, check=False
    )
    elapsed = time.perf_counter() - started
    status = "ok" if completed.returncode == 0 else f"FAILED ({completed.returncode})"
    print(f"\n{module}: {status} in {elapsed:.1f}s")
    return completed.returncode == 0


def rebuild_figures() -> list[str]:
    """Regenerate every figure from the CSVs already in ``results/raw``."""
    from experiments.static_experiment import SMALL_PANEL_LAYERS, REPORTED_METRICS, make_figures

    written: list[str] = []

    main_frame = load_raw("e1_static_main")
    small_frame = load_raw("e1_static_small")
    main_summary = summarise_by_method(main_frame, REPORTED_METRICS)
    small_summary = summarise_by_method(small_frame, REPORTED_METRICS)
    save_processed(main_summary, "e1_static_main_summary")
    save_processed(small_summary, "e1_static_small_summary")
    written.extend(make_figures(main_summary, small_summary))

    for tag in ("static",):
        try:
            curves = load_raw(f"e1_dqn_curves_{tag}")
        except FileNotFoundError:
            print(f"  no DQN curves for tag {tag!r}; skipping its training figures")
            continue
        written.append(
            str(
                visualization.training_curve(
                    curves,
                    f"fig01_dqn_training_return_{tag}",
                    title="DQN training return",
                    subtitle="Return equals the negative weighted objective, so higher is better",
                    x_column="step",
                    value_column="episode_return",
                    smoothed_column="moving_average_return",
                    x_label="Environment steps",
                )
            )
        )
        written.append(
            str(
                visualization.training_curve(
                    curves,
                    f"fig02_dqn_validation_return_{tag}",
                    title="DQN return on held-out validation scenarios",
                    subtitle="Measured during training on scenarios never used for gradient updates",
                    x_column="step",
                    value_column="validation_return",
                    smoothed_column="validation_return",
                    x_label="Environment steps",
                    y_label="Validation return",
                )
            )
        )

    for name, builder in (
        ("e2_scaling", _rebuild_scaling_figures),
        ("e3_dynamic_network", _rebuild_network_figures),
        ("e4_device_load", _rebuild_device_load_figures),
    ):
        try:
            frame = load_raw(name)
        except FileNotFoundError:
            print(f"  no {name} results; skipping its figures")
            continue
        written.extend(builder(frame))

    for mode in ("single_scenario", "pooled"):
        try:
            curve = load_raw(f"e1_tabular_curve_{mode}")
        except FileNotFoundError:
            print(f"  no tabular curve for mode {mode!r}; skipping its figure")
            continue
        written.append(
            str(
                visualization.training_curve(
                    curve,
                    f"fig11_tabular_training_{mode}",
                    title=f"Tabular Q-learning: {mode.replace('_', ' ')}",
                    subtitle="Return equals the negative weighted objective, so higher is better",
                )
            )
        )

    return written


def _rebuild_scaling_figures(frame) -> list[str]:
    """Rebuild the depth-scaling figures from saved rows."""
    from experiments.scaling_experiment import PLOTTED_METHODS, SCALING_METRICS

    summary = summarise_by_method(frame, SCALING_METRICS, group_columns=("num_layers",))
    save_processed(summary, "e2_scaling_summary")
    return [
        str(
            visualization.lines_by_x(
                summary, "num_layers", "objective", "fig07_objective_vs_depth",
                title="Placement quality as the DNN gets deeper",
                subtitle="Weighted objective on held-out scenarios, log scale; "
                "1.0 is the cost of random placement",
                x_label="Number of DNN layers", methods=PLOTTED_METHODS, log_y=True,
            )
        ),
        str(
            visualization.lines_by_x(
                summary, "num_layers", "gap_vs_best_pct", "fig07b_gap_vs_depth",
                title="Distance from the best placement found, by depth",
                subtitle="Log scale; lower is better",
                x_label="Number of DNN layers",
                methods=[m for m in PLOTTED_METHODS if m != "random"], log_y=True,
            )
        ),
        str(
            visualization.lines_by_x(
                summary, "num_layers", "decision_runtime_s", "fig07c_runtime_vs_depth",
                title="Time to place a whole DNN",
                subtitle="Log scale. Exhaustive search stops where it is no longer affordable",
                x_label="Number of DNN layers",
                methods=(*PLOTTED_METHODS, "exhaustive"), log_y=True,
            )
        ),
    ]


def _rebuild_network_figures(frame) -> list[str]:
    """Rebuild the network-regime figure from saved rows."""
    from experiments.dynamic_network_experiment import (
        PLOTTED_METHODS,
        REGIMES,
        REPORTED_METRICS,
    )

    summary = summarise_by_method(frame, REPORTED_METRICS, group_columns=("regime",))
    save_processed(summary, "e3_dynamic_network_summary")
    return [
        str(
            visualization.grouped_bars(
                summary, "regime", "objective", "fig08_network_conditions",
                title="Placement quality under three network regimes",
                subtitle="Weighted objective on identical held-out scenarios; lower is better",
                x_label="Network regime", methods=PLOTTED_METHODS,
                group_order=REGIMES, value_format="{:,.3f}",
            )
        )
    ]


def _rebuild_device_load_figures(frame) -> list[str]:
    """Rebuild the device-load figures from saved rows."""
    from experiments.dynamic_device_experiment import PLOTTED_METHODS, REPORTED_METRICS

    levels = sorted(set(frame["load_level"]))
    summary = summarise_by_method(
        frame, (*REPORTED_METRICS, "loaded_device_share"), group_columns=("load_level",)
    )
    save_processed(summary, "e4_device_load_summary")
    return [
        str(
            visualization.grouped_bars(
                summary, "load_level", "objective", "fig09_device_load",
                title="Placement quality as device load rises",
                subtitle="Weighted objective on identical held-out scenarios; lower is better",
                x_label="Background utilisation of the loaded device",
                methods=PLOTTED_METHODS, group_order=levels, value_format="{:,.3f}",
            )
        ),
        str(
            visualization.grouped_bars(
                summary, "load_level", "loaded_device_share", "fig09b_device_load_share",
                title="Share of layers still sent to the loaded device",
                subtitle="A method that adapts should route work away as the device fills up",
                x_label="Background utilisation of the loaded device",
                methods=PLOTTED_METHODS, group_order=levels, value_format="{:,.2f}",
            )
        ),
    ]


def main() -> int:
    """Run every experiment, or only rebuild figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="rebuild figures from results/raw without running any experiment",
    )
    parser.add_argument(
        "--quick", action="store_true", help="fewer scenarios, for a fast smoke run"
    )
    arguments = parser.parse_args()

    if arguments.figures_only:
        print("Rebuilding figures from existing CSVs")
        for path in rebuild_figures():
            print(f"  {path}")
        return 0

    failures = []
    for module, args in EXPERIMENTS:
        extra = list(args)
        if arguments.quick:
            extra += ["--scenarios", "50"]
            if module.endswith("static_experiment"):
                extra.append("--skip-bound")
            if module.endswith("scaling_experiment"):
                extra += ["--depths", "5", "10"]
        if not run_module(module, extra):
            failures.append(module)

    print("\n" + "=" * 78)
    if failures:
        print(f"{len(failures)} experiment(s) failed: {failures}")
        return 1
    print(f"all {len(EXPERIMENTS)} experiments completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
