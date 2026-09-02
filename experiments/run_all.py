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

    extra = ["--scenarios", "50", "--skip-bound"] if arguments.quick else []
    failures = [module for module, args in EXPERIMENTS if not run_module(module, args + extra)]

    print("\n" + "=" * 78)
    if failures:
        print(f"{len(failures)} experiment(s) failed: {failures}")
        return 1
    print(f"all {len(EXPERIMENTS)} experiments completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
