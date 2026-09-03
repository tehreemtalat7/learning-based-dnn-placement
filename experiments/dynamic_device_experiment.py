"""Experiment 4: behaviour as device load changes.

One device's background utilisation is swept through a series of levels while
everything else -- the workload, the network, the other devices -- is held fixed,
so the load is the only thing that varies. That is what
``environment.device_load_override`` exists for: it replaces the sampled
background load of a named device without disturbing the random stream any other
quantity is drawn from.

Loading a device does two things. It slows that device down directly, and it
raises the cost of every layer already placed there, because effective speed is
``capacity x (1 - utilisation)`` and utilisation also accumulates as work is
assigned. A method that keeps sending layers to a device that is already busy
therefore degrades twice over.

Every method observes the current utilisation of every device, so once again the
heuristics are reactive and no method has an information advantage.

Run it with::

    python -m experiments.dynamic_device_experiment
    python -m experiments.dynamic_device_experiment --device edge_server
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from experiments._shared import CHECKPOINTS, compare_methods, optional_agents
from src.config import config_summary, load_config
from src.environment.scenario import evaluation_seeds
from src.utils import visualization
from src.utils.io import save_processed, save_raw
from src.utils.metrics import format_summary_table, summarise_by_method

DEFAULT_LEVELS = (0.2, 0.5, 0.8)
DEFAULT_DEVICE = "gpu_server"

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
    "dp_relaxed",
)


def device_share(frame: pd.DataFrame, device_index: int) -> pd.Series:
    """Fraction of layers each method placed on one device."""
    def share(placement: str) -> float:
        devices = [int(part) for part in str(placement).split("-") if part != ""]
        return sum(1 for value in devices if value == device_index) / max(len(devices), 1)

    return frame["placement"].map(share)


def main() -> int:
    """Sweep one device's background load and compare every method at each level."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dynamic_device_load.yaml")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--levels", type=float, nargs="*", default=list(DEFAULT_LEVELS))
    parser.add_argument("--scenarios", type=int, default=None)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    arguments = parser.parse_args()

    print("=" * 78)
    print("EXPERIMENT 4: DYNAMIC DEVICE LOAD")
    print("=" * 78)
    print(f"sweeping {arguments.device} through {arguments.levels}")

    frames = []
    for level in arguments.levels:
        overrides = [
            *arguments.overrides,
            f"environment.device_load_override.{arguments.device}=[{level}, {level}]",
        ]
        config = load_config(arguments.config, overrides)
        seeds = evaluation_seeds(config, arguments.scenarios)
        device_index = config.device_index(arguments.device)
        agents = optional_agents(config, dqn_checkpoints={"dqn": CHECKPOINTS / "dqn.pt"})

        started = time.perf_counter()
        frame = compare_methods(config, seeds, extra_agents=agents)
        frame["load_level"] = level
        frame["loaded_device_share"] = device_share(frame, device_index)
        frames.append(frame)

        print(f"\n[{arguments.device} at {level:.0%} background load] "
              f"({time.perf_counter() - started:.1f}s)")
        print(config_summary(config))
        print(
            format_summary_table(
                summarise_by_method(
                    frame, (*REPORTED_METRICS, "loaded_device_share")
                ),
                ["objective", "total_latency_ms", "energy", "loaded_device_share"],
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    summary = summarise_by_method(
        combined, (*REPORTED_METRICS, "loaded_device_share"), group_columns=("load_level",)
    )
    save_raw(combined, "e4_device_load", config, device=arguments.device, levels=arguments.levels)
    save_processed(summary, "e4_device_load_summary")

    print(f"\nShare of layers still placed on {arguments.device} as it fills up")
    print("-" * 78)
    for method in sorted(set(combined["method"])):
        shares = [
            combined[
                (combined["method"] == method) & (combined["load_level"] == level)
            ]["loaded_device_share"].mean()
            for level in arguments.levels
        ]
        arrows = "  ->  ".join(f"{share:.0%}" for share in shares)
        print(f"  {method:<28} {arrows}")

    figures = [
        visualization.grouped_bars(
            summary,
            "load_level",
            "objective",
            "fig09_device_load",
            title=f"Placement quality as {arguments.device.replace('_', ' ')} load rises",
            subtitle="Weighted objective on identical held-out scenarios; lower is better",
            x_label=f"{arguments.device.replace('_', ' ')} background utilisation",
            methods=PLOTTED_METHODS,
            group_order=arguments.levels,
            value_format="{:,.3f}",
        ),
        visualization.grouped_bars(
            summary,
            "load_level",
            "loaded_device_share",
            "fig09b_device_load_share",
            title=f"Share of layers still sent to {arguments.device.replace('_', ' ')}",
            subtitle="A method that adapts should route work away as the device fills up",
            x_label=f"{arguments.device.replace('_', ' ')} background utilisation",
            methods=PLOTTED_METHODS,
            group_order=arguments.levels,
            value_format="{:,.2f}",
        ),
    ]
    print("\nFigures written:")
    for path in figures:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
