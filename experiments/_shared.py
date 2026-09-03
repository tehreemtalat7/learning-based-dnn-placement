"""Helpers shared by the Phase 5 experiment scripts.

Each experiment varies one thing -- DNN depth, network conditions, device load --
and holds everything else fixed, so they all need the same three pieces: build
the comparison set of agents, roll them over identical held-out scenarios, and
attach the exact-method references. Keeping that here means the three scripts
differ only in what they vary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.agents import build_heuristic_agents
from src.agents.dqn_agent import DQNAgent
from src.agents.q_learning_agent import TabularQAgent
from src.baselines import dp_optimal, exhaustive_search
from src.baselines.supervised_ml import SupervisedAgent, load_model
from src.config import Config
from src.training.evaluate import evaluate_agent, evaluate_placement_function
from src.utils.metrics import add_gap_vs_best, records_to_frame

CHECKPOINTS = Path("checkpoints")


def optional_agents(
    config: Config,
    *,
    dqn_checkpoints: dict[str, Path] | None = None,
    include_supervised: bool = True,
    include_tabular: bool = True,
) -> list:
    """Load whichever trained agents exist, naming any that are missing.

    Args:
        config: The configuration in force.
        dqn_checkpoints: Mapping of method name to checkpoint path. Each is
            loaded if present.
        include_supervised: Whether to load the Random Forest baseline.
        include_tabular: Whether to load the pooled tabular table.

    Returns:
        The agents that could be loaded.
    """
    agents = []

    for name, path in (dqn_checkpoints or {}).items():
        if Path(path).exists():
            agents.append(DQNAgent.load(Path(path), config.dqn, name=name))
        else:
            print(f"  {name}: no checkpoint at {path}, skipping")

    if include_supervised:
        supervised = CHECKPOINTS / "supervised_rf.joblib"
        if supervised.exists():
            agents.append(SupervisedAgent(load_model(supervised), config.num_devices))
        else:
            print(f"  supervised_rf: no model at {supervised}, skipping")

    if include_tabular:
        table = CHECKPOINTS / "q_table_pooled.joblib"
        if table.exists():
            agent = TabularQAgent.load(table)
            agent.name = "tabular_q_pooled"
            agents.append(agent)
        else:
            print(f"  tabular_q_pooled: no table at {table}, skipping")

    return agents


def compare_methods(
    config: Config,
    seeds: list[int],
    *,
    num_layers: int | None = None,
    extra_agents: list | None = None,
    include_exhaustive: bool = False,
    include_dp: bool = True,
) -> pd.DataFrame:
    """Roll every method over identical scenarios and return tidy rows.

    Args:
        config: The configuration in force.
        seeds: Held-out scenario seeds, the same for every method.
        num_layers: Depth override for this panel.
        extra_agents: Learned agents to include alongside the heuristics.
        include_exhaustive: Whether brute force is affordable here.
        include_dp: Whether to include the dynamic-programming placement.

    Returns:
        A frame with one row per (method, scenario), carrying a per-scenario gap
        against the best placement any method found.
    """
    records = []
    for agent in [*build_heuristic_agents(config), *(extra_agents or [])]:
        records.extend(evaluate_agent(config, agent, seeds, num_layers=num_layers))

    if include_dp:
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

    return add_gap_vs_best(records_to_frame(records))


def relative_to(frame: pd.DataFrame, baseline: str, column: str = "objective") -> pd.DataFrame:
    """Add each method's percentage difference from a baseline, per scenario."""
    reference = (
        frame.loc[frame["method"] == baseline, ["scenario_seed", column]]
        .rename(columns={column: "_baseline"})
        .drop_duplicates(subset=["scenario_seed"])
    )
    merged = frame.merge(reference, on="scenario_seed", how="left")
    merged[f"vs_{baseline}_pct"] = (
        (merged[column] - merged["_baseline"]) / merged["_baseline"] * 100.0
    )
    return merged.drop(columns=["_baseline"])


__all__ = ["CHECKPOINTS", "compare_methods", "optional_agents", "relative_to"]
