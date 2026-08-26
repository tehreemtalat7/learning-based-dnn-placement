"""Rollout and evaluation helpers shared by every experiment.

A single loop drives every method, so heuristics, the supervised baseline and
the learned policies are measured under identical conditions: the same scenario
seeds, the same environment, the same timing methodology.

Placement runtime is measured as the wall-clock time the *agent* spends deciding
(one timer around each ``act`` call), excluding the simulator's own bookkeeping.
That is the quantity that would matter in deployment, and it is what makes the
comparison against exhaustive search meaningful.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from src.agents.base import Agent
from src.config import Config
from src.environment.dnn_environment import DNNPlacementEnv
from src.environment.reward import PlacementResult
from src.environment.scenario import Scenario


@dataclass
class EpisodeRecord:
    """One episode's outcome, ready to be written to a CSV row."""

    method: str
    scenario_seed: int
    num_layers: int
    placement: tuple[int, ...]
    compute_latency_ms: float
    communication_latency_ms: float
    total_latency_ms: float
    energy: float
    objective: float
    memory_violations: int
    invalid_action_attempts: int
    device_switches: int
    episode_return: float
    decision_runtime_s: float
    congested: bool
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def decision_runtime_us_per_layer(self) -> float:
        """Mean time spent deciding one layer's placement, in microseconds."""
        return self.decision_runtime_s / max(self.num_layers, 1) * 1e6

    def to_row(self) -> dict[str, Any]:
        """Flatten into a mapping suitable for :class:`pandas.DataFrame`."""
        row: dict[str, Any] = {
            "method": self.method,
            "scenario_seed": self.scenario_seed,
            "num_layers": self.num_layers,
            "placement": "-".join(str(index) for index in self.placement),
            "compute_latency_ms": self.compute_latency_ms,
            "communication_latency_ms": self.communication_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "energy": self.energy,
            "objective": self.objective,
            "memory_violations": self.memory_violations,
            "invalid_action_attempts": self.invalid_action_attempts,
            "device_switches": self.device_switches,
            "episode_return": self.episode_return,
            "decision_runtime_s": self.decision_runtime_s,
            "decision_runtime_us_per_layer": self.decision_runtime_us_per_layer,
            "congested": self.congested,
        }
        row.update(self.extra)
        return row


def run_episode(
    env: DNNPlacementEnv,
    agent: Agent,
    *,
    scenario_seed: int | None = None,
    scenario: Scenario | None = None,
    num_layers: int | None = None,
    method_name: str | None = None,
) -> EpisodeRecord:
    """Run one agent on one scenario and record the outcome.

    Args:
        env: The environment to drive.
        agent: The placement method.
        scenario_seed: Seed of the scenario to solve.
        scenario: A ready-made scenario, taking precedence over ``scenario_seed``.
        num_layers: Optional depth override.
        method_name: Label for the record; defaults to ``agent.name``.

    Returns:
        An :class:`EpisodeRecord`.
    """
    options: dict[str, Any] = {}
    if scenario is not None:
        options["scenario"] = scenario
    elif scenario_seed is not None:
        options["scenario_seed"] = scenario_seed
    if num_layers is not None:
        options["num_layers"] = num_layers

    observation, info = env.reset(options=options)
    agent.reset()

    episode_return = 0.0
    decision_runtime_s = 0.0
    terminated = False
    while not terminated:
        started = time.perf_counter()
        action = agent.act(observation, info)
        decision_runtime_s += time.perf_counter() - started
        observation, reward, terminated, _truncated, info = env.step(action)
        episode_return += reward

    result: PlacementResult = info["result"]
    assert env.scenario is not None
    return EpisodeRecord(
        method=method_name if method_name is not None else agent.name,
        scenario_seed=env.scenario.seed,
        num_layers=result.num_layers,
        placement=result.placement,
        compute_latency_ms=result.compute_latency_ms,
        communication_latency_ms=result.communication_latency_ms,
        total_latency_ms=result.total_latency_ms,
        energy=result.energy,
        objective=result.objective,
        memory_violations=result.memory_violations,
        invalid_action_attempts=env.invalid_action_attempts,
        device_switches=result.device_switches,
        episode_return=episode_return,
        decision_runtime_s=decision_runtime_s,
        congested=env.scenario.has_congestion,
    )


def evaluate_agent(
    config: Config,
    agent: Agent,
    seeds: Sequence[int],
    *,
    num_layers: int | None = None,
    env: DNNPlacementEnv | None = None,
    method_name: str | None = None,
) -> list[EpisodeRecord]:
    """Evaluate one agent across a fixed set of scenario seeds.

    Args:
        config: The configuration in force.
        agent: The placement method.
        seeds: Scenario seeds; every method must be given the same list so that
            the comparison is paired.
        num_layers: Optional depth override applied to every scenario.
        env: Optional environment to reuse.
        method_name: Label for the records.

    Returns:
        One :class:`EpisodeRecord` per seed, in the order of ``seeds``.
    """
    environment = env if env is not None else DNNPlacementEnv(config, num_layers=num_layers)
    return [
        run_episode(
            environment,
            agent,
            scenario_seed=seed,
            num_layers=num_layers,
            method_name=method_name,
        )
        for seed in seeds
    ]


def evaluate_placement_function(
    config: Config,
    placement_fn: Callable[[Scenario], tuple[int, ...]],
    seeds: Sequence[int],
    method_name: str,
    *,
    num_layers: int | None = None,
) -> list[EpisodeRecord]:
    """Evaluate a method that computes a whole placement offline.

    Used by the exhaustive search and the dynamic-programming baseline, which
    solve the problem in one shot rather than layer by layer. The placement is
    scored with :func:`~src.environment.reward.evaluate_placement`, which applies
    exactly the same arithmetic as the environment but *counts* memory
    violations instead of rejecting them. That matters for the relaxed dynamic
    programme, whose solution can be infeasible once memory accumulation is
    switched on; rolling it through the masked environment would raise rather
    than report the problem.

    Args:
        config: The configuration in force.
        placement_fn: Maps a scenario to a complete placement.
        seeds: Scenario seeds.
        method_name: Label for the records.
        num_layers: Optional depth override.

    Returns:
        One :class:`EpisodeRecord` per seed.
    """
    from src.environment.reward import evaluate_placement
    from src.environment.scenario import sample_scenario  # local import avoids a cycle

    records = []
    for seed in seeds:
        scenario = sample_scenario(config, seed, num_layers=num_layers)
        started = time.perf_counter()
        placement = placement_fn(scenario)
        runtime_s = time.perf_counter() - started

        result = evaluate_placement(scenario, placement, config)
        records.append(
            EpisodeRecord(
                method=method_name,
                scenario_seed=scenario.seed,
                num_layers=result.num_layers,
                placement=result.placement,
                compute_latency_ms=result.compute_latency_ms,
                communication_latency_ms=result.communication_latency_ms,
                total_latency_ms=result.total_latency_ms,
                energy=result.energy,
                objective=result.objective,
                memory_violations=result.memory_violations,
                invalid_action_attempts=result.memory_violations,
                device_switches=result.device_switches,
                episode_return=-result.objective,
                decision_runtime_s=runtime_s,
                congested=scenario.has_congestion,
            )
        )
    return records


__all__ = [
    "EpisodeRecord",
    "evaluate_agent",
    "evaluate_placement_function",
    "run_episode",
]
