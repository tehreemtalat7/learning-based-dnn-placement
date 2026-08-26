"""Exhaustive search over every possible placement.

Evaluates all ``D**L`` device assignments under the *full* environment dynamics,
including memory and utilisation accumulation, and returns the cheapest feasible
one. It is therefore the true optimum -- but only where it is affordable.

This is the same baseline the previous project used, kept for two reasons:

1. It validates :mod:`src.baselines.dp_optimal`. A unit test asserts the two
   agree exactly whenever the dynamic programme is exact, which is a much
   stronger check on the dynamic programme than reading its recursion.
2. Where accumulation is enabled the dynamic programme only returns a lower
   bound, so exhaustive search supplies the true optimum for the small DNNs on
   which it can still be run -- which in turn lets the experiments *measure* how
   tight that bound is instead of assuming it.

The cost grows as ``D**L``: with four devices that is 1 024 placements at five
layers, about a million at ten, and a billion at fifteen. The search refuses to
start above ``experiment.max_exhaustive_combinations`` rather than appearing to
hang, and the experiments simply omit it beyond that point.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from src.config import Config
from src.environment.reward import evaluate_placement
from src.environment.scenario import Scenario


class SearchTooLargeError(RuntimeError):
    """Raised when the requested exhaustive search exceeds the configured budget."""


@dataclass(frozen=True)
class ExhaustiveSolution:
    """The best placement found, and how much work finding it took."""

    placement: tuple[int, ...]
    objective: float
    evaluated: int
    feasible_found: int


def count_placements(num_layers: int, num_devices: int) -> int:
    """Number of candidate assignments, ignoring feasibility."""
    if num_layers < 0:
        raise ValueError("num_layers must be non-negative")
    if num_devices <= 0:
        raise ValueError("num_devices must be positive")
    return num_devices**num_layers


def is_affordable(num_layers: int, num_devices: int, budget: int) -> bool:
    """Whether an exhaustive search of this size is within the budget."""
    return count_placements(num_layers, num_devices) <= budget


def solve(
    scenario: Scenario, config: Config, budget: int | None = None
) -> ExhaustiveSolution:
    """Find the cheapest feasible placement by brute force.

    Args:
        scenario: The scenario to solve.
        config: The configuration in force.
        budget: Maximum number of candidates to enumerate; defaults to
            ``experiment.max_exhaustive_combinations``.

    Returns:
        An :class:`ExhaustiveSolution`.

    Raises:
        SearchTooLargeError: If the search space exceeds the budget.
        ValueError: If no feasible placement exists.
    """
    effective_budget = (
        budget if budget is not None else config.experiment.max_exhaustive_combinations
    )
    total = count_placements(scenario.num_layers, scenario.num_devices)
    if total > effective_budget:
        raise SearchTooLargeError(
            f"exhaustive search over {total:,} placements exceeds the budget of "
            f"{effective_budget:,} ({scenario.num_layers} layers, "
            f"{scenario.num_devices} devices). Use the dynamic-programming baseline instead."
        )

    best_placement: tuple[int, ...] | None = None
    best_objective = float("inf")
    feasible_found = 0

    for candidate in itertools.product(range(scenario.num_devices), repeat=scenario.num_layers):
        result = evaluate_placement(scenario, candidate, config)
        if result.memory_violations:
            continue
        feasible_found += 1
        if result.objective < best_objective:
            best_objective = result.objective
            best_placement = candidate

    if best_placement is None:
        raise ValueError(f"no feasible placement exists for scenario {scenario.seed}")

    return ExhaustiveSolution(
        placement=best_placement,
        objective=best_objective,
        evaluated=total,
        feasible_found=feasible_found,
    )


def exhaustive_optimal_placement(scenario: Scenario, config: Config) -> tuple[int, ...]:
    """Convenience wrapper returning only the placement."""
    return solve(scenario, config).placement


__all__ = [
    "ExhaustiveSolution",
    "SearchTooLargeError",
    "count_placements",
    "exhaustive_optimal_placement",
    "is_affordable",
    "solve",
]
