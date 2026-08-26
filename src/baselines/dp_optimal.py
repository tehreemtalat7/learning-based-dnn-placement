"""Exact placement optimum by dynamic programming.

The previous project could only obtain an optimal reference by brute force over
all ``D**L`` assignments, which confined optimality gaps to five-layer DNNs.
This module removes that limitation.

**The structure that makes it work.** The cost of placing layer ``i`` depends on
the layer itself, on the device chosen for it, and -- through the activation
transfer -- on the device chosen for layer ``i-1``. Nothing else. A placement is
therefore a path through a layered graph whose nodes are ``(layer, device)``
pairs, and the cheapest placement is a shortest path, computable by the usual
Viterbi recursion in ``O(L * D^2)``::

    V[0][d]  = cost(layer 0 on d, arriving from the input source device)
    V[i][d]  = min over p of  V[i-1][p] + cost(layer i on d, arriving from p)

**When it is exact, and what it is otherwise.** That decomposition holds as long
as placing a layer does not change the cost of later layers. Two configured
effects break it: memory accumulation (an early choice can make a device too
full later) and utilisation accumulation (an early choice can slow a device
down). When both are disabled, this recursion returns the true optimum, and
:mod:`tests.test_baselines` asserts it agrees with exhaustive search.

When either is enabled, the recursion is solved on the *relaxed* problem in which
neither effect applies. Relaxing them can only make placements cheaper (devices
stay at full speed) and can only admit more placements (devices never fill up),
so the value returned is a **certified lower bound** on the true optimum. It is
reported and plotted as a lower bound, never as "the optimum", and the resulting
optimality gaps are conservative -- they overstate how far a method is from
optimal rather than flattering it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from src.config import Config
from src.environment.devices import DeviceFleet
from src.environment.reward import step_costs, weighted_cost
from src.environment.scenario import Scenario

INFEASIBLE = float("inf")


@dataclass(frozen=True)
class DPSolution:
    """The result of the dynamic programme.

    Attributes:
        placement: The cheapest path found, one device index per layer.
        objective: Its weighted objective *under the relaxed dynamics*. Equal to
            the true optimum when :attr:`is_exact` is true, and a lower bound on
            it otherwise.
        is_exact: Whether the configuration makes the recursion exact.
    """

    placement: tuple[int, ...]
    objective: float
    is_exact: bool

    @property
    def label(self) -> str:
        """How this value should be described in a table or figure."""
        return "optimum" if self.is_exact else "lower bound"


def relaxed_environment(config: Config) -> Config:
    """Return a copy of the configuration with both accumulation effects off.

    This is the relaxation the dynamic programme solves. Building it explicitly
    keeps the approximation visible instead of hiding it inside the solver.
    """
    environment = dataclasses.replace(
        config.environment,
        memory_accumulates=False,
        utilisation_accumulates=False,
    )
    return dataclasses.replace(config, environment=environment)


def is_exact_for(config: Config) -> bool:
    """Whether the dynamic programme returns the true optimum for this configuration."""
    return not (
        config.environment.memory_accumulates or config.environment.utilisation_accumulates
    )


def transition_costs(
    scenario: Scenario, config: Config, layer_index: int, fleet: DeviceFleet
) -> tuple[np.ndarray, np.ndarray]:
    """Cost of every ``(previous device, chosen device)`` pair for one layer.

    Args:
        scenario: The scenario being solved.
        config: The (already relaxed) configuration.
        layer_index: Index of the layer being placed.
        fleet: A fleet in its initial state, shared across layers because the
            relaxation keeps device state constant.

    Returns:
        A tuple ``(costs, feasible)``. ``costs`` has shape ``(D, D)`` with
        ``costs[p, d]`` the weighted cost of placing this layer on ``d`` when the
        activation arrives from ``p``. ``feasible`` has shape ``(D,)``.
    """
    layer = scenario.workload[layer_index]
    payload_mb = scenario.payload_before(layer_index)
    num_devices = len(fleet)

    costs = np.empty((num_devices, num_devices), dtype=np.float64)
    for previous in range(num_devices):
        execution_ms, communication_ms, energy = step_costs(
            fleet=fleet,
            network=scenario.network,
            layer_index=layer_index,
            layer_compute_cost=layer.compute_cost,
            payload_mb=payload_mb,
            previous_device_index=previous,
            environment=config.environment,
        )
        for chosen in range(num_devices):
            costs[previous, chosen] = weighted_cost(
                latency_ms=float(execution_ms[chosen] + communication_ms[chosen]),
                communication_ms=float(communication_ms[chosen]),
                energy=float(energy[chosen]),
                references=scenario.references,
                weights=scenario.weights,
                comm_double_count=config.objective.comm_double_count,
            )

    return costs, fleet.feasibility_mask(layer.memory_gb)


def solve(scenario: Scenario, config: Config) -> DPSolution:
    """Find the cheapest placement by dynamic programming.

    Args:
        scenario: The scenario to solve.
        config: The configuration in force. It is relaxed internally; the caller
            passes the real configuration.

    Returns:
        A :class:`DPSolution`.

    Raises:
        ValueError: If no feasible placement exists, which the workload
            generator's memory cap is designed to prevent.
    """
    exact = is_exact_for(config)
    relaxed = relaxed_environment(config)
    fleet = DeviceFleet(scenario.devices, relaxed.environment)

    depth = scenario.num_layers
    num_devices = len(fleet)

    values = np.full(num_devices, INFEASIBLE, dtype=np.float64)
    backpointers = np.zeros((depth, num_devices), dtype=np.int64)

    for layer_index in range(depth):
        costs, feasible = transition_costs(scenario, relaxed, layer_index, fleet)
        if not feasible.any():
            raise ValueError(
                f"layer {scenario.workload[layer_index].name!r} fits on no device in "
                f"scenario {scenario.seed}"
            )

        if layer_index == 0:
            candidate = costs[scenario.input_source_index].copy()
            backpointers[0, :] = scenario.input_source_index
        else:
            # candidate[d] = min over p of values[p] + costs[p, d]
            totals = values[:, None] + costs
            best_previous = np.argmin(totals, axis=0)
            candidate = totals[best_previous, np.arange(num_devices)]
            backpointers[layer_index, :] = best_previous

        candidate = np.where(feasible, candidate, INFEASIBLE)
        values = candidate

    final_device = int(np.argmin(values))
    if not np.isfinite(values[final_device]):
        raise ValueError(f"no feasible placement exists for scenario {scenario.seed}")

    placement = [0] * depth
    placement[depth - 1] = final_device
    for layer_index in range(depth - 1, 0, -1):
        placement[layer_index - 1] = int(backpointers[layer_index, placement[layer_index]])

    return DPSolution(
        placement=tuple(placement),
        objective=float(values[final_device]),
        is_exact=exact,
    )


def dp_optimal_placement(scenario: Scenario, config: Config) -> tuple[int, ...]:
    """Convenience wrapper returning only the placement."""
    return solve(scenario, config).placement


__all__ = [
    "DPSolution",
    "dp_optimal_placement",
    "is_exact_for",
    "relaxed_environment",
    "solve",
    "transition_costs",
]
