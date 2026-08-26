"""Phase 1 gate: check the simulator before any learning code touches it.

Runs the non-learning agents over a sample of scenarios and asserts the
invariants the rest of the project depends on:

1. **Return equals the negative objective.** Summing per-step rewards must
   reproduce the weighted objective of the finished placement, otherwise the
   training signal and the evaluation metric are different quantities.
2. **The environment and the offline evaluator agree.** Rolling an episode must
   produce exactly what :func:`~src.environment.reward.evaluate_placement`
   computes for the same placement, so the search baselines and the agents are
   scored identically.
3. **Masking guarantees feasibility.** No memory violation may occur while
   action masking is enabled.
4. **The normalisation is calibrated.** Random placement should score near 1.0,
   because the cost references are the expected cost of random placement.
5. **The heuristics are ordered sensibly.** Objective-aware greedy should beat
   random placement on the objective it optimises.
6. **Scenarios are reproducible.** The same seed must rebuild the same problem
   and produce the same numbers.

Run it with::

    python -m experiments.validate_environment
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.round_robin_agent import RoundRobinAgent
from src.config import Config, config_summary, load_config
from src.environment.dnn_environment import DNNPlacementEnv
from src.environment.reward import evaluate_placement
from src.environment.scenario import evaluation_seeds, sample_scenario, summarise_scenarios
from src.training.evaluate import EpisodeRecord, evaluate_agent

TOLERANCE = 1e-6


def build_agents(config: Config) -> list:
    """Instantiate every non-learning agent."""
    num_actions = config.num_devices
    return [
        RandomAgent(num_actions, seed=config.seed),
        RoundRobinAgent(num_actions),
        GreedyAgent(num_actions, "fastest_device"),
        GreedyAgent(num_actions, "communication_aware"),
        GreedyAgent(num_actions, "objective_aware"),
    ]


def describe_scenarios(config: Config, seeds: list[int]) -> None:
    """Print what the sampled scenarios look like."""
    scenarios = [sample_scenario(config, seed) for seed in seeds]
    print("\nScenario sample")
    print("-" * 78)
    for scenario in scenarios[:3]:
        print(f"  {scenario.describe()}")
        for device in scenario.devices:
            print(
                f"     {device.name:<13} capacity={device.compute_capacity:6.2f} "
                f"memory={device.memory_gb:6.2f} GB  energy/compute={device.energy_per_compute:4.2f} "
                f"base_load={device.base_utilisation:4.2f}"
            )
        print(
            f"     layers: compute={np.round(scenario.workload.compute_costs, 2)}"
        )
        print(
            f"             memory={np.round(scenario.workload.memory_requirements_gb, 2)} GB"
        )
        print(
            f"             output={np.round(scenario.workload.output_sizes_mb, 2)} MB"
        )
        print(
            f"     references: latency={scenario.references.latency_ms:9.1f} ms  "
            f"energy={scenario.references.energy:7.2f}  "
            f"communication={scenario.references.communication_ms:9.1f} ms"
        )
    summary = summarise_scenarios(scenarios)
    print("\n  Aggregate over the sample:")
    for key, value in summary.items():
        print(f"     {key:<32} {value:,.4f}")


def check_return_matches_objective(records: list[EpisodeRecord]) -> list[str]:
    """Verify invariant 1 for every episode."""
    failures = []
    for record in records:
        difference = abs(record.episode_return + record.objective)
        if difference > TOLERANCE * max(1.0, abs(record.objective)):
            failures.append(
                f"{record.method} scenario {record.scenario_seed}: "
                f"return {record.episode_return:.9f} != -objective {-record.objective:.9f}"
            )
    return failures


def check_evaluator_agrees(config: Config, records: list[EpisodeRecord]) -> list[str]:
    """Verify invariant 2 by re-scoring each placement offline."""
    failures = []
    for record in records:
        scenario = sample_scenario(config, record.scenario_seed, num_layers=record.num_layers)
        placement = tuple(int(part) for part in record.placement)
        offline = evaluate_placement(scenario, placement, config)
        for label, rolled, computed in (
            ("latency", record.total_latency_ms, offline.total_latency_ms),
            ("energy", record.energy, offline.energy),
            ("objective", record.objective, offline.objective),
        ):
            if abs(rolled - computed) > TOLERANCE * max(1.0, abs(computed)):
                failures.append(
                    f"{record.method} scenario {record.scenario_seed}: environment {label} "
                    f"{rolled:.9f} != evaluator {computed:.9f}"
                )
    return failures


def check_masking(records: list[EpisodeRecord], masking_enabled: bool) -> list[str]:
    """Verify invariant 3."""
    if not masking_enabled:
        return []
    return [
        f"{record.method} scenario {record.scenario_seed}: "
        f"{record.memory_violations} memory violations under action masking"
        for record in records
        if record.memory_violations > 0
    ]


def check_reproducibility(config: Config, seeds: list[int]) -> list[str]:
    """Verify invariant 6 by replaying the first scenarios."""
    failures = []
    for seed in seeds[:20]:
        first = sample_scenario(config, seed)
        second = sample_scenario(config, seed)
        if first.workload.compute_costs.tolist() != second.workload.compute_costs.tolist():
            failures.append(f"scenario {seed}: workload differs between two samplings")
        if first.devices != second.devices:
            failures.append(f"scenario {seed}: devices differ between two samplings")

        agent = GreedyAgent(config.num_devices, "objective_aware")
        environment = DNNPlacementEnv(config)
        from src.training.evaluate import run_episode

        one = run_episode(environment, agent, scenario_seed=seed)
        two = run_episode(environment, agent, scenario_seed=seed)
        if abs(one.objective - two.objective) > TOLERANCE:
            failures.append(f"scenario {seed}: repeated rollout produced a different objective")
    return failures


def main() -> int:
    """Run the validation harness and report pass/fail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="experiment configuration file")
    parser.add_argument("--scenarios", type=int, default=100, help="number of scenarios to check")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="configuration override, e.g. workload.num_layers=5",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config, arguments.overrides)
    seeds = evaluation_seeds(config, arguments.scenarios)

    print("=" * 78)
    print("ENVIRONMENT VALIDATION")
    print("=" * 78)
    print(config_summary(config))
    describe_scenarios(config, seeds)

    print("\nAgent comparison over "
          f"{len(seeds)} held-out scenarios ({config.workload.num_layers} layers)")
    print("-" * 78)
    header = (
        f"{'method':<28}{'objective':>11}{'latency ms':>13}"
        f"{'energy':>10}{'comm ms':>11}{'switches':>10}"
    )
    print(header)

    all_records: list[EpisodeRecord] = []
    objectives: dict[str, float] = {}
    for agent in build_agents(config):
        records = evaluate_agent(config, agent, seeds)
        all_records.extend(records)
        objective = float(np.mean([record.objective for record in records]))
        objectives[agent.name] = objective
        print(
            f"{agent.name:<28}{objective:>11.4f}"
            f"{np.mean([r.total_latency_ms for r in records]):>13.1f}"
            f"{np.mean([r.energy for r in records]):>10.2f}"
            f"{np.mean([r.communication_latency_ms for r in records]):>11.1f}"
            f"{np.mean([r.device_switches for r in records]):>10.2f}"
        )

    print("\nInvariant checks")
    print("-" * 78)
    failures: list[str] = []
    checks = {
        "return == -objective": check_return_matches_objective(all_records),
        "environment == offline evaluator": check_evaluator_agrees(config, all_records),
        "masking prevents violations": check_masking(
            all_records, config.environment.uses_action_masking
        ),
        "scenarios are reproducible": check_reproducibility(config, seeds),
    }

    random_objective = objectives["random"]
    if not 0.5 <= random_objective <= 1.8:
        checks["random placement scores near 1.0"] = [
            f"random placement scored {random_objective:.3f}, expected roughly 1.0; "
            "the cost references may be miscalibrated"
        ]
    else:
        checks["random placement scores near 1.0"] = []

    if objectives["greedy_objective_aware"] >= random_objective:
        checks["objective-aware greedy beats random"] = [
            f"greedy scored {objectives['greedy_objective_aware']:.3f} vs "
            f"random {random_objective:.3f}"
        ]
    else:
        checks["objective-aware greedy beats random"] = []

    for label, problems in checks.items():
        status = "PASS" if not problems else f"FAIL ({len(problems)})"
        print(f"  {label:<40} {status}")
        for problem in problems[:5]:
            print(f"      - {problem}")
        failures.extend(problems)

    print("-" * 78)
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} problem(s)")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
