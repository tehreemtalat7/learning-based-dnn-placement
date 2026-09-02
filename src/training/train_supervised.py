"""Train the Random Forest imitation baseline.

Collects oracle demonstrations on training scenarios, fits the classifier, and
reports how faithfully it reproduces the oracle *and*, more importantly, what
its placements actually cost when rolled out on held-out scenarios.

Those two numbers usually disagree, and the gap between them is the point:
imitation accuracy measures agreement with a teacher, while the objective
measures the quality of the placements the agent actually produces. A classifier
that is right 80 % of the time can still be expensive if its mistakes fall on the
layers that matter.

Run it with::

    python -m src.training.train_supervised
    python -m src.training.train_supervised --scenarios 500   # quick
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from src.baselines.supervised_ml import (
    DEFAULT_MODEL_PATH,
    SupervisedAgent,
    collect_demonstrations,
    evaluate_imitation_accuracy,
    save_model,
    train_random_forest,
)
from src.config import config_summary, load_config
from src.environment.scenario import evaluation_seeds, training_seeds
from src.training.evaluate import evaluate_agent
from src.utils.seed import make_rng


def main() -> int:
    """Collect demonstrations, fit the classifier and report its quality."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="static_environment.yaml")
    parser.add_argument(
        "--scenarios", type=int, default=None, help="training scenarios to collect"
    )
    parser.add_argument("--layers", type=int, default=None, help="DNN depth override")
    parser.add_argument("--seed", type=int, default=0, help="seed for the forest")
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config, arguments.overrides)
    count = arguments.scenarios or config.supervised.n_training_scenarios
    seeds = training_seeds(config, count, make_rng(config.seed, "supervised_scenarios"))

    print("=" * 78)
    print("TRAINING THE SUPERVISED BASELINE")
    print("=" * 78)
    print(config_summary(config))
    print(f"teacher: {config.supervised.teacher}, training scenarios: {count}")

    started = time.perf_counter()
    demonstrations = collect_demonstrations(config, seeds, num_layers=arguments.layers)
    collection_time = time.perf_counter() - started
    print(
        f"\ncollected {len(demonstrations):,} (state, action) pairs from "
        f"{count:,} scenarios in {collection_time:.1f}s"
    )
    print("teacher sources:")
    for source, occurrences in sorted(demonstrations.teacher_sources.items()):
        print(f"  {source:<28} {occurrences:>6} scenarios ({occurrences / count:.1%})")

    started = time.perf_counter()
    model = train_random_forest(demonstrations, config, seed=arguments.seed)
    print(f"\nfitted {config.supervised.n_estimators} trees in {time.perf_counter() - started:.1f}s")

    training_accuracy = evaluate_imitation_accuracy(model, demonstrations)
    print("\nAgreement with the oracle on the training trajectories")
    print(f"  per-layer accuracy        {training_accuracy['per_layer_accuracy']:.4f}")
    print(f"  exact placement accuracy  {training_accuracy['exact_placement_accuracy']:.4f}")

    # Held-out demonstrations measure agreement on states the oracle visits;
    # the rollout below measures what the agent's own placements actually cost.
    held_out_seeds = evaluation_seeds(config, min(config.experiment.n_eval_scenarios, 150))
    held_out = collect_demonstrations(
        config, held_out_seeds, num_layers=arguments.layers, progress_every=0
    )
    held_out_accuracy = evaluate_imitation_accuracy(model, held_out)
    print("\nAgreement with the oracle on held-out scenarios")
    print(f"  per-layer accuracy        {held_out_accuracy['per_layer_accuracy']:.4f}")
    print(f"  exact placement accuracy  {held_out_accuracy['exact_placement_accuracy']:.4f}")

    agent = SupervisedAgent(model, config.num_devices)
    records = evaluate_agent(config, agent, held_out_seeds, num_layers=arguments.layers)
    print("\nRolled out on the same held-out scenarios")
    print(f"  mean objective            {np.mean([r.objective for r in records]):.4f}")
    print(f"  mean latency (ms)         {np.mean([r.total_latency_ms for r in records]):,.1f}")
    print(f"  mean energy               {np.mean([r.energy for r in records]):.2f}")
    print(f"  memory violations         {sum(r.memory_violations for r in records)}")
    print(
        f"  decision time per layer   "
        f"{np.mean([r.decision_runtime_us_per_layer for r in records]):,.1f} us"
    )

    path = save_model(model, __import__("pathlib").Path(arguments.output))
    print(f"\nmodel written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
