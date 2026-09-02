"""Supervised placement: a Random Forest that imitates the best available oracle.

This is the direct descendant of the previous project's approach, rebuilt so
that it can be compared fairly against sequential decision making. The previous
version trained one classifier per layer *position*, using scenario-level
features, and repaired the infeasible placements that independent predictions
inevitably produce. Here a single classifier is trained on the same state vector
the reinforcement learning agent sees, and is rolled out sequentially through
the same environment with the same action masking.

That makes the comparison clean. Both methods see identical features, face
identical constraints and are scored by the identical simulator. What differs is
only the learning signal:

* the Random Forest is trained to **reproduce an oracle's decision** at states
  the oracle visits;
* the reinforcement learner is trained to **maximise return** at states its own
  policy visits.

The distinction matters because of compounding error. A classifier is only ever
shown expert trajectories, so the first time its own mistake takes it somewhere
the oracle never went, nothing in its training says what to do -- and under
memory accumulation the consequences of an early mistake persist for the rest of
the episode. This is the standard behaviour-cloning distribution shift, and it is
one of the concrete reasons to expect a sequential method to differ.

**Choice of oracle.** The teacher is the strongest supervision that is
computationally available: exhaustive search where brute force is affordable, and
otherwise the cheapest placement among the dynamic programme and the greedy
heuristics. Using a weaker teacher would turn the headline comparison into a
comparison of teachers rather than of learning formulations -- which matters
here, because Experiment 1 showed the dynamic programme is *not* the best
available placement at ten layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.agents.base import action_mask_from_info
from src.agents.greedy_agent import GreedyAgent
from src.baselines import dp_optimal, exhaustive_search
from src.config import Config
from src.environment.dnn_environment import DNNPlacementEnv
from src.environment.observation import feature_names
from src.environment.reward import evaluate_placement
from src.environment.scenario import Scenario, sample_scenario
from src.training.evaluate import run_episode

DEFAULT_MODEL_PATH = Path("checkpoints") / "supervised_rf.joblib"


@dataclass
class Demonstrations:
    """A behaviour-cloning dataset collected along oracle trajectories."""

    observations: np.ndarray
    actions: np.ndarray
    scenario_seeds: np.ndarray
    layer_indices: np.ndarray
    feature_names: list[str]
    teacher_sources: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of ``(state, action)`` pairs."""
        return int(self.observations.shape[0])


def teacher_placement(
    scenario: Scenario, config: Config, env: DNNPlacementEnv
) -> tuple[tuple[int, ...], str]:
    """Return the strongest placement available for a scenario, and its source.

    Args:
        scenario: The scenario to solve.
        config: The configuration in force.
        env: An environment used to roll out the greedy candidates.

    Returns:
        ``(placement, source)`` where source names the method that produced it.

    Raises:
        ValueError: If no feasible teacher placement could be produced, which the
            workload generator's memory cap is designed to prevent.
    """
    preference = config.supervised.teacher
    affordable = exhaustive_search.is_affordable(
        scenario.num_layers, scenario.num_devices, config.experiment.max_exhaustive_combinations
    ) and scenario.num_layers <= config.supervised.max_exhaustive_layers

    if preference == "exhaustive" or (preference == "auto" and affordable):
        if not affordable:
            raise ValueError(
                f"supervised.teacher='exhaustive' but {scenario.num_layers} layers exceeds "
                f"supervised.max_exhaustive_layers={config.supervised.max_exhaustive_layers}"
            )
        return exhaustive_search.solve(scenario, config).placement, "exhaustive"

    dp_placement = dp_optimal.solve(scenario, config).placement
    if preference == "dp":
        return dp_placement, "dp"

    candidates: list[tuple[float, tuple[int, ...], str]] = []
    dp_result = evaluate_placement(scenario, dp_placement, config)
    if dp_result.memory_violations == 0:
        candidates.append((dp_result.objective, dp_placement, "dp"))

    for criterion in ("objective_aware", "communication_aware"):
        record = run_episode(
            env, GreedyAgent(config.num_devices, criterion), scenario=scenario
        )
        candidates.append((record.objective, record.placement, f"greedy_{criterion}"))

    if not candidates:
        raise ValueError(f"no feasible teacher placement for scenario {scenario.seed}")
    _objective, placement, source = min(candidates, key=lambda item: item[0])
    return placement, source


def collect_demonstrations(
    config: Config,
    seeds: Sequence[int],
    *,
    num_layers: int | None = None,
    progress_every: int = 500,
) -> Demonstrations:
    """Roll the oracle through the environment and record its decisions.

    States are collected *along the oracle's own trajectory*, which is what makes
    this behaviour cloning rather than supervised regression on arbitrary states.

    Args:
        config: The configuration in force.
        seeds: Training scenario seeds. These must come from the training pool.
        num_layers: Optional depth override.
        progress_every: How often to print progress, or 0 to stay silent.

    Returns:
        A :class:`Demonstrations` dataset.
    """
    env = DNNPlacementEnv(config, num_layers=num_layers)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    seed_column: list[int] = []
    layer_column: list[int] = []
    sources: dict[str, int] = {}

    for index, seed in enumerate(seeds):
        scenario = sample_scenario(config, seed, num_layers=num_layers)
        placement, source = teacher_placement(scenario, config, env)
        sources[source] = sources.get(source, 0) + 1

        observation, info = env.reset(options={"scenario": scenario})
        for layer_index, action in enumerate(placement):
            observations.append(observation)
            actions.append(int(action))
            seed_column.append(int(seed))
            layer_column.append(layer_index)
            observation, _reward, terminated, _truncated, info = env.step(int(action))
            if terminated:
                break

        if progress_every and (index + 1) % progress_every == 0:
            print(f"  collected {index + 1}/{len(seeds)} scenarios")

    return Demonstrations(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        scenario_seeds=np.asarray(seed_column, dtype=np.int64),
        layer_indices=np.asarray(layer_column, dtype=np.int64),
        feature_names=feature_names(config.num_devices, config.device_names),
        teacher_sources=sources,
    )


def train_random_forest(demonstrations: Demonstrations, config: Config, seed: int = 0):
    """Fit the classifier on a demonstration dataset.

    Args:
        demonstrations: The collected ``(state, action)`` pairs.
        config: The configuration in force.
        seed: Random seed for the forest.

    Returns:
        A fitted ``RandomForestClassifier``.
    """
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=config.supervised.n_estimators,
        max_depth=config.supervised.max_depth,
        min_samples_leaf=config.supervised.min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(demonstrations.observations, demonstrations.actions)
    # Fitting benefits from every core, but inference here is one row at a time,
    # where thread dispatch costs more than the work itself (measured 13.7 ms per
    # call with n_jobs=-1 against 4.9 ms with a single thread).
    model.n_jobs = 1
    return model


class SupervisedAgent:
    """Rolls a fitted classifier through the environment, respecting the mask.

    The classifier proposes a distribution over devices; infeasible devices are
    removed and the most probable remaining device is chosen. Because masking is
    applied at decision time, this agent -- unlike the previous project's
    per-layer classifiers -- can never emit an infeasible placement, so no repair
    pass exists or is needed.
    """

    def __init__(self, model, num_actions: int, name: str = "supervised_rf") -> None:
        """Wrap a fitted scikit-learn classifier as an agent."""
        self.model = model
        self.num_actions = num_actions
        self.name = name
        self._classes = np.asarray(model.classes_, dtype=np.int64)

    def reset(self) -> None:
        """No per-episode state to clear."""

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> int:
        """Return the most probable feasible device."""
        mask = action_mask_from_info(info, self.num_actions)
        probabilities = self.model.predict_proba(observation.reshape(1, -1))[0]

        scores = np.full(self.num_actions, -np.inf, dtype=np.float64)
        scores[self._classes] = probabilities
        scores = np.where(mask, scores, -np.inf)
        return int(np.argmax(scores))


def evaluate_imitation_accuracy(
    model, demonstrations: Demonstrations
) -> dict[str, float]:
    """Measure how often the classifier reproduces the oracle's decision.

    Reports both per-layer accuracy and exact whole-placement match, the two
    numbers the previous project reported, so the two studies can be compared.
    """
    predictions = model.predict(demonstrations.observations)
    correct = predictions == demonstrations.actions

    exact = []
    for seed in np.unique(demonstrations.scenario_seeds):
        rows = demonstrations.scenario_seeds == seed
        exact.append(bool(correct[rows].all()))

    return {
        "per_layer_accuracy": float(correct.mean()),
        "exact_placement_accuracy": float(np.mean(exact)),
        "samples": float(len(demonstrations)),
        "scenarios": float(len(exact)),
    }


def save_model(model, path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist a fitted classifier, creating the directory if needed."""
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path = DEFAULT_MODEL_PATH):
    """Load a previously fitted classifier.

    Raises:
        FileNotFoundError: With a hint about the command that creates it.
    """
    import joblib

    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} does not exist. Train the supervised baseline first: "
            "python -m src.training.train_supervised"
        )
    return joblib.load(path)


__all__ = [
    "DEFAULT_MODEL_PATH",
    "Demonstrations",
    "SupervisedAgent",
    "collect_demonstrations",
    "evaluate_imitation_accuracy",
    "load_model",
    "save_model",
    "teacher_placement",
    "train_random_forest",
]
