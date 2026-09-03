"""Configuration schema and YAML loading for the placement study.

Every tunable quantity in this project lives in ``configs/*.yaml`` and reaches
the code through the :class:`Config` dataclass tree built here. Modules never
hard-code experiment parameters; they receive a :class:`Config` (or one of its
sub-configurations) instead.

Loading works in three layers, each merged on top of the previous one:

1. ``configs/default.yaml`` -- the complete set of defaults.
2. An optional experiment file, e.g. ``configs/dynamic_network.yaml``.
3. Optional command-line overrides such as ``dqn.total_steps=50000``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIRECTORY = REPOSITORY_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIRECTORY / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class Range:
    """An inclusive interval that scenario sampling draws from.

    A range written as ``[a, b]`` in YAML is sampled uniformly; a bare number is
    parsed as the degenerate range ``[x, x]`` so that fixed values and sampled
    values are interchangeable everywhere in the configuration.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ConfigError(f"range upper bound {self.high} is below lower bound {self.low}")

    @property
    def is_fixed(self) -> bool:
        """Whether the range collapses to a single value."""
        return self.low == self.high

    def sample(self, rng: Any) -> float:
        """Draw one value uniformly from the range.

        Args:
            rng: A ``numpy.random.Generator`` (or any object exposing ``uniform``).

        Returns:
            A float inside ``[low, high]``.
        """
        if self.is_fixed:
            return float(self.low)
        return float(rng.uniform(self.low, self.high))

    @property
    def midpoint(self) -> float:
        """The centre of the range, used for reference/normalisation values."""
        return (self.low + self.high) / 2.0

    @staticmethod
    def parse(value: Any, *, field_name: str) -> "Range":
        """Build a range from a YAML scalar, two-element sequence, or mapping."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return Range(float(value), float(value))
        if isinstance(value, Mapping):
            try:
                return Range(float(value["min"]), float(value["max"]))
            except KeyError as error:  # pragma: no cover - defensive
                raise ConfigError(f"{field_name}: mapping range needs 'min' and 'max'") from error
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise ConfigError(f"{field_name}: range must have exactly two entries")
            return Range(float(value[0]), float(value[1]))
        raise ConfigError(f"{field_name}: cannot interpret {value!r} as a range")


@dataclass(frozen=True)
class ObjectiveConfig:
    """Weights of the placement cost function.

    The weighted objective minimised by every method is::

        alpha * latency / latency_reference
      + beta  * energy  / energy_reference
      + gamma * communication / communication_reference

    Weights must be non-negative and sum to one so that objective values stay
    comparable between configurations and DNN sizes.
    """

    alpha: float
    beta: float
    gamma: float
    comm_double_count: bool
    randomise_weights: bool = False

    def __post_init__(self) -> None:
        for name, weight in (("alpha", self.alpha), ("beta", self.beta), ("gamma", self.gamma)):
            if weight < 0:
                raise ConfigError(f"objective.{name} must be non-negative, got {weight}")
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(f"objective weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Dynamics and constraint handling of the placement environment."""

    memory_accumulates: bool
    utilisation_accumulates: bool
    utilisation_load_scale: float
    utilisation_max: float
    effective_speed_floor: float
    invalid_action_mode: str
    invalid_action_penalty: float
    input_source_device: str
    include_transmission_energy: bool
    transmission_energy_per_mb: float
    device_load_override: Mapping[str, Range] = field(default_factory=dict)

    VALID_INVALID_ACTION_MODES = ("mask", "penalty")

    def __post_init__(self) -> None:
        if self.invalid_action_mode not in self.VALID_INVALID_ACTION_MODES:
            raise ConfigError(
                "environment.invalid_action_mode must be one of "
                f"{self.VALID_INVALID_ACTION_MODES}, got {self.invalid_action_mode!r}"
            )
        if not 0.0 < self.utilisation_max < 1.0:
            raise ConfigError("environment.utilisation_max must lie strictly between 0 and 1")
        if not 0.0 < self.effective_speed_floor <= 1.0:
            raise ConfigError("environment.effective_speed_floor must lie in (0, 1]")
        if self.utilisation_load_scale <= 0:
            raise ConfigError("environment.utilisation_load_scale must be positive")

    @property
    def uses_action_masking(self) -> bool:
        """Whether infeasible devices are removed from the action space."""
        return self.invalid_action_mode == "mask"


@dataclass(frozen=True)
class DeviceArchetypeConfig:
    """Sampling ranges for one heterogeneous device archetype."""

    name: str
    compute_capacity: Range
    memory_gb: Range
    energy_per_compute: Range
    base_utilisation: Range


@dataclass(frozen=True)
class LinkConfig:
    """Sampling ranges for one symmetric device-to-device link."""

    endpoints: tuple[str, str]
    latency_ms: Range
    bandwidth_mbps: Range


@dataclass(frozen=True)
class NetworkProfileConfig:
    """Multiplicative congestion profile applied to sampled link characteristics."""

    name: str
    latency_scale: Range
    bandwidth_scale: Range
    event_probability: float
    mid_episode_probability: float

    def __post_init__(self) -> None:
        for name, probability in (
            ("event_probability", self.event_probability),
            ("mid_episode_probability", self.mid_episode_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ConfigError(f"network profile {self.name!r}: {name} must lie in [0, 1]")


@dataclass(frozen=True)
class NetworkConfig:
    """Link topology plus the congestion profile currently selected."""

    profile: str
    links: tuple[LinkConfig, ...]
    profiles: Mapping[str, NetworkProfileConfig]

    def __post_init__(self) -> None:
        if self.profile not in self.profiles:
            raise ConfigError(
                f"network.profile {self.profile!r} is not defined in network.profiles "
                f"({sorted(self.profiles)})"
            )

    @property
    def active_profile(self) -> NetworkProfileConfig:
        """The profile named by ``network.profile``."""
        return self.profiles[self.profile]


@dataclass(frozen=True)
class WorkloadConfig:
    """Parameters of the synthetic CNN-like DNN generator."""

    num_layers: int
    input_size_mb: Range
    base_compute: Range
    compute_growth: Range
    base_activation_mb: Range
    pool_every: int
    activation_decay: Range
    feature_memory_gb: Range
    head_fraction: float
    head_compute_scale: Range
    head_memory_gb: Range
    head_output_mb: Range
    jitter_sigma: float

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ConfigError("workload.num_layers must be at least 1")
        if self.pool_every < 1:
            raise ConfigError("workload.pool_every must be at least 1")
        if not 0.0 <= self.head_fraction < 1.0:
            raise ConfigError("workload.head_fraction must lie in [0, 1)")
        if self.jitter_sigma < 0:
            raise ConfigError("workload.jitter_sigma must be non-negative")


@dataclass(frozen=True)
class ExperimentConfig:
    """Seed pools and evaluation sizes shared by every experiment script."""

    train_seed_start: int
    train_seed_count: int
    eval_seed_start: int
    n_eval_scenarios: int
    max_exhaustive_combinations: int
    valid_seed_start: int = 5_000_000
    n_valid_scenarios: int = 100

    def __post_init__(self) -> None:
        pools = {
            "training": (self.train_seed_start, self.train_seed_start + self.train_seed_count),
            "validation": (self.valid_seed_start, self.valid_seed_start + self.n_valid_scenarios),
            "evaluation": (self.eval_seed_start, self.eval_seed_start + self.n_eval_scenarios),
        }
        names = list(pools)
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                first_start, first_end = pools[first]
                second_start, second_end = pools[second]
                if first_start < second_end and second_start < first_end:
                    raise ConfigError(
                        f"the {first} and {second} seed pools overlap; scenarios would no "
                        "longer be held out"
                    )

    def eval_seeds(self) -> list[int]:
        """The held-out scenario seeds, identical for every method compared."""
        return [self.eval_seed_start + index for index in range(self.n_eval_scenarios)]

    def valid_seeds(self) -> list[int]:
        """Validation seeds, used for checkpoint selection and progress checks."""
        return [self.valid_seed_start + index for index in range(self.n_valid_scenarios)]


@dataclass(frozen=True)
class DQNConfig:
    """Hyper-parameters of the hand-written Deep Q-Network agent."""

    hidden_sizes: tuple[int, ...]
    learning_rate: float
    buffer_capacity: int
    batch_size: int
    discount: float
    learning_starts: int
    train_frequency: int
    target_update_interval: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_fraction: float
    total_steps: int
    double_dqn: bool
    grad_clip_norm: float
    eval_interval: int
    eval_episodes: int
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon_decay_fraction <= 1.0:
            raise ConfigError("dqn.epsilon_decay_fraction must lie in (0, 1]")
        if self.batch_size > self.buffer_capacity:
            raise ConfigError("dqn.batch_size cannot exceed dqn.buffer_capacity")
        if self.learning_starts < self.batch_size:
            raise ConfigError("dqn.learning_starts must be at least dqn.batch_size")


@dataclass(frozen=True)
class QLearningConfig:
    """Hyper-parameters of the tabular Q-learning agent."""

    learning_rate: float
    discount: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_fraction: float
    episodes: int
    memory_buckets: int
    seeds: tuple[int, ...]
    mode: str = "single_scenario"

    VALID_MODES = ("single_scenario", "pooled")

    def __post_init__(self) -> None:
        if self.mode not in self.VALID_MODES:
            raise ConfigError(
                f"q_learning.mode must be one of {self.VALID_MODES}, got {self.mode!r}"
            )


@dataclass(frozen=True)
class SupervisedConfig:
    """Hyper-parameters and supervision source of the Random Forest baseline."""

    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    n_training_scenarios: int
    teacher: str = "auto"
    max_exhaustive_layers: int = 6

    VALID_TEACHERS = ("auto", "exhaustive", "dp", "best_known")

    def __post_init__(self) -> None:
        if self.teacher not in self.VALID_TEACHERS:
            raise ConfigError(
                f"supervised.teacher must be one of {self.VALID_TEACHERS}, got {self.teacher!r}"
            )


@dataclass(frozen=True)
class Config:
    """The full configuration tree for one experiment run."""

    seed: int
    experiment: ExperimentConfig
    objective: ObjectiveConfig
    environment: EnvironmentConfig
    devices: tuple[DeviceArchetypeConfig, ...]
    network: NetworkConfig
    workload: WorkloadConfig
    dqn: DQNConfig
    q_learning: QLearningConfig
    supervised: SupervisedConfig
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def num_devices(self) -> int:
        """Number of device archetypes, which is also the size of the action space."""
        return len(self.devices)

    @property
    def device_names(self) -> tuple[str, ...]:
        """Device names in action-index order."""
        return tuple(device.name for device in self.devices)

    def device_index(self, name: str) -> int:
        """Return the action index of a named device.

        Raises:
            ConfigError: If the name does not match any configured device.
        """
        try:
            return self.device_names.index(name)
        except ValueError as error:
            raise ConfigError(
                f"unknown device {name!r}; configured devices are {self.device_names}"
            ) from error

    def to_dict(self) -> dict[str, Any]:
        """Return the merged configuration as plain data, for provenance dumps."""
        return copy.deepcopy(dict(self.raw))


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` without mutating either."""
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_overrides(overrides: Iterable[str]) -> dict[str, Any]:
    """Turn ``a.b=value`` strings from the command line into a nested mapping.

    Values are parsed as YAML, so ``true``, ``3``, ``0.5`` and ``[1, 2]`` all
    arrive with the type they look like.

    Raises:
        ConfigError: If an override is not of the form ``dotted.key=value``.
    """
    nested: dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"override {override!r} must look like 'section.key=value'")
        dotted_key, raw_value = override.split("=", 1)
        keys = [part for part in dotted_key.strip().split(".") if part]
        if not keys:
            raise ConfigError(f"override {override!r} has an empty key")
        cursor = nested
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
            if not isinstance(cursor, dict):  # pragma: no cover - defensive
                raise ConfigError(f"override {override!r} conflicts with an earlier override")
        cursor[keys[-1]] = yaml.safe_load(raw_value)
    return nested


def _require(mapping: Mapping[str, Any], key: str, *, section: str) -> Any:
    """Fetch a required key, raising a helpful error when it is missing."""
    if key not in mapping:
        raise ConfigError(f"configuration section {section!r} is missing required key {key!r}")
    return mapping[key]


def _build_devices(entries: Sequence[Mapping[str, Any]]) -> tuple[DeviceArchetypeConfig, ...]:
    """Build device archetypes and check that their names are unique."""
    if not entries:
        raise ConfigError("at least one device must be configured")
    devices = tuple(
        DeviceArchetypeConfig(
            name=str(_require(entry, "name", section="devices")),
            compute_capacity=Range.parse(
                _require(entry, "compute_capacity", section="devices"),
                field_name="devices.compute_capacity",
            ),
            memory_gb=Range.parse(
                _require(entry, "memory_gb", section="devices"),
                field_name="devices.memory_gb",
            ),
            energy_per_compute=Range.parse(
                _require(entry, "energy_per_compute", section="devices"),
                field_name="devices.energy_per_compute",
            ),
            base_utilisation=Range.parse(
                entry.get("base_utilisation", 0.0), field_name="devices.base_utilisation"
            ),
        )
        for entry in entries
    )
    names = [device.name for device in devices]
    if len(set(names)) != len(names):
        raise ConfigError(f"device names must be unique, got {names}")
    return devices


def _build_network(section: Mapping[str, Any], device_names: Sequence[str]) -> NetworkConfig:
    """Build the network configuration and validate the topology is complete."""
    links = []
    seen: set[frozenset[str]] = set()
    for entry in _require(section, "links", section="network"):
        endpoints = tuple(str(name) for name in _require(entry, "between", section="network.links"))
        if len(endpoints) != 2:
            raise ConfigError(f"network link {endpoints} must connect exactly two devices")
        for name in endpoints:
            if name not in device_names:
                raise ConfigError(f"network link references unknown device {name!r}")
        key = frozenset(endpoints)
        if len(key) != 2:
            raise ConfigError(f"network link {endpoints} connects a device to itself")
        if key in seen:
            raise ConfigError(f"network link {endpoints} is configured more than once")
        seen.add(key)
        links.append(
            LinkConfig(
                endpoints=(endpoints[0], endpoints[1]),
                latency_ms=Range.parse(
                    _require(entry, "latency_ms", section="network.links"),
                    field_name="network.links.latency_ms",
                ),
                bandwidth_mbps=Range.parse(
                    _require(entry, "bandwidth_mbps", section="network.links"),
                    field_name="network.links.bandwidth_mbps",
                ),
            )
        )

    expected_link_count = len(device_names) * (len(device_names) - 1) // 2
    if len(links) != expected_link_count:
        raise ConfigError(
            f"network topology must be fully connected: expected {expected_link_count} links "
            f"for {len(device_names)} devices, found {len(links)}"
        )

    profiles = {
        name: NetworkProfileConfig(
            name=name,
            latency_scale=Range.parse(
                profile.get("latency_scale", 1.0), field_name=f"network.profiles.{name}.latency_scale"
            ),
            bandwidth_scale=Range.parse(
                profile.get("bandwidth_scale", 1.0),
                field_name=f"network.profiles.{name}.bandwidth_scale",
            ),
            event_probability=float(profile.get("event_probability", 0.0)),
            mid_episode_probability=float(profile.get("mid_episode_probability", 0.0)),
        )
        for name, profile in _require(section, "profiles", section="network").items()
    }
    return NetworkConfig(
        profile=str(_require(section, "profile", section="network")),
        links=tuple(links),
        profiles=profiles,
    )


def build_config(data: Mapping[str, Any]) -> Config:
    """Build a validated :class:`Config` from a merged configuration mapping."""
    devices = _build_devices(_require(data, "devices", section="<root>"))
    device_names = [device.name for device in devices]

    environment_section = dict(_require(data, "environment", section="<root>"))
    overrides = environment_section.get("device_load_override") or {}
    for name in overrides:
        if name not in device_names:
            raise ConfigError(
                f"environment.device_load_override names unknown device {name!r}; "
                f"configured devices are {device_names}"
            )
    environment_section["device_load_override"] = {
        name: Range.parse(value, field_name=f"environment.device_load_override.{name}")
        for name, value in overrides.items()
    }
    if environment_section["input_source_device"] not in device_names:
        raise ConfigError(
            f"environment.input_source_device {environment_section['input_source_device']!r} "
            f"is not one of the configured devices {device_names}"
        )

    workload_section = _require(data, "workload", section="<root>")
    dqn_section = _require(data, "dqn", section="<root>")
    q_learning_section = _require(data, "q_learning", section="<root>")
    supervised_section = _require(data, "supervised", section="<root>")

    return Config(
        seed=int(data.get("seed", 0)),
        experiment=ExperimentConfig(**_require(data, "experiment", section="<root>")),
        objective=ObjectiveConfig(**_require(data, "objective", section="<root>")),
        environment=EnvironmentConfig(**environment_section),
        devices=devices,
        network=_build_network(_require(data, "network", section="<root>"), device_names),
        workload=WorkloadConfig(
            num_layers=int(_require(workload_section, "num_layers", section="workload")),
            input_size_mb=Range.parse(
                workload_section["input_size_mb"], field_name="workload.input_size_mb"
            ),
            base_compute=Range.parse(
                workload_section["base_compute"], field_name="workload.base_compute"
            ),
            compute_growth=Range.parse(
                workload_section["compute_growth"], field_name="workload.compute_growth"
            ),
            base_activation_mb=Range.parse(
                workload_section["base_activation_mb"], field_name="workload.base_activation_mb"
            ),
            pool_every=int(workload_section["pool_every"]),
            activation_decay=Range.parse(
                workload_section["activation_decay"], field_name="workload.activation_decay"
            ),
            feature_memory_gb=Range.parse(
                workload_section["feature_memory_gb"], field_name="workload.feature_memory_gb"
            ),
            head_fraction=float(workload_section["head_fraction"]),
            head_compute_scale=Range.parse(
                workload_section["head_compute_scale"], field_name="workload.head_compute_scale"
            ),
            head_memory_gb=Range.parse(
                workload_section["head_memory_gb"], field_name="workload.head_memory_gb"
            ),
            head_output_mb=Range.parse(
                workload_section["head_output_mb"], field_name="workload.head_output_mb"
            ),
            jitter_sigma=float(workload_section["jitter_sigma"]),
        ),
        dqn=DQNConfig(
            **{
                **dqn_section,
                "hidden_sizes": tuple(dqn_section["hidden_sizes"]),
                "seeds": tuple(dqn_section["seeds"]),
            }
        ),
        q_learning=QLearningConfig(
            **{**q_learning_section, "seeds": tuple(q_learning_section["seeds"])}
        ),
        supervised=SupervisedConfig(**supervised_section),
        raw=copy.deepcopy(dict(data)),
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a mapping.

    Raises:
        ConfigError: If the file does not exist or does not contain a mapping.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        candidate = CONFIG_DIRECTORY / resolved
        resolved = candidate if candidate.exists() else Path(path)
    if not resolved.exists():
        raise ConfigError(f"configuration file not found: {path}")
    with resolved.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"configuration file {resolved} must contain a mapping at the top level")
    return dict(loaded)


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] | Mapping[str, Any] | None = None,
) -> Config:
    """Load defaults, merge an experiment file and overrides, and validate.

    Args:
        path: Optional experiment configuration file. Bare file names are looked
            up inside ``configs/``.
        overrides: Either an iterable of ``section.key=value`` strings (as passed
            on the command line) or an already-nested mapping.

    Returns:
        A validated :class:`Config`.

    Raises:
        ConfigError: If any file is missing or the merged configuration is invalid.
    """
    data = load_yaml(DEFAULT_CONFIG_PATH)
    if path is not None:
        data = _deep_merge(data, load_yaml(path))
    if overrides is not None:
        nested = overrides if isinstance(overrides, Mapping) else parse_overrides(overrides)
        data = _deep_merge(data, nested)
    return build_config(data)


def config_summary(config: Config) -> str:
    """Return a short human-readable summary used in experiment logs."""
    return (
        f"devices={config.num_devices} "
        f"layers={config.workload.num_layers} "
        f"network={config.network.profile} "
        f"weights=({config.objective.alpha}, {config.objective.beta}, {config.objective.gamma}) "
        f"invalid_actions={config.environment.invalid_action_mode} "
        f"seed={config.seed}"
    )


__all__ = [
    "Config",
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "DeviceArchetypeConfig",
    "DQNConfig",
    "EnvironmentConfig",
    "ExperimentConfig",
    "LinkConfig",
    "NetworkConfig",
    "NetworkProfileConfig",
    "ObjectiveConfig",
    "QLearningConfig",
    "Range",
    "REPOSITORY_ROOT",
    "SupervisedConfig",
    "WorkloadConfig",
    "build_config",
    "config_summary",
    "load_config",
    "load_yaml",
    "parse_overrides",
]
