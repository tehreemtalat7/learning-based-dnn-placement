# Adaptive DNN Placement Using Reinforcement Learning in Heterogeneous Edge–Cloud Environments

An independent research and portfolio project studying whether reinforcement
learning can learn **adaptive, sequential** policies for placing the layers of a
deep neural network across heterogeneous edge, GPU and cloud devices — and
measuring honestly how it compares with heuristic, supervised and optimal
placement strategies.

> This is a personal research project, not peer-reviewed work. Every number
> reported here is produced by the scripts in this repository from the raw CSVs
> in `results/raw/`; nothing is quoted from elsewhere or estimated by hand.

**Status:** the simulation environment, the heuristic baselines, the exact
baselines (exhaustive search and dynamic programming) and the static comparison
experiment are complete. The supervised, tabular and deep RL agents and the
dynamic-condition experiments are being added phase by phase; the results
sections below are populated from generated data as each phase lands.

---

## 1. Motivation

Executing a DNN across an edge–cloud continuum means deciding, for each layer,
*where it should run*. A fast device may be far away, so moving a large
activation tensor to it costs more than the computation saves. A cheap edge
device may be energy-efficient but too small to hold a dense layer's weights.
And the choice made for one layer changes the cost of the next, because the
activation must be transferred whenever consecutive layers sit on different
devices.

That coupling is the heart of the problem, and it is what motivates treating
placement as a **sequential decision process** rather than a set of independent
per-layer predictions.

## 2. Connection to the previous project

This project builds directly on
[`distributed-dnn-device-placement`](https://github.com/tehreemtalat7/distributed-dnn-device-placement),
which compared random, round-robin, greedy, communication-aware and
objective-aware placement against exhaustive search, and trained five
independent Random Forest classifiers — one per layer position — to predict the
optimal placement.

Its main limitation is structural: **per-layer independent prediction cannot
represent the dependency it is trying to model.** The presence of a
placement-repair pass in that repository is the symptom — independent
predictions can be jointly infeasible.

This project keeps the same cost model (so the two studies stay comparable) and
changes the formulation. `docs/RELATION_TO_PREVIOUS_WORK.md` records in detail
what carries over, what was redesigned and why, and why the previous trained
model cannot be applied to this environment.

## 3. Research questions

**RQ1.** Can reinforcement learning learn an adaptive DNN layer placement policy
that achieves competitive latency–energy performance compared with heuristic,
supervised ML and optimal placement strategies in heterogeneous edge–cloud
environments?

**RQ2.** How does the RL-based strategy perform when network conditions and
device loads change dynamically, including shifts it did not see during
training?

The project does **not** assume that RL wins. The environment is fully
observable with a known cost model, which means an exact optimum is computable
and the strongest greedy heuristic is already reactive; a plausible honest
outcome is that RL matches the best heuristic and trails the exact optimum.
Where that happens, it is reported and analysed rather than hidden.

## 4. System architecture

### Devices

Four heterogeneous archetypes, sampled per scenario from configured ranges:

| Device | Compute capacity | Memory | Energy / compute | Role |
|---|---|---|---|---|
| `edge_device` | 0.8 – 1.4 | 2 – 5 GB | 0.4 – 0.7 | Phone class: slow, cheap to run, tight memory |
| `edge_server` | 4 – 7 | 10 – 18 GB | 1.0 – 1.5 | On-premise micro server |
| `gpu_server` | 12 – 20 | 24 – 40 GB | 2.0 – 3.0 | Accelerator: fast, power hungry |
| `cloud_server` | 25 – 40 | 64 – 128 GB | 2.5 – 4.0 | Very fast and roomy, but far away |

Each device also carries a background utilisation, and — while an episode runs —
accumulates the memory and load of the layers assigned to it:

```
effective_speed = compute_capacity × (1 − utilisation)
utilisation     = base_utilisation + assigned_compute / (capacity × load_scale)
```

This is what makes the problem genuinely **path dependent**: piling work onto one
device slows it down and can make it too full to host a later layer.

### Network

Symmetric links between every pair of devices, each with latency and bandwidth:

```
communication_time_ms = latency_ms + (activation_mb × 8 / bandwidth_mbps) × 1000
```

Transfers within a device are free. Three profiles — `normal`, `congested` and
`dynamic` — scale link characteristics; a `dynamic` congestion event can begin
*part way through an episode*, which is what Experiment 3 measures.

### Workload

Synthetic but structured CNN-like DNNs of configurable depth: a feature stage
whose activations start large and shrink through pooling, followed by a dense
head with large weight memory and negligible activations. That produces the
tension the placement problem is about — layers that are expensive to move but
fit anywhere, versus layers that are cheap to move but exclude small devices.

These workloads are **not profiled from real networks**; the generative process
and its rationale are documented in `src/environment/workload.py`.

## 5. Reinforcement learning formulation

**Episode.** One episode places one DNN, one layer per step, in order.

**Action space.** `Discrete(4)` — which device hosts the current layer. Fixed
regardless of DNN depth.

**State.** A 58-dimensional `float32` vector: 14 global features (progress,
current and next layer characteristics, remaining workload, objective weights)
plus 11 features per device (effective speed, free memory, utilisation, energy
rate, whether it holds the incoming activation, the link's latency and
bandwidth, feasibility, and the estimated execution, communication and immediate
objective cost of choosing it). Its width does not depend on depth, so one
policy handles 5- and 30-layer DNNs. The full specification is generated from
the code into [`docs/STATE_SPEC.md`](docs/STATE_SPEC.md).

**Reward.**

```
reward_t = −( α·latency_t/latency_ref + β·energy_t/energy_ref + γ·comm_t/comm_ref )
```

Because the objective is linear in its components, the undiscounted episode
return is **exactly the negative of the placement's weighted objective** — the
training signal and the evaluation metric are the same quantity.

The references are the expected cost of placing the DNN uniformly at random, so
an objective near `1.0` means "no better than random", and objective values stay
comparable across DNN depths. (Measured: random placement scores 1.19 on the
default configuration.) Weights default to α=0.5, β=0.3, γ=0.2 and live only in
configuration.

**Infeasible actions are masked, not penalised.** A device without enough free
memory is removed from the action space. Masking is preferred because every
placement is then feasible by construction — no repair pass — and there is no
penalty magnitude to trade off against the reward scale. A `penalty` mode is
implemented as well, and Experiment 5 measures the difference rather than
asserting it.

## 6. Baselines

| Method | Kind | Notes |
|---|---|---|
| Random | heuristic | Uniform over feasible devices; the objective's calibration point |
| Round robin | heuristic | Cycles through devices, skipping infeasible ones |
| Fastest-device greedy | heuristic | Minimises execution time; ignores communication |
| Communication-aware greedy | heuristic | Minimises execution + transfer time |
| Objective-aware greedy | heuristic | Minimises the immediate weighted objective — the strongest heuristic |
| Random Forest | supervised | Imitates the exact optimum from the same 58 features, rolled out sequentially |
| Exhaustive search | exact | All `D^L` assignments; small DNNs only |
| Dynamic programming | exact | Viterbi-style, `O(L·D²)`; gives an optimality reference at every depth |
| Tabular Q-learning | RL | On a fixed scenario, for comparison and to show why tabular does not scale |
| DQN | RL | Hand-written PyTorch: replay buffer, target network, Double DQN, masked actions |

All heuristics read exactly the same per-device costs that are encoded in the
agent's observation, so comparisons measure decision quality, not information
asymmetry.

## 7. Experimental setup

* 300 held-out evaluation scenarios drawn from a seed pool **disjoint** from the
  training pool, so no learning method is ever evaluated on a scenario it saw.
* Every method is evaluated on the *same* scenario seeds, making all comparisons
  paired.
* Three training seeds per learning method; results are reported as mean with a
  95 % confidence interval.
* Memory constraints genuinely bind: with the default configuration a device is
  masked out at some point in roughly 31 % of episodes under objective-aware
  greedy placement.

## 8. Results

*Populated from `results/processed/` as the experiment phases land. Learning
methods are not in these tables yet — they arrive with the phases that implement
them.*

### Experiment 1 — static comparison

300 held-out scenarios, 10-layer DNNs, stable network. Reproduce with
`python -m experiments.static_experiment`.

| Method | Objective | Latency (ms) | Energy | Gap vs best known |
|---|---:|---:|---:|---:|
| Random | 1.181 | 10 411 | 27.8 | 162.98 % |
| Round robin | 0.992 | 8 325 | 25.8 | 119.57 % |
| Greedy (fastest device) | 0.696 | 1 441 | 46.8 | 54.38 % |
| Greedy (communication-aware) | 0.530 | 1 128 | 40.1 | 17.06 % |
| **Greedy (objective-aware)** | **0.458** | 2 728 | 24.6 | **0.34 %** |
| DP (relaxed problem) | 0.473 | 3 573 | 20.9 | 3.24 % |

On 5-layer DNNs, where exhaustive search is affordable (1 024 candidates per
scenario), the gaps are measured against the true optimum: objective-aware
greedy 0.26 %, the dynamic programme 0.12 %.

**The result worth pausing on.** At ten layers the dynamic-programming placement
is *beaten* by objective-aware greedy — 0.473 against 0.458, with greedy cheaper
in 90 % of scenarios (paired Wilcoxon, p = 8.6 × 10⁻¹⁹). This is not a bug in the
solver. The dynamic programme is exact for the *relaxed* problem in which
devices never slow down and never fill up; it therefore concentrates work on the
fastest devices without accounting for the congestion its own choices create.
The greedy heuristic, deciding inside the real environment, sees the accumulated
utilisation and routes around it.

Two consequences run through the rest of the project. Anything computed on the
relaxation must be labelled as such — hence `DP (relaxed problem)` rather than
"optimal". And the interesting question for the learning agent is sharpened: in
the static setting the strongest heuristic is already within 0.26 % of optimal,
so there is almost no headroom; whatever advantage sequential learning has must
come from *anticipating* accumulation and from the dynamic conditions of
Experiments 3 and 4.

Measured on 8-layer DNNs, where both the relaxation and brute force can be run:
the DP lower bound sits 4.89 % below the true optimum, while the DP *placement*
lands 1.47 % above it.

### Cost of the optimal reference

| Depth | Candidate placements | Exhaustive search | Dynamic programming |
|---:|---:|---:|---:|
| 5 | 1 024 | 0.04 s | 0.17 ms |
| 8 | 65 536 | 4.4 s | 0.27 ms |
| 10 | 1 048 576 | 83.7 s | 0.33 ms |
| 50 | 4⁵⁰ | infeasible | ~1 ms |

This is why the previous project could only report optimality gaps for five-layer
DNNs, and why the dynamic programme was added here.

## 9. Visualisations

Generated into `results/figures/` by the experiment scripts; every figure is
built from the CSVs in `results/`, never hand-authored.

| Figure | Content |
|---|---|
| `fig03_latency_by_method.png` | Mean end-to-end latency by method |
| `fig04_energy_by_method.png` | Mean energy consumption by method |
| `fig05_objective_by_method.png` | Mean weighted objective by method |
| `fig06_runtime_by_method.png` | Placement decision time per layer (log scale) |
| `fig10_optimality_gap.png` | Optimality gap against exhaustive search, 5-layer DNNs |

## 10. Key findings

*Written once the experiments have run — including any finding that reinforcement
learning does **not** improve on a baseline.*

## 11. Limitations

* Workloads are synthetic; no real network is profiled on real hardware.
* Only chain-structured DNNs are modelled — no residual or inception branching.
* A single inference pass is simulated: no pipelining, batching or concurrency.
* Device and network parameters come from hand-specified ranges, not measured
  traces.
* Exhaustive search is only tractable for small depths; beyond that the dynamic
  programme provides the optimality reference, and it is exact only when memory
  and utilisation accumulation are disabled (otherwise it is reported as a lower
  bound).

## 12. Future work

*To be written alongside the findings.*

## 13. Installation

Requires Python 3.11 (PyTorch, Gymnasium and scikit-learn all support it).

```bash
make setup
```

This creates `.venv` and installs `requirements.txt`. To use a different
interpreter: `make setup PYTHON_BOOTSTRAP=/path/to/python3.11`.

## 14. Reproducing the experiments

```bash
make test        # unit tests
make validate    # environment sanity checks with non-learning agents
make experiments # every experiment; writes results/raw/*.csv
make figures     # rebuilds every figure from those CSVs
```

Individual scripts run as modules from the repository root, for example:

```bash
.venv/bin/python -m experiments.validate_environment --scenarios 200
```

Any configuration value can be overridden on the command line:

```bash
.venv/bin/python -m experiments.validate_environment --set workload.num_layers=20 --set network.profile=congested
```

## 15. Repository structure

```text
configs/          YAML configuration; defaults plus one file per experiment
docs/             Design notes, generated state specification, relation to the previous project
src/config.py     Configuration schema, loading, merging and validation
src/environment/  Devices, network, workload, scenario sampling, cost model, Gymnasium environment
src/agents/       Random, round-robin, greedy, tabular Q-learning and DQN agents
src/baselines/    Exhaustive search, dynamic-programming optimum, supervised Random Forest
src/training/     Training loops and the shared evaluation harness
src/utils/        Seeding, metrics, statistics and plotting helpers
experiments/      Runnable experiment scripts
results/          raw/ per-episode CSVs, processed/ aggregates, figures/ plots
tests/            Unit tests for the cost model, the environment and the baselines
```

## 16. License

MIT — see [LICENSE](LICENSE).
