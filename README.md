# Adaptive DNN Placement Using Reinforcement Learning in Heterogeneous Edge–Cloud Environments

An independent research and portfolio project studying whether reinforcement
learning can learn **adaptive, sequential** policies for placing the layers of a
deep neural network across heterogeneous edge, GPU and cloud devices — and
measuring honestly how it compares with heuristic, supervised and optimal
placement strategies.

> This is a personal research project, not peer-reviewed work. Every number
> reported here is produced by the scripts in this repository from the raw CSVs
> in `results/raw/`; nothing is quoted from elsewhere or estimated by hand.

**Status:** the simulation environment, every baseline (heuristic, exact,
supervised and tabular), the deep Q-network and Experiments 1 to 4 are complete.
Ablations and the final write-up are still to come; the results sections below
are populated from generated data as each phase lands.

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

* Scenario seeds come from **three disjoint pools**: training, validation and
  evaluation. Training draws from the first, checkpoint selection and progress
  checks use the second, and every reported number comes from the third — so no
  decision about a model is ever made using the data its results are reported on.
  The configuration refuses to load if the pools overlap.
* Every method is evaluated on the *same* scenario seeds, making all comparisons
  paired.
* Three training seeds per learning method; results are reported as mean with a
  95 % confidence interval, and `experiments/dqn_seed_spread.py` reports every
  seed rather than only the one that validated best.
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

| Method | Objective | Latency (ms) | Energy | Gap vs best known | Decision time / layer |
|---|---:|---:|---:|---:|---:|
| Random | 1.181 | 10 411 | 27.8 | 163.22 % | 6 µs |
| Round robin | 0.992 | 8 325 | 25.8 | 119.79 % | 2 µs |
| Greedy (fastest device) | 0.696 | 1 441 | 46.8 | 54.53 % | 3 µs |
| Greedy (communication-aware) | 0.530 | 1 128 | 40.1 | 17.17 % | 4 µs |
| **Greedy (objective-aware)** | **0.458** | 2 728 | 24.6 | **0.45 %** | 3 µs |
| Random Forest (supervised) | 0.461 | 2 969 | 23.3 | 1.08 % | 6 034 µs |
| Tabular Q (pooled) | 0.480 | 2 310 | 28.7 | 5.66 % | 16 µs |
| Tabular Q (single scenario) | 0.586 | 3 114 | 33.2 | 29.22 % | 16 µs |
| **DQN** | **0.456** | 2 562 | 25.5 | **0.07 %** | 37 µs |
| DP (relaxed problem) | 0.473 | 3 573 | 20.9 | 3.35 % | 33 µs |

No method records a memory violation: action masking makes every placement
feasible by construction, so the repair pass the previous project needed has no
counterpart here.

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

### Learning from an oracle is not the same as being good at the task

The supervised baseline is trained on the same 58 features the RL agent sees,
imitating the strongest oracle available for each scenario (exhaustive search
where affordable; otherwise the cheapest of the dynamic programme and the greedy
heuristics — the teacher mix at ten layers came out 52.7 % dynamic programme,
45.7 % objective-aware greedy, 1.6 % communication-aware greedy).

| | Value |
|---|---:|
| Per-layer agreement with the oracle, held out | 93.7 % |
| Exact whole-placement agreement, held out | 44.7 % |
| Rolled-out objective | 0.461 |

Reproducing 93.7 % of the oracle's individual decisions still lands **0.65 %
behind the myopic heuristic** it partly imitates, winning only 2 % of scenarios
(paired Wilcoxon, p = 5.3 × 10⁻²⁷). This is behaviour cloning's characteristic
failure: the classifier only ever sees states the *oracle* visits, so the first
time its own mistake leads somewhere unfamiliar, nothing in its training says
what to do — and under memory accumulation an early mistake keeps mattering. It
is a concrete motivation for optimising return over the agent's own state
distribution rather than imitating decisions.

Worth noting for the runtime figure: the Random Forest costs ~5 ms per decision
against ~3 µs for the greedy heuristics, three orders of magnitude more. That is
scikit-learn's per-call overhead on single-row inference over 200 trees rather
than anything intrinsic to the model, but it is what this implementation costs.

### Why the deep agent is necessary, measured rather than asserted

Tabular Q-learning needs a finite state space, so the state collapses to
*(layer index, device holding the activation, memory bucket per device)* — device
speeds, energy rates, link latencies and bandwidths are all discarded, being
continuous quantities that differ in every scenario.

| Setting | Objective on held-out scenarios |
|---|---:|
| Trained on one fixed scenario | 0.586 |
| Trained across sampled scenarios ("pooled") | 0.480 |
| Objective-aware greedy | 0.458 |

On the single scenario it trains on, the table converges **exactly** to the
dynamic-programming placement (0.4699 against 0.4699, a 0.00 % gap) — the
abstraction is lossless when the discarded quantities are constants. Transferred
to unseen scenarios the same table is 29 % worse than greedy.

The sharpest detail is this: **0 % of the states met at evaluation are states the
table never visited in training.** The agent is not guessing on unfamiliar
states; it has an entry for every state it meets and is still wrong, because the
state does not contain what the decision depends on. Pooled training recovers
much of the loss (0.480) by learning a good *average* placement pattern, but it
cannot condition on the scenario in front of it. That is precisely the gap
function approximation exists to close.

### Does reinforcement learning help? A little, and not everywhere

The DQN is the cheapest method in the table, at 0.456 against 0.458 for the
strongest heuristic — **0.35 % cheaper on average**. All three training seeds
land in the same place:

| Seed | Objective (95 % CI) | vs greedy | Scenarios won | p (paired Wilcoxon) |
|---|---|---:|---:|---:|
| 0 | 0.4564 [0.4508, 0.4620] | −0.35 % | 25 % | 0.0022 |
| 1 | 0.4566 [0.4510, 0.4622] | −0.32 % | 27 % | 0.069 |
| 2 | 0.4565 [0.4509, 0.4621] | −0.35 % | 30 % | 0.0093 |

Seed variance is negligible here — the spread across seeds is 0.04 % of the
objective, far smaller than the gap between methods — but **one of the three
seeds does not reach significance at the 5 % level**, and that is reported rather
than dropped.

The average hides something more interesting. Comparing placements scenario by
scenario:

| | Scenarios | Mean effect | Largest |
|---|---:|---:|---:|
| Identical placement to greedy | 182 (61 %) | — | — |
| DQN cheaper | 74 (25 %) | −1.61 % | −21.2 % |
| DQN more expensive | 44 (15 %) | +0.30 % | +1.43 % |

So the agent agrees with the heuristic on most problems, and its advantage comes
from a **minority of scenarios where myopia is expensive**, where it wins by far
more than it loses elsewhere. On the fifteen scenarios where it gains most it
cuts end-to-end latency by 39 %, accepting 15 % more energy and 24 % more
communication to do it, and makes 60 % fewer device switches — it commits to a
fast device earlier and pays a small, immediate cost to avoid a later blow-up.
That is precisely the trade a one-layer-lookahead heuristic cannot make, and it
is the clearest evidence in this project that the sequential formulation is doing
real work.

It is also a **small** effect, and it should be read as such. In the static
setting the strongest heuristic is already within 0.26 % of the exhaustive
optimum, so there was never much room. The claim supported by this experiment is
"reinforcement learning matches the best heuristic and improves slightly on it by
handling a hard minority of cases", not "reinforcement learning solves DNN
placement". Whether the advantage grows when conditions change is what
Experiments 3 and 4 are for.

Two practical notes. Validation return plateaus by roughly 30 000 environment
steps (`fig02`), so the configured 150 000 is generous — a shorter budget would
reach the same policy. And a decision costs 37 µs against 3 µs for greedy: an
order of magnitude more, but a whole 10-layer placement still takes ~370 µs
against the 83.7 s exhaustive search needs at this depth — a factor of about
230 000.

### Experiment 2 — scaling with DNN depth

Weighted objective at each depth, 300 held-out scenarios per depth. `DQN` was
trained only on 10-layer networks; `DQN (mixed depths)` on a mixture.

| Method | 5 layers | 10 layers | 20 layers | 30 layers |
|---|---:|---:|---:|---:|
| Greedy (communication-aware) | 0.488 | 0.530 | 0.562 | 0.579 |
| Greedy (objective-aware) | 0.403 | 0.458 | 0.510 | 0.546 |
| Random Forest (supervised) | 0.404 | 0.461 | 0.519 | 0.667 |
| **DQN** | 0.403 | 0.456 | 0.508 | 0.542 |
| DQN (mixed depths) | 0.403 | 0.459 | 0.511 | 0.551 |
| Tabular Q (pooled) | 0.407 | 0.480 | 0.878 | 2.279 |
| DP (relaxed problem) | 0.403 | 0.473 | 1.002 | 1.659 |
| Random | 1.020 | 1.181 | 2.055 | 3.151 |

Three things happen as the DNN gets deeper.

**The dynamic programme collapses.** From 0.403 at five layers to 1.659 at
thirty — worse than the fastest-device heuristic, and worse than random
placement was at five layers. It is exact for the relaxed problem in which
devices never slow down, and the error in that relaxation compounds with every
layer: it keeps loading the fast devices that its model says are still fast. This
is the clearest possible vindication of labelling it "DP (relaxed problem)"
rather than "optimal", and the strongest argument in the project for learning a
policy in the real environment rather than solving a tractable approximation of
it.

**Behaviour cloning degrades.** The Random Forest tracks the heuristics to twenty
layers and then breaks down at thirty (0.667 against 0.546). It was trained on
10-layer demonstrations, and thirty-layer episodes take it far outside the state
distribution it was shown.

**The learned policy transfers.** The DQN is the cheapest method at every depth,
having been trained only on 10-layer networks — so this is generalisation, not
memorisation: the fixed-width state vector carries the shape of the workload
rather than its length.

| Depth | Greedy | DQN | Margin | Scenarios won | p |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.4030 | 0.4028 | +0.04 % | 4 % | 0.48 |
| 10 | 0.4583 | 0.4564 | +0.35 % | 25 % | 0.0022 |
| 20 | 0.5097 | 0.5077 | +0.34 % | 38 % | 0.17 |
| 30 | 0.5459 | 0.5419 | +0.66 % | 46 % | 0.0029 |

The margin is larger at thirty layers than at five, but it is **not monotone and
not significant at every depth**: the twenty-layer margin is no larger than the
ten-layer one and does not reach significance (p = 0.17), and at five layers the
two methods are indistinguishable. The trend that does hold cleanly is the win
rate — 4 % → 25 % → 38 % → 46 % — so as the DNN deepens the agent departs from
the heuristic more often and is right to. On this evidence "the advantage grows
with depth" is a reasonable reading, not a demonstrated one.

Training on mixed depths did **not** help: it is slightly worse at every depth
than the policy trained on a single size, which is worth reporting precisely
because it contradicts the obvious expectation.

At five layers every method except random and tabular Q produces *the same
placement* — all layers on the edge server — so that panel discriminates between
nothing. Depth is what makes this problem interesting.

### Experiment 3 — dynamic network conditions

Identical held-out scenarios under three regimes. `DQN` was trained only on the
normal network, so its congested and dynamic columns measure **robustness to a
shift it never saw**; `DQN (dynamic-trained)` saw the dynamic distribution.

| Method | Normal | Congested | Congested change | Dynamic change |
|---|---:|---:|---:|---:|
| Random | 1.181 | 2.027 | +71.7 % | +20.6 % |
| Round robin | 0.992 | 1.628 | +64.2 % | +22.7 % |
| Greedy (fastest device) | 0.696 | 1.025 | +47.2 % | +10.8 % |
| Greedy (objective-aware) | 0.458 | 0.541 | +18.0 % | +3.6 % |
| Random Forest (supervised) | 0.461 | 0.501 | +8.6 % | +2.2 % |
| DQN | 0.456 | 0.494 | +8.2 % | +1.9 % |
| Tabular Q (pooled) | 0.480 | 0.516 | +7.6 % | +1.9 % |
| DQN (dynamic-trained) | 0.457 | 0.490 | +7.2 % | +1.6 % |
| DP (relaxed problem) | 0.473 | 0.505 | +6.9 % | +1.7 % |
| Greedy (communication-aware) | 0.530 | 0.540 | +1.8 % | -0.7 % |

Congestion separates the methods far more sharply than the static setting did.
The strongest heuristic degrades by 18.0 %; the DQN by 8.2 %, less than half as
much, despite never having been trained on a congested network. Under congestion
it beats objective-aware greedy by **6.72 %** (paired Wilcoxon, p = 6.2 × 10⁻⁶)
against 0.35 % in the normal regime — the advantage is roughly twenty times
larger where conditions are harder.

Training on the dynamic distribution helps further, but modestly: it wins **55 %
of congested scenarios against the normal-trained agent's 36 %**, at 7.43 %
better than greedy (p = 2.4 × 10⁻²³). Most of the robustness is already present
without ever seeing congestion, which suggests the policy learned something
structural — keep consecutive layers together when moving them is expensive —
rather than memorising a regime.

One row deserves care rather than celebration. Communication-aware greedy
degrades least of all (+1.8 %), because it already refuses to move data. It is
simply *starting from a much worse place* (0.530), and a method that is bad
everywhere is not robust — it is uniformly mediocre. Degradation percentages have
to be read next to the level they degrade from.

### Experiment 4 — device load

`gpu_server` background utilisation is swept from 20 % to 80 % while the
workload, the network and the other devices are held fixed.

| Method | Layers still on the loaded GPU | Weighted objective |
|---|---|---|
| Greedy (objective-aware) | 32% → 18% → 1% | 0.458 → 0.469 → 0.461 |
| DQN | 37% → 21% → 2% | 0.456 → 0.468 → 0.461 |
| Random Forest (supervised) | 25% → 18% → 11% | 0.461 → 0.471 → 0.483 |
| DP (relaxed problem) | 18% → 7% → 0% | 0.472 → 0.482 → 0.468 |
| Tabular Q (pooled) | 50% → 50% → 50% | 0.479 → 0.505 → 0.611 |
| Round robin | 24% → 24% → 24% | 0.992 → 0.991 → 0.989 |
| Random | 25% → 25% → 25% | 1.181 → 1.178 → 1.169 |

The share column is the direct test of adaptivity, and it separates the methods
cleanly. Every method that can see utilisation routes work away as the GPU fills:
greedy 32 % → 1 %, the DQN 37 % → 2 %, the dynamic programme 18 % → 0 %. Every
method that cannot keeps feeding a device that can no longer do the work — random
and round-robin by construction, and **tabular Q at a flat 50 %**, because
utilisation is not part of its discrete state at all. Its objective pays for it,
rising 28 % across the sweep while the adaptive methods stay flat.

The Random Forest sits in between (25 % → 11 %): it can see utilisation, but it
imitates decisions taken on a distribution where the GPU was rarely this loaded.

Note the non-monotonic middle column: every method does slightly *worse* at 50 %
than at 80 %. At 80 % the GPU is so slow that it is obviously the wrong choice;
at 50 % it is a genuinely marginal one, and marginal choices are where methods
lose ground.

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
| `fig01_dqn_training_return_static.png` | DQN episode return and its moving average during training |
| `fig02_dqn_validation_return_static.png` | DQN return on validation scenarios, all three seeds |
| `fig03_latency_by_method.png` | Mean end-to-end latency by method |
| `fig04_energy_by_method.png` | Mean energy consumption by method |
| `fig05_objective_by_method.png` | Mean weighted objective by method |
| `fig06_runtime_by_method.png` | Placement decision time per layer (log scale) |
| `fig07_objective_vs_depth.png` | Placement quality against DNN depth (log scale) |
| `fig07b_gap_vs_depth.png` | Distance from the best placement found, by depth |
| `fig07c_runtime_vs_depth.png` | Time to place one DNN against depth, all methods |
| `fig08_network_conditions.png` | Performance under normal, congested and dynamic networks |
| `fig09_device_load.png` | Performance as one device's background load rises |
| `fig09b_device_load_share.png` | Share of layers still sent to the loaded device |
| `fig10_optimality_gap.png` | Optimality gap against exhaustive search, 5-layer DNNs |
| `fig11_tabular_training_single_scenario.png` | Tabular Q-learning converging onto the exact solution |
| `fig11_tabular_training_pooled.png` | The same table trained across scenarios |

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
make train       # train the learned agents into checkpoints/
make experiments # every experiment; writes results/raw/*.csv
make figures     # rebuilds every figure from those CSVs
```

The learned methods are optional: the experiments run without them and report
which checkpoints are missing, so a partial results table can never be mistaken
for a complete one.

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
