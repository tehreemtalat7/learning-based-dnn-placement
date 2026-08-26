# Relation to the previous project

This project is a direct successor to
[`tehreemtalat7/distributed-dnn-device-placement`](https://github.com/tehreemtalat7/distributed-dnn-device-placement).
This document records what the earlier work did, what carries over, what was
deliberately redesigned, and why. It exists so that the two repositories can be
read together without either one having to be taken on trust.

## 1. What the previous project built

A simulation framework for placing the layers of a 5-layer DNN across three
heterogeneous devices (edge phone, edge GPU, cloud server).

| Component | Behaviour |
|---|---|
| `simulator/device.py` | Static device: `compute_speed`, `memory_gb`, `energy_per_compute` |
| `simulator/layer.py` | Layer: `compute_cost`, `memory_gb`, `output_size_mb` |
| `simulator/network.py` | `transfer_time_ms = latency_ms + (size_mb * 8 / bandwidth_mbps) * 1000` |
| `simulator/network_topology.py` | Symmetric links keyed by the unordered device pair; intra-device transfer is free |
| `simulator/placement.py` | Sums execution time and inter-layer transfer time; energy counts computation only |
| `simulator/objectives.py` | `w_latency * (latency / latency_ref) + w_energy * (energy / energy_ref)`, weights summing to 1 |
| `simulator/greedy.py`, `communication_greedy.py`, `objective_greedy.py` | Three myopic heuristics of increasing sophistication |
| `simulator/exhaustive_search.py` | Brute force over all `D^L` assignments, filtered by memory feasibility |
| `simulator/placement_repair.py` | Repairs infeasible ML predictions one layer at a time |
| `experiments/*` | 5 000 random scenarios, exhaustive labels, per-layer Random Forests, comparison scripts |

Its headline result over 1 000 held-out scenarios: objective-aware greedy reached
a 4.63 % mean optimality gap, the Random Forest 6.86 %, and a hybrid of the
Random Forest with a small local search 0.91 %.

## 2. What carries over

The cost model is kept **deliberately identical in form**, so that results from
the two projects are readable against each other:

```
execution_time_ms     = compute_cost / effective_speed * 1000
communication_time_ms = latency_ms + (output_size_mb * 8 / bandwidth_mbps) * 1000
energy                = compute_cost * energy_per_compute
```

Also carried over: the weighted, normalised objective; the family of heuristic
baselines; exhaustive search as ground truth; optimality gap as the headline
metric; and the idea of sampling randomised scenarios rather than studying a
single hand-picked one.

The code itself is re-implemented rather than copied, because the surrounding
data model changed (see below).

## 3. What was redesigned, and why

### 3.1 The formulation: independent classification to a sequential decision process

The previous supervised approach trained five classifiers, one per layer
position, each predicting its layer's device **independently**. That is a
modelling mismatch with the phenomenon under study: the communication cost of
layer *i* is a function of where layer *i-1* was placed, so the decisions are
coupled by construction. The symptom is visible in the previous repository
itself — `placement_repair.py` exists because independent predictions can
produce placements that are jointly infeasible.

This project models placement as a Markov decision process: one episode places
one DNN, one layer per step, and the state carries the previous layer's device
and the current device occupancy. Feasibility is enforced *during* the decision
by masking infeasible devices out of the action space, so no repair pass is
needed and every produced placement is feasible by construction.

### 3.2 Static devices to devices with load

Previously a device's speed never changed. Here each device carries a
`base_utilisation` and, when `environment.utilisation_accumulates` is enabled,
gains utilisation as work is assigned to it:

```
effective_speed = compute_capacity * (1 - utilisation)
```

Resident memory accumulates the same way. This matters for the research
question: it is what makes the problem genuinely path-dependent (an early
decision can make a later one infeasible), and it is what Experiment 4 varies.

### 3.3 Three devices and five layers to four devices and any depth

The state vector here has a fixed length that does not depend on the number of
layers, so a single policy handles 5-, 10-, 20- and 30-layer DNNs. The previous
feature schema was hard-wired to 5 layers x 3 devices.

### 3.4 Fixed normalisation constants to per-scenario references

The previous objective normalised by the constants 10 000 ms and 500 energy
units. Those do not scale with DNN depth, so objective values for a 5-layer and
a 20-layer network were not comparable. Here the references are derived
analytically from the sampled workload, so the objective stays on a comparable
scale across sizes.

### 3.5 Exhaustive search only, to exhaustive search plus exact dynamic programming

Exhaustive search costs `D^L` and is therefore unusable past roughly ten layers,
which is why the previous project could only report optimality gaps for 5-layer
DNNs. Because the per-step cost here depends only on the previous device and the
current device, the optimum is a shortest path through a layered graph and can be
computed exactly by dynamic programming in `O(L * D^2)`.

Both are implemented. Exhaustive search is retained as an independent check: a
unit test asserts the two agree on small random instances. The DP optimum is
exact when `memory_accumulates` and `utilisation_accumulates` are both disabled;
when they are enabled it is a relaxation, and is reported and plotted as a lower
bound rather than as the optimum.

## 4. Why the previous Random Forest model is not reused directly

The supervised baseline here is re-created rather than imported. The trained
model from the previous repository cannot be applied to this environment:

1. **Feature schema.** Its inputs are fixed to 3 devices x 5 layers and contain
   no utilisation, no free-memory state, and no per-link features relative to
   the current decision. This environment has 4 devices, variable depth, and
   device state that changes within an episode.
2. **Label space.** It predicts a 5-tuple over a different device set.
3. **Cost model calibration.** It was fitted against a different scenario
   distribution and different normalisation constants, so its decision
   boundaries encode assumptions that no longer hold.

Instead, the same *idea* is reproduced faithfully inside the new environment: a
`RandomForestClassifier` is trained on `(state_vector, oracle_device)` pairs,
where the oracle is the dynamic-programming optimum, and is then rolled out
sequentially through the same environment with the same action masking. This
gives the cleanest possible comparison — identical inputs, identical oracle,
identical constraints — between **myopic supervised imitation** and
**sequential reinforcement learning**.

## 5. Limitations of the previous work that this project addresses (and those it does not)

Addressed:

- Sequential coupling is now modelled explicitly rather than ignored.
- Feasibility is enforced during decision-making instead of repaired afterwards.
- Optimality gaps are available at every DNN size, not just five layers.
- Device load and network conditions can change, so adaptivity can actually be
  measured rather than assumed.
- The marginal contribution of each component is isolated by ablation.

Not addressed, and still open in both projects:

- Workloads remain synthetic; neither project profiles real networks on real
  hardware.
- The simulator assumes sequential (chain) DNNs; branching architectures such as
  residual or inception blocks are not modelled.
- Execution is modelled as a single inference pass; pipelining, batching and
  concurrent requests are out of scope.
- Network and device parameters are sampled from hand-specified ranges rather
  than measured traces.
