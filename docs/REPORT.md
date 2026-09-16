# GOAT-Lite: Multi-Modal Lifelong Object Navigation under Constrained Compute

**Project report (working draft).** Chapters 1–3 are complete; Chapter 4 is an
interim account of implementation and results as of **2026-09-16**; Chapter 5
is a placeholder until the evaluation is complete.

Course: Project I (BCSE497J) · Team: Nishaanth S (23BRS1114), Muhammed Mujtaba
Dar (23BRS1118), Shaurya Jha (23BAI1464) · Guide: Dr. Nithyanandam P

> **Status note.** Every number in this document is a measurement taken from
> this repository, and the measurement that produced it is named. Where a
> result does not exist yet it is marked *pending*, not estimated. The
> end-to-end success rate is currently **0%**, and Chapter 4 explains precisely
> where the pipeline stops short. Formatting will be adapted once the official
> template is released; only the content is fixed here.

---

## Chapter 1 — Introduction

### 1.1 Background and Motivation

A service robot in a home is useless if it can only be told where to go. It has
to be told *what* to go to — "the fridge", "the wooden chair near the window",
or a photograph of a particular cushion — and work out the rest for itself. That
capability is **object-goal navigation**: placed in a building it has never
seen, with no floor plan and no prior map, an agent must explore, find the named
object, and decide by itself that it has arrived.

Three properties make the problem harder than path planning.

1. **The environment is unknown.** The agent must map and explore at the same
   time, using only what its sensors have seen so far.
2. **Goals arrive in three different forms.** A *category* ("refrigerator"), a
   *language description* ("the wooden chair near the window"), or an *image* of
   the target. These require different reasoning, but must all resolve to the
   same output: a physical position to drive to.
3. **Instances matter, not just classes.** A house contains many pictures and
   many pillows. A language or image goal refers to *one specific instance*, so
   the agent must distinguish between objects of the same class.

A fourth property is what makes the problem interesting rather than merely hard.
Real deployments are not one-shot. A robot is asked for one thing, then another,
then another, in the same building. This is **lifelong** navigation: a single
episode issues 5–10 goals in sequence in one house. An agent that remembers what
it saw while pursuing earlier goals should reach later goals faster — having
walked past a couch on the way to the refrigerator, "go to the couch" should be
immediate. Memory, not perception, is what turns a navigator into something
usable.

**GOAT** (Chang et al., RSS 2024) demonstrated exactly this: a modular system
with an instance-level memory that persists across sequential goals, supporting
all three goal modalities, deployed on physical robots. It is an existence
proof. What it does not report is which of its components are actually
load-bearing, or what happens to the system when it is made small.

That gap is the motivation for this project. Published systems report the
configuration that performs best; there is no incentive to strip a system down
until it breaks, and therefore no published account of where it breaks first.
For anyone hoping to run such a system on affordable hardware — a laptop GPU, an
embedded board, a low-cost robot — that is precisely the missing information.

### 1.2 Problem Statement

> **Determine which components of the GOAT architecture are load-bearing, and
> characterise the achievable performance of multi-modal lifelong object
> navigation under a hard compute constraint, by designing and constructing a
> minimal modular replication and measuring where and why it fails.**

The literature establishes that multi-modal lifelong object navigation is
achievable with a heavyweight modular stack. It does not establish:

- **Which components are necessary and which are redundant.** A published
  ablation removes components from a system that already works. It does not
  report the operating point at which the system stops working at all.
- **What is attainable inside a small compute envelope.** Existing work reports
  a single operating point at high compute. The performance curve below that
  point is unmeasured.
- **Whether the central claim survives unreliable perception.** GOAT's claim is
  that persistent instance memory improves performance across sequential goals.
  Memory retains whatever perception gives it, so a false detection can persist
  and be re-targeted for every later goal in the episode. Whether memory still
  helps when the detector is weak is an open question, and it is exactly the
  question a resource-constrained build is positioned to answer.

### 1.3 Objectives

| # | Objective | Status (2026-09-16) |
|---|---|---|
| 1 | Design and implement a complete modular navigation stack — detection, vision-language embedding, occupancy mapping, instance memory, multi-modal goal matching, planning and control — within a 4 GB VRAM budget | **Complete** |
| 2 | Evaluate on GOAT-Bench `val_unseen` (36 scenes, 360 episodes, 2,669 subtasks); report SR and SPL per modality | In progress — harness built, dev-subset results pending |
| 3 | Quantify the lifelong effect: SR by subtask index, plus a memory-on vs memory-off ablation over identical episodes | Pending — gated on end-to-end success |
| 4 | Attribute end-to-end failure to individual components, producing a per-module failure taxonomy backed by measurements | In progress — 32 findings logged |
| 5 | Quantify the data-availability ceiling that HM3D-Semantics imposes on the benchmark vocabulary | **Complete** — 59.2% |
| 6 | Release a reproducible public repository with a documented engineering record, including negative results | Ongoing |

### 1.4 Scope

**In scope.** A simulation-only system, built on habitat-sim and evaluated on
the GOAT-Bench `val_unseen` split, using a classical modular pipeline with no
learned navigation policy and no end-to-end training. All three goal modalities
are supported. Development targets a 4 GB laptop GPU (RTX 3050), and that
constraint drives every model choice: half precision throughout, 256×256
sensors, YOLOv8n rather than a larger detector, CLIP ViT-B/32 rather than a
large vision-language model.

**Out of scope.** Deployment on physical hardware; training a navigation policy;
open-vocabulary detection; dynamic environments or moving obstacles;
manipulation of any kind. The agent is a camera on a moving base: it observes
and navigates, and never interacts with objects.

**Deliberate constraint.** The 4 GB budget is not a limitation to apologise for;
it is the independent variable. The project's contribution comes from operating
below the compute regime the literature reports, and measuring what fails there.

### 1.5 Expected Outcomes

- **Primary quantitative result:** SR and SPL on GOAT-Bench `val_unseen`,
  reported per modality, together with SR as a function of subtask index within
  an episode — the *lifelong curve*, which tests the reference work's central
  claim directly.
- **Ablation:** memory-enabled versus memory-disabled over identical episodes,
  isolating the contribution of persistent instance memory.
- **Failure taxonomy:** a per-component attribution of end-to-end failure,
  supported by measurements, identifying which module binds performance.
- **Benchmark data ceiling:** the proportion of benchmark subtasks that no
  detector trained on HM3D-Semantics can win, with per-category detail.
- **Artefacts:** a working agent, a reproducible public repository, a
  documented engineering log including negative results, and a demonstration
  video.

**Hypothesis under test.** Instance memory acts as an *amplifier of perception
precision*. Above some precision threshold it compounds correct observations and
produces the lifelong improvement GOAT reports. Below that threshold, a false
detection admitted into memory persists and may be re-targeted for every
subsequent goal in the episode, so memory could actively degrade performance.
Confirming or refuting this would identify a precondition on the memory
mechanism that the original work does not state.

**Note on interpretation.** Results are stated as deltas and per-component
attributions rather than as competitive absolute numbers, so that they remain
meaningful at a reduced operating point. This project does not attempt to beat
published GOAT results; it attempts to explain what happens beneath them.

### 1.6 Organisation of the Report

Chapter 2 surveys object-goal navigation, contrasts the main families of
approach, and identifies the gap this work addresses. Chapter 3 gives the
methodology and system design: architecture, module-by-module description,
algorithms, experimental protocol, and the tools and datasets used. Chapter 4
reports implementation, testing and results obtained so far, with analysis and
limitations. Chapter 5 concludes and sets out future work.

---

## Chapter 2 — Literature Review

### 2.1 Object-Goal Navigation

Object-goal navigation asks an agent to find and reach an instance of a
specified object in an unseen environment, using only onboard sensing. It is
standardly evaluated in photorealistic simulation, where ground-truth object
positions and a navigation mesh make automatic scoring possible. Anderson et al.
(2018) defined the evaluation conventions still in use, including **Success
weighted by Path Length (SPL)**, which discounts success by how efficiently the
agent travelled.

Approaches fall into four families.

### 2.2 End-to-End Reinforcement Learning

A policy is learned directly from pixels to actions. Memory, where it exists, is
implicit in the weights of a recurrent network. These methods are conceptually
simple and require no hand-designed representation, but have three properties
that make them unsuitable here:

- Goals are effectively restricted to a fixed **category** vocabulary seen
  during training.
- There is **no explicit notion of object identity**, so "the chair near the
  window" cannot be distinguished from any other chair.
- Training cost is very high, and failures are **not attributable** — when the
  agent fails there is no module to point to.

### 2.3 Modular Object-Goal Navigation

Chaplot et al. (2020) separated the problem into a learned semantic mapping
module and a planning module, showing that an explicit spatial representation
improves both sample efficiency and transfer to unseen scenes. The map is
semantic but **per-episode**, and goals remain **category-only**: the
representation records *where a class was seen*, not *which object it was*.

This family establishes the architectural pattern this project follows — explicit
map, explicit planner, attributable failures — but stops short of instance-level
reasoning and multi-modal goals.

### 2.4 Open-Vocabulary and Language-Driven Navigation

A third family attaches vision-language features to a spatial representation, so
that goals can be given in natural language. CoWs (Gadre et al., 2023) applies
CLIP to zero-shot object localisation during exploration. VLMaps (Huang et al.,
2023) fuses vision-language features into a spatial map that language queries
can index. ConceptFusion (Jatavallabhula et al., 2023) builds open-set
multimodal 3D maps supporting text, image and audio queries.

These admit language goals, but share two limitations relevant here:

- Features are stored **per voxel or per pixel**, not per object. A dense
  feature map answers "where is there something matching this description", not
  "which distinct objects exist and which one is this".
- They depend on **large vision-language models**, placing them well outside a
  4 GB budget.

### 2.5 GOAT and Multi-Modal Lifelong Navigation

GOAT (Chang et al., 2024) is the direct antecedent of this project. It
introduces an **instance-level semantic memory** — a set of distinct remembered
objects, each with a position and a visual embedding — that persists across a
sequence of goals within a deployment. Goals may be given as a category, a
language description, or an image, and all three are resolved against the same
memory. The system is modular, and is demonstrated on physical robots.

GOAT-Bench (Khanna et al., 2024) is the accompanying benchmark, built on
HM3D (Ramakrishnan et al., 2021) and HM3D-Semantics (Yadav et al., 2023) in the
Habitat simulator (Savva et al., 2019). An episode is a sequence of 5–10
subtasks in one scene, with goal modality varying between subtasks. The
`val_unseen` split used here comprises 36 scenes, 360 episodes and 2,669
subtasks.

### 2.6 Comparison of Approaches

| Approach | Goal modalities | Memory representation | Instance-level? | Compute demand |
|---|---|---|---|---|
| End-to-end RL ObjectNav | Category | Implicit in policy weights | No | Very high (training) |
| Modular ObjectNav (Chaplot et al., 2020) | Category | Semantic map, per episode | No | Moderate |
| Open-vocabulary mapping (Gadre 2023; Huang 2023; Jatavallabhula 2023) | Category, language | Dense per-voxel / per-pixel features | No | High (large VLMs) |
| GOAT (Chang et al., 2024) | Category, language, image | Instance-level, lifelong | Yes | High (large detectors, keypoint matching) |
| **GOAT-Lite (this work)** | Category, language, image | Instance-level, lifelong | Yes | **Constrained to 4 GB VRAM** |

### 2.7 Identified Gap

The literature establishes *that* multi-modal lifelong object navigation works
with a heavyweight stack. It leaves three things unestablished:

1. **Component necessity.** No published account states which components are
   load-bearing, because no published system is reported at the point where it
   breaks.
2. **The low-compute operating point.** Results exist at one high-compute
   configuration. What the same architecture achieves at an order of magnitude
   less compute is unmeasured.
3. **The precondition on memory.** The claim that persistent memory helps is
   demonstrated with strong perception. Whether it still holds — or reverses —
   when perception is weak is not addressed, although this is the regime any
   resource-constrained deployment occupies.

This project addresses all three by building the smallest faithful version of
the architecture that can run, and measuring where it fails.

---

## Chapter 3 — Methodology and System Design

### 3.1 Design Principle

The system is a **classical sense-plan-act architecture** executing one cycle
per simulator step. Perception, mapping and memory update on **every** cycle
regardless of what the agent is currently doing; this is the mechanism by which
memory becomes lifelong, since objects are recorded whether or not they are
currently being sought.

Modularity is a deliberate methodological choice, not a stylistic one. The
project's objective is to attribute failure to components, which requires that
components exist as separable, individually measurable units. An end-to-end
learned policy would make the central research question unanswerable.

### 3.2 System Architecture

```
HabitatEnv.reset / step
   │  Observation(rgb, depth, pose, compass, gps, current_goal, collided)
   ▼
┌─────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌──────────┐   ┌─────────┐
│  PERCEPTION │──▶│ MAPPING + MEMORY │──▶│   MATCHING   │──▶│ PLANNING │──▶│ CONTROL │
│ YOLOv8n     │   │ occupancy grid   │   │ GoalMatcher  │   │ frontier │   │  FSM    │
│ CLIP        │   │ instance database│   │              │   │ A* / FMM │   │ action  │
└─────────────┘   └──────────────────┘   └──────────────┘   └──────────┘   └─────────┘
```

Layer ownership in the repository: `sim/` → `perception/` → `mapping/` +
`memory/` → `matching/` → `planning/` → `agent/` → `eval/`.

### 3.3 Module Descriptions

#### 3.3.1 Perception — detect, describe, localise

**Detect.** YOLOv8n produces bounding boxes and class labels from each 256×256
RGB frame. Only **4 of the 36** GOAT-Bench categories exist in the COCO
vocabulary YOLO ships with, so the detector is finetuned on images rendered from
HM3D training scenes using their semantic annotations (§3.6.2). Detection
confidence is set to **0.15**, not the conventional 0.35, for reasons measured
in §4.4.

**Describe.** Each detection crop is encoded by CLIP (OpenCLIP ViT-B/32,
LAION-2B weights) into a 512-dimensional embedding, L2-normalised so that
similarity is a plain dot product. Because CLIP places images and text in a
shared space, the similarity between a language description and a stored object
crop is directly computable.

**Localise.** The bounding-box centre pixel, with the median depth of a small
surrounding patch, is back-projected through the inverse pinhole camera model
and transformed to world coordinates using the **camera** pose (not the agent
base pose; the camera sits 1.41 m above the floor). Habitat uses the OpenGL
camera convention, so back-projection produces `y = -(v - cy)·d/fy`, `z = -d`.

#### 3.3.2 Mapping — evidence-based occupancy

Depth and pose are projected into a 2D top-down occupancy grid on the XZ plane
at **5 cm** resolution, with three states: unknown (−1), free (0), occupied (1).
The grid is derived from an underlying **clamped log-odds** accumulation:

- An observed surface contributes positive evidence to its cell, and a cell must
  receive at least `min_points_per_cell = 3` hits in a frame to count — a real
  surface fills a 5 cm cell with many pixels, a stray sample contributes one.
- **Ray clearing** between the camera and each observed surface contributes
  negative evidence, and may overwrite a cell previously marked occupied.
- Evidence is clamped, bounding how confident the map can become, so that any
  single erroneous observation remains recoverable.

The floor height is derived per frame as `camera_y − camera_height`, because
HM3D scenes place floors at arbitrary world heights. Map extent is derived from
the scene bounds rather than fixed.

#### 3.3.3 Memory — the instance database

The central representation. Each node is a distinct remembered object with a
class label, a world position, a CLIP embedding, a confidence and an observation
count. For each new detection, the database merges it into an existing node when
**all three** gates pass:

1. the class labels match,
2. the ground-plane distance is within **0.75 m**, and
3. the CLIP embedding similarity exceeds **0.85**;

otherwise a new node is created. Positions and embeddings are maintained as
running means (embeddings re-normalised after each merge), confidence as a
running maximum. This converts a stream of per-frame detections into a
persistent set of distinct object instances.

#### 3.3.4 Matching — resolving three modalities

| Modality | Resolution strategy |
|---|---|
| Category | Retrieve all nodes of the target class; select the **nearest to the agent**, since any instance satisfies the task |
| Image | Same as category — the goal image's category is known, and any instance of it satisfies the task |
| Language | Encode the description with CLIP; **restricted to nodes of the subtask's stated category**; select the highest similarity above threshold 0.24 |

Returning **no match** is a meaningful output, not a failure: it directs the
agent to keep exploring, which is strictly better than committing to a wrong
object (§4.4).

#### 3.3.5 Planning and Control — the state machine

| State | Condition | Behaviour |
|---|---|---|
| `SEARCHING` | No match in memory | Detect frontiers (boundaries between known-free and unknown space); navigate to the highest-scoring frontier, scored by `size / (distance + 1)` |
| `APPROACHING` | A match exists | Plan with A\* over the occupancy grid to the nearest standable cell **within the success radius of the object**, not the object itself; follow with pure pursuit |
| `VERIFYING` | Within the success radius | Confirm the target class is detected in the current frame before committing |
| `DONE` | Verified | Emit `STOP` |

A\* treats obstacle inflation as a **hard constraint** but allows unknown cells
at a penalty — frontier goals sit on the unknown boundary, so forbidding unknown
would stall exploration. Path following is pure pursuit with distinct arrival
radii for intermediate and final waypoints, since one forward step (0.25 m)
spans five map cells.

**Action space:** `0 = stop`, `1 = forward 0.25 m`, `2 = turn left 30°`,
`3 = turn right 30°`. Budget: 500 steps per subtask.

### 3.4 Success Criterion

Adopted from GOAT-Bench without modification. A subtask succeeds if and only if
**all three** hold:

1. the agent itself calls `STOP`,
2. it is within **1.0 m** of a valid instance of the goal, and
3. that instance is **within its field of view** at that moment.

The third condition is what couples navigation quality to perception quality: an
agent can be in exactly the right place and still fail.

### 3.5 Evaluation Protocol

**Metrics.**
- **Success Rate (SR):** fraction of subtasks meeting the criterion above.
- **SPL:** `success × shortest_path / max(shortest_path, actual_path)`.
- Both are disaggregated by **modality** and by **subtask index within the
  episode**. The latter is the *lifelong curve*, and is the headline result: it
  tests whether memory makes later goals easier.

**Splits.** A 30-episode subset (218 subtasks) is sampled deterministically from
`val_unseen` (seed 42) as the development loop. The full `val_unseen` split is
reserved for exactly one final run, to prevent the benchmark becoming a
development target.

**Memory policy.** Memory and the occupancy map persist **across subtasks within
an episode** and are cleared between episodes, so the lifelong effect is
measured within episodes only.

**Planned ablations.**
- Memory on vs. off over identical episodes — the key experiment for the
  hypothesis in §1.5.
- Matcher parameters: language threshold, merge distance, merge similarity.
- Planner: A\* vs. Fast Marching; success distance; frontier scoring rule.

### 3.6 Tools, Technologies and Datasets

#### 3.6.1 Software stack

| Layer | Component |
|---|---|
| Simulation | habitat-sim 0.3.1 (headless, Bullet), Python 3.9 |
| Detection | Ultralytics YOLOv8n |
| Embeddings | OpenCLIP ViT-B/32, LAION-2B weights, fp16 |
| Numerics | PyTorch 2.6.0+cu124, NumPy 1.26.4, SciPy, scikit-image, scikit-fmm, OpenCV |
| Engineering | pytest, Git with conventional commits, Weights & Biases |

#### 3.6.2 Datasets

| Dataset | Role |
|---|---|
| HM3D v0.2 | Photorealistic 3D scans of real interiors |
| HM3D-Semantics | Instance annotations, used to render detector training data |
| GOAT-Bench `val_unseen` | 36 scenes, 360 episodes, 2,669 subtasks — the evaluation set |

**Split separation.** Detector training data is rendered exclusively from HM3D
**train** scenes. No evaluation scene contributes a single training image, by
construction, making validation leakage structurally impossible rather than
merely avoided.

#### 3.6.3 Hardware

Development and current evaluation: NVIDIA RTX 3050 Laptop GPU, **4 GB VRAM**.
The full `val_unseen` run is planned for Kaggle (P100/T4), with a mid-scale
local run as a documented fallback.

---

## Chapter 4 — Implementation and Results *(interim, 2026-09-16)*

### 4.1 Implementation Status

All modules in the approved design are implemented, integrated and under test.

| Module | Status | Notes |
|---|---|---|
| Habitat wrapper, GOAT-Bench parsing | Complete | Scene grouping so one load serves all its episodes |
| Detector (finetuned YOLOv8n) | Complete | Trained to 36 GOAT categories; §4.3 |
| CLIP encoder, perception pipeline | Complete | fp16, L2-normalised embeddings |
| Occupancy mapping, frontier detection | Complete | Log-odds with ray clearing; §4.4 |
| Instance database | Complete | Three-gate merge; **no reset method** — §4.7 |
| Goal matcher (3 modalities) | Complete | Language restricted to category; §4.4 |
| Planning (A\*, FMM), pure pursuit | Complete | Validated against navmesh; §4.4 |
| Agent state machine | Complete | Stop logic defective; §4.6 |
| Evaluation runner, metrics | Complete | Per-scene checkpointing |
| Debug visualiser, demo tooling | Complete | §4.2 |

The system runs within budget: peak GPU memory stays **under 3 GB** of the 4 GB
available, and the most recent plumbing smoke test peaked at **590 MB**.

**Testing.** **166 automated tests** pass. They use dependency-injected fakes —
a fake detector, a fake encoder returning deterministic normalised embeddings,
synthetic observations — so the entire suite runs without a GPU, model downloads
or scene files, in about 16 seconds.

A caution recorded from experience: the suite pins conventions, which cuts both
ways. Three transform tests asserted the *wrong* camera convention and had been
encoding a serious geometric bug for weeks (§4.4). A green suite only confirms
the conventions the tests were written against.

### 4.2 Validation Methodology

Three independent sources of truth are used, in order of strength.

1. **The simulator's navigation mesh.** The map's claims about free and occupied
   space are compared cell-by-cell against habitat's own navmesh. This is
   ground truth the agent never sees, and it converts "the map looks wrong" into
   a percentage.
2. **Ground-truth goal positions.** Episode metadata gives true instance
   positions, so the error between a remembered node and the real object is
   directly measurable.
3. **A purpose-built visualiser.** `src/utils/viz.py` and
   `scripts/debug_episode.py` render, side by side: the occupancy grid, the
   current plan, the agent's actual track, every remembered instance with the
   matched one highlighted, the ground-truth goals with their success radii, the
   first-person view with detector boxes, and the live FSM state.

The visualiser was written late, and that was a mistake worth recording: it
found two substantial bugs in two frames that fourteen rounds of numerical
debugging had not.

### 4.3 Results I — Detector Finetuning

**Training data.** 5,971 images and 23,177 boxes rendered from all 145 annotated
HM3D training scenes. No evaluation scene appears anywhere in it. Training used
YOLOv8n at 512 px, batch 24, peaking at 2.2 GB VRAM.

**Caveat.** Training reached **epoch 39 of 60** before being interrupted, so the
learning-rate schedule never annealed. All figures below are a **lower bound**
on what the configuration would give.

**Overall: mAP50 = 0.149, mAP50-95 = 0.099.**

Aggregate mAP over 36 heavily imbalanced classes is the wrong lens for
navigation; what matters is recall on categories that actually appear as goals.
Per class, performance tracks training-box count almost monotonically:

| Class | Training boxes | Recall | mAP50 |
|---|---|---|---|
| picture | 4,171 | 0.53 | 0.51 |
| microwave | 268 | 0.53 | 0.57 |
| refrigerator | 404 | 0.50 | 0.32 |
| pillow | 4,653 | 0.47 | 0.34 |
| mirror | 1,188 | 0.35 | 0.35 |
| plant | 981 | 0.33 | 0.15 |
| nightstand | 202 | 0.31 | 0.42 |
| book | 1,624 | 0.17 | 0.19 |
| piano / statue / boiler / calendar / footrest | 40–62 | **0.00** | <0.05 |
| christmas tree | 0 | 0.00 | 0.00 |

**Only 7 of 36 classes reach recall ≥ 0.30.** Twelve have zero recall, typically
reporting `P = 1.000, R = 0.000` — the model emits one confident box and misses
everything else, the classic long-tail collapse.

**Observation.** This is a **data-availability** ceiling in HM3D-Semantics, not
a modelling failure. Anything built on this detector will do well on
`picture`/`pillow`/`mirror`/`refrigerator` subtasks and poorly on the tail,
independent of navigation quality.

### 4.4 Results II — Debugging by Measurement

Development proceeded by measuring against ground truth rather than inspecting
behaviour. Each defect below was found by a measurement, and the fix verified by
re-measuring the same quantity. Thirty-two findings are recorded in the
engineering log; the load-bearing ones are summarised here.

| Measurement | Before | After |
|---|---|---|
| Depth points reconstructed in front of the agent | 0% | **100%** |
| Map cells marked occupied that are actually walkable (vs navmesh) | 45% | **0%** |
| Forward commands physically blocked | 63% | **0%** |
| Subtasks with no valid plan | 10 / 17 | 0 / 17 |
| Median final distance to goal | 6.39 m | **3.31 m** |

**The most serious defect: camera convention.** Back-projection built OpenCV
points (Y down, Z = +depth) while applying a rotation from habitat's sensor
quaternion, which is OpenGL (Y up, camera looks down −Z). Mixing the two mirrors
the point cloud through the camera. Measured with the agent upright, the
fraction of reconstructed points landing in front of the agent was **0.0%**
under the code's convention and **100.0%** under habitat's. Every occupancy cell
and every memory node was displaced; instance nodes sat roughly 3 m from the
true object, which is why the agent could stand 0.60 m from a dresser and
believe its target was metres away.

**Other defects worth reporting in full:**

- **Occupancy was one-shot and irreversible.** A single depth pixel marked a
  cell occupied forever, and ray clearing was gated on "only if not already
  occupied". Projection error accumulated monotonically, so the map could only
  degrade as an episode ran. Replaced with the evidence-based scheme of §3.3.2:
  blocked forwards 63% → 0%, occupied cells 12,057 → 640.
- **The confidence gate blinded the agent.** At `conf = 0.35` the detector fired
  in only 20% of frames; at 0.15, 50%. Memory stayed empty and the matcher had
  nothing to work with. Lowered to 0.15, accepting false positives as the cost —
  a decision revisited in §4.6.
- **The matcher chased confidence, not proximity.** For category goals the agent
  walked past an instance 0.98 m away — inside the success radius — to pursue a
  better-scored one across the scene. Now selects on distance.
- **Language goals could never be verified.** Verification compared the detected
  class name against the goal *value*, which for a language goal is the entire
  description. Verification therefore always failed and the agent looped to
  timeout **while standing on the target**. This affected 856 of 2,669
  `val_unseen` subtasks (32%) structurally.
- **Language goals matched any class.** Language matching scored the description
  against every remembered instance regardless of class. With no microwave in
  memory, a window scored 0.257 against a 0.24 threshold and won; the agent
  navigated to it, verified it (the node carries the detector's own label, so the
  false detection confirms itself) and stopped **12.5 m** from the real target.
  Language matching is now restricted to the subtask's stated category.
- **Wall-mounted goals were unreachable by construction.** Pictures, mirrors and
  radiators sit on walls, so an instance's estimated position lands on an
  occupied cell and A\* refused to plan to it. Most of the vocabulary was
  affected. The agent now plans to the nearest free cell within the success
  radius, which is what the success criterion actually asks for.

**The pattern that matters.** Each fix moved the failure *forward* through the
pipeline rather than sideways: detector vocabulary → map geometry → goal
reachability → detector precision → the final stop decision. No stage regressed
after being fixed and covered by tests.

### 4.5 Results III — The Detector-Limited Success Ceiling

Since success requires the goal to be **in view** at the moment of stopping, and
"in view" means the detector fires on it, the honest question is not "why is SR
low" but "how many subtasks are winnable at all".

**Method.** For every ground-truth goal instance in the 30-episode development
subset, the agent is parked at a ring of navigable poses facing the object — 8
angles × 3 radii (0.8 m, 1.5 m, 2.5 m) = **24 poses** — and the detector is
queried directly. If it never fires on the goal class from any of them, no
navigator can win that subtask.

**Result over 218 subtasks:**

| | |
|---|---|
| Detector fires on the goal from at least one pose | **129 (59.2%)** |
| **Unwinnable by any navigator** | **89 (40.8%)** |

| By modality | Ceiling |
|---|---|
| Category | 75.3% |
| Language | 55.7% |
| Image | 43.3% |

**Seven categories sit at 0%:** `calendar`, `christmas tree`, `island`, `glass`,
`statue`, `piano`, `hanging clothes` — 32 subtasks that cannot succeed however
good navigation becomes.

**Analysis.** Training-set size predicts the ceiling only loosely.
`pillow`/`picture`/`book` (4,653/4,171/1,624 boxes) reach 100%, and
`christmas tree` at 0 boxes is trivially 0%. But `refrigerator` has 404 boxes and
reaches only 26.7%, while `hanging clothes` has 241 and reaches 0%. For some
categories the binding factor is domain shift between HM3D train and validation
scenes, or object scale, rather than raw data volume. The error analysis should
say so rather than implying a clean correlation.

**Two consequences.**

1. **A 2-episode smoke test is a poor benchmark.** Episode 0 of the development
   subset has a ceiling of **0%**, and its 7 subtasks were 7 of the 17 being
   iterated against. Some tuning was therefore measured against noise. Smoke
   tests must be drawn from episodes with a non-zero ceiling.
2. **The ceiling does not excuse the result.** 59.2% being winnable means
   SR = 0% still reflects genuine navigation and stop-logic gaps. SR should be
   reported against the achievable subset as well as the raw total.

### 4.6 Results IV — End-to-End Status

**Current end-to-end Success Rate: 0%** (2-episode smoke test, 17 subtasks). All
subtasks end in timeout or a stop at the wrong object. What has changed over
development is not the number but **where the failure sits**.

Per-subtask traces on the development episodes, 250 steps each:

| Episode / subtask | Goal | Closest approach | Outcome |
|---|---|---|---|
| 24 / 2 | category/mirror | **0.19 m** | Reached the goal; never confirmed it |
| 19 / 0 | category/picture | 3.20 m | Called `STOP` at the wrong instance, 3.72 m out |
| 19 / 5 | category/picture | 2.79 m | Never approached |
| 14 / 0 | category/decorative plant | 3.52 m | Never approached |

**The decisive trace is episode 24, subtask 2.** The goal is a mirror with one
valid instance in the scene:

| Quantity | Value |
|---|---|
| Error between matched memory node and true instance | **0.15 m** |
| Closest approach to the true goal | **0.19 m** (step 36) |
| Success radius | 1.00 m |
| State on arrival | `VERIFYING` |
| `goal_in_view` | **False, for the entire subtask** |
| Outcome | Timeout at 0.40 m |

**This isolates the failure precisely.** Memory located the mirror to within
0.15 m — perception and the instance database both worked. A\* and pure pursuit
delivered the agent to 0.19 m of it — planning and control both worked. What
failed is the final decision. The agent approaches the *node position*, which for
a wall-mounted object lies on the wall, so it arrives nose-to-surface with the
object out of frame, and `goal_in_view` can never become true. Nothing in
`VERIFYING` turns the agent to *face* the object; it continues issuing forward
commands while already inside the success radius.

**The remaining failure mode is different and unresolved.** On `picture`
subtasks, where the scene contains 25 valid instances and the detector is at its
most productive, the agent stops **confidently at the wrong object**. Lowering
detection confidence to 0.15 (§4.4) admits false positives into memory, and a
*consistent* false positive also defeats verification — the node carries the
detector's own label, so when the same wrong detection reappears on arrival,
verification passes. Memory has no mechanism for disbelief: unlike the occupancy
grid, which accumulates clamped evidence in both directions, a node created from
a single low-confidence detection is as targetable as one seen fifty times.

### 4.7 Known Limitations

1. **The stop decision does not orient toward the goal** (§4.6). The identified
   fix is a bearing check on arrival: turn toward the matched node until it is in
   frame, then decide.
2. **Instance memory has no evidence threshold.** A node observed once at
   confidence 0.15 can become a navigation target. The candidate fix mirrors the
   occupancy grid's log-odds scheme: require repeated observations, or an
   aggregate confidence, before a node is targetable.
3. **Memory is not cleared between episodes.** `InstanceDatabase` has no reset
   method, so `reset(keep_memory=False)` is currently a no-op, and the agent is
   constructed once per scene. Memory and map therefore leak across episodes
   within a scene. This confounds the lifelong curve and makes the memory on/off
   ablation — the key experiment of §3.5 — unimplementable until fixed.
4. **The detector is undertrained** (epoch 39 of 60), and its ceiling caps the
   whole system at 59.2%.
5. **No full evaluation has been run.** Results to date come from smoke tests
   and traces, not from the 30-episode development evaluation.
6. **Scale of evidence.** Conclusions currently rest on tens of subtasks, not
   hundreds. They should be treated as diagnostic, not statistical.

### 4.8 Comparison with the Reference Work

A quantitative comparison with published GOAT results is **not yet meaningful**,
and will be framed carefully when it is. GOAT operates with large detectors and
keypoint-based image matching at a compute budget an order of magnitude above
this one, and reports results on physical robots as well as in simulation.

The intended comparison is therefore not "our SR versus theirs" but:

- **Which components proved load-bearing** in a minimal implementation, and
  which could be reduced without consequence;
- **What the compute reduction costs**, expressed per component;
- **Whether the lifelong memory effect survives** at this perception quality —
  the hypothesis of §1.5, and the one finding that would generalise beyond this
  implementation.

### 4.9 Immediate Next Steps

1. **Bearing check on arrival** — turn toward the matched node before deciding,
   in place of blind forward motion inside the success radius (§4.6).
2. **Evidence-gated memory** — require repeated observation before a node
   becomes targetable (§4.7.2).
3. **Add `clear()` to the instance database and map**, and honour
   `keep_memory=False`, unblocking the memory ablation (§4.7.3).
4. **Run the full 30-episode development evaluation** and populate the results
   tables. All 36 `val_unseen` scenes now resolve locally, so this is no longer
   blocked on data.
5. **Complete detector training** to epoch 60 and re-measure both mAP and the
   ceiling.

---

## Chapter 5 — Conclusion and Future Work

*To be written once the development evaluation and ablations are complete.*

Planned content: summary of work completed; major findings (the failure
taxonomy, the 59.2% data ceiling, and the outcome of the memory hypothesis);
conclusion; limitations; and future enhancements — completing detector training,
open-vocabulary detection, stronger image-goal verification via keypoint
matching, and deployment on physical hardware.

---

## References

1. Chang, M., Gervet, T., Khanna, M., et al. "GOAT: GO to Any Thing." *Robotics: Science and Systems (RSS)*, 2024.
2. Khanna, M., Ramrakhya, R., Chhablani, G., et al. "GOAT-Bench: A Benchmark for Multi-Modal Lifelong Navigation." *CVPR*, 2024.
3. Chaplot, D. S., Gandhi, D., Gupta, A., Salakhutdinov, R. "Object Goal Navigation using Goal-Oriented Semantic Exploration." *NeurIPS*, 2020.
4. Gadre, S. Y., Wortsman, M., Ilharco, G., et al. "CoWs on Pasture: Baselines and Benchmarks for Language-Driven Zero-Shot Object Navigation." *CVPR*, 2023.
5. Huang, C., Mees, O., Zeng, A., Burgard, W. "Visual Language Maps for Robot Navigation." *ICRA*, 2023.
6. Jatavallabhula, K. M., Kuwajerwala, A., Gu, Q., et al. "ConceptFusion: Open-set Multimodal 3D Mapping." *RSS*, 2023.
7. Ramakrishnan, S. K., Gokaslan, A., Wijmans, E., et al. "Habitat-Matterport 3D Dataset (HM3D)." *NeurIPS Datasets and Benchmarks*, 2021.
8. Yadav, K., Ramrakhya, R., Ramakrishnan, S. K., et al. "Habitat-Matterport 3D Semantics Dataset." *CVPR*, 2023.
9. Savva, M., Kadian, A., Maksymets, O., et al. "Habitat: A Platform for Embodied AI Research." *ICCV*, 2019.
10. Radford, A., Kim, J. W., Hallacy, C., et al. "Learning Transferable Visual Models From Natural Language Supervision." *ICML*, 2021.
11. Anderson, P., Chang, A., Chaplot, D. S., et al. "On Evaluation of Embodied Navigation Agents." arXiv:1807.06757, 2018.
12. Yamauchi, B. "A Frontier-Based Approach for Autonomous Exploration." *IEEE CIRA*, 1997.
13. Elfes, A. "Using Occupancy Grids for Mobile Robot Perception and Navigation." *Computer*, 22(6), 1989.
14. Hart, P. E., Nilsson, N. J., Raphael, B. "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." *IEEE Trans. Systems Science and Cybernetics*, 4(2), 1968.

---

## Appendices *(to be assembled)*

- **A — Engineering log.** 32 dated findings with the measurement that produced
  each, in `TECHNICAL_GUIDE.md` §6, including negative results and failed fixes.
- **B — Reproduction.** Environment, commands and locally runnable episodes, in
  `docs/DEMO.md`.
- **C — Full hyperparameter table.** *Pending.*
- **D — Per-scene and per-category results.** *Pending the development evaluation.*
- **E — Individual contribution statement.** *Pending.*
