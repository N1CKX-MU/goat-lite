# GOAT-Lite: Multi-Modal Lifelong Object Navigation under Constrained Compute

**Project report — complete draft, 2026-10-15.** All five chapters are
written. Chapter 4 reports the final held-out evaluation.

Course: Project I (BCSE497J) · Team: Nishaanth S (23BRS1114), Muhammed Mujtaba
Dar (23BRS1118), Shaurya Jha (23BAI1464) · Guide: Dr. Nithyanandam P

> **Status note.** Every number in this document is a measurement taken from
> this repository, and the measurement that produced it is named. Where a
> result does not exist it is marked so, never estimated. Formatting will be
> adapted to the official template; only the content is fixed here.

---

## Abstract

Embodied navigation agents are usually asked for one kind of goal at a time: a
category name, or a photograph, or a sentence. GOAT (Chang et al., RSS 2024)
asks for all three in sequence within a single episode, and requires the agent
to carry an instance-level memory of what it has seen from one goal to the
next — the *lifelong* property. This report replicates that setting at a
reduced operating point: a fully modular, classical pipeline (YOLOv8n, CLIP
ViT-B/32, a log-odds occupancy grid, A\* and frontier exploration driven by a
finite state machine), with no learned policy and no end-to-end training,
constrained throughout to a 4 GB laptop GPU.

Evaluated once on the held-out GOAT-Bench `val_unseen` split — 360 episodes,
2,669 subtasks, 36 unseen HM3D scenes — the agent reaches **9.44% success rate
and 0.0830 SPL**: 13.82% on category goals, 8.27% on image goals and 5.49% on
language goals. These are modest absolute figures, and the contribution of
this work is the attributed account of *why* they are modest rather than the
figures themselves.

Three findings are the substance of that account. First, the binding
constraint is perception and not navigation: the agent explores roughly 97% of
the navigable area, and 58% of failures never match any candidate instance for
the goal. Second, the instance memory demonstrably works — it roughly doubles
the rate at which a candidate is recalled (43.6% against 27.1%, p = 0.016) with
the first subtask of each episode serving as a built-in negative control — yet
that advantage does not survive the stages between a match and a scored stop,
because the extra candidates it supplies sit a median 3.43 m from the true
object. Third, and most consequential for how such systems are evaluated, the
*lifelong curve* that this class of work reports is, in our data, an artifact:
pooled across episodes success appears to rise with subtask index, but only
long episodes contribute late subtasks and episode length is strongly
confounded with difficulty; controlling for it removes the effect entirely
(p = 0.188 paired within episode).

A supporting methodological result is that the 30-episode development subset
on which such systems are routinely iterated carries only 61 distinct goals and
has 36.7% power to detect even a ten-point change in success rate — it reported
image-goal success as exactly 0.0% where the full split gives 8.27%. The work
is accompanied by 67 dated engineering-log entries, each paired with the
measurement that produced it and including negative results and withdrawn
claims, and by a test suite of 350 cases.

**Keywords:** embodied AI, object-goal navigation, multi-modal goals, lifelong
memory, open-vocabulary perception, habitat-sim, HM3D, constrained compute.

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

| # | Objective | Status (2026-10-15) |
|---|---|---|
| 1 | Design and implement a complete modular navigation stack — detection, vision-language embedding, occupancy mapping, instance memory, multi-modal goal matching, planning and control — within a 4 GB VRAM budget | **Complete** |
| 2 | Evaluate on GOAT-Bench `val_unseen` (36 scenes, 360 episodes, 2,669 subtasks); report SR and SPL per modality | **Complete** — SR 9.44%, SPL 0.0830 over all 2,669 subtasks; category 13.82%, image 8.27%, language 5.49% |
| 3 | Quantify the lifelong effect: SR by subtask index, plus a memory-on vs memory-off ablation over identical episodes | **Complete** — attenuation chain measured: matching 43.6% vs 27.1% (p = 0.016 clustered by goal), usable matches 12.8% vs 7.8% (p = 0.08), SR 3.2% vs 3.2% (p = 1.000), with subtask 0 as a negative control |
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

## Chapter 4 — Implementation and Results

### 4.1 Implementation Status

All modules in the approved design are implemented, integrated and under test.

| Module | Status | Notes |
|---|---|---|
| Habitat wrapper, GOAT-Bench parsing | Complete | Scene grouping so one load serves all its episodes |
| Detector (finetuned YOLOv8n) | Complete | Trained to 36 GOAT categories; §4.3 |
| CLIP encoder, perception pipeline | Complete | fp16, L2-normalised embeddings |
| Occupancy mapping, frontier detection | Complete | Log-odds with ray clearing; §4.4 |
| Instance database | Complete | Three-gate merge, clearable, evidence-gated; §4.4 |
| Goal matcher (3 modalities) | Complete | Language restricted to category; image matching outstanding (§4.5.3) |
| Planning (A\*, FMM), pure pursuit | Complete | Validated against navmesh; §4.4 |
| Agent state machine | Complete | Stop decision reworked; §4.4 |
| Evaluation runner, metrics | Complete | Per-scene checkpointing; geometric visibility test (§4.5.1) |
| Debug visualiser, demo tooling | Complete | §4.2 |
| Failure-taxonomy tooling | Complete | Report tables generated from `results.csv` |

The system runs within budget: peak GPU memory stays **under 3 GB** of the 4 GB
available, and the most recent plumbing smoke test peaked at **590 MB**.

**Testing.** **200 automated tests** pass. They use dependency-injected fakes —
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

**The stop decision (2026-10-13).** Three further defects, all in the eight
lines of the state machine that decide whether to stop.

- **VERIFYING drove forward** while the agent was already inside the success
  radius, which is what pressed it nose-to-surface against the wall a mirror
  hangs on and pushed the object out of frame.
- **The 5-step verification budget was geometrically impossible.** One turn is
  30°, so five steps cannot bring a goal more than 150° off-axis into frame: a
  goal behind the agent was unconfirmable however long the subtask ran.
- **Budget expiry was a livelock.** It reverted to APPROACHING while still
  inside the radius, so the next tick re-entered VERIFYING with the counter
  reset — measured at **32 of 40 ticks** on a synthetic arrival, which is the
  500-step timeout observed on every traced subtask that reached its goal.

Verification now turns the short way toward the matched node, closes to within
0.8 of the success distance so the node's own position error cannot push the
stop outside the radius, and commits. It no longer waits for a fresh detection,
because §4.5.1 shows the detector is nearly blind at that range; the decision
rests on accumulated memory instead, which was built at distances where the
detector works. A node that cannot be brought into frame within a full 360°
survey is rejected for the rest of the subtask, and rejections are passed to the
matcher so the second-best instance can win.

Two supporting changes: `InstanceDatabase` and `SemanticMap` gained `clear()`,
so `reset(keep_memory=False)` stopped being a no-op and memory no longer leaks
between episodes — which is what made the memory ablation possible at all — and
the matcher now requires `n_observations ≥ 2`, or a single sighting above 0.5
confidence, before an instance may be targeted.

On the trace that motivated the work (episode 24, subtask 2):

| | before | after |
|---|---|---|
| Outcome | timeout at 250 steps | **STOP at step 15** |
| Closest approach | 0.19 m | 0.91 m |
| Goal in view at stop | False | **True** |
| Scored | failure | **SUCCESS** |

The closest approach moving *outward* is the fix working: the agent no longer
drives through its own viewing distance.

**The pattern that matters.** Each fix moved the failure *forward* through the
pipeline rather than sideways: detector vocabulary → map geometry → goal
reachability → detector precision → the stop decision → and now the success
criterion itself and the question of whether the goal is ever found at all. No
stage regressed after being fixed and covered by tests.

### 4.5 Results III — The Detector-Limited Success Ceiling

Since success requires the goal to be **in view** at the moment of stopping, the
honest question is not "why is SR low" but "how many subtasks are winnable at
all".

> **Correction (2026-10-13).** The figures in this section were computed while
> "in view" was implemented as *the agent's own detector firing on the final
> frame*. That is not the benchmark's criterion (§4.5.1), and the radii used
> here include two that lie **outside** the success radius. The 59.2% below is
> therefore an upper bound on "the detector can see this object from
> somewhere", not on "from a pose that could score". The in-radius ceiling,
> which is the number that actually bounds SR, is reported in §4.5.2.

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

#### 4.5.1 The criterion the agent was being held to was not the benchmark's

§3.4 adopts GOAT-Bench's success criterion: STOP within 1.0 m of a valid
instance, **and that instance within the agent's field of view**. Being within
the field of view is a geometric property of the scene and the camera. The
evaluation code decided it by running the agent's own YOLO on the final frame
and looking for the goal class.

That substitution is wrong on three counts. GOAT-Bench has no access to an
agent's internals, so it cannot be the benchmark's definition — and the episode
files ship 30 ground-truth `view_points` per goal, each with an IoU, which is
how the benchmark does express visibility. It makes SR a function of detector
recall twice over, once to find the goal and again to be credited with reaching
it, which makes the number incomparable with published GOAT results. And it is
very nearly unsatisfiable, for a reason that only became visible on measuring
it.

**Detection probability against viewing distance.** Probing one development
episode's goals at five radii, 8 angles each, asking the detector directly
whether it fires on the goal class while facing the instance:

| Radius | 0.5 m | 0.8 m | 1.2 m | 1.5 m | 2.5 m |
|---|---|---|---|---|---|
| Detector fires | 7% | 9% | 20% | **34%** | 28% |

Detection probability climbs steeply out to about 1.5 m. **Success requires a
stop inside 1.0 m**, where the detector fires from roughly one pose in twelve:
an object that fills the frame with untextured surface and no surrounding
context does not resemble the room-scale viewpoints the detector was trained
on. A geometrically perfect arrival was therefore rejected about nine times in
ten.

The scenes are not the obstacle. Over all 1,246 annotated `val_unseen` goals
and their 26,714 ground-truth viewpoints, the median viewpoint sits **1.10 m**
from its object and **90% of goals have at least one viewpoint inside 1.0 m**.
The conflict is between the detector's working range and the criterion.

Clause (b) is now `instance_in_view()`: project the ground-truth centroid into
the camera, require it inside the image and in front of it, and test occlusion
against the depth buffer. Depth rather than a semantic mask, because the local
HM3D validation scenes ship no semantic assets — a stated limitation. Success
is granted if **any** valid instance is both inside the radius and visible,
since a category subtask accepts any instance of its class.

The detector's answer is still recorded per subtask and aggregated as
`sr_detector_agrees`. **Both are reported**: the benchmark's SR, and the
stricter rate at which the agent also recognised what it had reached.

#### 4.5.2 Two limits that are properties of the embodiment, not the policy

Two further ceilings were measured, both geometric, both independent of how
good the navigation policy becomes.

**The in-radius detector ceiling.** The ceiling of §4.5 counts a subtask as
winnable if the detector fires from *any* probe radius, including 1.5 m and
2.5 m, which lie outside the success radius. The probe now measures each
snapped pose's true distance and reports the in-radius figure separately. This
is the number that bounds SR under the benchmark criterion.

**Vertical framing.** The camera sits 1.41 m up, the sensor is square with a 90°
horizontal FOV, so the vertical FOV is also 90°, and the action space has no
pitch — only stop, forward and two turns. At horizontal distance *d* the camera
sees heights `camera_y ± d`, so an instance whose centroid is *h* above or
below the camera needs at least *h* of standoff before it enters frame at all.

Over the 218 development subtasks, taking the easiest valid instance each:

| Minimum standoff for the centroid to be in frame | |
|---|---|
| p50 | 0.27 m |
| p75 | 0.56 m |
| p95 | 1.42 m |

**31 of 218 subtasks (14.2%) need more standoff than the 1.0 m success radius
allows** — the goal cannot be framed from anywhere a scoring stop could happen.
It is even across modalities (category 12.3%, language 15.7%, image 14.9%) and
concentrated in two categories: **every one of the 20 `rug` subtasks**, and 7 of
9 `carpet` subtasks. A rug's centroid lies on the floor 1.41 m below the camera,
and the downward 45° ray only reaches the floor at 1.41 m out.

This cuts both ways and the report should say so. It is partly a limitation of
`instance_in_view()`, which projects the **centroid** — the right test for a
compact object and too strict for an extended one, where part of the object can
be in frame while its centroid is not. The principled fix is the dataset's own
`view_points`, which encode visibility per pose with an IoU and need no object
extents. That change would *raise* SR, so retaining the stricter test keeps the
reported figure conservative; the affected 14.2% is named here rather than left
inside the detector's share of the blame.

#### 4.5.3 Instance ambiguity in the single-instance modalities

A language or image subtask targets **one** referenced instance, and only that
instance is credited. Over the development subset:

| Modality | n | Median instances of the goal category in the scene | With >1 |
|---|---|---|---|
| Category | 81 | 4 | 58 (72%) |
| Language | 70 | 2 | 47 (67%) |
| Image | 67 | 2 (max 47) | 41 (61%) |

The matcher routes image goals through category matching — the *nearest*
instance of the class — and never uses the goal image's pixels. The expected
chance that the nearest instance is the referenced one is **0.58 for image** and
0.49 for language. Image is 67 of 218 development subtasks, so this alone caps
image SR near 25% before any other limit applies. Implementing true image-goal
matching is therefore reclassified from an optional extension to required work
(§4.9).

### 4.6 Results IV — The full `val_unseen` evaluation

The one reserved run (Section 8 protocol), 2026-10-14/15, on the configuration
locked by the Week 10 ablations: 512 px sensors, memory on,
`language_threshold` 0.24, `merge_dist` 0.75, A*, nearest-largest frontier,
seed 42. **All 360 episodes, 2,669 subtasks, 36 unseen HM3D scenes.** 14.9 h
locally at 20.5 s/subtask.

| | n | SR | SPL | `ever_matched` |
|---|---|---|---|---|
| **overall** | **2,669** | **9.44%** | **0.0830** | 65.2% |
| category | 991 | 13.82% | 0.1238 | 73.3% |
| image | 822 | 8.27% | 0.0737 | 72.0% |
| language | 856 | 5.49% | 0.0446 | 49.4% |

**The development subset misreported nearly all of this.** It gave 3.2% SR at
256 px and 7.8% at 512, against 9.44% here, and it put image goals at exactly
**0.0%** — described in an earlier draft of this report as "a measurable zero
rather than an inactive code path". The true figure is 8.27%, 68 successes.
Section 4.6.2 and §6.64 give the reason: 30 episodes carry only 61 distinct
goals, which cannot resolve a success rate near 5%. The development subset is
adequate for perception metrics and was never capable of answering the
question it was being asked.

**Language is the weakest modality, and the cause is one stage upstream.** Its
`ever_matched` is 49.4% against roughly 73% for category and image. The agent
is not failing to navigate to described objects; it is failing to work out
which instance a description denotes. That is CLIP text-instance matching, and
it is the single clearest lever this evaluation exposes.

#### The lifelong curve, and why its obvious reading is wrong

Pooled across all episodes, success rate by position looks like the result the
project set out to find:

| subtask index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| n | 360 | 360 | 360 | 360 | 360 | 284 | 232 | 168 | 121 | 64 |
| SR | 8.1% | 8.3% | 10.3% | 11.7% | 11.1% | 10.6% | 7.3% | 4.8% | 9.1% | 12.5% |

Grouped, subtasks 0–1 score 8.19%, subtasks 2–5 score 10.92% and subtasks 6–9
score 7.52%; Fisher gives p = 0.055 early-vs-mid and p = 0.021 mid-vs-late.
Performance appears to rise as memory accumulates and fall as it fills.

**It is an artifact of mixing episode lengths.** Only long episodes have a
subtask 9, and length tracks difficulty hard:

| episode length | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| n | 380 | 312 | 448 | 376 | 513 | 640 |
| SR | 9.47% | **14.74%** | 12.95% | 8.51% | **5.85%** | 7.81% |

A 6-subtask episode scores two and a half times a 9-subtask one. The early
bins are dominated by short easy episodes, the late bins contain only long
hard ones. Two controls remove the effect completely:

- **Within the 64 episodes of exactly 10 subtasks**, where composition cannot
  change: 0–1 7.03%, 2–5 8.20%, 6–9 7.81%. p = 0.84 and p = 1.00.
- **Paired within every episode**, first half against second half, controlling
  for scene, length and difficulty at once: the second half is better in 64
  episodes, worse in 49, tied in 247. Sign test **p = 0.188**.

**Success rate does not measurably improve within an episode.** The direction
is positive and the instrument cannot resolve it, which is not the same as
memory doing nothing: Section 4.6.1 shows memory significantly improves whether a
candidate is matched at all, and the attenuation chain shows why that does not
reach the score.

So the result of this project is not the lifelong curve. It is the chain —
memory measurably improves candidate recall; the improvement does not survive
the stages between a match and a scored stop; and the pooled curve that
appears to show otherwise is an artifact of episode-length composition.

#### 4.6.1 What the development subset said, and why it was wrong

**Success Rate on the development subset: 3.2%** (7 of 218 subtasks, 30
episodes of `val_unseen`, seed 42), SPL 0.028. Two full evaluations were run,
before and after the fixes of §4.5.1, and the pair is more informative than
either number:

| | run #2 | run #3 (current) |
|---|---|---|
| SR, memory on | 2.8% (6/218) | **3.2% (7/218)** |
| SR, memory off | 2.3% (5/218) | **3.2% (7/218)** |
| SPL, memory on | 0.026 | 0.028 |
| subtasks stationary for the whole budget | 22.4% | **0%** |
| image goals that produced an embedding | 0 of 5 | **67 of 67** |

Between the runs, four defects were fixed: an unreachable goal that consumed the
whole step budget, a stop decision that halted on the success boundary, an
evidence gate on memory nodes, and image-goal embedding. **Every one is
independently verified as working, and together they bought one extra success.**
The distance between mechanism and outcome is the result of this section.

#### By modality and by position in the episode

| modality | n | SR | SPL | median final distance |
|---|---|---|---|---|
| category | 81 | 4.9% | 0.045 | 3.72 m |
| language | 70 | 4.3% | 0.036 | 6.20 m |
| image | 67 | **0.0%** | 0.000 | 5.19 m |

Image goals now embed and match (34 of 67 matched a node) and still never score.
That is a measurable zero rather than an inactive code path, which is the
difference between a known limitation and a silent one.

**The memory on/off delta on SR is exactly 0.0 pp**, paired
position-by-position over all 218 subtasks — 7/218 against 7/218, with five
discordant pairs each way and an exact two-sided sign test of p = 1.000. Run
#2's +0.5 pp sat in the same noise. That is a balanced coin, not a weak
effect, and no amount of additional episodes at this SR would resolve it: one
subtask is 1.4 pp.

**Measured on the metric memory actually acts on, the ablation is decisive.**
SR sits four stages downstream of the instance database — match, approach,
verify, stop — while `ever_matched` (did the matcher ever return a node for
this goal) is the immediate output of memory, and every subtask contributes to
it rather than only the successful 3.2%. On the same paired 218 subtasks, with
no new runs:

| metric | memory on | memory off | discordant | p | clustered by goal |
|---|---|---|---|---|---|
| SR | 3.2% | 3.2% | 5 / 5 | 1.000 | 3 / 4, p = 1.000 |
| `ever_matched` | **43.6%** | **27.1%** | **48 / 12** | < 0.0001 | **22 / 8, p = 0.016** |

The final column matters and is explained in §4.6.2: the 218 subtasks carry
only 61 distinct `(episode_id, category)` goals, so the subtask-level test
overstates its evidence. The effect survives the conservative reading.

The effect also has the shape the lifelong claim requires, which is what
separates it from a nuisance difference between two runs:

| | n | `ever_matched` on | off | discordant | p |
|---|---|---|---|---|---|
| subtask 0 | 30 | 46.7% | 43.3% | 1 / 0 | **1.000** |
| subtasks 1+ | 188 | 43.1% | 24.5% | 47 / 12 | **0.000005** |

Subtask 0 is a **built-in negative control**. It is the first goal of its
episode, so there is nothing to inherit and the two arms are the same system;
they agree, on 29 of 30 subtasks. Every later subtask inherits a map and an
instance database under memory-on and begins blind under memory-off, and there
the arms separate by 18.6 points. The effect is present exactly where the
mechanism is and absent exactly where it is not — the strongest available
evidence that it is the memory doing the work and not a difference between
runs.

The matches memory adds are, however, **mostly badly localised**, and that is
where the effect dies. `node_error_m` is the distance from the matched node to
the nearest true instance; a success needs the agent within 1 m of the goal.
Of the 48 subtasks matched only with memory on, the median node error is
**3.43 m** and only 16 (33%) fall within 1 m. Where both arms matched (n = 47)
memory does not improve localisation at all: median 4.07 m on against 3.75 m
off, p = 0.864. **Memory adds recall, not precision.**

Carried through every stage, on the same pair of runs:

| stage | memory on | memory off | discordant | p | clustered |
|---|---|---|---|---|---|
| matched anything | 43.6% | 27.1% | 48 / 12 | < 0.0001 | **0.016** |
| matched within 1 m | 12.8% | 7.8% | 22 / 11 | 0.080 | — |
| scored a success | 3.2% | 3.2% | 5 / 5 | 1.000 | 1.000 |

Memory also *appears* to make the agent faster — `steps_taken` averages 380
with memory against 422 without, 45 subtasks shorter against 22 longer,
p = 0.0067, with budget exhaustion falling from 78.4% to 73.4% — but that one
does not survive clustering by goal (19 against 11, p = 0.200). The direction
is right and the evidence is not there; it is reported here as suggestive
rather than as a result.

This chain is the central result of the project, and it is more informative
than any single row of it. The lifelong memory does its job — it roughly
doubles the rate at which the agent recalls a candidate for its goal on
subtasks with history — but two thirds of the candidates it adds sit more than
a metre from the object, so the advantage is already half gone by the
usable-match stage and entirely gone by the time it reaches a score. Both
numbers are true of the same two runs, and quoting either alone misrepresents
the system.

#### 4.6.2 Why these tests are clustered by goal

The 218 dev subtasks carry only **61 distinct `(episode_id, category)`
goals**. Episode 3 asks for the same mirror thirteen times, episode 6 the same
mirror ten times, episode 9 the same refrigerator nine times; 157 of the 218
subtasks repeat a goal already seen in their own episode. That is the
benchmark working as intended — asking for one object by category, then by
description, then by photograph is what makes it multi-modal — but it means
subtasks are **not independent observations**, and a sign test over subtasks
overstates its evidence.

The cost of ignoring this is not hypothetical. Comparing a 512 px sensor
against 256 px, the higher resolution won 13 subtasks to 4 (p = 0.049).
Six of those thirteen were episode 3's single mirror, each ending at 0.86 m or
0.93 m, five of them after a single step — the agent spawned inside the
success radius and stopped. One physical arrival, counted six times. Collapsed
to distinct goals the comparison is 6 to 3, **p = 0.508**.

Every paired test in this report therefore carries both numbers. Clustering is
conservative in the other direction — it also merges subtasks that genuinely
differ, since the same mirror asked for by photograph and by description
exercises different matching paths — so the truth lies between the two, and
printing both prevents the question being quietly skipped. A goal whose
subtasks flip both ways counts on both sides.

#### Why the failures fail

| diagnosis | n | share of failures |
|---|---|---|
| never matched any candidate | 123 | 58.3% |
| committed to the wrong object (>3 m away) | 54 | 25.6% |
| candidate was the goal; stop rule or budget lost it | 22 | 10.4% |
| candidate near the goal (1–3 m localisation error) | 12 | 5.7% |

Two numbers bound the system. Only **43.6%** of subtasks ever match a candidate,
and among the 95 that do, the matched node sits a **median of 3.83 m** from the
nearest true instance, with just **28 of 95 within 1 m**. The matcher is
simultaneously recall-starved and imprecise.

#### Exploration is not the binding constraint

The agent explores a median of **86.0 m²** of floor per subtask. The 22
development scenes have a median navigable area of **88.9 m²**. It covers
substantially the whole scene and still does not match the goal in 58% of
failures.

The clearest statement of this comes from the 51 subtasks (23.4%) that stop early
because frontier exploration is exhausted. They stop at a median of step 30 of
500 — having explored a median of **104.3 m²**, *more* than the subtasks that ran
the full budget (81.3 m²). There is genuinely nowhere left for them to go.

This bounds the remaining work. Searching better, planning better and stopping
better all act on stages the median subtask never reaches, and none of the
hyperparameters in the planned sweep acts on detection at all — so the sweep's
table should be read as a ranking at fixed perception, not as headroom. The
measurement that bounds all of them is the **in-radius detector ceiling**: how
many of the 218 subtasks are winnable by *any* navigator carrying this
detector. §4.5 establishes that the detector fires from roughly 9% of poses
at 0.8 m against 34% at 1.5 m, and §4.5.2 that 14.2% of subtasks cannot be framed
from any scoring position at all. Until the ceiling is measured end-to-end, agent
improvements are bets on headroom that may not exist.

#### The traces that localised the stop decision

These per-subtask traces drove the §4.5.1 fixes and are retained because they
isolate each stage cleanly:

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
`VERIFYING` turned the agent to *face* the object; it continued issuing forward
commands while already inside the success radius. That is now fixed, and
`mirror` is the project's best-scoring category at 10.0% of 40 subtasks.

**The second failure mode is unresolved and is the 25.6% row above.** On
`picture` subtasks, where the scene contains 25 valid instances and the detector
is at its most productive, the agent stops **confidently at the wrong object**.
Lowering detection confidence to 0.15 (§4.4) admits false positives into memory,
and a *consistent* false positive also defeats verification — the node carries
the detector's own label, so when the same wrong detection reappears on arrival,
verification passes. The evidence gate added since run #2 (two observations, or
one confident sighting) was aimed at exactly this and did not resolve it: a
false positive that is *stable across frames* clears the gate as easily as a true
one. Memory still has no mechanism for disbelief, unlike the occupancy grid,
which accumulates clamped evidence in both directions.

#### Hyperparameter sensitivity, and what a 10-episode arm can resolve

Three axes were swept over nine arms of 80 subtasks each (10 episodes of the
development subset), paired positionally so that every arm is read over the same
subtasks in the same order.

| axis | 0.20 / 0.50 / 0.75 | middle | 0.28 / 1.00 / 1.50 |
|---|---|---|---|
| language threshold | 3.8% | **6.2%** (0.24) | 1.2% |
| merge distance (m) | 5.0% | **6.2%** (0.75) | 6.2% |
| success distance (m) | 2.5% | **6.2%** (1.00) | 0.0% |

Every grid selected the configuration the system already ran, so the sweep
confirms the operating point and changes nothing in it.

It does not, however, rank the arms. The arms differ by one to five subtasks out
of eighty; an exact two-sided sign test over the paired disagreements returns
p >= 0.062 for every pair in the grid, and at this sample size six one-way flips
are needed before p < 0.05. The honest result is a statement about the
instrument: **at a success rate near 5%, a 10-episode arm is below the
resolution needed to compare configurations**, and the remaining axes were not
run for that reason rather than for lack of time.

Three observations from the paired rows are more informative than the table:

1. **The axes are not independent of the modality they name.** Raising the
   language threshold flipped two *category* subtasks, which it cannot affect
   directly. In episode 1, two language subtasks stopped matching at the higher
   threshold; the agent explored instead of approaching, and reached a later
   category subtask with a different map and a different instance database,
   which it then solved. Because memory persists across subtasks by design, an
   arm is not a per-subtask intervention but a different trajectory through the
   whole episode — the lifelong property under study is also a confound in every
   ablation of it.

2. **Scoring at the margin is close to a coin toss.** One subtask stopped 0.63 m
   from its goal and was scored a failure, and 0.68 m from it and was scored a
   success: the nearer stop lost, because success also requires the benchmark's
   geometric visibility test, which inside 1 m with a 1.41 m fixed-pitch camera
   fails more often than it passes (§4.5.2). A few centimetres of final pose,
   which any parameter change perturbs, moves the score.

3. **One axis has no effect whatsoever.** Merge distances of 0.75 m and 1.00 m
   produced byte-identical results over all 80 subtasks — the same successes,
   step counts, final distances and explored areas. No two same-class detections
   ever fell between those radii while also clearing the embedding-similarity
   gate, so the distance test is not what decides a merge at this operating
   point.

The sweep therefore reinforces §4.5: the parameters all act on stages the median
subtask never reaches. The way to make an arm distinguishable is to raise the
success rate — which means the detector — not to search the configuration space
more finely.

### 4.7 Known Limitations

The first three limitations of the previous revision — no orientation on
arrival, no evidence threshold in memory, and memory that could not be cleared —
have been addressed and are reported in §4.4. What remains:

1. **Visibility is tested on the object's centroid.** `instance_in_view()`
   projects the ground-truth centroid, which is correct for a compact object and
   too strict for an extended one: part of a rug, carpet or bed can be in frame
   while its centroid is not. This costs 14.2% of development subtasks (§4.5.2).
   The principled fix is the dataset's own `view_points` with their per-pose IoU.
   Since that change would *raise* SR, the stricter test is retained and the
   affected share is named rather than absorbed into the detector's blame.
2. **Occlusion is tested against depth, not semantics.** The local HM3D
   validation scenes ship no semantic assets, so the test allows the nearest
   visible surface to sit up to `max(0.5·range, range − 1 m)` in front of the
   centroid. Downloading `hm3d-val-semantic-annots` would permit the faithful
   semantic-mask test.
3. **Image goals use the goal image, but only as an appearance prior.** All
   822 image subtasks of the final run embedded their goal view
   (`image_goal_embedded` is true for every one), and the modality scores
   8.27% — the development subset's 0.0% was a sample-size artifact. What
   remains is that matching is a CLIP cosine ranking within the goal category
   rather than a geometric verification, so where a category has several
   instances the referenced one is still selected at close to chance
   (§4.5.3).
4. **No pitch action.** The agent cannot look up or down, which is what makes
   the vertical framing limit of §4.5.2 binding rather than merely inconvenient.
5. **Detector recall remains the binding constraint.** The detector was
   retrained to convergence (48 of 60 epochs, early-stopped on patience, best
   at epoch 33) on 2.5x the data, raising held-out mAP50 from 0.128 to 0.170
   and recall from 0.17 to 0.22. That improvement did not move end-to-end
   success measurably (p = 0.549), and six goal categories remain at zero
   recall, so the in-radius ceiling still caps the system.
6. **The evidence gate is set, not swept.** `min_observations = 2` and
   `trust_confidence = 0.5` are reasoned from the probe's confidence range but
   the values are not yet measured; {1, 2, 3} is planned on the development
   subset.
7. **Scale of evidence.** The headline numbers are the full split — 2,669
   subtasks over 36 scenes — but per-category conclusions still rest on small
   counts (`photo mount` has 2 subtasks, `hanger` 4) and should be read as
   diagnostic. The development subset used for all configuration decisions
   carries only 61 distinct goals and is underpowered for success rate
   (§4.6.2); choices locked on it should be revisited at adequate power.
8. **No outdoor environments.** All 36 scenes are building interiors, 43-258 m²
   of explored area. Goal categories that sound exterior — `parapet`,
   `handrail`, `statue` — occur as interior architectural features; the one
   long-navigation `parapet` success is an indoor room at a window. Nothing in
   this evaluation speaks to outdoor performance.

### 4.8 Comparison with the Reference Work

This work now has a number on the same split the reference work uses:
**9.44% SR / 0.0830 SPL on GOAT-Bench `val_unseen`**, from one held-out run of
all 360 episodes.

A direct "our SR versus theirs" table should be assembled from the published
figures in [1] and [2] when this document is prepared for submission; those
numbers are *not* reproduced here, because they were not re-derived in this
repository and everything else in this report is a measurement taken from it.
The comparison also needs framing rather than tabulating: GOAT uses large
detectors and keypoint-based image matching at a compute budget roughly an
order of magnitude above this one, and reports on physical robots as well as
in simulation, so the gap conflates several independent reductions.

The comparison this work can make on its own terms is more specific:

- **Which components proved load-bearing.** Perception, overwhelmingly. The
  agent explores ~97% of the navigable area and 58% of failures never match a
  candidate; neither the planner nor the exploration rule was ever the limiting
  stage, and three planner and frontier variants could not be separated at the
  resolution available (§4.6.2).
- **What the compute reduction costs, per component.** A YOLOv8n detector at
  mAP50 0.170 with six categories at zero recall; a CLIP ViT-B/32 text encoder
  that grounds descriptions to the right instance only 49.4% of the time; and a
  256 px sensor that was costing a third of the detector's own accuracy before
  it was raised to 512 (§6.60).
- **Whether the lifelong memory effect survives at this perception quality.**
  This was the hypothesis of §1.5 and the answer is partial and precise: the
  memory effect is real and significant at the stage it acts on — candidate
  recall, p = 0.016 — and is undetectable by the time it reaches success rate.
  The pooled lifelong curve that appears to show otherwise is an artifact of
  episode-length composition (§4.6). This is the finding most likely to
  generalise beyond this implementation, because it is a property of how the
  benchmark is aggregated rather than of this agent.

### 4.9 Status of the Planned Work

Everything listed as an immediate next step in the previous revision has been
carried out, with one exception:

| planned | outcome |
|---|---|
| Image-goal matching | Done. All 822 image subtasks embed their goal view; the modality scores 8.27%, against the development subset's apparent 0.0% |
| Memory on/off ablation | Done on identical episodes. Significant on candidate recall (p = 0.016), null on success (§4.6.1) |
| Complete detector training | Done. 48 of 60 epochs, early-stopped; mAP50 0.128 → 0.170, recall 0.17 → 0.22 (§4.3) |
| Re-measure the in-radius ceiling | Done (§4.5) |
| Sweep `min_observations` ∈ {1, 2, 3} | **Not done.** Deferred on measured grounds: at the available resolution no arm of any swept axis could be separated (§4.6.2), so the sweep would cost GPU time to produce a non-answer |
| Semantic assets for the validation split | Not done. Train-split semantics were obtained and used for detector training; validation semantics would replace the depth-occlusion proxy with the faithful test |

Remaining work is forward-looking rather than remedial and is set out in §5.4.

---

## Chapter 5 — Conclusion and Future Work

### 5.1 Summary of Work

A complete multi-modal lifelong object-navigation agent was built and
evaluated. Every stage is classical and inspectable: a YOLOv8n detector
finetuned to the 36-category GOAT vocabulary, CLIP ViT-B/32 embeddings for
appearance and language matching, a log-odds occupancy grid with ray clearing,
frontier exploration, A\* planning with pure-pursuit control, and a
finite-state machine arbitrating search, approach and verification. An
instance-level memory persists across the subtasks of an episode, which is the
property under test. The system runs within 3 GB of VRAM on an RTX 3050.

It was evaluated once, as the protocol in Section 3.5 requires, on the
held-out `val_unseen` split: 360 episodes, 2,669 subtasks, 36 unseen scenes,
14.9 hours of compute.

### 5.2 Principal Findings

**The agent works, modestly.** SR 9.44%, SPL 0.0830. Category goals 13.82%,
image goals 8.27%, language goals 5.49%.

**The binding constraint is perception, not navigation.** The agent covers
roughly 97% of the navigable area of a scene, and 58% of failures never match
a candidate for the goal at all. Exploration is not what is missing.

**Language is the weakest modality for a locatable reason.** Its
`ever_matched` rate is 49.4% against roughly 73% for the other two modalities.
The agent is not failing to walk to described objects; it is failing to
determine which instance a description denotes. That is CLIP text-instance
matching, and it is the single clearest lever this evaluation exposes.

**The memory works and does not reach the score.** Memory roughly doubles the
rate at which a candidate is recalled — 43.6% against 27.1%, p = 0.016 when
clustered by distinct goal — and the first subtask of each episode acts as a
built-in negative control, with the two arms agreeing on 29 of 30. But the
candidates it adds sit a median 3.43 m from the true object, so the advantage
attenuates at every subsequent stage: matching p = 0.016, usable matches
within 1 m p = 0.080, scored success p = 1.000.

**The lifelong curve is an artifact, and this is the result we would most want
carried forward.** Pooled across episodes, success rises from 8.1% at the first
subtask to 11.7% by the fourth and falls thereafter — the shape this class of
work reports. It does not survive control. Only long episodes contribute late
subtasks, and episode length is strongly confounded with difficulty
(6-subtask episodes score 14.74%, 9-subtask episodes 5.85%). Within episodes of
a fixed length the curve is flat, and paired within every episode the second
half beats the first in 64 episodes against 49, p = 0.188. Any report of a
lifelong curve that pools across episode lengths should be read with this in
mind.

**A data-availability ceiling bounds what is achievable.** Across the goal
categories with fewer than twenty annotated training instances, roughly 4% of
subtasks cannot be won by any detector trained on HM3D-Semantics;
`refrigerator` alone is 17.4% of all `val_unseen` goals against 124 training
instances.

**The development subset cannot answer the questions it was asked.** Thirty
episodes carry only 61 distinct goals and have 36.7% power to detect even a
ten-point change in success rate. It reported image-goal success as exactly
0.0% where the truth is 8.27%. Related: GOAT-Bench repeats the same object
within an episode, so subtask-level significance tests overstate themselves —
218 dev subtasks carry 61 goals, and two claims made during this work were
withdrawn once that was corrected.

### 5.3 Limitations

The agent is modular and unlearned by design, so it cannot recover from a
perception failure through policy. Detector recall is low in absolute terms
(mAP50 0.170 on held-out scenes) and six goal categories remain at zero
recall. The success criterion requires the goal to be visible from within
1.0 m, and with a 1.41 m camera, a 90° vertical field of view and no pitch
action, 14.2% of subtasks have a goal that cannot be framed from anywhere a
scoring stop could occur. All evaluation is in simulation, on building
interiors only; the split contains no outdoor environments. Finally, the
single reserved held-out run means configuration choices were locked on a
development subset now known to be underpowered.

### 5.4 Future Work

In the order we would attempt them, each targeted at a measured bottleneck:

1. **Open-vocabulary detection** in place of the closed 36-class detector,
   removing the HM3D-Semantics annotation ceiling that bounds roughly 4% of
   subtasks outright and depresses the rare categories generally.
2. **Stronger language-to-instance grounding.** Language trails the other
   modalities by 24 points of `ever_matched`; a cross-encoder or a
   region-grounded model in place of plain CLIP cosine similarity addresses
   precisely that stage.
3. **Localisation of matched instances.** Memory supplies more candidates at
   an unchanged median error of 3.8 m. Multi-view triangulation of an
   instance's position, rather than single back-projected bounding-box centres,
   would convert existing matches into successes without improving recall at
   all.
4. **A pitch action or a second camera**, to address the 14.2% of subtasks
   whose goals cannot be framed from a legal stopping pose.
5. **Evaluation at adequate power.** Development iteration should be done on
   `ever_matched` and `node_error_m`, which every subtask contributes to,
   rather than on success rate; or on a development split selected for
   distinct-goal count rather than episode count.
6. **Physical deployment**, which the modular design and the 3 GB memory
   envelope were chosen to keep feasible.

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

## List of Figures and Tables

Figures are not embedded in this Markdown source; the data behind each is in
the repository and the caption states where. When the official template is
applied, these are the intended figures:

| # | Figure | Source |
|---|---|---|
| 4.1 | System architecture: the sense-plan-act tick | §3.2 diagram |
| 4.2 | Detector precision-recall by goal category | `runs/detect/runs/goat_yolo/finetune_v2/` |
| 4.3 | Lifelong curve, pooled vs. controlled for episode length | `outputs/full_eval/results.csv`, §4.6 |
| 4.4 | Success rate and SPL by modality | `outputs/full_eval/summary.json` |
| 4.5 | The attenuation chain: matching → usable match → success | §4.6.1 |
| 4.6 | Failure taxonomy by diagnosis | `outputs/full_eval/failures.jsonl` |
| 4.7 | Top-down trajectory of a successful episode | `outputs/demo_video/` frames |

Principal tables are inline: objectives (§1.3), approach comparison (§2.6),
module responsibilities (§3.3), final results by modality (§4.6), episode
length against difficulty (§4.6), per-scene results (Appendix D) and the full
hyperparameter table (Appendix C).

---

## Appendices

- **A — Engineering log.** 67 dated findings with the measurement that produced
  each, in `TECHNICAL_GUIDE.md` §6, including negative results and failed fixes.
- **B — Reproduction.** Environment, commands and locally runnable episodes, in
  `docs/DEMO.md`.
- **C — Full hyperparameter table.** `configs/agent_default.yaml`, which
  mirrors the constructor defaults that actually run and is held to them by
  `tests/test_config.py`; each measured value is annotated with the
  engineering-log section that produced it.
- **D — Per-scene and per-category results.** Reproduced in full below, from
  the single held-out run (`outputs/full_eval/results.csv`).
- **E — Individual contribution statement.** To be completed by the team; the
  git history (`git shortlog -sn`, and the dated entries of
  `TECHNICAL_GUIDE.md` §6) is the supporting record.

### Appendix D.1 — Results by scene

All 36 `val_unseen` scenes. Scene success rate ranges from 0.0% to 39.7%, with
a median of 6.3%; two scenes yield no successes at all. The spread is wider
than any configuration change measured during development, which is worth
stating plainly: scene identity matters more to the outcome than most of the
hyperparameters that were swept.

| scene | subtasks | SR | SPL | `ever_matched` |
|---|---|---|---|---|
| `00800-TEEsavR23oF` | 73 | 5.5% | 0.054 | 32% |
| `00802-wcojb4TFT35` | 78 | 15.4% | 0.120 | 68% |
| `00803-k1cupFYWXJ6` | 71 | 32.4% | 0.273 | 86% |
| `00808-y9hTuugGdiq` | 72 | 8.3% | 0.066 | 54% |
| `00810-CrMo8WxCyVb` | 85 | 15.3% | 0.140 | 85% |
| `00813-svBbv1Pavdk` | 78 | 5.1% | 0.051 | 94% |
| `00814-p53SfW6mjZe` | 86 | 2.3% | 0.023 | 55% |
| `00815-h1zeeAwLh9Z` | 79 | 2.5% | 0.025 | 46% |
| `00820-mL8ThkuaVTM` | 78 | 24.4% | 0.225 | 92% |
| `00821-eF36g7L6Z9M` | 72 | 15.3% | 0.145 | 68% |
| `00823-7MXmsvcQjpJ` | 65 | 3.1% | 0.018 | 57% |
| `00824-Dd4bFSTQ8gi` | 63 | 3.2% | 0.032 | 63% |
| `00827-BAbdmeyTvMZ` | 69 | 7.2% | 0.040 | 59% |
| `00829-QaLdnwvtxbs` | 78 | 39.7% | 0.349 | 97% |
| `00831-yr17PDCnDDW` | 70 | 8.6% | 0.086 | 77% |
| `00832-qyAac8rV8Zk` | 66 | 6.1% | 0.061 | 70% |
| `00835-q3zU7Yy5E5s` | 72 | 2.8% | 0.028 | 57% |
| `00839-zt1RVoi7PcG` | 78 | 3.8% | 0.038 | 79% |
| `00843-DYehNKdT76V` | 76 | 1.3% | 0.005 | 25% |
| `00844-q5QZSEeHe5g` | 77 | 1.3% | 0.013 | 58% |
| `00847-bCPU9suPUw9` | 81 | 7.4% | 0.066 | 94% |
| `00848-ziup5kvtCCR` | 67 | 0.0% | 0.000 | 54% |
| `00849-a8BtkwhxdRV` | 79 | 6.3% | 0.056 | 85% |
| `00853-5cdEh9F2hJL` | 71 | 4.2% | 0.032 | 73% |
| `00861-GLAQ4DNUx5U` | 75 | 8.0% | 0.062 | 61% |
| `00862-LT9Jq6dN3Ea` | 69 | 13.0% | 0.116 | 74% |
| `00869-MHPLjHsuG27` | 68 | 26.5% | 0.226 | 66% |
| `00871-VBzV5z6i1WS` | 71 | 9.9% | 0.087 | 54% |
| `00873-bxsVRursffK` | 81 | 2.5% | 0.025 | 46% |
| `00876-mv2HUxq3B53` | 72 | 0.0% | 0.000 | 60% |
| `00877-4ok3usBNeis` | 78 | 3.8% | 0.038 | 44% |
| `00878-XB4GS9ShBRE` | 76 | 17.1% | 0.171 | 54% |
| `00880-Nfvxx8J5NCo` | 76 | 13.2% | 0.103 | 97% |
| `00890-6s7QHgap2fW` | 70 | 12.9% | 0.119 | 57% |
| `00891-cvZr5TUy5C5` | 79 | 5.1% | 0.040 | 58% |
| `00894-HY1NcmCgn3n` | 70 | 5.7% | 0.042 | 41% |

### Appendix D.2 — Results by goal category

All 36 GOAT categories as they appear in `val_unseen`. Note the shape of the
distribution: `mirror`, `refrigerator` and `picture` are 48% of all subtasks
between them, and `refrigerator` — 464 subtasks, 17.4% of the split — scores
1.7% against the 124 annotated training instances available for it.

| goal category | subtasks | SR | `ever_matched` |
|---|---|---|---|
| mirror | 483 | 22.8% | 84% |
| refrigerator | 464 | 1.7% | 75% |
| picture | 325 | 22.2% | 89% |
| dresser | 158 | 0.0% | 53% |
| rug | 156 | 0.0% | 86% |
| microwave | 132 | 2.3% | 41% |
| plant | 128 | 14.1% | 95% |
| statue | 69 | 0.0% | 0% |
| decorative plant | 68 | 29.4% | 85% |
| carpet | 57 | 0.0% | 84% |
| hanging clothes | 46 | 10.9% | 50% |
| book | 45 | 8.9% | 47% |
| glass | 44 | 0.0% | 2% |
| piano | 43 | 0.0% | 5% |
| island | 38 | 0.0% | 47% |
| calendar | 36 | 0.0% | 0% |
| parapet | 35 | 5.7% | 29% |
| handrail | 34 | 0.0% | 0% |
| nightstand | 34 | 0.0% | 44% |
| window glass | 33 | 6.1% | 88% |
| vase | 31 | 0.0% | 52% |
| pillow | 29 | 0.0% | 24% |
| freezer | 25 | 0.0% | 0% |
| flowerpot | 23 | 0.0% | 78% |
| flower vase | 21 | 23.8% | 57% |
| radiator | 18 | 0.0% | 50% |
| christmas tree | 18 | 0.0% | 0% |
| stair | 15 | 0.0% | 93% |
| shower glass | 13 | 0.0% | 0% |
| boiler | 12 | 25.0% | 0% |
| footrest | 10 | 0.0% | 40% |
| photo | 9 | 0.0% | 0% |
| exercise bike | 7 | 0.0% | 0% |
| printer | 4 | 0.0% | 0% |
| hanger | 4 | 0.0% | 25% |
| photo mount | 2 | 0.0% | 0% |
