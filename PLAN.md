# GOAT-Lite — Master Build Plan

> **How to use this file.** This document is the single source of truth for the project. Read Sections 1–4 once to understand what you're building. Read Section 5 before touching any code. From then on, the weekly sections (7.x) are self-contained: each one states what should already exist, what to build, and how to verify it. When starting a Claude Code session, paste the relevant week's section along with this preamble — Claude Code should be able to work off it without further context.

---

## 1. Project Overview

**What we're building.** A scaled-down replication of the GOAT (Go to Any Thing) navigation system from Chang et al. (RSS 2024), evaluated in the Habitat simulator on the GOAT-Bench benchmark. Given a goal specified as either an object category (e.g. `"chair"`) or a free-form language description (e.g. `"the wooden chair near the window"`), our agent must navigate to that specific object instance in a photorealistic indoor scene. Across an episode of 5–10 sequential goals in the same scene, the agent builds up an instance-aware semantic memory that makes later goals easier — this is the "lifelong" property.

**What we're deliberately *not* building.**
- Real robot deployment. Simulation only.
- Image-goal modality. Category + language only (image goals are a stretch goal).
- Fine-tuned or from-scratch neural models. All perception models are off-the-shelf, used as black boxes.
- End-to-end learned policies. The architecture is modular and classical.

**Hardware constraint.** All development on a 4 GB laptop NVIDIA GPU (RTX 3050 Laptop). Final full-benchmark evaluation runs on Kaggle Notebooks (free P100/T4 tier). This constraint is a first-class design driver, not an afterthought.

**Deliverables at end of semester.**
1. A working agent that runs on the GOAT-Bench `val_unseen` split.
2. Quantitative results: Success Rate (SR) and SPL, per modality and overall, on a mid-scale eval subset (development) and one full `val_unseen` run (final).
3. Ablations showing (a) the value of instance memory (memory on vs off), (b) the accuracy cost of small models vs large.
4. A written report (~15–25 pages) and a 3–5 minute demo video rendered from the simulator.
5. Public Git repo with reproducible setup instructions.

**Realistic success target.** 25–40% overall Success Rate on `val_unseen`. Anything above 20% is a legitimate final-year project result on this hardware. The compelling story in the report is the **lifelong-improvement delta** (SR on goal #1 vs goal #5+ within an episode) — that shows the memory is doing real work regardless of absolute numbers.

---

## 2. Architecture

The agent is a classical sense→plan→act loop with a persistent semantic memory. At every timestep:

```
   ┌────────────────────────────────────────────────────────────────┐
   │                    HABITAT SIMULATOR                            │
   │  (Stretch embodiment, HM3D scene, forward/turn/stop actions)   │
   └──────────┬────────────────────────────────────┬────────────────┘
              │ RGB, Depth, Pose, Compass          │ action
              ▼                                    │
   ┌────────────────────┐                          │
   │  PERCEPTION        │                          │
   │  - YOLOv8n / -World│                          │
   │    (object boxes)  │                          │
   │  - CLIP img encoder│                          │
   │    (per-box embed) │                          │
   └────────┬───────────┘                          │
            │ detections + embeddings              │
            ▼                                      │
   ┌────────────────────┐    ┌───────────────────┐ │
   │  SEMANTIC MAP      │    │  INSTANCE MEMORY  │ │
   │  - 2D occupancy    │───▶│  Nodes: {id, cat, │ │
   │  - explored mask   │    │   embed, xy, view,│ │
   │  - built from      │    │   first_seen_step}│ │
   │    depth + pose    │    │  Dedup by dist +  │ │
   │                    │    │  embedding sim    │ │
   └────────┬───────────┘    └─────────┬─────────┘ │
            │                          │           │
            └───────────┬──────────────┘           │
                        ▼                          │
              ┌──────────────────┐                 │
              │ GOAL MATCHING    │                 │
              │ Text goal →      │                 │
              │  CLIP text embed │                 │
              │ Score all nodes  │                 │
              │ Return best or ∅ │                 │
              └────────┬─────────┘                 │
                       ▼                           │
              ┌──────────────────┐                 │
              │ PLANNER          │                 │
              │ If match: A* to  │                 │
              │  target xy       │                 │
              │ Else: frontier   │                 │
              │  exploration     │                 │
              └────────┬─────────┘                 │
                       └─────────────────────────► │
```

**Key design decisions & why:**

- **Fixed-vocab detector (YOLOv8n) over open-vocab.** GOAT-Bench uses 36 fixed categories anyway. YOLOv8n at fp16 fits in ~30 MB VRAM and runs at ~50 FPS on a 3050. We finetune it once on the 36 categories using GOAT-Bench training scenes. Open-vocab detection (YOLO-World-S) is a Week-13 upgrade path if VRAM allows.
- **MobileCLIP-S0 for text-image matching.** ~55 MB vs 175 MB for CLIP ViT-B/32, similar accuracy on our use case. If MobileCLIP has install trouble on your setup, fall back to open_clip ViT-B/32 — the pipeline is otherwise identical.
- **Store embeddings, not images, in memory.** Every instance node holds a pre-computed 512-dim CLIP image embedding (2 KB) rather than the raw image crop. Goal matching is then a cheap cosine similarity against a list of vectors — no re-encoding on GPU during matching. Keep one small thumbnail per node for the report/video.
- **2D top-down map, not 3D.** Depth + pose → project into a 2D occupancy grid at 5 cm resolution. Enough for planning. Enough for exploration. A 3D map would be right for a research paper on mapping but wrong for our compute budget.
- **Sequential inference, single process.** No parallel perception/planning. Simpler, more predictable VRAM, easier to debug. Costs ~15% wall-clock; worth it.

---

## 3. Tech Stack (pinned)

**Python:** 3.9 (Habitat is picky; 3.10 works but 3.9 has the best wheel coverage).

**Core simulation (install order matters — see Section 5):**
- `habitat-sim==0.3.1` (conda, headless build)
- `habitat-lab==0.3.1` (pip, editable install from git tag `v0.3.1`)
- HM3D v0.2 scenes (requires Matterport agreement — start on Day 1)
- GOAT-Bench task config from `facebookresearch/habitat-lab` examples

**Deep learning:**
- `torch==2.1.2` with `cu118` wheels (works on 30-series with driver ≥ 525)
- `torchvision==0.16.2`
- `ultralytics==8.1.34` (YOLOv8n)
- `open_clip_torch==2.24.0` (fallback if MobileCLIP flaky)
- `mobileclip` (from apple/ml-mobileclip if it installs cleanly on your system)
- `lightglue` (from cvg/LightGlue) — Week 13, optional

**Numerics / classical:**
- `numpy==1.24.4` (Habitat 0.3 dislikes 2.x)
- `scipy==1.11.4`
- `scikit-image==0.22.0` (for map operations)
- `opencv-python-headless==4.8.1.78`
- `matplotlib==3.7.4` (viz only)

**Utilities:**
- `hydra-core==1.3.2` (Habitat uses this)
- `omegaconf==2.3.0`
- `wandb==0.16.2` (experiment tracking, optional but recommended)
- `tqdm`, `rich` (nice logging)

**Reproducibility.** All of the above go in `environment.yml` (conda) and `requirements.txt` (pip). Both are committed. Do **not** upgrade any of them mid-semester unless you have a specific bug forcing it — Habitat's dependency web is fragile.

---

## 4. Repository Layout

Create this exact structure on Day 1. Do not deviate — Claude Code sessions will assume these paths.

```
goat-lite/
├── PLAN.md                       # this file
├── README.md                     # setup + how-to-run for graders
├── environment.yml               # conda env spec
├── requirements.txt              # pip deps
├── LICENSE                       # MIT
├── .gitignore                    # ignore data/, checkpoints/, wandb/, outputs/
├── configs/
│   ├── agent_default.yaml        # agent hyperparams
│   ├── perception.yaml           # detector + clip settings
│   ├── map.yaml                  # map resolution, size, etc.
│   ├── planner.yaml              # planner + exploration params
│   └── eval_dev.yaml             # dev-subset eval config
├── data/                         # gitignored
│   ├── hm3d/                     # symlink to HM3D download
│   ├── goat_bench/               # GOAT-Bench episode dataset
│   └── yolo_ft/                  # YOLOv8 finetuning data
├── checkpoints/                  # gitignored, model weights
├── outputs/                      # gitignored, eval results + videos
├── src/
│   ├── __init__.py
│   ├── sim/
│   │   ├── __init__.py
│   │   ├── env.py                # HabitatEnv wrapper
│   │   └── goat_dataset.py       # GOAT-Bench dataset loading
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── detector.py           # YOLOv8 wrapper
│   │   ├── encoder.py            # CLIP image + text encoder
│   │   └── pipeline.py           # end-to-end perception per frame
│   ├── mapping/
│   │   ├── __init__.py
│   │   ├── occupancy.py          # 2D grid + updates from depth
│   │   ├── frontier.py           # frontier detection
│   │   └── transforms.py         # coordinate transforms
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── instance_db.py        # instance node + database
│   │   └── merger.py             # dedup / merging logic
│   ├── matching/
│   │   ├── __init__.py
│   │   └── goal_matcher.py       # category / language matching
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── astar.py              # A* on 2D grid
│   │   └── fmm.py                # Fast Marching alternative
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── goat_agent.py         # main agent class + state machine
│   │   └── action.py             # low-level action selection
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py            # SR, SPL calculation
│   │   ├── runner.py             # eval loop
│   │   └── report.py             # aggregate + write CSV/JSON
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── viz.py                # matplotlib visualizations
│       └── seeds.py
├── scripts/
│   ├── setup_hm3d.sh             # HM3D symlink + verify
│   ├── smoke_test.py             # env + models load + one action
│   ├── finetune_yolo.py          # YOLOv8 finetuning
│   ├── run_dev_eval.py           # eval on dev subset
│   ├── run_full_eval.py          # eval on val_unseen (Kaggle)
│   └── make_video.py             # render episode video
├── tests/
│   ├── __init__.py
│   ├── test_map.py
│   ├── test_memory.py
│   ├── test_matcher.py
│   ├── test_planner.py
│   └── test_agent_smoke.py
└── notebooks/
    ├── 01_habitat_smoke.ipynb    # sanity check environment
    ├── 02_perception_check.ipynb # detector + clip outputs
    ├── 03_map_viz.ipynb          # visualize map building
    └── 04_results_analysis.ipynb # post-eval analysis
```

**Git strategy.** `main` is always working. Feature branches for each week's work: `feat/week03-mapping`, `feat/week05-matching`, etc. Merge via PR (even if just self-reviewing) so you have a clean history to point to in the report.

---

## 5. Environment Setup (Week 1 — do this before anything else)

Everything in Week 1 is failure-prone plumbing. Follow the order below exactly. Do not "just try `pip install habitat-sim`" — it will not work.

### 5.1 Prerequisites (Day 1)
- Ubuntu 22.04 (dual-boot or clean install; **not** WSL2 for real dev — WSL2 tolerates habitat-sim but the debugging story is bad).
- NVIDIA driver ≥ 525 (`nvidia-smi` should show CUDA 11.8+).
- Miniconda installed.
- Git + git-lfs.
- ~50 GB free disk (HM3D minival + code) up front; 200 GB if you'll do full `val_unseen` locally too.

### 5.2 Conda env (Day 1)
```bash
conda create -n goat python=3.9 -y
conda activate goat
conda install habitat-sim=0.3.1 headless -c conda-forge -c aihabitat -y
```

Test:
```python
import habitat_sim
print(habitat_sim.__version__)  # should print 0.3.1
```
If this fails, do not proceed. Fix it first. Common causes: driver mismatch, wrong CUDA, mixing pip and conda installs.

### 5.3 Habitat-Lab + GOAT-Bench task (Day 1–2)
```bash
git clone --branch v0.3.1 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```

### 5.4 PyTorch (Day 2)
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

Verify:
```python
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

### 5.5 Rest of pip deps (Day 2)
Create `requirements.txt` with the versions in Section 3, then `pip install -r requirements.txt`.

### 5.6 HM3D access + download (Day 1, in parallel with above — access takes 1–3 days)
1. Register at https://matterport.com/partners/facebook and accept the terms.
2. Follow instructions at https://github.com/matterport/habitat-matterport-3dresearch to get the download token.
3. Download at minimum: `hm3d-val-habitat-v0.2.tar` (for eval) and `hm3d-minival-habitat-v0.2.tar` (fast smoke tests).
4. Extract into `data/hm3d/`.

### 5.7 GOAT-Bench episode dataset (Day 3)
The GOAT-Bench episode JSONs are hosted alongside habitat-lab data. Follow instructions in `habitat-lab/DATA.md` — specifically the goat_bench section. Land these in `data/goat_bench/`.

### 5.8 VRAM smoke test (Day 3)
Before writing any real code, write `scripts/smoke_test.py` that:
1. Loads a Habitat env with an HM3D scene at 256×256 RGB+depth.
2. Loads YOLOv8n (`ultralytics` default weights) to CUDA in fp16.
3. Loads open_clip ViT-B/32 to CUDA in fp16.
4. Steps the env forward 100 times, calling detector + CLIP each step.
5. Prints peak VRAM usage every 10 steps.

Expected peak VRAM: **~2.6–3.0 GB**. If you see > 3.5 GB, something is wrong (probably fp32 by accident). If under 2 GB, you have headroom for stretch features.

### 5.9 Kaggle setup (Day 3)
1. Create a Kaggle account (both team members).
2. Verify phone number (unlocks GPU).
3. Create a private dataset placeholder (you'll upload weights + code here later).
4. Run a "hello GPU" notebook and confirm you get a P100 or T4.

**Week 1 Done Criteria (must all be true before Week 2 starts):**
- [ ] Both team members can `conda activate goat` and run `python scripts/smoke_test.py` locally with no errors.
- [ ] `smoke_test.py` reports peak VRAM under 3 GB.
- [ ] HM3D val scenes are extracted and one loads in Habitat.
- [ ] GOAT-Bench episode JSON parses and yields at least one episode.
- [ ] Kaggle accounts have verified GPU access.
- [ ] Repo skeleton (Section 4) exists and is pushed to GitHub.

If you miss any of these, spend Week 2 finishing them. Do not build on a broken foundation.

---

## 6. Team Split (2 people)

Roles are load-bearing but not walls. Pair up for architectural decisions and code review.

### Person A — "Perception & Memory Lead"
Owns everything below `src/perception/`, `src/memory/`, `src/matching/`.
- Detector wrapper + finetuning
- CLIP encoder wrapper (image + text)
- Per-frame perception pipeline
- Instance database schema + merging logic
- Goal matching module
- Ablations on model choice and memory config

### Person B — "Mapping & Planning Lead"
Owns everything below `src/sim/`, `src/mapping/`, `src/planning/`, `src/agent/`.
- Habitat env wrapper
- Coordinate transforms (this is where ROS background pays off)
- 2D occupancy + semantic map
- Frontier detection
- A* / Fast Marching planner
- Agent state machine and action loop

### Shared (do together)
- Week 1 environment setup
- Weeks 8–9 integration (this is where the code from A and B meets)
- Weeks 10–11 evaluation runs
- Report and demo video
- Third team member (if active) picks up: eval automation scripts, video rendering, figure generation for report, README polish, running ablations while others code.

**Working rhythm.** Daily 15-min sync (voice call is fine). Weekly 90-min planning session Monday. Both PRs reviewed by the other person before merge, however briefly. Do not let PRs pile up.

---

## 7. Weekly Plan

Timeline assumes ~14 productive weeks starting the week of the Zeroth Review. Adjust dates when your actual course calendar solidifies. **The Week X sections below are self-contained — Claude Code should be able to pick up any single week and execute it.**

### 7.1 Week 1 — Environment (see Section 5)
Covered above. No agent code yet.

---

### 7.2 Week 2 — Habitat wrapper & perception scaffolding

**Prereqs:** Week 1 done criteria met.

**Person B: `src/sim/env.py`**

Build a thin wrapper around Habitat that exposes a clean API for the rest of the codebase:

```python
class HabitatEnv:
    def __init__(self, config_path: str): ...
    def reset(self, episode_id: str | None = None) -> Observation: ...
    def step(self, action: int) -> tuple[Observation, bool, dict]: ...
    # action: 0=stop, 1=forward, 2=turn_left, 3=turn_right
    def get_pose(self) -> np.ndarray: ...   # 4x4 SE(3) in world frame
    def get_scene_bounds(self) -> tuple[np.ndarray, np.ndarray]: ...
    def close(self) -> None: ...
```

`Observation` is a dataclass with `rgb: np.uint8 [H,W,3]`, `depth: np.float32 [H,W]`, `pose: np.ndarray [4,4]`, `compass: float`, `gps: np.ndarray [2]`, `current_goal: GoalSpec`.

`GoalSpec` is a dataclass with `modality: {"category", "language"}`, `value: str`, `episode_step: int`, `subtask_index: int`.

**Do:**
- Load config via Hydra.
- Set the Stretch embodiment (1.41 m height, 0.17 m radius, 25 cm step, 30° turn).
- RGB+depth at 256×256 for dev. Config-flag it up to 640×480 for final eval.
- Enforce seeds (`utils/seeds.py`).

**Verify:** `python scripts/smoke_test.py` should now use this wrapper instead of raw habitat-sim and produce identical behavior.

**Person A: `src/perception/detector.py` + `src/perception/encoder.py`**

Two thin wrappers:

`YOLODetector`:
```python
class YOLODetector:
    def __init__(self, weights: str, device: str, fp16: bool = True,
                 conf: float = 0.35, iou: float = 0.5): ...
    def detect(self, rgb: np.ndarray) -> list[Detection]: ...
```
`Detection` = `{cls_id: int, cls_name: str, conf: float, bbox: [x1,y1,x2,y2], mask: None}`. Use ultralytics YOLOv8n with default COCO weights for now — we'll swap in finetuned weights in Week 3.

`ClipEncoder`:
```python
class ClipEncoder:
    def __init__(self, model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k",
                 device: str = "cuda", fp16: bool = True): ...
    def encode_image(self, crops: list[np.ndarray]) -> np.ndarray:  # [N, D]
        ...
    def encode_text(self, prompts: list[str]) -> np.ndarray:  # [N, D]
        ...
```
All outputs L2-normalized. Try MobileCLIP-S0 first; if install is painful, use open_clip ViT-B/32 as spec'd.

**Verify:** `notebooks/02_perception_check.ipynb` — load a Habitat frame, detect, encode, print detected classes + top-5 CLIP text matches for "a photo of a chair" vs "a photo of a couch". Sanity check with your eyes.

**Week 2 Done Criteria:**
- [ ] `HabitatEnv.reset()` and `.step()` work on ≥ 3 different HM3D val scenes.
- [ ] Detector returns non-empty detections on scenes containing furniture.
- [ ] CLIP encoder produces normalized 512-dim embeddings.
- [ ] Peak VRAM under 3 GB with env + both models loaded.
- [ ] Unit tests for both modules exist (mocked env, real models).

---

### 7.3 Week 3 — Detector finetuning + per-frame perception pipeline

**Prereqs:** Week 2 done.

**Person A: `scripts/finetune_yolo.py`**

Finetune YOLOv8n on the 36 GOAT-Bench categories.

Data preparation:
1. GOAT-Bench provides a training split (`train` episodes on HM3D train scenes) with instance annotations. Iterate through ~50 train scenes, spawn the Stretch agent at random reachable locations, render RGB, project ground-truth semantic instances into pixel space to get 2D boxes.
2. Filter: only categories in the 36-class list, boxes ≥ 400 px², IoU with edge < 0.9.
3. Aim for ~15k training images, ~2k val images. Save in YOLO format under `data/yolo_ft/`.

Finetuning:
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data="data/yolo_ft/data.yaml",
    epochs=30,
    imgsz=256,
    batch=32,
    device=0,
    half=True,
    patience=8,
)
```
Save best weights to `checkpoints/yolo_goat.pt`. Target: **mAP@0.5 ≥ 0.55** on val. If below 0.4, dig into class imbalance or box quality.

**Person A: `src/perception/pipeline.py`**

```python
class PerceptionPipeline:
    def __init__(self, detector: YOLODetector, encoder: ClipEncoder,
                 min_crop_size: int = 32): ...

    def process(self, obs: Observation) -> list[PerceivedInstance]:
        # 1. Run detector on obs.rgb
        # 2. For each detection, crop and encode with CLIP image encoder
        # 3. Back-project bbox center + depth to 3D world coordinates using
        #    obs.pose and camera intrinsics
        # 4. Return list of PerceivedInstance
```

`PerceivedInstance`:
```python
@dataclass
class PerceivedInstance:
    cls_id: int
    cls_name: str
    conf: float
    bbox: tuple[int, int, int, int]
    crop_thumbnail: np.ndarray   # 64x64 for later viz
    clip_embed: np.ndarray       # [512], L2-normalized
    world_xyz: np.ndarray        # [3], meters, world frame
    seen_step: int
```

Camera intrinsics for the Stretch config are in the habitat-lab GOAT-Bench task config — extract them at env init, don't hardcode.

**Verify:** In a notebook, run the pipeline for 20 steps on one scene, plot `world_xyz` of all `PerceivedInstance`s on a top-down scatter alongside the ground-truth object positions. They should visibly cluster around real objects.

**Person B: `src/mapping/transforms.py`**

Utilities for:
- Habitat world frame ↔ map grid frame (map is XY plane, world Y is up)
- Depth image → point cloud in camera frame
- Point cloud in camera frame → world frame using pose
- Bounding-box center + depth → world xyz

Pure functions, well-tested. This is boring but critical — every downstream bug in mapping or memory will trace back here if it's wrong. Write the unit tests first.

**Week 3 Done Criteria:**
- [ ] Finetuned YOLOv8n weights saved, val mAP@0.5 ≥ 0.55.
- [ ] `PerceptionPipeline.process()` returns valid `PerceivedInstance`s on real frames.
- [ ] `transforms.py` has ≥ 8 unit tests covering forward and inverse transforms.
- [ ] `world_xyz` for a chair placed at a known location in a scene matches ground truth within 30 cm.

---

### 7.4 Week 4 — 2D semantic occupancy map

**Prereqs:** Week 3 done.

**Person B: `src/mapping/occupancy.py`**

```python
class SemanticMap:
    def __init__(self, size_m: float = 24.0, resolution_m: float = 0.05,
                 num_classes: int = 37):  # 36 + background
        # grids: occupancy [H,W] in {-1 unknown, 0 free, 1 occupied}
        #        explored [H,W] bool
        #        class_counts [H, W, num_classes] int16
        ...

    def update_from_depth(self, depth: np.ndarray, pose: np.ndarray,
                          intrinsics: np.ndarray) -> None: ...

    def update_from_detections(self, instances: list[PerceivedInstance]) -> None: ...

    def get_occupancy(self) -> np.ndarray: ...  # [H,W]
    def get_frontiers(self) -> list[tuple[int,int]]: ...
    def world_to_grid(self, xy: np.ndarray) -> tuple[int, int]: ...
    def grid_to_world(self, ij: tuple[int, int]) -> np.ndarray: ...
```

**Algorithm for `update_from_depth`:**
1. Project depth into a point cloud in world frame using `transforms.py`.
2. For each point, mark the grid cell it falls into as occupied. Mark cells along the ray from the camera to the point as free (Bresenham on the 2D projection).
3. Filter: only project points at Z ∈ [0.1 m, 1.5 m] above floor to avoid marking floor and ceiling as occupied.

**Algorithm for `update_from_detections`:**
For each `PerceivedInstance`, increment `class_counts[i, j, cls_id]` at the projected grid cell (small kernel, e.g. 3×3 Gaussian).

Update the map at ~2 Hz, not every step, to save compute. Configurable.

**Verify:** `notebooks/03_map_viz.ipynb` — run a random-walk agent for 200 steps in one scene, plot the top-down occupancy map alongside the ground-truth top-down map from Habitat. They should visibly resemble each other. Also plot the class-count map as an overlay.

**Person B: `src/mapping/frontier.py`**

```python
def find_frontiers(occupancy: np.ndarray, explored: np.ndarray,
                   min_size: int = 20) -> list[Frontier]:
    # Frontier = boundary between free and unknown, connected component
    # Return list sorted by size descending
```

`Frontier` = `{centroid_ij: tuple, size: int, cells: np.ndarray}`.

Use scikit-image's `label` on the boolean mask `(occupancy == 0) & neighbors_unknown`.

**Verify:** Test on synthetic maps (write these — a partial hallway should give one frontier at the end).

**Week 4 Done Criteria:**
- [ ] `SemanticMap.update_from_depth` produces recognizable top-down maps.
- [ ] `find_frontiers` correctly identifies frontiers on synthetic tests.
- [ ] Map update runs at < 100 ms per call on the 3050.
- [ ] Class-count grid shows chair detections concentrated near real chairs.

---

### 7.5 Week 5 — Instance memory database

**Prereqs:** Week 4 done.

**Person A: `src/memory/instance_db.py`**

The core data structure of the whole system.

```python
@dataclass
class InstanceNode:
    node_id: int
    cls_id: int
    cls_name: str
    world_xyz: np.ndarray            # [3], running mean of all merged detections
    clip_embed: np.ndarray           # [512], running mean of all merged, re-normalized
    confidence: float                # running max
    first_seen_step: int
    last_seen_step: int
    n_observations: int
    best_thumbnail: np.ndarray       # 64x64, kept from highest-confidence view

class InstanceDatabase:
    def __init__(self, merge_dist_m: float = 0.75,
                 merge_embed_sim: float = 0.85): ...
    def update(self, perceived: list[PerceivedInstance], step: int) -> None: ...
    def all_nodes(self) -> list[InstanceNode]: ...
    def query_by_class(self, cls_id: int) -> list[InstanceNode]: ...
    def query_by_embedding(self, embed: np.ndarray,
                           top_k: int = 5) -> list[tuple[InstanceNode, float]]: ...
```

**Merging logic in `update` (this is the sneaky-important part):**

For each new `PerceivedInstance`:
1. Find all existing nodes with the same `cls_id`.
2. Among those, filter to nodes within `merge_dist_m` in xy.
3. If none, create a new node.
4. If one, merge: update running means for `world_xyz` and `clip_embed` (weighted by `n_observations`), update `last_seen_step`, increment `n_observations`, keep the higher-confidence thumbnail.
5. If multiple: merge with the *closest* one. Consider merging the others if their embeddings are also similar (`> merge_embed_sim` cosine) — this handles the case where earlier noisy detections split one real instance into two nodes.

**Justify your thresholds in the report.** Sweep `merge_dist_m ∈ {0.5, 0.75, 1.0}` and `merge_embed_sim ∈ {0.80, 0.85, 0.90}` on a small dev subset in Week 10.

**Testing (`tests/test_memory.py`):**
- Add 3 detections of the same chair at slightly different xy → 1 node with `n_observations = 3`.
- Add 2 chairs 2 m apart → 2 nodes.
- Add a chair, then a chair 60 cm away with dissimilar embedding → 2 nodes (embedding overrides distance for close-but-different).

**Verify at scene level:** Run random-walk for 500 steps in a scene, inspect the resulting node list. Number of nodes should roughly match number of visible objects — an order of magnitude off means merging is broken.

**Week 5 Done Criteria:**
- [ ] `InstanceDatabase` passes all unit tests.
- [ ] After a 500-step scene walk, node count is within 2× of the ground-truth visible-object count for that scene.
- [ ] Memory of the entire DB for a full scene is < 20 MB.

---

### 7.6 Week 6 — Goal matching + Midterm demo

**Prereqs:** Week 5 done. This is your **internal midterm milestone** — an end-to-end perception + memory + matching pipeline, no navigation yet.

**Person A: `src/matching/goal_matcher.py`**

```python
class GoalMatcher:
    def __init__(self, encoder: ClipEncoder,
                 category_threshold: float = 0.0,   # category always matches
                 language_threshold: float = 0.24): ...

    def match(self, goal: GoalSpec, db: InstanceDatabase
              ) -> tuple[InstanceNode | None, float]:
        """
        For category goals: return the highest-confidence node with cls_name == goal.value,
                            or None if no such node exists.
        For language goals: encode goal.value as text, cosine-sim vs all node embeddings,
                            return best if score > threshold, else None.
        """
```

**Language matching prompt engineering.** Use CLIP-style templates: `f"a photo of {goal.value}"`. If the language description is already a full sentence like `"the chair with red cushion near the window"`, use it directly. Have one config flag `use_template`.

**Threshold calibration.** In a notebook, run matching on a dev scene where you know the ground truth, plot precision-recall as a function of `language_threshold`. Pick the elbow. Default 0.24 is a starting guess — you'll re-tune in Week 10.

**Midterm demo script (`scripts/midterm_demo.py`):**
1. Load a scene.
2. Random-walk the agent for 300 steps.
3. Print the instance DB summary (nodes, classes, positions).
4. For each of 5 hand-crafted goals (mix of category and language), run `GoalMatcher.match` and print the match + score.
5. Save a top-down plot with the map, agent trajectory, all instance nodes, and matched goal instance highlighted.

**Week 6 Done Criteria:**
- [ ] Category matching: 100% precision on scenes where the category exists in the DB.
- [ ] Language matching: 60%+ precision at threshold 0.24 on a manual eval of 20 queries across 3 scenes.
- [ ] Midterm demo plot is presentable — show it to your guide if you have a mid-semester meeting.

---

### 7.7 Week 7 — Planner and frontier exploration

**Prereqs:** Weeks 4 and 6 done.

**Person B: `src/planning/astar.py`**

A* on the 2D occupancy grid. Standard 8-connected, Euclidean heuristic. Cost: 1 for free cells, ∞ for occupied, small penalty (e.g. 3) for cells within `robot_radius / resolution` of an occupied cell (inflates obstacles).

```python
def plan_astar(occupancy: np.ndarray, start_ij: tuple[int,int],
               goal_ij: tuple[int,int], inflate_cells: int = 4
               ) -> list[tuple[int,int]] | None:
    ...
```

Alternative in `fmm.py`: Fast Marching. `skfmm.travel_time` on an inflated free-space mask, then gradient descent from start. Often smoother paths than A*. Have both, config-select.

**Person B: frontier-based exploration policy**

```python
def choose_frontier(frontiers: list[Frontier], agent_ij: tuple[int,int],
                    visited_frontiers: set) -> Frontier | None:
    # Score = size / (distance + 1) — greedy nearest-large
    # Skip frontiers within X cells of any visited_frontier
```

`visited_frontiers` is maintained by the agent — when we plan to a frontier and reach within N cells, we log it.

**Person B: action selection**

Given a path in grid cells, convert to a sequence of `{forward, turn_left, turn_right}` actions given the agent's current pose. This is the classical "path following" step. Watch out for angle wrapping.

**Verify:** In a notebook, load a scene, build a full map by exploration for 300 steps, then plan to a known coordinate. The agent should reach within 1 m in ≤ 100 additional steps.

**Week 7 Done Criteria:**
- [ ] A* returns paths on tested maps; returns None when no path exists.
- [ ] Frontier exploration in a fresh scene covers ≥ 60% of free space within 500 steps.
- [ ] Path following reaches a target within 1 m in a corridor scene.

---

### 7.8 Week 8 — Full agent integration

**Prereqs:** All prior weeks.

**Both people, together. This is the week where the two halves meet.**

**`src/agent/goat_agent.py`:**

```python
class GoatAgent:
    """
    States:
      - SEARCHING: no match in memory, exploring via frontiers
      - APPROACHING: match found, planning + executing path to it
      - VERIFYING: within threshold distance of matched node, confirm visually
      - DONE: called stop
    """
    def __init__(self, config, perception, memory, matcher, mapper, planner): ...
    def reset(self, obs: Observation) -> None: ...
    def act(self, obs: Observation) -> int: ...
```

**High-level `act()` loop:**
1. Run perception. Update map and memory with the new observations.
2. Run goal matcher against current memory.
3. State transitions:
   - SEARCHING → APPROACHING if match found with score > threshold.
   - APPROACHING → VERIFYING if agent is within `success_distance` (default 1.0 m from paper) of the matched node's `world_xyz`.
   - VERIFYING → DONE (return STOP action) if the goal object is currently in view (any detection of the matched class within some frame region) OR after 5 verification steps without visual confirmation, revert to APPROACHING and mark that node stale.
4. If APPROACHING: plan A* to match `world_xyz`, follow path.
5. If SEARCHING: pick a frontier, plan to it, follow path.
6. If no plan possible in 2 consecutive steps in either state, force STOP (fail gracefully).

**Between subtasks in an episode:** `agent.reset(obs)` is called with `keep_memory=True`. The map and instance DB persist. This is the lifelong part. Only the current goal changes.

**Timeout:** hard 500-step limit per subtask. Return STOP if exceeded.

**Verify:** Run one episode of GOAT-Bench end-to-end. Print per-subtask outcome (success or timeout). Even a 1/5 success rate here is fine — you're testing that the loop runs, not that it's good.

**Week 8 Done Criteria:**
- [ ] Agent runs through a full 5-subtask GOAT-Bench episode without crashing.
- [ ] At least one subtask succeeds on ≥ 3 different episodes.
- [ ] Per-step wall time on 3050: < 400 ms (target 200 ms).
- [ ] VRAM stable across a full episode (no leaks).

---

### 7.9 Week 9 — Development eval + debugging

> **STATUS (2026-08-08) — pick up here tomorrow.**
> Week 9 blocked on one thing: the detector. Baseline SR = 0% because the stock
> COCO YOLOv8n can't see 30/36 GOAT categories. Fix in flight: friend is
> finetuning YOLOv8n on HM3D-Semantics → will send back `checkpoints/yolo_goat.pt`
> (see `TRAIN_YOLO.md`). All wiring + two control bugs already fixed and pushed:
> - matcher image-modality routing, FSM premature-quit (commit 3398064)
> - **heading convention in `path_to_action`** (commit 2f7693d) — VERIFIED
>   end-to-end in sim: agent turns the correct way in all 4 directions and
>   drives to target. Was inverted 180° + swapped turn signs → spin-in-place.
>
> **Tomorrow's checklist (in order):**
> 1. [ ] Drop `yolo_goat.pt` into `checkpoints/` once it arrives; confirm it loads:
>        `python -c "from ultralytics import YOLO; m=YOLO('checkpoints/yolo_goat.pt'); print(len(m.names))"` (expect 36).
> 2. [ ] Smoke test: run `python scripts/run_dev_eval.py --n-episodes 2` and
>        confirm **SR > 0** (proves detector + control now close the loop).
> 3. [ ] If SR>0: launch full 30-ep dev eval in BACKGROUND with `python -u`
>        (buffered stdout otherwise), notify-on-done.
> 4. [ ] Diagnose top failure bucket from `failures.jsonl`, one fix cycle.
> 5. [ ] Then proceed to Week 10 ablations below.
>
> If SR still 0 after the finetuned detector: check (a) cls_name equality
> between detector vocab and GOAT categories, (b) `success_distance`/`world_xyz`
> noise, (c) whether `image` subtasks localize by category as intended.

**Prereqs:** Week 8 done.

**Both people: `src/eval/runner.py` and `src/eval/metrics.py`**

Metrics:
- **Success Rate (SR):** subtask succeeded iff agent called STOP within 1.0 m of the correct instance AND with the correct instance in view. Note: GOAT-Bench measures success against a specific ground-truth instance, not just any instance of the class.
- **SPL:** `success * shortest_path / max(shortest_path, actual_path)`. Shortest path is provided in the episode metadata.
- Break both down by modality and by subtask-index-in-episode (this is the lifelong curve).

Build a **30-episode dev subset** (~180 subtasks) sampled deterministically from `val_unseen`. This is your fast iteration loop for the rest of the semester. Runtime: 2–4 hours on the 3050.

Run the dev eval. Log everything to `outputs/dev_eval_<timestamp>/`:
- `results.csv` — one row per subtask
- `summary.json` — aggregated metrics
- `failures.jsonl` — one line per failure with episode_id, subtask_index, reason (timeout / wrong stop / no plan)
- Random sample of 5 failure trajectories with map + trajectory PNG

**Debugging is the actual Week 9 work.** Expect: most failures fall into 2–3 buckets. Diagnose top bucket, fix, re-run. Common early failures:
- Agent stops far from target: `success_distance` threshold, or noisy `world_xyz` in memory.
- Agent walks past target repeatedly: planner not smooth enough, or verification loop too strict.
- Wrong instance selected: language threshold too low, or embedding merging too aggressive.
- Never finds it: exploration policy loops.

**Baseline target after Week 9:** ≥ 15% SR on dev subset. If below 10%, stop and diagnose before Week 10.

**Week 9 Done Criteria:**
- [ ] `run_dev_eval.py` produces the full result artifacts.
- [ ] At least one debugging cycle completed with measurable improvement.
- [ ] Failure taxonomy documented in `outputs/failure_notes.md`.

---

### 7.10 Week 10 — Hyperparameter sweep + ablation setup

**Prereqs:** Week 9 baseline established.

**Person A: matcher and memory ablations**
- `language_threshold ∈ {0.20, 0.24, 0.28}`
- `merge_dist_m ∈ {0.5, 0.75, 1.0}` × `merge_embed_sim ∈ {0.80, 0.85, 0.90}`
- Encoder choice: MobileCLIP-S0 vs open_clip ViT-B/32

Each config → dev-subset eval → summary metrics. Small grid, ~6 configs, budget 3 days of overnight runs.

**Person B: planner and exploration ablations**
- Success distance: 0.75 m vs 1.0 m vs 1.5 m
- Planner: A* vs Fast Marching
- Frontier score: nearest-largest vs largest-only vs random

**Both: memory-on vs memory-off ablation.** This is the big one for the report. "Memory off" = clear the instance DB and map at the start of every subtask. Run on the same 30 episodes. The delta between the two is your key result.

Log everything in `outputs/ablations/`. Wandb project is very useful here — create a shared project between both team members.

**Week 10 Done Criteria:**
- [ ] All planned ablations run and logged.
- [ ] Best config identified and set as `configs/agent_default.yaml`.
- [ ] Memory on-vs-off delta computed and > 0 (if not, memory is broken — go back).

---

### 7.11 Week 11 — Full val_unseen evaluation on Kaggle

**Prereqs:** Week 10 best config locked.

**Both people:**

Package the code and models for Kaggle:
1. Push repo state to GitHub, tag as `v1.0-eval`.
2. Upload finetuned YOLO weights + any custom checkpoints as a private Kaggle dataset.
3. Create a Kaggle notebook that: `git clones` your repo at the tag, installs deps, downloads HM3D val + GOAT-Bench episodes (script provided in `scripts/setup_kaggle.sh`), runs `run_full_eval.py`.
4. First run: 50 episodes as a smoke test. Verify SR is in the same ballpark as your dev subset.
5. Full run: all 360 val_unseen episodes. Expected wall time: 6–10 hours on P100. Budget one weekly Kaggle quota window (30 GPU hours) for the run + 1 backup attempt.

Save all outputs to a Kaggle dataset, then download for local analysis.

**Have a backup plan.** If Kaggle GPU is unavailable or the run fails midway, fall back to the 100-episode mid-scale eval on local 3050 (~20 hours). This is what you cite in the report if the full run doesn't happen. Not a failure — a documented compute constraint.

**Week 11 Done Criteria:**
- [ ] Final `val_unseen` (or mid-scale fallback) results saved and analyzed.
- [ ] Per-modality SR/SPL numbers.
- [ ] Lifelong curve (SR by subtask index within episode) plotted.

---

### 7.12 Week 12 — Report writing (draft) + demo video

**Both people.**

Report structure (adjust to your college's template):
1. **Abstract** (150 words — expand from the Zeroth Review abstract)
2. **Introduction** — motivation, problem, contributions (frame the "resource-constrained replication" as the contribution)
3. **Related Work** — GOAT, GOAT-Bench, ObjectNav literature (SemExp, CoWs, PONI), semantic mapping (Chaplot 2020), foundation models in navigation
4. **Method** — architecture diagram, module descriptions, key algorithms (map update, memory merging, matching, agent state machine)
5. **Implementation** — stack, hardware, key optimizations for 4 GB
6. **Experiments** — GOAT-Bench setup, dev vs full eval, main results table, ablations, lifelong curve
7. **Discussion** — what worked, what didn't, failure analysis, comparison to published GOAT numbers (note the compute gap)
8. **Conclusion & Future Work** — image goals, open-vocab detection swap, real robot deployment
9. **References**
10. **Appendix** — full hyperparameter table, per-scene results, additional viz

Aim for 15–25 pages including figures.

Demo video (`scripts/make_video.py`):
- Pick 3 successful episodes covering both modalities.
- Render at 30 FPS: RGB view + top-down map with instance nodes + current goal text overlay.
- Cut to 90 s total. Add subtitle explanations. Simple `ffmpeg` compositing.

**Week 12 Done Criteria:**
- [ ] Report draft complete through Experiments section.
- [ ] Demo video 90 s cut ready.
- [ ] All figures generated from actual result data (not schematics).

---

### 7.13 Week 13 — Stretch features (optional) + report polish

Choose one stretch feature if the schedule allows. In priority order:

**Stretch A: Image goals (third modality).** Add `modality == "image"` to `GoalMatcher`. Encode goal image with CLIP image encoder, cosine-sim vs stored instance embeddings. Easy — mostly a matcher change. Re-run dev eval with all three modalities. ~2 days.

**Stretch B: LightGlue for image-image verification.** In VERIFYING state, use LightGlue to keypoint-match the goal image against the current view for a stronger success signal. Improves image-goal accuracy. ~3 days.

**Stretch C: Open-vocab detection swap.** Replace YOLOv8n with YOLO-World-S. Re-run key benchmarks and add an ablation row. Only if VRAM budget was underutilized in earlier smoke tests. ~2 days.

**Stretch D: Interactive live demo.** Build `scripts/interactive_demo.py` — a live mode where the user types a natural language goal into the terminal and watches the agent navigate in real-time via a matplotlib/cv2 window showing first-person RGB + top-down map updating each step. Much more impressive for presentations than pre-rendered video. ~1 day. (Added 2026-08-07.)

Do not attempt more than one. Report polish and dry-runs matter more than another feature.

Finish full report. Circulate to guide for feedback.

---

### 7.14 Week 14 — Presentation prep + submission

- Presentation slides (~15 slides): problem, related work (2 slides), architecture, method (3 slides), results (3 slides), demo video (1 slide with embed), ablations (2 slides), conclusion, Q&A.
- Two dry-runs. Time them. Cut to fit the slot.
- Final report proofread.
- Repo: clean commit history, working README, `LICENSE`, `CITATION.cff`.
- Push a final tagged release: `v1.0-final`.
- Rehearse Q&A on: why not open-vocab? why 4 GB? why not real robot? how did you validate ground truth? what's the delta from published GOAT and why?

---

## 8. Evaluation Protocol (reference — do not deviate mid-semester)

**Splits:**
- `train` — used only for YOLOv8n finetuning.
- `val_dev` — our 30-episode deterministic subset, sampled from `val_unseen` (see `src/eval/runner.py` for exact seed). Used every dev cycle.
- `val_mid` — 100-episode subset. Used for major decisions (Weeks 10–11).
- `val_unseen` — the full 360 episodes. Used exactly **once**, at the end of Week 11, on the locked config.

**Deterministic subsampling.** Fixed seed. Episode selection code committed. Reviewers can reproduce our exact subset.

**Metrics reported:**
- SR overall, SR by modality (category, language), SR by subtask index (1..10)
- SPL overall, SPL by modality
- Per-scene SR (in appendix)
- Wall-clock time per subtask (median, 95th percentile)

**What is "success"?** GOAT-Bench's own definition: agent called STOP action, and at STOP time (a) the agent is within 1.0 m euclidean of the correct instance's ground-truth position, and (b) the correct instance is within the agent's current field of view. We do not reinvent this definition.

**Reporting the memory ablation.** Report memory-on and memory-off numbers side-by-side on the same episode set. Report the lifelong curve for memory-on. Interpret honestly — if goal-5 SR is not higher than goal-1 SR, say so.

---

## 9. Common Pitfalls (learn from these upfront)

1. **Coordinate frames.** Habitat's world frame has Y up. Your 2D map is XZ. Every bug in mapping traces back here. Write and lean on the transform unit tests.
2. **Depth units.** Habitat depth is in meters as float32. If you accidentally divide by 1000 (as with some ROS conventions), everything is 1000× too close. Log a depth histogram once at env init and eyeball the range.
3. **fp16 quirks.** Some ops (softmax on large tensors, some norms) are unstable in fp16. If you see NaN in CLIP outputs, cast the specific op to fp32.
4. **YOLO input resolution.** Ultralytics rescales inputs. Do not train at 256 and infer at 640 (or vice versa). Match them, or explicitly pass `imgsz=`.
5. **CLIP normalization.** L2-normalize both image and text embeddings, always, before cosine sim. Forgetting one is the classic bug — silently gives you bad matching.
6. **Habitat headless.** On a laptop, install the `headless` variant of habitat-sim. Do not try to use the interactive viewer for dev — it fights with your display drivers.
7. **HM3D scene loading time.** ~5–10 s per scene load. Do not reload between subtasks in the same episode — reset the agent instead. This is a huge speedup for eval.
8. **Ground-truth instance selection in GOAT-Bench.** A scene may have 5 chairs. The goal targets one specific one. Your matcher must return that instance, not just any chair. The instance memory + lifelong nature is precisely what makes this tractable.
9. **Wandb from Kaggle.** Set `WANDB_API_KEY` via Kaggle secrets. Do not paste it in the notebook.
10. **Committing large files.** `data/`, `checkpoints/`, `outputs/`, `wandb/` all in `.gitignore`. Verify with `git status` before every push. If a big file sneaks in, use `git filter-repo` to purge before others pull.

---

## 10. Instructions for Claude Code Sessions

When starting a work session with Claude Code, do this:

1. `cd` into the repo. Check `git status` is clean.
2. Open a new Claude Code session in the repo.
3. Start the session by pasting:
   > "Read `PLAN.md`. Today we are working on **Week N — [name]** (Section 7.N). The Week N-1 done criteria are met. Confirm your understanding of what needs to be built this week, list the files you'll create or modify, and identify anything you'd want me to clarify before writing code."

4. Do not skip that confirmation step. If Claude Code jumps straight to code without restating the plan, ask it to back up. This catches misunderstandings before an hour is wasted.

5. For each subtask, prefer this workflow:
   - Ask Claude Code to write the **tests first** based on the spec in Section 7.N.
   - Confirm the tests are correct (this is your job — they encode the spec).
   - Ask Claude Code to implement until tests pass.
   - Review the diff before committing.

6. Commit at least once per subtask. Use conventional commits: `feat(mapping): add frontier detection`, `test(memory): merging thresholds`, etc.

7. If Claude Code proposes deviating from `PLAN.md` (different library, different algorithm), stop and think. Sometimes the deviation is right. But it must be a conscious call, not drift. Update `PLAN.md` with a dated note if you decide to change course.

8. End of each week, both team members skim the merged diffs and update the **Done Criteria checklist** in Section 7.N with actuals. This is your paper trail for the final report.

---

## 11. Deferrable Ideas (do not attempt in v1)

These would be great follow-ups but are explicitly out of scope for the final year project:
- End-to-end learned policy (imitation or RL from the modular baseline)
- Cross-episode memory (memory persisting across separate GOAT-Bench episodes in the same scene)
- Real robot transfer (Stretch or a low-cost custom base)
- 3D semantic mapping (voxel grid or NeRF-lite)
- Language description generation (agent describing what it sees)

Note them in the Future Work section of the report — they make the story feel forward-looking.

---

**Last updated:** at project start. Update the "Last updated" date and add a dated note under each Section 7.N as you close out that week.
