# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GOAT-Lite: a scaled-down replication of GOAT (Go to Any Thing, Chang et al. RSS 2024), run in
habitat-sim and evaluated on GOAT-Bench `val_unseen` (HM3D). An episode is 5–10 sequential subtasks
in one scene; goals arrive as a category, a free-form language description, or an image. The agent
builds an instance-level semantic memory that persists across subtasks within an episode — the
"lifelong" property is the headline result, not the absolute SR.

Everything is modular and classical (YOLOv8n + CLIP + occupancy grid + A*/frontier FSM). No learned
policy, no end-to-end training. Development targets a 4 GB laptop GPU (RTX 3050); that constraint
drives fp16 everywhere, 256×256 sensors, and the model choices.

### Documentation map

- `PLAN.md` — the project spec: architecture, pinned stack, week-by-week build plan with done
  criteria, eval protocol (§8), pitfalls (§9), and session workflow (§10). Section 7.9 carries a
  dated STATUS block for wherever work stands.
- `TECHNICAL_GUIDE.md` — module-by-module walkthrough (§3), testing philosophy (§4), glossary (§5),
  and **§6 Engineering Log** — every defect found so far with the measurement that found it. Read
  §6.16–6.27 before touching mapping, geometry, or navigation; §6.27 is an *open* regression, and
  it lists what has already been ruled out so you don't re-investigate it.
- `TRAIN_YOLO.md` — the detector finetune pipeline, which runs on a *different* machine (the main
  box lacks disk for HM3D-Semantics). §0 documents the WSL2/Windows setup and the dependency
  combination that actually installs.

## Environment and commands

Everything runs in the `goat` conda env (python 3.9, headless habitat-sim). `src/sim/env.py`
imports `habitat_sim` at module scope and nearly every module imports through it, so without the
env even the pure-logic tests fail to collect.

```bash
conda activate goat
```

`requirements.txt` is *not* installable as written on a modern stack (`numpy==1.24.4` breaks
habitat-sim's quaternion ext; `ultralytics==8.1.34` fails under torch ≥ 2.6). The verified working
combination is recorded at the top of `requirements.txt` and in `TRAIN_YOLO.md` §0.

```bash
# Tests — always scope to tests/. Bare `pytest` also collects scripts/smoke_test.py
# (it matches *_test.py) and dies importing torch at module level.
pytest tests/ -q
pytest tests/test_planner.py -q
pytest tests/test_planner.py::TestAstar::test_straight_line_path -q

# VRAM/plumbing check: loads a scene + YOLO + CLIP, steps 100 times. Expect 2.6–3.0 GB peak.
python scripts/smoke_test.py

# Dev eval — 30 deterministic episodes sampled from val_unseen (seed 42).
# Use `python -u`; a full run is hours and buffered stdout hides progress.
python -u scripts/run_dev_eval.py --n-episodes 30
python -u scripts/run_dev_eval.py --n-episodes 2          # SR>0 smoke check after a nav change

# Perception + memory + matching on one scene, no navigation (writes a top-down PNG).
python scripts/midterm_demo.py --scene data/hm3d/<scene-dir>/<scene>.basis.glb

# Detector pipeline (runs on the machine that holds HM3D train + semantics; see TRAIN_YOLO.md)
python scripts/make_yolo_dataset.py --hm3d-root <hm3d>/train --out data/yolo_goat --frames-per-scene 60 --resolution 512
python scripts/finetune_yolo.py --data data/yolo_goat/data.yaml --epochs 60 --imgsz 512 --batch 16
```

On WSL2, export `HABITAT_GPU_DEVICE_ID=-1` plus the Mesa/d3d12 vars from `TRAIN_YOLO.md` §0 —
`default_gpu_device_id()` in `src/sim/env.py` reads it, and native Linux is unaffected.

### Expected on-disk data (all gitignored)

- `data/hm3d/<NNNNN-HASH>/<HASH>.basis.glb` — scenes; `_resolve_scene_path()` maps GOAT-Bench
  `scene_id` strings onto this layout.
- `data/datasets/goat_bench/hm3d/v1/val_unseen/content/*.json.gz` — episodes.
- `checkpoints/yolo_goat.pt` — finetuned 36-class detector. **Without it the detector silently
  falls back to COCO `yolov8n.pt`**, whose vocabulary covers almost none of the GOAT categories;
  SR goes to 0 and `run_dev_eval.py` prints a dropped-class-id warning at the end of each scene.
- `outputs/dev_eval_<timestamp>/` — `results.csv`, `summary.json`, `failures.jsonl`.

## Architecture

One sense-plan-act tick per step, driven by `GoatAgent.act()` (`src/agent/goat_agent.py`):

```
HabitatEnv.reset/step ──► Observation(rgb, depth, pose, compass, gps, current_goal, collided)
   │
   ├─ PerceptionPipeline ─► YOLODetector ─► crops ─► ClipEncoder ─► back-project bbox centre
   │                                                            └─► [PerceivedInstance]
   ├─ SemanticMap.update_from_depth   (log-odds occupancy + ray clearing)
   ├─ SemanticMap.update_from_detections (per-class count channels)
   ├─ InstanceDatabase.update         (merge by class + XZ distance + CLIP similarity)
   │
   └─ GoalMatcher.match(goal, db, agent_xy) ─► (InstanceNode | None, score)
          │
          FSM: SEARCHING ──match──► APPROACHING ──within 1.0 m──► VERIFYING ──seen──► DONE(stop)
                   │ frontier                │ A* + pure pursuit      │ ≥5 steps unseen
                   │ find_frontiers          │                        └─► back to APPROACHING
                   └─ choose_frontier ──► plan_astar ──► path_to_action ──► action int
```

`src/eval/runner.py` drives this per subtask: it sets the goal, calls `agent.reset(obs,
keep_memory=i>0)` — memory and map deliberately persist across subtasks within an episode — runs to
success or 500 steps, then scores with `subtask_success` + `compute_spl`. Scenes are grouped so one
habitat load serves all its episodes (a load is 5–10 s).

Layer ownership: `sim/` (habitat wrapper + GOAT-Bench parsing) → `perception/` (detector, CLIP,
per-frame pipeline) → `mapping/` (transforms, occupancy, frontiers) + `memory/` (instance DB) →
`matching/` → `planning/` (A*, FMM, frontier choice) → `agent/` (FSM, action) → `eval/`.

Empty stubs that exist only because `PLAN.md` §4 fixed the layout: `scripts/run_full_eval.py`,
`scripts/make_video.py`, `scripts/setup_hm3d.sh`, `src/utils/{viz,logging}.py`,
`src/memory/merger.py` (merging lives in `instance_db.py`).

### Configuration

`configs/*.yaml` are mostly aspirational: `map.yaml`, `perception.yaml` and `planner.yaml` are
empty, and `build_agent()` in `scripts/run_dev_eval.py` does not read `agent_default.yaml`.
**Constructor defaults in the code are the effective configuration** — change behaviour there, and
keep the YAML in sync as documentation rather than trusting it.

## Conventions that have already caused bugs

These are load-bearing. Each one has an entry in `TECHNICAL_GUIDE.md` §6 with the measurement that
exposed it; violating one produces a silently wrong agent, not a crash.

- **Frames.** Habitat world is Y-up; the map is the XZ plane with `row ← world Z`, `col ← world X`.
  The camera is **OpenGL convention**: back-projection must produce `y = -(v-cy)·d/fy`, `z = -d`.
  Building OpenCV-style points and applying habitat's quaternion mirrors the cloud through the
  camera (0% of points landed in front of the agent instead of 100%).
- **Camera pose, never agent base pose.** `HabitatEnv.get_pose()` returns the *sensor* pose; depth
  is measured from the camera, which sits 1.41 m up. `SemanticMap(camera_height=...)` then derives
  the floor per-frame as `camera_y - camera_height`, because HM3D floors are at arbitrary world Y.
- **Quaternion order.** Habitat JSON stores `(x, y, z, w)`; `numpy-quaternion.from_float_array`
  reads `(w, x, y, z)`. Always use `quaternion_from_coeffs()` — the mix-up leaves the forward vector
  intact but spawns the agent upside down.
- **Heading/actions.** Actions are ints `0=stop, 1=forward, 2=turn_left, 3=turn_right`. At heading
  `h`, forward is `(-cos h, -sin h)` in `(row, col)`; `turn_left` increases the compass.
  `path_to_action` returning 0 means "waypoint reached", *not* "end the subtask".
- **Occupancy is tri-state `{-1 unknown, 0 free, 1 occupied}` and evidence-based.** Cells accumulate
  clamped log-odds and need `min_points_per_cell` hits in a frame; ray clearing may overwrite
  occupied cells. Never write `_occupancy` directly — go through the update methods or
  `mark_occupied()` (used for collision feedback).
- **A\*: inflation is a hard constraint, unknown is passable but penalised.** Frontier goals sit on
  the unknown boundary, so blocking unknown stalls exploration; treating the inflation band as mere
  cost routes the agent through its own safety margin.
- **Map extent must come from the scene.** Build with `map_extent_for_scene(*env.get_scene_bounds())`;
  the default 24 m window on the world origin fails to contain 18 of the 36 val scenes.
- **Path following is pure pursuit** with distinct arrival radii (`ARRIVE_CELLS` for intermediate
  waypoints, `FINAL_ARRIVE_CELLS` for the last one) — a forward step is 0.25 m = 5 cells, so any
  lookahead shorter than that overshoots every waypoint.
- **Detector vocabulary.** `src/perception/goat_classes.py` is the single source of truth: 36 GOAT
  categories, ids 0–35, shared by the dataset builder, the trainer's `data.yaml`, the detector and
  the matcher. `SemanticMap` allocates 37 channels and drops out-of-vocabulary ids into
  `dropped_cls_ids` rather than crashing — a non-empty set means the wrong weights are loaded.
  Ingestion-side HM3D aliases (`stairs`→`stair`, `kitchen island`→`island`) never change emitted names.
- **CLIP embeddings are always L2-normalised**, so similarity is a plain dot product. The instance
  DB re-normalises after every running-mean merge.
- **Detector recall is the binding constraint** (mAP50 ≈ 0.149), which is why `conf` is 0.15, not
  0.35 — false positives are cheap because VERIFYING re-checks before declaring success.

## Testing

Tests use duck-typed fakes (`FakeDetector`, `FakeEncoder` returning deterministic normalised
embeddings, synthetic `Observation`s with `np.eye(4)` poses) — no GPU, no model downloads, no scene
files. Habitat-dependent tests in `tests/test_agent_smoke.py` `pytest.skip` when `data/hm3d` is empty.

The suite pins conventions, which cuts both ways: three transform tests asserted `z = +depth` and
had been encoding the camera-convention bug for weeks. When you change a geometric convention,
expect to fix tests, and check whether the test or the code was wrong.

## Working practices

From `PLAN.md` §10, and reflected in the git history:

- Tests first from the spec, then implement until they pass.
- Conventional commits (`fix(mapping): ...`, `feat(eval): ...`) with a body that states the
  **measurement** behind the change, not just the change. Feature branches per week
  (`feat/week07-planning`); `main` stays working.
- Fix one defect per commit, verified in isolation.
- Record non-obvious findings — including failed fixes and regressions — as a dated entry in
  `TECHNICAL_GUIDE.md` §6, and add dated notes to `PLAN.md` when deviating from it.
- Never commit `data/`, `checkpoints/`, `outputs/`, `wandb/`.
- `val_unseen` in full is reserved for exactly one final run; iterate on the 30-episode dev subset.
