# GOAT-Lite

A scaled-down replication of **GOAT** ("Go to Any Thing", Chang et al., RSS
2024), run in habitat-sim and evaluated on **GOAT-Bench** `val_unseen` (HM3D).
An episode is 5–10 sequential subtasks in one scene, and each goal arrives as a
**category**, a **free-form language description** or an **image**. The agent
builds an instance-level semantic memory that persists across the subtasks of an
episode — the "lifelong" property is the subject of the experiment, not the
absolute success rate.

Everything is modular and classical: YOLOv8n + CLIP + an occupancy grid +
A\*/frontier exploration driven by a state machine. **No learned policy and no
end-to-end training.** Development targets a **4 GB laptop GPU**, and that
constraint is the independent variable rather than an apology — it drives fp16
throughout, 256×256 sensors, and every model choice. Peak usage stays under 3 GB.

## What this repository is for

The interesting output is not a leaderboard number but an **attributed account
of what fails** at a reduced operating point. Six results so far:

- **The headline evaluation.** On the full GOAT-Bench `val_unseen` split — 360
  episodes, 2,669 subtasks, 36 unseen HM3D scenes, one reserved run —
  **SR 9.44%, SPL 0.0830**: category 13.82%, image 8.27%, language 5.49%.
  Language trails because its matcher does, not its navigator: `ever_matched`
  is 49.4% against ~73% for the other two modalities. The 30-episode
  development subset had reported 3.2% and had put image goals at exactly
  0.0%; both were artifacts of its 61 distinct goals (§6.64, §6.67).
- **The lifelong curve is a composition artifact.** Pooled over episodes, SR
  rises from 8.1% at the first subtask to 11.7% by the fourth and falls after
   — the shape the project set out to find. It does not survive the control:
  only long episodes have late subtasks, and episode length tracks difficulty
  (6-subtask episodes score 14.74%, 9-subtask ones 5.85%). Within episodes of
  a fixed length the curve is flat, and paired within every episode the second
  half beats the first 64 times against 49, **p = 0.188**. §6.67.
- **The lifelong memory works, and does not reach the score.** Measured over
  218 subtasks paired position-by-position, memory on against memory off, the
  effect attenuates at every stage: the agent matches a candidate for its goal
  43.6% of the time against 27.1% (p = 0.016), but the matched node is no
  better localised (4.07 m against 3.75 m, p = 0.864) and the success rate is
  identical, 3.2% against 3.2% (p = 1.000). **Subtask 0 is a built-in negative
  control** — nothing to inherit, and the two arms agree on 29 of 30 — so the
  effect appears exactly where the mechanism is and nowhere else. Memory
  recalls roughly twice as many candidates, in the same wrong places. The
  p-values are clustered by distinct goal, because the 218 subtasks carry only
  61 of them and the uncorrected tests overstate themselves badly — §6.63,
  which is the entry to read before trusting any number here.
  §6.62.
- **An instrument that cannot answer its own question.** The 30-episode
  development subset has 218 subtasks but only **61 distinct goals**, because
  GOAT-Bench asks for the same object by category, by description and by
  photograph within one episode. Its power to detect even a 10-point change in
  success rate is **36.7%**; the full `val_unseen` run reaches 99.4% at 5
  points. Every null result here is therefore consistent with a real effect
  too small for this instrument, which is a different claim from "nothing
  happened". §6.63, §6.64.
- **A data-availability ceiling.** Of the development subtasks, a large share
  cannot be won by *any* navigator, because the detector never fires on the goal
  — a property of HM3D-Semantics, not of the policy. See
  `TECHNICAL_GUIDE.md` §6.32 and §6.35.
- **Two limits that belong to the embodiment.** The detector is nearly blind
  inside the 1.0 m success radius it is required to stop within (9% of poses at
  0.8 m against 34% at 1.5 m), and with a 1.41 m camera, a 90° vertical FOV and
  no pitch action, 14.2% of subtasks have a goal that cannot be framed from
  anywhere a scoring stop could happen — every `rug` subtask among them. §6.35,
  §6.42.
- **A per-component failure taxonomy**, with 67 dated findings in
  `TECHNICAL_GUIDE.md` §6, each paired with the measurement that produced it,
  **including the hypotheses that turned out to be wrong**.

## Documentation map

| File | What is in it |
|---|---|
| `PLAN.md` | The project spec: architecture, pinned stack, week-by-week plan with done criteria, the evaluation protocol (§8) and pitfalls (§9) |
| `TECHNICAL_GUIDE.md` | Module-by-module walkthrough (§3), testing philosophy (§4), glossary (§5) and the **Engineering Log** (§6) |
| `docs/REPORT.md` | The project report: motivation, literature, method, results, limitations |
| `docs/DEMO.md` | Runbook for this machine — what to run to see the system working, and what each command proves |
| `TRAIN_YOLO.md` | The detector finetune pipeline, including the WSL2/Windows setup and the dependency combination that actually installs |
| `configs/agent_default.yaml` | Every effective hyperparameter, annotated with the measurement behind it and held to the code by `tests/test_config.py` |

## Setup

`PLAN.md` §5 has the full instructions. The short version:

```bash
conda activate goat            # python 3.9, headless habitat-sim
pytest tests/ -q               # 300 tests, ~20 s; 286 need no GPU at all
python scripts/smoke_test.py   # loads a scene + YOLO + CLIP and steps 100 times
```

`requirements.txt` is **not installable as written** — the verified working
combination is recorded at the top of that file and in `TRAIN_YOLO.md` §0
(numpy 1.26.4, torch 2.6.0+cu124, ultralytics 8.4.117, habitat-sim 0.3.1).

Expected on-disk data, all gitignored: HM3D scenes under `data/hm3d/`,
GOAT-Bench episodes under `data/datasets/goat_bench/`, and the finetuned
detector at `checkpoints/yolo_goat.pt`. **Without that checkpoint the detector
silently falls back to COCO weights**, whose vocabulary covers almost none of
the GOAT categories.

## Running things

```bash
# One subtask, traced to video: occupancy, plan, track, memory nodes, the
# ground-truth goal the agent cannot see, and the live state machine
python -u scripts/debug_episode.py --n-episodes 30 --episode 24 --subtask 2 \
    --steps 250 --out outputs/trace

# The development evaluation: 30 deterministic episodes, 218 subtasks
python -u scripts/run_dev_eval.py --n-episodes 30 --output-dir outputs/dev_eval

# Memory on versus off, the headline ablation, over identical episodes
python -u scripts/run_dev_eval.py --n-episodes 30 --no-memory

# Failure taxonomy, lifelong curve and per-modality tables, from the artifacts
python scripts/failure_report.py outputs/dev_eval --compare outputs/dev_eval_nomem

# A hyperparameter grid; sweep on 10 episodes, confirm the winner on 30
python -u scripts/sweep.py --grid min-observations --n-episodes 10

# How many subtasks are winnable at all, given the detector
python -u scripts/detector_ceiling.py --n-episodes 30
```

Long runs checkpoint after every scene and take `--resume`.

## Architecture

One sense-plan-act tick per step, driven by `GoatAgent.act()`:

```
HabitatEnv.step ─► Observation(rgb, depth, pose, compass, gps, goal, collided)
   │
   ├─ PerceptionPipeline   YOLO ─► crops ─► CLIP ─► back-projected world XYZ
   ├─ SemanticMap          log-odds occupancy + ray clearing + class channels
   ├─ InstanceDatabase     merge on class + XZ distance + CLIP similarity
   └─ GoalMatcher          category / language / image ─► (InstanceNode, score)
        │
        SEARCHING ─match─► APPROACHING ─within 1 m─► VERIFYING ─face it─► stop
             └─ frontiers ◄──────── reject what cannot be reached ────────┘
```

Layer ownership runs one way: `sim/` → `perception/` → `mapping/` + `memory/` →
`matching/` → `planning/` → `agent/` → `eval/`.

## Conventions that have already caused bugs

Each has an entry in `TECHNICAL_GUIDE.md` §6 with the measurement that exposed
it. Violating one produces a silently wrong agent, not a crash.

- **Frames.** Habitat world is Y-up; the map is the XZ plane with `row ← Z`,
  `col ← X`. The camera is **OpenGL** convention, so back-projection must give
  `y = -(v-cy)·d/fy` and `z = -d`. Mixing in OpenCV points mirrored the cloud
  through the camera: 0% of points landed in front of the agent instead of 100%.
- **Camera pose, never agent base pose.** Depth is measured from a sensor 1.41 m
  up, and HM3D floors sit at arbitrary world Y.
- **Quaternion order.** Habitat JSON stores `(x, y, z, w)`; `numpy-quaternion`
  reads `(w, x, y, z)`. The mix-up leaves the forward vector intact and spawns
  the agent upside down — in all 360 episodes.
- **Occupancy is evidence-based**, tri-state `{-1, 0, 1}`. Never write
  `_occupancy` directly.
- **Episode ids are per-scene, not global.** 218 subtasks share only 86 distinct
  `(episode_id, subtask_index)` pairs.
- **Report both success numbers.** Success uses the benchmark's geometric
  visibility test; `sr_detector_agrees` is the stricter rate where the agent's
  own detector also recognised what it reached.

## Testing

300 tests, about 20 seconds. 286 of them need no GPU, no model downloads and no
scene files — dependency-injected fakes throughout. The other 14 load real YOLO
and CLIP weights, so run them on a free card: against a GPU already busy with an
eval they fail with out-of-memory errors that look exactly like regressions
(§6.53), and `--ignore=tests/test_agent_smoke.py` is the right scope mid-run. The suite pins conventions, which cuts
both ways: three transform tests once asserted the *wrong* camera convention and
encoded a serious geometric bug for weeks. A green suite confirms only the
conventions the tests were written against.

## Citation

GOAT-Lite replicates and measures against:

- Chang et al., "GOAT: GO to Any Thing", RSS 2024.
- Khanna et al., "GOAT-Bench: A Benchmark for Multi-Modal Lifelong Navigation", CVPR 2024.
- Ramakrishnan et al., "Habitat-Matterport 3D Dataset (HM3D)", NeurIPS D&B 2021.

## License

MIT — see `LICENSE`.
