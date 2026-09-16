# Running the simulator on the 3050 laptop

Everything below runs on this machine as-is. Activate the env first; the
`activate.d` hook clears the ROS `PYTHONPATH` that `~/.bashrc` exports, which
otherwise leaks python3.12 packages into this python3.9 env.

```bash
conda activate goat
```

## 1. Is anything broken? (30 s)

```bash
python scripts/smoke_test.py          # loads a scene + YOLO + CLIP, steps 100x
```

Expect `DONE. Peak VRAM: ~600 MB`. Detections are often 0 here — the walk is a
straight line into whatever is in front of the start pose, so it proves nothing
about the detector.

## 2. Scene tour — the pipeline, no agent (2–4 min)

Best live demo of the system itself: RGB with live detections, depth, the
occupancy map filling in, and a panel naming each pipeline stage.

```bash
# pick the cleanest 3 of the 10 local scenes by depth validity, 15 s each
python -u scripts/scene_tour.py --scenes 3 --candidates 10 --seconds 15 \
    --out outputs/demo_scene_tour.mp4

# one known-good scene
python -u scripts/scene_tour.py --scene-dirs 00809-Qpor2mEya8F --seconds 20 \
    --out outputs/tour_00809.mp4
```

Routes are random per `--seed`, and some legs hug a wall, which renders as a
flat grey camera view. Re-run with another seed if a take looks bad.

## 3. Agent trace — the real FSM on a real episode (1–2 min)

Runs the full agent on one subtask and writes an MP4: occupancy map, planned
path, actual track, memory nodes, the matched node, the ground-truth goal the
agent cannot see, plus live FSM state.

```bash
python -u scripts/debug_episode.py --n-episodes 30 --episode 24 --subtask 2 \
    --steps 250 --out outputs/demo_agent
```

**`--n-episodes 30` is required.** The default is 2, which samples a different
subset, and any `--episode` above 1 fails with `IndexError`.

### Episodes that exist locally

`data/hm3d` is the **minival** split (10 scenes). Only 4 val_unseen scenes are
among them, so only these dev-subset indices run here:

| `--episode` | scene | subtasks | notes |
|---|---|---|---|
| 11 | wcojb4TFT35 | 10 | mostly `christmas tree` — **0% detector ceiling, do not demo** |
| 12 | y9hTuugGdiq | 5 | refrigerator, picture, rug |
| 14 | k1cupFYWXJ6 | 10 | decorative plant, picture |
| 19 | y9hTuugGdiq | 10 | picture, rug, mirror, refrigerator |
| 24 | TEEsavR23oF | 9 | pillow, picture, mirror, refrigerator, microwave |
| 29 | y9hTuugGdiq | 5 | refrigerator, rug |

Measured 2026-09-16 (250 steps each):

| episode / subtask | goal | closest approach | outcome |
|---|---|---|---|
| **24 / 2** | category/mirror | **0.19 m** | reaches the goal, never confirms it — best demo |
| 19 / 0 | category/picture | 3.20 m | stops at the wrong instance, 3.72 m out |
| 19 / 5 | category/picture | 2.79 m | never approaches |
| 14 / 0 | category/decorative plant | 3.52 m | never approaches |

`24 / 2` is the one to show: `MATCH ERROR 0.15 m` (memory has the mirror almost
exactly right), `dist to TRUE goal 0.19 m`, state `VERIFYING`, and a
first-person view flat against the wall — the agent drives onto the node, the
mirror leaves the frame, and `goal_in_view` never becomes true.

## What does not run here

- **`scripts/robot_tour.py`** — needs a habitat-sim built with Bullet. This
  build raises `ESP_CHECK failed: Physics has been enabled ... not built with
  Bullet support`. The asset itself is present at `data/robots/hab_stretch`.
  Nishaanth's machine has the Bullet build; the recorded video came from there.
- **A full dev eval** (`scripts/run_dev_eval.py --n-episodes 30`) — 24 of the
  30 episodes reference val_unseen scenes that are not on this disk.
