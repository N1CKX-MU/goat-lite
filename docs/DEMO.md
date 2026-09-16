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

## 4. Robot tour — Stretch on a chase camera (2 min)

Needs Bullet physics. `goat` now carries the bullet build, so no separate env:

```bash
python -u scripts/robot_tour.py --scene-dirs 00809-Qpor2mEya8F --seconds 12 \
    --out outputs/demo_robot_tour.mp4
```

The env originally shipped `habitat-sim-mutex 1.0 headless_nobullet`, which
raises `ESP_CHECK failed: Physics has been enabled ... not built with Bullet
support`. Swapped in place, tested first in a throwaway clone:

```bash
conda install -c aihabitat -c conda-forge \
  habitat-sim=0.3.1=py3.9_headless_bullet_linux_3d6d67d6deae4ab2472cc84df7a3cef1503f606d
```

Pin the full build string. A bare `habitat-sim withbullet` resolves to 0.3.3, a
different simulator from the one behind every measurement so far. The pinned
build is the same version *and source hash* as before, so only Bullet changed:
numpy 1.26.4, torch 2.6.0+cu124 and ultralytics 8.4.117 are untouched and all
166 tests still pass.

## What does not run here

- **A full dev eval** (`scripts/run_dev_eval.py --n-episodes 30`) — 24 of the
  30 episodes reference val_unseen scenes that are not on this disk. Fix by
  downloading `hm3d_val_habitat_v0.2` with
  `python -m habitat_sim.utils.datasets_download` and repointing the
  `data/hm3d` symlink at `scene_datasets/hm3d/val`.
