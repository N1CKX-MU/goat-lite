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

**All of them, as of 2026-10-13.** `data/hm3d` now points at the full HM3D val
split (100 scene dirs), and every one of the 36 `val_unseen` scenes the
benchmark references resolves on disk — verified by walking all 360 episodes
through `_resolve_scene_path`. The 30-episode dev subset spans 22 scenes, all
present. The earlier 10-scene minival restriction no longer applies.

Measured 2026-10-13 after the stop-decision fixes (250 steps each, scored the
same way `src/eval/runner.py` scores):

| episode / subtask | goal | instances | closest approach | outcome |
|---|---|---|---|---|
| **24 / 2** | category/mirror | 1 | **0.91 m** | **STOP at step 15 — SUCCESS** |
| 19 / 0 | category/picture | 25 | 0.40 m, visible | timeout |
| 19 / 5 | category/picture | 25 | 3.02 m | timeout |
| 14 / 0 | category/decorative plant | 4 | 3.52 m | timeout |

`24 / 2` is still the one to show, now as a success: the agent arrives at
0.91 m, turns to face the mirror instead of driving into the wall it hangs on,
and stops. The panel shows both predicates — `goal in view (GT)`, which is what
the scorer checks, and `detector fired`, which is what the agent's YOLO says.
They disagree often, and that gap is the subject of engineering-log 6.35.

For contrast, `19 / 0` shows the remaining failure: the agent passes 0.40 m from
a valid picture that is geometrically in view and does not stop, because the
detector never fired on it and so memory holds no node for it.

Episode 11 is still a poor demo — mostly `christmas tree`, a category with zero
training boxes and a 0% detector ceiling.

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

## 5. Full dev eval (hours)

Runs here as of 2026-10-13, now that all 36 `val_unseen` scenes are on disk.
218 subtasks over 22 scenes; budget 2–4 hours and use `python -u`.

```bash
python -u scripts/run_dev_eval.py --n-episodes 30 \
    --output-dir outputs/dev_eval_memon

# memory-off arm of the ablation: same episodes, memory and map cleared
# before every subtask instead of carried across the episode
python -u scripts/run_dev_eval.py --n-episodes 30 --no-memory
```

Results are checkpointed after every scene, so a crash costs at most one scene.
`summary.json` reports SR and SPL by modality and by subtask index (the
lifelong curve), plus `sr_detector_agrees` — the stricter reading where the
agent's own detector also recognised the goal in the final frame.

## What does not run here

- **The semantic-sensor visibility test.** The local val scenes ship only
  `.basis.glb` and `.basis.navmesh`, no `.semantic.glb`, so `instance_in_view`
  tests occlusion against the depth buffer rather than a semantic mask. Fix by
  downloading `hm3d-val-semantic-annots`.
