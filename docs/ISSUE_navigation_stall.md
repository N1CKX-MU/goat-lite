# Agent approaches goals but stalls ~1.2 m short; dev-eval SR still 0%

**Labels:** `bug`, `navigation`, `week-9`
**Blocks:** Week 9 done-criteria (`SR > 0` on the 30-episode dev eval)

## Summary

After fixing 14 defects across perception, mapping, matching and planning
(see `TECHNICAL_GUIDE.md` §6), the agent now explores correctly and gets
close to goals, but **never calls STOP and never scores a success**.

On the 2-episode smoke test all 17 subtasks end in `timeout`. Several finish
**1.16–1.20 m** from a valid goal instance against a **1.0 m** success radius.
Nine subtasks report `path_length = 0.00 m` despite running the full 500 steps.

## Current numbers

| metric | before this round | now |
|---|---|---|
| blocked forward commands | 41% | **0%** |
| map says OCCUPIED but navmesh says navigable | 45% | **0%** |
| map says FREE and navmesh agrees | 37% | 50% |
| median distance to goal | 6.39 m | **3.68 m** |
| successes (2-ep smoke, 17 subtasks) | 0 | **0** |

Mapping and planning are now measurably correct. The failure is in the last
metre: approach → arrive → stop.

## Reproduce

```bash
# WSL2. See TRAIN_YOLO.md §0 for the EGL/d3d12 environment variables.
source /mnt/d/goat-work/goatenv.sh
cd /mnt/d/goat-lite
python -u scripts/run_dev_eval.py --n-episodes 2 --output-dir outputs/repro
```

Then inspect `outputs/repro/results.csv`. Expect `reason=timeout` on every row
and `distance_to_goal` around 1.2 m on the closest subtasks.

## What has been ruled out (with measurements, don't re-investigate)

- **`path_to_action` heading convention** — predicted forward matches the
  simulator on 12/12 headings; turn signs correct; closed-loop drive to a point
  2.00 m away lands 0.22 m off.
- **A\*** — 111 waypoints to a chosen frontier, 24/26 frontiers reachable.
  Obstacle inflation is now a hard constraint; `test_narrow_gap_is_refused`
  pins it.
- **`_follow_path`** — `tests/test_path_following.py` drives it in closed loop
  against the real motion model (0.25 m step, 30° turns, 0.05 m cells) and
  asserts convergence, monotone cursor progress, recovery from a reversed
  start, and that short paths still drive to their end. All 8 pass.
- **Actuation / navmesh** — 10 forward steps move 2.495 m, `get_gps()` tracks
  it, motion works from all 8 headings, episode start poses are navigable.
- **Depth units** — metres, verified across random navigable poses.
- **Occupancy height band** — floor near 0.0, walls to +3.09 above floor,
  37–68% of each frame in-band.
- **Camera convention** — 100% of reconstructed points now land in front of
  the agent (was 0%).

## Leading hypotheses

1. **Instance position error.** Memory node positions come from back-projecting
   the detection bbox centre. Measured error against ground truth was ~3 m
   before the camera-convention fix and has not been re-measured since. If a
   node sits >1 m from the real object, the agent can arrive at its *estimate*
   and still be outside the success radius — which matches subtasks stalling at
   1.16–1.20 m.
2. **VERIFYING never completes.** The FSM only stops when `_goal_in_view`
   confirms a detection matching the node's class in the final frame. With
   detector recall at 0.149 mAP50, the object may simply not be detected from
   the arrival pose even when it is visible.
3. **The remaining `path=0.00 m` subtasks.** Distinct from the above; the agent
   is stationary for 500 steps. Cause unknown — the two previous explanations
   (orbiting a waypoint, unhandled collisions) are both fixed and verified.

## Recommended next step

**Build the top-down visualiser before attempting another fix.** All 14 bugs so
far were found by printing numbers, and three regressions were introduced by
tuning constants against a 12-minute end-to-end eval. A single frame showing the
occupancy map, the planned path, the agent's actual track, the memory nodes and
the ground-truth goal position would very likely explain the last metre
immediately. `scripts/make_video.py` exists as a starting point.

## Related

- `TECHNICAL_GUIDE.md` §6.16–6.27 — the full bug chain with measurements
- `PLAN.md` §7.9 — Week 9 done criteria
- Detector: `checkpoints/yolo_goat.pt`, mAP50 0.149, 7/36 classes above 0.30
  recall. Two classes (`christmas tree`, `shower glass`) have zero training
  boxes — a data-availability ceiling in HM3D-Semantics, documented in §6.20.
