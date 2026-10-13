# Agent approaches goals but stalls ~1.2 m short; dev-eval SR still 0%

**Labels:** `bug`, `navigation`, `week-9`
**Blocks:** Week 9 done-criteria (`SR > 0` on the 30-episode dev eval)
**Status: CLOSED 2026-10-13** — see *Resolution* below. None of it was a
navigation defect: two were in the stop decision, one was in the success
criterion itself, and the `path = 0.00 m` mystery was an unreachable goal being
re-approached until the step budget ran out.

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

## Resolution (2026-10-13)

All three hypotheses are settled. The headline is that **none of them was a
navigation problem**, which is why no amount of navigation tuning moved the
number.

**1. Instance position error — not the cause.** Traced on episode 24 subtask 2:
the matched memory node sat **0.15 m** from the true mirror, and pure pursuit
delivered the agent to **0.19 m** of it. Memory and control were both fine. The
error is now recorded per subtask as `node_error_m`, so this is measured rather
than assumed from here on.

**2. VERIFYING never completes — confirmed, and it had three separate causes**
(engineering log 6.35–6.37):

- VERIFYING issued **forward** while already inside the success radius, pressing
  the agent nose-to-surface against the wall a wall-mounted goal hangs on, which
  pushed the object out of frame.
- The 5-step budget was **geometrically impossible**: one turn is 30°, so it
  could not bring a goal more than 150° off-axis into frame.
- Expiry was a **livelock** — reverting to APPROACHING while still inside the
  radius re-entered VERIFYING with the counter reset. Measured at 32 of 40
  ticks.

And the deeper reason the stop never fired: **the scorer was asking the agent's
own detector** whether the goal was visible, which is not the benchmark's
criterion, while the detector fires from only 9% of poses at 0.8 m against 34%
at 1.5 m. Success *requires* a stop inside 1.0 m, so correct arrivals were
rejected roughly nine times in ten. Verification now turns to face the node,
closes to 0.8 of the success distance, and commits on accumulated memory.

**3. The `path = 0.00 m` subtasks — solved** (engineering log 6.47). This was
the most persistent one and it was never a motion problem at all. Over the first
125 subtasks of the 2026-10-13 dev eval, **28 (22.4%)** ran the full 500 steps
having moved less than 0.01 m, with **zero** successes. The distribution over
subtask index gave it away: 1 through 7 and **never 0**, because subtask 0
starts with empty memory and so has no remembered node to get stuck on.

When A\* could not reach the matched node, the approach give-up cleared
`_matched_node` *without rejecting it*, so the matcher returned the same node on
the next tick — APPROACHING, four failed plans, SEARCHING, re-match, for the
whole budget, turning in place throughout. Reproduced in closed loop at 500
ticks / 499 turns / 0.000 m moved, and after routing the give-up through
`_reject_matched_node()`: **15 ticks**, 3 in APPROACHING, node rejected.

## Status

**Closed.** The first scored successes followed these fixes; `SR` is no longer
0%. The remaining failure mode is different and is tracked in the engineering
log rather than here: with the stop decision working, most failures are now
*never reaching the goal*, and the binding constraints are detector recall at
close range (6.35), vertical framing inside the success radius (6.42) and
instance ambiguity for the single-instance modalities (6.43).

## Note on the recommended next step

The advice below — *build the visualiser before attempting another fix* — was
followed and was correct: it found two substantial bugs in two frames
(6.28–6.29). Worth keeping as a record of what the recommendation bought.

## Related

- `TECHNICAL_GUIDE.md` §6.16–6.27 — the full bug chain with measurements
- `PLAN.md` §7.9 — Week 9 done criteria
- Detector: `checkpoints/yolo_goat.pt`, mAP50 0.149, 7/36 classes above 0.30
  recall. Two classes (`christmas tree`, `shower glass`) have zero training
  boxes — a data-availability ceiling in HM3D-Semantics, documented in §6.20.
