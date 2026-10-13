#!/usr/bin/env python3
"""Measure the detector-limited upper bound on success rate.

A subtask can only succeed if the agent stops within `success_distance` of a
valid instance **with that instance in view** -- and "in view" means the
detector actually fires on the goal class. If the detector never fires on the
target from any reasonable viewpoint, the subtask is unwinnable no matter how
good navigation is.

This parks the agent at a ring of navigable poses around each ground-truth
instance, facing it, and asks the detector directly. The fraction of subtasks
where it ever fires is a hard ceiling on SR for the whole pipeline.

Motivation: a traced episode had the agent standing 1.20 m from a microwave
with zero microwave nodes in memory after ~3000 steps, two of them dedicated
microwave subtasks. Probing that episode showed the detector never fired on any
of its 7 goals from 48 poses each -- a theoretical maximum SR of 0%.

Usage:
    python scripts/detector_ceiling.py --n-episodes 30 --out outputs/ceiling.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import habitat_sim

from src.sim.env import HabitatEnv
from src.sim.goat_dataset import load_val_unseen, sample_dev_subset
from src.eval.runner import _resolve_scene_path
from src.perception.detector import YOLODetector


def probe_instance(env, det, goal_xyz, category, radii, n_angles,
                   success_distance=1.0):
    """Does the detector ever fire on `category` when facing this instance?

    Returns (hits, hits_inside_radius, poses_tried, poses_inside_radius,
    best_conf).

    The in-radius counts are the ones that bound SR. A subtask succeeds only if
    the agent stops WITHIN success_distance of the instance with it in view, so
    a detection obtained at 1.5 m or 2.5 m proves the object is findable but not
    that the subtask is winnable. Measured over episode 24 of the dev subset,
    that distinction is large: the detector fires from 9% of poses at 0.8 m and
    34% at 1.5 m, because an object filling the frame with untextured surface
    does not resemble the room-scale viewpoints it was trained on.

    Poses are snapped to the navmesh, so each pose's true distance is measured
    rather than assumed from its nominal radius.
    """
    pf = env._sim.pathfinder
    hits, tried, best = 0, 0, 0.0
    hits_in, tried_in = 0, 0
    for radius in radii:
        for k in range(n_angles):
            th = 2 * math.pi * k / n_angles
            p = [float(goal_xyz[0] + radius * math.cos(th)),
                 float(goal_xyz[1]),
                 float(goal_xyz[2] + radius * math.sin(th))]
            snapped = pf.snap_point(p)
            if not np.isfinite(snapped).all() or not pf.is_navigable(snapped):
                continue
            d = np.array([goal_xyz[0] - snapped[0], goal_xyz[2] - snapped[2]])
            pose_dist = float(np.linalg.norm(d))
            if pose_dist < 1e-6:
                continue
            in_radius = pose_dist <= success_distance
            yaw = math.atan2(-d[0], -d[1])
            st = habitat_sim.AgentState()
            st.position = snapped
            st.rotation = np.quaternion(math.cos(yaw / 2), 0, math.sin(yaw / 2), 0)
            env._agent.set_state(st)
            tried += 1
            tried_in += int(in_radius)
            for dd in det.detect(env._make_obs().rgb):
                if dd.cls_name == category:
                    hits += 1
                    hits_in += int(in_radius)
                    best = max(best, float(dd.conf))
    return hits, hits_in, tried, tried_in, best


def main() -> None:
    ap = argparse.ArgumentParser(description="Detector-limited SR ceiling")
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--angles", type=int, default=8)
    ap.add_argument("--radii", type=float, nargs="+",
                    default=[0.5, 0.8, 1.0, 1.5, 2.5])
    ap.add_argument("--success-distance", type=float, default=1.0,
                    help="poses at or inside this distance are the only ones "
                         "that can produce a scoring stop")
    ap.add_argument("--out", default="outputs/detector_ceiling.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset-root",
                    default="data/datasets/goat_bench/hm3d/v1/val_unseen")
    ap.add_argument("--hm3d-root", default="data/hm3d")
    args = ap.parse_args()

    eps = load_val_unseen(args.dataset_root)
    dev = sample_dev_subset(eps, n=args.n_episodes, seed=args.seed)

    by_scene = collections.OrderedDict()
    for ep in dev:
        by_scene.setdefault(ep.scene_id, []).append(ep)

    det = YOLODetector(device=args.device)
    rows = []
    # One instance can serve many subtasks; probing is the expensive part.
    cache: dict[tuple, tuple] = {}

    for si, (scene_id, scene_eps) in enumerate(by_scene.items(), 1):
        print(f"[{si}/{len(by_scene)}] {scene_id.split('/')[-2]}", flush=True)
        try:
            env = HabitatEnv(scene_path=_resolve_scene_path(scene_id, args.hm3d_root),
                             seed=args.seed)
        except Exception as e:
            print(f"    scene load failed: {type(e).__name__}", flush=True)
            continue

        for ep in scene_eps:
            for sub in ep.subtasks:
                if not sub.goal_candidates:
                    rows.append(dict(scene=scene_id, episode=ep.episode_id,
                                     subtask=sub.subtask_index,
                                     modality=sub.modality, category=sub.category,
                                     reachable=False, reachable_in_radius=False,
                                     hits=0, hits_in_radius=0, best_conf=0.0,
                                     note="no ground-truth instance"))
                    continue
                hits = tried = hits_in = tried_in = 0
                best = 0.0
                for cand in sub.goal_candidates:
                    key = (scene_id, sub.category,
                           tuple(np.round(cand.position, 2)))
                    if key not in cache:
                        cache[key] = probe_instance(
                            env, det, np.asarray(cand.position), sub.category,
                            args.radii, args.angles, args.success_distance)
                    h, hi, t, ti, b = cache[key]
                    hits += h; tried += t
                    hits_in += hi; tried_in += ti
                    best = max(best, b)
                rows.append(dict(scene=scene_id, episode=ep.episode_id,
                                 subtask=sub.subtask_index,
                                 modality=sub.modality, category=sub.category,
                                 reachable=hits > 0,
                                 reachable_in_radius=hits_in > 0,
                                 hits=hits, poses=tried,
                                 hits_in_radius=hits_in,
                                 poses_in_radius=tried_in,
                                 best_conf=round(best, 3)))
        env.close()

        # Checkpoint: the WSL GL context has died mid-run before.
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)

    # ── summary ──────────────────────────────────────────────────────────
    n = len(rows)
    ok = sum(r["reachable"] for r in rows)
    ok_in = sum(r.get("reachable_in_radius") for r in rows)
    print(f"\n{'=' * 58}")
    print(f"subtasks probed                   : {n}")
    print(f"detector fires from ANY pose      : {ok}  "
          f"({100.0 * ok / max(n, 1):.1f}%)")
    print(f"detector fires INSIDE {args.success_distance:.1f} m        : {ok_in}  "
          f"({100.0 * ok_in / max(n, 1):.1f}%)")
    print(f"IMPOSSIBLE for any navigator      : {n - ok_in}  "
          f"({100.0 * (n - ok_in) / max(n, 1):.1f}%)")
    print(f"\n>>> Detector-limited SR ceiling: "
          f"{100.0 * ok_in / max(n, 1):.1f}%  (in-radius; "
          f"{100.0 * ok / max(n, 1):.1f}% from any distance)")
    print("    Success needs the stop INSIDE the radius with the goal in view,")
    print("    so the in-radius figure is the one that bounds SR.")

    by_mod = collections.defaultdict(lambda: [0, 0, 0])
    by_cat = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        for agg, key in ((by_mod, r["modality"]), (by_cat, r["category"])):
            agg[key][0] += 1
            agg[key][1] += int(bool(r["reachable"]))
            agg[key][2] += int(bool(r.get("reachable_in_radius")))

    print(f"\n{'modality':<12}{'total':>7}{'any':>8}{'in-radius':>11}")
    for m, (t, o, oi) in sorted(by_mod.items()):
        print(f"{m:<12}{t:>7}{100.0 * o / t:>7.1f}%{100.0 * oi / t:>10.1f}%")

    print(f"\n{'category':<20}{'total':>7}{'any':>8}{'in-radius':>11}")
    for c, (t, o, oi) in sorted(by_cat.items(), key=lambda kv: -kv[1][0]):
        print(f"{c:<20}{t:>7}{100.0 * o / t:>7.1f}%{100.0 * oi / t:>10.1f}%")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
