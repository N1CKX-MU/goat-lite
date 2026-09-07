#!/usr/bin/env python3
"""Render a guided tour through HM3D scenes: a demo of the simulation itself.

Walks a smooth, collision-free route through each scene and records what the
robot's sensors see -- RGB with live detections, depth, and the occupancy map
building up underneath. This shows the environment and the perception stack
without any claim about navigation performance.

Usage:
    python scripts/scene_tour.py --scenes 3 --out outputs/tour.mp4
    python scripts/scene_tour.py --scene-dirs 00853-5cdEh9F2hJL --seconds 25
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import habitat_sim

from src.sim.env import HabitatEnv
from src.mapping.occupancy import SemanticMap, map_extent_for_scene
from src.utils.viz import render_topdown, draw_detections

W, H = 1280, 720
BG = (26, 22, 20)          # BGR, warm near-black
FG = (240, 240, 240)
ACCENT = (90, 160, 255)    # amber-ish in BGR
DIM = (150, 150, 150)


def _put(img, text, org, scale=0.5, colour=FG, thick=1):
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thick, cv2.LINE_AA)


def build_route(pf, rng, n_legs=6, min_leg=3.0):
    """A connected, walkable route: geodesic legs between navigable points."""
    pts = []
    cur = pf.get_random_navigable_point()
    tries = 0
    while len(pts) < n_legs and tries < 200:
        tries += 1
        nxt = pf.get_random_navigable_point()
        if not np.isfinite(nxt).all():
            continue
        if float(np.linalg.norm(np.asarray(nxt) - np.asarray(cur))) < min_leg:
            continue
        path = habitat_sim.ShortestPath()
        path.requested_start = cur
        path.requested_end = nxt
        if not pf.find_path(path) or len(path.points) < 2:
            continue
        pts.extend([np.asarray(p, dtype=np.float64) for p in path.points])
        cur = nxt
    return pts


def resample(points, spacing=0.12):
    """Even spacing along the polyline so the camera glides at constant speed."""
    if len(points) < 2:
        return points
    out = [points[0]]
    carry = 0.0
    for a, b in zip(points, points[1:]):
        seg = float(np.linalg.norm(b - a))
        if seg < 1e-6:
            continue
        t = carry
        while t < seg:
            out.append(a + (b - a) * (t / seg))
            t += spacing
        carry = t - seg
    return out


def score_scene(scene_dir, samples=14, seed=3):
    """How enclosed and well-reconstructed is this scene?

    HM3D contains outdoor courtyards and scans with large holes, where the
    camera stares at empty black space. Those look broken on a demo reel even
    though nothing is wrong. Sampling a few navigable poses and measuring the
    fraction of pixels with valid depth separates enclosed interiors (high)
    from open or hole-ridden scans (low).

    Returns (score, navigable_area) or (None, 0) if the scene is unusable.
    """
    glb = None
    for f in os.listdir(scene_dir):
        if f.endswith(".basis.glb"):
            glb = os.path.join(scene_dir, f)
    if glb is None:
        return None, 0.0
    try:
        env = HabitatEnv(scene_path=glb, seed=seed)
    except Exception:
        return None, 0.0
    pf = env._sim.pathfinder
    if not pf.is_loaded or pf.navigable_area < 12.0:
        env.close()
        return None, 0.0

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(samples):
        p = pf.get_random_navigable_point()
        if not np.isfinite(p).all():
            continue
        yaw = rng.uniform(-math.pi, math.pi)
        st = habitat_sim.AgentState()
        st.position = p
        st.rotation = np.quaternion(math.cos(yaw / 2), 0, math.sin(yaw / 2), 0)
        env._agent.set_state(st)
        d = env._make_obs().depth
        vals.append(float(np.mean((d > 0) & np.isfinite(d))))
    area = float(pf.navigable_area)
    env.close()
    if not vals:
        return None, 0.0
    return float(np.mean(vals)), area


def tour_scene(scene_dir, hm3d_root, seconds, fps, writer, rng, device, det):
    """Render one scene's leg of the tour. Returns frames written."""
    import cv2

    name = os.path.basename(scene_dir)
    glb = None
    for f in os.listdir(scene_dir):
        if f.endswith(".basis.glb"):
            glb = os.path.join(scene_dir, f)
    if glb is None:
        print(f"  {name}: no .basis.glb, skipping")
        return 0

    try:
        env = HabitatEnv(scene_path=glb, seed=1234)
    except Exception as e:
        print(f"  {name}: load failed ({type(e).__name__}), skipping")
        return 0

    pf = env._sim.pathfinder
    if not pf.is_loaded:
        print(f"  {name}: no navmesh, skipping")
        env.close()
        return 0

    size_m, origin = map_extent_for_scene(*env.get_scene_bounds())
    smap = SemanticMap(size_m=size_m, origin=origin, camera_height=1.41)

    route = resample(build_route(pf, rng))
    want = int(seconds * fps)
    if len(route) < 8:
        print(f"  {name}: route too short, skipping")
        env.close()
        return 0
    # loop or trim the route to the requested duration
    while len(route) < want:
        route = route + route[::-1]
    route = route[:want]

    # -- title card -------------------------------------------------------
    card = np.full((H, W, 3), BG, dtype=np.uint8)
    _put(card, "GOAT-Lite", (60, 300), 1.6, ACCENT, 3)
    _put(card, "simulated home environment", (62, 348), 0.75, FG, 1)
    _put(card, name, (62, 404), 0.95, FG, 2)
    _put(card, f"{pf.navigable_area:.0f} m2 navigable floor area",
         (62, 442), 0.6, DIM, 1)
    for _ in range(int(fps * 1.5)):
        writer.append_data(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))

    heading = 0.0
    track = []
    written = 0

    for i, p in enumerate(route):
        # face along the direction of travel, smoothed so it doesn't snap
        j = min(i + 4, len(route) - 1)
        d = route[j] - p
        if float(np.linalg.norm([d[0], d[2]])) > 1e-4:
            target = math.atan2(-d[0], -d[2])
            err = (target - heading + math.pi) % (2 * math.pi) - math.pi
            heading += 0.25 * err

        st = habitat_sim.AgentState()
        st.position = p.tolist()
        st.rotation = np.quaternion(math.cos(heading / 2), 0,
                                    math.sin(heading / 2), 0)
        env._agent.set_state(st)

        obs = env._make_obs()
        smap.update_from_depth(obs.depth, obs.pose, env.intrinsics)
        xy = env.get_gps()
        track.append(xy.copy())

        dets = det.detect(obs.rgb) if det is not None else []

        frame = np.full((H, W, 3), BG, dtype=np.uint8)

        # first-person, with what the detector sees
        rgb = draw_detections(obs.rgb, dets)
        rgb = cv2.resize(rgb, (600, 600), interpolation=cv2.INTER_LINEAR)
        frame[70:670, 24:624] = rgb
        _put(frame, "ROBOT CAMERA", (24, 60), 0.5, DIM)

        # depth
        d_img = obs.depth.copy()
        finite = np.isfinite(d_img) & (d_img > 0)
        vmax = float(np.percentile(d_img[finite], 95)) if finite.any() else 1.0
        d_norm = np.clip(d_img / max(vmax, 1e-3), 0, 1)
        d_col = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_BONE)
        d_col[~finite] = (40, 34, 30)
        d_col = cv2.resize(d_col, (300, 300), interpolation=cv2.INTER_NEAREST)
        frame[70:370, 648:948] = d_col
        _put(frame, "DEPTH", (648, 60), 0.5, DIM)

        # the map it is building
        top = render_topdown(smap, xy, heading, track=track, out_size=300)
        th, tw = top.shape[:2]
        frame[70:70 + th, 964:964 + tw] = top
        _put(frame, "MAP BEING BUILT", (964, 60), 0.5, DIM)

        # -- what the viewer is looking at --------------------------------
        # Three panels show three different things at once and nothing on
        # screen says how they relate. Spell out the pipeline and put a live
        # number beside each stage, so the demo explains itself unattended.
        occ = smap.get_occupancy()
        pct = 100.0 * (occ != -1).sum() / occ.size
        seen = sorted({dd.cls_name for dd in dets})

        _put(frame, "HOW IT WORKS", (648, 424), 0.5, DIM)
        cv2.line(frame, (648, 436), (1256, 436), (60, 52, 46), 1)

        stages = [
            ("1  RGB-D camera", "256 x 256, mounted 1.41 m up"),
            ("2  YOLOv8 detector", "36 classes | now: " +
             (", ".join(seen)[:32] if seen else "nothing")),
            ("3  CLIP encoder", "512-d embedding per object crop"),
            ("4  Instance memory", "repeat sightings merge into one object"),
            ("5  Occupancy map", f"log-odds grid | {pct:.1f}% explored"),
            ("6  A* + frontier", "route to the goal, or explore for it"),
        ]
        y = 464
        for label, detail in stages:
            _put(frame, label, (648, y), 0.46, ACCENT)
            _put(frame, detail, (846, y), 0.42, FG)
            y += 30

        # captions
        _put(frame, name, (24, 700), 0.6, FG)
        _put(frame, "simulated home  |  sensors only, no ground truth",
             (648, 700), 0.48, DIM)
        _put(frame, "GOAT-Lite", (1140, 700), 0.55, ACCENT)

        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        written += 1

    env.close()
    print(f"  {name}: {written} frames, {pf.navigable_area:.0f} m2")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Guided tour of HM3D scenes")
    ap.add_argument("--scenes", type=int, default=3,
                    help="how many scenes to visit (ignored if --scene-dirs given)")
    ap.add_argument("--scene-dirs", nargs="*", default=None,
                    help="explicit scene directory names under --hm3d-root")
    ap.add_argument("--candidates", type=int, default=12,
                    help="scenes to score before picking the most enclosed")
    ap.add_argument("--seconds", type=float, default=18.0,
                    help="tour length per scene")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-detector", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hm3d-root", default="data/hm3d")
    ap.add_argument("--out", default="outputs/scene_tour.mp4")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    root = args.hm3d_root
    if args.scene_dirs:
        names = args.scene_dirs
    else:
        allnames = sorted(d for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d)))
        pool = list(rng.permutation(allnames))[: args.candidates]
        print(f"scoring {len(pool)} candidate scenes for enclosure...")
        scored = []
        for n in pool:
            s, area = score_scene(os.path.join(root, n))
            if s is None:
                continue
            print(f"  {n:<24} coverage {s * 100:5.1f}%   {area:6.0f} m2")
            scored.append((s, area, n))
        # well-enclosed first; a little weight on size so tours aren't cramped
        scored.sort(key=lambda t: (t[0], min(t[1], 120) / 120), reverse=True)
        names = [n for _, _, n in scored]
        if names:
            print(f"chosen: {names[: args.scenes]}\n")

    det = None
    if not args.no_detector:
        try:
            from src.perception.detector import YOLODetector
            det = YOLODetector(device=args.device)
        except Exception as e:
            print(f"detector unavailable ({type(e).__name__}); rendering without")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import imageio.v2 as imageio
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=None,
                                quality=8)

    done = 0
    print(f"touring -> {args.out}")
    for name in names:
        if done >= (len(args.scene_dirs) if args.scene_dirs else args.scenes):
            break
        n = tour_scene(os.path.join(root, name), root, args.seconds, args.fps,
                       writer, rng, args.device, det)
        if n:
            done += 1

    writer.close()
    print(f"\nwrote {args.out}  ({done} scenes)")


if __name__ == "__main__":
    main()
