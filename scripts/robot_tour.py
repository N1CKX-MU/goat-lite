#!/usr/bin/env python3
"""Scene tour with a visible robot: a Hello Robot Stretch seen from a chase camera.

The evaluated agent is an invisible cylinder -- that is all GOAT-Bench simulates,
and every sensor we use is on the robot, so the body never appears on screen.
For a demo that hides what the project is. This loads the Stretch URDF (one of
the two robots the GOAT paper ran on) into the scene, drives it along a
walkable route, and renders a third-person camera alongside the robot's own
first-person view and the occupancy map it builds.

Demo only. Nothing here is imported by the agent or the eval: it builds its own
simulator with Bullet physics enabled (articulated objects need it), whereas
HabitatEnv keeps physics off.

Needs the robot asset (38.6 MB, ai-habitat/hab_stretch on Hugging Face):
    git clone https://huggingface.co/datasets/ai-habitat/hab_stretch.git data/robots/hab_stretch

Usage:
    python scripts/robot_tour.py --scene-dirs 00832-qyAac8rV8Zk --seconds 20
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import habitat_sim
import magnum as mn
import quaternion as qt

from src.sim.env import default_gpu_device_id
from src.mapping.occupancy import SemanticMap, map_extent_for_scene
from src.utils.viz import render_topdown, draw_detections
from scripts.scene_tour import resample, _put, BG, FG, ACCENT, DIM

W, H = 1280, 720
CAM_HEIGHT = 1.41          # matches HabitatEnv, so the map is built identically
FPV_RES = 256
CHASE_RES = (624, 832)     # (h, w)

# Stretch's URDF drives along +X with the arm on its right; our agent's forward
# is -Z. A +90 deg yaw lines the two up (verified by rendering all four
# candidate orientations and checking the arm extends to the robot's right).
ROBOT_YAW_OFFSET = math.pi / 2

# Chase camera boom, relative to a pivot 1.0 m above the robot's feet.
BOOM_BACK = 1.7
BOOM_UP = 0.85
PIVOT_UP = 1.0


def _yaw_quat(yaw: float) -> np.quaternion:
    return np.quaternion(math.cos(yaw / 2), 0, math.sin(yaw / 2), 0)


def _look_rotation(frm: np.ndarray, to: np.ndarray) -> np.quaternion:
    """Rotation that points a habitat camera (looks down -Z) from `frm` at `to`."""
    d = to - frm
    yaw = math.atan2(-d[0], -d[2])
    pitch = math.atan2(d[1], math.hypot(d[0], d[2]))
    q_pitch = np.quaternion(math.cos(pitch / 2), math.sin(pitch / 2), 0, 0)
    return _yaw_quat(yaw) * q_pitch


def make_sim(glb: str) -> habitat_sim.Simulator:
    b = habitat_sim.SimulatorConfiguration()
    b.scene_id = glb
    b.enable_physics = True            # articulated objects + ray casts need Bullet
    b.gpu_device_id = default_gpu_device_id()

    # agent 0: the robot's own sensors, identical to HabitatEnv's
    robot = habitat_sim.agent.AgentConfiguration()
    robot.height, robot.radius = CAM_HEIGHT, 0.17
    specs = []
    for uuid, kind in (("rgb", habitat_sim.SensorType.COLOR),
                       ("depth", habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type = uuid, kind
        s.resolution = [FPV_RES, FPV_RES]
        s.position = [0.0, CAM_HEIGHT, 0.0]
        specs.append(s)
    robot.sensor_specifications = specs

    # agent 1: a free-floating chase camera; its pose is set directly each frame
    chase = habitat_sim.agent.AgentConfiguration()
    c = habitat_sim.CameraSensorSpec()
    c.uuid, c.sensor_type = "chase", habitat_sim.SensorType.COLOR
    c.resolution = list(CHASE_RES)
    c.position = [0.0, 0.0, 0.0]
    c.hfov = 75
    chase.sensor_specifications = [c]

    return habitat_sim.Simulator(habitat_sim.Configuration(b, [robot, chase]))


def build_floor_route(pf, n_legs=6, min_leg=2.5, clearance=0.7, max_dy=0.3):
    """Walkable route that stays on one floor and turns in open space.

    The generic tour route happily climbs stairs -- fine for a floating camera,
    absurd for a wheeled Stretch. Waypoints are also kept >= `clearance` from
    obstacles so the chase camera has room to sit behind the robot.
    """
    def sample(y0=None):
        for _ in range(300):
            q = np.asarray(pf.get_random_navigable_point(), dtype=np.float64)
            if not np.isfinite(q).all():
                continue
            if y0 is not None and abs(q[1] - y0) > max_dy:
                continue
            if pf.distance_to_closest_obstacle(q, clearance + 0.1) >= clearance:
                return q
        return None

    cur = sample()
    if cur is None:
        return []
    y0, pts, tries = cur[1], [], 0
    while len(pts) < n_legs * 4 and tries < 60:
        tries += 1
        nxt = sample(y0)
        if nxt is None or np.linalg.norm(nxt - cur) < min_leg:
            continue
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = cur, nxt
        if not pf.find_path(path) or len(path.points) < 2:
            continue
        leg = [np.asarray(q, dtype=np.float64) for q in path.points]
        if max(abs(q[1] - y0) for q in leg) > max_dy:
            continue                      # the shortest path detours via stairs
        pts.extend(leg)
        cur = nxt
    return pts


def boom_length(sim, pivot: np.ndarray, desired: np.ndarray) -> float:
    """How far the camera can sit along pivot->desired before hitting geometry.

    Without this the camera ends up inside walls in every corridor and the
    frame shows the back of a mesh. Cast with the robot parked out of the way.
    """
    v = desired - pivot
    full = float(np.linalg.norm(v))
    ray = habitat_sim.geo.Ray(mn.Vector3(*pivot), mn.Vector3(*(v / full)))
    res = sim.cast_ray(ray, max_distance=full)
    if res.has_hits():
        return max(0.35, float(res.hits[0].ray_distance) - 0.2)
    return full


def tour_scene(scene_dir, seconds, fps, writer, rng, det):
    import cv2

    name = os.path.basename(scene_dir)
    glb = next((os.path.join(scene_dir, f) for f in os.listdir(scene_dir)
                if f.endswith(".basis.glb")), None)
    if glb is None:
        print(f"  {name}: no .basis.glb, skipping")
        return 0

    sim = make_sim(glb)
    pf = sim.pathfinder
    pf.seed(int(rng.integers(1 << 30)))    # reproducible routes
    robot_agent, chase_agent = sim.get_agent(0), sim.get_agent(1)

    ao = sim.get_articulated_object_manager().add_articulated_object_from_urdf(
        "data/robots/hab_stretch/urdf/hab_stretch.urdf", fixed_base=True)
    # URDF origin sits above the wheel bottoms; lift so the lowest point meets the floor.
    # ArticulatedObject.aabb only exists on newer habitat-sim; 0.3.1's bullet build
    # has no such attribute, so fall back to the root node's cumulative bounds.
    if hasattr(ao, "aabb"):
        foot = -float(ao.aabb.min[1])
    else:
        foot = -float(ao.root_scene_node.cumulative_bb.min[1])
    parked = mn.Vector3(0.0, -50.0, 0.0)

    lo, hi = pf.get_bounds()
    size_m, origin = map_extent_for_scene(np.array(lo), np.array(hi))
    smap = SemanticMap(size_m=size_m, origin=origin, camera_height=CAM_HEIGHT)
    fx = FPV_RES / (2.0 * math.tan(math.radians(90.0) / 2.0))
    K = np.array([[fx, 0, FPV_RES / 2], [0, fx, FPV_RES / 2], [0, 0, 1]], np.float32)

    # Retry from fresh starts, relaxing clearance: some floors are all corridor.
    route = []
    for clearance in (0.7, 0.7, 0.55, 0.55, 0.4, 0.4, 0.3):
        route = resample(build_floor_route(pf, clearance=clearance), spacing=0.05)
        if len(route) >= 200:
            break
    want = int(seconds * fps)
    if len(route) < 8:
        print(f"  {name}: route too short, skipping")
        sim.close()
        return 0
    while len(route) < want:
        route = route + route[::-1]
    route = route[:want]

    card = np.full((H, W, 3), BG, dtype=np.uint8)
    _put(card, "GOAT-Lite", (60, 290), 1.6, ACCENT, 3)
    _put(card, "Hello Robot Stretch in a simulated home", (62, 338), 0.75, FG)
    _put(card, name, (62, 394), 0.95, FG, 2)
    _put(card, f"{pf.navigable_area:.0f} m2 navigable floor area", (62, 432), 0.6, DIM)
    for _ in range(int(fps * 1.5)):
        writer.append_data(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))

    heading, boom, cam_az, cam_hi = None, BOOM_BACK, 0.0, 0.0
    cramped = 0
    track, written = [], 0

    for i, p in enumerate(route):
        j = min(i + 10, len(route) - 1)
        d = route[j] - p
        if float(np.hypot(d[0], d[2])) > 1e-4:
            target = math.atan2(-d[0], -d[2])
            if heading is None:
                heading = target
            err = (target - heading + math.pi) % (2 * math.pi) - math.pi
            heading += 0.12 * err
        heading = heading or 0.0

        st = habitat_sim.AgentState()
        st.position = p.tolist()
        st.rotation = _yaw_quat(heading)
        robot_agent.set_state(st)

        # ── first-person, robot parked away so it never occludes its own camera
        ao.translation = parked
        obs = sim.get_sensor_observations(agent_ids=0)
        rgb, depth = obs["rgb"][:, :, :3].copy(), obs["depth"].copy()
        ss = robot_agent.get_state().sensor_states["rgb"]
        pose = np.eye(4)
        pose[:3, :3] = qt.as_rotation_matrix(ss.rotation)
        pose[:3, 3] = ss.position
        smap.update_from_depth(depth, pose, K)
        xy = np.array([p[0], p[2]])
        track.append(xy)
        dets = det.detect(rgb) if det is not None else []

        # ── chase camera: boom behind the robot, shortened where walls intrude
        fwd = np.array([-math.sin(heading), 0.0, -math.cos(heading)])
        pivot = p + np.array([0.0, PIVOT_UP, 0.0])

        def boom_at(az, hi):
            # camera direction: behind the robot, swung `az` radians around it.
            # `hi` in [0, 1] blends from the normal boom to a short, high one
            # that looks down over the robot -- the only view that works in a
            # passage too narrow for any angle behind it.
            back = -np.array([-math.sin(heading + az), 0.0, -math.cos(heading + az)])
            b_len = BOOM_BACK + (0.8 - BOOM_BACK) * hi
            b_up = BOOM_UP + (1.25 - BOOM_UP) * hi
            want_ = pivot + back * b_len + np.array([0.0, b_up, 0.0])
            side = np.array([math.cos(heading + az), 0.0, -math.sin(heading + az)])
            # three rays -- centre and either shoulder -- because a door frame
            # can clear the centre line yet still hide the mast, which is off-centre
            # rays fan out from the centre: starting them beside the robot put
            # them almost inside the wall in a narrow passage, capping every
            # candidate at the 0.35 m minimum
            clear = min(boom_length(sim, pivot, want_ + side * o)
                        for o in (-0.25, 0.0, 0.25))
            return want_, clear

        # In a corridor the straight-back boom collapses onto the mast and the
        # frame is all robot. Score angles round the back by clearance and pick
        # the roomiest, preferring straight behind.
        #
        # Score by the WORST clearance along the swing from the current angle,
        # not just at the destination: the camera glides there, and when only
        # the endpoint was checked the intermediate angles hit walls, the boom
        # collapsed to 0.35 m mid-swing and took ~40 frames to regrow
        # (measured: cramped in 20-25% of frames).
        grid = np.radians(np.arange(-80, 81, 10))
        best_az, best_hi, best_score = cam_az, cam_hi, -1e9
        for hi in (0.0, 1.0):
            clear = np.array([boom_at(a, hi)[1] for a in grid])
            here = int(np.argmin(np.abs(grid - cam_az)))
            for k, az in enumerate(grid):
                # the angle we're leaving doesn't count -- if it's blocked,
                # every swing would inherit that and the camera stays stuck
                lo_k, hi_k = sorted((here, k))
                path = [c for j, c in enumerate(clear[lo_k:hi_k + 1], lo_k)
                        if j != here or k == here]
                swing = float(min(path))
                score = (min(swing, 1.6) - 0.35 * abs(az) - 0.15 * abs(az - cam_az)
                         - 0.3 * hi)
                if score > best_score:
                    best_az, best_hi, best_score = az, hi, score
        cam_az += 0.15 * (best_az - cam_az)
        cam_hi += 0.10 * (best_hi - cam_hi)
        desired, allowed = boom_at(cam_az, cam_hi)
        full = float(np.linalg.norm(desired - pivot))
        # shrink at once (never clip), grow back slowly (no jitter)
        boom = min(allowed, boom + 0.05 * full, full)
        cramped += boom < 0.9
        cam = pivot + (desired - pivot) / full * boom
        look_at = p + np.array([0.0, 0.55, 0.0]) + fwd * 0.6
        cs = habitat_sim.AgentState()
        cs.position = cam.tolist()
        cs.rotation = _look_rotation(cam, look_at)
        chase_agent.set_state(cs, infer_sensor_states=False)

        ao.translation = mn.Vector3(float(p[0]), float(p[1]) + foot, float(p[2]))
        ao.rotation = mn.Quaternion.rotation(mn.Rad(heading + ROBOT_YAW_OFFSET),
                                             mn.Vector3(0, 1, 0))
        chase = sim.get_sensor_observations(agent_ids=1)["chase"][:, :, :3]

        # ── compose
        frame = np.full((H, W, 3), BG, dtype=np.uint8)
        frame[64:64 + CHASE_RES[0], 24:24 + CHASE_RES[1]] = cv2.cvtColor(
            np.ascontiguousarray(chase), cv2.COLOR_RGB2BGR)
        _put(frame, "THIRD-PERSON VIEW", (24, 54), 0.5, DIM)

        fpv = cv2.resize(draw_detections(rgb, dets), (300, 300))
        frame[64:364, 900:1200] = fpv
        _put(frame, "ROBOT CAMERA  (what the agent sees)", (900, 54), 0.45, DIM)

        top = render_topdown(smap, xy, heading, track=track, out_size=300)
        frame[400:400 + top.shape[0], 900:900 + top.shape[1]] = top
        _put(frame, "MAP IT IS BUILDING", (900, 390), 0.45, DIM)

        occ = smap.get_occupancy()
        pct = 100.0 * (occ != -1).sum() / occ.size
        seen = sorted({dd.cls_name for dd in dets})
        _put(frame, name, (24, 708), 0.55, FG)
        _put(frame, f"explored {pct:4.1f}%   sees: "
                    f"{', '.join(seen)[:30] if seen else '-'}", (330, 708), 0.5, DIM)
        _put(frame, "GOAT-Lite", (1140, 708), 0.55, ACCENT)

        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        written += 1

    sim.close()
    print(f"  {name}: {written} frames, camera cramped (<0.9 m) in "
          f"{100.0 * cramped / max(written, 1):.1f}%", flush=True)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Scene tour with a visible Stretch robot")
    ap.add_argument("--scene-dirs", nargs="+",
                    default=["00832-qyAac8rV8Zk", "00810-CrMo8WxCyVb", "00819-6D36GQHuP8H"])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-detector", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hm3d-root", default="data/hm3d")
    ap.add_argument("--out", default="outputs/robot_tour.mp4")
    args = ap.parse_args()

    if not os.path.exists("data/robots/hab_stretch/urdf/hab_stretch.urdf"):
        sys.exit("missing data/robots/hab_stretch -- see the module docstring")

    det = None
    if not args.no_detector:
        try:
            from src.perception.detector import YOLODetector
            det = YOLODetector(device=args.device)
        except Exception as e:
            print(f"detector unavailable ({type(e).__name__}); rendering without")

    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=None, quality=7)
    rng = np.random.default_rng(args.seed)
    for n in args.scene_dirs:
        tour_scene(os.path.join(args.hm3d_root, n), args.seconds, args.fps, writer, rng, det)
    writer.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
