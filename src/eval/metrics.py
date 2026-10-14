"""Evaluation metrics for GOAT-Bench: Success Rate (SR) and SPL."""

from __future__ import annotations

import math

from collections import defaultdict

import numpy as np


def subtask_success(
    agent_pos: np.ndarray,
    goal_pos: np.ndarray,
    goal_in_view: bool,
    success_distance: float = 1.0,
) -> bool:
    """Check if a subtask was successful.

    Success requires BOTH:
    1. Agent stopped within success_distance of the goal (XZ plane distance).
    2. The correct goal instance is in the agent's view when it stops.

    Args:
        agent_pos: [3] agent position (x, y, z).
        goal_pos: [3] goal instance position (x, y, z).
        goal_in_view: whether the goal instance is visible in the final frame.
        success_distance: maximum allowed distance in meters.

    Returns:
        True if subtask succeeded.
    """
    # Euclidean distance on XZ plane (ignore Y / height)
    dx = agent_pos[0] - goal_pos[0]
    dz = agent_pos[2] - goal_pos[2]
    dist = np.sqrt(dx * dx + dz * dz)
    return bool(dist <= success_distance and goal_in_view)


def instance_in_view(
    instance_xyz: np.ndarray,
    camera_pose: np.ndarray,
    intrinsics: np.ndarray,
    depth: np.ndarray,
    window: int = 2,
    occlusion_tol_m: float = 1.0,
) -> bool:
    """Is a ground-truth instance inside the camera frustum and unoccluded?

    This is the benchmark's visibility criterion -- a geometric property of the
    scene and the camera. It deliberately does NOT ask the agent's detector.

    Scoring with the detector (which is what this module's caller used to do)
    is wrong on three counts: the benchmark has no access to the agent's
    internals, so it cannot be GOAT-Bench's definition; it makes SR a function
    of detector recall twice over, once for finding the goal and again for
    being credited with it; and it is measurably blind in exactly the band the
    success radius forces the agent into. Probed over episode 24 of the dev
    subset, the finetuned detector fired on the goal class from 4/56 poses at
    0.5 m and 5/56 at 0.8 m, against 19/56 at 1.5 m -- so a geometrically
    perfect stop inside 1.0 m was rejected about nine times in ten.

    Occlusion is tested against the depth buffer rather than a semantic mask,
    because the local HM3D val scenes ship no semantic assets. The instance
    position is a centroid, so the object's own surface renders NEARER than
    that: the test therefore allows the visible surface to sit up to
    ``occlusion_tol_m`` (or half the range, whichever is more permissive) in
    front of the centroid, and only calls it occluded beyond that. A wall in
    the way fails this comfortably -- the stated failure case is a wall-mounted
    goal approached from the next room, 0.9 m from the centroid with the wall
    rendering at 0.1 m.

    Args:
        instance_xyz: [3] ground-truth world position of the instance.
        camera_pose: [4, 4] world-from-camera SE(3). Must be the SENSOR pose.
        intrinsics: [3, 3] camera matrix.
        depth: [H, W] float32 depth in metres, as habitat returns it (0 where
            there is no geometry).
        window: half-width in pixels of the depth patch sampled around the
            projected centroid.
        occlusion_tol_m: how far in front of the centroid the nearest visible
            surface may sit before the instance counts as occluded.

    Returns:
        True if the instance projects inside the image, lies in front of the
        camera, and nothing nearer blocks it.
    """
    p_world = np.asarray(instance_xyz, dtype=np.float64)[:3]
    R = np.asarray(camera_pose, dtype=np.float64)[:3, :3]
    t = np.asarray(camera_pose, dtype=np.float64)[:3, 3]
    p_cam = R.T @ (p_world - t)

    # OpenGL convention: the camera looks down -Z, so a point in front has
    # z < 0 and its range is -z. Same frame as depth_to_pointcloud_camera,
    # whose inverse this is.
    rng = -float(p_cam[2])
    if rng <= 1e-6:
        return False

    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    u = cx + float(p_cam[0]) * fx / rng
    v = cy - float(p_cam[1]) * fy / rng

    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return False

    patch = depth[
        max(0, vi - window): min(h, vi + window + 1),
        max(0, ui - window): min(w, ui + window + 1),
    ]
    rendered = patch[patch > 0.0]
    if rendered.size == 0:
        return False  # nothing rendered there at all; do not claim visibility

    nearest_allowed = max(0.5 * rng, rng - occlusion_tol_m)
    return bool(float(rendered.max()) >= nearest_allowed)


def compute_spl(
    success: bool,
    shortest_path: float,
    actual_path: float,
) -> float:
    """Compute Success weighted by Path Length (SPL).

    SPL = success * shortest_path / max(shortest_path, actual_path)

    Args:
        success: whether the subtask succeeded.
        shortest_path: geodesic distance from start to goal (from episode metadata).
        actual_path: total distance the agent actually traveled.

    Returns:
        SPL value in [0, 1]. Never nan.

    A non-finite ``shortest_path`` is not hypothetical: habitat returns inf
    when start and goal sit on different navmesh islands, and on the full
    val_unseen run 48 of 1013 subtasks had one. Seven of those *succeeded*
    with ``actual_path`` 0.0 -- the agent spawned on the goal -- and inf/inf
    is nan, which propagated through the mean and made the run's headline
    ``overall_spl`` nan (6.66).

    Such a subtask has no defined path-efficiency, because the quantity SPL
    normalises by does not exist. It is scored 1.0 when the agent succeeded
    without travelling, since no shorter path is possible than none, and 0.0
    otherwise -- the pessimistic reading, so a broken geodesic can never
    flatter the result.
    """
    if not success:
        return 0.0
    if not math.isfinite(shortest_path):
        return 1.0 if actual_path == 0.0 else 0.0
    if shortest_path == 0.0 and actual_path == 0.0:
        return 1.0
    denom = max(shortest_path, actual_path)
    if denom == 0.0:
        return 1.0
    return float(shortest_path / denom)


def _mean_spl(rows: list[dict]) -> float:
    """Mean SPL over rows, ignoring any that are not finite.

    compute_spl no longer emits nan, but results.csv files written before
    6.66 contain some, and a single nan turns the whole aggregate into nan
    rather than into a slightly wrong number -- a failure mode that is silent
    in a summary and fatal in a report.
    """
    vals = [float(r["spl"]) for r in rows]
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return 0.0
    return sum(finite) / len(finite)


def aggregate_metrics(results: list[dict]) -> dict:
    """Aggregate per-subtask results into summary metrics.

    Args:
        results: list of dicts, each with keys:
            - success (bool)
            - spl (float)
            - modality (str): "category", "language", or "image"
            - subtask_index (int): 0-based index within the episode

    Returns:
        Dict with:
            - overall_sr, overall_spl
            - by_modality: {modality: {sr, spl, count}}
            - by_subtask_index: {index: {sr, spl, count}}
    """
    if not results:
        return {
            "overall_sr": 0.0,
            "overall_spl": 0.0,
            "by_modality": {},
            "by_subtask_index": {},
        }

    # Overall
    successes = [r["success"] for r in results]
    spls = [r["spl"] for r in results]
    overall_sr = sum(successes) / len(successes)
    overall_spl = _mean_spl(results)

    # By modality
    by_mod: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mod[r["modality"]].append(r)

    by_modality = {}
    for mod, rs in by_mod.items():
        s = [r["success"] for r in rs]
        sp = rs
        by_modality[mod] = {
            "sr": sum(s) / len(s),
            "spl": _mean_spl(sp),
            "count": len(rs),
        }

    # By subtask index
    by_idx: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_idx[r["subtask_index"]].append(r)

    by_subtask_index = {}
    for idx, rs in sorted(by_idx.items()):
        s = [r["success"] for r in rs]
        sp = rs
        by_subtask_index[idx] = {
            "sr": sum(s) / len(s),
            "spl": _mean_spl(sp),
            "count": len(rs),
        }

    return {
        "overall_sr": overall_sr,
        "overall_spl": overall_spl,
        "by_modality": by_modality,
        "by_subtask_index": by_subtask_index,
    }
