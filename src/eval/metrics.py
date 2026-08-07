"""Evaluation metrics for GOAT-Bench: Success Rate (SR) and SPL."""

from __future__ import annotations

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
        SPL value in [0, 1].
    """
    if not success:
        return 0.0
    if shortest_path == 0.0 and actual_path == 0.0:
        return 1.0
    denom = max(shortest_path, actual_path)
    return float(shortest_path / denom)


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
    overall_spl = sum(spls) / len(spls)

    # By modality
    by_mod: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mod[r["modality"]].append(r)

    by_modality = {}
    for mod, rs in by_mod.items():
        s = [r["success"] for r in rs]
        sp = [r["spl"] for r in rs]
        by_modality[mod] = {
            "sr": sum(s) / len(s),
            "spl": sum(sp) / len(sp),
            "count": len(rs),
        }

    # By subtask index
    by_idx: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_idx[r["subtask_index"]].append(r)

    by_subtask_index = {}
    for idx, rs in sorted(by_idx.items()):
        s = [r["success"] for r in rs]
        sp = [r["spl"] for r in rs]
        by_subtask_index[idx] = {
            "sr": sum(s) / len(s),
            "spl": sum(sp) / len(sp),
            "count": len(rs),
        }

    return {
        "overall_sr": overall_sr,
        "overall_spl": overall_spl,
        "by_modality": by_modality,
        "by_subtask_index": by_subtask_index,
    }
