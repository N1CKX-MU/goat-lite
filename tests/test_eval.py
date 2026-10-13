"""Tests for src/eval/metrics.py — SR, SPL, and per-modality aggregation."""

import math
import numpy as np
import pytest

from src.eval.metrics import (
    subtask_success,
    compute_spl,
    aggregate_metrics,
)


class TestSubtaskSuccess:
    def test_success_when_close_and_in_view(self):
        assert subtask_success(
            agent_pos=np.array([1.0, 0.0, 1.0]),
            goal_pos=np.array([1.0, 0.0, 1.5]),
            goal_in_view=True,
            success_distance=1.0,
        ) is True

    def test_fail_when_too_far(self):
        assert subtask_success(
            agent_pos=np.array([0.0, 0.0, 0.0]),
            goal_pos=np.array([5.0, 0.0, 5.0]),
            goal_in_view=True,
            success_distance=1.0,
        ) is False

    def test_fail_when_not_in_view(self):
        assert subtask_success(
            agent_pos=np.array([1.0, 0.0, 1.0]),
            goal_pos=np.array([1.0, 0.0, 1.2]),
            goal_in_view=False,
            success_distance=1.0,
        ) is False

    def test_exactly_at_threshold(self):
        # Distance == success_distance should still count
        assert subtask_success(
            agent_pos=np.array([0.0, 0.0, 0.0]),
            goal_pos=np.array([1.0, 0.0, 0.0]),
            goal_in_view=True,
            success_distance=1.0,
        ) is True

    def test_ignores_y_component(self):
        # Height difference shouldn't matter for distance
        assert subtask_success(
            agent_pos=np.array([0.0, 0.0, 0.0]),
            goal_pos=np.array([0.5, 5.0, 0.0]),
            goal_in_view=True,
            success_distance=1.0,
        ) is True


class TestComputeSPL:
    def test_perfect_path(self):
        # Agent took the shortest path
        spl = compute_spl(success=True, shortest_path=5.0, actual_path=5.0)
        assert spl == pytest.approx(1.0)

    def test_longer_path(self):
        spl = compute_spl(success=True, shortest_path=5.0, actual_path=10.0)
        assert spl == pytest.approx(0.5)

    def test_failed_subtask(self):
        spl = compute_spl(success=False, shortest_path=5.0, actual_path=3.0)
        assert spl == pytest.approx(0.0)

    def test_zero_shortest_path(self):
        # Edge case: already at goal
        spl = compute_spl(success=True, shortest_path=0.0, actual_path=0.0)
        assert spl == pytest.approx(1.0)

    def test_actual_shorter_than_shortest(self):
        # Can happen with noise — SPL formula uses max()
        spl = compute_spl(success=True, shortest_path=5.0, actual_path=3.0)
        assert spl == pytest.approx(1.0)


class TestAggregateMetrics:
    def test_basic_aggregation(self):
        results = [
            {"success": True, "spl": 1.0, "modality": "category", "subtask_index": 0},
            {"success": False, "spl": 0.0, "modality": "category", "subtask_index": 1},
            {"success": True, "spl": 0.5, "modality": "language", "subtask_index": 0},
            {"success": True, "spl": 0.8, "modality": "language", "subtask_index": 1},
        ]
        agg = aggregate_metrics(results)
        assert agg["overall_sr"] == pytest.approx(0.75)
        assert agg["overall_spl"] == pytest.approx((1.0 + 0.0 + 0.5 + 0.8) / 4)

    def test_per_modality_breakdown(self):
        results = [
            {"success": True, "spl": 1.0, "modality": "category", "subtask_index": 0},
            {"success": False, "spl": 0.0, "modality": "category", "subtask_index": 1},
            {"success": True, "spl": 0.5, "modality": "language", "subtask_index": 0},
        ]
        agg = aggregate_metrics(results)
        assert agg["by_modality"]["category"]["sr"] == pytest.approx(0.5)
        assert agg["by_modality"]["language"]["sr"] == pytest.approx(1.0)

    def test_per_subtask_index_breakdown(self):
        results = [
            {"success": True, "spl": 1.0, "modality": "category", "subtask_index": 0},
            {"success": False, "spl": 0.0, "modality": "category", "subtask_index": 1},
            {"success": True, "spl": 0.5, "modality": "language", "subtask_index": 0},
        ]
        agg = aggregate_metrics(results)
        # Subtask 0: 2/2 success, subtask 1: 0/1 success
        assert agg["by_subtask_index"][0]["sr"] == pytest.approx(1.0)
        assert agg["by_subtask_index"][1]["sr"] == pytest.approx(0.0)

    def test_empty_results(self):
        agg = aggregate_metrics([])
        assert agg["overall_sr"] == 0.0
        assert agg["overall_spl"] == 0.0


# ── Geometric visibility: the benchmark's "in the agent's field of view" ──

def _K(w=256, h=256, hfov_deg=90.0):
    f = w / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return np.array([[f, 0.0, w / 2.0],
                     [0.0, f, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _camera_at(x=0.0, z=0.0, y=1.41):
    """Camera at (x, y, z) looking down world -Z (habitat/OpenGL identity)."""
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3], pose[1, 3], pose[2, 3] = x, y, z
    return pose


def _depth(val, w=256, h=256):
    return np.full((h, w), val, dtype=np.float32)


class TestInstanceInView:
    """The scorer previously asked the agent's own YOLO whether the goal was
    visible. That is not the benchmark's criterion, it is self-referential, and
    it is measurably blind at the distance the criterion forces: probed over
    episode 24, the detector fired from 5 of 56 poses at 0.8 m but 19 of 56 at
    1.5 m, while success demands a stop inside 1.0 m."""

    def test_dead_ahead_is_visible(self):
        from src.eval.metrics import instance_in_view
        assert instance_in_view(
            np.array([0.0, 1.41, -2.0]), _camera_at(), _K(), _depth(1.8)
        ) is True

    def test_behind_the_camera_is_not_visible(self):
        from src.eval.metrics import instance_in_view
        assert instance_in_view(
            np.array([0.0, 1.41, 2.0]), _camera_at(), _K(), _depth(1.8)
        ) is False

    def test_outside_the_horizontal_fov_is_not_visible(self):
        from src.eval.metrics import instance_in_view
        # 90 deg hfov: at 2 m ahead, anything beyond +/-2 m lateral is out.
        assert instance_in_view(
            np.array([3.0, 1.41, -2.0]), _camera_at(), _K(), _depth(1.8)
        ) is False

    def test_wall_in_between_is_not_visible(self):
        from src.eval.metrics import instance_in_view
        # Object 3 m ahead, but the depth buffer says geometry at 0.3 m.
        assert instance_in_view(
            np.array([0.0, 1.41, -3.0]), _camera_at(), _K(), _depth(0.3)
        ) is False

    def test_close_range_object_is_visible(self):
        from src.eval.metrics import instance_in_view
        # Standing 0.8 m from the centroid, its surface renders at 0.65 m.
        assert instance_in_view(
            np.array([0.0, 1.41, -0.8]), _camera_at(), _K(), _depth(0.65)
        ) is True

    def test_wrong_side_of_a_thin_wall_is_not_visible(self):
        from src.eval.metrics import instance_in_view
        # A wall-mounted goal can be approached from the next room: 0.9 m from
        # the centroid, but the wall renders at 0.1 m.
        assert instance_in_view(
            np.array([0.0, 1.41, -0.9]), _camera_at(), _K(), _depth(0.1)
        ) is False

    def test_empty_depth_is_not_claimed_visible(self):
        from src.eval.metrics import instance_in_view
        # Habitat returns 0 where there is no geometry; HM3D scans have holes.
        assert instance_in_view(
            np.array([0.0, 1.41, -2.0]), _camera_at(), _K(), _depth(0.0)
        ) is False

    def test_translated_camera_is_handled(self):
        from src.eval.metrics import instance_in_view
        assert instance_in_view(
            np.array([5.0, 1.41, 3.0]), _camera_at(x=5.0, z=5.0), _K(), _depth(1.8)
        ) is True
