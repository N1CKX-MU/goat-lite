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
