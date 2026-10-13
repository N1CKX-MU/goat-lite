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


class TestSaveResults:
    """results.csv used a hand-listed fieldname set. Adding a field to
    SubtaskResult therefore broke the writer -- and because results are
    checkpointed per scene, the crash lands hours into a run."""

    def test_round_trip_includes_every_field(self, tmp_path):
        import csv as _csv
        import json as _json
        from dataclasses import fields
        from src.eval.report import save_results
        from src.eval.runner import SubtaskResult

        results = [
            SubtaskResult(
                episode_id="3", subtask_index=0, category="mirror",
                modality="category", success=True, spl=0.8,
                distance_to_goal=0.91, steps_taken=15,
                actual_path_length=3.2, shortest_path_length=2.9,
                reason="success", wall_time_s=1.5, detector_in_view=False,
            ),
            SubtaskResult(
                episode_id="3", subtask_index=1, category="picture",
                modality="language", success=False, spl=0.0,
                distance_to_goal=3.15, steps_taken=500,
                actual_path_length=9.0, shortest_path_length=4.0,
                reason="timeout", wall_time_s=40.0, detector_in_view=True,
            ),
        ]
        save_results(results, str(tmp_path))

        with open(tmp_path / "results.csv") as f:
            rows = list(_csv.DictReader(f))
        assert len(rows) == 2
        for f_ in fields(SubtaskResult):
            assert f_.name in rows[0]

        summary = _json.loads((tmp_path / "summary.json").read_text())
        assert summary["overall_sr"] == 0.5
        # Scored success whose detector stayed silent must not be counted here.
        assert summary["sr_detector_agrees"] == 0.0
        assert summary["detector_in_view_rate"] == 0.5

        lines = (tmp_path / "failures.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        assert _json.loads(lines[0])["reason"] == "timeout"


class TestResumeRoundTrip:
    def test_load_results_reads_back_what_save_wrote(self, tmp_path):
        from src.eval.report import save_results, load_results
        from src.eval.runner import SubtaskResult

        original = [
            SubtaskResult(
                episode_id="7", subtask_index=2, category="mirror",
                modality="category", success=True, spl=0.5,
                distance_to_goal=0.8, steps_taken=40,
                actual_path_length=5.0, shortest_path_length=4.0,
                reason="success", wall_time_s=9.0, detector_in_view=True,
                ever_matched=True, node_error_m=0.15,
                steps_searching=10, steps_approaching=25, steps_verifying=5,
                explored_m2=42.5,
            ),
        ]
        save_results(original, str(tmp_path))
        back = load_results(str(tmp_path))

        assert len(back) == 1
        r = back[0]
        assert r.episode_id == "7" and r.subtask_index == 2
        assert r.success is True and r.detector_in_view is True
        assert r.steps_taken == 40
        assert abs(r.node_error_m - 0.15) < 1e-9
        assert abs(r.explored_m2 - 42.5) < 1e-9

    def test_load_results_is_empty_without_a_file(self, tmp_path):
        from src.eval.report import load_results
        assert load_results(str(tmp_path)) == []

    def test_load_results_tolerates_a_missing_column(self, tmp_path):
        """A CSV written before a diagnostic field existed must still resume."""
        import csv as _csv
        from src.eval.report import load_results
        with open(tmp_path / "results.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=[
                "episode_id", "subtask_index", "category", "modality",
                "success", "spl", "distance_to_goal", "steps_taken",
                "actual_path_length", "shortest_path_length", "reason",
                "wall_time_s"])
            w.writeheader()
            w.writerow(dict(
                episode_id="1", subtask_index=0, category="rug",
                modality="image", success="False", spl="0.0",
                distance_to_goal="3.0", steps_taken="500",
                actual_path_length="7.0", shortest_path_length="5.0",
                reason="timeout", wall_time_s="30.0"))
        back = load_results(str(tmp_path))
        assert len(back) == 1
        assert back[0].detector_in_view is False   # default
        assert back[0].ever_matched is False


class TestResumablePrefix:
    """Resume used to key on (episode_id, subtask_index). GOAT-Bench numbers
    episodes PER SCENE, so over the 30-episode dev subset 218 subtasks share
    only 86 distinct such pairs and episode id "9" appears in five scenes. A
    scene was therefore skipped because an unrelated earlier scene had produced
    the same ids -- silently dropping scene 6 of 22 from a real run."""

    def _row(self, episode_id, idx):
        from src.eval.runner import SubtaskResult
        return SubtaskResult(
            episode_id=episode_id, subtask_index=idx, category="mirror",
            modality="category", success=False, spl=0.0,
            distance_to_goal=2.0, steps_taken=500, actual_path_length=1.0,
            shortest_path_length=1.0, reason="timeout",
        )

    def test_counts_only_complete_leading_scenes(self):
        from src.eval.report import resumable_prefix
        scene_keys = [[("0", 0), ("0", 1)], [("1", 0)], [("2", 0), ("2", 1)]]
        prior = [self._row("0", 0), self._row("0", 1), self._row("1", 0)]

        kept, n = resumable_prefix(prior, scene_keys)
        assert n == 2
        assert len(kept) == 3

    def test_colliding_episode_ids_do_not_over_skip(self):
        """The regression. Scene 1 is episode "9" with 5 subtasks; scene 2 is a
        DIFFERENT scene that also calls its episode "9" with 5 subtasks. With
        only scene 1 on disk, exactly one scene is resumable."""
        from src.eval.report import resumable_prefix
        keys = [("9", i) for i in range(5)]
        scene_keys = [list(keys), list(keys)]
        prior = [self._row("9", i) for i in range(5)]

        kept, n = resumable_prefix(prior, scene_keys)
        assert n == 1, "a later scene sharing episode ids must still be run"
        assert len(kept) == 5

    def test_stops_at_the_first_mismatch(self):
        from src.eval.report import resumable_prefix
        scene_keys = [[("0", 0)], [("1", 0)], [("2", 0)]]
        # Scene 2's row is missing, so scene 3's row cannot be trusted either.
        prior = [self._row("0", 0), self._row("2", 0)]

        kept, n = resumable_prefix(prior, scene_keys)
        assert n == 1
        assert len(kept) == 1

    def test_short_final_scene_is_redone(self):
        from src.eval.report import resumable_prefix
        scene_keys = [[("0", 0), ("0", 1), ("0", 2)]]
        prior = [self._row("0", 0), self._row("0", 1)]

        kept, n = resumable_prefix(prior, scene_keys)
        assert n == 0 and kept == []

    def test_nothing_on_disk(self):
        from src.eval.report import resumable_prefix
        assert resumable_prefix([], [[("0", 0)]]) == ([], 0)


class TestEpisodeCounting:
    def test_counts_episodes_that_share_an_id(self, tmp_path):
        """Episode ids repeat across scenes, so a distinct-id count undercounts.
        Each episode contributes exactly one subtask_index == 0."""
        import json as _json
        from src.eval.report import save_results
        from src.eval.runner import SubtaskResult

        def row(ep, idx):
            return SubtaskResult(
                episode_id=ep, subtask_index=idx, category="rug",
                modality="image", success=False, spl=0.0,
                distance_to_goal=2.0, steps_taken=500,
                actual_path_length=1.0, shortest_path_length=1.0,
                reason="timeout",
            )

        # Three different scenes, every one of them calling its episode "9".
        results = [row("9", 0), row("9", 1),
                   row("9", 0), row("9", 1),
                   row("9", 0)]
        save_results(results, str(tmp_path))
        summary = _json.loads((tmp_path / "summary.json").read_text())
        assert summary["total_episodes"] == 3
        assert summary["total_subtasks"] == 5
