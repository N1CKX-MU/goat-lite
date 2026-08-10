"""Tests for src/sim/goat_dataset.py — episode parsing and goal resolution.

Regression cover for a bug that made category subtasks unscoreable: GOAT-Bench
writes category ("object") tasks as ``[category, "object", None]``, so there is
no object_id to look up and ``goal_info`` is legitimately None. Scoring off
``goal_info`` alone therefore failed every category subtask — ~37% of
val_unseen — no matter how the agent behaved. ``goal_candidates`` carries the
instances that count instead.
"""

import gzip
import json
import os

import numpy as np
import pytest
import quaternion as qt

from src.sim.goat_dataset import load_scene_episodes, quaternion_from_coeffs


SCENE_BASE = "XXXXscene.basis.glb"


def _shard(tmp_path, tasks, goals):
    """Write a minimal GOAT-Bench content shard and return its path."""
    data = {
        "episodes": [{
            "episode_id": 0,
            "scene_id": f"hm3d/val//00000-XXXXscene/{SCENE_BASE}",
            "start_position": [0.0, 0.0, 0.0],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "tasks": tasks,
        }],
        "goals": goals,
    }
    p = os.path.join(tmp_path, "shard.json.gz")
    with gzip.open(p, "wt") as f:
        json.dump(data, f)
    return p


def _goal(cat, oid, pos, desc=""):
    return {
        "object_category": cat,
        "object_id": oid,
        "position": pos,
        "lang_desc": desc,
        "image_goals": [],
        "view_points": [],
    }


class TestCategorySubtasks:
    def test_category_subtask_gets_all_instances_as_candidates(self, tmp_path):
        goals = {
            f"{SCENE_BASE}_freezer": [
                _goal("freezer", "freezer_2", [1.0, 0.0, 5.0]),
                _goal("freezer", "freezer_3", [2.0, 0.0, 6.0]),
            ]
        }
        eps = load_scene_episodes(
            _shard(str(tmp_path), [["freezer", "object", None]], goals),
            SCENE_BASE,
        )
        sub = eps[0].subtasks[0]

        assert sub.modality == "category"
        assert sub.goal_key is None
        # No single referenced instance...
        assert sub.goal_info is None
        # ...but both instances count as reaching the goal.
        assert len(sub.goal_candidates) == 2
        assert {c.object_id for c in sub.goal_candidates} == {"freezer_2", "freezer_3"}

    def test_category_subtask_is_scoreable(self, tmp_path):
        """The actual regression: a category subtask must expose a position."""
        goals = {f"{SCENE_BASE}_rug": [_goal("rug", "rug_1", [3.0, 0.0, 4.0])]}
        eps = load_scene_episodes(
            _shard(str(tmp_path), [["rug", "object", None]], goals), SCENE_BASE
        )
        sub = eps[0].subtasks[0]
        positions = [c.position for c in sub.goal_candidates]
        assert positions, "category subtask has no goal position -> always fails"
        np.testing.assert_allclose(positions[0], [3.0, 0.0, 4.0])


class TestSpecificInstanceSubtasks:
    @pytest.mark.parametrize("raw_modality", ["description", "image"])
    def test_targets_exactly_the_referenced_instance(self, tmp_path, raw_modality):
        goals = {
            f"{SCENE_BASE}_piano": [
                _goal("piano", "piano_1", [1.0, 0.0, 1.0], "the upright piano"),
                _goal("piano", "piano_9", [9.0, 0.0, 9.0], "the grand piano"),
            ]
        }
        eps = load_scene_episodes(
            _shard(str(tmp_path), [["piano", raw_modality, "piano_9"]], goals),
            SCENE_BASE,
        )
        sub = eps[0].subtasks[0]

        assert sub.goal_info is not None
        assert sub.goal_info.object_id == "piano_9"
        # Must NOT widen to every piano in the scene.
        assert len(sub.goal_candidates) == 1
        assert sub.goal_candidates[0].object_id == "piano_9"

    def test_unresolvable_object_id_yields_no_candidates(self, tmp_path):
        goals = {f"{SCENE_BASE}_vase": [_goal("vase", "vase_1", [1.0, 0.0, 1.0])]}
        eps = load_scene_episodes(
            _shard(str(tmp_path), [["vase", "description", "vase_404"]], goals),
            SCENE_BASE,
        )
        sub = eps[0].subtasks[0]
        assert sub.goal_info is None
        assert sub.goal_candidates == []


class TestQuaternionFromCoeffs:
    """habitat JSON stores rotations as (x, y, z, w). Reading them as
    (w, x, y, z) turns a yaw into a ~180 degree roll -- the agent spawns upside
    down, depth projects below the map's height band, and the agent never
    moves. The forward vector is unchanged by the mix-up, so only the 'up'
    vector reveals it."""

    @staticmethod
    def _up(q):
        return qt.as_rotation_matrix(q) @ np.array([0.0, 1.0, 0.0])

    def test_identity(self):
        q = quaternion_from_coeffs([0.0, 0.0, 0.0, 1.0])
        assert q.w == pytest.approx(1.0)
        assert (q.x, q.y, q.z) == (0.0, 0.0, 0.0)

    def test_component_order_is_xyzw(self):
        q = quaternion_from_coeffs([0.1, 0.2, 0.3, 0.4])
        assert (q.x, q.y, q.z, q.w) == pytest.approx((0.1, 0.2, 0.3, 0.4))

    def test_yaw_only_rotation_keeps_up_vector_up(self):
        # ~180 degrees of yaw about Y, the shape of a real GOAT-Bench start pose
        q = quaternion_from_coeffs([0.0, 0.9995, 0.0, 0.0323])
        assert self._up(q)[1] == pytest.approx(1.0, abs=1e-6)

    def test_from_float_array_would_flip_the_agent(self):
        """Guards the actual regression rather than just the happy path."""
        coeffs = [0.0, 0.9995, 0.0, 0.0323]
        assert self._up(quaternion_from_coeffs(coeffs))[1] > 0.99
        assert self._up(qt.from_float_array(coeffs))[1] < -0.99

    def test_all_real_start_rotations_are_upright(self, tmp_path):
        goals = {f"{SCENE_BASE}_book": [_goal("book", "book_1", [0.0, 0.0, 1.0])]}
        for rot in ([0.0, 0.0323, 0.0, -0.9995],
                    [0.0, 0.9999, 0.0, 0.0115],
                    [0.0, 0.945, 0.0, 0.327]):
            data_path = _shard(str(tmp_path), [["book", "object", None]], goals)
            import gzip, json
            with gzip.open(data_path, "rt") as f:
                d = json.load(f)
            d["episodes"][0]["start_rotation"] = rot
            with gzip.open(data_path, "wt") as f:
                json.dump(d, f)
            ep = load_scene_episodes(data_path, SCENE_BASE)[0]
            assert self._up(quaternion_from_coeffs(ep.start_rotation))[1] > 0.9


class TestEpisodeParsing:
    def test_modalities_normalized_and_indexed(self, tmp_path):
        goals = {
            f"{SCENE_BASE}_book": [_goal("book", "book_1", [0.0, 0.0, 1.0], "a book")],
        }
        tasks = [
            ["book", "object", None],
            ["book", "description", "book_1"],
            ["book", "image", "book_1"],
        ]
        eps = load_scene_episodes(_shard(str(tmp_path), tasks, goals), SCENE_BASE)
        subs = eps[0].subtasks

        assert [s.modality for s in subs] == ["category", "language", "image"]
        assert [s.subtask_index for s in subs] == [0, 1, 2]
        assert subs[1].goal_info.lang_desc == "a book"

    def test_missing_category_in_goals_is_not_fatal(self, tmp_path):
        eps = load_scene_episodes(
            _shard(str(tmp_path), [["ghost", "object", None]], {}), SCENE_BASE
        )
        sub = eps[0].subtasks[0]
        assert sub.goal_candidates == []
        assert sub.goal_info is None
