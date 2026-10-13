"""Tests for the image-goal rendering path in src/eval/runner.py.

`_embed_image_goal` is the one piece of the image-goal work that only runs
inside a real evaluation, so without these it would first execute hours into a
multi-hour run. The habitat-specific half -- that writing
`AgentState.sensor_states` and calling `set_state(..., infer_sensor_states=False)`
actually moves the camera -- was verified directly against the simulator
(engineering log 6.43); what is left to pin is this function's own logic:
which view it picks, that the agent's pose is restored whatever happens, and
that every failure degrades to None rather than losing the subtask.

Fakes throughout, so no GPU, no scene files.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.runner import _embed_image_goal
from src.sim.goat_dataset import GoalInfo, SubtaskSpec


# ── Fakes ──

class FakeSensorState:
    def __init__(self):
        self.position = np.zeros(3)
        self.rotation = None


class FakeAgentState:
    def __init__(self, position=(0.0, 0.0, 0.0), tag="original"):
        self.position = np.asarray(position, dtype=np.float64)
        self.rotation = None
        self.tag = tag
        self.sensor_states = {"rgb": FakeSensorState(),
                              "depth": FakeSensorState()}


class FakeAgent:
    """Records every set_state call so pose restoration is checkable."""

    def __init__(self):
        self._state = FakeAgentState()
        self.set_calls = []

    def get_state(self):
        return self._state

    def set_state(self, state, infer_sensor_states=True):
        self.set_calls.append((state, infer_sensor_states))
        self._state = state


class FakeObs:
    def __init__(self, rgb):
        self.rgb = rgb


class FakeEnv:
    def __init__(self, rgb=None):
        self._agent = FakeAgent()
        self._rgb = rgb if rgb is not None else np.full((8, 8, 3), 7, np.uint8)
        self.render_count = 0

    def _make_obs(self):
        self.render_count += 1
        return FakeObs(self._rgb)


class FakePerception:
    """Captures what it was asked to embed."""

    def __init__(self, result=None, raises=False):
        self.result = result
        self.raises = raises
        self.calls = []

    def embed_goal_image(self, rgb, cls_name):
        if self.raises:
            raise RuntimeError("detector exploded")
        self.calls.append((rgb, cls_name))
        return self.result


class FakeAgentWrapper:
    def __init__(self, perception):
        self.perception = perception


def _goal_info(image_goals=None, view_points=None):
    return GoalInfo(
        object_category="microwave",
        object_id="microwave_1",
        position=np.array([1.0, 0.5, 2.0]),
        lang_desc="",
        image_goals=image_goals or [],
        view_points=view_points or [],
    )


def _subtask(goal_info):
    return SubtaskSpec(
        category="microwave",
        modality="image",
        goal_key="microwave_1",
        goal_info=goal_info,
        subtask_index=0,
        goal_candidates=[goal_info] if goal_info else [],
    )


def _image_goal(position, coverage, rotation=(0.0, 0.0, 0.0, 1.0)):
    return {"position": list(position), "rotation": list(rotation),
            "object_coverage": coverage, "frame_coverage": 0.3,
            "hfov": 75, "image_dimensions": [512, 512]}


# ── Tests ──

class TestViewSelection:
    def test_picks_the_best_framed_image_goal(self):
        gi = _goal_info(image_goals=[
            _image_goal((1.0, 0.6, 1.0), coverage=0.20),
            _image_goal((9.0, 0.6, 9.0), coverage=0.91),   # best
            _image_goal((5.0, 0.6, 5.0), coverage=0.55),
        ])
        env = FakeEnv()
        embed = np.ones(4, dtype=np.float32)
        agent = FakeAgentWrapper(FakePerception(result=embed))

        out = _embed_image_goal(env, agent, _subtask(gi))

        assert out is embed
        # The sensor was moved to the best-framed pose, and habitat was told not
        # to recompute it from the agent base.
        moved, infer = env._agent.set_calls[0]
        assert infer is False
        assert np.allclose(moved.sensor_states["rgb"].position,
                           [9.0, 0.6, 9.0])
        assert np.allclose(moved.sensor_states["depth"].position,
                           [9.0, 0.6, 9.0])
        assert moved.sensor_states["rgb"].rotation is not None

    def test_falls_back_to_the_highest_iou_viewpoint(self):
        """With no pre-rendered goal views, the best ground-truth viewpoint is
        an AGENT state, so it is set as one."""
        gi = _goal_info(view_points=[
            {"agent_state": {"position": [1.0, -0.5, 1.0],
                             "rotation": [0.0, 0.0, 0.0, 1.0]}, "iou": 0.11},
            {"agent_state": {"position": [4.0, -0.5, 4.0],
                             "rotation": [0.0, 0.0, 0.0, 1.0]}, "iou": 0.77},
        ])
        env = FakeEnv()
        agent = FakeAgentWrapper(FakePerception(result=np.zeros(4)))

        _embed_image_goal(env, agent, _subtask(gi))

        moved, infer = env._agent.set_calls[0]
        assert np.allclose(np.asarray(moved.position), [4.0, -0.5, 4.0])
        assert infer is True  # an agent pose; sensors SHOULD be inferred

    def test_passes_the_rendered_frame_and_category_to_perception(self):
        rgb = np.full((8, 8, 3), 3, np.uint8)
        gi = _goal_info(image_goals=[_image_goal((1.0, 0.6, 1.0), 0.5)])
        env = FakeEnv(rgb=rgb)
        perception = FakePerception(result=np.zeros(4))

        _embed_image_goal(env, FakeAgentWrapper(perception), _subtask(gi))

        assert env.render_count == 1
        sent_rgb, cls_name = perception.calls[0]
        assert cls_name == "microwave"
        assert np.array_equal(sent_rgb, rgb)


class TestPoseRestoration:
    """The agent is mid-episode. Rendering a goal view must not move it."""

    def test_restores_the_pose_on_success(self):
        gi = _goal_info(image_goals=[_image_goal((9.0, 0.6, 9.0), 0.9)])
        env = FakeEnv()
        original = env._agent.get_state()

        _embed_image_goal(env, FakeAgentWrapper(
            FakePerception(result=np.zeros(4))), _subtask(gi))

        assert env._agent.get_state() is original
        assert env._agent.set_calls[-1][0] is original

    def test_restores_the_pose_when_perception_raises(self):
        gi = _goal_info(image_goals=[_image_goal((9.0, 0.6, 9.0), 0.9)])
        env = FakeEnv()
        original = env._agent.get_state()

        out = _embed_image_goal(env, FakeAgentWrapper(
            FakePerception(raises=True)), _subtask(gi))

        assert out is None
        assert env._agent.get_state() is original

    def test_restores_the_pose_when_the_goal_data_is_malformed(self):
        gi = _goal_info(image_goals=[{"object_coverage": 0.9}])  # no position
        env = FakeEnv()
        original = env._agent.get_state()

        out = _embed_image_goal(env, FakeAgentWrapper(
            FakePerception(result=np.zeros(4))), _subtask(gi))

        assert out is None
        assert env._agent.get_state() is original


class TestDegradesToNone:
    """Every failure must fall back to category matching, never lose a subtask."""

    def test_no_goal_info(self):
        st = SubtaskSpec(category="microwave", modality="image",
                         goal_key=None, goal_info=None, subtask_index=0)
        env = FakeEnv()
        assert _embed_image_goal(
            env, FakeAgentWrapper(FakePerception()), st) is None
        assert env.render_count == 0

    def test_no_views_at_all(self):
        env = FakeEnv()
        out = _embed_image_goal(env, FakeAgentWrapper(FakePerception()),
                                _subtask(_goal_info()))
        assert out is None
        assert env.render_count == 0
        assert env._agent.set_calls[-1][0] is not None  # pose still restored

    def test_perception_without_the_method(self):
        class Bare:
            pass

        gi = _goal_info(image_goals=[_image_goal((1.0, 0.6, 1.0), 0.5)])
        env = FakeEnv()
        assert _embed_image_goal(env, FakeAgentWrapper(Bare()),
                                 _subtask(gi)) is None

    def test_detector_silent_on_the_goal_picture(self):
        """The detector not firing on the goal's own picture is a legitimate
        outcome: there is then nothing honest to compare against."""
        gi = _goal_info(image_goals=[_image_goal((1.0, 0.6, 1.0), 0.5)])
        env = FakeEnv()
        assert _embed_image_goal(
            env, FakeAgentWrapper(FakePerception(result=None)),
            _subtask(gi)) is None
