"""Tests for src/agent/goat_agent.py — GoatAgent state machine.

Uses fakes for all subsystems so tests run without GPU or Habitat.
"""

import math
import numpy as np
import pytest

from src.agent.goat_agent import GoatAgent, AgentState
from src.sim.env import Observation, GoalSpec
from src.perception.pipeline import PerceivedInstance
from src.memory.instance_db import InstanceNode


# ── Fakes ──

def _make_obs(
    gps=(0.0, 0.0), heading=0.0, depth_val=2.0, goal=None
) -> Observation:
    h, w = 256, 256
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = gps[0]
    pose[2, 3] = gps[1]
    return Observation(
        rgb=np.full((h, w, 3), 128, dtype=np.uint8),
        depth=np.full((h, w), depth_val, dtype=np.float32),
        pose=pose,
        compass=heading,
        gps=np.array(gps, dtype=np.float64),
        current_goal=goal or GoalSpec(modality="category", value="chair"),
    )


def _make_embed(seed=0):
    rng = np.random.RandomState(seed)
    e = rng.randn(512).astype(np.float32)
    return e / np.linalg.norm(e)


class FakePerception:
    """Returns configurable detections."""
    def __init__(self, detections=None):
        self.detections = detections or []
    def process(self, obs, step=0):
        return self.detections


class FakeMemory:
    """Tracks updates, returns configurable nodes."""
    def __init__(self, nodes=None):
        self._nodes = nodes or []
        self.update_calls = 0
    def update(self, perceived, step):
        self.update_calls += 1
    def all_nodes(self):
        return list(self._nodes)
    def query_by_class(self, cls_id):
        return [n for n in self._nodes if n.cls_id == cls_id]
    def query_by_class_name(self, cls_name):
        return [n for n in self._nodes if n.cls_name == cls_name]
    def query_by_embedding(self, embed, top_k=5):
        return [(n, 0.5) for n in self._nodes[:top_k]]


class FakeMatcher:
    """Returns configurable match results."""
    def __init__(self, result=None):
        self.result = result or (None, 0.0)
    def match(self, goal, db, agent_xy=None):
        return self.result


class FakeMap:
    """Minimal map fake."""
    def __init__(self, grid_size=100):
        self._gs = grid_size
        self._occ = np.zeros((grid_size, grid_size), dtype=np.int8)
        self._explored = np.zeros((grid_size, grid_size), dtype=bool)
    def update_from_depth(self, depth, pose, intrinsics):
        pass
    def update_from_detections(self, instances):
        pass
    def get_occupancy(self):
        return self._occ
    def get_explored(self):
        return self._explored
    def world_to_grid(self, xy):
        return (int(xy[1] * 10) + 50, int(xy[0] * 10) + 50)
    def grid_to_world(self, ij):
        return np.array([(ij[1] - 50) / 10.0, (ij[0] - 50) / 10.0])


def _make_node(node_id=0, cls_name="chair", world_xyz=(2.0, 0.5, 3.0)):
    return InstanceNode(
        node_id=node_id,
        cls_id=0,
        cls_name=cls_name,
        world_xyz=np.array(world_xyz, dtype=np.float64),
        clip_embed=_make_embed(node_id),
        confidence=0.9,
        first_seen_step=0,
        last_seen_step=0,
        n_observations=3,
        best_thumbnail=np.zeros((64, 64, 3), dtype=np.uint8),
    )


def _make_agent(**overrides) -> GoatAgent:
    defaults = dict(
        perception=FakePerception(),
        memory=FakeMemory(),
        matcher=FakeMatcher(),
        semantic_map=FakeMap(),
        intrinsics=np.eye(3, dtype=np.float32),
        max_steps=500,
        success_distance=1.0,
    )
    defaults.update(overrides)
    return GoatAgent(**defaults)


# ── Tests ──

class TestAgentInit:
    def test_initial_state_is_searching(self):
        agent = _make_agent()
        obs = _make_obs()
        agent.reset(obs)
        assert agent.state == AgentState.SEARCHING

    def test_reset_clears_step_count(self):
        agent = _make_agent()
        obs = _make_obs()
        agent.reset(obs)
        assert agent.step_count == 0


class TestStateTransitions:
    def test_searching_to_approaching_on_match(self):
        node = _make_node(world_xyz=(5.0, 0.5, 5.0))
        matcher = FakeMatcher(result=(node, 0.9))
        agent = _make_agent(matcher=matcher)
        agent.reset(_make_obs())

        agent.act(_make_obs())
        assert agent.state == AgentState.APPROACHING

    def test_stays_searching_when_no_match(self):
        agent = _make_agent(matcher=FakeMatcher(result=(None, 0.0)))
        agent.reset(_make_obs())
        agent.act(_make_obs())
        assert agent.state == AgentState.SEARCHING

    def test_approaching_to_verifying_when_close(self):
        # Node at (0.5, 0.5, 0.5), agent at origin — within 1.0m
        node = _make_node(world_xyz=(0.5, 0.5, 0.5))
        matcher = FakeMatcher(result=(node, 0.9))
        agent = _make_agent(matcher=matcher, success_distance=1.0)
        agent.reset(_make_obs(gps=(0.0, 0.0)))

        agent.act(_make_obs(gps=(0.0, 0.0)))
        # Should transition to APPROACHING then immediately to VERIFYING
        # since we're within success_distance
        assert agent.state in (AgentState.APPROACHING, AgentState.VERIFYING)

    def test_searching_surveys_before_giving_up(self):
        """With no match and no frontiers, the agent should keep surveying
        (turning) rather than quitting after a couple of steps."""
        agent = _make_agent(matcher=FakeMatcher(result=(None, 0.0)))
        agent.reset(_make_obs())
        for _ in range(5):
            action = agent.act(_make_obs())
            assert agent.state == AgentState.SEARCHING
            assert action == 2  # spin to survey, not stop

    def test_timeout_returns_stop(self):
        agent = _make_agent(max_steps=3)
        agent.reset(_make_obs())
        for _ in range(5):
            action = agent.act(_make_obs())
        assert action == 0  # stop
        assert agent.state == AgentState.DONE


class TestPerceptionUpdates:
    def test_perception_runs_each_step(self):
        det = PerceivedInstance(
            cls_id=0, cls_name="chair", conf=0.8,
            bbox=(50, 50, 150, 150),
            crop_thumbnail=np.zeros((64, 64, 3), dtype=np.uint8),
            clip_embed=_make_embed(0),
            world_xyz=np.array([1.0, 0.5, 2.0]),
            seen_step=0,
        )
        perception = FakePerception(detections=[det])
        memory = FakeMemory()
        agent = _make_agent(perception=perception, memory=memory)
        agent.reset(_make_obs())

        agent.act(_make_obs())
        assert memory.update_calls == 1

        agent.act(_make_obs())
        assert memory.update_calls == 2


class TestDoneState:
    def test_done_always_returns_stop(self):
        agent = _make_agent(max_steps=1)
        agent.reset(_make_obs())
        agent.act(_make_obs())  # hits timeout
        action = agent.act(_make_obs())
        assert action == 0

    def test_reset_keep_memory(self):
        """Reset with keep_memory=True should preserve memory but change state."""
        memory = FakeMemory(nodes=[_make_node()])
        agent = _make_agent(memory=memory)
        agent.reset(_make_obs())
        agent.act(_make_obs())

        agent.reset(_make_obs(), keep_memory=True)
        assert agent.state == AgentState.SEARCHING
        assert agent.step_count == 0
        # Memory should still have nodes
        assert len(memory.all_nodes()) == 1


class TestActionOutput:
    def test_act_returns_valid_action(self):
        agent = _make_agent()
        agent.reset(_make_obs())
        action = agent.act(_make_obs())
        assert action in (0, 1, 2, 3)

    def test_multiple_steps_no_crash(self):
        """Agent should run for many steps without errors."""
        agent = _make_agent(max_steps=100)
        agent.reset(_make_obs())
        for i in range(50):
            obs = _make_obs(gps=(float(i) * 0.1, 0.0))
            action = agent.act(obs)
            assert action in (0, 1, 2, 3)
