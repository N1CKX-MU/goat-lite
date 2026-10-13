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
    """Returns configurable match results, honouring the exclusion set."""
    def __init__(self, result=None):
        self.result = result or (None, 0.0)
        self.last_exclude = set()
    def match(self, goal, db, agent_xy=None, exclude_ids=None):
        self.last_exclude = set(exclude_ids or ())
        node, score = self.result
        if node is not None and node.node_id in self.last_exclude:
            return None, 0.0
        return node, score


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


# ── Verification: arriving is not the same as confirming ──

def _apply_turn(heading: float, action: int) -> float:
    """Advance a heading by one turn action (30 deg; left increases)."""
    if action == 2:
        return heading + math.radians(30.0)
    if action == 3:
        return heading - math.radians(30.0)
    return heading


def _verify_setup(node_xyz, detections=()):
    """An agent already inside the success radius of a matched node."""
    node = _make_node(world_xyz=node_xyz)
    matcher = FakeMatcher(result=(node, 0.9))
    agent = _make_agent(
        matcher=matcher,
        perception=FakePerception(detections=list(detections)),
        success_distance=1.0,
    )
    agent.reset(_make_obs(gps=(0.0, 0.0)))
    return agent, matcher, node


class TestVerification:
    """Measured on episode 24 subtask 2: the agent reached 0.19 m from the
    goal with `goal_in_view` False for the entire subtask, because VERIFYING
    drove forward instead of turning to face the object."""

    def test_verifying_turns_toward_a_goal_it_cannot_see(self):
        # Node 0.6 m BEHIND the agent: inside the success radius, out of frame.
        from src.agent.action import bearing_error
        agent, _, node = _verify_setup((0.0, 0.5, 0.6))

        heading = 0.0
        for _ in range(8):
            action = agent.act(_make_obs(gps=(0.0, 0.0), heading=heading))
            assert action in (2, 3), f"expected a turn, got {action}"
            heading = _apply_turn(heading, action)
            err = bearing_error(
                np.array([0.0, 0.0]), heading, node.world_xyz[[0, 2]]
            )
            if abs(err) <= math.radians(30.0):
                break
        else:
            pytest.fail("never turned to face the goal")

    def test_verifying_never_drives_forward(self):
        """Forward inside the success radius is what put the agent
        nose-to-surface with a wall-mounted goal, pushing it out of frame."""
        agent, _, _ = _verify_setup((0.0, 0.5, 0.6))
        heading = 0.0
        for _ in range(11):
            action = agent.act(_make_obs(gps=(0.0, 0.0), heading=heading))
            assert action != 1, "VERIFYING must not drive into the goal"
            heading = _apply_turn(heading, action)

    def test_verifying_stops_when_goal_comes_into_view(self):
        det = PerceivedInstance(
            cls_id=0, cls_name="chair", conf=0.8,
            bbox=(50, 50, 150, 150),
            crop_thumbnail=np.zeros((64, 64, 3), dtype=np.uint8),
            clip_embed=_make_embed(0),
            world_xyz=np.array([0.0, 0.5, 0.6]),
            seen_step=0,
        )
        agent, _, _ = _verify_setup((0.0, 0.5, 0.6), detections=[det])
        action = agent.act(_make_obs(gps=(0.0, 0.0)))
        assert action == 0
        assert agent.state == AgentState.DONE

    def test_commits_once_it_faces_the_goal(self):
        """Inside the success radius and facing its matched instance, the agent
        must decide. Waiting for a fresh detection instead is the defect: the
        detector fires from 5 of 56 poses at 0.8 m, so demanding one at the stop
        pose rejects nine arrivals in ten, and reverting to APPROACHING while
        still inside the radius re-enters VERIFYING on the next tick with the
        counter reset -- a livelock that ran to the 500-step timeout."""
        agent, _, _ = _verify_setup((0.0, 0.5, 0.6))
        heading = 0.0
        for tick in range(10):
            action = agent.act(_make_obs(gps=(0.0, 0.0), heading=heading))
            assert action != 1
            if action == 0:
                assert agent.state == AgentState.DONE
                break
            heading = _apply_turn(heading, action)
        else:
            pytest.fail("never committed to a decision")

    def test_gives_up_on_a_node_it_cannot_bring_into_frame(self):
        """Constant heading: the turns have no effect, standing in for an agent
        that cannot get the node into frame. It must drop the node and look
        elsewhere rather than verify it forever."""
        agent, matcher, node = _verify_setup((0.0, 0.5, 0.6))
        for _ in range(20):
            agent.act(_make_obs(gps=(0.0, 0.0), heading=0.0))
        assert node.node_id in matcher.last_exclude
        assert agent.state != AgentState.VERIFYING

    def test_rejection_lasts_only_for_the_subtask(self):
        agent, matcher, node = _verify_setup((0.0, 0.5, 0.6))
        for _ in range(20):
            agent.act(_make_obs(gps=(0.0, 0.0), heading=0.0))
        assert node.node_id in matcher.last_exclude

        agent.reset(_make_obs(gps=(0.0, 0.0)), keep_memory=True)
        agent.act(_make_obs(gps=(0.0, 0.0), heading=0.0))
        assert node.node_id not in matcher.last_exclude
