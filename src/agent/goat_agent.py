"""GoatAgent — top-level sense-plan-act state machine for object navigation."""

from __future__ import annotations

import enum
import math

import numpy as np

from src.sim.env import Observation
from src.planning.astar import plan_astar
from src.planning.exploration import choose_frontier
from src.mapping.frontier import find_frontiers
from src.agent.action import path_to_action


# Turn ~360° (30°/turn) surveying for frontiers before concluding the area is
# exhausted. Giving up after only a couple of turns quits with a near-empty map.
SEARCH_GIVEUP_TURNS = 12
# When a matched node is unreachable, try a few replans/turns before falling
# back to exploration rather than stopping (a stop counts as failure).
APPROACH_GIVEUP_TURNS = 4


class AgentState(enum.Enum):
    SEARCHING = "searching"
    APPROACHING = "approaching"
    VERIFYING = "verifying"
    DONE = "done"


class GoatAgent:
    """High-level agent that wires perception, memory, matching, mapping, and planning.

    States:
      SEARCHING   — no match in memory, exploring via frontiers
      APPROACHING — match found, planning + executing path to it
      VERIFYING   — within threshold distance, confirm visually
      DONE        — called stop
    """

    def __init__(
        self,
        perception,
        memory,
        matcher,
        semantic_map,
        intrinsics: np.ndarray,
        max_steps: int = 500,
        success_distance: float = 1.0,
    ):
        self.perception = perception
        self.memory = memory
        self.matcher = matcher
        self.semantic_map = semantic_map
        self.intrinsics = intrinsics
        self.max_steps = max_steps
        self.success_distance = success_distance

        self.state = AgentState.SEARCHING
        self.step_count = 0
        self._current_goal = None
        self._matched_node = None
        self._current_path: list[tuple[int, int]] | None = None
        self._path_idx = 0
        self._no_plan_count = 0
        self._visited_frontiers: set[tuple[int, int]] = set()
        self._verify_steps = 0

    def reset(self, obs: Observation, keep_memory: bool = False) -> None:
        """Reset for a new subtask. If keep_memory, preserve memory and map."""
        self.state = AgentState.SEARCHING
        self.step_count = 0
        self._current_goal = obs.current_goal
        self._matched_node = None
        self._current_path = None
        self._path_idx = 0
        self._no_plan_count = 0
        self._visited_frontiers = set()
        self._verify_steps = 0

        if not keep_memory:
            pass  # memory.clear() would go here if InstanceDatabase had a clear method

    def act(self, obs: Observation) -> int:
        """Run one sense-plan-act cycle. Returns action int: 0=stop, 1=fwd, 2=left, 3=right."""
        # Timeout check
        if self.state == AgentState.DONE:
            return 0

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.state = AgentState.DONE
            return 0

        # 1. Perception: detect objects, update map and memory
        detections = self.perception.process(obs, step=self.step_count)
        self.semantic_map.update_from_depth(obs.depth, obs.pose, self.intrinsics)
        if detections:
            self.semantic_map.update_from_detections(detections)
        self.memory.update(detections, self.step_count)

        # 2. Goal matching
        self._current_goal = obs.current_goal
        matched_node, score = self.matcher.match(self._current_goal, self.memory)

        # 3. State transitions
        if self.state == AgentState.SEARCHING:
            if matched_node is not None and score > 0.0:
                self._matched_node = matched_node
                self.state = AgentState.APPROACHING
                self._current_path = None
                self._no_plan_count = 0

        if self.state == AgentState.APPROACHING:
            # Update match if we get a better one
            if matched_node is not None and score > 0.0:
                self._matched_node = matched_node

            if self._matched_node is not None:
                dist = self._distance_to_node(obs, self._matched_node)
                if dist <= self.success_distance:
                    self.state = AgentState.VERIFYING
                    self._verify_steps = 0

        if self.state == AgentState.VERIFYING:
            self._verify_steps += 1
            # Check if goal object is in current detections
            if self._goal_in_view(detections):
                self.state = AgentState.DONE
                return 0  # success stop
            if self._verify_steps >= 5:
                # Revert to approaching, mark stale
                self.state = AgentState.APPROACHING
                self._current_path = None
            else:
                return 1  # move forward slowly to get a better view

        # 4. Action selection based on state
        if self.state == AgentState.APPROACHING:
            return self._act_approaching(obs)
        elif self.state == AgentState.SEARCHING:
            return self._act_searching(obs)

        return 0  # fallback stop

    def _act_approaching(self, obs: Observation) -> int:
        """Plan and follow path to matched node."""
        if self._matched_node is None:
            self.state = AgentState.SEARCHING
            return self._act_searching(obs)

        target_xy = self._matched_node.world_xyz[[0, 2]]  # x, z
        agent_ij = self.semantic_map.world_to_grid(self._agent_xy(obs))
        target_ij = self.semantic_map.world_to_grid(target_xy)

        # Replan if no path
        if self._current_path is None or self._path_idx >= len(self._current_path):
            occ = self.semantic_map.get_occupancy()
            self._current_path = plan_astar(occ, agent_ij, target_ij)
            self._path_idx = 0

            if self._current_path is None:
                self._no_plan_count += 1
                if self._no_plan_count >= APPROACH_GIVEUP_TURNS:
                    # Can't reach the matched node — go explore for a better
                    # vantage instead of stopping (which would count as failure).
                    self.state = AgentState.SEARCHING
                    self._matched_node = None
                    self._current_path = None
                    self._no_plan_count = 0
                return 2  # turn to try to find a way
            else:
                self._no_plan_count = 0

        return self._follow_path(obs, agent_ij)

    def _act_searching(self, obs: Observation) -> int:
        """Explore via frontiers."""
        agent_ij = self.semantic_map.world_to_grid(self._agent_xy(obs))
        occ = self.semantic_map.get_occupancy()
        explored = self.semantic_map.get_explored()

        # Need a new frontier target?
        if self._current_path is None or self._path_idx >= len(self._current_path):
            frontiers = find_frontiers(occ, explored)
            target = choose_frontier(frontiers, agent_ij, self._visited_frontiers)

            if target is None:
                self._no_plan_count += 1
                if self._no_plan_count >= SEARCH_GIVEUP_TURNS:
                    self.state = AgentState.DONE
                    return 0
                return 2  # spin to survey for frontiers

            self._current_path = plan_astar(occ, agent_ij, target.centroid_ij)
            self._path_idx = 0

            if self._current_path is None:
                self._visited_frontiers.add(target.centroid_ij)
                self._no_plan_count += 1
                if self._no_plan_count >= SEARCH_GIVEUP_TURNS:
                    self.state = AgentState.DONE
                    return 0
                return 2
            else:
                self._no_plan_count = 0

        return self._follow_path(obs, agent_ij)

    def _follow_path(self, obs: Observation, agent_ij: tuple[int, int]) -> int:
        """Follow the current path, advancing the path index."""
        if self._current_path is None or self._path_idx >= len(self._current_path):
            return 1  # forward fallback

        # Pick a waypoint a few steps ahead for smoother movement
        lookahead = min(self._path_idx + 3, len(self._current_path) - 1)
        next_ij = self._current_path[lookahead]

        action = path_to_action(
            agent_ij=agent_ij,
            heading=float(obs.compass),
            next_ij=next_ij,
        )

        # Advance path index if moving forward
        if action == 1:
            self._path_idx = lookahead + 1

        return action

    def _distance_to_node(self, obs: Observation, node) -> float:
        """Euclidean XZ distance from agent to node."""
        agent_xy = self._agent_xy(obs)
        node_xy = node.world_xyz[[0, 2]]
        return float(np.linalg.norm(agent_xy - node_xy))

    def _agent_xy(self, obs: Observation) -> np.ndarray:
        """Extract agent XZ position from observation."""
        return np.array([obs.gps[0], obs.gps[1]], dtype=np.float64)

    def _goal_in_view(self, detections) -> bool:
        """Check if any detection matches the current goal class."""
        if not detections or self._current_goal is None:
            return False
        goal_val = self._current_goal.value
        for d in detections:
            if hasattr(d, 'cls_name') and d.cls_name == goal_val:
                return True
        return False
