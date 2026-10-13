"""GoatAgent — top-level sense-plan-act state machine for object navigation."""

from __future__ import annotations

import enum
import math

import numpy as np

from src.sim.env import Observation
from src.planning.astar import plan_astar
from src.planning.exploration import choose_frontier
from src.mapping.frontier import find_frontiers
from src.agent.action import bearing_error, path_to_action


# Turn ~360° (30°/turn) surveying for frontiers before concluding the area is
# exhausted. Giving up after only a couple of turns quits with a near-empty map.
SEARCH_GIVEUP_TURNS = 12
# When a matched node is unreachable, try a few replans/turns before falling
# back to exploration rather than stopping (a stop counts as failure).
APPROACH_GIVEUP_TURNS = 4
# Waypoint distance ahead of the agent along the path, in grid cells. One
# forward action covers 0.25 m = 5 cells at the default 0.05 m resolution, so
# the aim point must be at least that far or every step overshoots it.
LOOKAHEAD_CELLS = 10
# A waypoint within this many cells counts as reached and is retired. Must be
# comfortably larger than one forward step (5 cells) or the agent can step past
# a waypoint without ever "arriving" at it and orbit it instead.
ARRIVE_CELLS = 7
# The end of the path is only "reached" this close (0.10 m). Using the looser
# intermediate radius here strands the agent just outside its goal.
FINAL_ARRIVE_CELLS = 2
# Re-plan periodically: the map changes every step, and a path planned through
# what later turns out to be a wall was otherwise followed to its end.
REPLAN_EVERY_STEPS = 15
# Verification budget, in steps. One full 360° survey at 30°/turn. The old
# budget of 5 could not even bring a goal 150° off-axis into frame, so a goal
# behind the agent was unconfirmable no matter how long the subtask ran.
VERIFY_GIVEUP_STEPS = 12
# A target within this bearing counts as "in frame". The camera's horizontal
# FOV is 90° (so 45° either side) and one turn is 30°, which leaves margin for
# the object's own extent at the edge of the image.
VERIFY_FACING_TOL = math.radians(30.0)
# Commit to a stop only once the matched node is this fraction of the success
# distance away, closing in first if necessary. Two measurements bracket it.
# Stopping at the boundary leaves no margin for the node's own position error:
# two dev-eval subtasks stopped 1.02 m from the true instance, failures by 2 cm.
# Going nearer than this is also wrong -- the detector fires from only 9% of
# poses at 0.8 m against 34% at 1.5 m (6.35), and 31.7% of dev subtasks need
# more than 0.5 m of standoff before the goal's centroid is even inside the
# vertical FOV (6.42). 0.8 keeps ~0.2 m of slack at both ends. Sweep axis.
COMMIT_DISTANCE_FRAC = 0.8


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
        self._steps_since_replan = 0
        self._frontier_target: tuple[int, int] | None = None
        self._rejected_nodes: set[int] = set()

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
        self._steps_since_replan = 0
        self._frontier_target = None
        # Rejections are per-subtask: a node that could not be confirmed from
        # one arrival may well be confirmable when approached from elsewhere.
        self._rejected_nodes = set()

        if not keep_memory:
            # Memory and map persist across subtasks within an episode -- that
            # is the lifelong property under test -- and are dropped between
            # episodes. This used to be a no-op for want of a clear() method on
            # either subsystem, so one agent per scene meant nodes and occupancy
            # leaked from one episode into the next, confounding the very curve
            # the experiment measures. Subsystems are duck-typed, so a fake
            # without clear() must still reset cleanly.
            for subsystem in (self.memory, self.semantic_map):
                clear = getattr(subsystem, "clear", None)
                if callable(clear):
                    clear()

    def act(self, obs: Observation) -> int:
        """Run one sense-plan-act cycle. Returns action int: 0=stop, 1=fwd, 2=left, 3=right."""
        # Timeout check
        if self.state == AgentState.DONE:
            return 0

        self.step_count += 1
        self._steps_since_replan += 1
        if self.step_count >= self.max_steps:
            self.state = AgentState.DONE
            return 0

        # A blocked forward means the map thinks a cell is free that is not.
        # Record that evidence and force a replan, otherwise the agent grinds
        # against the wall for the rest of the subtask.
        if getattr(obs, "collided", False):
            self._note_collision(obs)

        # 1. Perception: detect objects, update map and memory
        detections = self.perception.process(obs, step=self.step_count)
        self.semantic_map.update_from_depth(obs.depth, obs.pose, self.intrinsics)
        if detections:
            self.semantic_map.update_from_detections(detections)
        self.memory.update(detections, self.step_count)

        # 2. Goal matching
        self._current_goal = obs.current_goal
        # Pass the agent's position so category/image goals resolve to the
        # nearest qualifying instance rather than the most confident one.
        matched_node, score = self.matcher.match(
            self._current_goal,
            self.memory,
            self._agent_xy(obs),
            exclude_ids=self._rejected_nodes,
        )

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
            if self._matched_node is None:
                self.state = AgentState.SEARCHING
            else:
                decision = self._verify_decide(obs)
                if decision is not None:
                    return decision

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

        if self._needs_replan():
            occ = self.semantic_map.get_occupancy()
            goal_ij = self._approach_cell(occ, agent_ij, target_ij)
            self._set_path(plan_astar(occ, agent_ij, goal_ij))

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

        action = self._follow_path(obs, agent_ij)
        if action is None:
            # Path finished or invalid; force a fresh plan on the next tick.
            self._current_path = None
            return 2
        return action

    def _act_searching(self, obs: Observation) -> int:
        """Explore via frontiers."""
        agent_ij = self.semantic_map.world_to_grid(self._agent_xy(obs))
        occ = self.semantic_map.get_occupancy()
        explored = self.semantic_map.get_explored()

        # Need a new frontier target?
        if self._needs_replan():
            frontiers = find_frontiers(occ, explored)
            target = choose_frontier(frontiers, agent_ij, self._visited_frontiers)

            if target is None:
                self._no_plan_count += 1
                if self._no_plan_count >= SEARCH_GIVEUP_TURNS:
                    self.state = AgentState.DONE
                    return 0
                return 2  # spin to survey for frontiers

            self._set_path(plan_astar(occ, agent_ij, target.centroid_ij))

            if self._current_path is None:
                self._visited_frontiers.add(target.centroid_ij)
                self._no_plan_count += 1
                if self._no_plan_count >= SEARCH_GIVEUP_TURNS:
                    self.state = AgentState.DONE
                    return 0
                return 2
            else:
                self._frontier_target = target.centroid_ij
                self._no_plan_count = 0

        action = self._follow_path(obs, agent_ij)
        if action is None:
            # Path to this frontier is finished. Retire it now -- marking on
            # SELECTION instead blacklists a frontier the agent is still driving
            # toward, and with choose_frontier's proximity radius that quickly
            # exhausts every candidate and forces a premature give-up.
            if self._frontier_target is not None:
                self._visited_frontiers.add(self._frontier_target)
                self._frontier_target = None
            self._current_path = None
            return 2
        return action

    def _approach_cell(self, occ, agent_ij, target_ij):
        """A standable cell from which the target is within success_distance.

        Most GOAT categories are wall-mounted or wall-adjacent -- picture,
        mirror, window glass, radiator. Their estimated position therefore lands
        ON the wall, that cell reads OCCUPIED, and planning straight to it
        returns None every time, so the agent can never approach them at all.
        Observed as APPROACHING with a zero-length path, turning in place while
        2.7 m from a picture it had correctly matched.

        Navigate to a viewpoint beside the object instead, which is what the
        success criterion actually asks for: be within 1 m of it, not on it.
        """
        gs = occ.shape[0]
        tr, tc = int(target_ij[0]), int(target_ij[1])
        if not (0 <= tr < gs and 0 <= tc < occ.shape[1]):
            return target_ij
        if occ[tr, tc] != 1:
            return target_ij  # already standable, nothing to do

        radius = max(1, int(self.success_distance / self.semantic_map._resolution))
        ar, ac = agent_ij
        best, best_key = None, None
        r0, r1 = max(0, tr - radius), min(gs, tr + radius + 1)
        c0, c1 = max(0, tc - radius), min(occ.shape[1], tc + radius + 1)
        for r in range(r0, r1):
            for c in range(c0, c1):
                if occ[r, c] != 0:  # must be known-free to stand on
                    continue
                d_target = math.hypot(r - tr, c - tc)
                if d_target > radius:
                    continue
                # Closest to the object, breaking ties toward the agent so the
                # detour stays short.
                key = (round(d_target, 3), math.hypot(r - ar, c - ac))
                if best_key is None or key < best_key:
                    best, best_key = (r, c), key
        return best if best is not None else target_ij

    def _verify_decide(self, obs: Observation) -> int | None:
        """Turn to face the matched node, then commit. Never drives forward.

        Three measurements shape this.

        1. Episode 24 subtask 2: memory had the mirror to within 0.15 m and
           pure pursuit delivered the agent to 0.19 m of it, but VERIFYING
           issued FORWARD, so the agent pressed nose-to-surface against the
           wall the mirror hangs on and the object left the frame entirely.
           `goal_in_view` was False for all 250 steps. Arriving inside the
           success radius and facing the goal are separate requirements.
        2. Probing the detector around episode 24's goals: it fires from 4 of
           56 poses at 0.5 m and 5 of 56 at 0.8 m, against 19 of 56 at 1.5 m.
           Inside the success radius -- where the criterion *requires* the
           agent to be -- it is nearly blind, so waiting for a fresh detection
           before stopping discards almost every correct arrival. The decision
           therefore rests on accumulated instance memory, which was built from
           the distances where the detector does work.
        3. Holding still buys nothing: the detector is deterministic, so an
           unchanged pose yields the same frame and the same answer.

        Returns an action to take, or None to fall through to the normal
        state-based action selection.
        """
        err = bearing_error(
            self._agent_xy(obs),
            float(obs.compass),
            self._matched_node.world_xyz[[0, 2]],
        )
        if abs(err) <= VERIFY_FACING_TOL:
            dist = self._distance_to_node(obs, self._matched_node)
            if dist > COMMIT_DISTANCE_FRAC * self.success_distance:
                # Aligned but still on the edge of the radius. Close in, so the
                # node's own position error cannot push the stop outside it.
                # Forward is only safe here *because* it is bounded: it stops
                # at the commit band and never drives the agent onto the object.
                return 1
            # Looking at it, comfortably inside, and the detector has already
            # had its say on this very frame at the top of the tick.
            self.state = AgentState.DONE
            return 0
        if self._verify_steps < VERIFY_GIVEUP_STEPS:
            return 2 if err > 0 else 3
        # A full survey and the node never came into frame -- its estimated
        # position is not where the agent can look. Drop it and move on rather
        # than reverting to APPROACHING, which re-enters VERIFYING on the next
        # tick (still inside the radius) with the counter reset: measured as a
        # livelock of 32 VERIFYING ticks in 40 on a synthetic arrival.
        self._reject_matched_node()
        return None

    def _reject_matched_node(self) -> None:
        """Give up on the current node for the rest of this subtask."""
        if self._matched_node is not None:
            self._rejected_nodes.add(self._matched_node.node_id)
        self._matched_node = None
        self._current_path = None
        self._path_idx = 0
        self._no_plan_count = 0
        self._verify_steps = 0
        self.state = AgentState.SEARCHING

    def _note_collision(self, obs: Observation) -> None:
        """Mark the cell just ahead as occupied and invalidate the path.

        Habitat refused the motion, so whatever is there is solid regardless of
        what depth suggested. Without this the agent has no way to learn from a
        blocked step -- env.step() used to discard the collision flag entirely.
        """
        xy = self._agent_xy(obs)
        heading = float(obs.compass)
        ahead = xy + np.array([-math.sin(heading), -math.cos(heading)]) * 0.25
        r, c = self.semantic_map.world_to_grid(ahead)
        self.semantic_map.mark_occupied(r, c)
        self._current_path = None

    # ── Path bookkeeping ──

    def _set_path(self, path) -> None:
        self._current_path = path
        self._path_idx = 0
        self._steps_since_replan = 0

    def _needs_replan(self) -> bool:
        """Replan when there is no path, it is exhausted, it has gone stale, or
        the remaining route now runs through cells the map calls occupied."""
        if self._current_path is None or self._path_idx >= len(self._current_path):
            return True
        if self._steps_since_replan >= REPLAN_EVERY_STEPS:
            return True
        occ = self.semantic_map.get_occupancy()
        for (r, c) in self._current_path[self._path_idx:]:
            if 0 <= r < occ.shape[0] and 0 <= c < occ.shape[1] and occ[r, c] == 1:
                return True
        return False

    def _follow_path(self, obs: Observation, agent_ij: tuple[int, int]) -> int:
        """Follow the current path toward a waypoint at least one step away.

        Two unit bugs used to live here. The aim point was 3 cells (0.15 m)
        ahead while one forward action covers 0.25 m, so every step overshot the
        waypoint it was aimed at; and ``_path_idx`` advanced 4 cells per step
        while the body moved 5, so the cursor drifted monotonically behind the
        agent and was never resynchronised against its true position.

        The cursor is now re-anchored to the closest point on the path each
        call, and the waypoint is chosen a fixed distance ahead of that anchor.
        """
        if self._current_path is None or not self._current_path:
            return None  # caller replans; never a blind forward

        path = self._current_path
        ar, ac = agent_ij

        def dist_to(i: int) -> float:
            pr, pc = path[i]
            return math.hypot(pr - ar, pc - ac)

        # Pure pursuit. Retire INTERMEDIATE waypoints the agent has reached, then
        # aim at the first one at least LOOKAHEAD_CELLS away.
        #
        # Picking the globally *nearest* path cell instead does not work: once
        # the agent overshoots its aim point the nearest cell is still behind it,
        # the cursor never advances, and it turns back toward the same waypoint
        # forever -- 500 steps at 0.00 m with the index pinned at 0.
        last = len(path) - 1
        while self._path_idx < last and dist_to(self._path_idx) <= ARRIVE_CELLS:
            self._path_idx += 1

        target_i = self._path_idx
        while target_i < last and dist_to(target_i) < LOOKAHEAD_CELLS:
            target_i += 1

        # The FINAL waypoint gets a much tighter arrival radius than the
        # intermediate ones. Retiring it at ARRIVE_CELLS too means that as soon
        # as the remaining path is short, every waypoint counts as reached, this
        # returns None, the caller turns and replans, and the agent orbits just
        # outside the goal -- subtasks stalling at 1.4-2.1 m from a 1.0 m
        # success radius.
        if target_i == last and dist_to(last) <= FINAL_ARRIVE_CELLS:
            return None  # genuinely arrived; caller replans

        next_ij = path[target_i]

        action = path_to_action(
            agent_ij=agent_ij,
            heading=float(obs.compass),
            next_ij=next_ij,
        )
        # path_to_action returns 0 when the waypoint IS the agent's cell. That
        # means "waypoint reached", not "stop the episode" -- propagating it
        # would end the subtask on a bookkeeping artefact.
        if action == 0:
            return None
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
        """Is the goal object currently visible?

        Compares against the matched instance's class rather than
        ``goal.value``. For a language goal ``value`` is the whole description
        ("bathroom mirror. start by locating the sink..."), which never equals a
        detection's ``cls_name`` -- so verification could never succeed, the FSM
        bounced between VERIFYING and APPROACHING forever, and every language
        subtask timed out even while standing on top of the target.
        """
        if not detections or self._current_goal is None:
            return False
        if self._matched_node is not None:
            target = self._matched_node.cls_name
        else:
            target = self._current_goal.value
        for d in detections:
            if hasattr(d, 'cls_name') and d.cls_name == target:
                return True
        return False
