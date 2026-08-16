"""Closed-loop tests for GoatAgent._follow_path.

This function converts a grid path into discrete actions, and it has been the
single richest source of navigation bugs in the project -- every failure looked
identical from outside ("the agent spins in place and never moves"), and each
attempt to fix it by adjusting a constant broke a different case:

* aim point closer than one step  -> the agent overshoots and orbits it
* cursor advanced by a fixed count -> it drifts behind the agent forever
* cursor re-anchored to the NEAREST cell -> after any overshoot the nearest
  cell is behind the agent, so the cursor never advances
* final waypoint retired at the same radius as intermediate ones -> as soon as
  the remaining path is short every waypoint counts as reached and the agent
  orbits just outside its goal

So drive the follower in closed loop against a simulated body with the real
motion model (0.25 m forward, 30 degree turns, 0.05 m cells) and assert the
property that actually matters: it converges. These run in milliseconds, unlike
the 12-minute end-to-end eval they replace.
"""

import math

import numpy as np
import pytest

from src.agent.goat_agent import GoatAgent, LOOKAHEAD_CELLS, ARRIVE_CELLS

RES = 0.05          # metres per cell, matches SemanticMap default
STEP_M = 0.25       # HabitatEnv move_forward
TURN_RAD = math.radians(30.0)


class _Body:
    """Minimal kinematic stand-in for the habitat agent.

    Grid convention matches SemanticMap: row <- Z, col <- X. Heading follows
    HabitatEnv.get_compass, where forward is (-sin h, -cos h) in world XZ.
    """

    def __init__(self, xz=(0.0, 0.0), heading=0.0):
        self.xz = np.array(xz, dtype=np.float64)
        self.heading = heading

    @property
    def cell(self):
        return (int(round(self.xz[1] / RES)), int(round(self.xz[0] / RES)))

    def apply(self, action):
        if action == 1:
            fwd = np.array([-math.sin(self.heading), -math.cos(self.heading)])
            self.xz = self.xz + fwd * STEP_M
        elif action == 2:
            self.heading += TURN_RAD
        elif action == 3:
            self.heading -= TURN_RAD


class _Obs:
    def __init__(self, body):
        self.compass = body.heading
        self.gps = np.array([body.xz[0], body.xz[1]])
        self.collided = False


def _agent_with_path(path):
    """A GoatAgent with only the fields _follow_path touches."""
    agent = GoatAgent.__new__(GoatAgent)
    agent._current_path = path
    agent._path_idx = 0
    return agent


def _straight_path(n=60, row=0):
    """Path heading along +col (i.e. +X in world)."""
    return [(row, c) for c in range(n)]


def _run(agent, body, max_steps=400):
    """Drive the body with the follower. Returns (steps, actions, reached_end)."""
    actions = []
    for _ in range(max_steps):
        obs = _Obs(body)
        action = agent._follow_path(obs, body.cell)
        if action is None:
            return len(actions), actions, True
        actions.append(action)
        body.apply(action)
    return len(actions), actions, False


class TestConverges:
    def test_reaches_end_of_straight_path(self):
        path = _straight_path(60)          # 60 cells = 3.0 m
        agent = _agent_with_path(path)
        body = _Body(xz=(0.0, 0.0), heading=0.0)
        steps, actions, reached = _run(agent, body)

        assert reached, f"never reached the end in {steps} steps"
        # 3 m at 0.25 m/step is 12 forwards; allow generous turning overhead.
        assert steps < 80, f"took {steps} steps for a 3 m straight line"
        assert actions.count(1) >= 8, "barely moved forward"

    def test_cursor_advances_monotonically(self):
        """The failure signature of every past bug was a pinned cursor."""
        path = _straight_path(60)
        agent = _agent_with_path(path)
        body = _Body(xz=(0.0, 0.0), heading=0.0)

        seen = []
        for _ in range(120):
            obs = _Obs(body)
            action = agent._follow_path(obs, body.cell)
            seen.append(agent._path_idx)
            if action is None:
                break
            body.apply(action)

        assert seen == sorted(seen), "path index went backwards"
        assert seen[-1] > seen[0], "path index never advanced at all"

    def test_makes_real_ground_progress(self):
        """Guards the 500-steps-at-0.00-m symptom directly."""
        path = _straight_path(80)
        agent = _agent_with_path(path)
        body = _Body(xz=(0.0, 0.0), heading=0.0)
        start = body.xz.copy()
        _run(agent, body, max_steps=60)
        assert np.linalg.norm(body.xz - start) > 1.0, "agent barely moved"

    def test_does_not_orbit_when_starting_backwards(self):
        """Facing the wrong way must cost turns, not convergence."""
        path = _straight_path(60)
        agent = _agent_with_path(path)
        body = _Body(xz=(0.0, 0.0), heading=math.pi)   # facing away
        steps, actions, reached = _run(agent, body)
        assert reached, "did not recover from a reversed start"


class TestArrival:
    def test_short_path_still_drives_to_the_end(self):
        """A path shorter than the arrival radius must not instantly 'arrive'.

        Retiring the final waypoint at the intermediate radius made the agent
        stall 1.4-2.1 m short of goals with a 1.0 m success radius.
        """
        path = _straight_path(ARRIVE_CELLS + 3)   # deliberately short
        agent = _agent_with_path(path)
        body = _Body(xz=(0.0, 0.0), heading=0.0)
        start = body.xz.copy()
        _run(agent, body, max_steps=40)
        moved = float(np.linalg.norm(body.xz - start))
        assert moved > 0.10, f"only moved {moved:.3f} m on a short path"

    def test_reports_arrival_at_the_end(self):
        path = _straight_path(20)
        agent = _agent_with_path(path)
        # place the body essentially on the last cell
        body = _Body(xz=(19 * RES, 0.0), heading=0.0)
        agent._path_idx = 19
        obs = _Obs(body)
        assert agent._follow_path(obs, body.cell) is None

    def test_empty_and_missing_paths_are_handled(self):
        body = _Body()
        assert _agent_with_path(None)._follow_path(_Obs(body), body.cell) is None
        assert _agent_with_path([])._follow_path(_Obs(body), body.cell) is None


class TestAimPoint:
    def test_lookahead_exceeds_one_step(self):
        """A waypoint nearer than one step cannot be converged on, only orbited."""
        step_cells = STEP_M / RES          # 5 cells
        assert LOOKAHEAD_CELLS > step_cells
        assert ARRIVE_CELLS > step_cells
