"""Tests for src/planning/astar.py, src/planning/fmm.py, and src/agent/action.py."""

import math
import numpy as np
import pytest

from src.planning.astar import plan_astar
from src.mapping.frontier import Frontier


def _make_free_map(h=50, w=50):
    """All free (0) map."""
    return np.zeros((h, w), dtype=np.int8)


def _make_map_with_wall(h=50, w=50):
    """Free map with a wall across the middle, gap at col=25."""
    occ = np.zeros((h, w), dtype=np.int8)
    occ[25, :] = 1  # wall
    occ[25, 25] = 0  # gap
    return occ


# ── A* tests ──

class TestAstar:
    def test_straight_line_path(self):
        occ = _make_free_map()
        path = plan_astar(occ, (5, 5), (5, 40))
        assert path is not None
        assert len(path) >= 2
        assert path[0] == (5, 5)
        assert path[-1] == (5, 40)

    def test_path_around_wall(self):
        occ = _make_map_with_wall()
        path = plan_astar(occ, (10, 10), (40, 10))
        assert path is not None
        assert path[0] == (10, 10)
        assert path[-1] == (40, 10)
        # Path must go through the gap at col=25
        cols_at_wall_row = [c for r, c in path if r == 25]
        assert any(c == 25 for c in cols_at_wall_row)

    def test_no_path_fully_blocked(self):
        occ = np.zeros((50, 50), dtype=np.int8)
        occ[25, :] = 1  # complete wall, no gap
        path = plan_astar(occ, (10, 10), (40, 10))
        assert path is None

    def test_start_equals_goal(self):
        occ = _make_free_map()
        path = plan_astar(occ, (10, 10), (10, 10))
        assert path is not None
        assert len(path) == 1
        assert path[0] == (10, 10)

    def test_adjacent_cells(self):
        occ = _make_free_map()
        path = plan_astar(occ, (10, 10), (10, 11))
        assert path is not None
        assert len(path) == 2

    def test_diagonal_path(self):
        occ = _make_free_map()
        path = plan_astar(occ, (0, 0), (10, 10))
        assert path is not None
        # Diagonal should be shorter than Manhattan
        assert len(path) <= 12  # ~11 for pure diagonal

    def test_obstacle_inflation(self):
        """With inflation, path should stay away from obstacles."""
        occ = np.zeros((50, 50), dtype=np.int8)
        occ[25, 20:30] = 1  # small wall segment
        path = plan_astar(occ, (10, 25), (40, 25), inflate_cells=3)
        assert path is not None
        # Path should not pass through cells adjacent to the wall
        for r, c in path:
            if r == 25:
                assert c < 17 or c > 32, f"Path too close to wall at ({r},{c})"

    def test_out_of_bounds_start(self):
        occ = _make_free_map(50, 50)
        path = plan_astar(occ, (-1, -1), (10, 10))
        assert path is None

    def test_goal_on_occupied(self):
        occ = _make_free_map()
        occ[30, 30] = 1
        path = plan_astar(occ, (10, 10), (30, 30))
        assert path is None

    def test_large_map_performance(self):
        """A* on a 480x480 map (real size) should complete quickly."""
        import time
        occ = np.zeros((480, 480), dtype=np.int8)
        t0 = time.perf_counter()
        path = plan_astar(occ, (10, 10), (470, 470))
        elapsed = time.perf_counter() - t0
        assert path is not None
        assert elapsed < 2.0, f"A* took {elapsed:.2f}s on 480x480"


# ── FMM tests ──

_has_skfmm = False
try:
    import skfmm
    _has_skfmm = True
except ImportError:
    pass


@pytest.mark.skipif(not _has_skfmm, reason="skfmm not installed")
class TestFMM:
    def test_straight_line_path(self):
        from src.planning.fmm import plan_fmm
        occ = _make_free_map()
        path = plan_fmm(occ, (5, 5), (5, 40))
        assert path is not None
        assert len(path) >= 2
        assert abs(path[0][0] - 5) <= 1 and abs(path[0][1] - 5) <= 1
        assert abs(path[-1][0] - 5) <= 1 and abs(path[-1][1] - 40) <= 1

    def test_no_path_blocked(self):
        from src.planning.fmm import plan_fmm
        occ = np.zeros((50, 50), dtype=np.int8)
        occ[25, :] = 1
        path = plan_fmm(occ, (10, 10), (40, 10))
        assert path is None


# ── choose_frontier tests ──

class TestChooseFrontier:
    def test_chooses_nearest_large(self):
        from src.planning.exploration import choose_frontier
        f1 = Frontier(centroid_ij=(10, 10), size=50, cells=np.zeros((50, 2)))
        f2 = Frontier(centroid_ij=(40, 40), size=100, cells=np.zeros((100, 2)))
        agent_ij = (8, 8)
        chosen = choose_frontier([f1, f2], agent_ij, visited=set())
        # f1 is much closer, should win despite being smaller
        assert chosen is not None
        assert chosen.centroid_ij == (10, 10)

    def test_skips_visited(self):
        from src.planning.exploration import choose_frontier
        f1 = Frontier(centroid_ij=(10, 10), size=50, cells=np.zeros((50, 2)))
        f2 = Frontier(centroid_ij=(40, 40), size=100, cells=np.zeros((100, 2)))
        agent_ij = (8, 8)
        visited = {(10, 10)}  # f1's centroid is visited
        chosen = choose_frontier([f1, f2], agent_ij, visited=visited, visit_radius=3)
        assert chosen is not None
        assert chosen.centroid_ij == (40, 40)

    def test_no_frontiers_returns_none(self):
        from src.planning.exploration import choose_frontier
        chosen = choose_frontier([], (10, 10), visited=set())
        assert chosen is None


# ── Action selection tests ──

class TestActionSelection:
    def test_forward_when_facing_target(self):
        from src.agent.action import path_to_action
        # Agent at (10,10) facing along +col (heading=0 means facing +Z which maps to +row)
        # Target is the next cell in the path
        # heading=pi/2 means facing +col direction
        action = path_to_action(
            agent_ij=(10, 10),
            heading=math.pi / 2,  # facing +col
            next_ij=(10, 11),     # target is +col
            step_size_cells=1,
        )
        assert action == 1  # forward

    def test_turn_left_when_target_is_left(self):
        from src.agent.action import path_to_action
        action = path_to_action(
            agent_ij=(10, 10),
            heading=0.0,          # facing +row
            next_ij=(10, 9),      # target is -col (left of +row)
            step_size_cells=1,
        )
        assert action in (2, 3)  # should turn

    def test_turn_right_when_target_is_right(self):
        from src.agent.action import path_to_action
        action = path_to_action(
            agent_ij=(10, 10),
            heading=0.0,          # facing +row
            next_ij=(10, 11),     # target is +col (right of +row)
            step_size_cells=1,
        )
        assert action in (2, 3)  # should turn

    def test_stop_when_at_goal(self):
        from src.agent.action import path_to_action
        action = path_to_action(
            agent_ij=(10, 10),
            heading=0.0,
            next_ij=(10, 10),  # already there
            step_size_cells=1,
        )
        assert action == 0  # stop
