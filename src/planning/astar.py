"""A* pathfinding on a 2D occupancy grid."""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.ndimage import binary_dilation


def plan_astar(
    occupancy: np.ndarray,
    start_ij: tuple[int, int],
    goal_ij: tuple[int, int],
    inflate_cells: int = 4,
    obstacle_penalty: float = 3.0,
    unknown_penalty: float = 2.0,
) -> list[tuple[int, int]] | None:
    """Plan a path from start to goal on the occupancy grid using A*.

    Args:
        occupancy: [H, W] int8, {-1=unknown, 0=free, 1=occupied}.
        start_ij: (row, col) start position.
        goal_ij: (row, col) goal position.
        inflate_cells: obstacles are dilated by this many cells and the result
            is IMPASSABLE, not merely expensive. At 0.05 m/cell the default 4
            cells is 0.20 m, just over the 0.17 m agent radius.
        obstacle_penalty: retained for API compatibility; the inflated band is
            now a hard constraint, so this only affects cells adjacent to it.
        unknown_penalty: extra cost for traversing unknown cells.

    Returns:
        List of (row, col) from start to goal, or None if no path exists.

    Two behaviours here were previously wrong and caused the agent to drive into
    walls (~41% of forward commands physically blocked):

    * ``blocked`` was ``occupied`` alone, so UNKNOWN cells were traversable at
      ordinary free-space cost. The agent planned confidently through
      unexplored space that was frequently solid.
    * The inflation band was applied only as ``cost += obstacle_penalty``, so A*
      routed through its own safety margin whenever a detour cost more than a
      few cells -- i.e. most of the time -- leaving no clearance for the agent's
      radius.

    Unknown is now expensive but still passable, deliberately: frontier goals
    sit on the boundary of unknown space, so blocking it outright would stall
    exploration.
    """
    h, w = occupancy.shape
    sr, sc = start_ij
    gr, gc = goal_ij

    # Bounds check
    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return None

    # Build cost grid
    occupied = occupancy == 1
    unknown = occupancy == -1

    # Inflate obstacles and treat the whole dilated region as impassable, so a
    # planned path always keeps the agent's radius clear of geometry.
    if inflate_cells > 0:
        struct = np.ones((2 * inflate_cells + 1, 2 * inflate_cells + 1), dtype=bool)
        inflated = binary_dilation(occupied, structure=struct)
    else:
        inflated = occupied
    near_obstacle = inflated & ~occupied

    blocked = inflated.copy()

    if occupied[sr, sc]:
        return None

    # The agent legitimately ends up inside the inflated band (hugging a wall,
    # or right after a collision). Leaving it blocked there would strand it with
    # no expandable neighbour, so carve the band -- but not real obstacles --
    # out of a small pocket around the start.
    if blocked[sr, sc]:
        r0, r1 = max(0, sr - inflate_cells), min(h, sr + inflate_cells + 1)
        c0, c1 = max(0, sc - inflate_cells), min(w, sc + inflate_cells + 1)
        blocked[r0:r1, c0:c1] &= occupied[r0:r1, c0:c1]

    if blocked[gr, gc]:
        if occupied[gr, gc]:
            return None
        # Goal has no clearance (an object against a wall, or a frontier
        # centroid). Relax to the un-inflated constraint rather than refusing.
        blocked = occupied.copy()

    # Start == goal
    if sr == gr and sc == gc:
        return [(sr, sc)]

    # 8-connected neighbors with costs
    DIRS = [
        (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
        (0, -1, 1.0),                          (0, 1, 1.0),
        (1, -1, math.sqrt(2)),  (1, 0, 1.0),   (1, 1, math.sqrt(2)),
    ]

    # Heuristic: Euclidean distance
    def heuristic(r, c):
        return math.sqrt((r - gr) ** 2 + (c - gc) ** 2)

    # Priority queue: (f_score, counter, row, col)
    counter = 0
    open_set = [(heuristic(sr, sc), counter, sr, sc)]
    g_score = np.full((h, w), np.inf, dtype=np.float64)
    g_score[sr, sc] = 0.0
    came_from = {}
    closed = np.zeros((h, w), dtype=bool)

    while open_set:
        f, _, cr, cc = heapq.heappop(open_set)

        if cr == gr and cc == gc:
            # Reconstruct path
            path = [(gr, gc)]
            r, c = gr, gc
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                path.append((r, c))
            path.reverse()
            return path

        if closed[cr, cc]:
            continue
        closed[cr, cc] = True

        for dr, dc, move_cost in DIRS:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if closed[nr, nc] or blocked[nr, nc]:
                continue

            # Extra cost near obstacles and through unexplored space. Unknown is
            # passable (frontier goals live there) but should never be preferred
            # over a known-free route of similar length.
            cost = move_cost
            if near_obstacle[nr, nc]:
                cost += obstacle_penalty
            if unknown[nr, nc]:
                cost += unknown_penalty

            new_g = g_score[cr, cc] + cost
            if new_g < g_score[nr, nc]:
                g_score[nr, nc] = new_g
                came_from[(nr, nc)] = (cr, cc)
                counter += 1
                heapq.heappush(open_set, (new_g + heuristic(nr, nc), counter, nr, nc))

    return None  # no path found
