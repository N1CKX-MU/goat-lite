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
) -> list[tuple[int, int]] | None:
    """Plan a path from start to goal on the occupancy grid using A*.

    Args:
        occupancy: [H, W] int8, {-1=unknown, 0=free, 1=occupied}.
        start_ij: (row, col) start position.
        goal_ij: (row, col) goal position.
        inflate_cells: inflate obstacles by this many cells for safety margin.
        obstacle_penalty: extra cost for cells near obstacles.

    Returns:
        List of (row, col) from start to goal, or None if no path exists.
    """
    h, w = occupancy.shape
    sr, sc = start_ij
    gr, gc = goal_ij

    # Bounds check
    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return None

    # Build cost grid
    occupied = occupancy == 1
    if occupied[gr, gc]:
        return None

    # Inflate obstacles
    if inflate_cells > 0:
        struct = np.ones((2 * inflate_cells + 1, 2 * inflate_cells + 1), dtype=bool)
        near_obstacle = binary_dilation(occupied, structure=struct) & ~occupied
    else:
        near_obstacle = np.zeros_like(occupied)

    # Blocked = occupied or unknown treated as blocked for planning
    blocked = occupied

    if blocked[sr, sc]:
        return None

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

            # Extra cost near obstacles
            cost = move_cost
            if near_obstacle[nr, nc]:
                cost += obstacle_penalty

            new_g = g_score[cr, cc] + cost
            if new_g < g_score[nr, nc]:
                g_score[nr, nc] = new_g
                came_from[(nr, nc)] = (cr, cc)
                counter += 1
                heapq.heappush(open_set, (new_g + heuristic(nr, nc), counter, nr, nc))

    return None  # no path found
