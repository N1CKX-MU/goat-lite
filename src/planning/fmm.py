"""Fast Marching Method planner — smoother paths than A*."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def plan_fmm(
    occupancy: np.ndarray,
    start_ij: tuple[int, int],
    goal_ij: tuple[int, int],
    inflate_cells: int = 4,
) -> list[tuple[int, int]] | None:
    """Plan a path using Fast Marching Method.

    Args:
        occupancy: [H, W] int8, {-1=unknown, 0=free, 1=occupied}.
        start_ij: (row, col) start.
        goal_ij: (row, col) goal.
        inflate_cells: inflate obstacles by this many cells.

    Returns:
        List of (row, col) from start to goal, or None if no path.
    """
    try:
        import skfmm
    except ImportError:
        return None  # fall back to A* if skfmm not installed

    h, w = occupancy.shape
    sr, sc = start_ij
    gr, gc = goal_ij

    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return None

    occupied = occupancy == 1
    if occupied[gr, gc]:
        return None

    # Inflate
    if inflate_cells > 0:
        struct = np.ones((2 * inflate_cells + 1, 2 * inflate_cells + 1), dtype=bool)
        blocked = binary_dilation(occupied, structure=struct)
    else:
        blocked = occupied.copy()

    # Free mask (traversable)
    free = ~blocked
    if not free[sr, sc] or not free[gr, gc]:
        return None

    # Compute travel time from goal
    phi = np.ones((h, w), dtype=np.float64)
    phi[gr, gc] = 0.0
    speed = np.where(free, 1.0, 0.0)

    try:
        travel_time = skfmm.travel_time(phi, speed=speed, dx=1.0)
    except Exception:
        return None

    if np.isinf(travel_time[sr, sc]):
        return None

    # Gradient descent from start to goal
    path = [(sr, sc)]
    cr, cc = sr, sc
    max_steps = h * w
    for _ in range(max_steps):
        if cr == gr and cc == gc:
            break
        # Check 8 neighbors, move to one with lowest travel time
        best_r, best_c = cr, cc
        best_t = travel_time[cr, cc]
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and travel_time[nr, nc] < best_t:
                    best_t = travel_time[nr, nc]
                    best_r, best_c = nr, nc
        if best_r == cr and best_c == cc:
            break  # stuck
        cr, cc = best_r, best_c
        path.append((cr, cc))

    if abs(cr - gr) > 1 or abs(cc - gc) > 1:
        return None  # didn't reach goal

    return path
