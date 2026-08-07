"""Low-level action selection — convert a grid path to movement actions."""

from __future__ import annotations

import math


def path_to_action(
    agent_ij: tuple[int, int],
    heading: float,
    next_ij: tuple[int, int],
    step_size_cells: int = 1,
    turn_angle: float = math.radians(30.0),
    angle_tolerance: float = math.radians(20.0),
) -> int:
    """Decide the next action to move toward the next waypoint.

    Args:
        agent_ij: (row, col) current agent position on the grid.
        heading: agent heading in radians. 0 = facing +row, pi/2 = facing +col.
        next_ij: (row, col) next waypoint on the path.
        step_size_cells: how many cells one forward step covers.
        turn_angle: how much one turn action rotates (radians).
        angle_tolerance: if angular error is within this, go forward.

    Returns:
        Action int: 0=stop, 1=forward, 2=turn_left, 3=turn_right.
    """
    ar, ac = agent_ij
    nr, nc = next_ij

    dr = nr - ar
    dc = nc - ac

    # Already at the waypoint
    if dr == 0 and dc == 0:
        return 0  # stop

    # Desired heading: angle from agent to target
    # Convention: 0 = +row direction, pi/2 = +col direction
    desired = math.atan2(dc, dr)

    # Angular difference (signed, wrapped to [-pi, pi])
    diff = desired - heading
    diff = (diff + math.pi) % (2 * math.pi) - math.pi

    if abs(diff) <= angle_tolerance:
        return 1  # forward

    if diff > 0:
        return 3  # turn right (increase heading)
    else:
        return 2  # turn left (decrease heading)
