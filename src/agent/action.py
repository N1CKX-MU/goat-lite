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
        heading: agent compass heading in radians (Habitat convention). At
            heading 0 the agent's forward is -Z (grid -row); heading increases
            CW viewed from above. So forward(h) = (-cos h, -sin h) in (row, col).
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

    # Desired heading such that forward(h) = (-cos h, -sin h) points at the
    # target direction (dr, dc). Solving gives h = atan2(-dc, -dr).
    desired = math.atan2(-dc, -dr)

    # Angular difference (signed, wrapped to [-pi, pi])
    diff = desired - heading
    diff = (diff + math.pi) % (2 * math.pi) - math.pi

    if abs(diff) <= angle_tolerance:
        return 1  # forward

    # turn_left (2) increases compass heading; turn_right (3) decreases it.
    if diff > 0:
        return 2  # turn left (increase heading)
    else:
        return 3  # turn right (decrease heading)
