"""Frontier-based exploration policy."""

from __future__ import annotations

import math

import numpy as np

from src.mapping.frontier import Frontier


def choose_frontier(
    frontiers: list[Frontier],
    agent_ij: tuple[int, int],
    visited: set,
    visit_radius: int = 10,
) -> Frontier | None:
    """Choose the best frontier to explore.

    Score = size / (distance + 1) — greedy nearest-large.
    Skip frontiers whose centroid is within visit_radius of any visited centroid.

    Args:
        frontiers: list of Frontier objects.
        agent_ij: (row, col) of the agent.
        visited: set of (row, col) centroids already visited.
        visit_radius: skip frontiers within this many cells of a visited centroid.

    Returns:
        Best Frontier, or None if no valid options.
    """
    if not frontiers:
        return None

    ar, ac = agent_ij
    best = None
    best_score = -1.0

    for f in frontiers:
        fr, fc = f.centroid_ij

        # Skip if near a visited frontier
        skip = False
        for vr, vc in visited:
            if math.sqrt((fr - vr) ** 2 + (fc - vc) ** 2) < visit_radius:
                skip = True
                break
        if skip:
            continue

        dist = math.sqrt((fr - ar) ** 2 + (fc - ac) ** 2)
        score = f.size / (dist + 1.0)

        if score > best_score:
            best_score = score
            best = f

    return best
