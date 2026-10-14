"""Frontier-based exploration policy."""

from __future__ import annotations

import math

import numpy as np

from src.mapping.frontier import Frontier


# The frontier scoring rules PLAN.md Section 7.10 asks to compare. Kept as a
# tuple so a sweep can iterate them and tests can assert over all of them.
FRONTIER_STRATEGIES = ("nearest-largest", "largest", "random")


def choose_frontier(
    frontiers: list[Frontier],
    agent_ij: tuple[int, int],
    visited: set,
    visit_radius: int = 10,
    strategy: str = "nearest-largest",
    rng=None,
) -> Frontier | None:
    """Choose the best frontier to explore.

    Three scoring rules, so Section 7.10's exploration ablation can be run:

    - ``nearest-largest`` (default): ``size / (distance + 1)``, the rule every
      measurement in the engineering log so far was taken under. Trades area
      against travel, which matters because 500 steps buys at most 125 m.
    - ``largest``: ``size`` alone. Commits to the biggest unknown boundary
      however far away it is, so it pays more travel for more new area.
    - ``random``: a uniform pick among the eligible frontiers, the control arm.
      Requires ``rng`` to be reproducible; an unseeded random arm cannot be
      re-run and so is not an experiment.

    Skip frontiers whose centroid is within visit_radius of any visited
    centroid, under every rule -- retiring a frontier is what stops the agent
    oscillating between two of them for the whole budget.

    Args:
        frontiers: list of Frontier objects.
        agent_ij: (row, col) of the agent.
        visited: set of (row, col) centroids already visited.
        visit_radius: skip frontiers within this many cells of a visited centroid.
        strategy: one of FRONTIER_STRATEGIES.
        rng: numpy Generator, used by the ``random`` strategy only.

    Returns:
        Best Frontier, or None if no valid options.
    """
    if strategy not in FRONTIER_STRATEGIES:
        raise ValueError(
            f"unknown frontier strategy {strategy!r}; "
            f"expected one of {list(FRONTIER_STRATEGIES)}"
        )
    if not frontiers:
        return None

    ar, ac = agent_ij
    best = None
    best_score = -1.0
    eligible: list[Frontier] = []

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

        if strategy == "random":
            eligible.append(f)
            continue

        dist = math.sqrt((fr - ar) ** 2 + (fc - ac) ** 2)
        score = f.size if strategy == "largest" else f.size / (dist + 1.0)

        if score > best_score:
            best_score = score
            best = f

    if strategy == "random":
        if not eligible:
            return None
        if rng is None:
            rng = np.random.default_rng()
        return eligible[int(rng.integers(len(eligible)))]

    return best
