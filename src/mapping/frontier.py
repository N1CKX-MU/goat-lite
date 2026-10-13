"""Frontier detection for exploration.

A frontier is a connected boundary between free and unknown space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Frontier:
    centroid_ij: tuple[int, int]  # (row, col) centroid of the frontier
    size: int                     # number of frontier cells
    cells: np.ndarray             # [N, 2] array of (row, col) coordinates


def find_frontiers(
    occupancy: np.ndarray,
    explored: np.ndarray,
    min_size: int = 20,
) -> list[Frontier]:
    """Find frontiers between free and unknown space.

    A frontier cell is a free cell that has at least one unknown neighbor.

    Args:
        occupancy: [H, W] int8, values in {-1=unknown, 0=free, 1=occupied}.
        explored: [H, W] bool.
        min_size: minimum number of cells for a frontier to be kept.

    Returns:
        List of Frontier objects, sorted by size descending.
    """
    free_mask = occupancy == 0

    # Dilate the unknown mask by one pixel to find free cells adjacent to unknown
    unknown_mask = occupancy == -1
    kernel = np.ones((3, 3), dtype=bool)
    unknown_neighbor = ndimage.binary_dilation(unknown_mask, structure=kernel)

    # Frontier = free AND adjacent to unknown
    frontier_mask = free_mask & unknown_neighbor

    if not np.any(frontier_mask):
        return []

    # Label connected components
    labeled, n_labels = ndimage.label(frontier_mask)

    # Group cells by label in ONE pass. The obvious loop --
    #     for label_id in range(1, n_labels + 1):
    #         cells = np.argwhere(labeled == label_id)
    # rescans the whole grid per label. On a 29 m scene the grid is 580x580 =
    # 336k cells and a partly explored map produces dozens of frontier
    # fragments, so that loop measured **233 ms per call** against 8.8 ms for a
    # full depth update. Exploration replans whenever the remaining path crosses
    # a newly occupied cell, which while mapping a fresh scene is most steps, so
    # this was the single largest cost in the eval loop.
    #
    # Sorting the labelled cells is equivalent, not merely similar: np.argwhere
    # returns indices in row-major order, and a STABLE sort by label preserves
    # that order within each group, so the per-frontier `cells` arrays come out
    # identical element for element. tests/test_frontier.py asserts that
    # against the original implementation on random grids.
    flat_labels = labeled.ravel()
    nz = np.flatnonzero(flat_labels)
    frontiers = []
    if nz.size:
        lab = flat_labels[nz]
        order = np.argsort(lab, kind="stable")
        nz, lab = nz[order], lab[order]
        _, starts, counts = np.unique(lab, return_index=True,
                                         return_counts=True)
        rows, cols = np.unravel_index(nz, labeled.shape)
        for start, count in zip(starts, counts):
            if count < min_size:
                continue
            sl = slice(start, start + count)
            cells = np.stack((rows[sl], cols[sl]), axis=1)
            centroid = cells.mean(axis=0).astype(int)
            frontiers.append(Frontier(
                centroid_ij=(int(centroid[0]), int(centroid[1])),
                size=int(count),
                cells=cells,
            ))

    # Sort by size descending
    frontiers.sort(key=lambda f: f.size, reverse=True)
    return frontiers
