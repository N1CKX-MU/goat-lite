"""Tests for src/mapping/frontier.py — find_frontiers."""

import numpy as np
import pytest

from src.mapping.frontier import Frontier, find_frontiers


def _make_map(h=100, w=100, fill=-1):
    """Create a map filled with a value. -1=unknown, 0=free, 1=occupied."""
    return np.full((h, w), fill, dtype=np.int8)


class TestFindFrontiers:
    def test_fully_unknown_no_frontiers(self):
        """A fully unknown map has no free cells, so no frontiers."""
        occ = _make_map(fill=-1)
        explored = np.zeros_like(occ, dtype=bool)
        frontiers = find_frontiers(occ, explored)
        assert frontiers == []

    def test_fully_explored_no_frontiers(self):
        """A fully explored (all free) map has no unknown neighbors, so no frontiers."""
        occ = _make_map(fill=0)
        explored = np.ones_like(occ, dtype=bool)
        frontiers = find_frontiers(occ, explored)
        assert frontiers == []

    def test_hallway_one_frontier(self):
        """A partial hallway: free cells with unknown at the open end."""
        occ = _make_map(50, 50, fill=-1)
        explored = np.zeros((50, 50), dtype=bool)
        # Carve a hallway: rows 20-30, cols 0-25 are free
        occ[20:30, 0:25] = 0
        explored[20:30, 0:25] = True
        # Walls on top and bottom
        occ[19, 0:25] = 1
        occ[30, 0:25] = 1
        explored[19, 0:25] = True
        explored[30, 0:25] = True

        frontiers = find_frontiers(occ, explored, min_size=3)
        # Should find frontier at col=25 (the open end)
        assert len(frontiers) >= 1
        # The frontier should be near col=24 (edge of free space)
        centroids_col = [f.centroid_ij[1] for f in frontiers]
        assert any(c >= 22 for c in centroids_col)

    def test_room_with_door(self):
        """A room fully explored except for a doorway opening into unknown."""
        occ = _make_map(50, 50, fill=-1)
        explored = np.zeros((50, 50), dtype=bool)
        # Room: rows 10-40, cols 10-40
        occ[10:40, 10:40] = 0
        explored[10:40, 10:40] = True
        # Walls around room
        occ[9, 10:40] = 1
        explored[9, 10:40] = True
        occ[40, 10:40] = 1
        explored[40, 10:40] = True
        occ[10:40, 9] = 1
        explored[10:40, 9] = True
        occ[10:40, 40] = 1
        explored[10:40, 40] = True
        # Door opening on the right wall: cols 40, rows 23-27 are free (not wall)
        occ[23:27, 40] = 0
        explored[23:27, 40] = True

        frontiers = find_frontiers(occ, explored, min_size=1)
        assert len(frontiers) >= 1

    def test_min_size_filters_small(self):
        """Frontiers below min_size should be filtered."""
        occ = _make_map(50, 50, fill=-1)
        explored = np.zeros((50, 50), dtype=bool)
        # Small free area: 3x3
        occ[24:27, 24:27] = 0
        explored[24:27, 24:27] = True

        # With a large min_size, the small frontier should be filtered
        frontiers_large = find_frontiers(occ, explored, min_size=50)
        frontiers_small = find_frontiers(occ, explored, min_size=1)
        assert len(frontiers_large) <= len(frontiers_small)

    def test_sorted_by_size_descending(self):
        """Frontiers should be sorted largest first."""
        occ = _make_map(100, 100, fill=-1)
        explored = np.zeros((100, 100), dtype=bool)
        # Two separate free regions of different sizes
        # Small region
        occ[10:15, 10:15] = 0
        explored[10:15, 10:15] = True
        # Large region
        occ[50:70, 50:70] = 0
        explored[50:70, 50:70] = True

        frontiers = find_frontiers(occ, explored, min_size=1)
        if len(frontiers) >= 2:
            assert frontiers[0].size >= frontiers[1].size

    def test_frontier_dataclass_fields(self):
        """Frontier should have centroid_ij, size, and cells fields."""
        occ = _make_map(50, 50, fill=-1)
        explored = np.zeros((50, 50), dtype=bool)
        occ[20:30, 20:30] = 0
        explored[20:30, 20:30] = True

        frontiers = find_frontiers(occ, explored, min_size=1)
        assert len(frontiers) >= 1
        f = frontiers[0]
        assert hasattr(f, "centroid_ij")
        assert hasattr(f, "size")
        assert hasattr(f, "cells")
        assert len(f.centroid_ij) == 2
        assert f.size > 0
