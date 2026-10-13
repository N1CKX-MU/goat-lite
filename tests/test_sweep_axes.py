"""The three PLAN.md Section 7.10 ablations that had no way to be run.

Section 7.10 lists encoder choice, A* against Fast Marching, and three frontier
scoring rules among the Week 10 ablations. All three were unreachable: the
planner and the frontier rule were hard-coded in `GoatAgent`, and the encoder
was only reachable by editing `ClipEncoder`'s defaults. A sweep axis that cannot
be selected is an ablation that quietly does not happen -- the failure mode of
engineering log 6.46 and 6.49, where working code produced plausible numbers for
a feature that was never active.

These tests pin the selection itself, not the algorithms behind it (A*, FMM and
the frontier finder have their own tests). The emphasis is on the two ways a
selection can lie: silently falling back to the default, and accepting a name
nothing will ever load.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.planning.exploration import Frontier, choose_frontier, FRONTIER_STRATEGIES


def _f(centroid, size):
    return Frontier(cells=[centroid] * size, centroid_ij=centroid, size=size)


class TestFrontierStrategies:
    """size/(d+1) against size alone against random, over the same candidates."""

    # The two rules must disagree on these candidates or the test cannot tell
    # them apart, and the sizes have to be chosen for it: at size 6 against 400
    # the nearest-largest rule scores 6/2 = 3 against 400/99.3 = 4.03 and picks
    # the FAR one too. 60 against 400 separates them -- 30 against 4.03.
    NEAR_SMALL = (10, 10)
    FAR_HUGE = (80, 80)

    def _candidates(self):
        return [_f(self.NEAR_SMALL, 60), _f(self.FAR_HUGE, 400)]

    def test_nearest_largest_prefers_the_near_small_one(self):
        got = choose_frontier(self._candidates(), (10, 11), set(),
                              strategy="nearest-largest")
        assert got.centroid_ij == self.NEAR_SMALL

    def test_largest_only_prefers_the_far_huge_one(self):
        got = choose_frontier(self._candidates(), (10, 11), set(),
                              strategy="largest")
        assert got.centroid_ij == self.FAR_HUGE

    def test_default_is_unchanged_behaviour(self):
        """The default must stay the rule every prior measurement used."""
        a = choose_frontier(self._candidates(), (10, 11), set())
        b = choose_frontier(self._candidates(), (10, 11), set(),
                            strategy="nearest-largest")
        assert a.centroid_ij == b.centroid_ij == self.NEAR_SMALL

    def test_random_is_reproducible_for_a_given_seed(self):
        """An unseeded random arm is not an experiment -- it cannot be re-run."""
        cands = [_f((i, i), 10 + i) for i in range(20, 40)]
        first = choose_frontier(cands, (0, 0), set(), strategy="random",
                                rng=np.random.default_rng(42))
        again = choose_frontier(cands, (0, 0), set(), strategy="random",
                                rng=np.random.default_rng(42))
        assert first.centroid_ij == again.centroid_ij

    def test_random_actually_varies_across_seeds(self):
        cands = [_f((i, i), 10) for i in range(20, 60)]
        picks = {choose_frontier(cands, (0, 0), set(), strategy="random",
                                 rng=np.random.default_rng(s)).centroid_ij
                 for s in range(12)}
        assert len(picks) > 1, "a 'random' rule that always picks one frontier"

    def test_every_strategy_respects_the_visited_blacklist(self):
        """Whatever the rule, a retired frontier must not come back -- that is
        what stops the agent oscillating between two frontiers forever."""
        cands = self._candidates()
        for name in FRONTIER_STRATEGIES:
            got = choose_frontier(cands, (10, 11), {self.FAR_HUGE},
                                  strategy=name, rng=np.random.default_rng(0))
            assert got is not None
            assert got.centroid_ij != self.FAR_HUGE, name

    def test_every_strategy_returns_none_on_no_candidates(self):
        for name in FRONTIER_STRATEGIES:
            assert choose_frontier([], (0, 0), set(), strategy=name,
                                   rng=np.random.default_rng(0)) is None

    def test_an_unknown_strategy_is_refused(self):
        with pytest.raises(ValueError, match="frontier strategy"):
            choose_frontier(self._candidates(), (0, 0), set(),
                            strategy="whatever-looks-best")


class TestPlannerSelection:
    """GoatAgent must route through the planner it was asked for."""

    def test_default_planner_is_astar(self):
        from src.agent.goat_agent import GoatAgent
        import inspect
        d = inspect.signature(GoatAgent.__init__).parameters["planner"].default
        assert d == "astar"

    def test_selects_the_named_planner(self):
        from src.agent.goat_agent import resolve_planner
        from src.planning.astar import plan_astar
        from src.planning.fmm import plan_fmm
        assert resolve_planner("astar") is plan_astar
        assert resolve_planner("fmm") is plan_fmm

    def test_an_unknown_planner_is_refused(self):
        from src.agent.goat_agent import resolve_planner
        with pytest.raises(ValueError, match="planner"):
            resolve_planner("dijkstra-ish")

    def test_fmm_is_refused_when_skfmm_is_absent(self, monkeypatch):
        """plan_fmm returns None on a missing skfmm, which an agent cannot tell
        from 'no path exists'. Every subtask would end in no_plan and the arm
        would look like a planner result instead of a missing dependency."""
        import builtins
        from src.agent.goat_agent import resolve_planner
        real_import = builtins.__import__

        def no_skfmm(name, *a, **k):
            if name == "skfmm":
                raise ImportError("no skfmm")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_skfmm)
        with pytest.raises(RuntimeError, match="skfmm"):
            resolve_planner("fmm")


class TestEncoderSelection:
    """The encoder arm must fail at build time, not after loading a scene."""

    def test_known_model_passes_validation(self):
        from src.perception.encoder import validate_encoder
        validate_encoder("ViT-B-32", "laion2b_s34b_b79k")

    def test_unknown_pairing_is_refused_with_the_alternatives(self):
        """PLAN 7.10 names MobileCLIP-S0, which open_clip 3.3.0 does not carry
        (it has MobileCLIP-S1/S2/B and MobileCLIP2-S0). Naming a model that
        cannot load must say so here rather than hours into a sweep."""
        from src.perception.encoder import validate_encoder
        with pytest.raises(ValueError) as e:
            validate_encoder("MobileCLIP-S0", "datacompdr")
        msg = str(e.value)
        assert "MobileCLIP-S0" in msg
        assert "MobileCLIP" in msg  # suggests what does exist
