"""One definition of "the Nth episode", shared by the runner and the replayer.

The evaluation groups episodes by scene, because one habitat load serves all
of a scene's episodes and a load costs 5-10 s. results.csv is therefore in
scene-grouped order. `debug_episode.py` indexed the *dev subset* order
instead, so `--episode 1` replayed a different episode than row-block 1 of the
results it was replaying: eval-order 1 is dev-subset index 27, eval-order 4 is
index 21. The demo video silently filmed unrelated episodes -- a clip chosen
as a 51-step image/plant success rendered as a category/carpet failure (6.65).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.episode_order import eval_episode_order, group_by_scene


class _Ep:
    def __init__(self, scene_id, episode_id):
        self.scene_id = scene_id
        self.episode_id = episode_id

    def __repr__(self):
        return f"{self.scene_id}:{self.episode_id}"


class TestGroupByScene:
    def test_episodes_of_one_scene_become_contiguous(self):
        eps = [_Ep("a", 0), _Ep("b", 0), _Ep("a", 1), _Ep("b", 1)]

        order = eval_episode_order(eps)

        assert [e.scene_id for e in order] == ["a", "a", "b", "b"]

    def test_scene_order_is_first_appearance(self):
        """dict preserves insertion order; the runner relies on it, so pin it."""
        eps = [_Ep("z", 0), _Ep("a", 0), _Ep("z", 1)]

        assert [e.scene_id for e in eval_episode_order(eps)] == ["z", "z", "a"]

    def test_within_a_scene_the_original_order_is_kept(self):
        eps = [_Ep("a", 5), _Ep("a", 2), _Ep("a", 9)]

        assert [e.episode_id for e in eval_episode_order(eps)] == [5, 2, 9]

    def test_it_is_a_permutation_losing_nothing(self):
        eps = [_Ep("a", 0), _Ep("b", 0), _Ep("a", 1), _Ep("c", 0), _Ep("b", 1)]

        order = eval_episode_order(eps)

        assert len(order) == len(eps)
        assert {id(e) for e in order} == {id(e) for e in eps}

    def test_an_already_grouped_list_is_unchanged(self):
        eps = [_Ep("a", 0), _Ep("a", 1), _Ep("b", 0)]

        assert [id(e) for e in eval_episode_order(eps)] == [id(e) for e in eps]

    def test_empty(self):
        assert eval_episode_order([]) == []

    def test_group_by_scene_returns_the_runners_mapping(self):
        eps = [_Ep("a", 0), _Ep("b", 0), _Ep("a", 1)]

        g = group_by_scene(eps)

        assert list(g) == ["a", "b"]
        assert [e.episode_id for e in g["a"]] == [0, 1]


class TestTheRunnerAndReplayerAgree:
    def test_run_dev_eval_groups_with_the_shared_helper(self):
        """The bug was two independent groupings. Pin that there is one."""
        import inspect

        import scripts.run_dev_eval as rde

        src = inspect.getsource(rde.main)
        assert "group_by_scene" in src, (
            "run_dev_eval must use the shared grouping, or debug_episode's "
            "--episode index can drift from results.csv again"
        )

    def test_debug_episode_orders_with_the_shared_helper(self):
        import inspect

        import scripts.debug_episode as de

        assert "eval_episode_order" in inspect.getsource(de.main)


class TestReplaySetsGoalsLikeTheRunner:
    """An image goal carries an embedding the runner computes by rendering the
    goal viewpoint. debug_episode never computed it, so every image subtask
    replayed with goal_embedding None -- the matcher had nothing to match and
    the agent wandered. The 51-step image/plant success replayed as a failure
    that was closest to the goal at step 0 (6.65)."""

    def test_the_replayer_embeds_image_goals(self):
        import inspect

        import scripts.debug_episode as de

        src = inspect.getsource(de)
        assert "_embed_image_goal" in src, (
            "debug_episode must embed image goals exactly as runner.py does, "
            "or image-modality replays diverge from the run they replay"
        )

    def test_it_embeds_them_during_warmup_too(self):
        """Warmup rebuilds the memory the filmed subtask inherits, so an image
        subtask replayed without its embedding corrupts that state as well."""
        import inspect

        import scripts.debug_episode as de

        src = inspect.getsource(de.main)
        # two call sites: the warmup loop and the traced subtask
        assert src.count("_goal_spec_for(") >= 2
