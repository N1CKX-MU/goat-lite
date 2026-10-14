"""The order the evaluation visits episodes in — one definition.

The runner groups episodes by scene because a habitat scene load costs 5-10 s
and one load serves every episode in that scene. That grouping is what
``results.csv`` is ordered by, so anything mapping a results row back to an
episode has to use the same order.

It was previously open-coded in ``run_dev_eval`` and *not* used by
``debug_episode``, which indexed the raw dev subset. The two disagree
completely — eval-order episode 1 is dev-subset index 27 — so the demo video
replayed unrelated episodes and filmed failures where the results said
success (6.65). Hence this module: the grouping has exactly one definition and
both callers import it.
"""

from __future__ import annotations


def group_by_scene(episodes: list) -> dict:
    """Episodes grouped by ``scene_id``, scenes in first-appearance order.

    Insertion order is load-bearing, not incidental: it fixes the order the
    runner loads scenes in and therefore the order of ``results.csv``.
    """
    by_scene: dict = {}
    for ep in episodes:
        by_scene.setdefault(ep.scene_id, []).append(ep)
    return by_scene


def eval_episode_order(episodes: list) -> list:
    """The episodes flattened in the order the evaluation runs them.

    A permutation of the input: index ``n`` is the episode whose rows form the
    ``n``-th episode block of ``results.csv``.
    """
    return [ep for scene_eps in group_by_scene(episodes).values()
            for ep in scene_eps]
