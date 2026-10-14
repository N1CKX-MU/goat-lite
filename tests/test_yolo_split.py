"""The train/val split of the detector dataset must not strand a class.

Measured 2026-10-14 on the 150-frame build: `christmas tree` came out with 0
boxes in train and 8 in val. It exists in exactly one of the 145 annotated
scenes, the split is by scene at val_frac 0.1, and seed 42 put that scene in
val. Two consequences, both bad. The detector cannot learn the class at all,
and -- because mAP50 is averaged over classes -- it is then scored on 8 boxes
of a class it never saw, so a class the data cannot support drags the headline
metric down. `christmas tree` is 7 of the 218 dev subtasks.

So the split has to be class-aware: a scene may move to val only while every
class it carries still has a scene left in train. Training data wins ties,
because a class present in one scene cannot be in both splits and having it in
train is strictly better than having it only in val.
"""

from __future__ import annotations

import random

import pytest

from scripts.make_yolo_dataset import stratified_scene_split


def _split(classes_by_scene, val_frac=0.1, seed=42):
    return stratified_scene_split(classes_by_scene, val_frac,
                                  random.Random(seed))


class TestNoClassIsStranded:
    def test_a_class_in_one_scene_stays_in_train(self):
        """The christmas-tree case, reduced."""
        classes_by_scene = [{"picture"} for _ in range(20)]
        classes_by_scene[7] = {"picture", "christmas tree"}

        val = _split(classes_by_scene, val_frac=0.5)

        assert 7 not in val, "the only scene with the class went to val"

    def test_every_class_keeps_at_least_one_train_scene(self):
        rng = random.Random(0)
        names = [f"c{i}" for i in range(12)]
        classes_by_scene = [
            set(rng.sample(names, rng.randint(1, 4))) for _ in range(60)
        ]
        # make three classes singletons, the hard case
        classes_by_scene[3].add("rare_a")
        classes_by_scene[40].add("rare_b")
        classes_by_scene[59].add("rare_c")

        val = _split(classes_by_scene, val_frac=0.2)
        train_classes = set().union(
            *[c for i, c in enumerate(classes_by_scene) if i not in val])

        every = set().union(*classes_by_scene)
        assert every <= train_classes, f"stranded: {every - train_classes}"

    def test_classes_in_two_scenes_may_be_split(self):
        """Not every rare class has to stay wholly in train -- only the ones
        that would otherwise leave train empty."""
        classes_by_scene = [{"common"} for _ in range(20)]
        classes_by_scene[2].add("pair")
        classes_by_scene[3].add("pair")

        # With a large val fraction one of the two is allowed to move.
        val = _split(classes_by_scene, val_frac=0.5)
        assert not {2, 3} <= set(val), "both scenes with the class went to val"


class TestSplitShape:
    def test_val_is_about_the_requested_fraction(self):
        classes_by_scene = [{"a", "b"} for _ in range(100)]
        val = _split(classes_by_scene, val_frac=0.1)
        assert len(val) == 10

    def test_at_least_one_val_scene(self):
        classes_by_scene = [{"a"} for _ in range(5)]
        assert len(_split(classes_by_scene, val_frac=0.01)) >= 1

    def test_is_deterministic_for_a_seed(self):
        classes_by_scene = [{f"c{i % 7}"} for i in range(50)]
        assert _split(classes_by_scene, seed=1) == _split(
            classes_by_scene, seed=1)

    def test_different_seeds_can_differ(self):
        classes_by_scene = [{f"c{i % 7}"} for i in range(50)]
        a = _split(classes_by_scene, seed=1)
        b = _split(classes_by_scene, seed=99)
        assert a != b or len(a) <= 1

    def test_val_indices_are_valid_scene_indices(self):
        classes_by_scene = [{"a"} for _ in range(30)]
        val = _split(classes_by_scene, val_frac=0.2)
        assert all(0 <= i < 30 for i in val)
        assert len(set(val)) == len(val)

    def test_a_scene_with_no_classes_is_free_to_move(self):
        classes_by_scene = [{"a"} for _ in range(10)] + [set(), set()]
        val = _split(classes_by_scene, val_frac=0.2)
        assert len(val) == 2


class TestDegenerateInputs:
    def test_single_scene_corpus_keeps_it_in_train(self):
        """One scene cannot be both. Train wins, and val comes out empty --
        ultralytics tolerates that; a detector with no training data does not."""
        assert _split([{"a"}], val_frac=0.5) == []

    def test_no_scenes(self):
        assert _split([], val_frac=0.1) == []

    def test_cannot_reach_the_fraction_without_stranding(self):
        """Every scene carries a unique class, so none may move. The function
        returns fewer val scenes rather than stranding anything."""
        classes_by_scene = [{f"only_in_{i}"} for i in range(10)]
        assert _split(classes_by_scene, val_frac=0.5) == []
