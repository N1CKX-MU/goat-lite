"""Where the finetuner looks for the weights ultralytics just wrote.

finetune_v2 trained for 48 epochs and then died with "Training finished but
runs/goat_yolo/finetune_v2/weights/best.pt not found". The weights were at
runs/detect/runs/goat_yolo/finetune_v2/weights/best.pt: ultralytics resolves
`project` relative to the `runs_dir` in its settings.yaml, so composing
project/name/weights/best.pt only works when runs_dir happens to be ".".
The trainer knows where it wrote; ask it.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.finetune_yolo import resolve_best_weights


class _Trainer:
    def __init__(self, best=None, save_dir=None):
        if best is not None:
            self.best = best
        if save_dir is not None:
            self.save_dir = save_dir


class TestResolveBestWeights:
    def test_prefers_the_trainers_own_best_path(self, tmp_path):
        real = tmp_path / "runs" / "detect" / "goat" / "weights" / "best.pt"
        real.parent.mkdir(parents=True)
        real.write_bytes(b"w")
        trainer = _Trainer(best=str(real))

        assert resolve_best_weights(trainer, "runs/goat_yolo", "finetune") == str(real)

    def test_falls_back_to_save_dir_when_best_is_absent(self, tmp_path):
        save_dir = tmp_path / "runs" / "detect" / "goat"
        (save_dir / "weights").mkdir(parents=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"w")
        trainer = _Trainer(save_dir=str(save_dir))

        got = resolve_best_weights(trainer, "runs/goat_yolo", "finetune")
        assert got == str(save_dir / "weights" / "best.pt")

    def test_falls_back_to_the_composed_path_without_a_trainer(self, tmp_path):
        composed = tmp_path / "runs" / "goat_yolo" / "finetune" / "weights" / "best.pt"
        composed.parent.mkdir(parents=True)
        composed.write_bytes(b"w")

        got = resolve_best_weights(None, str(tmp_path / "runs" / "goat_yolo"), "finetune")
        assert got == str(composed)

    def test_ignores_a_trainer_path_that_does_not_exist(self, tmp_path):
        """A stale .best attribute must not shadow a file that is really there."""
        composed = tmp_path / "runs" / "goat_yolo" / "finetune" / "weights" / "best.pt"
        composed.parent.mkdir(parents=True)
        composed.write_bytes(b"w")
        trainer = _Trainer(best=str(tmp_path / "gone" / "best.pt"))

        got = resolve_best_weights(trainer, str(tmp_path / "runs" / "goat_yolo"), "finetune")
        assert got == str(composed)

    def test_raises_when_no_candidate_exists(self, tmp_path):
        trainer = _Trainer(best=str(tmp_path / "a" / "best.pt"),
                           save_dir=str(tmp_path / "b"))

        with pytest.raises(FileNotFoundError) as exc:
            resolve_best_weights(trainer, str(tmp_path / "c"), "finetune")

        # The message has to name every place looked, or the next person debugs blind.
        msg = str(exc.value)
        assert str(tmp_path / "a" / "best.pt") in msg
        assert str(tmp_path / "b" / "weights" / "best.pt") in msg
        assert str(tmp_path / "c" / "finetune" / "weights" / "best.pt") in msg

    def test_pathlike_attributes_are_accepted(self, tmp_path):
        """ultralytics stores save_dir as a pathlib.Path, not a str."""
        import pathlib

        save_dir = tmp_path / "runs" / "detect" / "goat"
        (save_dir / "weights").mkdir(parents=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"w")
        trainer = _Trainer(save_dir=pathlib.Path(save_dir))

        got = resolve_best_weights(trainer, "runs/goat_yolo", "finetune")
        assert got == str(save_dir / "weights" / "best.pt")
