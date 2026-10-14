"""Clip selection for the Week 12 demo video (PLAN.md Section 7.12).

The rendering and compositing need a GPU, a scene and ffmpeg, so what is
testable without them is the part most likely to be quietly wrong: which
subtasks get picked, and how a row in results.csv maps back to a
`debug_episode.py --episode` index.

That mapping is the one to be careful about. `--episode` is an index into the
deterministic dev subset, while results.csv carries GOAT-Bench's `episode_id`,
which is per-scene and collides -- 218 subtasks share 86 distinct
(episode_id, subtask_index) pairs. That collision has already produced three
defects (6.46, and the episode count and ablation pairing fixes). Selecting a
clip by episode_id would make this the fourth: it would render a subtask from
the wrong scene and look plausible doing it.
"""

from __future__ import annotations

import pytest

from scripts.make_video import Clip, episode_indices, select_clips


class Row:
    """Minimal stand-in for SubtaskResult."""

    def __init__(self, episode_id, subtask_index, modality, category,
                 success, steps_taken=50):
        self.episode_id = episode_id
        self.subtask_index = subtask_index
        self.modality = modality
        self.category = category
        self.success = success
        self.steps_taken = steps_taken
        self.distance_to_goal = 0.5


def _episode(ep_id, n_sub, success_at=(), modality="category", steps=50):
    return [Row(ep_id, i, modality, "mirror", i in success_at, steps)
            for i in range(n_sub)]


class TestEpisodeIndices:
    def test_counts_episodes_by_their_subtask_zero(self):
        rows = _episode("0", 3) + _episode("1", 2) + _episode("0", 2)
        # Third episode reuses id "0" -- a different scene. Positionally it is
        # episode 2, and anything keyed on the id would fold it into episode 0.
        assert episode_indices(rows) == [0, 0, 0, 1, 1, 2, 2]

    def test_a_repeated_id_does_not_collapse(self):
        rows = _episode("9", 2) + _episode("9", 2) + _episode("9", 2)
        assert episode_indices(rows) == [0, 0, 1, 1, 2, 2]
        assert max(episode_indices(rows)) == 2, "three scenes, not one"

    def test_handles_a_leading_row_that_is_not_subtask_zero(self):
        """A resumed or truncated CSV can start mid-episode; it must not
        produce a negative index."""
        rows = [Row("4", 3, "category", "rug", False)] + _episode("5", 2)
        idx = episode_indices(rows)
        assert min(idx) >= 0


class TestSelectClips:
    def test_picks_only_successes(self):
        rows = _episode("0", 4, success_at=(2,))
        clips = select_clips(rows, n_clips=3)
        assert [c.subtask_index for c in clips] == [2]

    def test_spreads_across_modalities_before_doubling_up(self):
        rows = (_episode("0", 3, success_at=(0, 1, 2), modality="category")
                + _episode("1", 2, success_at=(0, 1), modality="language"))
        clips = select_clips(rows, n_clips=3)
        mods = [c.modality for c in clips]
        assert "language" in mods, "a whole modality was left out"
        assert mods.count("category") <= 2

    def test_skips_clips_too_short_to_watch(self):
        """One run #3 success called STOP at step 1. A single frame is not a
        demonstration of anything."""
        rows = (_episode("0", 2, success_at=(0,), steps=1)
                + _episode("1", 2, success_at=(0,), steps=60))
        clips = select_clips(rows, n_clips=3, min_steps=8)
        assert [c.steps for c in clips] == [60]

    def test_returns_nothing_rather_than_padding_with_failures(self):
        rows = _episode("0", 5)
        assert select_clips(rows, n_clips=3) == []

    def test_maps_each_clip_to_its_subset_index(self):
        rows = (_episode("7", 2) + _episode("7", 3, success_at=(1,)))
        clips = select_clips(rows, n_clips=1)
        assert clips[0].episode_index == 1, "must be positional, not by id"
        assert clips[0].subtask_index == 1

    def test_respects_n_clips(self):
        rows = _episode("0", 6, success_at=(0, 1, 2, 3, 4))
        assert len(select_clips(rows, n_clips=2, min_steps=1)) == 2


class TestFrameHold:
    """At 30 fps a 60-step subtask is two seconds. Section 7.12 asks for a 90 s
    cut, and the successes available are 13-115 steps, so the step-to-frame
    hold has to be computed rather than assumed."""

    def test_hold_fills_the_target_duration(self):
        from scripts.make_video import frame_hold
        clips = [Clip(0, 0, "category", "mirror", 100),
                 Clip(1, 0, "language", "picture", 100)]
        # 200 steps into 90 s at 30 fps = 2700 frames = 13.5 frames per step.
        assert frame_hold(clips, target_seconds=90, fps=30) == 13

    def test_hold_is_never_below_one(self):
        from scripts.make_video import frame_hold
        clips = [Clip(0, 0, "category", "mirror", 10_000)]
        assert frame_hold(clips, target_seconds=5, fps=30) == 1

    def test_no_clips_does_not_divide_by_zero(self):
        from scripts.make_video import frame_hold
        assert frame_hold([], target_seconds=90, fps=30) == 1


class TestExcludesBudgetBurners:
    """A success that took the whole budget is a success and a bad clip."""

    def test_skips_a_success_that_consumed_the_budget(self):
        rows = (_episode("0", 2, success_at=(0,), steps=500)
                + _episode("1", 2, success_at=(0,), steps=90))
        clips = select_clips(rows, n_clips=3, max_steps=200)
        assert [c.steps for c in clips] == [90], \
            "a 500-step success would fill 50s of a 90s cut with wandering"

    def test_the_band_is_inclusive_at_both_ends(self):
        rows = (_episode("0", 1, success_at=(0,), steps=8)
                + _episode("1", 1, success_at=(0,), steps=200))
        got = sorted(c.steps for c in select_clips(rows, n_clips=4,
                                                   min_steps=8, max_steps=200))
        assert got == [8, 200]


class TestRenderMatchesTheRunItFilms:
    """A clip is chosen from a results.csv and then re-rendered by replaying
    the episode. If the replay runs a different configuration from the one
    that produced the row, it films a different outcome -- the 512 px run's
    successes replay at 256 px as failures, so the demo video would show the
    agent missing goals it actually reached."""

    def test_the_render_command_carries_the_resolution(self):
        from scripts.make_video import render_command
        from scripts.make_video import Clip

        c = Clip(episode_index=4, subtask_index=4, modality="category",
                 category="plant", steps=20)
        cmd = render_command(c, n_episodes=30, fps=30, hold=17,
                             out_dir="/tmp/x", resolution=512)

        assert "--resolution" in cmd
        assert cmd[cmd.index("--resolution") + 1] == "512"

    def test_the_default_resolution_is_the_sensor_default(self):
        from scripts.make_video import render_command, Clip

        c = Clip(episode_index=0, subtask_index=0, modality="image",
                 category="mirror", steps=10)
        cmd = render_command(c, n_episodes=30, fps=30, hold=1, out_dir="/tmp/x")

        assert cmd[cmd.index("--resolution") + 1] == "256"

    def test_it_renders_enough_steps_to_reach_the_success(self):
        """Filming exactly `steps` can stop one action short of the stop."""
        from scripts.make_video import render_command, Clip

        c = Clip(episode_index=4, subtask_index=4, modality="category",
                 category="plant", steps=20)
        cmd = render_command(c, n_episodes=30, fps=30, hold=1, out_dir="/tmp/x")

        assert int(cmd[cmd.index("--steps") + 1]) > c.steps


class TestWarmupMatchesTheEvalBudget:
    """debug_episode replays an episode's earlier subtasks to rebuild the map
    and memory the filmed subtask inherits. It capped each replay at 300 steps
    while the evaluation gives every subtask 500, so any episode with a long
    earlier subtask was replayed from a state the real run never had.
    Measured: episode index 4's warmup subtasks took 338 and 500 steps; the
    filmed subtask scored success in the run and failure on replay (6.65)."""

    def test_the_render_command_sets_the_eval_step_budget(self):
        from scripts.make_video import render_command, Clip, EVAL_MAX_STEPS

        c = Clip(episode_index=4, subtask_index=4, modality="category",
                 category="plant", steps=20)
        cmd = render_command(c, n_episodes=30, fps=30, hold=1, out_dir="/tmp/x")

        assert "--warmup-steps" in cmd
        assert int(cmd[cmd.index("--warmup-steps") + 1]) == EVAL_MAX_STEPS

    def test_the_budget_is_the_one_the_runner_actually_uses(self):
        """Pinned against the runner rather than copied, so raising one side
        cannot silently desynchronise the replay again."""
        import inspect
        from scripts.make_video import EVAL_MAX_STEPS
        from src.eval.runner import run_episode

        default = inspect.signature(run_episode).parameters[
            "max_steps_per_subtask"].default
        assert EVAL_MAX_STEPS == default

    def test_a_first_subtask_needs_no_warmup(self):
        from scripts.make_video import render_command, Clip

        c = Clip(episode_index=0, subtask_index=0, modality="image",
                 category="mirror", steps=10)
        cmd = render_command(c, n_episodes=30, fps=30, hold=1, out_dir="/tmp/x")

        assert "--warmup-steps" in cmd


class TestOnlyVerifiedSuccessesAreCut:
    """A clip is chosen because results.csv says that subtask succeeded, then
    re-rendered by replaying it. Three separate defects made the replay
    diverge from the run (6.65), and the cut included the footage regardless,
    so a video advertised as a working episode could show the agent failing.
    The replay prints its own verdict; use it."""

    def test_a_scored_success_is_accepted(self):
        from scripts.make_video import replay_succeeded

        out = ("tracing subtask 2: image/plant, 1 valid instance(s)\n"
               "CLOSEST approach: 0.89 m at step 50, goal_in_view=True\n"
               "SCORED: SUCCESS\n")
        assert replay_succeeded(out) is True

    def test_a_scored_failure_is_rejected(self):
        from scripts.make_video import replay_succeeded

        out = ("CLOSEST approach: 4.54 m at step 7, goal_in_view=False\n"
               "SCORED: failure\n")
        assert replay_succeeded(out) is False

    def test_output_without_a_verdict_is_rejected(self):
        """A crash before scoring must not read as success."""
        from scripts.make_video import replay_succeeded

        assert replay_succeeded("Traceback (most recent call last):\n") is False
        assert replay_succeeded("") is False

    def test_the_word_success_elsewhere_does_not_count(self):
        from scripts.make_video import replay_succeeded

        out = ("this episode had 3 successes in the run\n"
               "SCORED: failure\n")
        assert replay_succeeded(out) is False
