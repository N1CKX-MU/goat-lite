#!/usr/bin/env python3
"""Cut the demo video from real evaluation results (PLAN.md Section 7.12).

Section 7.12 asks for three successful episodes covering both modalities,
rendered at 30 fps as RGB plus a top-down map with instance nodes and the goal
overlaid, cut to 90 s with subtitles. This builds that from `results.csv`, so
every clip is a subtask that actually scored rather than a hand-picked
trajectory.

It does not re-implement the renderer. `scripts/debug_episode.py` already draws
the composed frame -- occupancy, plan, track, memory nodes, the ground-truth
goal and the live state machine -- and is the path that has been exercised. So
each clip is one `debug_episode.py` invocation, and this script selects, titles
and concatenates. Compositing is ffmpeg, as the section specifies.

Two things the data forces, both reported rather than hidden:

**Image goals cannot appear.** Image modality is 0/67 on the development
subset, so "covering both modalities" can only mean category and language. The
script says so instead of silently producing a two-modality cut that looks like
three.

**30 fps makes a success very short.** The successes available run 13-115
steps, which at 30 fps is 0.4-3.8 s each -- three of them is about five
seconds, not ninety. So each simulator step is held for several output frames,
computed from the clips actually selected (`frame_hold`). The video is a real
30 fps file; the agent simply moves at a watchable speed.

    # what it would cut, without rendering anything
    python scripts/make_video.py outputs/dev_eval_run3_memon --dry-run

    # the real thing (needs a free GPU, the scenes, and ffmpeg)
    python -u scripts/make_video.py outputs/dev_eval_run3_memon \
        --n-episodes 30 --out outputs/demo_video
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclasses.dataclass(frozen=True)
class Clip:
    episode_index: int          # index into the dev subset, NOT episode_id
    subtask_index: int
    modality: str
    category: str
    steps: int


def episode_indices(rows) -> list[int]:
    """Map each row to its position-based episode index.

    GOAT-Bench episode ids are per-scene: the 218 development subtasks share
    only 86 distinct (episode_id, subtask_index) pairs, and id "9" appears in
    five different scenes. `debug_episode.py --episode` is an index into the
    deterministic dev subset, so the two must be bridged by counting episode
    boundaries -- a row with subtask_index 0 -- exactly as the episode count
    and resume fixes do. Keying on episode_id would render a subtask from the
    wrong scene and look entirely plausible doing it.
    """
    out: list[int] = []
    idx = -1
    for r in rows:
        if int(r.subtask_index) == 0:
            idx += 1
        out.append(max(idx, 0))
    return out


def select_clips(rows, n_clips: int = 3, min_steps: int = 8,
                 max_steps: int = 200) -> list[Clip]:
    """Pick up to n_clips successful subtasks, spread across modalities.

    Successes are scarce (7 of 218 on the development subset), so the choice is
    nearly forced -- but not entirely: one of them called STOP at step 1, and a
    single frame demonstrates nothing, hence `min_steps`.

    `max_steps` excludes a success that consumed most of the budget before
    scoring. One run #3 success took all 500 steps, and as the first clip it
    would have filled 50 s of a 90 s cut with the agent wandering -- a true
    success, and the weakest possible illustration of the system working.

    Never pads with failures. A short video of real successes is honest; a
    video padded to length with subtasks that did not score is not.
    """
    idxs = episode_indices(rows)
    cands = [
        Clip(idxs[i], int(r.subtask_index), r.modality, r.category,
             int(r.steps_taken))
        for i, r in enumerate(rows)
        if bool(r.success) and min_steps <= int(r.steps_taken) <= max_steps
    ]
    if not cands:
        return []

    # Longest first within a modality: more steps means more of the approach
    # and verification visible, which is what the video is meant to show.
    by_mod: dict[str, list[Clip]] = {}
    for c in cands:
        by_mod.setdefault(c.modality, []).append(c)
    for v in by_mod.values():
        v.sort(key=lambda c: -c.steps)

    # Round-robin over modalities so a modality is never left out while another
    # contributes three clips.
    chosen: list[Clip] = []
    while len(chosen) < n_clips:
        took = False
        for mod in sorted(by_mod):
            if by_mod[mod] and len(chosen) < n_clips:
                chosen.append(by_mod[mod].pop(0))
                took = True
        if not took:
            break
    return chosen


def frame_hold(clips, target_seconds: float = 90.0, fps: int = 30) -> int:
    """How many output frames each simulator step should occupy.

    A 60-step success is two seconds at 30 fps. Filling Section 7.12's 90 s
    from clips totalling a few hundred steps therefore needs a hold, and the
    factor depends on what was selected, so it is computed.
    """
    total_steps = sum(c.steps for c in clips)
    if total_steps <= 0:
        return 1
    return max(1, int((target_seconds * fps) / total_steps))


CARD_SECONDS = 3.0


def _title_card(text: str, subtitle: str, path: str, size=(1280, 720)) -> None:
    import cv2
    import numpy as np

    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[:] = (24, 20, 18)
    cv2.putText(img, text, (60, size[1] // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (240, 240, 240), 2, cv2.LINE_AA)
    y = size[1] // 2 + 40
    for line in subtitle.split("\n"):
        cv2.putText(img, line, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (150, 190, 150), 1, cv2.LINE_AA)
        y += 36
    cv2.imwrite(path, img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="a dev-eval output dir with results.csv")
    ap.add_argument("--out", default="outputs/demo_video")
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--n-episodes", type=int, default=30,
                    help="dev-subset size the results were produced with; "
                         "must match or --episode indices point elsewhere")
    ap.add_argument("--target-seconds", type=float, default=90.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--min-steps", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=200,
                    help="exclude a success that burned most of the budget "
                         "before scoring; it fills the cut with wandering")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the cut without rendering")
    args = ap.parse_args()

    from src.eval.report import load_results

    rows = load_results(args.results_dir)
    if not rows:
        sys.exit(f"no results in {args.results_dir}")

    n_success = sum(1 for r in rows if r.success)
    clips = select_clips(rows, args.n_clips, args.min_steps, args.max_steps)
    # The title cards are part of the 90 s, so the agent footage gets the
    # remainder -- otherwise a three-clip cut overruns by nine seconds.
    footage_budget = max(5.0, args.target_seconds - len(clips) * CARD_SECONDS)
    hold = frame_hold(clips, footage_budget, args.fps)

    print(f"{len(rows)} subtasks, {n_success} successes")
    if not clips:
        sys.exit("no success long enough to film. Section 7.12 needs a run "
                 f"with successes of at least {args.min_steps} steps.")

    mods = sorted({c.modality for c in clips})
    print(f"selected {len(clips)} clip(s) over modalities {mods}:")
    for c in clips:
        print(f"  episode {c.episode_index:2d} subtask {c.subtask_index}  "
              f"{c.modality}/{c.category}  {c.steps} steps  "
              f"-> {c.steps * hold / args.fps:.1f}s")
    missing = {"category", "language", "image"} - set(mods)
    if missing:
        print(f"NOT represented: {sorted(missing)} -- no scored success exists "
              f"in {'it' if len(missing) == 1 else 'them'} on this run.")
    footage = sum(c.steps for c in clips) * hold / args.fps
    print(f"step hold {hold} frame(s); {footage:.1f}s of footage + "
          f"{len(clips) * CARD_SECONDS:.0f}s of title cards = "
          f"{footage + len(clips) * CARD_SECONDS:.1f}s "
          f"(target {args.target_seconds:.0f}s)")

    if args.dry_run:
        return
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found; it does the compositing")

    os.makedirs(args.out, exist_ok=True)
    pieces: list[str] = []

    for n, c in enumerate(clips):
        say = f"[{n+1}/{len(clips)}] {c.modality}/{c.category}"
        print(f"\n{say}: rendering episode {c.episode_index} "
              f"subtask {c.subtask_index}", flush=True)
        clip_dir = os.path.join(args.out, f"clip{n}")
        os.makedirs(clip_dir, exist_ok=True)
        cmd = [sys.executable, "-u", "scripts/debug_episode.py",
               "--episode", str(c.episode_index),
               "--subtask", str(c.subtask_index),
               "--n-episodes", str(args.n_episodes),
               "--steps", str(c.steps + 5),
               "--fps", str(max(1, args.fps // hold)),
               "--out", clip_dir]
        rc = subprocess.call(cmd)
        src = os.path.join(clip_dir,
                           f"ep{c.episode_index}_sub{c.subtask_index}.mp4")
        if rc != 0 or not os.path.exists(src):
            print(f"  skipped: debug_episode returned {rc}")
            continue

        card = os.path.join(clip_dir, "card.png")
        _title_card(
            f"{c.modality} goal: {c.category}",
            f"episode {c.episode_index}, subtask {c.subtask_index}\n"
            f"{c.steps} steps to a scored success\n"
            "left: occupancy, plan, memory nodes   right: what the agent sees",
            card,
        )
        card_mp4 = os.path.join(clip_dir, "card.mp4")
        subprocess.call(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1",
                         "-t", str(CARD_SECONDS), "-i", card, "-r", str(args.fps),
                         "-pix_fmt", "yuv420p", "-vf",
                         "scale=1280:720:force_original_aspect_ratio=decrease,"
                         "pad=1280:720:(ow-iw)/2:(oh-ih)/2", card_mp4])
        norm = os.path.join(clip_dir, "norm.mp4")
        # Re-encode every piece to one geometry and frame rate; the concat
        # demuxer joins streams without checking they match, and a mismatch
        # produces a file that plays only the first clip.
        subprocess.call(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                         "-r", str(args.fps), "-pix_fmt", "yuv420p", "-vf",
                         "scale=1280:720:force_original_aspect_ratio=decrease,"
                         "pad=1280:720:(ow-iw)/2:(oh-ih)/2", norm])
        if os.path.exists(card_mp4):
            pieces.append(card_mp4)
        if os.path.exists(norm):
            pieces.append(norm)

    if not pieces:
        sys.exit("nothing rendered")

    listing = os.path.join(args.out, "concat.txt")
    with open(listing, "w") as f:
        for p in pieces:
            f.write(f"file '{os.path.abspath(p)}'\n")
    final = os.path.join(args.out, "demo.mp4")
    subprocess.call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                     "-safe", "0", "-i", listing, "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", final])
    print(f"\nwrote {final}")
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", final],
        capture_output=True, text=True).stdout.strip()
    if dur:
        print(f"duration {float(dur):.1f}s (target {args.target_seconds:.0f}s)")


if __name__ == "__main__":
    main()
