#!/usr/bin/env python3
"""Run a small grid of configurations and tabulate them (PLAN.md Section 7.10).

Each configuration is a separate `run_dev_eval.py` invocation into its own
output directory, so every arm keeps its full artifacts and any one of them can
be resumed or re-read later. The table at the end is generated from those
artifacts, not accumulated in memory, which means an interrupted sweep still
reports everything that finished.

Two things to know before using it.

**Sweep on a subset, confirm on the full one.** PLAN.md Section 8 reserves the
30-episode subset as the development loop, and a single 218-subtask arm takes
roughly an hour on this machine. A six-arm grid is therefore a day. Use
`--n-episodes 10` (about 70 subtasks) to rank configurations and then re-run
the winner over the full 30, rather than spending the day up front.

**Arms are compared on the subtasks they share.** Different configurations can
fail differently enough to produce different row counts, so the table reports
each arm over its own completed rows; use `scripts/failure_report.py --compare`
for a strict paired comparison of two arms.

    # the gate that matters most: is memory earning its keep?
    python -u scripts/sweep.py --n-episodes 10 --grid memory

    # evidence threshold, which is currently reasoned rather than measured
    python -u scripts/sweep.py --n-episodes 10 --grid min-observations

    # anything else, spelled out
    python -u scripts/sweep.py --n-episodes 10 \
        --arm "lang020:--language-threshold 0.20" \
        --arm "lang024:--language-threshold 0.24" \
        --arm "lang028:--language-threshold 0.28"
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.report import load_results

# Named grids, so the common sweeps are one word rather than a shell incantation.
GRIDS: dict[str, list[tuple[str, str]]] = {
    # The headline experiment: memory on versus off over identical episodes.
    "memory": [
        ("memon", ""),
        ("nomem", "--no-memory"),
    ],
    # How much evidence an instance needs before it may be targeted. 1 disables
    # the gate; 3 risks starving a recall-limited detector.
    "min-observations": [
        ("minobs1", "--min-observations 1"),
        ("minobs2", "--min-observations 2"),
        ("minobs3", "--min-observations 3"),
    ],
    # Where inside the success radius the agent commits. Stopping on the
    # boundary failed by 2 cm; going too near is where the detector goes blind.
    "commit": [
        ("commit06", "--commit-distance-frac 0.6"),
        ("commit08", "--commit-distance-frac 0.8"),
        ("commit095", "--commit-distance-frac 0.95"),
    ],
    "language": [
        ("lang020", "--language-threshold 0.20"),
        ("lang024", "--language-threshold 0.24"),
        ("lang028", "--language-threshold 0.28"),
    ],
    # Section 7.10's planner axis. A* against Fast Marching, same map, same
    # goals. resolve_planner() refuses 'fmm' outright if skfmm is missing.
    "planner": [
        ("astar", "--planner astar"),
        ("fmm", "--planner fmm"),
    ],
    # Section 7.10's exploration axis. 'random' is the control: if it ties the
    # scored rules, frontier choice is not what limits the agent.
    "frontier": [
        ("nearlarge", "--frontier-strategy nearest-largest"),
        ("largest", "--frontier-strategy largest"),
        ("random", "--frontier-strategy random"),
    ],
    # Section 7.10's encoder axis. It names MobileCLIP-S0, which open_clip
    # 3.3.0 does not have; MobileCLIP2-S0 is the nearest equivalent and
    # validate_encoder() will reject a name that cannot load.
    "encoder": [
        ("vitb32", ""),
        ("mobileclip2s0",
         "--encoder-model MobileCLIP2-S0 --encoder-pretrained dfndr2b"),
    ],
    "merge": [
        ("merge050", "--merge-dist 0.50"),
        ("merge075", "--merge-dist 0.75"),
        ("merge100", "--merge-dist 1.00"),
    ],
}


def _summarise(out_dir: str) -> dict | None:
    rows = load_results(out_dir)
    if not rows:
        return None
    n = len(rows)
    succ = sum(1 for r in rows if r.success)
    return {
        "n": n,
        "sr": 100.0 * succ / n,
        "spl": sum(r.spl for r in rows) / n,
        "detector": 100.0 * sum(1 for r in rows if r.success
                                and r.detector_in_view) / n,
        "stopped": 100.0 * sum(1 for r in rows if r.steps_taken < 500) / n,
        "median_dist": sorted(r.distance_to_goal for r in rows)[n // 2],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=sorted(GRIDS), default=None,
                    help="a named grid; see GRIDS in this file")
    ap.add_argument("--arm", action="append", default=[],
                    help='"name:flags", repeatable, combined with --grid')
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--out-root", default="outputs/sweep")
    ap.add_argument("--resume", action="store_true",
                    help="pass --resume to each arm, so an interrupted sweep "
                         "picks up where it stopped")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arms = list(GRIDS[args.grid]) if args.grid else []
    for spec in args.arm:
        name, _, flags = spec.partition(":")
        arms.append((name.strip(), flags.strip()))
    if not arms:
        ap.error("nothing to run: pass --grid and/or --arm")

    print(f"{len(arms)} arms x {args.n_episodes} episodes -> {args.out_root}/")
    for name, flags in arms:
        print(f"  {name:12s} {flags or '(defaults)'}")
    if args.dry_run:
        return

    results: dict[str, dict | None] = {}
    for i, (name, flags) in enumerate(arms, 1):
        out_dir = os.path.join(args.out_root, name)
        cmd = [sys.executable, "-u", "scripts/run_dev_eval.py",
               "--n-episodes", str(args.n_episodes), "--output-dir", out_dir]
        if args.resume:
            cmd.append("--resume")
        cmd += shlex.split(flags)

        print(f"\n[{i}/{len(arms)}] {name}: {' '.join(cmd[2:])}", flush=True)
        t0 = time.perf_counter()
        # Output goes to a log per arm; a sweep is long and interleaved stdout
        # from several hours of runs is unreadable.
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "run.log")
        with open(log_path, "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        elapsed = (time.perf_counter() - t0) / 60.0
        results[name] = _summarise(out_dir)
        got = results[name]
        print(f"      rc={rc}  {elapsed:.1f} min  "
              f"{'SR %.1f%% over %d subtasks' % (got['sr'], got['n']) if got else 'no results'}"
              f"   log: {log_path}", flush=True)

    # ── table ──
    print(f"\n{'arm':<14}{'n':>5}{'SR':>8}{'SPL':>8}{'det':>7}"
          f"{'stopped':>9}{'med dist':>10}")
    print("-" * 61)
    for name, _ in arms:
        r = results.get(name)
        if not r:
            print(f"{name:<14}{'-':>5}")
            continue
        print(f"{name:<14}{r['n']:>5}{r['sr']:>7.1f}%{r['spl']:>8.3f}"
              f"{r['detector']:>6.1f}%{r['stopped']:>8.1f}%"
              f"{r['median_dist']:>9.2f}m")
    print("\n'det' is the share of subtasks that succeeded AND whose detector "
          "fired at the\nfinal pose; 'stopped' is the share that called STOP "
          "rather than timing out.")
    print(f"\nPaired comparison of two arms:\n  python scripts/failure_report.py "
          f"{args.out_root}/{arms[0][0]} --compare {args.out_root}/{arms[-1][0]}")


if __name__ == "__main__":
    main()
