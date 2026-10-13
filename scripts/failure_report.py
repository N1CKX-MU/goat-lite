#!/usr/bin/env python3
"""Turn a dev-eval results.csv into the failure taxonomy the report needs.

PLAN.md Section 7.9 asks for a documented failure taxonomy; Section 8 asks for
SR and SPL by modality and by subtask index (the lifelong curve). This produces
all of it as markdown, from the artifacts run_dev_eval.py already writes, so the
numbers in the report are generated rather than transcribed.

Two notes on reading the output.

The `reason` column written by the runner is coarse: "no_plan" is its catch-all
for "the agent stopped early and was not close", which is not necessarily a
planning failure. The buckets below are therefore derived from what was actually
measured -- did the agent stop or run out of steps, and where did it end up --
rather than taken from that label.

`detector_agrees` is reported separately from success throughout. Success uses
the benchmark's geometric visibility test; the detector column records whether
the agent's own YOLO also fired on the goal class in the final frame. The gap
between them is the subject of engineering-log 6.35.

Usage:
    python scripts/failure_report.py outputs/dev_eval_20261013_memon
    python scripts/failure_report.py outputs/a --compare outputs/b   # ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from statistics import median

MAX_STEPS = 500


def _load(results_dir: str) -> list[dict]:
    path = os.path.join(results_dir, "results.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["success"] = r["success"] == "True"
        r["detector_in_view"] = r.get("detector_in_view", "False") == "True"
        r["spl"] = float(r["spl"])
        r["steps_taken"] = int(r["steps_taken"])
        r["subtask_index"] = int(r["subtask_index"])
        r["distance_to_goal"] = float(r["distance_to_goal"])
        r["actual_path_length"] = float(r["actual_path_length"])
        # Diagnostics, absent from runs made before they were added.
        if "ever_matched" in r:
            r["ever_matched"] = r["ever_matched"] == "True"
        for k in ("node_error_m", "explored_m2"):
            if k in r:
                r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
        for k in ("steps_searching", "steps_approaching", "steps_verifying"):
            if k in r:
                r[k] = int(r[k])
    return rows


def _rate(rows: list[dict], key="success") -> float:
    return 100.0 * sum(bool(r[key]) for r in rows) / max(len(rows), 1)


def _bucket(r: dict) -> str:
    """What actually happened, from the measurements rather than the label."""
    if r["success"]:
        return "success"
    stopped = r["steps_taken"] < MAX_STEPS
    d = r["distance_to_goal"]
    if stopped and d <= 1.0:
        # Inside the radius, stopped on purpose, still not credited: the goal
        # was not visible from where it stopped -- wrong side of a wall, or
        # facing away, or the matched node was a different object.
        return "stopped in radius, goal not visible"
    if stopped and d <= 3.0:
        return "stopped just short (1-3 m)"
    if stopped:
        return "stopped at the wrong object (>3 m)"
    if d <= 1.0:
        return "reached the goal, never stopped"
    if d <= 3.0:
        return "timed out just short (1-3 m)"
    return "timed out far away (>3 m)"


def _diagnosis(rows: list[dict]) -> str:
    """Split the failures by what the agent actually believed.

    "Timed out far away" is three different problems wearing one label. The
    matcher either never produced a candidate (perception or exploration), or
    produced one that was not the goal (matcher precision), or produced the
    right one and the agent still did not finish (stop rule or step budget).
    """
    fails = [r for r in rows if not r["success"]]
    buckets = Counter()
    for r in fails:
        if not r["ever_matched"]:
            buckets["never matched any candidate"] += 1
            continue
        e = r["node_error_m"]
        if e != e:  # nan: matched at some point, nothing held at the end
            buckets["matched, then dropped the candidate"] += 1
        elif e <= 1.0:
            buckets["candidate was the goal (stop rule or budget)"] += 1
        elif e <= 3.0:
            buckets["candidate near the goal (1-3 m localisation error)"] += 1
        else:
            buckets["committed to the wrong object (>3 m)"] += 1

    n = max(len(fails), 1)
    out = _table(
        "Why the failures failed", ("diagnosis", "n", "share of failures"),
        [(k, v, f"{100.0 * v / n:.1f}%") for k, v in buckets.most_common()],
    )

    def _med(key):
        vals = [r[key] for r in rows if key in r]
        return median(vals) if vals else float("nan")

    total = _med("steps_searching") + _med("steps_approaching") + _med("steps_verifying")
    out += _table(
        "Where the step budget went (median per subtask)",
        ("state", "steps", "share"),
        [(s_, f"{_med('steps_' + s_):.0f}",
          f"{100.0 * _med('steps_' + s_) / max(total, 1):.0f}%")
         for s_ in ("searching", "approaching", "verifying")],
    )
    out += (f"\nMedian floor area explored by the end: "
            f"**{_med('explored_m2'):.1f} m2**. For scale, the 22 dev-subset "
            f"scenes have a median navigable area of 88.9 m2 (max 291.8), and "
            f"500 steps buys at most 125 m of travel -- less once turning is "
            f"paid for.\n")
    return out


def _table(title: str, header: tuple, rows: list[tuple]) -> str:
    out = [f"\n### {title}\n", "| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def _group(rows: list[dict], key: str, order=None) -> list[tuple]:
    groups: dict = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    keys = order if order else sorted(groups)
    out = []
    for k in keys:
        g = groups.get(k if not isinstance(k, str) else k, [])
        if not g:
            continue
        out.append((
            k, len(g), f"{_rate(g):.1f}%",
            f"{sum(r['spl'] for r in g) / len(g):.3f}",
            f"{_rate(g, 'detector_in_view'):.1f}%",
            f"{median(r['distance_to_goal'] for r in g):.2f}",
        ))
    return out


def report(results_dir: str) -> str:
    rows = _load(results_dir)
    summary_path = os.path.join(results_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        summary = json.load(open(summary_path))

    n = len(rows)
    out = [f"# Failure taxonomy — `{results_dir}`\n",
           f"{n} subtasks over "
           f"{len(set(r['episode_id'] for r in rows))} episodes.\n"]

    out.append(_table(
        "Headline", ("metric", "value"),
        [("SR", f"{_rate(rows):.1f}%  ({sum(r['success'] for r in rows)}/{n})"),
         ("SPL", f"{sum(r['spl'] for r in rows) / max(n, 1):.3f}"),
         ("SR where the detector also fired",
          f"{100.0 * sum(r['success'] and r['detector_in_view'] for r in rows) / max(n, 1):.1f}%"),
         ("detector fired at the final pose",
          f"{_rate(rows, 'detector_in_view'):.1f}%"),
         ("median final distance to goal",
          f"{median(r['distance_to_goal'] for r in rows):.2f} m"),
         ("subtasks that called STOP",
          f"{100.0 * sum(r['steps_taken'] < MAX_STEPS for r in rows) / max(n, 1):.1f}%"),
         ("median steps", f"{median(r['steps_taken'] for r in rows):.0f}")],
    ))

    cols = ("", "n", "SR", "SPL", "detector agrees", "median dist (m)")
    out.append(_table("By modality", cols,
                      _group(rows, "modality",
                             ["category", "language", "image"])))
    out.append(_table("By subtask index within the episode (the lifelong curve)",
                      cols, _group(rows, "subtask_index",
                                   sorted({r["subtask_index"] for r in rows}))))

    buckets = Counter(_bucket(r) for r in rows)
    out.append(_table(
        "What happened", ("outcome", "n", "share"),
        [(k, v, f"{100.0 * v / max(n, 1):.1f}%")
         for k, v in buckets.most_common()],
    ))

    if "ever_matched" in rows[0]:
        out.append(_diagnosis(rows))

    by_cat = _group(rows, "category")
    by_cat.sort(key=lambda t: (-t[1], t[0]))
    out.append(_table("By goal category", cols, by_cat))

    # The runner's own label, kept for continuity with earlier runs.
    out.append(_table(
        "Runner `reason` label", ("reason", "n", "share"),
        [(k, v, f"{100.0 * v / max(n, 1):.1f}%")
         for k, v in Counter(r["reason"] for r in rows).most_common()],
    ))

    if summary:
        out.append(f"\n_summary.json: SR {summary.get('overall_sr', 0):.3f}, "
                   f"SPL {summary.get('overall_spl', 0):.3f}, "
                   f"avg wall time "
                   f"{summary.get('avg_wall_time_s', 0):.1f}s per subtask._\n")
    return "\n".join(out)


def compare(a_dir: str, b_dir: str) -> str:
    """Side-by-side for the memory on/off ablation."""
    a, b = _load(a_dir), _load(b_dir)
    # Only subtasks present in both arms, so the delta is on identical work.
    key = lambda r: (r["episode_id"], r["subtask_index"])
    common = {key(r) for r in a} & {key(r) for r in b}
    a = [r for r in a if key(r) in common]
    b = [r for r in b if key(r) in common]

    out = [f"\n# Ablation — `{a_dir}` vs `{b_dir}`\n",
           f"{len(common)} subtasks present in both arms.\n"]
    rows = [("SR", f"{_rate(a):.1f}%", f"{_rate(b):.1f}%",
             f"{_rate(a) - _rate(b):+.1f} pp"),
            ("SPL", f"{sum(r['spl'] for r in a) / max(len(a), 1):.3f}",
             f"{sum(r['spl'] for r in b) / max(len(b), 1):.3f}", ""),
            ("median dist (m)",
             f"{median(r['distance_to_goal'] for r in a):.2f}",
             f"{median(r['distance_to_goal'] for r in b):.2f}", "")]
    out.append(_table("Overall", ("metric", "A", "B", "delta"), rows))

    per_idx = []
    for i in sorted({r["subtask_index"] for r in a}):
        ga = [r for r in a if r["subtask_index"] == i]
        gb = [r for r in b if r["subtask_index"] == i]
        if not ga or not gb:
            continue
        per_idx.append((i, len(ga), f"{_rate(ga):.1f}%", f"{_rate(gb):.1f}%",
                        f"{_rate(ga) - _rate(gb):+.1f} pp"))
    out.append(_table("By subtask index — this is the lifelong result",
                      ("index", "n", "A", "B", "delta"), per_idx))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--compare", default=None,
                    help="second results dir, e.g. the --no-memory arm")
    ap.add_argument("--out", default=None,
                    help="write markdown here (default: print)")
    args = ap.parse_args()

    text = report(args.results_dir)
    if args.compare:
        text += compare(args.results_dir, args.compare)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
