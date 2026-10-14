#!/usr/bin/env python3
"""Pair two dev-eval arms and report whether they actually differ.

Every comparison in this project is paired: the same episodes in the same
order, one knob changed. The arithmetic is easy to get wrong in three specific
ways, and all three have happened:

  * ``results.csv`` carries no ``scene_id``, and GOAT-Bench numbers episodes
    per scene, so 218 dev subtasks share only 86 distinct
    ``(episode_id, subtask_index)`` pairs (6.46). Arms must be paired
    POSITIONALLY; matching on the id cross-pairs different scenes.
  * Positional pairing is only sound if the metadata agrees row by row, so
    this refuses to pair rather than silently comparing different subtasks.
  * Comparing the two success RATES ignores that the arms agree on most rows.
    What carries the evidence is the DISAGREEMENTS, which is an exact
    two-sided sign test -- at n=80 it needs six one-way flips before p < 0.05
    (6.54), and nearly every arm run this term has fallen short of that.

Arms still running can be compared: they are truncated to their common prefix,
since both walk the same episodes in the same order.

Usage:
    python scripts/compare_arms.py outputs/dev_eval_A outputs/dev_eval_B
    python scripts/compare_arms.py A B --label-a "memory on" --label-b "off"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from math import comb

TRUE = ("True", "true", "1")

# Metadata that must agree for two rows to be the same subtask. Deliberately
# not 'success' or any other outcome -- those are what is being compared.
IDENTITY = ("episode_id", "subtask_index", "category", "modality")

# Reported for every run. Ordered by distance from the perception stack, which
# is how 6.61 read them: the signal attenuates with every stage downstream.
BOOL_METRICS = ("success", "ever_matched", "detector_in_view")


class ArmMismatch(ValueError):
    """The two arms are not the same subtasks in the same order."""


def sign_test(a_only: int, b_only: int) -> float:
    """Exact two-sided sign test over the discordant pairs.

    Concordant pairs carry no information about a difference, so only the
    disagreements count. Returns 1.0 when there are none.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    p = sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n
    return min(p, 1.0)


def discordant_clusters(a: list[dict], b: list[dict], metric: str) -> tuple[int, int]:
    """Discordant pairs, counted once per distinct goal rather than per subtask.

    GOAT-Bench episodes revisit the same object: the 218-subtask dev subset
    carries only 61 distinct ``(episode_id, category)`` goals, and episode 3
    asks for the same mirror thirteen times, each arrival ending at the same
    distance. Those are not independent observations, so a sign test over
    subtasks overstates its evidence -- 13 flips against 4 (p = 0.049) is 8
    distinct goals against 4 (p = 0.388). See 6.63.

    A cluster that flips both ways counts on both sides: it is evidence for
    neither arm, and assigning it to one would be the convenient choice rather
    than the honest one.
    """
    a_flip, b_flip = set(), set()
    for x, y in zip(a, b):
        key = (x.get("episode_id"), x.get("category"))
        if x.get(metric) in TRUE and y.get(metric) not in TRUE:
            a_flip.add(key)
        elif y.get(metric) in TRUE and x.get(metric) not in TRUE:
            b_flip.add(key)
    return len(a_flip), len(b_flip)


def pair_rows(a: list[dict], b: list[dict]) -> tuple[list[dict], list[dict]]:
    """Truncate both arms to their common prefix, verifying row identity."""
    n = min(len(a), len(b))
    for i in range(n):
        for field in IDENTITY:
            if a[i].get(field) != b[i].get(field):
                raise ArmMismatch(
                    f"arms diverge at row {i}: {field} is "
                    f"{a[i].get(field)!r} in A and {b[i].get(field)!r} in B. "
                    "These are not the same subtasks; pairing them positionally "
                    "would compare unrelated runs."
                )
    return a[:n], b[:n]


def load(path: str) -> list[dict]:
    csv_path = path if path.endswith(".csv") else os.path.join(path, "results.csv")
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _config(path: str) -> dict:
    summary = os.path.join(path, "summary.json")
    if os.path.isdir(path) and os.path.exists(summary):
        with open(summary) as f:
            return json.load(f).get("config", {})
    return {}


def compare(a: list[dict], b: list[dict], label_a: str, label_b: str) -> None:
    a, b = pair_rows(a, b)
    n = len(a)
    if n == 0:
        print("No paired subtasks yet (one arm has written no rows).")
        return

    print(f"paired on {n} subtasks, identity verified row by row "
          f"({', '.join(IDENTITY)})\n")
    # Cells read like "102/218 (46.8%)"; size the columns to the widest one
    # rather than to the label, or the table loses its alignment exactly when
    # the counts get interesting.
    w = max(len(label_a), len(label_b), len(f"{n}/{n} (100.0%)"))
    print(f"{'metric':18s} {label_a:>{w}s} {label_b:>{w}s} "
          f"{'only A':>7s} {'only B':>7s} {'p':>8s}   "
          f"{'clA':>5s} {'clB':>5s} {'p(clust)':>8s}")
    for metric in BOOL_METRICS:
        if metric not in a[0]:
            continue
        ca = sum(1 for r in a if r[metric] in TRUE)
        cb = sum(1 for r in b if r[metric] in TRUE)
        oa = sum(1 for x, y in zip(a, b)
                 if x[metric] in TRUE and y[metric] not in TRUE)
        ob = sum(1 for x, y in zip(a, b)
                 if y[metric] in TRUE and x[metric] not in TRUE)
        p = sign_test(oa, ob)
        ka, kb = discordant_clusters(a, b, metric)
        pc = sign_test(ka, kb)
        star = " *" if p < 0.05 else ""
        starc = " *" if pc < 0.05 else ""
        print(f"{metric:18s} {f'{ca}/{n} ({ca/n*100:.1f}%)':>{w}s} "
              f"{f'{cb}/{n} ({cb/n*100:.1f}%)':>{w}s} "
              f"{oa:7d} {ob:7d} {p:8.4f}{star:2s} "
              f"{ka:5d} {kb:5d} {pc:8.4f}{starc}")

    goals = len({(r.get("episode_id"), r.get("category")) for r in a})
    print(f"\n* p < 0.05. Left p is over subtasks, right over the {goals} "
          f"distinct (episode, category) goals these {n} subtasks carry.")
    print("GOAT-Bench episodes revisit the same object, so subtasks are not "
          "independent; the clustered column is the conservative read (6.63).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm_a")
    ap.add_argument("arm_b")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    args = ap.parse_args()

    la = args.label_a or os.path.basename(args.arm_a.rstrip("/"))[:18]
    lb = args.label_b or os.path.basename(args.arm_b.rstrip("/"))[:18]

    for path, label in ((args.arm_a, la), (args.arm_b, lb)):
        cfg = _config(path)
        if cfg:
            keys = ("resolution", "no_memory", "language_threshold",
                    "merge_dist", "success_distance", "n_episodes")
            shown = {k: cfg[k] for k in keys if k in cfg}
            print(f"{label}: {shown}")
    print()

    compare(load(args.arm_a), load(args.arm_b), la, lb)


if __name__ == "__main__":
    main()
