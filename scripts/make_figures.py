#!/usr/bin/env python3
"""Produce the report's figures from the evaluation artifacts.

Every figure is derived from `results.csv` and `failures.jsonl`, so re-running
an evaluation and re-running this regenerates the report's plots unchanged.
The data shaping is separated from the drawing and tested in
`tests/test_figures.py`: the arithmetic is what can be wrong, and in one case
spectacularly so -- a lifelong curve pooled across episode lengths tells the
opposite story from one controlled for it (6.67).

Usage:
    python scripts/make_figures.py outputs/full_eval --out docs/figures
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRUE = ("True", "true", "1")


# --------------------------------------------------------------------------
# data shaping (pure; tested)
# --------------------------------------------------------------------------

def _ok(row, field="success") -> bool:
    return row.get(field) in TRUE


def episodes_from_rows(rows: list[dict]) -> list[list[dict]]:
    """Split a results.csv into episodes.

    Split on ``subtask_index == 0`` rather than on ``episode_id``: GOAT-Bench
    numbers episodes per scene, so ids repeat and grouping by them silently
    merges unrelated episodes (6.46).
    """
    eps: list[list[dict]] = []
    cur: list[dict] = []
    for r in rows:
        if int(r["subtask_index"]) == 0 and cur:
            eps.append(cur)
            cur = []
        cur.append(r)
    if cur:
        eps.append(cur)
    return eps


def lifelong_pooled(rows: list[dict]) -> tuple[list[int], list[float], list[int]]:
    """SR by subtask index, pooled over every episode.

    This is the curve that appears to show the lifelong effect, and it is an
    artifact: only long episodes reach the high indices, and episode length is
    confounded with difficulty (6.67).
    """
    by = collections.defaultdict(list)
    for e in episodes_from_rows(rows):
        for i, r in enumerate(e):
            by[i].append(_ok(r))
    idx = sorted(by)
    return idx, [sum(by[i]) / len(by[i]) for i in idx], [len(by[i]) for i in idx]


def lifelong_fixed_length(rows: list[dict], length: int):
    """SR by subtask index within episodes of exactly ``length`` subtasks.

    The control: the set of episodes contributing is identical at every x, so
    composition cannot produce a trend.
    """
    sel = [e for e in episodes_from_rows(rows) if len(e) == length]
    if not sel:
        return [], [], []
    by = collections.defaultdict(list)
    for e in sel:
        for i, r in enumerate(e):
            by[i].append(_ok(r))
    idx = sorted(by)
    return idx, [sum(by[i]) / len(by[i]) for i in idx], [len(by[i]) for i in idx]


def paired_halves(rows: list[dict]) -> tuple[int, int, int]:
    """Per episode, is the second half better than the first?

    Paired within an episode, so scene, episode length and difficulty are all
    held constant. An odd-length episode puts the extra subtask in the second
    half.
    """
    better = worse = tied = 0
    for e in episodes_from_rows(rows):
        h = len(e) // 2
        if h == 0:
            continue
        first = sum(_ok(r) for r in e[:h]) / h
        second = sum(_ok(r) for r in e[h:]) / (len(e) - h)
        if second > first:
            better += 1
        elif second < first:
            worse += 1
        else:
            tied += 1
    return better, worse, tied


def by_modality(rows: list[dict]) -> dict:
    out: dict = {}
    for m in {r["modality"] for r in rows}:
        rs = [r for r in rows if r["modality"] == m]
        out[m] = {
            "n": len(rs),
            "sr": sum(_ok(r) for r in rs) / len(rs),
            "spl": sum(float(r["spl"]) for r in rs if float(r["spl"]) == float(r["spl"])) / len(rs),
            "matched": sum(_ok(r, "ever_matched") for r in rs) / len(rs),
        }
    return out


def by_episode_length(rows: list[dict]) -> dict:
    out = collections.defaultdict(list)
    for e in episodes_from_rows(rows):
        for r in e:
            out[len(e)].append(_ok(r))
    return {k: (sum(v) / len(v), len(v)) for k, v in sorted(out.items())}


def failure_taxonomy(failures: list[dict]) -> list[tuple[str, int]]:
    return collections.Counter(f["reason"] for f in failures).most_common()


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

INK = "#1b1b1b"
ACCENT = "#1f6feb"
MUTED = "#9aa0a6"
WARN = "#d1495b"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_lifelong(rows, path):
    """The headline figure: the curve, and the same curve controlled."""
    import matplotlib.pyplot as plt

    idx, sr, n = lifelong_pooled(rows)
    fidx, fsr, fn = lifelong_fixed_length(rows, 10)
    better, worse, tied = paired_halves(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    ax = axes[0]
    ax.plot(idx, [v * 100 for v in sr], marker="o", color=ACCENT, linewidth=2)
    # n below the marker, so the label never sits on the line it describes
    for i, v, c in zip(idx, sr, n):
        ax.annotate(f"{c}", (i, v * 100), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=6.5, color=MUTED)
    _style(ax, "Pooled over all episodes\n(apparent lifelong effect)",
           "subtask index within episode", "success rate (%)")
    ax.set_ylim(0, 16)
    ax.annotate("small grey numbers are the subtasks contributing at each index;\n"
                "they fall from 360 to 64 because only long episodes reach the right",
                xy=(0.03, 0.04), xycoords="axes fraction", fontsize=7,
                color=MUTED, va="bottom")

    ax = axes[1]
    n10 = len([e for e in episodes_from_rows(rows) if len(e) == 10])
    mean = sum(fsr) / len(fsr) * 100 if fsr else 0.0
    # A band of +/- one binomial standard error at n per point. Each point is
    # 64 subtasks, so it scatters by several points on its own; the band makes
    # "consistent with flat" visible instead of asking the reader to infer it.
    inside = 0
    if fsr:
        # +/-2 s.e., not 1: at n=64 a point scatters by ~3.3 points on its own,
        # so a 1 s.e. band excludes about a third of the points even when the
        # underlying rate is perfectly constant. The claim being made is "this
        # is consistent with a flat line", and 2 s.e. is the band that states
        # it honestly.
        se = (mean / 100 * (1 - mean / 100) / fn[0]) ** 0.5 * 100
        lo, hi = mean - 2 * se, mean + 2 * se
        inside = sum(1 for v in fsr if lo <= v * 100 <= hi)
        ax.axhspan(lo, hi, color=MUTED, alpha=0.18, zorder=0)
        ax.axhline(mean, color=MUTED, linewidth=1.1, linestyle="--", zorder=1)
        ax.annotate(f"mean {mean:.1f}%  ±2 s.e. (n={fn[0]} per point)",
                    (idx[-1], mean), textcoords="offset points", xytext=(-4, 5),
                    ha="right", fontsize=7, color=MUTED)
    ax.plot(fidx, [v * 100 for v in fsr], marker="o", color=WARN, linewidth=2, zorder=2)
    _style(ax, f"Only the {n10} episodes of exactly 10 subtasks\n"
               "(composition held constant)",
           "subtask index within episode", "success rate (%)")
    ax.set_ylim(0, 16)
    ax.annotate(f"{inside} of {len(fsr)} points inside the band\n"
                f"paired within episode, all 360: {better} better, {worse} worse, "
                f"{tied} tied\nexact two-sided sign test p = 0.188",
                xy=(0.03, 0.04), xycoords="axes fraction", fontsize=7.5,
                color=INK, va="bottom")

    fig.suptitle("The lifelong curve is an artifact of episode-length composition",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_episode_length(rows, path):
    import matplotlib.pyplot as plt

    d = by_episode_length(rows)
    ks = list(d)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar([str(k) for k in ks], [d[k][0] * 100 for k in ks], color=ACCENT, width=0.6)
    for i, k in enumerate(ks):
        ax.annotate(f"n={d[k][1]}", (i, d[k][0] * 100), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=6.5, color=MUTED)
    _style(ax, "Episode length is confounded with difficulty",
           "subtasks in the episode", "success rate (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_modality(rows, path):
    import matplotlib.pyplot as plt
    import numpy as np

    d = by_modality(rows)
    order = [m for m in ("category", "image", "language") if m in d]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(x - 0.21, [d[m]["sr"] * 100 for m in order], 0.42,
           label="success rate", color=ACCENT)
    ax.bar(x + 0.21, [d[m]["matched"] * 100 for m in order], 0.42,
           label="ever matched a candidate", color=MUTED)
    for i, m in enumerate(order):
        ax.annotate(f"{d[m]['sr']*100:.1f}%", (i - 0.21, d[m]["sr"] * 100),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
        ax.annotate(f"{d[m]['matched']*100:.0f}%", (i + 0.21, d[m]["matched"] * 100),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(n={d[m]['n']})" for m in order])
    _style(ax, "Language trails because its matcher does, not its navigator",
           "", "percent of subtasks")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_attenuation(path):
    """The memory ablation, stage by stage. Values from 6.62, which is a
    paired comparison of two dev-subset runs rather than this results file."""
    import matplotlib.pyplot as plt
    import numpy as np

    stages = ["matched a\ncandidate", "matched within\n1 m of the goal", "scored a\nsuccess"]
    on = [43.6, 12.8, 3.2]
    off = [27.1, 7.8, 3.2]
    ps = ["p = 0.016", "p = 0.080", "p = 1.000"]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - 0.21, on, 0.42, label="memory on", color=ACCENT)
    ax.bar(x + 0.21, off, 0.42, label="memory off", color=MUTED)
    for i, p in enumerate(ps):
        ax.annotate(p, (i, max(on[i], off[i]) + 2.6), ha="center", fontsize=8,
                    color=INK if "0.0" in p and p != "p = 0.080" else MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=8.5)
    _style(ax, "Memory's advantage attenuates at every stage after matching",
           "", "percent of subtasks")
    ax.set_ylim(0, 56)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    # Provenance matters here: unlike every other figure, this one is NOT from
    # the full val_unseen run. It is the memory on/off ablation, which is a
    # paired pair of 218-subtask development runs -- memory cannot be switched
    # off in the single reserved held-out run.
    ax.annotate("from the paired memory on/off ablation: 218 development subtasks, "
                "identical episodes.\np-values are exact two-sided sign tests "
                "clustered by distinct goal (6.62, 6.63).",
                xy=(0.0, -0.22), xycoords="axes fraction", fontsize=7,
                color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# What runner.py's reason codes actually mean, read off the branch that sets
# each (runner.py:377-383). "no_plan" is a misnomer kept for compatibility with
# every results.csv written so far: it is not a planner failure, it is the
# agent issuing STOP while still more than a metre from the goal.
REASON_LABELS = {
    "timeout": "exhausted the 500-step budget",
    "no_plan": "stopped early, >1 m from the goal",
    "wrong_stop": "stopped within 1 m, goal not visible",
}


def fig_failures(failures, path):
    import matplotlib.pyplot as plt

    tax = failure_taxonomy(failures)
    labels = [REASON_LABELS.get(t[0], t[0]) for t in tax][::-1]
    vals = [t[1] for t in tax][::-1]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=(7.4, 0.5 * len(labels) + 1.7))
    ax.barh(labels, vals, color=ACCENT, height=0.6)
    for i, v in enumerate(vals):
        ax.annotate(f"{v}  ({v/total*100:.0f}%)", (v, i), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=8, color=INK)
    _style(ax, f"How the {total} failed subtasks failed", "subtasks", "")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=MUTED, alpha=0.25, linewidth=0.6)
    ax.set_xlim(0, max(vals) * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_scene_spread(rows, scenes, path):
    import matplotlib.pyplot as plt

    d = sorted(((sum(_ok(r) for r in v) / len(v) * 100, sc) for sc, v in scenes.items()))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(range(len(d)), [v for v, _ in d], color=ACCENT, width=0.8)
    overall = sum(_ok(r) for r in rows) / len(rows) * 100
    ax.axhline(overall, color=WARN, linewidth=1.2, linestyle="--")
    # left-hand side: the low scenes leave space there, the high ones do not
    ax.annotate(f"overall {overall:.2f}%", (0.3, overall + 1.4),
                fontsize=8, color=WARN)
    zeros = sum(1 for v, _ in d if v == 0.0)
    if zeros:
        ax.annotate(f"{zeros} scenes yield no successes at all "
                    f"(zero-height bars, far left)",
                    xy=(0.03, 0.88), xycoords="axes fraction",
                    fontsize=7.5, color=MUTED)
    best = d[-1][0]
    ax.annotate(f"best scene {best:.1f}%", (len(d) - 1, best),
                textcoords="offset points", xytext=(-6, 4), ha="right",
                fontsize=7.5, color=INK)
    _style(ax, "Scene identity matters more than any hyperparameter swept",
           f"the {len(d)} val_unseen scenes, ordered by success rate",
           "success rate (%)")
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    with open(os.path.join(args.results_dir, "results.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    fpath = os.path.join(args.results_dir, "failures.jsonl")
    failures = [json.loads(l) for l in open(fpath)] if os.path.exists(fpath) else []

    os.makedirs(args.out, exist_ok=True)

    # scene grouping needs the episode order, which needs the dataset
    scenes = {}
    try:
        from src.sim.goat_dataset import load_val_unseen
        from src.eval.episode_order import eval_episode_order
        order = eval_episode_order(load_val_unseen(
            "data/datasets/goat_bench/hm3d/v1/val_unseen"))
        idx, e = [], -1
        for r in rows:
            if int(r["subtask_index"]) == 0:
                e += 1
            idx.append(e)
        g = collections.defaultdict(list)
        for r, ei in zip(rows, idx):
            g[order[ei].scene_id.split("/")[-2]].append(r)
        scenes = dict(g)
    except Exception as exc:          # pragma: no cover - needs the dataset
        print(f"  (skipping the per-scene figure: {exc})")

    made = []
    fig_lifelong(rows, os.path.join(args.out, "fig_lifelong.png")); made.append("fig_lifelong")
    fig_episode_length(rows, os.path.join(args.out, "fig_episode_length.png")); made.append("fig_episode_length")
    fig_modality(rows, os.path.join(args.out, "fig_modality.png")); made.append("fig_modality")
    fig_attenuation(os.path.join(args.out, "fig_attenuation.png")); made.append("fig_attenuation")
    if failures:
        fig_failures(failures, os.path.join(args.out, "fig_failures.png")); made.append("fig_failures")
    if scenes:
        fig_scene_spread(rows, scenes, os.path.join(args.out, "fig_scene_spread.png")); made.append("fig_scene_spread")

    print(f"{len(rows)} subtasks -> {len(made)} figures in {args.out}")
    for m in made:
        print(f"  {m}.png")


if __name__ == "__main__":
    main()
