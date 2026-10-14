"""Save evaluation results to disk: CSV, JSON summary, and failure logs."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, fields

from src.eval.runner import SubtaskResult
from src.eval.metrics import aggregate_metrics


def load_results(output_dir: str) -> list[SubtaskResult]:
    """Read back a previous run's results.csv as SubtaskResult objects.

    Used by --resume. A dev eval takes hours, and losing a nearly complete run
    to a crash on the last scene is not acceptable; results are already
    checkpointed per scene, and this is the other half of that.

    Rows are matched to the dataclass by field name and converted by the
    declared type, so a CSV written before a field existed still loads (the
    missing field keeps its default).
    """
    path = os.path.join(output_dir, "results.csv")
    if not os.path.exists(path):
        return []

    types = {f.name: f.type for f in fields(SubtaskResult)}

    def _coerce(name: str, raw: str):
        t = types.get(name, "str")
        if t == "bool" or t is bool:
            return raw == "True"
        if t == "int" or t is int:
            return int(float(raw))
        if t == "float" or t is float:
            return float(raw)
        return raw

    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            kwargs = {k: _coerce(k, v) for k, v in row.items()
                      if k in types and v != ""}
            try:
                out.append(SubtaskResult(**kwargs))
            except TypeError:
                continue  # a row missing a required field is not resumable
    return out


def resumable_prefix(prior, scene_keys):
    """How many leading scenes of a run the rows already on disk really cover.

    Args:
        prior: rows from load_results(), in the order they were written.
        scene_keys: for each scene in run order, the list of
            ``(episode_id, subtask_index)`` it is expected to produce.

    Returns:
        ``(rows_to_keep, n_scenes_done)`` -- the verified prefix of ``prior``
        and how many scenes it accounts for. Resuming stops at the first
        mismatch and everything from there is redone.

    Matching is POSITIONAL, and that is the whole point.
    ``(episode_id, subtask_index)`` is not unique across a run: GOAT-Bench
    numbers episodes per scene, so over the 30-episode dev subset 218 subtasks
    share only 86 distinct such pairs, and episode id "9" appears in five
    different scenes. Keying resume on that pair -- which the first
    implementation did -- lets a scene be skipped because an *earlier, unrelated*
    scene happened to produce the same ids: it silently dropped scene 6 of 22
    from a run, and would have dropped more as the covered set grew. Nothing
    errors; the output is just quietly short.

    The sequence, by contrast, is deterministic: the dev subset is sampled with
    a fixed seed and scenes are visited in a fixed order, so the n-th row of
    results.csv always belongs to the same subtask. save_results() is called
    only after a scene completes, so the file always holds a whole number of
    scenes.
    """
    rows_done, cursor = 0, 0
    for keys in scene_keys:
        chunk = prior[cursor:cursor + len(keys)]
        if len(chunk) != len(keys):
            break
        if [(r.episode_id, r.subtask_index) for r in chunk] != list(keys):
            break
        cursor += len(keys)
        rows_done += 1
    return prior[:cursor], rows_done


def save_results(
    results: list[SubtaskResult],
    output_dir: str,
    config: dict | None = None,
) -> None:
    """Save full evaluation results to output_dir.

    Creates:
        results.csv       — one row per subtask
        summary.json      — aggregated metrics (plus ``config``, if given)
        failures.jsonl    — one JSON line per failed subtask

    ``config`` is the resolved run configuration. Without it an output
    directory does not say what produced it: today's three sweep grids had to
    be told apart by directory name and paired positionally, because
    results.csv carries no scene_id and summary.json carried no settings.
    """
    os.makedirs(output_dir, exist_ok=True)

    # results.csv
    csv_path = os.path.join(output_dir, "results.csv")
    # Derived from the dataclass rather than hand-listed: the hand-listed
    # version silently went stale when SubtaskResult gained a field, and
    # DictWriter raises on the extra key -- at the first per-scene checkpoint,
    # hours into a run.
    fieldnames = [f.name for f in fields(SubtaskResult)]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # summary.json
    agg_input = [
        {
            "success": r.success,
            "spl": r.spl,
            "modality": r.modality,
            "subtask_index": r.subtask_index,
        }
        for r in results
    ]
    summary = aggregate_metrics(agg_input)
    summary["total_subtasks"] = len(results)
    # Count episodes by their subtask 0, not by distinct episode_id: GOAT-Bench
    # numbers episodes per scene, so 218 dev subtasks share only 86 distinct
    # (episode_id, subtask_index) pairs and 30 episodes reported as 10 (6.46).
    # Every episode contributes exactly one subtask_index == 0.
    summary["total_episodes"] = sum(1 for r in results if r.subtask_index == 0)
    summary["avg_steps"] = sum(r.steps_taken for r in results) / max(len(results), 1)
    summary["avg_wall_time_s"] = sum(r.wall_time_s for r in results) / max(len(results), 1)
    # Success is scored on the benchmark's geometric visibility test. This is
    # the stricter reading -- navigated there AND the detector recognised it in
    # the final frame -- kept alongside so neither number is lost.
    n = max(len(results), 1)
    summary["sr_detector_agrees"] = sum(
        1 for r in results if r.success and r.detector_in_view
    ) / n
    summary["detector_in_view_rate"] = sum(
        1 for r in results if r.detector_in_view
    ) / n

    if config is not None:
        summary["config"] = config

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # failures.jsonl
    failures_path = os.path.join(output_dir, "failures.jsonl")
    with open(failures_path, "w") as f:
        for r in results:
            if not r.success:
                line = {
                    "episode_id": r.episode_id,
                    "subtask_index": r.subtask_index,
                    "category": r.category,
                    "modality": r.modality,
                    "reason": r.reason,
                    "distance_to_goal": r.distance_to_goal,
                    "steps_taken": r.steps_taken,
                    "detector_in_view": r.detector_in_view,
                }
                f.write(json.dumps(line) + "\n")

    print(f"Results saved to {output_dir}/")
    print(f"  SR: {summary['overall_sr']:.1%}  SPL: {summary['overall_spl']:.3f}"
          f"  (detector also agrees: {summary['sr_detector_agrees']:.1%})")
    print(f"  {len(results)} subtasks, {summary['total_episodes']} episodes")
