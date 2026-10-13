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


def save_results(
    results: list[SubtaskResult],
    output_dir: str,
) -> None:
    """Save full evaluation results to output_dir.

    Creates:
        results.csv       — one row per subtask
        summary.json      — aggregated metrics
        failures.jsonl    — one JSON line per failed subtask
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
    summary["total_episodes"] = len(set(r.episode_id for r in results))
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
