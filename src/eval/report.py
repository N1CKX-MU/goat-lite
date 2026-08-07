"""Save evaluation results to disk: CSV, JSON summary, and failure logs."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict

from src.eval.runner import SubtaskResult
from src.eval.metrics import aggregate_metrics


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
    fieldnames = [
        "episode_id", "subtask_index", "category", "modality",
        "success", "spl", "distance_to_goal", "steps_taken",
        "actual_path_length", "shortest_path_length", "reason", "wall_time_s",
    ]
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
                }
                f.write(json.dumps(line) + "\n")

    print(f"Results saved to {output_dir}/")
    print(f"  SR: {summary['overall_sr']:.1%}  SPL: {summary['overall_spl']:.3f}")
    print(f"  {len(results)} subtasks, {summary['total_episodes']} episodes")
