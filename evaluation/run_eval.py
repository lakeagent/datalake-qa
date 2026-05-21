#!/usr/bin/env python3
"""
Run evaluation across multiple models on benchmark tasks.

Usage:
    # Evaluate all models on a single task directory
    python evaluation/run_eval.py --task-dir tasks/k-3-d-2/

    # Evaluate specific models
    python evaluation/run_eval.py --task-dir tasks/k-3-d-2/ --models gpt-5.2 bedrock/claude-opus-4.5

    # Evaluate all tasks in default 'tasks' directory
    python evaluation/run_eval.py --all-tasks

    # Evaluate all tasks in 'tasks_mini' directory
    python evaluation/run_eval.py --all-tasks --task-base tasks_mini

    # Evaluate all tasks in 'wikipedia_tasks_case_study' directory
    python evaluation/run_eval.py --all-tasks --task-base wikipedia_tasks_case_study
"""

import argparse
import glob
import json
import os
import sys
import csv
from datetime import datetime
from pathlib import Path


def _sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in file paths (replace / with _)."""
    return model.replace("/", "_")

def _ensure_venv_with_boto3():
    if os.environ.get("EVAL_USE_VENV") == "1":
        return
    try:
        import boto3  # noqa: F401
        return
    except Exception:
        venv_python = Path(__file__).resolve().parent / "venv" / "bin" / "python"
        if venv_python.exists():
            env = os.environ.copy()
            env["EVAL_USE_VENV"] = "1"
            os.execve(str(venv_python), [str(venv_python)] + sys.argv, env)


_ensure_venv_with_boto3()

if __package__:
    from .agent_runner import BatchRunner, AgentConfig
else:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from evaluation.agent_runner import BatchRunner, AgentConfig


DEFAULT_MODELS = [
    "gpt-5.2",
    "gpt-5-mini",
    "bedrock/claude-opus-4.5",
    "bedrock/claude-sonnet-4.5",
    "bedrock/claude-haiku-4.5",
]


def find_all_task_dirs(base_dir: str = "tasks") -> list:
    """Find all task directories matching k-*-d-* pattern."""
    pattern = os.path.join(base_dir, "k-*-d-*")
    return sorted(glob.glob(pattern))


def run_evaluation(
    task_dir: str,
    models: list,
    verbose: bool = False,
    max_turns: int = 100,
    only_new: bool = False,
    reasoning_effort: str = "medium",
    parallel: int = 6,
) -> dict:
    """
    Run evaluation on a task directory across multiple models.

    Args:
        task_dir: Directory containing task JSON files
        models: List of model names to evaluate
        verbose: Print verbose output
        max_turns: Max agent turns per task
        reasoning_effort: For OpenAI models - "low", "medium", or "high"
        parallel: Number of parallel processes (max_workers)

    Returns:
        Dict mapping model -> results
    """
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    task_files = sorted(glob.glob(os.path.join(task_dir, "*.json")))
    if not task_files:
        print(f"No task files found in {task_dir}")
        return {}

    task_dir_name = os.path.basename(task_dir)
    print(f"\nEvaluating {len(task_files)} tasks from {task_dir_name}")
    print(f"Models: {', '.join(models)}")
    print(f"Timeout: 600s per task")
    print("=" * 60)

    config = AgentConfig(max_turns=max_turns, verbose=verbose, reasoning_effort=reasoning_effort)
    all_results = {}

    tasks_by_id = {}
    for path in task_files:
        with open(path) as f:
            task = json.load(f)
            task["id"] = path
            tasks_by_id[path] = task

    def extract_dataset_id(source: str) -> str:
        if not source:
            return ""
        parts = [p for p in source.strip("/").split("/") if p]
        if not parts:
            return ""
        if parts[0] in ("datagov", "wikipedia"):
            return parts[1] if len(parts) > 1 else ""
        return parts[0]

    for model in models:
        print(f"\n--- Running {model} ---")

        try:
            task_files_to_run = task_files
            safe_model_name = _sanitize_model_name(model)
            csv_path = os.path.join(output_dir, f"{safe_model_name}_eval.csv")
            if only_new and os.path.exists(csv_path):
                existing_task_ids = set()
                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        task_id = row.get("task_id", "")
                        if task_id:
                            existing_task_ids.add(task_id)
                task_files_to_run = [p for p in task_files if p not in existing_task_ids]
                if not task_files_to_run:
                    print("  No new tasks to evaluate for this model.")
                    all_results[model] = {
                        "summary": {
                            "model": model,
                            "task_dir": task_dir_name,
                            "total_tasks": 0,
                            "exact_match_count": 0,
                            "exact_match_rate": 0,
                            "avg_f1_score": 0,
                            "avg_tool_calls": 0,
                            "avg_tokens": 0,
                            "avg_time": 0,
                        },
                        "results": [],
                    }
                    continue

            batch = BatchRunner(model=model, config=config, max_workers=parallel)
            results = batch.run_from_files(task_files_to_run, verbose=verbose)

            # Compute summary
            total = len(results)
            exact_matches = sum(r.get("exact_match", 0) for r in results)
            f1_scores = [r.get("f1_score", 0) for r in results if "f1_score" in r]
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

            total_cost = sum(r.get("cost", 0) for r in results)
            summary = {
                "model": model,
                "task_dir": task_dir_name,
                "total_tasks": total,
                "exact_match_count": exact_matches,
                "exact_match_rate": exact_matches / total if total else 0,
                "avg_f1_score": avg_f1,
                "avg_tool_calls": sum(r.get("tool_calls", 0) for r in results) / total if total else 0,
                "avg_tokens": sum(r.get("tokens", 0) for r in results) / total if total else 0,
                "avg_time": sum(r.get("time", 0) for r in results) / total if total else 0,
                "total_cost": total_cost,
                "avg_cost": total_cost / total if total else 0,
            }

            all_results[model] = {
                "summary": summary,
                "results": results,
            }

            print(f"  Exact Match: {exact_matches}/{total} ({100*exact_matches/total:.1f}%)")
            print(f"  Avg F1: {avg_f1:.3f}")

        except Exception as e:
            print(f"  Error: {e}")
            all_results[model] = {"error": str(e)}
            continue

        csv_path = os.path.join(output_dir, f"{safe_model_name}_eval.csv")
        fieldnames = [
            "task_id",
            "expected_answer",
            "answer",
            "required_datasets",
            "actual_datasets_used",
            "datasets_discovered",
            "runtime_seconds",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
        ]
        existing_rows = {}
        if os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    task_id = row.get("task_id", "")
                    if task_id:
                        existing_rows[task_id] = row

        for r in results:
            task_id = r.get("task_id", "")
            task = tasks_by_id.get(task_id, {})
            required = task.get("datasets_used", [])
            existing_rows[task_id] = {
                "task_id": task_id,
                "expected_answer": task.get("answer", ""),
                "answer": r.get("predicted_answer", ""),
                "required_datasets": json.dumps(sorted(required)),
                "actual_datasets_used": json.dumps(r.get("datasets_used", [])),
                "datasets_discovered": json.dumps(r.get("datasets_discovered", [])),
                "runtime_seconds": r.get("time", 0),
                "input_tokens": r.get("input_tokens", 0),
                "output_tokens": r.get("output_tokens", 0),
                "total_tokens": r.get("tokens", 0),
                "cost_usd": r.get("cost", 0),
            }

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for task_id in sorted(existing_rows.keys()):
                writer.writerow(existing_rows[task_id])

    return all_results


def print_comparison_table(results: dict):
    """Print a comparison table of results across models."""
    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    print(f"{'Model':<25} {'EM Rate':<10} {'Avg F1':<10} {'Avg Tools':<10} {'Avg Time':<10} {'Total Cost':<12}")
    print("-" * 90)

    for model, data in results.items():
        if "error" in data:
            print(f"{model:<25} ERROR: {data['error'][:50]}")
        else:
            s = data["summary"]
            total_cost = s.get('total_cost', 0)
            cost_str = f"${total_cost:.4f}" if total_cost > 0 else "N/A"
            print(f"{model:<25} {s['exact_match_rate']*100:>5.1f}%    {s['avg_f1_score']:>6.3f}    {s['avg_tool_calls']:>6.1f}     {s['avg_time']:>6.1f}s    {cost_str:>10}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Run multi-model evaluation")
    parser.add_argument("--task-dir", "-d", help="Task directory to evaluate")
    parser.add_argument("--all-tasks", action="store_true", help="Evaluate all task directories")
    parser.add_argument("--task-base", "-b", default="tasks",
                        help="Base directory for tasks (default: tasks). Use 'tasks_mini' or 'wikipedia_tasks_case_study' for other task sets")
    parser.add_argument("--models", "-m", nargs="+", default=DEFAULT_MODELS, help="Models to evaluate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--max-turns", type=int, default=25, help="Max agent turns")
    parser.add_argument("--only-new", action="store_true", help="Evaluate only tasks not already in the model CSV")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"], default="medium",
                        help="Reasoning effort for OpenAI models (low/medium/high/xhigh)")
    # for parallel
    parser.add_argument("--parallel", type=int, default=10, help="Number of parallel processes")

    args = parser.parse_args()

    start_time = datetime.now()

    if args.all_tasks:
        task_dirs = find_all_task_dirs(base_dir=args.task_base)
        print(f"Found {len(task_dirs)} task directories in {args.task_base}")

        all_summaries = {}
        for task_dir in task_dirs:
            results = run_evaluation(
                task_dir=task_dir,
                models=args.models,
                verbose=args.verbose,
                max_turns=args.max_turns,
                only_new=args.only_new,
                reasoning_effort=args.reasoning_effort,
                parallel=args.parallel,
            )
            all_summaries[os.path.basename(task_dir)] = results
            print_comparison_table(results)

    elif args.task_dir:
        results = run_evaluation(
            task_dir=args.task_dir,
            models=args.models,
            verbose=args.verbose,
            max_turns=args.max_turns,
            only_new=args.only_new,
            reasoning_effort=args.reasoning_effort,
            parallel=args.parallel,
        )
        print_comparison_table(results)

    else:
        parser.print_help()

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"\nTotal evaluation time: {elapsed}")

if __name__ == "__main__":
    main()
