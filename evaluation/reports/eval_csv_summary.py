#!/usr/bin/env python3
"""
Compute summary metrics from an eval CSV.
"""

import argparse
import csv
import json
import os
import sys
from typing import List, Dict, Any


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    out = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def _parse_json_list(value: str) -> List[str]:
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [str(x) for x in data]
    return []


def _load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_summary(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "task_count": 0,
            "avg_accuracy": 0.0,
            "avg_used_coverage": 0.0,
            "avg_discovered_coverage": 0.0,
        }

    acc_sum = 0.0
    used_cov_sum = 0.0
    disc_cov_sum = 0.0

    for row in rows:
        expected = row.get("expected_answer", "")
        answer = row.get("answer", "")
        acc = 1.0 if _normalize_text(expected) == _normalize_text(answer) else 0.0
        acc_sum += acc

        required = set(_parse_json_list(row.get("required_datasets", "")))
        actual = set(_parse_json_list(row.get("actual_datasets_used", "")))
        discovered = set(_parse_json_list(row.get("datasets_discovered", "")))

        if required:
            used_cov_sum += len(required & actual) / len(required)
            disc_cov_sum += len(required & discovered) / len(required)
        else:
            used_cov_sum += 0.0
            disc_cov_sum += 0.0

    task_count = len(rows)
    return {
        "task_count": task_count,
        "avg_accuracy": acc_sum / task_count,
        "avg_used_coverage": used_cov_sum / task_count,
        "avg_discovered_coverage": disc_cov_sum / task_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval CSV metrics.")
    parser.add_argument("csv_path", help="Path to <model>_eval.csv")
    args = parser.parse_args()

    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    rows = _load_rows(csv_path)
    summary = compute_summary(rows)

    print(f"Tasks: {summary['task_count']}")
    print(f"Avg accuracy: {summary['avg_accuracy']:.4f}")
    print(f"Avg datasets used coverage: {summary['avg_used_coverage']:.4f}")
    print(f"Avg datasets discovered coverage: {summary['avg_discovered_coverage']:.4f}")


if __name__ == "__main__":
    main()
