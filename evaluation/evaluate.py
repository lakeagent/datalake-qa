#!/usr/bin/env python3
"""
Main evaluation script for Data Lake Benchmark.

Usage:
    python evaluate.py --task_file tasks/task.json --predictions pred.json --output results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
from metrics import (
    compute_exact_match,
    compute_f1_score,
    compute_evidence_metrics,
    compute_efficiency_metrics
)


def load_task(task_file: Path) -> Dict[str, Any]:
    """Load task definition from JSON file."""
    with open(task_file, 'r') as f:
        return json.load(f)


def load_predictions(predictions_file: Path) -> Dict[str, Any]:
    """Load model predictions from JSON file."""
    with open(predictions_file, 'r') as f:
        return json.load(f)


def evaluate_answer(
    prediction: str,
    ground_truth: str,
    answer_type: str,
    matching_mode: str = "fuzzy"
) -> Dict[str, float]:
    """
    Evaluate a single answer.

    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer
        answer_type: Type of answer (entity, number, date, etc.)
        matching_mode: Matching strategy (exact, fuzzy, semantic)

    Returns:
        Dictionary with evaluation metrics
    """
    results = {}

    # Exact match
    results['exact_match'] = compute_exact_match(prediction, ground_truth)

    # F1 score (token overlap)
    results['f1_score'] = compute_f1_score(prediction, ground_truth)

    # Type-specific evaluation
    if answer_type == "number":
        # TODO: Implement numerical tolerance matching
        pass
    elif answer_type == "date":
        # TODO: Implement date matching with flexibility
        pass

    return results


def evaluate_evidence(
    predicted_sources: List[Dict[str, Any]],
    ground_truth_sources: List[Dict[str, Any]],
    evidence_criteria: Dict[str, Any]
) -> Dict[str, float]:
    """
    Evaluate quality of evidence sources.

    Args:
        predicted_sources: Sources cited by model
        ground_truth_sources: Ground truth evidence sources
        evidence_criteria: Evaluation criteria from task definition

    Returns:
        Dictionary with evidence quality metrics
    """
    return compute_evidence_metrics(
        predicted_sources,
        ground_truth_sources,
        evidence_criteria
    )


def evaluate_efficiency(
    queries_issued: List[Dict[str, Any]],
    expected_queries: List[str],
    optimal_query_count: int
) -> Dict[str, float]:
    """
    Evaluate retrieval efficiency.

    Args:
        queries_issued: List of queries issued by model
        expected_queries: Expected optimal queries
        optimal_query_count: Optimal number of queries

    Returns:
        Dictionary with efficiency metrics
    """
    return compute_efficiency_metrics(
        queries_issued,
        expected_queries,
        optimal_query_count
    )


def evaluate_question(
    prediction: Dict[str, Any],
    ground_truth: Dict[str, Any],
    evaluation_criteria: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a single question's prediction.

    Args:
        prediction: Model's prediction for this question
        ground_truth: Ground truth for this question
        evaluation_criteria: Task-level evaluation criteria

    Returns:
        Dictionary with all evaluation metrics for this question
    """
    results = {
        'question_id': prediction['question_id'],
        'answer_metrics': {},
        'evidence_metrics': {},
        'efficiency_metrics': {}
    }

    # Evaluate answer
    results['answer_metrics'] = evaluate_answer(
        prediction.get('answer', ''),
        ground_truth.get('answer', ''),
        ground_truth.get('answer_type', 'text'),
        evaluation_criteria.get('answer_matching', 'fuzzy')
    )

    # Evaluate evidence if required
    if evaluation_criteria.get('evidence_required', False):
        results['evidence_metrics'] = evaluate_evidence(
            prediction.get('evidence_sources', []),
            ground_truth.get('evidence_sources', []),
            evaluation_criteria.get('evidence_evaluation', {})
        )

    # Evaluate efficiency if tracking queries
    if 'queries_issued' in prediction:
        efficiency_config = evaluation_criteria.get('efficiency_metrics', {})
        results['efficiency_metrics'] = evaluate_efficiency(
            prediction.get('queries_issued', []),
            ground_truth.get('expected_queries', []),
            efficiency_config.get('optimal_query_count', 0)
        )

    # Compute overall score
    results['correct'] = results['answer_metrics']['exact_match'] == 1.0
    results['score'] = compute_overall_score(results, evaluation_criteria)

    return results


def compute_overall_score(
    question_results: Dict[str, Any],
    evaluation_criteria: Dict[str, Any]
) -> float:
    """
    Compute overall score for a question based on all metrics.

    Args:
        question_results: Results from evaluate_question
        evaluation_criteria: Task-level evaluation criteria

    Returns:
        Overall score (0.0 to 1.0)
    """
    # Default: use F1 score as primary metric
    score = question_results['answer_metrics'].get('f1_score', 0.0)

    # Apply partial credit rules if specified
    partial_credit = evaluation_criteria.get('partial_credit', {})

    if evaluation_criteria.get('evidence_required', False):
        evidence_f1 = question_results['evidence_metrics'].get('f1', 0.0)

        if score > 0 and evidence_f1 == 0:
            # Correct answer but no evidence
            score *= partial_credit.get('verdict_without_evidence', 0.0)
        elif score == 0 and evidence_f1 > 0:
            # Wrong answer but has evidence
            score = evidence_f1 * partial_credit.get('evidence_without_verdict', 0.0)

    return score


def evaluate_task(
    task: Dict[str, Any],
    predictions: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate all predictions for a task.

    Args:
        task: Task definition
        predictions: Model predictions

    Returns:
        Complete evaluation results
    """
    results = {
        'task_id': task['task_id'],
        'category': task['category'],
        'difficulty': task['difficulty'],
        'per_question_results': [],
        'overall_metrics': {}
    }

    # Build lookup for predictions
    pred_lookup = {
        p['question_id']: p
        for p in predictions.get('predictions', [])
    }

    # Evaluate each question
    questions_key = 'questions' if 'questions' in task else 'claims'
    for gt_item in task.get(questions_key, []):
        item_id = gt_item.get('question_id') or gt_item.get('claim_id')

        if item_id not in pred_lookup:
            print(f"Warning: No prediction for {item_id}")
            continue

        question_results = evaluate_question(
            pred_lookup[item_id],
            gt_item,
            task.get('evaluation_criteria', {})
        )
        results['per_question_results'].append(question_results)

    # Compute aggregate metrics
    results['overall_metrics'] = compute_aggregate_metrics(
        results['per_question_results']
    )

    return results


def compute_aggregate_metrics(
    per_question_results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute aggregate metrics across all questions.

    Args:
        per_question_results: List of per-question results

    Returns:
        Dictionary of aggregate metrics
    """
    if not per_question_results:
        return {}

    n = len(per_question_results)

    # Answer metrics
    exact_match = sum(r['answer_metrics']['exact_match'] for r in per_question_results) / n
    f1_score = sum(r['answer_metrics']['f1_score'] for r in per_question_results) / n
    avg_score = sum(r['score'] for r in per_question_results) / n

    metrics = {
        'exact_match': exact_match,
        'f1_score': f1_score,
        'avg_score': avg_score,
        'num_questions': n
    }

    # Evidence metrics (if available)
    if per_question_results[0].get('evidence_metrics'):
        evidence_precision = sum(
            r['evidence_metrics'].get('precision', 0)
            for r in per_question_results
        ) / n
        evidence_recall = sum(
            r['evidence_metrics'].get('recall', 0)
            for r in per_question_results
        ) / n

        metrics['evidence_precision'] = evidence_precision
        metrics['evidence_recall'] = evidence_recall

    # Efficiency metrics (if available)
    if per_question_results[0].get('efficiency_metrics'):
        avg_query_count = sum(
            r['efficiency_metrics'].get('query_count', 0)
            for r in per_question_results
        ) / n

        metrics['avg_query_count'] = avg_query_count

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate predictions for Data Lake Benchmark'
    )
    parser.add_argument(
        '--task_file',
        type=Path,
        required=True,
        help='Path to task definition JSON file'
    )
    parser.add_argument(
        '--predictions',
        type=Path,
        required=True,
        help='Path to predictions JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to output results JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed results'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading task from {args.task_file}")
    task = load_task(args.task_file)

    print(f"Loading predictions from {args.predictions}")
    predictions = load_predictions(args.predictions)

    # Run evaluation
    print(f"Evaluating task {task['task_id']}...")
    results = evaluate_task(task, predictions)

    # Save results
    print(f"Saving results to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print(f"Task: {task['task_id']} ({task['category']}, {task['difficulty']})")
    print("="*60)
    print(f"Exact Match:  {results['overall_metrics']['exact_match']:.2%}")
    print(f"F1 Score:     {results['overall_metrics']['f1_score']:.2%}")
    print(f"Avg Score:    {results['overall_metrics']['avg_score']:.2%}")

    if 'avg_query_count' in results['overall_metrics']:
        print(f"Avg Queries:  {results['overall_metrics']['avg_query_count']:.1f}")

    if args.verbose:
        print("\nPer-question results:")
        for qr in results['per_question_results']:
            print(f"  {qr['question_id']}: {qr['score']:.2%}")


if __name__ == '__main__':
    main()
