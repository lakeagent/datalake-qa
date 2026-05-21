"""
Core metrics implementation for Data Lake Benchmark evaluation.
"""

from typing import List, Dict, Any, Set
import re
from collections import Counter


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Ensure we have a string
    if not isinstance(text, str):
        text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute exact match score (0 or 1).

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        1.0 if exact match (after normalization), 0.0 otherwise
    """
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    return 1.0 if pred_norm == gt_norm else 0.0


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score based on token overlap.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score (0.0 to 1.0)
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    # Count token occurrences
    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)

    # Compute overlap
    overlap = sum((pred_counts & gt_counts).values())

    if overlap == 0:
        return 0.0

    # Compute precision and recall
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)

    # Compute F1
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def extract_source_ids(sources: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract source IDs from evidence sources.

    Args:
        sources: List of evidence sources

    Returns:
        Set of source IDs (table_id or passage_id)
    """
    ids = set()
    for source in sources:
        if 'table_id' in source:
            ids.add(source['table_id'])
        if 'passage_id' in source:
            ids.add(source['passage_id'])
        if 'article_id' in source:
            ids.add(source['article_id'])
    return ids


def compute_evidence_metrics(
    predicted_sources: List[Dict[str, Any]],
    ground_truth_sources: List[Dict[str, Any]],
    evidence_criteria: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute evidence quality metrics.

    Args:
        predicted_sources: Sources cited by model
        ground_truth_sources: Ground truth evidence sources
        evidence_criteria: Evaluation criteria

    Returns:
        Dictionary with precision, recall, F1 for evidence
    """
    pred_ids = extract_source_ids(predicted_sources)
    gt_ids = extract_source_ids(ground_truth_sources)

    if not pred_ids and not gt_ids:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

    if not pred_ids:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    if not gt_ids:
        return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0}

    # Compute overlap
    overlap = len(pred_ids & gt_ids)

    # Precision: what fraction of predicted sources are correct
    precision = overlap / len(pred_ids) if pred_ids else 0.0

    # Recall: what fraction of ground truth sources were found
    recall = overlap / len(gt_ids) if gt_ids else 0.0

    # F1 score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_predicted': len(pred_ids),
        'num_ground_truth': len(gt_ids),
        'num_overlap': overlap
    }


def compute_efficiency_metrics(
    queries_issued: List[Dict[str, Any]],
    expected_queries: List[str],
    optimal_query_count: int
) -> Dict[str, float]:
    """
    Compute retrieval efficiency metrics.

    Args:
        queries_issued: List of queries issued by model
        expected_queries: Expected optimal queries
        optimal_query_count: Optimal number of queries

    Returns:
        Dictionary with efficiency metrics
    """
    query_count = len(queries_issued)

    # Query efficiency: how close to optimal
    if optimal_query_count > 0:
        query_efficiency = min(1.0, optimal_query_count / query_count)
    else:
        query_efficiency = 1.0

    # Token efficiency: total tokens retrieved
    total_tokens_retrieved = sum(
        q.get('results_count', 0) for q in queries_issued
    )

    # Query relevance: proportion of queries that yielded results
    queries_with_results = sum(
        1 for q in queries_issued if q.get('results_count', 0) > 0
    )
    query_relevance = queries_with_results / query_count if query_count > 0 else 0.0

    return {
        'query_count': query_count,
        'optimal_query_count': optimal_query_count,
        'query_efficiency': query_efficiency,
        'total_tokens_retrieved': total_tokens_retrieved,
        'query_relevance': query_relevance,
        'queries_with_results': queries_with_results
    }


def compute_numerical_match(
    prediction: str,
    ground_truth: str,
    tolerance: float = 0.01
) -> float:
    """
    Compute match for numerical answers with tolerance.

    Args:
        prediction: Predicted numerical answer
        ground_truth: Ground truth numerical answer
        tolerance: Relative tolerance for match

    Returns:
        1.0 if within tolerance, 0.0 otherwise
    """
    # Extract numbers from strings
    pred_numbers = re.findall(r'-?\d+\.?\d*', prediction)
    gt_numbers = re.findall(r'-?\d+\.?\d*', ground_truth)

    if not pred_numbers or not gt_numbers:
        # Fall back to string matching
        return compute_exact_match(prediction, ground_truth)

    try:
        pred_val = float(pred_numbers[0])
        gt_val = float(gt_numbers[0])

        if gt_val == 0:
            return 1.0 if pred_val == 0 else 0.0

        # Check relative difference
        rel_diff = abs(pred_val - gt_val) / abs(gt_val)
        return 1.0 if rel_diff <= tolerance else 0.0

    except (ValueError, IndexError):
        return compute_exact_match(prediction, ground_truth)


def compute_date_match(
    prediction: str,
    ground_truth: str,
    require_full_date: bool = False
) -> float:
    """
    Compute match for date answers with flexibility.

    Args:
        prediction: Predicted date
        ground_truth: Ground truth date
        require_full_date: Whether full date (year-month-day) is required

    Returns:
        1.0 if match (with appropriate flexibility), 0.0 otherwise
    """
    # Extract years
    pred_years = re.findall(r'\b\d{4}\b', prediction)
    gt_years = re.findall(r'\b\d{4}\b', ground_truth)

    if not require_full_date:
        # Year-only matching
        if pred_years and gt_years:
            return 1.0 if pred_years[0] == gt_years[0] else 0.0

    # Fall back to exact match
    return compute_exact_match(prediction, ground_truth)


# Semantic similarity (requires sentence-transformers)
def compute_semantic_similarity(prediction: str, ground_truth: str) -> float:
    """
    Compute semantic similarity using sentence embeddings.

    Note: Requires sentence-transformers library.
    Install with: pip install sentence-transformers

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # Load model (cache it in production)
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Compute embeddings
        pred_emb = model.encode([prediction])[0]
        gt_emb = model.encode([ground_truth])[0]

        # Cosine similarity
        similarity = np.dot(pred_emb, gt_emb) / (
            np.linalg.norm(pred_emb) * np.linalg.norm(gt_emb)
        )

        # Normalize to [0, 1]
        return (similarity + 1) / 2

    except ImportError:
        print("Warning: sentence-transformers not installed. Using F1 instead.")
        return compute_f1_score(prediction, ground_truth)
