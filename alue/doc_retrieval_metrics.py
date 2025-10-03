from typing import List

"""Evaluation metrics for information retrieval and RAG systems.

This module provides metrics for evaluating retrieval quality, specifically
recall@k which measures what proportion of relevant documents are retrieved
in the top-k results.
"""

def recall_at_k_per_query(
    ground_truth_ids: List[List[str]], 
    predicted_ids: List[List[str]]
) -> List[float]:
    """Calculate recall@k for each individual query.
    
    Recall@k measures the proportion of relevant documents that appear in the
    top-k retrieved results. For each query, it is calculated as:
    recall@k = (# relevant docs in top-k) / (total # relevant docs)
    
    Args:
        ground_truth_ids: List of lists where each inner list contains the
            document IDs that are relevant for that query. For example:
            [['doc1', 'doc2'], ['doc3']] means query 0 has 2 relevant docs
            and query 1 has 1 relevant doc.
        predicted_ids: List of lists where each inner list contains the
            top-k retrieved document IDs for that query, in ranked order.
            Must have the same length as ground_truth_ids.
            
    Returns:
        List of recall values (floats between 0.0 and 1.0), one per query.
        A value of 1.0 means all relevant documents were retrieved.
        A value of 0.0 means no relevant documents were retrieved.
        
    Raises:
        ZeroDivisionError: If any query has an empty ground_truth_ids list.
        
    Example:
        >>> ground_truth = [['doc1', 'doc2'], ['doc3', 'doc4']]
        >>> predictions = [['doc1', 'doc5'], ['doc3', 'doc4', 'doc5']]
        >>> recall_at_k_per_query(ground_truth, predictions)
        [0.5, 1.0]  # Query 0: 1/2 relevant docs found, Query 1: 2/2 found
    """
    recall_values = []
    for gt_ids, pred_ids in zip(ground_truth_ids, predicted_ids, strict=False):
        relevant_retrieved = len(set(gt_ids).intersection(set(pred_ids)))
        recall_k = relevant_retrieved / len(gt_ids)
        recall_values.append(recall_k)
    return recall_values


def overall_recall_at_k(recall_values: List[float]) -> float:
    """Calculate mean recall@k across all queries.
    
    Takes per-query recall values and computes the average to produce
    an overall recall@k metric for the entire evaluation set.
    
    Args:
        recall_values: List of recall@k values (floats between 0.0 and 1.0)
            for individual queries, typically from recall_at_k_per_query().
            
    Returns:
        The mean recall value as a float between 0.0 and 1.0.
        
    Raises:
        ZeroDivisionError: If recall_values is an empty list.
        
    Example:
        >>> recall_values = [0.5, 1.0, 0.75, 1.0]
        >>> overall_recall_at_k(recall_values)
        0.8125
        
    Note:
        This is a macro-averaged metric, treating all queries equally
        regardless of how many relevant documents each has.
    """    
    return sum(recall_values) / len(recall_values)
