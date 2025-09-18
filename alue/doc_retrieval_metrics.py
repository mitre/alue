def recall_at_k_per_query(
    ground_truth_ids: list[list[str]], predicted_ids: list[list[str]]
):
    """
    Calculates recall@k per query

    Parameters
    ----------
    ground_truth_ids: List[List[str]]
        List of list containing ground truth document ids
    prediction_ids: List[List[str]]
        List of list containin predicted document ids

    Returns
    -------
    List[float]
    List of recall values per query
    """
    recall_values = []
    for gt_ids, pred_ids in zip(ground_truth_ids, predicted_ids, strict=False):
        relevant_retrieved = len(set(gt_ids).intersection(set(pred_ids)))
        recall_k = relevant_retrieved / len(gt_ids)
        recall_values.append(recall_k)
    return recall_values


def overall_recall_at_k(recall_values: list[float]):
    """
    Calcluates overall recall value

    Parameters
    ----------
    recall_values: List[float]
        List of recall values per query

    Returns
    -------
    float
    recall@k score
    """
    return sum(recall_values) / len(recall_values)
