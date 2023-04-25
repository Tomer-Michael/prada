import numpy as np


def topk(array: np.ndarray, k: int) -> np.ndarray:
    """
    Returns the indices of the top k values in the array, sorted in descending order.
    """
    assert len(array.shape) == 1
    k = min(k, array.shape[0])
    top_k_indices = np.argpartition(array, -k)[-k:]
    top_k_values = array[top_k_indices]
    top_k_indices_sorted = top_k_indices[np.argsort(-top_k_values)]
    return top_k_indices_sorted
