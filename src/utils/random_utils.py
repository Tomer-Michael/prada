# todo tomer go over this file
import random
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

from src.utils import python_utils

T = TypeVar("T")


def shuffled_copy(data: Sequence[T], seed: Optional[int]) -> List[T]:
    shuffled = sample(data, k=len(data), seed=seed)
    return shuffled


def sample(data: Sequence[T], k: int, seed: Optional[int]) -> List[T]:
    if seed is not None:
        random.seed(seed)
    if k > len(data):
        k = len(data)
    sampled = random.sample(data, k)
    return sampled

# todo tomer better name and/or doc
def sample_2d(data: Sequence[T], k: int, weight_function: Callable[[T], int], seed: Optional[int]) -> List[T]:
    shuffled_data = shuffled_copy(data, seed)
    split_index = python_utils.get_split_index(shuffled_data, k, weight_function, take_at_least_k=True)
    return shuffled_data[: split_index]

# todo tomer better name and/or doc
def random_split(data: Sequence[T], k: int, weight_function: Callable[[T], int], seed: Optional[int]) -> Tuple[List[T], List[T]]:
    shuffled_data = shuffled_copy(data, seed)
    min_weight_for_split = 2 * k
    split_index = python_utils.get_split_index(shuffled_data, k, weight_function, take_at_least_k=False,
                                               min_weight_for_split=min_weight_for_split)
    if split_index == -1:
        return list(), shuffled_data
    return shuffled_data[: split_index], shuffled_data[split_index:]
