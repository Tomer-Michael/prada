# todo tomer go over this file
import bisect
import itertools
from typing import Callable, Optional, Sequence, Tuple, TypeVar, Union

T = TypeVar("T")

_TRUE_STRINGS: Tuple[str, ...] = ("true", "t", "yes", "y", "on", "1")
_FALSE_STRINGS: Tuple[str, ...] = ("false", "f", "no", "n", "off", "0")
_BOOLEAN_STRINGS: Tuple[str, ...] = _TRUE_STRINGS + _FALSE_STRINGS


def str_to_bool(value: str) -> bool:
    lower_value = value.lower()
    if lower_value in _TRUE_STRINGS:
        return True
    if lower_value in _FALSE_STRINGS:
        return False
    raise ValueError(f"Got an unsupported string representation of a boolean: [{value}]. "
                     f"The supported values are [{_BOOLEAN_STRINGS}].")


def convert_to_bool(value: Union[bool, str, int, float]) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return str_to_bool(value)

    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True

    raise ValueError(f"Got an unsupported representation of a boolean: [{value}].")

# todo tomer why is it here?
def get_weight_function(is_sequence_level: bool) -> Callable[[T], int]:
    if is_sequence_level:
        return lambda x: 1
    else:
        return len

# todo tomer why is it here?
def get_split_index(data: Sequence[T], k: int, weight_function: Callable[[T], int],
                    take_at_least_k: bool = True, min_weight_for_split: Optional[int] = None) -> int:
    weights = map(weight_function, data)
    cumulative_weights = list(itertools.accumulate(weights))

    if min_weight_for_split is not None:
        total_weight = cumulative_weights[-1]
        if total_weight < min_weight_for_split:
            return -1

    search_method = bisect.bisect_right if take_at_least_k else bisect.bisect_left
    split_index = search_method(cumulative_weights, k)
    return split_index
