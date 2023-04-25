import heapq
from dataclasses import dataclass, field
from typing import Generic, List, TypeVar

T = TypeVar("T")


@dataclass(order=True, frozen=True)
class PrioritizedItem(Generic[T]):
    priority: float
    item: T = field(compare=False)


class PriorityCache(Generic[T]):
    def __init__(self, capacity: int) -> None:
        self._capacity: int = capacity
        self._cache: List[PrioritizedItem[T]] = list()

    def add_item(self, priority: float, item: T) -> bool:
        prioritized_item = PrioritizedItem(priority, item)

        if len(self._cache) < self._capacity:
            heapq.heappush(self._cache, prioritized_item)
            return True

        removed_item = heapq.heappushpop(self._cache, prioritized_item)
        return removed_item is not prioritized_item

    def get_items(self) -> List[T]:
        return [prioritized_item.item for prioritized_item in self._cache]
