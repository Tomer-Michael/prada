# todo tomer go over this file
from typing import List, Tuple

import torch

from src.utils import torch_utils
from src.utils.priority_cache import PriorityCache


class ScoresTracker:
    def __init__(self, capacity: int, is_sequence_level: bool) -> None:
        self._scores_cache: PriorityCache[torch.Tensor] = PriorityCache(capacity)
        self._is_sequence_level: bool = is_sequence_level

    def register_score(self, batched_features: torch.Tensor, scores: Tuple[float, ...]) -> None:
        assert len(batched_features.shape) == 2
        batch_size = batched_features.shape[1] # todo tomer could be a mistake?

        if self._is_sequence_level:  # todo tomer what is the difference?
            assert len(scores) == 1, f"Should use 1 feature for entire sequence, but got {len(scores)} scores."
            sample_priority = -scores[0]  # Lower score == Less anomalous == Better == Should have higher priority
            self._scores_cache.add_item(sample_priority, batched_features)
            return

        assert len(scores) == batch_size, f"batch_size is {batch_size}, but got {len(scores)} scores."
        for sample_index in range(batch_size):
            sample_score = scores[sample_index]
            sample_priority = -sample_score  # Lower score == Less anomalous == Better == Should have higher priority
            sample_feature = torch_utils.copy_tensor(batched_features[:, sample_index].unsqueeze(1))
            self._scores_cache.add_item(sample_priority, sample_feature)

    def get_best_features(self) -> List[torch.Tensor]:
        best_features = self._scores_cache.get_items()
        return best_features
