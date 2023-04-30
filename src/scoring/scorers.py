# todo tomer go over this file
import abc
from typing import List, Optional, Tuple

import faiss
import faiss.contrib.torch_utils
import torch

from src.utils import torch_utils

class Scorer(abc.ABC):
    _NUM_DIMENSIONS: int = 2
    _BATCH_DIMENSION_INDEX: int = 0
    _CHANNELS_DIMENSION_INDEX: int = 1

    @abc.abstractmethod
    def fit(self, features: List[torch.Tensor]) -> None:
        """
        Fits the scorer to the features.

        :param features: A list of tensors of the shape (num_channels, num_samples)
        """

    def __call__(self, features: torch.Tensor) -> Tuple[float, ...]:
        """
        Scores the features. Gets the features in the shape of (num_channels, num_samples).

        :param features: A tensor of the shape (num_channels, num_samples)
        """


class Scorer2(abc.ABC):
    _NUM_DIMENSIONS: int = 2
    _BATCH_DIMENSION_INDEX: int = 0
    _CHANNELS_DIMENSION_INDEX: int = 1

    def __init__(self) -> None:
        self._is_fitted: bool = False

    @abc.abstractmethod
    def fit(self, features: List[torch.Tensor]) -> None:
        """
        Fits the scorer to the features.

        :param features: A list of tensors of the shape (num_channels, num_samples)
        """
        print(f"Fitting scorer {self}")
        self.reset()
        self._fit(features)
        self._is_fitted = True

    def _fit(self, features: List[torch.Tensor]) -> None:
        return

    def reset(self) -> None:
        self._is_fitted = False

    def score(self, features: torch.Tensor) -> Tuple[float, ...]:  # todo tomer change to __call__
        """
        Scores the features. Gets the features in the shape of (num_channels, num_samples).

        :param features: A tensor of the shape (num_channels, num_samples)
        """
        assert self._is_fitted
        processed_features = self._process_features(features)
        scores = self._score(processed_features)
        list_or_single_value = scores.tolist()
        if isinstance(list_or_single_value, list):
            scores_as_tuple = tuple(list_or_single_value)
        else:
            scores_as_tuple = (list_or_single_value, )
        return scores_as_tuple

    def _process_features(self, features: torch.Tensor) -> torch.Tensor:
        assert len(features.shape) == self._NUM_DIMENSIONS
        # Transpose because we want to score each amino acid separately.
        processed_features = features.t() \
            .to(self._device) \
            .contiguous()
        return processed_features

    @abc.abstractmethod
    def _score(self, features: torch.Tensor) -> torch.Tensor:
        """
        The actual implementation of the scoring method. Gets the features in the proper (batch_size, num_channels) shape.

        :param features: A tensor of the shape (batch_size, num_channels)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"


class KNNScorer(Scorer):
    def __init__(self, device: torch.device, num_neighbors: int) -> None:
        super().__init__()
        self._device: torch.device = device
        self._num_neighbors: int = num_neighbors
        self._index: Optional[faiss.Index] = None

    def _fit(self, features: List[torch.Tensor]) -> None:
        data = torch_utils.cat_samples_on_device(features, self._device, permute_cols=True)
        self._index = self._create_index(data, self._device)

    @staticmethod
    def _create_index(data: torch.Tensor, device: torch.device) -> faiss.Index:
        num_channels = data.shape[KNNScorer._CHANNELS_DIMENSION_INDEX]
        use_gpu = device.type == "cuda"
        if use_gpu:
            res = faiss.StandardGpuResources()
            faiss_index = faiss.GpuIndexFlatL2(res, num_channels)
        else:
            faiss_index = faiss.IndexFlatL2(num_channels)
        faiss_index.add(data)
        return faiss_index

    def reset(self) -> None:
        super().reset()
        if self._index is not None:
            self._index.reset()
            self._index = None

    def _score(self, data: torch.Tensor) -> torch.Tensor:
        knn_distances, _ = self._index.search(data, self._num_neighbors)
        scores = knn_distances.sum(dim=1)
        return scores

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(device={self._device}, num_neighbors={self._num_neighbors})"


class NormSizeScorer(Scorer):
    def _score(self, data: torch.Tensor) -> torch.Tensor:
        scores = torch.linalg.norm(data, dim=self._CHANNELS_DIMENSION_INDEX)
        return scores


def create_scorer(device: torch.device, knn_num_neighbors: int) -> Scorer:
    use_knn = knn_num_neighbors > 0
    if use_knn:
        return KNNScorer(device, knn_num_neighbors)
    else:
        return NormSizeScorer()
