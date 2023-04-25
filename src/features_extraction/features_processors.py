# todo tomer go over this file
import abc
from typing import Iterable, List, Optional

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from src.utils import torch_utils


class FeaturesProcessor(abc.ABC):
    """
    This class represents a robust processing of the entire set, that runs on the CPU.
    It is allowed to change the size of the samples dimension.
    For more light-weight processing, working on a single protein at a time, see FeaturesTransform.
    """
    _NUM_DIMENSIONS: int = 2
    _CHANNELS_DIMENSION_INDEX: int = 0
    _NUM_SAMPLES_DIMENSION_INDEX: int = 1

    def __init__(self) -> None:
        self._is_fitted: bool = False

    @abc.abstractmethod
    def is_sequence_level(self) -> bool:
        """
        Whether this processor produces sequence level features.
        """
        pass

    def fit(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Fits the processor to the data, and returns the processed data.

        :param data: A list of tensors of the shape (num_channels, num_samples) (num_samples could be different for each).
        """
        print(f"Fitting processor {self}")
        self.reset()
        self._is_fitted = True
        transformed_data = self._fit_and_process(data)
        return transformed_data

    def _fit_and_process(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.batch_process(data)

    def reset(self) -> None:
        self._is_fitted = False

    @abc.abstractmethod
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Applies the transform on the raw features from the initial features extractor.

         :param features: A tensor of the shape (num_channels, num_samples).

         :returns The transformed features, a tensor of the shape (new_num_channels, new_num_samples).
        """
        pass

    def batch_process(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        processed = list()
        print(f"Batch processing {len(data):,} samples with processor {self}.")
        for features in data:
            processed_features = self(features)
            processed.append(processed_features)
        print(f"Done batch processing with processor {self}")
        return processed

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"


class SequenceMean(FeaturesProcessor):
    def is_sequence_level(self) -> bool:
        return True

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        sequence_mean = torch.mean(features, dim=self._NUM_SAMPLES_DIMENSION_INDEX)
        sequence_mean = sequence_mean.unsqueeze(1).contiguous()
        return sequence_mean

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"


class CDFHistogram(FeaturesProcessor):
    def __init__(self, num_quantiles: int) -> None:
        super().__init__()
        self._num_quantiles = num_quantiles
        self._min_vals: Optional[torch.Tensor] = None
        self._max_vals: Optional[torch.Tensor] = None

    def is_sequence_level(self) -> bool:
        return True

    def _fit_and_process(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        for sample in torch_utils.iter_samples(data):
            if self._min_vals is None:
                self._min_vals = sample
                self._max_vals = sample
                continue
            self._min_vals = torch.minimum(self._min_vals, sample)
            self._max_vals = torch.maximum(self._max_vals, sample)

        return super()._fit_and_process(data)

    def reset(self) -> None:
        super().reset()
        self._min_vals = None
        self._max_vals = None

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        num_channels = features.shape[self._CHANNELS_DIMENSION_INDEX]

        cdf = torch.zeros((num_channels, self._num_quantiles))
        for quantile in range(self._num_quantiles):
            threshold = self._min_vals + (self._max_vals - self._min_vals) * (quantile + 1) / (self._num_quantiles + 1)
            threshold = threshold.unsqueeze(1)
            cdf[:, quantile] = (features < threshold).float().mean(dim=self._NUM_SAMPLES_DIMENSION_INDEX)

        radon_histogram = torch.flatten(cdf)
        radon_histogram = radon_histogram.unsqueeze(self._NUM_SAMPLES_DIMENSION_INDEX).contiguous()
        return radon_histogram

    def __repr__(self) -> str:
        return self.__class__.__qualname__ + f"(n_quantiles={self._num_quantiles})"


class ZCAWhitening(FeaturesProcessor):
    def __init__(self) -> None:
        super().__init__()
        self._mu: Optional[torch.Tensor] = None
        self._whitening_mat: Optional[torch.Tensor] = None

    def is_sequence_level(self) -> bool:
        return False

    def _fit_and_process(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        all_samples = torch_utils.cat_samples_on_cpu(data, permute_cols=True)
        num_samples_dimension = self._CHANNELS_DIMENSION_INDEX  # cpu_data is transposed
        self._mu = all_samples.mean(dim=num_samples_dimension).unsqueeze(self._NUM_SAMPLES_DIMENSION_INDEX)  # todo tomer why? and dim was 0

        self._whitening_mat = self._get_whitening_mat(all_samples)

        del all_samples
        return super()._fit_and_process(data)

    @staticmethod
    def _get_whitening_mat(data: torch.Tensor) -> torch.Tensor:
        """
        :param data: A tensor of the shape (num_samples, num_channels).
        """
        ledoit_wolf = LedoitWolf()
        np_data = data.numpy()
        ledoit_wolf.fit(np_data)

        cov = ledoit_wolf.covariance_
        u, s, vh = np.linalg.svd(cov, hermitian=True, full_matrices=True)
        D_norm = np.diag(1 / np.sqrt(s))
        W = np.matmul(D_norm, vh)
        whitening_mat = W.T

        whitening_mat = torch.from_numpy(whitening_mat)
        return whitening_mat

    def reset(self) -> None:
        super().reset()
        self._mu = None
        self._whitening_mat = None

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        centered_data = features - self._mu
        whitened_data = torch.matmul(centered_data, self._whitening_mat).contiguous()
        return whitened_data

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"


class Compose(FeaturesProcessor):
    def __init__(self, processors: Iterable[FeaturesProcessor]) -> None:
        super().__init__()
        self._processors = tuple(processors)

    def is_sequence_level(self) -> bool:
        for processor in self._processors:
            if processor.is_sequence_level():
                return True
        return False

    def _fit_and_process(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        for processor in self._processors:
            data = processor.fit(data)
        return data

    def reset(self) -> None:
        super().reset()
        for processor in self._processors:
            processor.reset()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        for processor in self._processors:
            data = processor(data)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"


def create_features_processor(num_quantiles: int, use_zca: bool, use_sequence_mean: bool) -> FeaturesProcessor:
    processors = list()

    if _use_cdf(num_quantiles):
        projections_processor = CDFHistogram(num_quantiles)
        processors.append(projections_processor)

    if use_zca:
        zca_processor = ZCAWhitening()
        processors.append(zca_processor)

    if use_sequence_mean:
        sequence_mean_processor = SequenceMean()
        processors.append(sequence_mean_processor)

    processor = Compose(processors)
    return processor


def _use_cdf(num_quantiles: int) -> bool:
    return num_quantiles > 0
