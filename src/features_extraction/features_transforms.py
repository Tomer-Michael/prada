# todo tomer go over this file
import abc
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.utils import torch_utils


class FeaturesTransform(abc.ABC):
    """
    This class represents a lightweight function that works on a single protein without changing its size in the samples
    dimension.
    For heavier, more robust processing, that work on the entire protein set and\or alter the samples dimension, see FeaturesProcessor.
    """
    _NUM_DIMENSIONS: int = 2
    _CHANNELS_DIMENSION_INDEX: int = 0
    _NUM_SAMPLES_DIMENSION_INDEX: int = 1

    @abc.abstractmethod
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Applies the transform on the raw features from the initial features' extractor.
        Must maintain the number of samples.

         :param features: A tensor of the shape (num_channels, num_samples).

         :returns The transformed features, a tensor of the shape (new_num_channels, num_samples).
        """
        pass

    def apply_on_batch(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        result = list()
        for cur_features in features:
            transformed_features = self(cur_features)
            result.append(transformed_features)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"


class WindowsSet(FeaturesTransform):
    def __init__(self, max_sequence_len: int, num_windows: int, windows_size: int, stride_increments: int) -> None:
        self._max_sequence_len = max_sequence_len
        self._windows_size = windows_size
        self._num_windows = num_windows
        self._stride_increments = stride_increments
        self._all_left_indices, self._all_right_indices = \
            self._get_indices_left_and_right(max_sequence_len, num_windows, windows_size, stride_increments)
        self._longest_left_window = max(map(len, self._all_left_indices))
        self._longest_right_window = max(map(len, self._all_right_indices))
        self._total_windows_length = self._longest_left_window + 1 + self._longest_right_window  # +1 for itself

    @staticmethod
    def _get_indices_left_and_right(sequence_len: int, num_windows: int, windows_size: int,
                                    stride_increments: int) -> Tuple[List[List[int]], List[List[int]]]:
        # Create the indices range [1, sequence_len] and then subtract 1, since windowing pads with 0
        indices = torch.arange(1, sequence_len + 1, dtype=torch.float64) \
            .unsqueeze(FeaturesTransform._CHANNELS_DIMENSION_INDEX)
        windowed_indices = _multiple_windows(indices, num_windows, windows_size, stride_increments)
        windowed_indices = windowed_indices.to(torch.int) - 1
        all_left_indices, all_right_indices = WindowsSet._process_indices(windowed_indices)
        return all_left_indices, all_right_indices

    @staticmethod
    def _process_indices(indices: torch.Tensor) -> Tuple[List[List[int]], List[List[int]]]:
        num_samples = indices.shape[FeaturesTransform._NUM_SAMPLES_DIMENSION_INDEX]

        all_left_indices = list()
        all_right_indices = list()
        for i in range(num_samples):
            all_indices = indices[:, i].tolist()
            no_dups = set(all_indices)
            cur_left_indices = sorted(filter(lambda neighbor_ind: 0 <= neighbor_ind < i, no_dups))
            all_left_indices.append(cur_left_indices)
            cur_right_indices = sorted(filter(lambda neighbor_ind: i < neighbor_ind, no_dups))
            all_right_indices.append(cur_right_indices)

        return all_left_indices, all_right_indices

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        num_channels = features.shape[FeaturesTransform._CHANNELS_DIMENSION_INDEX]
        num_samples = features.shape[FeaturesTransform._NUM_SAMPLES_DIMENSION_INDEX]
        shape = (num_channels, self._total_windows_length, num_samples)
        windowed_features = torch.zeros(shape, dtype=features.dtype, layout=features.layout, device=features.device)

        all_left_indices = self._filter_out_of_range_indices(self._all_left_indices, num_samples)
        all_right_indices = self._filter_out_of_range_indices(self._all_right_indices, num_samples)

        for i in range(num_samples):
            self._fill_sample(features, i, all_left_indices[i], all_right_indices[i],
                              self._longest_left_window, windowed_features)
        windowed_features = windowed_features.reshape(-1, num_samples)
        return windowed_features

    @staticmethod
    def _filter_out_of_range_indices(all_indices: List[List[int]], max_index_exclusive: int) -> List[List[int]]:
        return [list(filter(lambda index: index < max_index_exclusive, cur_indices)) for cur_indices in all_indices]

    @staticmethod
    def _fill_sample(features: torch.Tensor, sample_index: int, left_indices: List[int], right_indices: List[int],
                     longest_left_window: int, windowed_features: torch.Tensor) -> None:
        if len(left_indices) > 0:
            left_start_inclusive = longest_left_window - len(left_indices)
            left_end_exclusive = longest_left_window
            windowed_features[:, left_start_inclusive: left_end_exclusive, sample_index] = features[:, left_indices]

        self_index = longest_left_window
        windowed_features[:, self_index, sample_index] = features[:, sample_index]

        if len(right_indices) > 0:
            right_start_inclusive = self_index + 1
            right_end_exclusive = right_start_inclusive + len(right_indices)
            windowed_features[:, right_start_inclusive: right_end_exclusive, sample_index] = features[:, right_indices]

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(" \
               f"windows_size={self._windows_size}" \
               f", num_windows={self._num_windows}" \
               f", stride_increments={self._stride_increments}" \
               f")"


class Projection(FeaturesTransform):
    def __init__(self, num_projections: int, random_seed: Optional[int]) -> None:
        self._num_projections = num_projections
        self._random_seed: Optional[int] = random_seed
        self._projections: Optional[torch.Tensor] = None

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if self._projections is None:
            self._projections = self.create_projections(features, self._num_projections, self._random_seed)
        features = features.unsqueeze(0)
        projected_features = F.conv1d(features, self._projections)
        projected_features = projected_features.squeeze()
        return projected_features

    @staticmethod
    def create_projections(features: torch.Tensor, num_projections: int, random_seed: Optional[int]) -> torch.Tensor:
        num_channels = features.shape[FeaturesTransform._CHANNELS_DIMENSION_INDEX]
        device = features.device
        generator = torch_utils.get_generator(random_seed, device)
        projections = torch.randn((num_projections, num_channels, 1), device=device, generator=generator)
        return projections

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(num_projections={self._num_projections})"


class ToDevice(FeaturesTransform):
    def __init__(self, device: torch.device) -> None:
        self._device = device

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data.to(self._device) \
            .contiguous()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(device={self._device})"


class Compose(FeaturesTransform):
    def __init__(self, transforms: Iterable[FeaturesTransform]) -> None:
        self._transforms = tuple(transforms)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        for transform in self._transforms:
            data = transform(data)
        return data

    def __repr__(self) -> str:
        if self._transforms:
            transforms_repr = ", ".join(map(repr, self._transforms))
        else:
            transforms_repr = "EMPTY"
        return f"{self.__class__.__qualname__}({transforms_repr})"


def _multiple_windows(features: torch.Tensor, num_windows: int, windows_size: int,
                      stride_increments: int) -> torch.Tensor:
    unsqueezed = features.unsqueeze(0).unsqueeze(3)
    windows = list()
    stride = 1
    for i in range(1, num_windows + 1):
        padding = (stride * windows_size - stride) // 2
        window = F.unfold(unsqueezed, kernel_size=(windows_size, 1), dilation=(stride, 1), padding=(padding, 0))
        window = window.squeeze()
        windows.append(window)
        stride += stride_increments
    windowed_features = torch.cat(windows, dim=FeaturesTransform._CHANNELS_DIMENSION_INDEX)
    return windowed_features


def create_default_features_transform(device: torch.device) -> FeaturesTransform:
    return create_features_transform(max_sequence_length=-1, num_windows=-1, windows_size=-1, stride_increments=-1,
                                     num_projections=-1, device=device, random_seed=None)


def create_features_transform(max_sequence_length: int, num_windows: int, windows_size: int, stride_increments: int,
                              num_projections: int, device: torch.device, random_seed: Optional[int]) -> FeaturesTransform:
    transforms = list()

    to_device_transform = ToDevice(device)
    transforms.append(to_device_transform)

    if _use_windows(max_sequence_length, num_windows, windows_size, stride_increments):
        window_transform = WindowsSet(max_sequence_length, num_windows, windows_size, stride_increments)
        transforms.append(window_transform)

    if _use_projections(num_projections):
        projections_transform = Projection(num_projections, random_seed)
        transforms.append(projections_transform)

    to_cpu_transform = ToDevice(torch.device("cpu"))
    transforms.append(to_cpu_transform)

    transform = Compose(transforms)
    return transform


def _use_windows(max_sequence_length: int, num_windows: int, windows_size: int, stride_increments: int) -> bool:
    return max_sequence_length > 0 and num_windows > 0 and windows_size > 0 and stride_increments > 0


def _use_projections(num_projections: int) -> bool:
    return num_projections > 0
