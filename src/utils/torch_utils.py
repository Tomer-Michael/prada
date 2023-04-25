# todo tomer go over this file
from typing import Iterable, List, Optional, Sequence

import torch


def get_generator(random_seed: Optional[int], device: Optional[torch.device]) -> Optional[torch.Generator]:
    if random_seed is None:
        return None
    return torch.Generator(device).manual_seed(random_seed)


def get_device(no_cuda: Optional[bool] = False, index: Optional[int] = None) -> torch.device:
    use_cpu = no_cuda or not is_cuda_available()
    if use_cpu:
        return torch.device("cpu")

    device_name = "cuda"
    if index is not None:
        assert index in range(torch.cuda.device_count())
        device_name += f":{index}"
    return torch.device(device_name)


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def copy_tensor(data: torch.Tensor) -> torch.Tensor:
    return data.detach().clone()


def choose_samples(data: torch.Tensor, indices: Sequence[int]) -> List[torch.Tensor]:
    """
    Takes a tensor of shape (num_channels, num_samples), and returns a tensor of shape (num_channels, len(indices)) with
    only the chosen samples included.
    """
    chosen_samples = list()
    for i in indices:
        sample = copy_tensor(data[:, i].unsqueeze(dim=1))
        chosen_samples.append(sample)
    return chosen_samples


def iter_samples(data: List[torch.Tensor]) -> Iterable[torch.Tensor]:
    """
    Takes a list of tensors of the shape (num_channels, num_samples), and returns an iterable that yields each sample in
    the data separately, as a 1D tensor in the shape (num_channels, ).
    """
    for x in data:
        num_samples = x.shape[1]
        for i in range(num_samples):
            yield x[:, i]


def cat_samples_on_cpu(data: List[torch.Tensor], permute_cols: bool = False) -> torch.Tensor:
    """
    Concatenates a list of tensors of the shape (num_channels, num_samples) along the samples dimension on the cpu.
    """
    return cat_samples_on_device(data, torch.device("cpu"), permute_cols)


def cat_samples_on_device(data: List[torch.Tensor], device: torch.device, permute_cols: bool = False) -> torch.Tensor:
    """
    Concatenates a list of tensors of the shape (num_channels, num_samples) along the samples dimension on the given
    device.
    """
    total_num_samples = sum(map(lambda it: it.shape[1], data))

    num_channels = data[0].shape[0]
    dtype = data[0].dtype
    layout = data[0].layout

    if permute_cols:
        shape = (total_num_samples, num_channels)
    else:
        shape = (num_channels, total_num_samples)
    out = torch.empty(shape, dtype=dtype, layout=layout, device=device, memory_format=torch.contiguous_format)

    start_index = 0
    for x in data:
        cur_num_samples = x.shape[1]
        x = x.to(device)
        end_index = start_index + cur_num_samples
        if permute_cols:
            x = x.t()
            out[start_index: end_index, :] = x
        else:
            out[:, start_index: end_index] = x
        start_index = end_index

    return out
