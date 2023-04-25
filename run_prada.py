# todo tomer go over this file
import sys
from dataclasses import dataclass, replace
from typing import Optional, Tuple

from simple_parsing import ArgumentParser, field

from src import prada
from src.features_extraction.models import SupportedModels
from src.utils.torch_utils import is_cuda_available

_DEFAULT_MODEL_TO_USE: SupportedModels = SupportedModels.ESM_1b
_DEFAULT_LAYERS: Tuple[int, ...] = (_DEFAULT_MODEL_TO_USE.number_of_layers - 1, )


@dataclass(frozen=True)
class RunConfig:
    model_to_use: SupportedModels = field(alias=["-model"], default=_DEFAULT_MODEL_TO_USE)
    """ The model to use. See 'src/features_extraction/models.py'. """

    layers: Tuple[int, ...] = field(alias=["-representation_layers", "-repr_layers"], default=_DEFAULT_LAYERS)
    """ The layers in the model to extract the representations from. """

    path_to_dataset: str = field(alias=["-dataset", "-ds"])
    """ Path to the dataset. See 'src/protein_datasets.py' for information about the dataset's expected format. """

    train_portion: float = field(alias=["-train_p"], default=0.5)
    """ Percentage of the normal proteins to use for training. Rest will be used for testing. """

    contamination_portion: float = field(alias=["-leak_p"], default=0.0)
    """ Percentage of the anomalous proteins to mislabel as normal (they will be added to the train dataset). """

    num_samples_evaluation: int = field(alias=["-num_samples", "-n_samples"], default=50_000)
    """ The amount of samples to use for evaluation.
    A 'sample' could be a feature vector of an amino-acid, in case we're in the segmentation task; or a feature vector
    of an entire protein, if we're in the detection task. 
    This will be use for evaluating the test samples, and also for evaluating the train candidates against the
    train evaluators in the features selection stage. """

    num_windows: int = field(alias=["-n_windows", "-num_win", "-n_win"], default=-1)
    """ Amount of windows in the spatial windowing of the features. """

    windows_size: int = field(alias=["-win_size"], default=-1)
    """ Size of each window in the spatial windowing of the features. """

    num_projections: int = field(alias=["-n_projections", "-num_proj", "-n_proj"], default=-1)
    """ Number of projections for Radon. """

    num_quantiles: int = field(alias=["-n_quantiles", "-num_quant", "-n_quant"], default=-1)
    """ Number of quantiles for Radon. """

    use_zca: bool = field(alias=["-zca"], default=False, action="store_true")
    """  """

    use_sequence_mean: bool = field(alias=["-sequence_mean", "-seq_mean", "-mean"], default=False)
    """  """

    knn_num_neighbors: int = field(alias=["-knn", "-KNN", "-kNN"], default=2)
    """ Number of neighbors for kNN. """

    output_path: str = field(alias=["-"], default="./results/")  # todo tomer
    """  """

    random_seed: Optional[int] = field(alias=["-seed"], default=None)
    """ For constant results between runs. """

    no_subsampling: bool = field(alias=["-"], default=False)  # todo tomer
    """  """

    no_cuda: bool = field(alias=["-"], default=False)  # todo tomer
    """ Do not use CUDA. """

    no_cuda_for_extract: bool = field(alias=["-extract_on_cpu"], default=False)
    """ Do not use CUDA for the initial features extraction stage. """

    no_cuda_for_scorer: bool = field(alias=["-score_on_cpu"], default=False)
    """ Do not use CUDA for the scoring stage. """

    extract_device_index: Optional[int] = field(alias=["-"], default=None)
    """  """  # todo tomer

    scorer_device_index: Optional[int] = field(alias=["-"], default=None)
    """  """  # todo tomer

    is_quick_run: bool = field(alias=["-"], default=False)  # todo tomer remove
    """ Should we subsample the train and . """


def main() -> None:
    try:
        print("*************     Starting.     *************")
        print(f"Command line arguments are:\n{sys.argv}")
        config = get_run_config()
        print(f"Run config is:\n{config}")
        prada.run(config.model_to_use, config.layers, config.path_to_dataset, config.train_portion,
                  config.contamination_portion, config.num_samples_evaluation, config.num_windows, config.windows_size,
                  config.num_projections, config.num_quantiles, config.use_zca, config.use_sequence_mean,
                  config.knn_num_neighbors, config.output_path, config.random_seed, config.stride_increments,
                  config.num_samples_evaluation_pretrain, config.no_subsampling, config.no_cuda,
                  config.no_cuda_for_extract, config.no_cuda_for_transform, config.no_cuda_for_scorer,
                  config.extract_device_index, config.transform_device_index, config.scorer_device_index,
                  config.is_quick_run)
    finally:
        print("*************        Done.        *************")


def get_run_config() -> RunConfig:
    parser = _create_args_parser()
    run_config = _parse_args(parser)
    return run_config


def _create_args_parser() -> ArgumentParser:
    parser = ArgumentParser(description="PRADA - Protein Anomaly Detection and Analysis")
    parser.add_arguments(RunConfig, dest="run_config")
    return parser


def _parse_args(parser: ArgumentParser) -> RunConfig:
    args = parser.parse_args()
    original_run_config: RunConfig = args.run_config
    run_config = _handle_cuda(original_run_config)
    validate_configuration(run_config)
    return run_config


def _handle_cuda(config: RunConfig) -> RunConfig:  # todo tomer also merge no_cuda with no_cuda_for_X, and device index
    did_request_cuda = not config.no_cuda
    can_use_cuda = is_cuda_available()

    if did_request_cuda and not can_use_cuda:
        print("CUDA is unavailable. Using CPU.")
    elif (not did_request_cuda) and can_use_cuda:
        print("CUDA is available, but you requested to use CPU.")

    should_use_cuda = did_request_cuda and can_use_cuda
    should_not_use_cuda = not should_use_cuda
    new_config = replace(config, no_cuda=should_not_use_cuda)
    return new_config


def validate_configuration(config: RunConfig) -> None:
    if config.num_windows == -1 or config.windows_size == -1:
        assert config.num_windows == config.windows_size, "We won't use windowing if either is -1"

    if config.windows_size != -1:
        assert config.windows_size != 0 or config.windows_size != 1, "Meaningless windows size"
        assert config.windows_size % 2 != 0, "Windows size should be odd (since it includes the middle element)"

    if config.num_projections == -1 or config.num_quantiles == -1:
        assert config.num_projections == config.num_quantiles, "We only consider projections with CDF"  # todo tomer why?


if __name__ == '__main__':
    main()
