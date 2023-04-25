import sys
from dataclasses import dataclass, replace
from typing import Tuple

from simple_parsing import ArgumentParser, field

from src.features_extraction.models import SupportedModels
from src.other_methods import classifier
from src.utils.torch_utils import is_cuda_available

_DEFAULT_MODEL_TO_USE: SupportedModels = SupportedModels.ESM_1b
_DEFAULT_LAYERS: Tuple[int, ...] = (_DEFAULT_MODEL_TO_USE.number_of_layers - 1, )


@dataclass(frozen=True)
class RunConfig:
    model_to_use: SupportedModels = field(alias=["-model"], default=_DEFAULT_MODEL_TO_USE)
    """ The model to use. See 'src/features_extraction/models.py'. """

    layers: Tuple[int, ...] = field(alias=["-representation_layers", "-repr_layers"], default=_DEFAULT_LAYERS)
    """ The layers in the model to extract the representations from. """

    path_to_dataset: str = field(alias=["-dataset", "-ds"], default=TOMER)
    """ Path to the dataset. See 'src/protein_datasets.py' for information about the dataset's expected format. """

    num_splits: int = field(default=5)
    """ todo tomer """

    max_iter: int = field(default=150)
    """ todo tomer """

    num_jobs: int = field(default=2)
    """ todo tomer """

    solver: str = field(default="saga")
    """ todo tomer """


def main() -> None:
    try:
        print("*************     Starting.     *************")
        print(f"Command line arguments are:\n{sys.argv}")
        config = get_run_config()
        print(f"Run config is:\n{config}")
        classifier.run(config.model_to_use, config.layers, config.path_to_dataset, config.num_splits, config.max_iter,
                       config.num_jobs, config.solver)
    finally:
        print("*************        Done.        *************")


def get_run_config() -> RunConfig:
    parser = _create_args_parser()
    run_config = _parse_args(parser)
    return run_config


def _create_args_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Anomaly Detection Using Classifier")
    parser.add_arguments(RunConfig, dest="run_config")
    return parser


def _parse_args(parser: ArgumentParser) -> RunConfig:
    args = parser.parse_args()
    original_run_config: RunConfig = args.run_config
    run_config = _handle_cuda(original_run_config)
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


if __name__ == '__main__':
    main()
