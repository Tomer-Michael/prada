import sys
from dataclasses import dataclass
from typing import Optional

from simple_parsing import ArgumentParser, field

from src.other_methods import ngrams


@dataclass(frozen=True)
class RunConfig:
    path_to_dataset: str = field(alias=["-dataset", "-ds"])
    """ Path to the dataset. See 'src/protein_datasets.py' for information about the dataset's expected format. """

    train_portion: float = field(alias=["-train_p"], default=0.5)
    """ Percentage of the normal proteins to use for training. Rest will be used for testing. """

    ngram_length: int = field(default=6)
    """ Length of each ngram. """

    num_ngram_to_score: int = field(default=20)
    """ How many ngrams to group together. """

    random_seed: Optional[int] = field(alias=["-seed"], default=None)
    """ For constant results between runs. """


def main() -> None:
    try:
        print("*************     Starting.     *************")
        print(f"Command line arguments are:\n{sys.argv}")
        config = get_run_config()
        print(f"Run config is:\n{config}")
        ngrams.run(config.path_to_dataset, config.train_portion, config.ngram_length, config.num_ngram_to_score,
                   config.random_seed)
    finally:
        print("*************        Done.        *************")


def get_run_config() -> RunConfig:
    parser = _create_args_parser()
    run_config = _parse_args(parser)
    return run_config


def _create_args_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Anomaly Detection Using N-Grams")
    parser.add_arguments(RunConfig, dest="run_config")
    return parser


def _parse_args(parser: ArgumentParser) -> RunConfig:
    args = parser.parse_args()
    run_config: RunConfig = args.run_config
    return run_config


if __name__ == '__main__':
    main()
