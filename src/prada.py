# todo tomer go over this file
from typing import List, Optional, Tuple

import torch
from tqdm.auto import tqdm

from src.features_extraction.features_extractors import create_features_extractor, FeaturesExtractor
from src.features_extraction.features_processors import create_features_processor, FeaturesProcessor
from src.features_extraction.features_transforms import create_features_transform
from src.features_extraction.models import SupportedModels
from src.protein_datasets import Protein, read_proteins_dataset, subsample_pre_train, train_test_split
from src.scoring import analyze_scores, scores_dataset
from src.scoring.scorers import create_scorer, Scorer
from src.utils import file_utils, python_utils, random_utils, torch_utils
from src.utils.scores_tracker import ScoresTracker


def run(model_to_use: SupportedModels, layers: Tuple[int, ...],
        path_to_dataset: str, train_portion: float, contamination_portion: float, num_samples_evaluation: int,
        num_windows: int, windows_size: int, num_projections: int, num_quantiles: int, use_zca: bool,
        use_sequence_mean: bool, knn_num_neighbors: int, output_path: str, random_seed: Optional[int],
        stride_increments: int = 1, num_samples_evaluation_pretrain: Optional[int] = None,
        no_subsampling: bool = False, no_cuda: bool = False, no_cuda_for_extract: bool = False,
        no_cuda_for_transform: bool = False, no_cuda_for_scorer: bool = False,
        extract_device_index: Optional[int] = None, transform_device_index: Optional[int] = None,
        scorer_device_index: Optional[int] = None, is_quick_run: bool = False) -> None:
    file_utils.mkdir(output_path)

    if num_samples_evaluation_pretrain is None:
        num_samples_evaluation_pretrain = num_samples_evaluation

    extract_device = torch_utils.get_device(no_cuda=(no_cuda or no_cuda_for_extract), index=extract_device_index)
    transform_device = torch_utils.get_device(no_cuda=(no_cuda or no_cuda_for_transform), index=transform_device_index)
    scorer_device = torch_utils.get_device(no_cuda=(no_cuda or no_cuda_for_scorer), index=scorer_device_index)

    features_transform = create_features_transform(model_to_use.max_sequence_len, num_windows, windows_size,
                                                   stride_increments, num_projections, transform_device, random_seed)
    features_extractor = create_features_extractor(model_to_use, layers, extract_device, features_transform)
    features_processor = create_features_processor(num_quantiles, use_zca, use_sequence_mean)
    scorer = create_scorer(scorer_device, knn_num_neighbors)

    proteins = read_proteins_dataset(path_to_dataset, model_to_use.max_sequence_len)
    train_proteins, test_proteins = train_test_split(proteins, train_portion, contamination_portion, random_seed,
                                                     use_small_subset=is_quick_run)

    with torch.inference_mode():
        fit(features_extractor, features_processor, scorer, train_proteins, num_samples_evaluation,
            num_samples_evaluation_pretrain, no_subsampling, random_seed)
        predict(features_extractor, features_processor, scorer, test_proteins, output_path)

    print("DONE")


def fit(features_extractor: FeaturesExtractor, features_processor: FeaturesProcessor, scorer: Scorer,
        train_proteins: List[Protein], num_samples_evaluation: int, num_samples_evaluation_pretrain: int,
        no_subsampling: bool, random_seed: Optional[int]) -> None:
    if no_subsampling:
        print("Not subsampling the training set (because no_subsampling flag was passed).")
        fit_without_subsampling(features_extractor, features_processor, scorer, train_proteins, num_samples_evaluation,
                                random_seed)
        return

    pre_train_proteins, candidate_proteins = subsample_train(train_proteins, features_processor.is_sequence_level(),
                                                             num_samples_evaluation_pretrain, random_seed)

    if len(pre_train_proteins) == 0:
        print("Not subsampling (because there are not enough proteins).")
        fit_without_subsampling(features_extractor, features_processor, scorer, train_proteins, num_samples_evaluation,
                                random_seed)
        return

    print("Doing subsampling.")
    fit_with_subsampling(features_extractor, features_processor, scorer, pre_train_proteins, candidate_proteins,
                         num_samples_evaluation)


def fit_without_subsampling(features_extractor: FeaturesExtractor, features_processor: FeaturesProcessor,
                            scorer: Scorer, train_proteins: List[Protein], num_samples_evaluation: int,
                            random_seed: Optional[int]) -> None:
    weight_function = python_utils.get_weight_function(features_processor.is_sequence_level())
    candidate_proteins = random_utils.sample_2d(train_proteins, num_samples_evaluation, weight_function, random_seed)
    chosen_features = features_extractor.extract_batch(candidate_proteins, treat_errors_silently=True)
    processed_features = features_processor.fit(chosen_features)
    scorer.fit(processed_features)


def fit_with_subsampling(features_extractor: FeaturesExtractor, features_processor: FeaturesProcessor, scorer: Scorer,
                         pre_train_proteins: List[Protein], candidate_proteins: List[Protein],
                         num_samples_evaluation: int) -> None:
    pre_train_features = features_extractor.extract_batch(pre_train_proteins, treat_errors_silently=True)
    processed_pre_train_features = features_processor.fit(pre_train_features)
    scorer.fit(processed_pre_train_features)

    scores_tracker = ScoresTracker(num_samples_evaluation, features_processor.is_sequence_level())
    for protein in tqdm(candidate_proteins, desc="Evaluating train candidates"):
        initial_features = features_extractor.extract(protein, treat_errors_silently=True)
        if initial_features is None:
            continue
        processed_features = features_processor(initial_features)
        scores = scorer.score(processed_features)
        scores_tracker.register_score(initial_features, scores)

    best_candidates = scores_tracker.get_best_features()
    processed_best_candidates = features_processor.fit(best_candidates)
    scorer.fit(processed_best_candidates)


def predict(features_extractor: FeaturesExtractor, features_processor: FeaturesProcessor, scorer: Scorer,
            test_proteins: List[Protein], output_path: str) -> None:
    results_path = output_path + ".results"
    roc_auc_path = output_path + ".rocauc"
    id_to_scores = dict()
    id_to_label = dict()
    with tqdm(test_proteins, desc="Evaluating test proteins") as pbar:
        for protein in pbar:
            initial_features = features_extractor.extract(protein, treat_errors_silently=True)
            if initial_features is None:
                continue
            features = features_processor(initial_features)
            scores = scorer.score(features)
            scores_dataset.save_score(results_path, protein.id, scores)

            id_to_scores[protein.id] = scores
            id_to_label[protein.id] = protein.label

    print("*" * 10)  # todo tomer doesn't belong here
    roc_auc_score = analyze_scores.run_from_segmentations(id_to_scores, id_to_label)
    print(f"ROC AUC score is [{roc_auc_score}].")
    with open(roc_auc_path, "w") as roc_auc_file:
        roc_auc_file.write(f"ROC AUC score is [{roc_auc_score}].")

    print("*" * 10)  # todo tomer doesn't belong here
