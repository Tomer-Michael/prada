# todo tomer go over this file
import os
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score as _sk_roc_auc_score

from src import protein_datasets
from src.scoring import scores_dataset
from src.utils import np_utils


def run_for_multiple_segmentations_files(path_to_segmentations_dir: str, path_to_dataset: str) -> None:
    id_to_label = protein_datasets.read_labels(path_to_dataset)

    scores_files = sorted(map(lambda file_name: path_to_segmentations_dir + file_name, os.listdir(path_to_segmentations_dir)))
    print(f"There are {len(scores_files):,} scores files.")
    for i, scores_file in enumerate(scores_files):
        print(f"File {i:,}/{len(scores_files):,}: {scores_file}")
        id_to_scores = scores_dataset.read_scores(scores_file)
        print(f"Has {len(id_to_scores):,} scores.")
        roc_auc_score = run_from_segmentations(id_to_scores, id_to_label)
        print(f"Has ROC AUC score of {roc_auc_score}")


def run_from_segmentations_file(path_to_segmentations: str, path_to_dataset: str) -> None:
    id_to_scores = scores_dataset.read_scores(path_to_segmentations)
    id_to_label = protein_datasets.read_labels(path_to_dataset)
    roc_auc_score = run_from_segmentations(id_to_scores, id_to_label)
    print(f"ROC AUC score of {path_to_segmentations} is {roc_auc_score}")


def run_from_segmentations(id_to_segmentations: Dict[str, Tuple[float, ...]], id_to_label: Dict[str, int]) -> float:
    id_to_score = {protein_id: max(protein_segmentations)
                   for protein_id, protein_segmentations in id_to_segmentations.items()}
    return run(id_to_score, id_to_label)


def run(id_to_score: Dict[str, float], id_to_label: Dict[str, int]) -> float:
    scores, labels = _scores_as_array(id_to_score, id_to_label)
    return _sk_roc_auc_score(labels, scores)


def _scores_as_array(id_to_score: Dict[str, float], id_to_label: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    scores = list()
    labels = list()

    for protein_id in id_to_score.keys():
        protein_score = id_to_score[protein_id]
        scores.append(protein_score)

        protein_label = id_to_label.get(protein_id, 0)
        labels.append(protein_label)

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    return scores, labels


def enrichment_report_from_segmentations_file(path_to_segmentations: str, path_to_dataset: str) -> Dict[str, float]:
    report = dict()

    id_to_segmentations = scores_dataset.read_scores(path_to_segmentations)
    id_to_score = {protein_id: max(protein_segmentations)
                   for protein_id, protein_segmentations in id_to_segmentations.items()}
    id_to_label = protein_datasets.read_labels(path_to_dataset)

    full_dataset_size = len(id_to_label)
    num_anomalous = sum(id_to_label.values())
    report["full"] = num_anomalous / full_dataset_size

    scores, labels = _scores_as_array(id_to_score, id_to_label)

    report["test_set"] = _enrichment_report_top_k(labels, scores)

    size_of_1_percent = int(scores.size * 0.01)
    print(f"Size of 1 percent is {size_of_1_percent:,}")
    report["top_1%"] = _enrichment_report_top_k(labels, scores, size_of_1_percent)

    size_of_5_percent = int(scores.size * 0.05)
    report["top_5%"] = _enrichment_report_top_k(labels, scores, size_of_5_percent)

    size_of_10_percent = int(scores.size * 0.1)
    report["top_10%"] = _enrichment_report_top_k(labels, scores, size_of_10_percent)

    size_of_20_percent = int(scores.size * 0.2)
    report["top_20%"] = _enrichment_report_top_k(labels, scores, size_of_20_percent)

    print(f"Enrichment report: {report}")
    return report


def _enrichment_report_top_k(labels: np.ndarray, scores: np.ndarray, k: int = -1) -> float:
    if k > -1:
        top_k_scores_indices = np_utils.topk(scores, k)
        top_k_labels = labels[top_k_scores_indices]
    else:
        top_k_labels = labels

    return np.mean(top_k_labels)
