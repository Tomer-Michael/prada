# todo tomer go over this file
from typing import List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as sk_calc_roc_auc_score
from sklearn.model_selection import cross_val_predict
from tqdm.auto import tqdm

from src.features_extraction.features_extractors import create_features_extractor, FeaturesExtractor, SupportedModels
from src.features_extraction.features_processors import create_features_processor, FeaturesProcessor
from src.features_extraction.features_transforms import create_default_features_transform
from src.protein_datasets import Protein, read_proteins_dataset


def run(model_to_use: SupportedModels, layers: Tuple[int, ...], path_to_dataset: str, num_splits: int, max_iter: int,
        num_jobs: int, solver: str) -> None:
    extract_device = torch.device("cuda")  # todo tomer make arg
    features_transform = create_default_features_transform(device=extract_device)
    features_extractor = create_features_extractor(model_to_use, layers, extract_device, features_transform)
    features_processor = create_features_processor(num_quantiles=-1, use_zca=False, use_sequence_mean=True)
    proteins = read_proteins_dataset(path_to_dataset, max_sequence_len=model_to_use.max_sequence_len)
    with torch.inference_mode():
        features, labels = extract_features(proteins, features_extractor, features_processor)

    clf = LogisticRegression(max_iter=max_iter, solver=solver, n_jobs=num_jobs)
    print("Generating cross-validated estimates")
    cv_predictions = cross_val_predict(estimator=clf, X=features, y=labels, cv=num_splits, method="predict_proba")

    print("Calculating ROC AUC score")
    scores = cv_predictions[:, 1]
    roc_auc_score = sk_calc_roc_auc_score(y_true=labels, y_score=scores)
    print(f"Got ROC AUC [{roc_auc_score}]")

    print("DONE")
    return roc_auc_score


def extract_features(proteins: List[Protein], features_extractor: FeaturesExtractor,
                     features_processor: FeaturesProcessor) -> Tuple[np.ndarray, np.ndarray]:
    features = list()
    labels = list()
    with tqdm(proteins, desc="Extracting features") as pbar:
        for protein in pbar:
            initial_features = features_extractor.extract(protein, treat_errors_silently=True)
            if initial_features is None:
                continue
            protein_features = features_processor(initial_features).squeeze()
            features.append(protein_features)
            labels.append(protein.label)

    features = torch.stack(features).numpy()
    labels = np.asarray(labels)
    return features, labels
