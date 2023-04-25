# todo tomer go over this file
from collections import defaultdict
from typing import Dict, List, Optional

from nltk.util import ngrams

from src.protein_datasets import Protein, read_proteins_dataset, train_test_split
from src.scoring import analyze_scores


def run(path_to_dataset: str, train_portion: float, ngram_length: int, num_ngram_to_score: int,
        random_seed: Optional[int]) -> float:
    proteins = read_proteins_dataset(path_to_dataset)
    train_proteins, test_proteins = train_test_split(proteins, train_portion, contamination_portion=0, random_seed=random_seed)

    print("Building ngram bank")
    ngram_bank = build_ngram_bank(train_proteins, ngram_length)

    print("Counting ngram")
    protein_id_to_ngram_count = count_ngrams(test_proteins, ngram_length, ngram_bank)

    print("Scoring proteins")
    protein_id_to_score = score_ngrams(protein_id_to_ngram_count, num_ngram_to_score)

    print("Calculating ROC AUC score")
    protein_id_to_label = {protein.id: protein.label for protein in test_proteins}
    roc_auc_score = analyze_scores.run(protein_id_to_score, protein_id_to_label)
    print(f"Got ROC AUC [{roc_auc_score}]")

    return roc_auc_score


def build_ngram_bank(proteins: List[Protein], ngram_length: int) -> Dict[str, int]:
    ngram_bank = defaultdict(lambda: 0)
    for protein in proteins:
        for ngram in ngrams(protein.sequence, ngram_length):
            ngram_bank[ngram] += 1
    return ngram_bank


def count_ngrams(proteins: List[Protein], ngram_length: int, ngram_bank: Dict[str, int]) -> Dict[str, List[int]]:
    protein_id_to_ngram_count = defaultdict(list)
    for protein in proteins:
        for ngram in ngrams(protein.sequence, ngram_length):
            ngram_count = ngram_bank[ngram]
            protein_id_to_ngram_count[protein.id].append(ngram_count)
    return protein_id_to_ngram_count


def score_ngrams(protein_id_to_ngram_count: Dict[str, List[int]], num_ngram_to_score: int) -> Dict[str, float]:
    scores = dict()
    for protein_id, protein_ngram_count in protein_id_to_ngram_count.items():
        protein_ngram_score = list(map(ngram_count_to_score, protein_ngram_count))
        windows = ngrams(protein_ngram_score, num_ngram_to_score)
        max_window_score = 0
        for window in windows:
            window_score = sum(window)
            max_window_score = max(max_window_score, window_score)
        scores[protein_id] = max_window_score
    return scores


def ngram_count_to_score(ngram_count: int) -> float:
    return 1.0 / (1.0 + ngram_count)
