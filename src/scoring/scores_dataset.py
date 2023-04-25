# todo tomer go over this file
from ast import literal_eval
from typing import Dict, Iterable, Tuple

_DELIMITER: str = ":"


def read_scores(path_to_scores: str) -> Dict[str, Tuple[float, ...]]:
    scores = dict()

    with open(path_to_scores, "r") as scores_file:
        line = scores_file.readline()
        while line:
            protein_id, protein_scores = _process_line(line)
            scores[protein_id] = protein_scores
            line = scores_file.readline()
    return scores


def _process_line(line: str) -> Tuple[str, Tuple[float, ...]]:
    line = line.rstrip()
    protein_id, protein_scores_str = line.split(_DELIMITER)
    protein_scores = literal_eval(protein_scores_str)
    return protein_id, protein_scores


def save_score(path_to_scores: str, protein_id: str, protein_scores: Tuple[float, ...]) -> None:
    scores = [(protein_id, protein_scores)]
    save_scores(path_to_scores, scores)


def save_scores(path_to_scores: str, scores: Iterable[Tuple[str, Tuple[float, ...]]]) -> None:
    with open(path_to_scores, "a") as scores_file:
        for protein_id, protein_scores in scores:
            scores_file.write(f"{protein_id}{_DELIMITER}{protein_scores}\n")
