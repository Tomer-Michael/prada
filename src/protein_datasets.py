# todo tomer go over this file
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split as _sk_train_test_split

from src.utils import python_utils, random_utils

PROTEIN_ID_COL_NAME: str = "ID"
PROTEIN_SEQUENCE_COL_NAME: str = "Sequence"
PROTEIN_LABEL_COL_NAME: str = "Label"
PROTEIN_COLS: Tuple[str, ...] = (PROTEIN_ID_COL_NAME, PROTEIN_SEQUENCE_COL_NAME, PROTEIN_LABEL_COL_NAME)

NORMAL_LABEL: int = 0
ANOMALOUS_LABEL: int = 1
POSSIBLE_LABELS: Tuple[int, ...] = (NORMAL_LABEL, ANOMALOUS_LABEL)

_SMALL_SUBSET_SIZE: int = 100


@dataclass(frozen=True)
class Protein:
    id: str
    sequence: str
    label: int

    def __post_init__(self) -> None:
        assert isinstance(self.id, str)
        assert isinstance(self.sequence, str)
        assert isinstance(self.label, int)

        assert self.label in POSSIBLE_LABELS

    def is_normal(self) -> bool:
        return self.label == NORMAL_LABEL

    def is_anomalous(self) -> bool:
        return self.label == ANOMALOUS_LABEL

    def sequence_len(self) -> int:
        return len(self.sequence)

    def __len__(self) -> int:
        return self.sequence_len()


def read_labels(path_to_dataset: str) -> Dict[str, int]:
    proteins = _read_proteins_from_csv(path_to_dataset)
    labels = {protein.id: protein.label for protein in proteins}
    return labels


def read_proteins_dataset(path_to_dataset: str, max_sequence_len: int = -1) -> List[Protein]:
    proteins = _read_proteins_from_csv(path_to_dataset)
    if max_sequence_len > 0:
        proteins = _trim_sequences(proteins, max_sequence_len)
    proteins = _sort_proteins_in_place(proteins)  # Sort now, so that shuffling will only depend on the seed.
    return proteins


def _read_proteins_from_csv(path_to_csv: str) -> List[Protein]:
    dataframe = pd.read_csv(path_to_csv, usecols=PROTEIN_COLS)
    dataframe.dropna(inplace=True)

    proteins = list()
    for _, row in dataframe.iterrows():
        protein_id = row[PROTEIN_ID_COL_NAME]
        protein_sequence = row[PROTEIN_SEQUENCE_COL_NAME]
        protein_label = row[PROTEIN_LABEL_COL_NAME]
        protein_label = python_utils.convert_to_bool(protein_label)

        protein = Protein(protein_id, protein_sequence, protein_label)
        proteins.append(protein)
    return proteins


def _trim_sequences(proteins: List[Protein], max_sequence_len: int) -> List[Protein]:
    print(f"Trimming sequences to a maximal length of [{max_sequence_len:,}].")

    trimmed = list()
    for protein in proteins:
        trimmed_sequence = protein.sequence[:max_sequence_len]
        new_protein = replace(protein, sequence=trimmed_sequence)
        trimmed.append(new_protein)

    return trimmed


def _sort_proteins_in_place(proteins: List[Protein]) -> List[Protein]:
    proteins.sort(key=lambda protein: protein.id)
    return proteins


def train_test_split(proteins: List[Protein], train_portion: float, contamination_portion: float,
                     random_seed: Optional[int], use_small_subset: bool = False) -> Tuple[List[Protein], List[Protein]]:
    normal_proteins, anomalous_proteins = _normal_anomalous_split(proteins, contamination_portion, random_seed)

    if use_small_subset:
        normal_proteins = normal_proteins[: _SMALL_SUBSET_SIZE]
        anomalous_proteins = anomalous_proteins[: _SMALL_SUBSET_SIZE]
        print(f"Using a small subset of the dataset. "
              f"Taking only the first {_SMALL_SUBSET_SIZE} normal and anomalous proteins.")

    train_normal, test_normal = _sk_train_test_split(normal_proteins, train_size=train_portion,
                                                     random_state=random_seed)

    # Sort now, so that shuffling will only depend on the seed.
    train_proteins = _sort_proteins_in_place(train_normal)
    test_proteins = _sort_proteins_in_place(test_normal + anomalous_proteins)
    print(f"There are [{len(train_proteins):,}] train proteins and [{len(test_proteins):,}] test proteins.")
    print(f"Out of the test proteins, [{len(test_normal):,}] are normal and [{len(anomalous_proteins):,}] are anomalous.")
    return train_proteins, test_proteins


def _normal_anomalous_split(proteins: List[Protein], contamination_portion: float,
                            random_seed: Optional[int]) -> Tuple[List[Protein], List[Protein]]:
    normal_proteins = list()
    anomalous_proteins = list()

    for protein in proteins:
        if protein.is_normal():
            normal_proteins.append(protein)
        else:
            anomalous_proteins.append(protein)

    normal_proteins = _sort_proteins_in_place(normal_proteins)
    anomalous_proteins = _sort_proteins_in_place(anomalous_proteins)
    print(f"There are [{len(normal_proteins):,}] normal proteins.")
    print(f"There are [{len(anomalous_proteins):,}] anomalous proteins.")

    normal_proteins, anomalous_proteins = _contaminate_proteins(normal_proteins, anomalous_proteins,
                                                                contamination_portion, random_seed)
    return normal_proteins, anomalous_proteins


def _contaminate_proteins(normal_proteins: List[Protein], anomalous_proteins: List[Protein],
                          contamination_portion: float, random_seed: Optional[int]) -> Tuple[List[Protein], List[Protein]]:
    if contamination_portion == 0:
        return normal_proteins, anomalous_proteins

    if len(anomalous_proteins) == 0:
        print(f"Warning! Can't contaminate training set with anomalous proteins since there are no anomalous proteins.")
        return normal_proteins, anomalous_proteins

    contamination_size = _calculate_contamination_size(contamination_portion, num_normal_proteins=len(normal_proteins),
                                                       num_anomalous_proteins=len(anomalous_proteins))
    anomalous, fake_normal = _sk_train_test_split(anomalous_proteins, test_size=contamination_size, random_state=random_seed)
    fake_normal = _change_labels(fake_normal, new_label=NORMAL_LABEL)

    normal_proteins = _sort_proteins_in_place(normal_proteins + fake_normal)
    anomalous_proteins = _sort_proteins_in_place(anomalous)
    print(f"After mislabeling and contaminating the normal proteins with anomalous proteins, "
          f"there are [{len(normal_proteins):,}] normal proteins "
          f"and [{len(anomalous_proteins):,}] anomalous proteins.")
    return normal_proteins, anomalous_proteins


def _change_labels(proteins: List[Protein], new_label: int) -> List[Protein]:
    assert new_label in POSSIBLE_LABELS
    updated = list()
    for protein in proteins:
        new_protein = replace(protein, label=new_label)
        updated.append(new_protein)
    return updated


def _calculate_contamination_size(contamination_portion: float, num_normal_proteins: int,
                                  num_anomalous_proteins: int) -> int:
    portion_of_normal = contamination_portion / (1.0 - contamination_portion)
    contamination_size = int(num_normal_proteins * portion_of_normal)
    print(f"In order to have [{contamination_portion}%] of the normal proteins actually be anomalous proteins that we "
          f"mislabeled as normal, we need to contaminate the normal proteins with "
          f"[{portion_of_normal:.2%}] * [{num_normal_proteins:,}] = [{contamination_size:,}] "
          f"anomalous proteins.")

    if contamination_size >= num_anomalous_proteins:
        print(f"Warning! The contamination size [{contamination_size:,}] is larger than the number of anomalous "
              f"proteins [{num_anomalous_proteins:,}]. If we mislabel all the anomalous proteins, we'll have no "
              f"anomalous proteins left for the test set.")
        contamination_size = num_anomalous_proteins // 2
        portion_of_normal = contamination_size / (num_normal_proteins + contamination_size)
        print(f"We'll mislabel half of the anomalous proteins ([{contamination_size:,}] proteins), which will result "
              f"in having [{portion_of_normal:.2%}] of the normal proteins actually be mislabeled "
              f"anomalous proteins.")

    return contamination_size


def subsample_train(proteins: List[Protein], is_sequence_level: bool, num_samples_evaluation: int,
                    random_seed: Optional[int]) -> Tuple[List[Protein], List[Protein]]:
    weight_function = python_utils.get_weight_function(is_sequence_level)
    return random_utils.random_split(proteins, num_samples_evaluation, weight_function, random_seed)
