# todo tomer go over this file
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from src import protein_datasets
from src.domains import domains_datasets
from src.domains.domains_datasets import ProteinDomain
from src.scoring import scores_dataset

COLOR_FOR_EVEN: str = "blue"
COLOR_FOR_ODD: str = "red"
FIGURES_FILE_EXTENSION = "png"
FIGURES_PATH_FORMAT = "/cs/labs/yedid/tomermichael/proteins/figures/{}." + FIGURES_FILE_EXTENSION

"""
todo tomer fill this file -
1. get mean in domains and mean outside domains, mean per protein AND over all proteins n
2. verify graphs are correct 
"""

in_domains = list()
outside_domains = list()
ratios = list()


def run(path_to_dataset: str, path_to_scores: str) -> None:
    print("STARTING")

    print("Reading scores")
    id_to_scores = scores_dataset.read_scores(path_to_scores)
    print("Normalizing scores")
    id_to_scores = normalize_scores(id_to_scores)
    print("Reading domains")
    id_to_domains = domains_datasets.get_domains_for_proteins(list(id_to_scores.keys()))
    print("Reading labels")
    id_to_label = protein_datasets.read_labels(path_to_dataset)

    missing_domains = set()

    for protein_id, protein_scores in tqdm(id_to_scores.items(), desc="Analyzing proteins"):
        protein_domains = id_to_domains.get(protein_id, None)
        if protein_domains is None:
            missing_domains.add(protein_id)
        protein_label = id_to_label[protein_id]
        analyze_protein(protein_id, protein_scores, protein_domains, protein_label)

    if len(missing_domains) != 0:
        print(f"Missing domains for {len(missing_domains)} proteins:\n{missing_domains}")
    else:
        print("All proteins have domains.")

    print("DONE")


def normalize_scores(scores: Dict[str, Tuple[float, ...]]) -> Dict[str, np.ndarray]:
    max_score = max(max(protein_scores) for protein_scores in scores.values())
    min_score = min(min(protein_scores) for protein_scores in scores.values())

    normalization_factor = 100.0 / (max_score - min_score)

    normalized_scores = dict()
    for protein_id, protein_scores in scores.items():
        protein_scores_np = np.asarray(protein_scores)
        normalized_protein_scores = normalization_factor * (protein_scores_np - min_score)
        normalized_scores[protein_id] = normalized_protein_scores
    return normalized_scores


def analyze_protein(protein_id: str, protein_scores: np.ndarray, domains: Optional[List[ProteinDomain]],
                    protein_label: int) -> None:
    statistics = dict()
    statistics["all"] = get_statistics(protein_scores)
    statistics["all"]["label"] = protein_label
    statistics["all"]["num_domains"] = len(domains) if domains is not None else "N/A"

    if domains is not None and len(domains) != 0:
        domains_statistics = analyze_domains(protein_scores, domains)
        statistics.update(domains_statistics)

    if protein_label == 1:
        print(f"Protein {protein_id} statistics:\n{statistics}")
        plot_protein(protein_id, protein_scores, statistics, domains, dict())  # todo tomer last arg


def analyze_domains(protein_scores: np.ndarray, domains: List[ProteinDomain]) -> Dict[str, Any]:
    domains_statistics = dict()

    for i, domain in enumerate(domains):
        domain_scores = protein_scores[domain.start: domain.end + 1]
        domain_statistics = get_statistics(domain_scores)
        domains_statistics[f"domain_{i}"] = domain_statistics

    domains_mask = get_domains_mask(protein_scores.shape[0], domains)
    domains_only = protein_scores[domains_mask]
    domains_statistics["domains_only"] = get_statistics(domains_only)
    in_domains.append(domains_only.mean())
    not_domains_only = protein_scores[~domains_mask]
    if not_domains_only.shape[0] != 0:
        outside_domains.append(not_domains_only.mean())
        domains_statistics["not_domains_only"] = get_statistics(not_domains_only)
        ratios.append(domains_only.mean() / not_domains_only.mean())
    else:
        domains_statistics["not_domains_only"] = "N\A"

    # todo tomer maybe try looking at domains (and intra-domains?) as a whole - i.e. avg(domain.avg for domain in domains)

    return domains_statistics


def get_statistics(scores: np.ndarray) -> Dict[str, Any]:
    return {
        "length": scores.shape[0],
        "min": scores.min(),
        "max": scores.max(),
        "mean": scores.mean(),
        "median": np.median(scores),
        "std": scores.std(),
    }


def get_domains_mask(protein_length: int, domains: List[ProteinDomain]) -> np.ndarray:
    """
    Returns a mask for the domains (domain region = 1, non domain = 0)
    """
    mask = np.zeros(protein_length, dtype=int)
    for domain in domains:
        mask[domain.start: domain.end] = 1
    mask = mask.astype(bool)
    return mask


def plot_protein(protein_id: str, protein_scores: np.ndarray, protein_info: Dict[str, Any],
                 protein_domains: Optional[List[ProteinDomain]], global_statistics: Dict[str, Any]) -> None:
    plt.clf()

    fig, ax = plt.subplots()

    # fig.set_url(protein_info.url)  # todo tomer

    ax.set_title(f"Anomaly Score Per Amino Acid - {protein_id}")
    ax.set_xlabel("Amino Acid's Position")
    ax.set_ylabel("Amino Acid's Anomaly Score")
    ax.set_ylim(0, 105)

    x_axis = range(protein_scores.shape[0])
    y_axis = protein_scores
    ax.plot(x_axis, y_axis)

    if protein_domains is not None:
        mark_domains(ax, protein_domains)
    # write_info(ax, protein_info, protein_domains, global_statistics)

    save_and_show(fig, protein_id)


def mark_domains(ax: plt.Axes, protein_domains: List[ProteinDomain]) -> None:
    for i, domain in enumerate(protein_domains):
        color = COLOR_FOR_EVEN if i % 2 == 0 else COLOR_FOR_ODD
        ax.axvspan(domain.start, domain.end, color=color, alpha=0.5)


def write_info(ax: plt.Axes, protein_info: Dict[str, Any], protein_domains: Optional[List[ProteinDomain]],
               global_statistics: Dict[str, Any]) -> None:
    bbox = dict(boxstyle="square", facecolor="lavender", alpha=0.5)

    global_info_text = "Global Info:\n" + \
                       "\n".join(f"{key}:\t{value}" for key, value in global_statistics.items())  # todo tomer
    global_info_text = ""

    protein_info_text = "Protein Info:\n" + \
                        "\n".join(f"{key}:\t{value}" for key, value in protein_info.items())

    if protein_domains is None:
        protein_domains_txt = "Unavailable."
    elif len(protein_domains) == 0:
        protein_domains_txt = "There are none."
    else:
        protein_domains_txt = "Domains Info:\n" + \
                              "\n".join(f"{domain.start}-{domain.end}:\t{domain.name}" for domain in protein_domains)

    info_txt = f"{global_info_text}\n\n{protein_info_text}\n\n{protein_domains_txt}"

    ax.text(1.1, 1, info_txt, fontsize=10, transform=ax.transAxes, bbox=bbox, verticalalignment="top")


def main() -> None:
    """
    todo tomer
    /cs/labs/yedid/tomermichael/proteins/data/mammals_dataset.csv
    /cs/labs/yedid/tomermichael/proteins/results/results_1/run_id_6_mammals_dataset.results
    /cs/labs/yedid/tomermichael/proteins/results/results_1/run_id_6_mammals_dataset_no_pre_training.results

    """
    path_to_dataset = "/cs/labs/yedid/tomermichael/proteins/data/mammals_dataset.csv"
    path_to_scores = "/cs/labs/yedid/tomermichael/proteins/results/results_1/run_id_6_mammals_dataset.results"
    run(path_to_dataset, path_to_scores)
    # print(f"Ratios are {ratios}")
    print(f"{len(ratios)} proteins have both domains and non domains seg ments. "
          f"We computed the ratio between the mean score inside the domain and outside the domain, for each protein.")
    print(f"Mean ratio is {sum(ratios) / len(ratios)}")
    print(f"Median ratio is {np.median(np.asarray(ratios))}")

    print(f"Number of ratios above 1 is {len(filter_above(ratios, 1))}")
    print(f"Number of ratios above 1.5 is {len(filter_above(ratios, 1.5))}")
    print(f"Number of ratios above 2 is {len(filter_above(ratios, 2))}")
    print(f"Number of ratios above 3 is {len(filter_above(ratios, 3))}")
    plot_mean_scores()


def filter_above(nums: List[float], threshold: float) -> List[float]:
    return [num for num in nums if num > threshold]


def plot_mean_scores() -> None:
    plt.clf()
    sns.set_theme()
    fig, ax = plt.subplots()

    all_scores = in_domains + outside_domains
    labels = ["In Domains"] * len(in_domains) + ["Outside Domains"] * len(outside_domains)
    data = pd.DataFrame({"Scores": all_scores, "InDomain": labels})
    sns.kdeplot(data=data, x="Scores", ax=ax, hue="InDomain")

    ax.set_title(f"Mean Anomaly Score Inside Domains vs. Outside Domains")
    ax.set_xlabel("Mean Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 105)

    save_and_show(fig, "Mean Scores Distribution")


def save_and_show(fig: plt.Figure, name: str) -> None:
    figure_path = FIGURES_PATH_FORMAT.format(name)
    fig.savefig(figure_path, format=FIGURES_FILE_EXTENSION)
    fig.show()


if __name__ == '__main__':
    main()
