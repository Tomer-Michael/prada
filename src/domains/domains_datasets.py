import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from src.utils import file_utils

DOMAINS_DIR: str = os.path.join(file_utils.DATA_BASE_DIR, "domains/")
DOMAIN_FILE_NAME_FORMAT: str = os.path.join(DOMAINS_DIR, "{}.json")  # protein_id


@dataclass(frozen=True)
class ProteinDomain:
    name: str
    start: int
    end: int


def get_domains_for_proteins(protein_ids: List[str]) -> Dict[str, List[ProteinDomain]]:
    all_existing_domains = file_utils.get_file_names_in_dir(DOMAINS_DIR)
    existing_domains = sorted(set(protein_ids).intersection(set(all_existing_domains)))

    domains = dict()
    for protein_id in tqdm(existing_domains, desc="Reading domains"):
        protein_domains = load_protein_domains(protein_id)
        domains[protein_id] = protein_domains
    return domains


def save_protein_domains(protein_id: str, domains: List[ProteinDomain]) -> None:
    domain_file_name = DOMAIN_FILE_NAME_FORMAT.format(protein_id)

    domains_as_dicts = [asdict(domain) for domain in domains]
    with open(domain_file_name, "w") as json_file:
        json.dump(domains_as_dicts, json_file)


def load_protein_domains(protein_id: str) -> Optional[List[ProteinDomain]]:
    domain_file_name = DOMAIN_FILE_NAME_FORMAT.format(protein_id)

    with open(domain_file_name, "r") as json_file:
        domains_as_dicts = json.load(json_file)

    domains = [ProteinDomain(**kwarg) for kwarg in domains_as_dicts]
    return domains
