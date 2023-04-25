# todo tomer go over this file - improve, maybe move to scripts or smthing
from io import StringIO
from typing import List, Optional

import requests
from Bio import SeqIO
from tqdm.auto import tqdm

from src.domains.domains_datasets import DOMAINS_DIR, ProteinDomain, save_protein_domains
from src.protein_datasets import read_proteins_dataset
from src.utils import file_utils

_UNIPROT_URL_FORMAT: str = "https://www.uniprot.org/uniprot/{}.xml"  # protein_id


def download_missing_domains(protein_ids: List[str]) -> None:
    print("STARTING")
    existing_domains = file_utils.get_file_names_in_dir(DOMAINS_DIR)
    missing_domains = sorted(set(protein_ids) - set(existing_domains))

    num_requested = len(protein_ids)
    num_existing_all = len(existing_domains)
    num_missing = len(missing_domains)
    num_existing_requested = num_requested - num_missing

    print(f"Requested to download domains for [{num_requested:,}] proteins. "
          f"[{num_existing_all:,}] already downloaded in total. "
          f"Of the already downloaded proteins, [{num_existing_requested:,}] are for the requested proteins. "
          f"Downloading domains for the [{num_missing:,}] missing proteins.")

    errors = set()

    for protein_id in tqdm(missing_domains):
        domains = get_domains_for_protein(protein_id)
        if domains is None:
            errors.add(protein_id)
            continue
        save_protein_domains(protein_id, domains)

    if errors:
        print(f"Couldn't download domains for {len(errors)} proteins:\n{errors}")
    else:
        print("Downloaded all domains with no errors.")
    print("DONE")


def get_domains_for_protein(protein_id: str) -> Optional[List[ProteinDomain]]:
    handle_to_data = query_uniprot_for_protein(protein_id)
    if handle_to_data is None:
        return None
    domains = parse_domains(handle_to_data)
    return domains


def query_uniprot_for_protein(protein_id: str) -> Optional[StringIO]:
    url = _UNIPROT_URL_FORMAT.format(protein_id)
    response = requests.get(url)
    if not response:
        print(f"Couldn't get uniprot data for protein [{protein_id}]. Status code is [{response.status_code}].")
        return None
    data = response.text
    return StringIO(data)


def parse_domains(handle_to_data: StringIO) -> Optional[List[ProteinDomain]]:
    try:
        record = SeqIO.read(handle_to_data, "uniprot-xml")
    except ValueError:
        return None

    domains = list()
    for feature in record.features:
        if feature.type == "domain":
            domain_name = feature.qualifiers["description"]
            domain_start = feature.location.nofuzzy_start
            domain_end = feature.location.nofuzzy_end
            domain = ProteinDomain(domain_name, domain_start, domain_end)
            domains.append(domain)
    return domains


def _main() -> None:
    proteins = read_proteins_dataset("/cs/labs/yedid/tomermichael/proteins/data/mammals_dataset.csv", 1022)
    proteins_id = list(map(lambda prot: prot.id, proteins))
    download_missing_domains(proteins_id)


if __name__ == '__main__':
    _main()
