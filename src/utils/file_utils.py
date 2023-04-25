import os.path
import pathlib
from typing import List


DATA_BASE_DIR: str = "./data/"


def mkdir(path: str) -> None:
    # todo tomer find better method
    dir_name = os.path.dirname(path)
    dir_path = pathlib.Path(dir_name)
    dir_path.mkdir(exist_ok=True, parents=True)


def get_file_name(file_path: str) -> str:
    file_name = pathlib.Path(file_path).stem.split(".")[0]
    return file_name


def get_file_names_in_dir(path_to_dir: str) -> List[str]:
    # todo tomer find better method
    all_files = os.listdir(path_to_dir)
    file_names = sorted(filter(None, map(get_file_name, all_files)))
    return file_names
