import os
from pathlib import Path


def to_rdir(name: str = "RP#0_RNN-behaviour"):
    curr_path = Path.cwd()
    while len(curr_path.name) and curr_path.name != name:
        curr_path = curr_path.parent

    if len(curr_path.name):
        os.chdir(curr_path)
    else:
        raise ValueError(f"Failed to find the root directory of the project: {name}")
