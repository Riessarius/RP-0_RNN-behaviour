import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch


def to_rdir(name: str = 'RP#0_RNN-behaviour'):
    curr_path = Path.cwd()
    while len(curr_path.name) and curr_path.name != name:
        curr_path = curr_path.parent

    if len(curr_path.name):
        os.chdir(curr_path)
    else:
        raise ValueError(f"Failed to find the root directory of the project: {name}")


def structural_tensor_to_numpy(obj: Union[Dict, List, Tuple, torch.tensor]) -> Union[Dict, List, Tuple, np.ndarray]:
    """Convert all tensors (nested) in best_model_pass to numpy arrays."""
    if isinstance(obj, dict):
        obj_new = {k: structural_tensor_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        obj_new = [structural_tensor_to_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        obj_new = tuple(structural_tensor_to_numpy(v) for v in obj)
    elif isinstance(obj, torch.Tensor):
        obj_new = obj.detach().cpu().numpy()
        if obj_new.size == 1:  # transform 1-element array to scalar
            obj_new = obj_new.item()
    else:
        obj_new = obj
    return obj_new


def demask(array: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
    assert array.shape[:2] == mask.shape, f"The first 2 components of shape of the processed array {array.shape} does not match the shape of the mask {mask.shape}."

    return [array[i, mask[i].astype(bool)] for i in range(array.shape[0])]
