from copy import deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils import data as torch_data

from dataset import Dataset


class DezfouliDataset(Dataset, torch_data.Dataset):
    """
    The dataset which loads data from Dezfouli et al., 2019.

    Attributes
    ----------
    _src_path : str
        The path to the source file.
    _mode : str
        The mode of the dataset.
    _original_data : pd.DataFrame
        The original data.
    _input : torch.Tensor
        The input of the dataset.
    _output : torch.Tensor
        The _output of the dataset.
    _mask : torch.Tensor
        The mask of the dataset.

    Methods
    -------
    __len__() -> int
        Get the length of the dataset.
    __getitem__(idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Get the item at the given index.
    subset(idx: Union[List[int], np.ndarray]) -> "DezfouliDataset"
        Get the subset of the dataset.
    """

    def __init__(self, src_path: str, mode: str = "prediction") -> None:
        super().__init__()
        self._src_path = src_path
        self._mode = mode
        self._original_data = self._load_original_data()  # type: pd.DataFrame
        self._input, self._output, self._mask, self._info = self._generate_data()  # type: torch.Tensor

    def __len__(self) -> int:
        return self._input.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        return self._input[idx], self._output[idx], self._mask[idx], {k: v[idx] for k, v in self._info.items()}

    def subset(self, idx: Union[List[int], np.ndarray]) -> "DezfouliDataset":
        """
        Get the subset of the dataset.

        Parameters
        ----------
        idx : Union[List[int], np.ndarray]
            The indices of the subset.

        Returns
        -------
        DezfouliDataset
            The subset of the dataset.
        """
        result = deepcopy(self)
        result._input = result._input[idx]
        result._output = result._output[idx]
        result._mask = result._mask[idx]
        result._info = {k: v[idx] for k, v in result._info.items()}
        return result

    def get_info_num_unique(self) -> Dict:
        """
        Get the number of unique values in each column of the info.

        Returns
        -------
        Dict
            The number of unique values in each column of the info.
        """
        num_unique = {k: len(np.unique(v)) for k, v in self._info.items()}
        return num_unique

    def _load_original_data(self) -> pd.DataFrame:
        raw = pd.read_csv(self._src_path).groupby(["ID", "block"]).agg(list)

        assert "key" in raw.columns
        raw.rename(columns = {"key": "action"}, inplace = True)

        action_values_are_expected = raw["action"].apply(lambda x: set(x) <= {"R1", "R2"}).all()
        assert action_values_are_expected, "Unexpected actions."
        raw["action"] = raw["action"].apply(lambda x: [int(y == "R2") for y in x])

        assert "best_action" in raw.columns
        raw.rename(columns = {"best_action": "is_best"}, inplace = True)

        block_best_is_consistent = raw.apply(lambda x: len(pd.unique(np.equal(x["action"], x["is_best"]))) == 1, axis = 1).all()
        assert block_best_is_consistent, "Inconsistent indication of the best action within a block."

        raw["best_action"] = raw.apply(lambda x: int(x["action"][0] == x["is_best"][0]), axis = 1)
        raw.drop(columns = ["is_best"], inplace = True)

        assert "diag" in raw.columns
        raw.rename(columns = {"diag": "diagnosis"}, inplace = True)
        block_diagnosis_is_consistent = raw["diagnosis"].apply(lambda x: len(pd.unique(x)) == 1).all()
        assert block_diagnosis_is_consistent, "Inconsistent diagnosis within a block."
        raw["diagnosis"] = raw["diagnosis"].apply(lambda x: x[0])
        subject_diagnosis_is_consistent = raw["diagnosis"].groupby(["ID"]).agg(list).apply(lambda x: len(pd.unique(x)) == 1).all()
        assert subject_diagnosis_is_consistent, "Inconsistent diagnosis within a subject."

        raw["reward_count"] = raw["reward"].apply(lambda x: sum(x))
        raw["trial_count"] = raw["reward"].apply(len)

        max_len = raw["action"].apply(len).max()
        raw["_mask"] = raw["action"].apply(lambda x: [1] * len(x) + [0] * (max_len - len(x)))
        raw["action"] = raw["action"].apply(lambda x: x + [0] * (max_len - len(x)))
        raw["reward"] = raw["reward"].apply(lambda x: x + [0] * (max_len - len(x)))

        raw.reset_index(inplace = True)
        id_no_map = {id: no for no, id in enumerate(raw["ID"].unique())}
        raw["no"] = raw["ID"].apply(lambda x: id_no_map[x])
        block_no_map = {block: no for no, block in enumerate(raw["block"].unique())}
        raw["block_no"] = raw["block"].apply(lambda x: block_no_map[x])
        diag_no_map = {diag: no for no, diag in enumerate(raw["diagnosis"].unique())}
        raw["diagnosis_no"] = raw["diagnosis"].apply(lambda x: diag_no_map[x])

        return raw

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        action = torch.tensor(np.stack(self._original_data["action"].values))
        reward = torch.tensor(np.stack(self._original_data["reward"].values))
        mask = torch.tensor(np.stack(self._original_data["_mask"].values))
        info = {
            "subject_id": self._original_data["ID"].values,
            "block": self._original_data["block"].values,
            "diagnosis": self._original_data["diagnosis"].values,
            "subject_no": torch.tensor(self._original_data["no"].values),
            "block_no": torch.tensor(self._original_data["block_no"].values),
            "diagnosis_no": torch.tensor(self._original_data["diagnosis_no"].values),
            "reward_count": torch.tensor(self._original_data["reward_count"].values),
            "trial_count": torch.tensor(self._original_data["trial_count"].values),
        }

        if self._mode == "prediction":
            input = torch.stack((action[:, :-1], reward[:, :-1]), dim = 2).to(dtype = torch.float32)  # type: torch.Tensor[batch, seq, 2]
            output = action[:, 1:].to(dtype = torch.int64)  # type: torch.Tensor[batch, seq]
            mask = mask[:, 1:].to(dtype = torch.bool)  # type: torch.Tensor[batch, seq]
            return input, output, mask, info
        else:
            raise ValueError(f"Unexpected mode: {self._mode}")
