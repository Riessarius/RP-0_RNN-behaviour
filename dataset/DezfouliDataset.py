from collections.abc import Sized
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DezfouliDataset(Dataset, Sized):
    def __init__(self, src_path: str, mode: str = "prediction") -> None:
        super().__init__()
        self._src_path = src_path
        self._mode = mode
        self._original_data = self._load_original_data()  # type: pd.DataFrame
        self.input, self.output, self.mask = self._generate_data()  # type: torch.tensor

    def __len__(self) -> int:
        return self.input.shape[0]

    def __getitem__(self, idx: Union[int, List[int]]) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        return self.input[idx], self.output[idx], self.mask[idx]

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
        block_diagnosis_is_consistent = raw["diag"].apply(lambda x: len(pd.unique(x)) == 1).all()
        assert block_diagnosis_is_consistent, "Inconsistent diagnosis within a block."
        raw["diag"] = raw["diag"].apply(lambda x: x[0])
        subject_diagnosis_is_consistent = raw["diag"].groupby(["ID"]).agg(list).apply(lambda x: len(pd.unique(x)) == 1).all()
        assert subject_diagnosis_is_consistent, "Inconsistent diagnosis within a subject."

        max_len = raw["action"].apply(len).max()
        raw["mask"] = raw["action"].apply(lambda x: [1] * len(x) + [0] * (max_len - len(x)))
        raw["action"] = raw["action"].apply(lambda x: x + [0] * (max_len - len(x)))
        raw["reward"] = raw["reward"].apply(lambda x: x + [0] * (max_len - len(x)))

        return raw

    def _generate_data(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        action = torch.tensor(np.stack(self._original_data["action"].values))
        reward = torch.tensor(np.stack(self._original_data["reward"].values))
        mask = torch.tensor(np.stack(self._original_data["mask"].values))

        if self._mode == "prediction":
            input = torch.stack((action[:, :-1], reward[:, :-1]), dim = 2).to(dtype = torch.float32)  # type: torch.tensor[batch, seq, 2]
            output = action[:, 1:].to(dtype = torch.int64)  # type: torch.tensor[batch, seq]
            mask = mask[:, 1:].to(dtype = torch.bool)  # type: torch.tensor[batch, seq]
            return input, output, mask
        else:
            raise ValueError(f"Unexpected mode: {self._mode}")
