from functools import reduce
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from dataset import Dataset


class DezfouliDataset(Dataset):
    """
    The dataset which loads data from Dezfouli et al., 2019.

    Attributes
    ----------
    _src_path : str
        The path to the source file.
    _original_data : pd.DataFrame
        The original data.

    Methods
    -------
    __len__() -> int
        Get the length of the dataset.
    __getitem__(idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Get the item at the given index.
    """

    def __init__(self, src_path: str, default_mode: str = 'default', *args, **kwargs) -> None:
        super().__init__(default_mode, *args, **kwargs)
        self._src_path = src_path
        self._original_data = self._load_original_data()  # type: pd.DataFrame
        self._data = self._generate_data()

    def __len__(self) -> int:
        return next(iter(self._data.values())).shape[0]

    def __getitem__(self, idx: Any) -> Dict:
        if isinstance(idx, str):  # Return all values of the property.
            return self._data[idx]
        if isinstance(idx, dict):  # Return items based on the query dictionary (convert into indices).
            # Items whose values are contained in the corresponding list for all given keys are selected.
            idx = reduce(lambda x, y: x & y, [np.isin(self._data[k], v) for k, v in idx.items()])
        return {k: v[idx] for k, v in self._data.items()}  # Return items based on indices.

    def _load_original_data(self) -> pd.DataFrame:
        raw = pd.read_csv(self._src_path)

        assert 'ID' in raw.columns
        raw.rename(columns = {'ID': 'subject_id'}, inplace = True)

        raw = raw.groupby(['subject_id', 'block']).agg(list)

        assert 'key' in raw.columns
        raw.rename(columns = {'key': 'action'}, inplace = True)

        action_values_are_expected = raw['action'].apply(lambda x: set(x) <= {'R1', 'R2'}).all()
        assert action_values_are_expected, f"Unexpected actions."
        raw['action'] = raw['action'].apply(lambda x: [int(y == 'R2') for y in x])

        assert 'best_action' in raw.columns
        raw.rename(columns = {'best_action': 'is_best'}, inplace = True)

        block_best_is_consistent = raw.apply(lambda x: len(pd.unique(np.equal(x['action'], x['is_best']))) == 1, axis = 1).all()
        assert block_best_is_consistent, f"Inconsistent indication of the best action within a block."

        raw['best_action'] = raw.apply(lambda x: int(x['action'][0] == x['is_best'][0]), axis = 1)
        raw.drop(columns = ['is_best'], inplace = True)

        assert 'diag' in raw.columns
        raw.rename(columns = {'diag': 'diagnosis'}, inplace = True)
        block_diagnosis_is_consistent = raw['diagnosis'].apply(lambda x: len(pd.unique(x)) == 1).all()
        assert block_diagnosis_is_consistent, f"Inconsistent diagnosis within a block."
        raw['diagnosis'] = raw['diagnosis'].apply(lambda x: x[0])
        subject_diagnosis_is_consistent = raw['diagnosis'].groupby(['subject_id']).agg(list).apply(lambda x: len(pd.unique(x)) == 1).all()
        assert subject_diagnosis_is_consistent, f"Inconsistent diagnosis within a subject."

        raw['reward_count'] = raw['reward'].apply(lambda x: sum(x))
        raw['trial_count'] = raw['reward'].apply(len)

        max_len = raw['action'].apply(len).max()
        raw['mask'] = raw['action'].apply(lambda x: [1] * len(x) + [0] * (max_len - len(x)))
        raw['action'] = raw['action'].apply(lambda x: x + [0] * (max_len - len(x)))
        raw['reward'] = raw['reward'].apply(lambda x: x + [0] * (max_len - len(x)))

        raw.reset_index(inplace = True)
        sid_no_map = {sid: no for no, sid in enumerate(raw['subject_id'].unique())}
        raw['subject_no'] = raw['subject_id'].apply(lambda x: sid_no_map[x])
        block_no_map = {block: no for no, block in enumerate(raw['block'].unique())}
        raw['block_no'] = raw['block'].apply(lambda x: block_no_map[x])
        diag_no_map = {diag: no for no, diag in enumerate(raw['diagnosis'].unique())}
        raw['diagnosis_no'] = raw['diagnosis'].apply(lambda x: diag_no_map[x])

        return raw

    def _generate_data(self) -> Dict:
        action = torch.tensor(np.stack(self._original_data['action'].values))
        reward = torch.tensor(np.stack(self._original_data['reward'].values))
        mask = torch.tensor(np.stack(self._original_data['mask'].values))

        # All keys must be strings, all values must be either np.ndarray or torch.Tensor.
        data = {
            'input': torch.stack((action[:, :-1], reward[:, :-1]), dim = 2).to(dtype = torch.float32),  # type: torch.Tensor[batch, seq, 2]
            'output': action[:, 1:].to(dtype = torch.int64),  # type: torch.Tensor[batch, seq]
            'mask': mask[:, 1:].to(dtype = torch.bool),  # type: torch.Tensor[batch, seq]
            'subject_id': self._original_data['subject_id'].values,
            'block': self._original_data['block'].values,
            'diagnosis': self._original_data['diagnosis'].values,
            'subject_no': torch.tensor(self._original_data['subject_no'].values),
            'block_no': torch.tensor(self._original_data['block_no'].values),
            'diagnosis_no': torch.tensor(self._original_data['diagnosis_no'].values),
            'reward_count': torch.tensor(self._original_data['reward_count'].values),
            'trial_count': torch.tensor(self._original_data['trial_count'].values),
        }
        return data
