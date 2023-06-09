from typing import Any, Dict

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
        return {k: v[idx] for k, v in self._data.items()}

    def get_num_unique(self) -> Dict:
        """
        Get the number of unique values for each column.

        Returns
        -------
        Dict
            The number of unique values for each column.
        """
        num_unique = {k: len(np.unique(v)) for k, v in self._data.items()}
        return num_unique

    def _load_original_data(self) -> pd.DataFrame:
        raw = pd.read_csv(self._src_path).groupby(['ID', 'block']).agg(list)

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
        subject_diagnosis_is_consistent = raw['diagnosis'].groupby(['ID']).agg(list).apply(lambda x: len(pd.unique(x)) == 1).all()
        assert subject_diagnosis_is_consistent, f"Inconsistent diagnosis within a subject."

        raw['reward_count'] = raw['reward'].apply(lambda x: sum(x))
        raw['trial_count'] = raw['reward'].apply(len)

        max_len = raw['action'].apply(len).max()
        raw['mask'] = raw['action'].apply(lambda x: [1] * len(x) + [0] * (max_len - len(x)))
        raw['action'] = raw['action'].apply(lambda x: x + [0] * (max_len - len(x)))
        raw['reward'] = raw['reward'].apply(lambda x: x + [0] * (max_len - len(x)))

        raw.reset_index(inplace = True)
        sid_no_map = {sid: no for no, sid in enumerate(raw['ID'].unique())}
        raw['no'] = raw['ID'].apply(lambda x: sid_no_map[x])
        block_no_map = {block: no for no, block in enumerate(raw['block'].unique())}
        raw['block_no'] = raw['block'].apply(lambda x: block_no_map[x])
        diag_no_map = {diag: no for no, diag in enumerate(raw['diagnosis'].unique())}
        raw['diagnosis_no'] = raw['diagnosis'].apply(lambda x: diag_no_map[x])

        return raw

    def _generate_data(self) -> Dict:
        action = torch.tensor(np.stack(self._original_data['action'].values))
        reward = torch.tensor(np.stack(self._original_data['reward'].values))
        mask = torch.tensor(np.stack(self._original_data['mask'].values))

        data = {
            'input': torch.stack((action[:, :-1], reward[:, :-1]), dim = 2).to(dtype = torch.float32),  # type: torch.Tensor[batch, seq, 2]
            'output': action[:, 1:].to(dtype = torch.int64),  # type: torch.Tensor[batch, seq]
            'mask': mask[:, 1:].to(dtype = torch.bool),  # type: torch.Tensor[batch, seq]
            'subject_id': self._original_data['ID'].values,
            'block': self._original_data['block'].values,
            'diagnosis': self._original_data['diagnosis'].values,
            'subject_no': torch.tensor(self._original_data['no'].values),
            'block_no': torch.tensor(self._original_data['block_no'].values),
            'diagnosis_no': torch.tensor(self._original_data['diagnosis_no'].values),
            'reward_count': torch.tensor(self._original_data['reward_count'].values),
            'trial_count': torch.tensor(self._original_data['trial_count'].values),
        }
        return data
