from abc import ABC
from copy import copy
from functools import reduce
from typing import Any, Dict, List, Optional

import numpy as np
import torch.utils.data as torch_data

IndexType = dict | int | list | slice | np.integer | np.ndarray


class Dataset(ABC, torch_data.Dataset):
    """
    Base class for all datasets.

    Methods
    -------
    __getitem__(idx: Any) -> Any
        Get the item at the given index.
    __len__() -> int
        Get the length of the dataset.

    Notes
    -----
    This is an abstract class which must be implemented in the subclass.
    """

    def __init__(self, default_mode: str = 'default', *args, **kwargs):
        self._default_mode = default_mode
        self._mode = default_mode
        self._sub_indices = None
        self._full_data = {
            default_mode: {}
        }
        pass

    @property
    def _data(self) -> Dict[str, Any]:
        return self._full_data[self._mode]

    @_data.setter
    def _data(self, data: Dict[str, Any]):
        assert isinstance(data, dict), f"Unexpected data type: {type(data)}, must be a dict."
        self._full_data[self._mode] = data

    @property
    def keys(self) -> List[str]:
        """
        Get the keys of the dataset.
        """
        return list(self._data.keys())

    @property
    def sub_indices(self) -> Optional[np.ndarray]:
        """
        Get the sub-indices of the dataset.
        """
        return self._sub_indices

    def __len__(self) -> int:
        return len(self._sub_indices) if self._sub_indices is not None else next(iter(self._data.values())).shape[0]

    def convert_index(self, idx: IndexType) -> np.ndarray:
        """
        Convert the given index into a np.ndarray with dtype of np.integer.

        Parameters
        ----------
        idx : IndexType
            The index to be converted.

        Returns
        -------
        np.ndarray
            The converted index.

        Raises
        ------
        ValueError
            If the given index is of an unexpected type.

        Notes
        -----
        This method is used to convert the index before indexing the dataset.
        """

        # Return indices of items based on the query dictionary.
        # Indices of items whose values are contained in the corresponding list for all given keys are selected.
        if isinstance(idx, dict):
            sub_indices = self._sub_indices if self._sub_indices is not None else np.arange(len(self))
            if len(idx) == 0:
                idx = np.arange(len(sub_indices))
            else:
                idx = reduce(lambda x, y: x & y, [np.isin(self._data[k][sub_indices], v) for k, v in idx.items()])
                idx = np.where(idx)[0]  # type = np.ndarray, dim = 1, dtype = np.int_
            assert isinstance(idx, np.ndarray) and len(idx.shape) == 1 and np.issubdtype(idx.dtype, np.integer), f"Assertion broken."
            return idx

        if isinstance(idx, slice):
            idx = np.arange(*idx.indices(len(self)))  # type = np.ndarray, dim = 1, dtype = np.int_
            assert isinstance(idx, np.ndarray) and len(idx.shape) == 1 and np.issubdtype(idx.dtype, np.integer), f"Assertion broken."
            return idx

        if isinstance(idx, int | np.integer):
            idx = np.int_(idx)  # type = np.int_
            return idx

        if isinstance(idx, list | np.ndarray):
            idx = np.array(idx).astype(int)  # type = np.ndarray, dim unknown, dtype = np.int_
            return idx

        raise ValueError(f"Unexpected index type: {type(idx)}, which cannot be transformed into np.ndarray.")

    def __getitem__(self, idx: IndexType) -> Dict:
        idx = self.convert_index(idx)
        if self._sub_indices is not None:
            idx = self._sub_indices[idx]
        return {k: v[idx] for k, v in self._data.items()}

    def __eq__(self, other: 'Dataset') -> bool:
        # Two datasets equal if they have the same keys and the same data, considered under their current modes and sub-indices.
        if not isinstance(other, Dataset) or self.keys != other.keys:
            return False
        else:
            self_data = self[:]
            other_data = other[:]
            return all([np.array_equal(self_data[k], other_data[k]) for k in self.keys])

    def get_by_prop(self, prop: str) -> np.ndarray:
        """
        Get the data of the given property.

        Parameters
        ----------
        prop : str
            The property of the data to be returned.

        Returns
        -------
        np.ndarray
            The data of the given property.
        """

        sub_indices = self._sub_indices if self._sub_indices is not None else np.arange(len(self))
        return self._data[prop][sub_indices]

    def subset(self, idx: IndexType) -> 'Dataset':
        """
        Get the subset of the dataset. This will create a shallow copy of the dataset that shares the same data.

        Parameters
        ----------
        idx : IndexType
            The indices of the subset.

        Returns
        -------
        Dataset
            The subset of the dataset.
        """
        result = copy(self)
        result._mode = copy(self._mode)
        idx = self.convert_index(idx)
        if len(idx.shape) == 0:
            idx = idx[np.newaxis]
        result._sub_indices = idx if self._sub_indices is None else self._sub_indices[idx]
        return result

    def set_mode(self, mode: str, as_default: bool = False):
        """
        Set the mode of the dataset.

        Parameters
        ----------
        mode : str
            The mode of the dataset.
        as_default : bool
            Whether to set the mode as the default mode.

        Returns
        -------
        Dataset
            The dataset.
        """
        self._mode = mode if mode in self._full_data else self._default_mode
        if as_default:
            self._default_mode = self._mode
        return self

    def add_mode(self, mode: str, data: Dict):
        """
        Add or replace a new mode to the dataset.

        Parameters
        ----------
        mode : str
            The mode to be added or replaced.
        data : Dict
            The data of the mode. All keys not existed will be filled with the default mode.

        Returns
        -------
        Dataset
            The dataset.
        """
        if mode == self._default_mode:
            raise ValueError(f"Cannot add mode with the same name as default mode.")
        self._full_data[mode] = data
        default_data = self._full_data[self._default_mode]
        for key in default_data:
            if key not in self._full_data[mode]:
                self._full_data[mode][key] = default_data[key]
        return self

    def remove_mode(self, mode: str):
        """
        Remove a mode from the dataset.

        Parameters
        ----------
        mode : str
            The mode to be removed.

        Returns
        -------
        Dataset
            The dataset.
        """
        if mode == self._default_mode:
            raise ValueError(f"Cannot remove default mode.")
        if mode not in self._full_data:
            raise ValueError(f"Mode not found.")
        del self._full_data[mode]
        if self._mode == mode:
            self._mode = self._default_mode
        return self
