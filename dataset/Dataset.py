from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict


class Dataset(ABC):
    """
    Base class for all datasets.

    Methods
    -------
    __getitem__(idx: int) -> Any
        Get the item at the given index.
    __len__() -> int
        Get the length of the dataset.
    subset(idx: Union[List[int], np.ndarray]) -> "Dataset"
        Get the subset of the dataset.

    Notes
    -----
    This is an abstract class which must be implemented in the subclass.
    """

    def __init__(self, default_mode: str = "default", *args, **kwargs):
        self._default_mode = default_mode
        self._mode = default_mode
        self._full_data = {
            default_mode: None
        }
        pass

    @property
    def _data(self) -> Dict[str, Any]:
        return self._full_data[self._mode]

    @_data.setter
    def _data(self, data: Dict[str, Any]):
        self._full_data[self._mode] = data

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: Any) -> Any:
        pass

    def subset(self, idx: Any) -> "Dataset":
        """
        Get the subset of the dataset.

        Parameters
        ----------
        idx : Any
            The indices of the subset.

        Returns
        -------
        Dataset
            The subset of the dataset.
        """
        result = deepcopy(self)
        for mode in result._full_data.keys():
            result._full_data[mode] = {k: v[idx] for k, v in result._full_data[mode].items()}
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
        """
        if mode == self._default_mode:
            raise ValueError(f"Cannot remove default mode.")
        if mode not in self._full_data:
            raise ValueError(f"Mode not found.")
        del self._full_data[mode]
        if self._mode == mode:
            self._mode = self._default_mode
        return self
