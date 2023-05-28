from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

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
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def subset(self, idx: Union[List[int], np.ndarray]) -> "Dataset":
        """
        Get the subset of the dataset.

        Parameters
        ----------
        idx : Union[List[int], np.ndarray]
            The indices of the subset.

        Returns
        -------
        Dataset
            The subset of the dataset.
        """
        pass
