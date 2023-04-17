from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

class Dataset(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def subset(self, idx: Union[List[int], np.ndarray]) -> "Dataset":
        pass
