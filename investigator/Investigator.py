from abc import ABC, abstractmethod
import os
from pathlib import Path
import pickle
from typing import Optional


class Investigator(ABC):
    """
    Base class for all investigators.

    Methods
    -------
    investigate(...) -> None
        Investigate the agent.
    save(save_dir: Path, *args, **kwargs) -> None
        Save the investigation.

    Notes
    -----
    This is an abstract class which must be implemented in the subclass.
    """
    def __init__(self, name: Optional[str] = None, *args, **kwargs) -> None:
        self._name = name
        self._agent_names = []
        self._info = []
        pass

    @abstractmethod
    def investigate(self, *args, **kwargs) -> None:
        """
        Investigate the agent.

        Notes
        -----
        This is an abstract method which must be implemented in the subclass.
        """
        pass

    def save(self, save_dir: Path, *args, **kwargs) -> None:
        """
        Save the investigation.

        Parameters
        ----------
        save_dir : Path
            The directory to save the investigation.

        Returns
        -------

        """
        save_dir.mkdir(parents = True, exist_ok = True)
        for i, info in enumerate(self._info):
            for k, v in info.items():
                pickle.dump(v, open(save_dir / f"{self._agent_names[i]}_{self._name}_{k}.pkl", "wb"))
