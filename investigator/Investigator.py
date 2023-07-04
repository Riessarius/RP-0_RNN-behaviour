from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from typing import Dict, List, Optional


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

    @property
    def info(self) -> List[Dict]:
        return self._info

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
        """
        for i, info in enumerate(self._info):
            agent_dir = save_dir / f"{self._agent_names[i]}_{self._name}"
            agent_dir.mkdir(parents = True, exist_ok = True)
            for k, v in info.items():
                with open(agent_dir / f"{k}.pkl", 'wb') as f:
                    pickle.dump(v, f)
