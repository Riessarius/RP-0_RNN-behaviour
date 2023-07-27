from abc import ABC, abstractmethod
import json, jsbeautifier
from pathlib import Path
import pickle
from typing import Dict, List, Optional

import agent


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
    def __init__(self, config: Optional[Dict] = None, *args, **kwargs) -> None:
        self._config = config if config is not None else {}
        self._agents = []
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

    def load(self, load_dir: Path, *args, **kwargs) -> None:
        """
        Load the investigation.

        Parameters
        ----------
        load_dir : Path
            The directory to load the investigation.
        """

        for agent_dir in load_dir.iterdir():
            if agent_dir.is_dir():
                ag = agent.load(agent_dir)
                self._agents.append(ag)
                info = {}
                for info_file in agent_dir.iterdir():
                    if info_file.is_file() and info_file.name.endswith(".pkl"):
                        with open(info_file, 'rb') as f:
                            info[info_file.stem] = pickle.load(f)
                self._info.append(info)

    def save(self, save_dir: Path, *args, **kwargs) -> None:
        """
        Save the investigation.

        Parameters
        ----------
        save_dir : Path
            The directory to save the investigation.
        """

        save_dir.mkdir(parents = True, exist_ok = True)
        cfg_path = save_dir / f"config.json"
        with cfg_path.open('w') as f:
            opts = jsbeautifier.default_options()
            opts.indent_size = 4
            f.write(jsbeautifier.beautify(json.dumps(self._config), opts))

        for i, (ag, info) in enumerate(zip(self._agents, self._info)):
            agent_dir = save_dir / f"{self._agents[i].name}_{self._config['name']}"
            agent_dir.mkdir(parents = True, exist_ok = True)
            ag.save(agent_dir)

            for k, v in info.items():
                with open(agent_dir / f"{k}.pkl", 'wb') as f:
                    pickle.dump(v, f)
