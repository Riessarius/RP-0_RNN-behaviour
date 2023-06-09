from abc import ABC, abstractmethod
import json
import jsbeautifier
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent import Agent


class Trainer(ABC):
    """
    Base class for all trainers.

    Attributes
    ----------
    _name : Optional[str]
        The name of the trainer.
    _agents : List[Agent]
        The agents.
    _configs : List[Dict]
        The configurations.

    Methods
    -------
    train(...) -> None
        Train the agent.
    save(save_dir: Path, *args, **kwargs) -> None
        Save the configuration and generated agents.

    Notes
    -----
    This is an abstract class which must be implemented in the subclass.
    """

    def __init__(self, name: Optional[str] = None, *args, **kwargs):
        self._name = name
        self._agents = []
        self._configs = []
        pass

    def __getitem__(self, idx: int) -> Tuple[Agent, Dict]:
        return self._agents[idx], self._configs[idx]

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        """
        Train the agent.

        Notes
        -----
        This is an abstract method which must be implemented in the subclass.

        Returns
        -------
        None
        """

        pass

    def save(self, save_dir: Path, *args, **kwargs):
        """
        Save the configuration and generated agents.

        Parameters
        ----------
        save_dir : Path
            The directory to save the configuration and generated agents.

        Returns
        -------
        None
        """

        save_dir.mkdir(parents = True, exist_ok = True)
        for i, (ag, cfg) in enumerate(zip(self._agents, self._configs)):
            agent_dir = save_dir / ag.get_name()
            ag.save(save_dir = agent_dir)

            cfg_path = save_dir / f"{ag.get_name()}_config.json"
            with cfg_path.open('w') as f:
                opts = jsbeautifier.default_options()
                opts.indent_size = 4
                f.write(jsbeautifier.beautify(json.dumps(cfg), opts))
