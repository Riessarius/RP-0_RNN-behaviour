from abc import ABC, abstractmethod
import json
import jsbeautifier
from pathlib import Path
from typing import List, Optional

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
    _configs : List[dict]
        The configurations.

    Methods
    -------
    train(...) -> None
        Train the agent.
    get_agents(...) -> List[Agent]
        Get the agents.
    save(save_rdir: Path, *args, **kwargs) -> None
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

    def get_agents(self, *args, **kwargs) -> List[Agent]:
        """
        Get the agents.

        Returns
        -------
        List[Agent]
            The agents.
        """

        return self._agents

    def save(self, save_rdir: Path, *args, **kwargs):
        """
        Save the configuration and generated agents.

        Parameters
        ----------
        save_rdir : Path
            The root directory to save the configuration and generated agents.

        Returns
        -------
        None
        """

        trainer_dir = Path(save_rdir) / self._name
        trainer_dir.mkdir(parents = True, exist_ok = True)
        for i, (ag, cfg) in enumerate(zip(self._agents, self._configs)):
            ag.save(trainer_dir)
            config_path = trainer_dir / f"trainer_config_{ag.get_name()}.json"
            with config_path.open("w") as f:
                opts = jsbeautifier.default_options()
                opts.indent_size = 4
                f.write(jsbeautifier.beautify(json.dumps(cfg), opts))
