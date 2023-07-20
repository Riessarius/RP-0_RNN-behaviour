from abc import ABC, abstractmethod
import json, jsbeautifier
from pathlib import Path
from typing import Dict, List, Optional

import agent


class Trainer(ABC):
    """
    Base class for all trainers.

    Attributes
    ----------
    _name : Optional[str]
        The name of the trainer.
    _agents : List[Agent]
        The agents.
    _agent_configs : List[Dict]
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

    def __init__(self, config: Optional[Dict] = None, *args, **kwargs):
        self._config = config if config is not None else {}
        self._agents = []
        self._agent_configs = []
        pass

    @property
    def agents(self) -> List[agent.Agent]:
        return self._agents

    @property
    def agent_configs(self) -> List[Dict]:
        return self._agent_configs

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

    def load(self, load_dir: Path, *args, **kwargs):
        """
        Load the configuration and generated agents.

        Parameters
        ----------
        load_dir : Path
            The directory to load the configuration and generated agents.

        Returns
        -------
        None
        """

        for agent_dir in load_dir.iterdir():
            if agent_dir.is_dir():
                ag = agent.load(agent_dir)
                self._agents.append(ag)

                agcfg_path = load_dir / f"{ag.get_name()}_config.json"
                with agcfg_path.open('r') as f:
                    agcfg = json.load(f)
                self._agent_configs.append(agcfg)

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
        cfg_path = save_dir / f"config.json"
        with cfg_path.open('w') as f:
            opts = jsbeautifier.default_options()
            opts.indent_size = 4
            f.write(jsbeautifier.beautify(json.dumps(self._config), opts))

        for i, (ag, agcfg) in enumerate(zip(self._agents, self._agent_configs)):
            agent_dir = save_dir / ag.get_name()
            ag.save(agent_dir)

            agcfg_path = save_dir / f"{ag.get_name()}_config.json"
            with agcfg_path.open('w') as f:
                opts = jsbeautifier.default_options()
                opts.indent_size = 4
                f.write(jsbeautifier.beautify(json.dumps(agcfg), opts))
