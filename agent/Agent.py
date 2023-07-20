from abc import ABC, abstractmethod
import jsbeautifier, json
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

import torch

import agent


class Agent(ABC):
    """
    Base class for all agents.

    Attributes
    ----------
    _name : Optional[str]
        The name of the agent.

    Methods
    -------
    __call__(x: torch.tensor) -> torch.tensor
        Synonym for predict.
    get_name() -> Optional[str]
        Get the name of the agent.
    train(...) -> None
        Train the agent.
    get_internal_state(...) -> None
        Get the internal state of the agent.
    predict(...) -> torch.tensor
        Predict the _output of the agent.
    save(...) -> None
        Save the agent.

    Notes
    -----
    This is an abstract class which must be implemented in the subclass.
    """

    def __init__(self, config: Optional[Dict] = None, *args, **kwargs):
        self._config = config if config is not None else {}
        pass

    def __call__(self, x: Union[torch.tensor, Tuple[torch.Tensor]]) -> torch.tensor:
        """
        Synonym for predict.

        Parameters
        ----------
        x : torch.tensor
            The _input tensor.

        Returns
        -------
        torch.tensor
            The _output tensor.
        """

        return self.predict(x)

    def get_name(self) -> Optional[str]:
        """
        Get the name of the agent.

        Returns
        -------
        Optional[str]
            The name of the agent.
        """

        return self._config['name']

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

    @abstractmethod
    def get_internal_state(self, *args, **kwargs) -> None:
        """
        Get the internal state of the agent.

        Notes
        -----
        This is an abstract method which must be implemented in the subclass.

        Returns
        -------
        None
        """

        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> torch.tensor:
        """
        Predict the _output of the agent.

        Notes
        -----
        This is an abstract method which must be implemented in the subclass.

        Returns
        -------
        torch.tensor
            The _output tensor.
        """

        pass

    def load(self, load_dir: Path, *args, **kwargs) -> None:
        cfg_path = load_dir / f"config.json"
        with cfg_path.open('r') as f:
            self._config = json.load(f)

    def save(self, save_dir: Path, *args, **kwargs) -> None:
        save_dir.mkdir(parents = True, exist_ok = True)

        cfg_path = save_dir / f"config.json"
        with cfg_path.open('w') as f:
            opts = jsbeautifier.default_options()
            opts.indent_size = 4
            f.write(jsbeautifier.beautify(json.dumps(self._config), opts))
