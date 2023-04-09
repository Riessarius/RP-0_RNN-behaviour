from abc import ABC, abstractmethod
from typing import Optional

import torch


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
        Predict the output of the agent.

    Notes
    -----
    This is an abstract class which must be implemented in the subclass.
    """

    def __init__(self, name: Optional[str] = None, *args, **kwargs):
        self._name = name
        pass

    def __call__(self, x: torch.tensor) -> torch.tensor:
        """
        Synonym for predict.

        Parameters
        ----------
        x : torch.tensor
            The input tensor.

        Returns
        -------
        torch.tensor
            The output tensor.
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

        return self._name

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
        Predict the output of the agent.

        Notes
        -----
        This is an abstract method which must be implemented in the subclass.

        Returns
        -------
        torch.tensor
            The output tensor.
        """

        pass

    @abstractmethod
    def save(self, path: str, *args, **kwargs) -> None:
        """
        Save the agent.

        Notes
        -----
        This is an abstract method which must be implemented in the subclass.

        Parameters
        ----------
        path : str
            The path to save the agent.

        Returns
        -------
        None
        """

        pass

    @abstractmethod
    def load(self, path: str, *args, **kwargs) -> None:
        """
        Load the agent.

        Notes
        -----
        This is an abstract method which must be implemented in the subclass.

        Parameters
        ----------
        path : str
            The path to load the agent.

        Returns
        -------
        None
        """

        pass
