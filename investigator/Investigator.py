from abc import ABC, abstractmethod


class Investigator(ABC):
    """
    Base class for all investigators.

    Methods
    -------
    investigate(...) -> None
        Investigate the agent.
    save(...) -> None
        Save the investigation.

    Notes
    -----
    This is an abstract class which must be implemented in the subclass.
    """
    def __init__(self, *args, **kwargs) -> None:
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

    @abstractmethod
    def save(self, *args, **kwargs) -> None:
        """
        Save the investigation.

        Notes
        -----
        This is an abstract method which must be implemented in the subclass.
        """
        pass
