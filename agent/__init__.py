import importlib
from .Agent import Agent
from .RNNAgent import RNNAgent

def FromString(class_name: str) -> type:
    """
    Convert the string to the agent class.

    Parameters
    ----------
    class_name : str, optional
        The agent class. Default is None.

    Returns
    -------
    type
        The agent class.
    """

    module = importlib.import_module("." + class_name, package = "agent")
    return getattr(module, class_name)
