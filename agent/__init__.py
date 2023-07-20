import importlib
import json
from pathlib import Path

from .Agent import Agent
from .RNNAgent import RNNAgent


def generate(class_name: str) -> type:
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


def load(load_dir: Path) -> Agent:
    """
    Load an agent.

    Parameters
    ----------
    load_dir : Path
        The directory to load the agent.

    Returns
    -------
    Agent
        The agent.
    """

    cfg_path = load_dir / f"config.json"
    with cfg_path.open('r') as f:
        config = json.loads(f.read())

    ag = generate(config['type'])(**config)
    ag.load(load_dir = load_dir)
    return ag
