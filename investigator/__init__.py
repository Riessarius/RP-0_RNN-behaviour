import importlib
import json
from pathlib import Path

from .Investigator import Investigator
from .RNNInvestigator import RNNInvestigator


def generate(class_name: str) -> type:
    """
    Convert the string to the investigator class.

    Parameters
    ----------
    class_name : str, optional
        The investigator class. Default is None.

    Returns
    -------
    type
        The investigator class.
    """

    module = importlib.import_module("." + class_name, package = "investigator")
    return getattr(module, class_name)


def load(load_dir: Path) -> Investigator:
    """
    Load an investigator.

    Parameters
    ----------
    load_dir : Path
        The directory to load the investigator.

    Returns
    -------
    Investigator
        The investigator.
    """

    cfg_path = load_dir / f"config.json"
    with cfg_path.open('r') as f:
        config = json.loads(f.read())

    iv = generate(config['type'])(**config)
    iv.load(load_dir = load_dir)
    return iv

