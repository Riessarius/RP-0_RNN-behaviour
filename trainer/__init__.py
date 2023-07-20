import importlib
import json
from pathlib import Path

from .Trainer import Trainer
from .CVTrainer import CVTrainer
from .SplitTrainer import SplitTrainer


def generate(class_name: str) -> type:
    """
    Convert the string to the trainer class.

    Parameters
    ----------
    class_name : str, optional
        The trainer class. Default is None.

    Returns
    -------
    type
        The trainer class.
    """

    module = importlib.import_module("." + class_name, package = "trainer")
    return getattr(module, class_name)


def load(load_dir: Path) -> Trainer:
    """
    Load a trainer.

    Parameters
    ----------
    load_dir : Path
        The directory to load the trainer.

    Returns
    -------
    Trainer
        The trainer.
    """

    cfg_path = load_dir / f"config.json"
    with cfg_path.open('r') as f:
        config = json.loads(f.read())

    tr = generate(config['type'])(**config)
    tr.load(load_dir = load_dir)
    return tr
