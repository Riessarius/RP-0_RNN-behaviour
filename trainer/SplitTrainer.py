from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import agent
from dataset import Dataset
from .Trainer import Trainer


class SplitTrainer(Trainer):
    """
    The split trainer.

    This trainer splits the dataset into a train and test set.

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
    get_agents(...) -> List[Agent]
        Get the agents.
    save(save_rdir: Path, *args, **kwargs) -> None
        Save the configuration and generated agents.
    """
    def __init__(self, name: Optional[str] = None, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)

    def train(self, dataset: Dataset, agent_model_config: Dict, agent_training_config: Dict,
              test_ratio: float = 0.2, shuffle: bool = True, random_state: Optional[int] = None,
              verbose_level: int = 0, tensorboard_rdir: Optional[Path] = None, *args, **kwargs) -> None:
        """
        Train the agent using train-test split.

        Parameters
        ----------
        dataset : Dataset
            The dataset to use.
        agent_model_config : Dict
            The agent model configuration.
        agent_training_config : Dict
            The agent training configuration.
        test_ratio : float, optional
            The ratio of the test set. Default is 0.2.
        shuffle : bool, optional
            Whether to shuffle the data. Default is True.
        random_state : int, optional
            The random state. Default is None.
        verbose_level : int, optional
            The verbose level. Default is 0.
        tensorboard_rdir : Path, optional
            The root directory of the tensorboard. Default is None.

        Returns
        -------
        None
        """

        common_config = {
            'test_ratio': test_ratio,
            'verbose_level': verbose_level,
        }

        train_dataset = deepcopy(dataset.set_mode('train'))
        test_dataset = deepcopy(dataset.set_mode('test'))

        train_indices, test_indices = train_test_split(range(len(dataset)), test_size = test_ratio, shuffle = shuffle, random_state = random_state)

        if verbose_level >= 1:
            print(f"Random split: Train size: {len(train_indices)}; Test size: {len(test_indices)}.")

        agent_model_config['args']['name'] = f"{agent_model_config['common_name']}_{self._name}"
        agent_model_config['args']['tensorboard_rdir'] = tensorboard_rdir
        ag = agent.FromString(agent_model_config['class'])(**agent_model_config['args'])
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)
        ag.train(train_subset, test_subset, **agent_training_config)
        self._agents.append(ag)

        cfg = common_config | {
            'train_indices': train_indices,
            'test_indices': test_indices,
        }
        self._configs.append(cfg)
