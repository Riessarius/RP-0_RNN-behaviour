from pathlib import Path
from typing import Dict, Optional

from sklearn.model_selection import KFold, StratifiedKFold

import agent
from dataset import Dataset
from .Trainer import Trainer


class CVTrainer(Trainer):
    """
    The cross validation trainer.

    This trainer splits the dataset into n_splits folds and trains the agent on each fold.

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
              split_config: Dict, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None,
              verbose_level: int = 0, tensorboard_rdir: Optional[Path] = None, *args, **kwargs) -> None:
        """
        Train the agent using cross validation.

        Parameters
        ----------
        dataset : Dataset
            The dataset to use.
        agent_model_config : Dict
            The agent model configuration.
        agent_training_config : Dict
            The agent training configuration.
        split_config : Dict
            The split configuration.
        n_splits : int, optional
            The number of splits. Default is 5.
        shuffle : bool, optional
            Whether to shuffle the data. Default is True.
        random_state : int, optional
            The random state. Default is None.
        verbose_level : int, optional
            The verbose level. Default is 0.
        tensorboard_rdir : Path, optional
            The tensorboard root directory. Default is None.

        Returns
        -------
        None
        """
        common_config = {
            'n_splits': n_splits,
            'verbose_level': verbose_level,
        }

        if split_config['mode'] == 'KFold':
            kf = KFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
            kfsp = kf.split(dataset)
        elif split_config['mode'] == 'StratifiedKFold':
            kf = StratifiedKFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
            kfsp = kf.split(dataset, dataset.get_by_prop(split_config['label']))
        else:
            raise NotImplementedError

        for f, (train_indices, test_indices) in enumerate(kfsp):
            if verbose_level >= 1:
                print(f"Cross validation - Fold {f}: Train size: {len(train_indices)}; Test size: {len(test_indices)}.")

            agent_model_config['args']['name'] = f"{agent_model_config['common_name']}_{self._name}_Fold{f}"
            agent_model_config['args']['tensorboard_rdir'] = tensorboard_rdir
            ag = agent.FromString(agent_model_config['class'])(**agent_model_config['args'])
            train_subset = dataset.subset(train_indices).set_mode('train')
            test_subset = dataset.subset(test_indices).set_mode('test')
            ag.train(train_subset, test_subset, **agent_training_config)
            self._agents.append(ag)

            cfg = common_config | {
                'train_indices': train_indices.tolist(),
                'test_indices': test_indices.tolist(),
            }
            self._configs.append(cfg)
