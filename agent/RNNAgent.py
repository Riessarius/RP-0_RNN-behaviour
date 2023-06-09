import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

import torch
from torch import nn, optim
from torch.utils import data as torch_data
from torch.utils.tensorboard import SummaryWriter

from .Agent import Agent
from .model.RNNModel import RNNModel


class RNNAgent(Agent):
    """
    RNN agents.

    Attributes
    ----------
    _name : Optional[str]
        The name of the agent.
    _model : RNNModel
        The RNN model.
    _hyperparameters : Dict
        The hyperparameters.
    _tensorboard_rdir : Optional[Path]
        The root directory of the tensorboard.

    Methods
    -------
    train(...) -> None
        Train the agent.
    predict(...) -> Tuple[torch.Tensor, torch.Tensor]
        Predict the labels of the dataset.
    save(save_dir: Path, *args, **kwargs) -> None
        Save the agent.

    See Also
    --------
    Agent : The base class for all agents.
    RNNModel : The RNN model.

    Notes
    -----
    The hyperparameters are stored in the attribute _hyper_parameters.
    """

    def __init__(self, rnn_type: str, input_dim: int, rnn_dim: int, output_dim: int,
                 embedding_keys: Optional[List[str]] = None, num_embeddings: Optional[Tuple[int]] = None, embedding_dims: Optional[Tuple[int]] = None,
                 name: Optional[str] = None, tensorboard_rdir: Optional[Path] = None, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)
        self._tensorboard_rdir = tensorboard_rdir
        self._model = RNNModel(rnn_type, input_dim, rnn_dim, output_dim, num_embeddings, embedding_dims)
        self._hyperparameters = {
            "rnn_type": rnn_type,
            "input_dim": input_dim,
            "rnn_dim": rnn_dim,
            "output_dim": output_dim,
            "embedding_keys": embedding_keys,
            "num_embeddings": num_embeddings,
            "embedding_dims": embedding_dims,
        }

    def train(self, train_set: torch_data.Dataset, test_set: torch_data.Dataset,
              device: str = "cpu", batch_size: Optional[int] = None, max_num_epoch: int = 1000, early_stopping: Optional[int] = None,
              criterion: str = "CrossEntropy", optimizer: str = "SGD", lr: float = 0.01, weight_decay: float = 0.01,
              verbose_level: int = 0, *args, **kwargs) -> None:
        """
        Train the agent.
        The model with the least loss on the test set will be saved.

        Parameters
        ----------
        train_set : torch_data.Dataset
            The training set.
        test_set : torch_data.Dataset
            The test set.
        device : str, optional
            The device to use. Default is "cpu".
        batch_size : int, optional
            The batch size. Default is None.
        max_num_epoch : int
            The maximum number of epochs. Default is 1000.
        early_stopping : int, optional
            The number of epochs to wait before early stopping. Default is None.
        criterion : str, optional
            The criterion to use. Default is "CrossEntropy".
        optimizer : str, optional
            The optimizer to use. Default is "SGD".
        lr : float, optional
            The learning rate. Default is 0.01.
        weight_decay : float, optional
            The weight decay. Default is 0.01.
        verbose_level : int, optional
            The verbose level. Default is 0.

        Returns
        -------
        None

        Notes
        -----
        The criterion can be either "CrossEntropy".
        The optimizer can be either "SGD" or "AdamW".
        """

        self._hyperparameters |= {
            "device": device,
            "batch_size": batch_size,
            "max_epochs": max_num_epoch,
            "early_stopping": early_stopping,
            "criterion": criterion,
            "optimizer": optimizer,
            "lr": lr,
            "weight_decay": weight_decay,
            "verbose_level": verbose_level,
        }

        _device = self._hyperparameters["device"]
        _embedding_keys = self._hyperparameters["embedding_keys"]

        if criterion == "CrossEntropy":
            criterion = nn.CrossEntropyLoss(reduction = "none")
        else:
            raise NotImplementedError

        if optimizer == "SGD":
            optimizer = optim.SGD(self._model.parameters(), lr = lr, weight_decay = weight_decay)
        elif optimizer == "AdamW":
            optimizer = optim.AdamW(self._model.parameters(), lr = lr, weight_decay = weight_decay)
        else:
            raise NotImplementedError

        writer = SummaryWriter(str(self._tensorboard_rdir / self._name)) if self._tensorboard_rdir is not None and self._name is not None else None

        model = self._model.to(_device)
        best_loss = float("inf")
        early_stopping_counter = 0
        train_loader = torch_data.DataLoader(train_set, batch_size = batch_size, shuffle = False)
        test_loader = torch_data.DataLoader(test_set, batch_size = len(test_set), shuffle = False)

        for e in range(max_num_epoch):
            model.train()
            total_train_loss = 0.
            total_train_trials = 0
            with torch.enable_grad():
                for i, (x, y, m, info) in enumerate(train_loader):
                    x, y, m = x.to(_device), y.to(_device), m.to(_device)
                    if _embedding_keys is not None:
                        x = tuple([x, torch.stack([info[key] for key in self._hyperparameters["embedding_keys"]]).transpose(0, 1).to(_device)])
                    optimizer.zero_grad()
                    y_hat = model(x)
                    loss = (criterion(y_hat.flatten(end_dim = -2), y.flatten()) * m.flatten()).sum() / m.flatten().sum()
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item() * m.flatten().sum().item()
                    total_train_trials += m.flatten().sum().item()

                    if verbose_level >= 3:
                        print(f"Epoch {e} / {max_num_epoch}: Batch {i} / {len(train_loader)}: Train loss: {loss.item()}.")

            model.eval()
            total_test_loss = 0.
            total_test_trials = 0
            with torch.no_grad():
                for i, (x, y, m, info) in enumerate(test_loader):
                    x, y, m = x.to(_device), y.to(_device), m.to(_device)
                    if _embedding_keys is not None:
                        x = tuple([x, torch.stack([info[key] for key in self._hyperparameters["embedding_keys"]]).transpose(0, 1).to(_device)])
                    y_hat = model(x)
                    loss = (criterion(y_hat.reshape(-1, y_hat.shape[2]), y.flatten()) * m.flatten()).sum() / m.flatten().sum()
                    total_test_loss += loss.item() * m.flatten().sum().item()
                    total_test_trials += m.flatten().sum().item()

                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        early_stopping_counter = 0
                        self._model = deepcopy(model)
                    else:
                        early_stopping_counter += 1

                    if early_stopping is not None and early_stopping_counter >= early_stopping:
                        if verbose_level >= 1:
                            print(f"Early stopping after {e} epochs.")
                        return

                    if verbose_level >= 3:
                        print(f"Epoch {e} / {max_num_epoch}: Batch {i} / {len(test_loader)}: Test loss: {loss.item()}.")

            train_loss = total_train_loss / total_train_trials
            test_loss = total_test_loss / total_test_trials

            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, e)
                writer.add_scalar("Loss/test", test_loss, e)

            if verbose_level >= 2:
                print(f"Epoch {e} / {max_num_epoch}: Train loss: {train_loss}; Test loss: {test_loss}.")

        if verbose_level >= 1:
            print(f"Model trained and stored with the best test loss: {best_loss}.")

        if writer is not None:
            writer.close()

    def predict(self, pred_set: torch_data.Dataset, *args, **kwargs) -> torch.tensor:
        """
        Predict the _output of the agent.

        Parameters
        ----------
        pred_set : torch_data.Dataset
            The dataset to predict.

        Returns
        -------
        torch.tensor
            The _output of the agent.

        Notes
        -----
        The _output will be a tensor of shape (pred_size, seq_len, output_dim).
        """
        _device = self._hyperparameters["device"]
        _embedding_keys = self._hyperparameters["embedding_keys"]

        pred_loader = torch_data.DataLoader(pred_set, batch_size = len(pred_set), shuffle = False)
        self._model.eval()
        with torch.no_grad():
            x, _, m, info = next(iter(pred_loader))  # type: x: torch.Tensor[batch, seq, _input]; m: torch.Tensor[batch, seq]
            x, m = x.to(_device), m.to(_device)
            if _embedding_keys is not None:
                x = tuple([x, torch.stack([info[key] for key in self._hyperparameters["embedding_keys"]]).transpose(0, 1).to(_device)])
            y_hat = self._model(x)  # type: torch.Tensor[batch, seq, _output]
            y_hat = y_hat * m.unsqueeze(-1)
        return y_hat

    def get_internal_state(self, *args, **kwargs) -> Tuple[torch.tensor, Any]:
        """
        Get the internal state of the agent.

        Parameters
        ----------
        None

        Returns
        -------
        Tuple[torch.tensor, Any]
            The internal state of the agent.
        """

        return self._model.get_internal_state()

    def save(self, save_dir: Path, *args, **kwargs) -> None:
        """
        Save the agent, including model states and hyper-parameters.

        Parameters
        ----------
        save_dir : Path
            The directory to save the agent.

        Returns
        -------
        None

        Notes
        -----
        The hyperparameters will be saved as "hyperparameters.json".
        The model will be saved as "model.pt".
        """
        save_dir.mkdir(parents = True, exist_ok = True)

        hp_path = save_dir / "hyperparameters.json"
        with open(hp_path, "w") as f:
            json.dump(self._hyperparameters, f, indent = 4)

        model_path = save_dir / "model.pt"
        torch.save(self._model.state_dict(), model_path)
