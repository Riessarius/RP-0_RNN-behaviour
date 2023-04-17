from pathlib import Path

import torch
from torch.utils import data as torch_data

from agent import Agent
from .Investigator import Investigator


class RNNInvestigator(Investigator):
    """
    The RNN investigator.

    This investigator investigates the output and internal state of the agent.

    Attributes
    ----------
    _agent_name : str
        The name of the agent.
    _output : torch.tensor
        The output of the agent.
    _internal_state : Any
        The internal state of the agent.

    Methods
    -------
    investigate(agent: Agent, dataset: Dataset, *args, **kwargs) -> None
        Investigate the agent.
    save(investigator_dir: str, *args, **kwargs) -> None
        Save the investigation.
    """
    def __init__(self) -> None:
        super().__init__()
        self._agent_name = None
        self._output = None
        self._internal_state = None
        pass

    def investigate(self, agent: Agent, dataset: torch_data.Dataset,
                    *args, **kwargs) -> None:
        """
        Investigate the agent.

        Parameters
        ----------
        agent : Agent
            The agent to investigate.
        dataset : torch_data.Dataset
            The dataset to investigate.

        Returns
        -------
        None
        """

        self._agent_name = agent.get_name()
        self._output = agent.predict(dataset)
        self._internal_state = agent.get_internal_state()

    def save(self, save_rdir: Path, *args, **kwargs) -> None:
        """
        Save the investigation.

        Parameters
        ----------
        save_rdir : Path
            The root directory to save the investigation.

        Returns
        -------
        None
        """

        investigator_dir = Path(save_rdir) / (self._agent_name + "_investigation")
        investigator_dir.mkdir(parents = True, exist_ok = True)
        torch.save(self._output, investigator_dir / "output.pt")
        torch.save(self._internal_state, investigator_dir / "internal_state.pt")
