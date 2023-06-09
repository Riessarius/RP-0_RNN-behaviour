from typing import List, Optional

import torch
from torch.utils import data as torch_data

from agent import Agent
from utils import demask, structural_tensor_to_numpy as s_t2n
from .Investigator import Investigator


class RNNInvestigator(Investigator):
    """
    The RNN investigator.

    This investigator investigates the _output and internal state of the agent.

    Attributes
    ----------
    _agent_name : str
        The name of the agent.
    _output : torch.tensor
        The _output of the agent.
    _internal_state : Any
        The internal state of the agent.

    Methods
    -------
    investigate(agent: Agent, dataset: Dataset, *args, **kwargs) -> None
        Investigate the agent.
    """
    def __init__(self, name: Optional[str] = None, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)

    def investigate(self, agents: List[Agent], dataset: torch_data.Dataset,
                    *args, **kwargs) -> None:
        """
        Investigate the agent.

        Parameters
        ----------
        agents : Agent
            The agents to investigate.
        dataset : torch_data.Dataset
            The datasets to investigate.

        Returns
        -------
        None
        """
        for ag in agents:
            self._agent_names.append(ag.get_name())

            mask = dataset[:][2].cpu().numpy()
            output = demask(s_t2n(ag.predict(dataset)), mask)
            internal_state = ag.get_internal_state()
            internal_state['rnn_output'] = demask(s_t2n(internal_state['rnn_output']), mask)
            internal_state['final_rnn_state'] = s_t2n(internal_state['final_rnn_state'])
            info = {
                'output': output,
                'internal_state': internal_state,
            }
            self._info.append(info)
