from typing import List, Optional

from agent import Agent
from dataset import Dataset
from .Investigator import Investigator


class RNNInvestigator(Investigator):
    """
    The RNN investigator.

    This investigator investigates the _output and internal state of the agent.

    Attributes
    ----------
    _name : Optional[str]
        The name of the investigator.
    _agent_names : List[str]
        The names of the agents.
    _info : List[Dict]
        The information of the agents.

    Methods
    -------
    investigate(agents: List[Agent], dataset: torch_data.Dataset, *args, **kwargs) -> None
        Investigate the agent.
    """
    def __init__(self, name: Optional[str] = None, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)

    def investigate(self, agents: List[Agent], dataset: Dataset,
                    *args, **kwargs) -> None:
        """
        Investigate the agent.

        Parameters
        ----------
        agents : Agent
            The agents to investigate.
        dataset : Dataset
            The datasets to investigate.

        Returns
        -------
        None
        """
        for ag in agents:
            self._agent_names.append(ag.get_name())
            mask = dataset.get_by_prop('mask')
            output = ag.predict(dataset)
            internal_state = ag.get_internal_state()
            info = {
                'mask': mask,
                'output': output,
                'internal_state': internal_state,
            }
            self._info.append(info)
