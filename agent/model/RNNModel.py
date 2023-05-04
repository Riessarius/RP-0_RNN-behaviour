from typing import Any, Tuple

import torch
from torch import nn


class RNNModel(nn.Module):
    def __init__(self, rnn_type: str, input_dim: int, rnn_dim: int, output_dim: int) -> None:
        super().__init__()

        self._rnn_type, self._input_dim, self._hidden_dim, self._output_dim = rnn_type, input_dim, rnn_dim, output_dim
        self._rnn_output = self._final_rnn_state = None

        if self._rnn_type == "LSTM":
            self._rnn = nn.LSTM(input_dim, rnn_dim, batch_first = True)
        elif self._rnn_type == "GRU":
            self._rnn = nn.GRU(input_dim, rnn_dim, batch_first = True)
        else:
            raise NotImplementedError

        self._lin = nn.Linear(rnn_dim, output_dim)

    def __call__(self, x: torch.tensor, /) -> torch.tensor:
        return self.forward(x)

    def forward(self, x: torch.tensor, /) -> torch.tensor:
        r, s = self._rnn(x)
        o = self._lin(r)
        self._rnn_output, self._final_rnn_state = r, s
        return o

    def get_internal_state(self) -> Tuple[torch.tensor, Any]:
        internal_state = {
            "rnn_output": self._rnn_output,
            "final_rnn_state": self._final_rnn_state
        }
        return internal_state

    def reset(self) -> None:
        self._rnn_output = self._final_rnn_state = None
        self._rnn.reset_parameters()
        self._lin.reset_parameters()
