from typing import Any, Tuple, Union

import torch
from torch import nn


class RNNModel(nn.Module):
    def __init__(self, rnn_type: str,
                 input_dim: int, rnn_dim: int, output_dim: int,
                 num_embeddings: Tuple[int] = None, embedding_dims: Tuple[int] = None,
                 ) -> None:
        super().__init__()

        self._num_embeddings, self._embedding_dims = num_embeddings, embedding_dims
        if self._num_embeddings is not None and self._embedding_dims is not None:
            assert len(self._num_embeddings) == len(self._embedding_dims), "The list of number of embedding and embedding dimension must have the same length."
            embeddings = []
            for n, d in zip(self._num_embeddings, self._embedding_dims):
                embeddings.append(nn.Embedding(n, d, _weight = torch.zeros([n, d])))
            self._embeddings = nn.ModuleList(embeddings)
        else:
            self._embeddings, self._num_embeddings, self._embedding_dims = None, [], []

        self._rnn_type = rnn_type
        self._input_dim, self._hidden_dim, self._output_dim = input_dim, rnn_dim, output_dim
        self._rnn_output = self._final_rnn_state = None

        if self._rnn_type == "LSTM":
            self._rnn = nn.LSTM(input_dim + sum(self._embedding_dims), rnn_dim, batch_first = True)
        elif self._rnn_type == "GRU":
            self._rnn = nn.GRU(input_dim + sum(self._embedding_dims), rnn_dim, batch_first = True)
        else:
            raise NotImplementedError

        self._lin = nn.Linear(rnn_dim, output_dim)

    def __call__(self, x: Union[torch.Tensor, Tuple[torch.Tensor]], /) -> torch.tensor:
        return self.forward(x)

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor]], /) -> torch.tensor:
        if self._embeddings is not None:
            assert isinstance(x, tuple) and len(x) == 2, "The input with embeddings must be a tuple of two tensors."
            original_input, info = x[0], x[1]
            assert info.shape[1] == len(self._embeddings), "The number of embeddings must be the same as the number of embedding layers."
            embedded_info = torch.hstack([e(info[:, i]) for i, e in enumerate(self._embeddings)]).unsqueeze(1).expand(-1, original_input.shape[1], -1)
            x = torch.cat([original_input, embedded_info], dim = -1)
        else:
            assert isinstance(x, torch.Tensor), "The input without embeddings must be a tensor."
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
