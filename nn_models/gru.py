# ret_pred/nn_models/gru.py
from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


@register_model("gru")
class GRUModel(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)             # (N, L, H[*2])
        last = out[:, -1, :]             # (N, H[*2])
        y = self.head(last).squeeze(-1)  # (N,)
        return y
