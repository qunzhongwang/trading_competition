from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAlphaModel(nn.Module):
    """LSTM-based alpha signal predictor.

    Architecture:
        Input: (batch, seq_len, n_features)
        → LSTM(n_features, hidden_size=128, num_layers=2, dropout=0.2)
        → Last hidden state
        → Linear(128, 32) → ReLU → Dropout(0.2) → Linear(32, 1) → Tanh
        Output: alpha score in [-1, 1]
    """

    def __init__(
        self,
        n_features: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, n_features) tensor

        Returns:
            (batch, 1) tensor with alpha scores in [-1, 1]
        """
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, (h_n, _) = self.lstm(x)
        # Take the last hidden state from the top layer
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.head(last_hidden)
