from __future__ import annotations

import math

import torch
import torch.nn as nn


class TransformerAlphaModel(nn.Module):
    """Transformer encoder-based alpha signal predictor.

    Architecture:
        Input: (batch, seq_len, n_features)
        → Linear(n_features, d_model) — feature projection
        → + Sinusoidal positional encoding
        → TransformerEncoder(num_layers, nhead, d_ff, dropout)
        → Mean pooling across sequence
        → Linear(d_model, 32) → ReLU → Dropout → Linear(32, 1) → Tanh
        Output: alpha score in [-1, 1]
    """

    def __init__(
        self,
        n_features: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 120,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Project input features to d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # Sinusoidal positional encoding (fixed, not learned)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
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
        seq_len = x.size(1)

        # Project features and add positional encoding
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :]

        # Transformer encoder
        x = self.encoder(x)  # (batch, seq_len, d_model)

        # Mean pooling across sequence
        x = x.mean(dim=1)  # (batch, d_model)

        return self.head(x)
