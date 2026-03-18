from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class LSTMAlphaModel(nn.Module):
    """LSTM-based alpha signal predictor.

    Input ``(batch, seq_len, n_features)`` → last-layer LSTM hidden state → MLP head.
    Output: alpha in ``[-1, 1]`` (Tanh).
    """

    def __init__(
        self,
        n_features: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

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
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])

    def hf_config(self) -> dict[str, Any]:
        """Fields needed to reconstruct the architecture (e.g. after download)."""
        return {
            "model_type": "lstm",
            "n_features": self.n_features,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    def upload_to_hf(
        self,
        repo_id: str,
        model_dir: str | Path,
        *,
        extra_config: dict[str, Any] | None = None,
        prefer_checkpoint: bool = True,
        force_safetensors: bool = False,
    ) -> None:
        """Write ``config.json`` + ``model.safetensors`` under ``model_dir`` and push to the Hub.

        Weights: ``model.pt`` in that folder if ``prefer_checkpoint`` and it exists; else this
        module's ``state_dict()`` (CPU). Requires ``pip install huggingface_hub safetensors``.

        Args:
            repo_id: ``username/repo`` on Hugging Face.
            model_dir: Directory containing optional ``model.pt`` / ``model.onnx``; files are
                written here then uploaded.
            extra_config: Merged into ``hf_config()`` (e.g. ``seq_len``, ``n_features`` from training).
            prefer_checkpoint: If True and ``model.pt`` exists, load weights from disk.
            force_safetensors: If True, always regenerate ``model.safetensors``.
        """
        try:
            from huggingface_hub import HfApi
            from safetensors.torch import save_file
        except ImportError as e:
            raise ImportError(
                "upload_to_hf needs huggingface_hub and safetensors. "
                "Install: uv add huggingface_hub safetensors"
            ) from e

        root = Path(model_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)

        cfg = {**self.hf_config(), **(extra_config or {})}
        (root / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        st_path = root / "model.safetensors"
        if force_safetensors or not st_path.exists():
            pt_path = root / "model.pt"
            if prefer_checkpoint and pt_path.is_file():
                state = torch.load(pt_path, map_location="cpu", weights_only=True)
            else:
                state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
            save_file(state, str(st_path))

        api = HfApi()
        api.create_repo(repo_id, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=str(root),
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["*.safetensors", "*.onnx", "*.json", "README.md"],
        )
        print(f"https://huggingface.co/{repo_id}")
