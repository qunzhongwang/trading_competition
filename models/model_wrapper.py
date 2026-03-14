from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ModelWrapper:
    """Loads and serves a pre-trained model for alpha inference.

    Supports both ONNX (.onnx) and PyTorch (.pt) checkpoints.
    ONNX is the primary inference path (faster, no torch dependency in hot loop).
    PyTorch is the fallback.
    """

    def __init__(self, model_path: str, n_features: int = 6, use_compile: bool = False):
        self._model_path = model_path
        self._n_features = n_features
        self._use_compile = use_compile
        self._backend: Optional[str] = None  # "onnx" or "pytorch"
        self._session = None  # onnxruntime.InferenceSession
        self._torch_model = None  # LSTMAlphaModel

    def load(self) -> None:
        if self._model_path.endswith(".onnx"):
            self._load_onnx()
        elif self._model_path.endswith(".pt"):
            self._load_pytorch()
        else:
            raise ValueError(f"Unknown model format: {self._model_path}")

    def _load_onnx(self) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            self._model_path,
            providers=["CPUExecutionProvider"],
        )
        self._backend = "onnx"
        logger.info("Loaded ONNX model from %s", self._model_path)

    def _load_pytorch(self) -> None:
        import torch

        from models.lstm_model import LSTMAlphaModel

        model = LSTMAlphaModel(n_features=self._n_features)
        state_dict = torch.load(self._model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        if self._use_compile:
            try:
                model = torch.compile(model)
                logger.info("torch.compile enabled for inference")
            except Exception as e:
                logger.warning("torch.compile failed for inference: %s", e)

        self._torch_model = model
        self._backend = "pytorch"
        logger.info("Loaded PyTorch model from %s", self._model_path)

    def predict(self, feature_sequence: np.ndarray) -> float:
        """Run inference on a feature sequence.

        Args:
            feature_sequence: (seq_len, n_features) numpy array

        Returns:
            Alpha score in [-1.0, 1.0]
        """
        # Add batch dimension: (1, seq_len, n_features)
        x = feature_sequence[np.newaxis, :, :].astype(np.float32)

        if self._backend == "onnx":
            return self._predict_onnx(x)
        elif self._backend == "pytorch":
            return self._predict_pytorch(x)
        else:
            raise RuntimeError("Model not loaded. Call load() first.")

    def _predict_onnx(self, x: np.ndarray) -> float:
        input_name = self._session.get_inputs()[0].name
        result = self._session.run(None, {input_name: x})
        return float(result[0][0][0])

    def _predict_pytorch(self, x: np.ndarray) -> float:
        import torch

        tensor = torch.from_numpy(x)
        with torch.no_grad():
            output = self._torch_model(tensor)
        return float(output[0][0])

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None
