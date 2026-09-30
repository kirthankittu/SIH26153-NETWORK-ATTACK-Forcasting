"""Inference helpers for saved network attack stage classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .model import DEFAULT_CLASS_NAMES, NetworkAttackLSTM, logits_to_probabilities


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> NetworkAttackLSTM:
    """Load a saved model checkpoint onto CPU by default."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    try:
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing model_config or model_state_dict.")
    model = NetworkAttackLSTM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_names = tuple(checkpoint.get("class_names", DEFAULT_CLASS_NAMES))
    model.feature_names = tuple(checkpoint.get("feature_names", ()))
    model.to(device)
    model.eval()
    return model


def _prediction_result(
    model: NetworkAttackLSTM,
    sequence: np.ndarray | Sequence[float] | torch.Tensor,
    class_names: Sequence[str] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    model_device = torch.device(device) if device is not None else next(model.parameters()).device
    tensor = torch.as_tensor(sequence, dtype=torch.float32, device=model_device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError("sequence must have shape (time, features) or (batch, time, features).")
    if tensor.shape[-1] != model.input_size:
        raise ValueError(f"Expected {model.input_size} features, received {tensor.shape[-1]}.")
    names = tuple(class_names or getattr(model, "class_names", DEFAULT_CLASS_NAMES))
    if len(names) != model.num_classes:
        raise ValueError(f"Expected {model.num_classes} class names, received {len(names)}.")
    model.eval()
    with torch.inference_mode():
        probabilities = logits_to_probabilities(model(tensor))
    results = []
    for row in probabilities.cpu().numpy():
        index = int(np.argmax(row))
        results.append(
            {
                "predicted_stage": names[index],
                "probabilities": {name: float(row[i]) for i, name in enumerate(names)},
                "confidence": float(row[index]),
            }
        )
    return results[0] if len(results) == 1 else {"predictions": results}


def predict_probabilities(
    model: NetworkAttackLSTM,
    sequence: np.ndarray | Sequence[float] | torch.Tensor,
    class_names: Sequence[str] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Return predicted stage, per-stage probabilities, and confidence."""
    return _prediction_result(model, sequence, class_names, device)


def predict_next_stage(
    model: NetworkAttackLSTM,
    sequence: np.ndarray | Sequence[float] | torch.Tensor,
    class_names: Sequence[str] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Predict the next stage and return its probabilities and confidence."""
    return _prediction_result(model, sequence, class_names, device)
