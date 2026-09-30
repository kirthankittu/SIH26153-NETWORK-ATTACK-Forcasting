"""Multi-task LSTM world model for future feature state and attack stage."""

from __future__ import annotations

import torch
from torch import nn

from .model import DEFAULT_CLASS_NAMES


class NetworkAttackWorldModel(nn.Module):
    """Forecast the next normalized feature vector and next attack stage."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1 or num_classes < 2:
            raise ValueError("input_size, hidden_size, num_layers must be positive; num_classes must be at least 2.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.state_head = nn.Linear(hidden_size, input_size)
        self.stage_head = nn.Linear(hidden_size, num_classes)
        self.class_names = DEFAULT_CLASS_NAMES[:num_classes]

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(next_state, next_stage_logits)`` for a temporal batch."""
        if inputs.ndim != 3:
            raise ValueError(
                "Expected inputs with shape (batch, sequence_length, input_size); "
                f"received {tuple(inputs.shape)}."
            )
        if inputs.shape[1] < 1:
            raise ValueError("sequence_length must be at least one.")
        if inputs.shape[-1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, received {inputs.shape[-1]}.")
        sequence_outputs, _ = self.lstm(inputs)
        representation = self.dropout(sequence_outputs[:, -1, :])
        return self.state_head(representation), self.stage_head(representation)


def load_world_model(
    checkpoint_path: str,
    device: str | torch.device = "cpu",
) -> NetworkAttackWorldModel:
    """Load a saved world-model checkpoint on CPU by default."""
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing model_config or model_state_dict.")
    model = NetworkAttackWorldModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_names = tuple(checkpoint.get("class_names", DEFAULT_CLASS_NAMES))
    model.feature_names = tuple(checkpoint.get("feature_names", ()))
    model.feature_encoder_state = checkpoint.get("feature_encoder")
    model.to(target_device)
    model.eval()
    return model
