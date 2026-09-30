"""PyTorch LSTM classifier for next-stage network attack forecasting."""

from __future__ import annotations

import torch
from torch import nn


DEFAULT_CLASS_NAMES = (
    "BENIGN",
    "RECONNAISSANCE",
    "INITIAL_ACCESS",
    "LATERAL_MOVEMENT",
    "COMMAND_AND_CONTROL",
)


class NetworkAttackLSTM(nn.Module):
    """Predict the next attack stage from a batch of temporal features.

    Inputs have shape ``(batch, sequence_length, input_size)``. The final
    recurrent output is passed through dropout and a linear classification
    layer; the forward method returns raw logits for ``CrossEntropyLoss``.
    """

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
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.class_names = DEFAULT_CLASS_NAMES[:num_classes]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits with shape ``(batch, num_classes)``."""
        if inputs.ndim != 3:
            raise ValueError(
                "Expected inputs with shape (batch, sequence_length, input_size); "
                f"received {tuple(inputs.shape)}."
            )
        if inputs.shape[-1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, received {inputs.shape[-1]}.")
        sequence_outputs, _ = self.lstm(inputs)
        final_output = sequence_outputs[:, -1, :]
        return self.classifier(self.dropout(final_output))


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert class logits to probabilities along the class dimension."""
    if logits.ndim < 1:
        raise ValueError("logits must have a class dimension.")
    return torch.softmax(logits, dim=-1)
