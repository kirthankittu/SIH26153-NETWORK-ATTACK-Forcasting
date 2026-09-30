"""Iterative autoregressive K-step forecasting for the network world model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .model import DEFAULT_CLASS_NAMES, logits_to_probabilities


@dataclass(frozen=True)
class RolloutResult:
    """Forecast trajectory, including each predicted state and stage distribution."""

    future_steps: list[dict[str, Any]]
    predicted_feature_vectors: np.ndarray
    stage_probabilities: np.ndarray
    predicted_stages: list[str]
    confidences: np.ndarray
    confidence_status: list[str]

    @property
    def probability_matrix(self) -> np.ndarray:
        """Return the full steps-by-classes probability matrix."""
        return self.stage_probabilities


def rollout_future(
    model: torch.nn.Module,
    initial_sequence: np.ndarray | Sequence[Sequence[float]] | torch.Tensor,
    steps: int = 5,
    device: str | torch.device = "cpu",
    confidence_threshold: float = 0.6,
    class_names: Sequence[str] | None = None,
) -> RolloutResult:
    """Autoregressively predict states/stages, feeding each state into the next step.

    Accepts one sequence with shape ``(sequence_length, features)`` or a batch
    of one with shape ``(1, sequence_length, features)``.
    """
    if steps < 1:
        raise ValueError("steps must be a positive integer.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    device = torch.device(device)
    model.to(device)
    sequence = torch.as_tensor(initial_sequence, dtype=torch.float32, device=device)
    if sequence.ndim == 2:
        sequence = sequence.unsqueeze(0)
    if sequence.ndim != 3 or sequence.shape[0] != 1 or sequence.shape[1] < 1:
        raise ValueError("initial_sequence must have shape (time, features) or (1, time, features).")
    model_input_size = getattr(model, "input_size", sequence.shape[-1])
    if sequence.shape[-1] != model_input_size:
        raise ValueError(f"Expected {model_input_size} features, received {sequence.shape[-1]}.")
    names = tuple(class_names or getattr(model, "class_names", DEFAULT_CLASS_NAMES))
    model.eval()

    states: list[np.ndarray] = []
    probability_rows: list[np.ndarray] = []
    predicted_stages: list[str] = []
    confidences: list[float] = []
    statuses: list[str] = []
    future_steps: list[dict[str, Any]] = []
    with torch.inference_mode():
        for step in range(1, steps + 1):
            next_state, stage_logits = model(sequence)
            # Features are min-max normalized; constrain autoregressive
            # state predictions to that range before feeding them back.
            next_state = next_state.clamp(0.0, 1.0)
            if next_state.shape != (1, sequence.shape[-1]):
                raise ValueError(
                    "World model next_state must have shape (batch, features); "
                    f"received {tuple(next_state.shape)}."
                )
            if stage_logits.ndim != 2 or stage_logits.shape[0] != 1:
                raise ValueError("World model stage logits must have shape (1, num_classes).")
            if stage_logits.shape[-1] != len(names):
                raise ValueError(
                    f"Model produced {stage_logits.shape[-1]} classes but {len(names)} names were supplied."
                )
            probabilities = logits_to_probabilities(stage_logits)[0]
            class_index = int(probabilities.argmax().item())
            confidence = float(probabilities[class_index].item())
            state_array = next_state[0].detach().cpu().numpy().astype(np.float32, copy=True)
            probability_array = probabilities.detach().cpu().numpy().astype(np.float32, copy=True)
            stage = names[class_index]
            status = "LOW_CONFIDENCE" if confidence < confidence_threshold else "OK"
            states.append(state_array)
            probability_rows.append(probability_array)
            predicted_stages.append(stage)
            confidences.append(confidence)
            statuses.append(status)
            future_steps.append(
                {
                    "step": step,
                    "predicted_feature_vector": state_array.copy(),
                    "stage_probabilities": probability_array.copy(),
                    "predicted_stage": stage,
                    "confidence": confidence,
                    "confidence_status": status,
                }
            )
            # Core rollout: the next prediction consumes this predicted state,
            # not the original input sequence or a repeated classification call.
            sequence = torch.cat((sequence[:, 1:, :], next_state.unsqueeze(1)), dim=1)

    return RolloutResult(
        future_steps=future_steps,
        predicted_feature_vectors=np.stack(states),
        stage_probabilities=np.stack(probability_rows),
        predicted_stages=predicted_stages,
        confidences=np.asarray(confidences, dtype=np.float32),
        confidence_status=statuses,
    )


def summarize_trajectory(
    result: RolloutResult,
    class_names: Sequence[str] | None = None,
) -> str:
    """Format each forecast step and its confidence status as readable text."""
    lines = []
    for item in result.future_steps:
        suffix = " LOW_CONFIDENCE" if item["confidence_status"] == "LOW_CONFIDENCE" else ""
        lines.append(
            f"Step {item['step']}: {item['predicted_stage']} "
            f"({item['confidence']:.2f}){suffix}"
        )
    return "\n".join(lines)

