"""Forecast-specific metrics implemented with NumPy and Python only."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def evaluate_trajectories(
    y_true: Sequence[Sequence[str]] | np.ndarray,
    y_pred: Sequence[Sequence[str]] | np.ndarray,
    confidences: Sequence[Sequence[float]] | np.ndarray,
    low_confidence_threshold: float = 0.6,
) -> dict[str, float | int]:
    """Measure stage, trajectory, transition, and confidence performance.

    ``transition_accuracy`` compares whether adjacent-stage changes occurred
    at the same points in predicted and true trajectories.
    """
    truth = np.asarray(y_true, dtype=str)
    predicted = np.asarray(y_pred, dtype=str)
    confidence = np.asarray(confidences, dtype=np.float64)
    if truth.ndim != 2 or predicted.shape != truth.shape or confidence.shape != truth.shape:
        raise ValueError("y_true, y_pred, and confidences must be equal-shape 2D arrays.")
    if truth.size == 0:
        raise ValueError("Cannot evaluate empty forecast trajectories.")
    if not np.isfinite(confidence).all() or np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidences must be finite values between 0 and 1.")
    if not 0.0 <= low_confidence_threshold <= 1.0:
        raise ValueError("low_confidence_threshold must be between 0 and 1.")
    if truth.shape[1] > 1:
        true_changes = truth[:, 1:] != truth[:, :-1]
        predicted_changes = predicted[:, 1:] != predicted[:, :-1]
        transition_accuracy = float(np.mean(true_changes == predicted_changes))
    else:
        transition_accuracy = 1.0
    return {
        "transition_accuracy": transition_accuracy,
        "top1_stage_accuracy": float(np.mean(truth == predicted)),
        "trajectory_exact_match_rate": float(np.mean(np.all(truth == predicted, axis=1))),
        "average_confidence": float(np.mean(confidence)),
        "low_confidence_rate": float(np.mean(confidence < low_confidence_threshold)),
        "low_confidence_threshold": float(low_confidence_threshold),
        "trajectory_count": int(truth.shape[0]),
        "steps_per_trajectory": int(truth.shape[1]),
    }
