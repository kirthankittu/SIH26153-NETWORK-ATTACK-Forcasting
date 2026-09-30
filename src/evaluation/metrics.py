"""Classification metrics for held-out attack-stage forecasts."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classification(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> dict[str, object]:
    """Calculate measured multiclass metrics and one-vs-rest false-positive rates.

    Accuracy is the standard overall score. Precision, recall, F1, and false
    positive rate are macro-averaged so each class contributes equally.
    ``false_positive_rate_by_class`` uses FP/(FP+TN) for each one-vs-rest class.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) != len(prediction):
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length.")
    if len(truth) == 0:
        raise ValueError("Cannot evaluate an empty held-out set.")
    labels = list(range(len(class_names)))
    if np.any((truth < 0) | (truth >= len(labels))) or np.any((prediction < 0) | (prediction >= len(labels))):
        raise ValueError("Class indices must map to the supplied class_names.")
    matrix = confusion_matrix(truth, prediction, labels=labels)
    per_class_fpr: dict[str, float | None] = {}
    all_fpr = []
    total = int(matrix.sum())
    for index, name in enumerate(class_names):
        false_positive = int(matrix[:, index].sum() - matrix[index, index])
        true_negative = total - int(matrix[index, :].sum()) - false_positive
        denominator = false_positive + true_negative
        value = float(false_positive / denominator) if denominator else None
        per_class_fpr[name] = value
        if value is not None:
            all_fpr.append(value)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision_macro": float(precision_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "class_names": list(class_names),
        "false_positive_rate_macro_ovr": float(np.mean(all_fpr)) if all_fpr else None,
        "false_positive_rate_by_class": per_class_fpr,
    }


# Short alias for callers that use the conventional evaluation name.
evaluate = evaluate_classification
