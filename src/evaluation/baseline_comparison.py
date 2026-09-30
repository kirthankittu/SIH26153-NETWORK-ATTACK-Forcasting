"""NumPy-only baseline comparison on the SIH26153 chronological demo split.

The baseline is a multinomial logistic regression over only the observation
immediately before each next-stage target. The saved world model is evaluated
on the same targets with its existing 20-observation temporal input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.forecasting.model import DEFAULT_CLASS_NAMES
from src.forecasting.world_model import load_world_model
from src.preprocessing.feature_engineering import FeatureEncoder, create_temporal_windows, extract_labels, load_network_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "sample" / "sample_network_traffic.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "network_attack_world_model.pt"
DEFAULT_RESULTS = PROJECT_ROOT / "models" / "baseline_comparison_results.json"
DISPLAY_METRICS = (
    ("Accuracy", "accuracy"),
    ("Macro Precision", "precision_macro"),
    ("Macro Recall", "recall_macro"),
    ("Macro F1", "f1_macro"),
    ("Macro OVR FPR", "false_positive_rate_macro_ovr"),
)


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> dict[str, Any]:
    """Calculate accuracy, macro P/R/F1, confusion matrix, and macro OVR FPR."""
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    names = tuple(class_names)
    if truth.ndim != 1 or prediction.ndim != 1 or truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must be one-dimensional and have equal length.")
    if truth.size == 0:
        raise ValueError("Cannot evaluate an empty held-out set.")
    class_count = len(names)
    if class_count < 2 or np.any((truth < 0) | (truth >= class_count)) or np.any((prediction < 0) | (prediction >= class_count)):
        raise ValueError("Every target and prediction must map to a supplied class name.")

    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (truth, prediction), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(true_positive, predicted_count, out=np.zeros_like(true_positive), where=predicted_count != 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
    total = int(confusion.sum())
    fpr_by_class: dict[str, float | None] = {}
    fpr_values = []
    for index, name in enumerate(names):
        false_positive = int(confusion[:, index].sum() - confusion[index, index])
        true_negative = total - int(confusion[index, :].sum()) - false_positive
        denominator = false_positive + true_negative
        rate = float(false_positive / denominator) if denominator else None
        fpr_by_class[name] = rate
        if rate is not None:
            fpr_values.append(rate)

    return {
        "accuracy": float(true_positive.sum() / total),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "false_positive_rate_macro_ovr": float(np.mean(fpr_values)) if fpr_values else None,
        "false_positive_rate_by_class_ovr": fpr_by_class,
        "confusion_matrix": confusion.tolist(),
        "class_names": list(names),
        "sample_count": total,
    }


def chronological_target_boundaries(
    observation_count: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[int, int]:
    """Return the existing trainer's observation boundaries (70% and 85%)."""
    if observation_count < 3:
        raise ValueError("At least three observations are required.")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must be positive and sum to less than one.")
    train_end = int(observation_count * train_fraction)
    validation_end = int(observation_count * (train_fraction + validation_fraction))
    return train_end, validation_end


@dataclass
class NumpyMultinomialLogisticRegression:
    """Fitted, standardized multinomial softmax regression baseline."""

    class_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    best_epoch: int
    best_validation_loss: float
    learning_rate: float
    l2: float

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_mean):
            raise ValueError("features must be a 2D matrix with the fitted feature width.")
        if not np.isfinite(matrix).all():
            raise ValueError("features must be finite.")
        return ((matrix - self.feature_mean) / self.feature_scale) @ self.weights + self.bias

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_logits(features), axis=1).astype(np.int64)


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    row_max = logits.max(axis=1, keepdims=True)
    log_partition = row_max[:, 0] + np.log(np.exp(logits - row_max).sum(axis=1))
    return float(np.mean(log_partition - logits[np.arange(len(labels)), labels]))


def fit_multinomial_logistic_regression(
    x_train: np.ndarray,
    y_train: Sequence[int] | np.ndarray,
    x_validation: np.ndarray,
    y_validation: Sequence[int] | np.ndarray,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    learning_rate: float = 0.1,
    l2: float = 1e-3,
    max_epochs: int = 2000,
    patience: int = 120,
) -> NumpyMultinomialLogisticRegression:
    """Fit full-batch softmax regression, early-stopping on chronological validation."""
    train = np.asarray(x_train, dtype=np.float64)
    validation = np.asarray(x_validation, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_validation, dtype=np.int64)
    names = tuple(class_names)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("Training and validation features must be 2D with equal feature widths.")
    if len(train) != len(y) or len(validation) != len(y_val) or not len(train) or not len(validation):
        raise ValueError("Training and validation features/labels must be non-empty and aligned.")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("Training and validation features must be finite.")
    if learning_rate <= 0 or l2 < 0 or max_epochs < 1 or patience < 1:
        raise ValueError("learning_rate/max_epochs/patience must be positive and l2 cannot be negative.")
    k = len(names)
    if np.any((y < 0) | (y >= k)) or np.any((y_val < 0) | (y_val >= k)):
        raise ValueError("Labels must map to the supplied class_names.")

    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0.0] = 1.0
    z_train = (train - mean) / scale
    z_validation = (validation - mean) / scale
    weights = np.zeros((train.shape[1], k), dtype=np.float64)
    bias = np.zeros(k, dtype=np.float64)
    one_hot = np.eye(k, dtype=np.float64)[y]
    best_loss = float("inf")
    best_weights = weights.copy()
    best_bias = bias.copy()
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        logits = z_train @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        residual = probabilities - one_hot
        weights -= learning_rate * (z_train.T @ residual / len(y) + l2 * weights)
        bias -= learning_rate * residual.mean(axis=0)

        validation_loss = _cross_entropy(z_validation @ weights + bias, y_val) + 0.5 * l2 * float(np.square(weights).sum())
        if validation_loss < best_loss - 1e-9:
            best_loss = validation_loss
            best_weights = weights.copy()
            best_bias = bias.copy()
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    return NumpyMultinomialLogisticRegression(
        class_names=names,
        feature_mean=mean,
        feature_scale=scale,
        weights=best_weights,
        bias=best_bias,
        best_epoch=best_epoch,
        best_validation_loss=float(best_loss),
        learning_rate=float(learning_rate),
        l2=float(l2),
    )


def _prepare_comparable_data(
    data_path: str | Path,
    checkpoint_path: str | Path,
    sequence_length: int = 20,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, Any]:
    records = load_network_csv(data_path)
    labels = extract_labels(records)
    n = len(records)
    train_end, validation_end = chronological_target_boundaries(n, train_fraction, validation_fraction)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder_state = checkpoint.get("feature_encoder")
    if not encoder_state:
        raise ValueError("World-model checkpoint has no training-fitted feature encoder.")
    encoder = FeatureEncoder(
        vocabularies={key: dict(value) for key, value in encoder_state["vocabularies"].items()},
        medians=np.asarray(encoder_state["medians"], dtype=np.float64),
        minima=np.asarray(encoder_state["minima"], dtype=np.float64),
        ranges=np.asarray(encoder_state["ranges"], dtype=np.float64),
        feature_names=tuple(encoder_state["feature_names"]),
    )
    features = encoder.transform(records)
    temporal_x, target_labels = create_temporal_windows(features, labels, window_size=sequence_length)
    target_indices = np.arange(sequence_length, n, dtype=np.int64)
    if len(temporal_x) != len(target_indices) or not np.array_equal(target_labels, labels[target_indices]):
        raise RuntimeError("Temporal windows and next-stage labels are not aligned.")
    if set(target_labels.tolist()) - set(DEFAULT_CLASS_NAMES):
        raise ValueError("Dataset contains labels that are not in the model's five-class mapping.")
    # Deliberately non-temporal baseline: only the final known observation x[t-1]
    # is used to forecast target stage y[t].
    baseline_x = features[target_indices - 1]
    class_to_index = {name: index for index, name in enumerate(DEFAULT_CLASS_NAMES)}
    y = np.asarray([class_to_index[label] for label in target_labels], dtype=np.int64)
    train_mask = target_indices < train_end
    validation_mask = (target_indices >= train_end) & (target_indices < validation_end)
    test_mask = target_indices >= validation_end
    if any(not mask.any() for mask in (train_mask, validation_mask, test_mask)):
        raise ValueError("Chronological split is too short to provide train, validation, and test targets.")
    return {
        "records": records,
        "features": features,
        "temporal_x": temporal_x,
        "baseline_x": baseline_x,
        "y": y,
        "target_indices": target_indices,
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "test_mask": test_mask,
        "train_end": train_end,
        "validation_end": validation_end,
        "sequence_length": sequence_length,
        "checkpoint": checkpoint,
    }


def run_baseline_comparison(
    data_path: str | Path = DEFAULT_DATA,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    results_path: str | Path = DEFAULT_RESULTS,
    sequence_length: int = 20,
    learning_rate: float = 0.1,
    l2: float = 1e-3,
    max_epochs: int = 2000,
    patience: int = 120,
) -> dict[str, Any]:
    """Train only the NumPy baseline, evaluate both models on identical test targets, and save JSON."""
    data = _prepare_comparable_data(data_path, checkpoint_path, sequence_length)
    train_mask, validation_mask, test_mask = data["train_mask"], data["validation_mask"], data["test_mask"]
    baseline = fit_multinomial_logistic_regression(
        data["baseline_x"][train_mask], data["y"][train_mask],
        data["baseline_x"][validation_mask], data["y"][validation_mask],
        learning_rate=learning_rate, l2=l2, max_epochs=max_epochs, patience=patience,
    )
    y_test = data["y"][test_mask]
    baseline_predictions = baseline.predict(data["baseline_x"][test_mask])
    baseline_metrics = classification_metrics(y_test, baseline_predictions)

    model = load_world_model(str(checkpoint_path), device="cpu")
    x_test = torch.as_tensor(data["temporal_x"][test_mask], dtype=torch.float32)
    with torch.inference_mode():
        _, logits = model(x_test)
        world_predictions = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
    world_metrics = classification_metrics(y_test, world_predictions)

    test_targets = data["target_indices"][test_mask]
    split_counts = {
        "train": int(train_mask.sum()),
        "validation": int(validation_mask.sum()),
        "test": int(test_mask.sum()),
    }
    result: dict[str, Any] = {
        "evaluation_label": "Synthetic development-data evaluation",
        "synthetic_demo_only": True,
        "real_world_performance_claim": False,
        "dataset_path": str(Path(data_path)),
        "target_definition": "Predict the attack_stage label at observation t (the next stage) using only observations before t.",
        "split_method": "Chronological observation target indices; 70% train, 15% validation, 15% held-out test, aligned to the world-model trainer.",
        "observation_count": len(data["records"]),
        "split_observation_boundaries": {
            "training_targets_before_index": data["train_end"],
            "validation_targets_before_index": data["validation_end"],
            "test_targets_from_index": data["validation_end"],
        },
        "split_target_counts": split_counts,
        "test_target_index_range_inclusive": [int(test_targets[0]), int(test_targets[-1])],
        "class_names": list(DEFAULT_CLASS_NAMES),
        "metrics_definition": {
            "accuracy": "Correct predictions divided by all held-out targets.",
            "precision_macro": "Unweighted mean of per-class precision; zero when a class has no predicted positives.",
            "recall_macro": "Unweighted mean of per-class recall.",
            "f1_macro": "Unweighted mean of per-class F1.",
            "false_positive_rate_macro_ovr": "Unweighted mean of one-vs-rest FP/(FP+TN) across classes.",
        },
        "baseline": {
            "name": "Multinomial Logistic Regression (NumPy softmax regression)",
            "temporal": False,
            "input_definition": "The normalized feature vector at observation t-1 only.",
            "standardization": "Feature mean and standard deviation fitted on training inputs only; reused for validation and test.",
            "optimizer": "Full-batch gradient descent with cross-entropy and L2 regularization.",
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "max_epochs": int(max_epochs),
            "early_stopping_patience": int(patience),
            "best_validation_epoch": int(baseline.best_epoch),
            "best_validation_loss": baseline.best_validation_loss,
            "metrics": baseline_metrics,
        },
        "world_model": {
            "name": "Saved temporal NetworkAttackWorldModel",
            "checkpoint_path": str(Path(checkpoint_path)),
            "checkpoint_modified": False,
            "temporal": True,
            "input_definition": f"The preceding {sequence_length} normalized observations (t-{sequence_length} through t-1).",
            "prediction_definition": "The saved model's stage-head argmax for the next-stage target at observation t.",
            "metrics": world_metrics,
        },
        "comparison_note": "The baseline is intentionally non-temporal and uses one observation. The world model uses the preceding temporal sequence. Both are measured on the same chronological held-out next-stage targets.",
        "limitations": [
            "All results use synthetic development data and are not public-benchmark or real-world performance.",
            "The saved encoder and world-model checkpoint are reused unchanged; only the NumPy baseline is fitted in this evaluation.",
            "This comparison scores one-step next-stage predictions, not five-step autoregressive trajectory accuracy.",
        ],
    }
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_baseline_comparison(path: str | Path = DEFAULT_RESULTS) -> dict[str, Any]:
    """Load and minimally validate comparison JSON for later dashboard display."""
    with Path(path).open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    for key in ("evaluation_label", "baseline", "world_model", "split_target_counts"):
        if key not in result:
            raise ValueError(f"Comparison results are missing required field {key!r}.")
    for name in ("baseline", "world_model"):
        metrics = result[name].get("metrics", {})
        for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "false_positive_rate_macro_ovr"):
            if metric not in metrics:
                raise ValueError(f"{name} results are missing metric {metric!r}.")
    return result


def format_baseline_comparison_table(comparison: dict[str, Any]) -> list[dict[str, str]]:
    """Format stored comparison metrics for a compact UI table; performs no scoring."""
    rows = []
    for label, key in DISPLAY_METRICS:
        row = {"Metric": label}
        for model_key, column_name in (
            ("baseline", "NumPy Logistic Regression"),
            ("world_model", "Temporal World Model"),
        ):
            try:
                value = float(comparison[model_key]["metrics"][key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Saved {model_key} results have no usable {key} value.") from exc
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Saved metric {model_key}.{key} must be between 0 and 1.")
            row[column_name] = f"{value:.2%}"
        rows.append(row)
    return rows


if __name__ == "__main__":
    comparison = run_baseline_comparison()
    print(comparison["evaluation_label"])
    print("Held-out chronological targets:", comparison["split_target_counts"]["test"])
    for model_key in ("baseline", "world_model"):
        print(model_key, comparison[model_key]["metrics"])
    print("Saved:", DEFAULT_RESULTS)
