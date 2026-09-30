"""Reproducible, chronological training for the SIH26153 synthetic demo."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.forecasting.model import DEFAULT_CLASS_NAMES, NetworkAttackLSTM
from src.preprocessing.feature_engineering import (
    FeatureEncoder,
    create_temporal_windows,
    extract_labels,
    load_network_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "sample" / "sample_network_traffic.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "network_attack_lstm.pt"
DEFAULT_RESULTS = PROJECT_ROOT / "models" / "evaluation_results.json"
CLASS_TO_INDEX = {name: index for index, name in enumerate(DEFAULT_CLASS_NAMES)}


@dataclass
class TrainConfig:
    data_path: Path = DEFAULT_DATA
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    results_path: Path = DEFAULT_RESULTS
    window_size: int = 20
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.20
    learning_rate: float = 1e-3
    epochs: int = 40
    batch_size: int = 32
    patience: int = 8
    seed: int = 26153
    device: str = "cpu"


def set_random_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _to_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )


def _run_epoch(
    model: NetworkAttackLSTM,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    weighted_loss = 0.0
    count = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(batch_X)
            loss = loss_function(logits, batch_y)
            if training:
                loss.backward()
                optimizer.step()
        weighted_loss += float(loss.item()) * len(batch_y)
        count += len(batch_y)
        predictions.append(logits.detach().argmax(dim=1).cpu().numpy())
        targets.append(batch_y.detach().cpu().numpy())
    if count == 0:
        raise ValueError("A chronological data split produced no sequences.")
    return weighted_loss / count, np.concatenate(predictions), np.concatenate(targets)


def _encoder_state(encoder: FeatureEncoder) -> dict[str, object]:
    return {
        "vocabularies": encoder.vocabularies,
        "medians": encoder.medians.tolist(),
        "minima": encoder.minima.tolist(),
        "ranges": encoder.ranges.tolist(),
        "feature_names": list(encoder.feature_names),
    }


def train(config: TrainConfig) -> dict[str, object]:
    """Train with chronological train/validation/test partitions.

    The encoder is fit only on the initial training observations. Windows are
    assigned to a split by the timestamp of their next-stage target. Validation
    and test inputs may use preceding observations as historical context; their
    target labels remain strictly within their respective later time partitions.
    """
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(config.device)
    if config.epochs < 1 or config.batch_size < 1 or config.patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive.")
    if not 0.0 < config.train_fraction < 1.0 or not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("train_fraction and validation_fraction must be between 0 and 1.")
    if config.train_fraction + config.validation_fraction >= 1.0:
        raise ValueError("The training and validation fractions must sum to less than 1.")

    set_random_seeds(config.seed)
    records = load_network_csv(config.data_path)
    labels = extract_labels(records)
    observation_count = len(records)
    train_end = int(observation_count * config.train_fraction)
    validation_end = int(observation_count * (config.train_fraction + config.validation_fraction))
    if train_end <= config.window_size:
        raise ValueError("Training partition is too short for the configured temporal window.")

    # Fit every scaling/imputation/category parameter on training observations only.
    encoder = FeatureEncoder().fit(records[:train_end])
    all_features = encoder.transform(records)
    all_windows, window_labels = create_temporal_windows(
        all_features, labels, window_size=config.window_size, stride=1, horizon=1
    )
    target_indices = np.arange(config.window_size, observation_count, dtype=np.int64)
    if len(target_indices) != len(all_windows):
        raise RuntimeError("Window target alignment does not match the observation timeline.")
    unknown_labels = sorted(set(window_labels.tolist()) - set(CLASS_TO_INDEX))
    if unknown_labels:
        raise ValueError(f"Dataset contains stage labels not in the model class list: {unknown_labels}")
    targets = np.asarray([CLASS_TO_INDEX[label] for label in window_labels], dtype=np.int64)

    train_mask = target_indices < train_end
    validation_mask = (target_indices >= train_end) & (target_indices < validation_end)
    test_mask = target_indices >= validation_end
    split_data = {
        "train": (all_windows[train_mask], targets[train_mask]),
        "validation": (all_windows[validation_mask], targets[validation_mask]),
        "test": (all_windows[test_mask], targets[test_mask]),
    }
    if any(len(X) == 0 for X, _ in split_data.values()):
        raise ValueError("Dataset is too short to create non-empty temporal train/validation/test splits.")
    print(
        "Chronological target splits: "
        f"train={len(split_data['train'][0])} (targets < {train_end}), "
        f"validation={len(split_data['validation'][0])} ({train_end} to {validation_end - 1}), "
        f"test={len(split_data['test'][0])} (targets >= {validation_end})"
    )

    model_config = {
        "input_size": len(encoder.feature_names),
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_classes": len(DEFAULT_CLASS_NAMES),
        "dropout": config.dropout,
    }
    model = NetworkAttackLSTM(**model_config).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loaders = {
        split_name: DataLoader(
            _to_tensor_dataset(X, y),
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
        )
        for split_name, (X, y) in split_data.items()
    }

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config.results_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        train_loss, _, _ = _run_epoch(model, loaders["train"], loss_function, device, optimizer)
        validation_loss, _, _ = _run_epoch(model, loaders["validation"], loss_function, device)
        print(
            f"Epoch {epoch:03d}/{config.epochs}: "
            f"train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "class_names": list(DEFAULT_CLASS_NAMES),
                "class_to_index": CLASS_TO_INDEX,
                "feature_names": list(encoder.feature_names),
                "feature_encoder": _encoder_state(encoder),
                "sequence_length": config.window_size,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "seed": config.seed,
                "dataset_type": "synthetic demonstration only",
            }
            torch.save(checkpoint, config.checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping at epoch {epoch}; best validation epoch was {best_epoch}.")
                break

    # Only the final chronological partition is used for reported evaluation.
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_loss, test_predictions, test_targets = _run_epoch(
        model, loaders["test"], loss_function, device
    )
    try:
        from src.evaluation.metrics import evaluate_classification

        metrics = evaluate_classification(test_targets, test_predictions, DEFAULT_CLASS_NAMES)
        evaluation_error = None
    except Exception as exc:
        # Preserve checkpoint/training completion but never invent metrics.
        metrics = None
        evaluation_error = f"{type(exc).__name__}: {exc}"
    results: dict[str, object] = {
        "evaluation_scope": "held-out chronological synthetic demonstration set",
        "dataset_path": str(config.data_path),
        "synthetic_demo_only": True,
        "benchmark_results": False,
        "split_observation_boundaries": {
            "training_targets_before_index": train_end,
            "validation_targets_before_index": validation_end,
            "test_targets_from_index": validation_end,
        },
        "split_window_counts": {name: int(len(X)) for name, (X, _) in split_data.items()},
        "test_loss": float(test_loss),
        "evaluation_status": "complete" if metrics is not None else "incomplete",
        "evaluation_error": evaluation_error,
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_validation_loss),
        "model_config": model_config,
        "sequence_length": config.window_size,
        "feature_names": list(encoder.feature_names),
        **(metrics or {}),
    }
    config.results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if metrics is None:
        print("Held-out test metric evaluation FAILED; no classification metrics are reported.")
        print(f"Metric error: {evaluation_error}")
    else:
        print("\nHeld-out test metrics (synthetic demonstration only):")
        print(f"accuracy: {metrics['accuracy']:.6f}")
        print(f"precision_macro: {metrics['precision_macro']:.6f}")
        print(f"recall_macro: {metrics['recall_macro']:.6f}")
        print(f"f1_macro: {metrics['f1_macro']:.6f}")
        print("confusion_matrix:")
        for row in metrics["confusion_matrix"]:
            print(row)
        print(f"false_positive_rate_macro_ovr: {metrics['false_positive_rate_macro_ovr']:.6f}")
    print(f"Saved best checkpoint: {config.checkpoint_path}")
    print(f"Saved held-out results: {config.results_path}")
    return results


def _parse_args(arguments: Sequence[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=26153)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    values = parser.parse_args(arguments)
    return TrainConfig(
        data_path=values.data,
        checkpoint_path=values.checkpoint,
        results_path=values.results,
        epochs=values.epochs,
        batch_size=values.batch_size,
        learning_rate=values.learning_rate,
        hidden_size=values.hidden_size,
        num_layers=values.num_layers,
        dropout=values.dropout,
        patience=values.patience,
        seed=values.seed,
        device=values.device,
    )


if __name__ == "__main__":
    train(_parse_args())
