"""Train and evaluate the multi-task temporal world model."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.forecast_metrics import evaluate_trajectories
from src.forecasting.model import DEFAULT_CLASS_NAMES
from src.forecasting.rollout import rollout_future, summarize_trajectory
from src.forecasting.world_model import NetworkAttackWorldModel, load_world_model
from src.preprocessing.feature_engineering import (
    FeatureEncoder,
    create_temporal_windows,
    extract_labels,
    load_network_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "sample" / "sample_network_traffic.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "network_attack_world_model.pt"
DEFAULT_RESULTS = PROJECT_ROOT / "models" / "world_model_evaluation_results.json"
CLASS_TO_INDEX = {stage: index for index, stage in enumerate(DEFAULT_CLASS_NAMES)}


@dataclass
class WorldTrainConfig:
    data_path: Path = DEFAULT_DATA
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    results_path: Path = DEFAULT_RESULTS
    window_size: int = 20
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    epochs: int = 40
    patience: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    state_loss_weight: float = 0.5
    stage_loss_weight: float = 1.0
    rollout_steps: int = 5
    confidence_threshold: float = 0.6
    seed: int = 26153
    device: str = "cpu"


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _tensor_dataset(
    windows: np.ndarray, next_states: np.ndarray, next_stages: np.ndarray
) -> TensorDataset:
    return TensorDataset(
        torch.as_tensor(windows, dtype=torch.float32),
        torch.as_tensor(next_states, dtype=torch.float32),
        torch.as_tensor(next_stages, dtype=torch.long),
    )


def _run_epoch(
    model: NetworkAttackWorldModel,
    loader: DataLoader,
    state_loss_fn: nn.Module,
    stage_loss_fn: nn.Module,
    state_loss_weight: float,
    stage_loss_weight: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss_sum = state_loss_sum = stage_loss_sum = 0.0
    sample_count = 0
    for windows, next_states, next_stages in loader:
        windows = windows.to(device)
        next_states = next_states.to(device)
        next_stages = next_stages.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predicted_states, stage_logits = model(windows)
            state_loss = state_loss_fn(predicted_states, next_states)
            stage_loss = stage_loss_fn(stage_logits, next_stages)
            total_loss = state_loss_weight * state_loss + stage_loss_weight * stage_loss
            if training:
                total_loss.backward()
                optimizer.step()
        batch_count = len(windows)
        sample_count += batch_count
        total_loss_sum += float(total_loss.detach().item()) * batch_count
        state_loss_sum += float(state_loss.detach().item()) * batch_count
        stage_loss_sum += float(stage_loss.detach().item()) * batch_count
    if sample_count == 0:
        raise ValueError("A temporal split produced no sequences.")
    return (
        total_loss_sum / sample_count,
        state_loss_sum / sample_count,
        stage_loss_sum / sample_count,
    )


def _encoder_state(encoder: FeatureEncoder) -> dict[str, object]:
    return {
        "vocabularies": encoder.vocabularies,
        "medians": encoder.medians.tolist(),
        "minima": encoder.minima.tolist(),
        "ranges": encoder.ranges.tolist(),
        "feature_names": list(encoder.feature_names),
    }


def _build_test_trajectories(
    model: NetworkAttackWorldModel,
    features: np.ndarray,
    labels: np.ndarray,
    test_start: int,
    window_size: int,
    rollout_steps: int,
    device: torch.device,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forecast every complete horizon in the held-out tail from actual past context."""
    last_start = len(features) - rollout_steps
    if test_start > last_start:
        raise ValueError("Held-out test partition is too short for the configured rollout horizon.")
    starts = range(test_start, last_start + 1)
    true_trajectories = []
    predicted_trajectories = []
    confidence_trajectories = []
    predicted_states = []
    for target_start in starts:
        initial_sequence = features[target_start - window_size : target_start]
        result = rollout_future(
            model,
            initial_sequence,
            steps=rollout_steps,
            device=device,
            confidence_threshold=confidence_threshold,
            class_names=DEFAULT_CLASS_NAMES,
        )
        true_trajectories.append(labels[target_start : target_start + rollout_steps])
        predicted_trajectories.append(result.predicted_stages)
        confidence_trajectories.append(result.confidences)
        predicted_states.append(result.predicted_feature_vectors)
    truth = np.asarray(true_trajectories, dtype=str)
    predictions = np.asarray(predicted_trajectories, dtype=str)
    confidences = np.asarray(confidence_trajectories, dtype=np.float32)
    states = np.asarray(predicted_states, dtype=np.float32)
    return truth, predictions, confidences, states


def train_world_model(config: WorldTrainConfig) -> dict[str, object]:
    """Train chronologically and report measured autoregressive test metrics."""
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if config.epochs < 1 or config.patience < 1 or config.batch_size < 1:
        raise ValueError("epochs, patience, and batch_size must all be positive.")
    if config.state_loss_weight < 0 or config.stage_loss_weight < 0:
        raise ValueError("Loss weights cannot be negative.")
    if config.state_loss_weight + config.stage_loss_weight <= 0:
        raise ValueError("At least one loss weight must be positive.")
    if config.train_fraction <= 0 or config.validation_fraction <= 0:
        raise ValueError("Training and validation fractions must be positive.")
    if config.train_fraction + config.validation_fraction >= 1:
        raise ValueError("Training and validation fractions must sum to less than 1.")
    if config.rollout_steps < 1:
        raise ValueError("rollout_steps must be positive.")

    set_random_seeds(config.seed)
    device = torch.device(config.device)
    records = load_network_csv(config.data_path)
    labels = extract_labels(records)
    observation_count = len(records)
    train_end = int(observation_count * config.train_fraction)
    validation_end = int(observation_count * (config.train_fraction + config.validation_fraction))
    if train_end <= config.window_size:
        raise ValueError("Training observations are insufficient for the configured sequence length.")

    # Fit encodings, imputations, and normalization on the chronological training
    # observations only. Transforming later observations applies these fixed values.
    encoder = FeatureEncoder().fit(records[:train_end])
    features = encoder.transform(records)
    windows, next_stage_names = create_temporal_windows(
        features, labels, window_size=config.window_size, stride=1, horizon=1
    )
    target_indices = np.arange(config.window_size, observation_count, dtype=np.int64)
    if len(target_indices) != len(windows):
        raise RuntimeError("Phase 2 window/next-state target alignment is inconsistent.")
    unknown = sorted(set(next_stage_names.tolist()) - set(CLASS_TO_INDEX))
    if unknown:
        raise ValueError(f"Unknown attack-stage labels: {unknown}")
    next_states = features[target_indices]
    next_stage_ids = np.asarray([CLASS_TO_INDEX[name] for name in next_stage_names], dtype=np.int64)

    train_mask = target_indices < train_end
    validation_mask = (target_indices >= train_end) & (target_indices < validation_end)
    test_mask = target_indices >= validation_end
    split_data = {}
    for name, mask in (("train", train_mask), ("validation", validation_mask), ("test", test_mask)):
        split_data[name] = (windows[mask], next_states[mask], next_stage_ids[mask])
    if any(len(part[0]) == 0 for part in split_data.values()):
        raise ValueError("Data is too short to form non-empty chronological splits.")
    stage_counts = {
        "training": Counter(next_stage_names[train_mask].tolist()),
        "validation": Counter(next_stage_names[validation_mask].tolist()),
        "test": Counter(next_stage_names[test_mask].tolist()),
    }
    for split_name, counts in stage_counts.items():
        print(f"{split_name.title()} stage counts: {dict(counts)}")
    for split_name, counts in stage_counts.items():
        missing_stages = sorted(set(DEFAULT_CLASS_NAMES) - set(counts))
        if missing_stages:
            raise ValueError(
                f"Chronological {split_name} split lacks stages {missing_stages}; "
                "adjust synthetic episode placement, not the temporal split."
            )
    print(
        "Chronological target splits: "
        f"train={len(split_data['train'][0])}, "
        f"validation={len(split_data['validation'][0])}, "
        f"test={len(split_data['test'][0])}; "
        f"observation boundaries={train_end}/{validation_end}/{observation_count}"
    )

    model_config = {
        "input_size": len(encoder.feature_names),
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_classes": len(DEFAULT_CLASS_NAMES),
        "dropout": config.dropout,
    }
    model = NetworkAttackWorldModel(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    state_loss_fn = nn.MSELoss()
    stage_loss_fn = nn.CrossEntropyLoss()
    loaders = {
        name: DataLoader(_tensor_dataset(*parts), batch_size=config.batch_size, shuffle=False)
        for name, parts in split_data.items()
    }

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config.results_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        train_total, train_state, train_stage = _run_epoch(
            model, loaders["train"], state_loss_fn, stage_loss_fn,
            config.state_loss_weight, config.stage_loss_weight, device, optimizer,
        )
        val_total, val_state, val_stage = _run_epoch(
            model, loaders["validation"], state_loss_fn, stage_loss_fn,
            config.state_loss_weight, config.stage_loss_weight, device,
        )
        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"train={train_total:.6f} (state={train_state:.6f}, stage={train_stage:.6f}) "
            f"validation={val_total:.6f} (state={val_state:.6f}, stage={val_stage:.6f})"
        )
        if val_total < best_validation_loss:
            best_validation_loss = val_total
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "class_names": list(DEFAULT_CLASS_NAMES),
                    "class_to_index": CLASS_TO_INDEX,
                    "feature_names": list(encoder.feature_names),
                    "feature_encoder": _encoder_state(encoder),
                    "sequence_length": config.window_size,
                    "state_loss_weight": config.state_loss_weight,
                    "stage_loss_weight": config.stage_loss_weight,
                    "rollout_steps": config.rollout_steps,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "seed": config.seed,
                    "dataset_type": "synthetic demonstration only",
                },
                config.checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"Early stopping at epoch {epoch}; best validation epoch was {best_epoch}.")
                break

    # Reload the best validation checkpoint before any held-out test evaluation.
    model = load_world_model(config.checkpoint_path, device=device)
    test_total, test_state_loss, test_stage_loss = _run_epoch(
        model, loaders["test"], state_loss_fn, stage_loss_fn,
        config.state_loss_weight, config.stage_loss_weight, device,
    )
    true_trajs, predicted_trajs, confidence_trajs, predicted_states = _build_test_trajectories(
        model=model,
        features=features,
        labels=labels,
        test_start=validation_end,
        window_size=config.window_size,
        rollout_steps=config.rollout_steps,
        device=device,
        confidence_threshold=config.confidence_threshold,
    )
    forecast_metrics = evaluate_trajectories(
        true_trajs, predicted_trajs, confidence_trajs, config.confidence_threshold
    )
    actual_states = np.stack([
        features[start : start + config.rollout_steps]
        for start in range(validation_end, observation_count - config.rollout_steps + 1)
    ])
    state_mse = float(np.mean(np.square(predicted_states - actual_states)))
    results: dict[str, object] = {
        "evaluation_status": "complete",
        "evaluation_scope": "held-out chronological synthetic demonstration set",
        "synthetic_demo_only": True,
        "benchmark_results": False,
        "dataset_path": str(config.data_path),
        "split_observation_boundaries": {
            "training_targets_before_index": train_end,
            "validation_targets_before_index": validation_end,
            "test_targets_from_index": validation_end,
        },
        "split_window_counts": {name: int(len(parts[0])) for name, parts in split_data.items()},
        "test_combined_loss": float(test_total),
        "test_state_mse_loss": float(test_state_loss),
        "test_stage_cross_entropy_loss": float(test_stage_loss),
        "autoregressive_test_state_mse": state_mse,
        "best_epoch": int(best_epoch),
        "best_validation_combined_loss": float(best_validation_loss),
        "model_config": model_config,
        "sequence_length": config.window_size,
        "rollout_steps": config.rollout_steps,
        "state_loss_weight": config.state_loss_weight,
        "stage_loss_weight": config.stage_loss_weight,
        "feature_names": list(encoder.feature_names),
        **forecast_metrics,
    }
    config.results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Requested five-step demonstration starts at the first held-out target.
    initial_sequence = features[validation_end - config.window_size : validation_end]
    demo = rollout_future(
        model,
        initial_sequence,
        steps=config.rollout_steps,
        device=device,
        confidence_threshold=config.confidence_threshold,
        class_names=DEFAULT_CLASS_NAMES,
    )
    print("\nHeld-out autoregressive forecast metrics (synthetic demo only):")
    for key in (
        "transition_accuracy", "top1_stage_accuracy", "trajectory_exact_match_rate",
        "average_confidence", "low_confidence_rate",
    ):
        print(f"{key}: {forecast_metrics[key]:.6f}")
    print(f"autoregressive_test_state_mse: {state_mse:.6f}")
    print(f"Initial stage: {labels[validation_end - 1]}")
    print("Forecast:")
    for item in demo.future_steps:
        print(f"t+{item['step']}:")
        print(f"stage: {item['predicted_stage']}")
        print(f"confidence: {item['confidence']:.6f} ({item['confidence_status']})")
        print("probabilities:")
        for stage, probability in zip(DEFAULT_CLASS_NAMES, item["stage_probabilities"]):
            print(f"  {stage}: {float(probability):.6f}")
        print(f"predicted state vector shape: {item['predicted_feature_vector'].shape}")
    print("\nTrajectory summary:")
    print(summarize_trajectory(demo))
    print(f"Probability matrix shape: {demo.probability_matrix.shape}")
    print(f"Checkpoint reload: OK ({config.checkpoint_path})")
    print(f"Saved actual evaluation results: {config.results_path}")
    return results


def parse_args(arguments: Sequence[str] | None = None) -> WorldTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--state-loss-weight", type=float, default=0.5)
    parser.add_argument("--stage-loss-weight", type=float, default=1.0)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=26153)
    values = parser.parse_args(arguments)
    return WorldTrainConfig(
        data_path=values.data,
        checkpoint_path=values.checkpoint,
        results_path=values.results,
        epochs=values.epochs,
        patience=values.patience,
        batch_size=values.batch_size,
        learning_rate=values.lr,
        state_loss_weight=values.state_loss_weight,
        stage_loss_weight=values.stage_loss_weight,
        rollout_steps=values.rollout_steps,
        device=values.device,
        seed=values.seed,
    )


if __name__ == "__main__":
    train_world_model(parse_args())
