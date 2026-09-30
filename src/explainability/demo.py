"""Run a held-out synthetic forecast evidence and ATT&CK context demo."""

from __future__ import annotations

from pathlib import Path


from src.explainability.explanation import (
    build_reference_state,
    explain_forecast,
    explain_trajectory,
    generate_forecast_summary,
)
from src.explainability.feature_metadata import FEATURE_METADATA
from src.explainability.mitre_mapping import get_mitre_mapping
from src.forecasting.model import logits_to_probabilities
from src.forecasting.rollout import rollout_future
from src.forecasting.world_model import load_world_model
from src.preprocessing.feature_engineering import FeatureEncoder, extract_labels, load_network_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "sample" / "sample_network_traffic.csv"
    checkpoint_path = PROJECT_ROOT / "models" / "network_attack_world_model.pt"
    records = load_network_csv(data_path)
    train_end = int(len(records) * 0.70)
    test_start = int(len(records) * 0.85)
    encoder = FeatureEncoder().fit(records[:train_end])
    features = encoder.transform(records)
    labels = extract_labels(records)
    model = load_world_model(checkpoint_path, device="cpu")
    names = tuple(model.feature_names) or tuple(encoder.feature_names)
    reference_state = build_reference_state(features[:train_end], names, method="mean")
    initial_sequence = features[test_start - 20 : test_start]
    trajectory = rollout_future(
        model,
        initial_sequence,
        steps=5,
        device="cpu",
        confidence_threshold=0.60,
    )

    first_step = trajectory.future_steps[0]
    first_explanation = explain_forecast(
        initial_sequence,
        first_step["predicted_stage"],
        first_step["stage_probabilities"],
        names,
        reference_state,
        top_n=5,
        confidence_threshold=0.60,
    )
    mapping = get_mitre_mapping(first_step["predicted_stage"])
    trajectory_explanations = explain_trajectory(
        trajectory,
        names,
        reference_state,
        top_n=3,
        confidence_threshold=0.60,
    )

    print("Current forecast (first held-out test forecast):")
    print(generate_forecast_summary(first_explanation, mapping))
    print("\nTop 5 forecast feature evidence:")
    for evidence in first_explanation["evidence"]:
        print(
            f"- {evidence['feature']}: {evidence['direction']}; "
            f"current={evidence['current_value']:.4f}, "
            f"reference={evidence['reference_value']:.4f}, "
            f"normalized_deviation={evidence['normalized_deviation']:.4f}, "
            f"importance={evidence['importance']:.4f}"
        )
    print("\nFive-step trajectory explanations:")
    for item in trajectory_explanations:
        print(f"t+{item['step']}: {item['predicted_stage']} (confidence={item['confidence']:.4f})")
        if item["low_confidence"]:
            print("  LOW_CONFIDENCE")
        for evidence in item["top_evidence_features"]:
            direction = "changed" if evidence["direction"] == "changed" else evidence["direction"]
            print(f"  - {evidence['feature']} {direction} relative to the reference")
    print("\nComplete trajectory probability matrix (rows=t+1 through t+5):")
    print(trajectory.probability_matrix)


if __name__ == "__main__":
    main()
