"""Tests for forecast feature evidence and ATT&CK context mappings."""

import unittest

import numpy as np
import torch

from src.explainability.explanation import (
    build_reference_state,
    explain_forecast,
    explain_trajectory,
    generate_forecast_summary,
)
from src.explainability.feature_metadata import FEATURE_METADATA, get_feature_metadata
from src.explainability.mitre_mapping import all_mitre_mappings, get_mitre_mapping
from src.forecasting.model import DEFAULT_CLASS_NAMES
from src.forecasting.rollout import rollout_future
from src.forecasting.world_model import NetworkAttackWorldModel
from src.preprocessing.feature_engineering import FeatureEncoder, load_network_csv

FEATURE_NAMES = (
    "src_ip_code", "dst_ip_code", "protocol_code", "tcp_flags_code", "src_port", "dst_port",
    "packet_count", "byte_count", "duration", "ttl", "tcp_window_size", "inter_arrival_mean",
    "payload_size_mean",
)


class ExplainabilityTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.full(13, 0.25, dtype=np.float32)
        self.current = np.full((20, 13), 0.25, dtype=np.float32)
        self.current[-1, 6] = 0.90
        self.current[-1, 7] = 0.80
        self.current[-1, 11] = 0.05
        self.probabilities = np.asarray([0.05, 0.70, 0.10, 0.10, 0.05], dtype=np.float32)

    def test_metadata_covers_all_thirteen_features(self):
        self.assertEqual(set(FEATURE_METADATA), set(FEATURE_NAMES))
        for name in FEATURE_NAMES:
            metadata = get_feature_metadata(name)
            self.assertTrue(metadata["description"])
            self.assertTrue(metadata["category"])
            self.assertTrue(metadata["interpretation"])

    def test_explanation_returns_top_finite_evidence_from_input(self):
        result = explain_forecast(
            self.current, "RECONNAISSANCE", self.probabilities, FEATURE_NAMES, self.reference, top_n=5
        )
        self.assertEqual(result["predicted_stage"], "RECONNAISSANCE")
        self.assertAlmostEqual(result["confidence"], 0.70)
        self.assertEqual(len(result["evidence"]), 5)
        for item in result["evidence"]:
            self.assertTrue(np.isfinite(item["importance"]))
            self.assertTrue(np.isfinite(item["normalized_deviation"]))
            self.assertIn(item["direction"], {"increased", "decreased", "stable", "changed"})
        packet_count = next(item for item in result["evidence"] if item["feature"] == "packet_count")
        self.assertEqual(packet_count["current_value"], float(self.current[-1, 6]))
        self.assertEqual(packet_count["direction"], "increased")

    def test_reference_state_uses_training_numeric_summary_and_categorical_mode(self):
        matrix = np.tile(self.reference, (4, 1))
        matrix[:, 4] = [0.1, 0.2, 0.3, 0.8]
        matrix[:, 0] = [0.1, 0.1, 0.8, 0.8]
        reference = build_reference_state(matrix, FEATURE_NAMES, method="mean")
        self.assertAlmostEqual(reference[4], 0.35)
        self.assertTrue(np.isclose(reference[0], 0.1) or np.isclose(reference[0], 0.8))

    def test_trajectory_explanation_has_one_step_per_rollout_item(self):
        torch.manual_seed(14)
        encoder = FeatureEncoder()
        records = load_network_csv("data/sample/sample_network_traffic.csv")
        encoder.fit(records[: int(len(records) * 0.70)])
        vectors = encoder.transform(records)
        reference = build_reference_state(vectors[:100], FEATURE_NAMES)
        model = NetworkAttackWorldModel(input_size=13, hidden_size=12, num_classes=5)
        rollout = rollout_future(model, vectors[100:120], steps=5, confidence_threshold=0.99)
        explanations = explain_trajectory(rollout, FEATURE_NAMES, reference, top_n=3, confidence_threshold=0.99)
        self.assertEqual(len(explanations), 5)
        self.assertEqual([item["step"] for item in explanations], [1, 2, 3, 4, 5])
        self.assertTrue(all(len(item["top_evidence_features"]) == 3 for item in explanations))

    def test_all_five_stages_have_mitre_context_and_benign_is_empty(self):
        mappings = all_mitre_mappings()
        self.assertEqual(set(mappings), set(DEFAULT_CLASS_NAMES))
        for stage, mapping in mappings.items():
            self.assertEqual(mapping["stage"], stage)
            self.assertIn("disclaimer", mapping)
            for technique in mapping["potential_techniques"]:
                self.assertTrue(technique["id"])
                self.assertTrue(technique["name"])
        self.assertEqual(get_mitre_mapping("BENIGN")["potential_techniques"], [])

    def test_low_confidence_is_explicit_and_summary_is_dynamic(self):
        low_probabilities = np.asarray([0.24, 0.22, 0.20, 0.18, 0.16])
        result = explain_forecast(
            self.current, "BENIGN", low_probabilities, FEATURE_NAMES, self.reference,
            confidence_threshold=0.60,
        )
        self.assertTrue(result["low_confidence"])
        summary = generate_forecast_summary(result, get_mitre_mapping("BENIGN"))
        self.assertIn("BENIGN", summary)
        self.assertIn("Confidence: 0.24", summary)
        self.assertIn("LOW_CONFIDENCE", summary)
        self.assertIn("do not prove causation", summary)


if __name__ == "__main__":
    unittest.main()
