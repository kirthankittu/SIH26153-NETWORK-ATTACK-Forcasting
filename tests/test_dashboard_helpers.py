from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from dashboard.helpers import (
    confidence_status,
    encoder_from_checkpoint,
    load_csv_bytes,
    prepare_records,
    validate_probability_matrix,
    window_stage,
)

ROOT = Path(__file__).resolve().parents[1]


class DashboardHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        checkpoint = torch.load(
            ROOT / "models" / "network_attack_world_model.pt",
            map_location="cpu",
            weights_only=True,
        )
        cls.encoder = encoder_from_checkpoint(checkpoint["feature_encoder"])

    def test_csv_loading(self):
        csv_bytes = b"timestamp,protocol,attack_stage\n2026-01-01T00:00:00Z,TCP,BENIGN\n"
        records = load_csv_bytes(csv_bytes)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["attack_stage"], "BENIGN")

    def test_unlabeled_inference_csv_is_accepted(self):
        records = load_csv_bytes(
            b"timestamp,protocol\n2026-01-01T00:00:00Z,TCP\n"
        )
        self.assertEqual(records[0]["attack_stage"], "UNKNOWN")
        ordered, features, labels = prepare_records(records, self.encoder, window_size=1)
        self.assertEqual(features.shape, (1, 13))
        self.assertEqual(labels.tolist(), ["UNKNOWN"])

    def test_feature_preparation_and_normalization(self):
        records = load_csv_bytes((ROOT / "data" / "sample" / "sample_network_traffic.csv").read_bytes())
        ordered, features, labels = prepare_records(records, self.encoder)
        self.assertEqual(features.shape, (len(ordered), 13))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreaterEqual(float(features.min()), 0.0)
        self.assertLessEqual(float(features.max()), 1.0)
        self.assertEqual(len(labels), len(ordered))

    def test_probability_validation_and_confidence_threshold(self):
        matrix = validate_probability_matrix([[0.1, 0.2, 0.3, 0.2, 0.2]])
        self.assertEqual(matrix.shape, (1, 5))
        self.assertEqual(confidence_status(0.59), "LOW_CONFIDENCE")
        self.assertEqual(confidence_status(0.60), "OK")
        with self.assertRaises(ValueError):
            validate_probability_matrix([[0.2, 0.2]])

    def test_window_stage_selection(self):
        labels = ["BENIGN", "BENIGN", "RECONNAISSANCE"]
        self.assertEqual(window_stage(labels, 0, 2), "BENIGN")
        with self.assertRaises(ValueError):
            window_stage(labels, 2, 2)


if __name__ == "__main__":
    unittest.main()
