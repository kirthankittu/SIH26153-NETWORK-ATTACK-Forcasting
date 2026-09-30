"""Unit tests for CSV traffic preprocessing."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.preprocessing.feature_engineering import (
    FeatureEncoder,
    create_temporal_windows,
    extract_features,
    extract_labels,
    load_network_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "sample_network_traffic.csv"


class PreprocessingTests(unittest.TestCase):
    def test_csv_loading_and_timestamp_sorting(self):
        records = load_network_csv(SAMPLE_PATH)
        self.assertGreaterEqual(len(records), 500)
        self.assertEqual(records[0]["timestamp"], "2026-01-01T00:00:00Z")
        self.assertEqual(records[0]["attack_stage"], "BENIGN")

    def test_feature_extraction_returns_named_finite_vectors(self):
        records = load_network_csv(SAMPLE_PATH)[:12]
        vectors, names, encoder = extract_features(records)
        self.assertEqual(vectors.shape, (12, len(names)))
        self.assertEqual(len(names), 13)
        self.assertEqual(names[0:4], ("src_ip_code", "dst_ip_code", "protocol_code", "tcp_flags_code"))
        self.assertTrue(np.isfinite(vectors).all())
        self.assertIsInstance(encoder, FeatureEncoder)

    def test_normalization_is_bounded_and_missing_numbers_are_imputed(self):
        records = load_network_csv(SAMPLE_PATH)[:10]
        records[0]["byte_count"] = ""
        records[1]["ttl"] = "not-a-number"
        vectors, _, _ = extract_features(records)
        self.assertTrue(np.isfinite(vectors).all())
        self.assertGreaterEqual(float(vectors.min()), 0.0)
        self.assertLessEqual(float(vectors.max()), 1.0)

    def test_temporal_windows_and_next_stage_targets(self):
        features = np.arange(30, dtype=np.float32).reshape(10, 3)
        labels = np.asarray([f"S{i}" for i in range(10)])
        X, y = create_temporal_windows(features, labels, window_size=3, stride=2, horizon=1)
        self.assertEqual(X.shape, (4, 3, 3))
        np.testing.assert_array_equal(y, ["S3", "S5", "S7", "S9"])
        np.testing.assert_array_equal(X[1], features[2:5])

    def test_attack_stage_extraction(self):
        records = load_network_csv(SAMPLE_PATH)
        labels = extract_labels(records)
        self.assertEqual(len(labels), len(records))
        self.assertEqual(labels[0], "BENIGN")
        self.assertEqual(labels[-1], "COMMAND_AND_CONTROL")
        self.assertEqual(set(labels), {
            "BENIGN", "RECONNAISSANCE", "INITIAL_ACCESS", "LATERAL_MOVEMENT", "COMMAND_AND_CONTROL"
        })

    def test_missing_required_columns_have_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            path.write_text("src_ip,attack_stage\n192.168.1.2,BENIGN\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "timestamp, protocol"):
                load_network_csv(path)


if __name__ == "__main__":
    unittest.main()
