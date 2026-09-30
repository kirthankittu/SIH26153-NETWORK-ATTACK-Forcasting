"""Tests for the multi-episode synthetic traffic generator."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from src.preprocessing.feature_engineering import (
    FeatureEncoder,
    create_temporal_windows,
    extract_labels,
    load_network_csv,
)
from src.preprocessing.synthetic_data import STAGES, count_attack_episodes, generate_synthetic_dataset


class SyntheticDataTests(unittest.TestCase):
    def generate_sample(self, directory):
        path = Path(directory) / "synthetic.csv"
        rows = generate_synthetic_dataset(path, num_episodes=20, seed=26153, minimum_records=1000)
        return path, rows

    def test_generator_writes_at_least_1000_rows_with_all_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rows = self.generate_sample(directory)
            loaded = load_network_csv(path)
            self.assertGreaterEqual(len(rows), 1000)
            self.assertEqual(len(loaded), len(rows))
            self.assertEqual(set(row["attack_stage"] for row in loaded), set(STAGES))
            self.assertEqual(loaded[0]["attack_stage"], "BENIGN")
            self.assertEqual(loaded[-1]["attack_stage"], "COMMAND_AND_CONTROL")

    def test_generator_creates_multiple_varied_episodes_with_early_termination(self):
        with tempfile.TemporaryDirectory() as directory:
            _, rows = self.generate_sample(directory)
            self.assertGreaterEqual(count_attack_episodes(rows), 10)
            episode_end_stages = []
            for index, row in enumerate(rows):
                if row["attack_stage"] == "BENIGN" and index > 0 and rows[index - 1]["attack_stage"] != "BENIGN":
                    episode_end_stages.append(rows[index - 1]["attack_stage"])
            self.assertIn("RECONNAISSANCE", episode_end_stages)
            self.assertIn("INITIAL_ACCESS", episode_end_stages)
            self.assertIn("LATERAL_MOVEMENT", episode_end_stages)
            self.assertIn("COMMAND_AND_CONTROL", episode_end_stages)

    def test_chronological_70_15_15_splits_all_contain_all_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            _, rows = self.generate_sample(directory)
            first = int(len(rows) * 0.70)
            second = int(len(rows) * 0.85)
            for part in (rows[:first], rows[first:second], rows[second:]):
                self.assertEqual(set(row["attack_stage"] for row in part), set(STAGES))

    def test_windows_target_the_observation_after_the_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.generate_sample(directory)
            rows = load_network_csv(path)
            labels = extract_labels(rows)
            encoder = FeatureEncoder().fit(rows[: int(len(rows) * 0.70)])
            features = encoder.transform(rows)
            X, y = create_temporal_windows(features, labels, window_size=20, stride=1, horizon=1)
            self.assertEqual(X.shape, (len(rows) - 20, 20, 13))
            self.assertEqual(y.shape, (len(rows) - 20,))
            np.testing.assert_allclose(X[0], features[:20])
            self.assertEqual(y[0], labels[20])
            np.testing.assert_allclose(X[1], features[1:21])
            self.assertEqual(y[1], labels[21])


if __name__ == "__main__":
    unittest.main()
