"""Tests for the SIH26153 temporal LSTM forecaster and inference helpers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.forecasting.inference import load_model, predict_next_stage, predict_probabilities
from src.forecasting.model import DEFAULT_CLASS_NAMES, NetworkAttackLSTM, logits_to_probabilities


class ForecastingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = NetworkAttackLSTM(
            input_size=13, hidden_size=16, num_layers=1, num_classes=5, dropout=0.1
        )
        self.sequence = np.random.default_rng(7).random((20, 13), dtype=np.float32)

    def test_model_instantiates_and_forward_shape_is_correct(self):
        self.assertIsInstance(self.model, torch.nn.Module)
        logits = self.model(torch.from_numpy(self.sequence).unsqueeze(0))
        self.assertEqual(tuple(logits.shape), (1, 5))

    def test_forward_accepts_a_batch_and_probabilities_sum_to_one(self):
        logits = self.model(torch.from_numpy(np.stack([self.sequence, self.sequence])))
        self.assertEqual(tuple(logits.shape), (2, 5))
        probabilities = logits_to_probabilities(logits)
        np.testing.assert_allclose(probabilities.detach().numpy().sum(axis=1), [1.0, 1.0], atol=1e-6)

    def test_inference_returns_stage_probabilities_and_confidence(self):
        result = predict_probabilities(self.model, self.sequence)
        self.assertIn(result["predicted_stage"], DEFAULT_CLASS_NAMES)
        self.assertEqual(set(result["probabilities"]), set(DEFAULT_CLASS_NAMES))
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=6)
        self.assertAlmostEqual(result["confidence"], max(result["probabilities"].values()), places=6)
        self.assertEqual(result, predict_next_stage(self.model, self.sequence))

    def test_saved_checkpoint_can_be_loaded_for_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            config = {
                "input_size": 13,
                "hidden_size": 16,
                "num_layers": 1,
                "num_classes": 5,
                "dropout": 0.1,
            }
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "model_config": config,
                "class_names": list(DEFAULT_CLASS_NAMES),
                "feature_names": [f"feature_{i}" for i in range(13)],
            }, path)
            loaded = load_model(path)
            result = predict_next_stage(loaded, self.sequence)
            self.assertEqual(len(result["probabilities"]), 5)
            self.assertEqual(tuple(loaded.feature_names), tuple(f"feature_{i}" for i in range(13)))


if __name__ == "__main__":
    unittest.main()
