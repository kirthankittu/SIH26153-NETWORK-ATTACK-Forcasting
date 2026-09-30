"""Autoregressive rollout and forecast metric tests."""

import unittest

import numpy as np
import torch
from torch import nn

from src.evaluation.forecast_metrics import evaluate_trajectories
from src.forecasting.model import DEFAULT_CLASS_NAMES, logits_to_probabilities
from src.forecasting.rollout import rollout_future, summarize_trajectory
from src.forecasting.world_model import NetworkAttackWorldModel


class RecordingWorldModel(nn.Module):
    """Simple deterministic model that reveals the sequence passed each step."""

    input_size = 13
    class_names = DEFAULT_CLASS_NAMES

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.inputs_seen = []

    def forward(self, sequence):
        self.inputs_seen.append(sequence.detach().cpu().clone())
        next_state = sequence[:, -1, :] + 0.1
        logits = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0]], device=sequence.device)
        return next_state, logits


class RolloutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = NetworkAttackWorldModel(
            input_size=13, hidden_size=16, num_layers=1, num_classes=5, dropout=0.1
        )
        self.sequence = np.random.default_rng(12).random((20, 13), dtype=np.float32)

    def test_world_model_forward_shapes_and_probabilities(self):
        next_state, logits = self.model(torch.from_numpy(self.sequence).unsqueeze(0))
        self.assertEqual(tuple(next_state.shape), (1, 13))
        self.assertEqual(tuple(logits.shape), (1, 5))
        probabilities = logits_to_probabilities(logits)
        self.assertAlmostEqual(float(probabilities.detach().sum()), 1.0, places=6)

    def test_rollout_shapes_probabilities_and_confidences(self):
        result = rollout_future(self.model, self.sequence, steps=5, confidence_threshold=0.99)
        self.assertEqual(len(result.future_steps), 5)
        self.assertEqual(result.predicted_feature_vectors.shape, (5, 13))
        self.assertTrue(np.all((result.predicted_feature_vectors >= 0.0) & (result.predicted_feature_vectors <= 1.0)))
        self.assertEqual(result.stage_probabilities.shape, (5, 5))
        self.assertEqual(len(result.predicted_stages), 5)
        self.assertTrue(np.allclose(result.stage_probabilities.sum(axis=1), 1.0, atol=1e-6))
        self.assertTrue(np.all((result.confidences >= 0.0) & (result.confidences <= 1.0)))
        self.assertEqual(result.confidence_status, ["LOW_CONFIDENCE"] * 5)
        self.assertIn("Step 1:", summarize_trajectory(result))

    def test_rollout_feeds_predicted_state_back_each_step(self):
        model = RecordingWorldModel()
        result = rollout_future(model, self.sequence, steps=3)
        self.assertEqual(len(model.inputs_seen), 3)
        first_predicted = result.predicted_feature_vectors[0]
        second_model_input = model.inputs_seen[1][0].numpy()
        np.testing.assert_allclose(second_model_input[-1], first_predicted)
        self.assertFalse(np.array_equal(second_model_input, self.sequence))
        self.assertFalse(np.array_equal(result.predicted_feature_vectors[1], self.sequence[-1]))

    def test_forecast_metrics_are_calculated(self):
        y_true = np.asarray([["BENIGN", "RECONNAISSANCE", "INITIAL_ACCESS"]])
        y_pred = np.asarray([["BENIGN", "RECONNAISSANCE", "LATERAL_MOVEMENT"]])
        confidence = np.asarray([[0.9, 0.7, 0.4]])
        metrics = evaluate_trajectories(y_true, y_pred, confidence, low_confidence_threshold=0.6)
        self.assertAlmostEqual(metrics["top1_stage_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["trajectory_exact_match_rate"], 0.0)
        self.assertAlmostEqual(metrics["average_confidence"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["low_confidence_rate"], 1.0 / 3.0)
        self.assertIn("transition_accuracy", metrics)


if __name__ == "__main__":
    unittest.main()


