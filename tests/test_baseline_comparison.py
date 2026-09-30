from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.baseline_comparison import (
    chronological_target_boundaries,
    classification_metrics,
    fit_multinomial_logistic_regression,
    format_baseline_comparison_table,
    load_baseline_comparison,
)


class BaselineComparisonTests(unittest.TestCase):
    def test_multiclass_metrics_and_one_vs_rest_false_positive_rates(self):
        result = classification_metrics([0, 1, 2, 0], [0, 2, 2, 1], ("A", "B", "C"))
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 0, 1], [0, 0, 1]])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["precision_macro"], (1.0 + 0.0 + 0.5) / 3)
        self.assertAlmostEqual(result["recall_macro"], (0.5 + 0.0 + 1.0) / 3)
        self.assertEqual(result["false_positive_rate_by_class_ovr"]["A"], 0.0)
        self.assertAlmostEqual(result["false_positive_rate_by_class_ovr"]["B"], 1 / 3)

    def test_chronological_boundaries_match_world_model_training_split(self):
        self.assertEqual(chronological_target_boundaries(1571), (1099, 1335))

    def test_numpy_logistic_regression_fits_and_predicts_actual_labels(self):
        x = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.asarray([0, 0, 1, 1])
        model = fit_multinomial_logistic_regression(
            x[:3], y[:3], x[3:], y[3:], class_names=("LEFT", "RIGHT"),
            max_epochs=100, patience=30,
        )
        predictions = model.predict(x)
        self.assertEqual(predictions.shape, y.shape)
        self.assertEqual(predictions[-1], 1)
        self.assertGreaterEqual(model.best_epoch, 1)
        self.assertTrue(np.isfinite(model.best_validation_loss))

    def test_comparison_loader_checks_required_metric_fields(self):
        valid = {
            "evaluation_label": "Synthetic development-data evaluation",
            "baseline": {"metrics": {k: 0.2 for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "false_positive_rate_macro_ovr")}},
            "world_model": {"metrics": {k: 0.8 for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "false_positive_rate_macro_ovr")}},
            "split_target_counts": {"test": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(load_baseline_comparison(path)["evaluation_label"], valid["evaluation_label"])
            rows = format_baseline_comparison_table(valid)
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]["NumPy Logistic Regression"], "20.00%")
            self.assertEqual(rows[0]["Temporal World Model"], "80.00%")
            self.assertEqual(rows[-1]["Metric"], "Macro OVR FPR")
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_baseline_comparison(path)


if __name__ == "__main__":
    unittest.main()
