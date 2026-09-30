from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from dashboard.helpers import (
    build_forward_looking_alert,
    build_forecast_probability_timeline,
    confidence_status,
    DEMO_SCENARIOS,
    encoder_from_checkpoint,
    load_csv_bytes,
    format_non_benign_forecast_index,
    non_benign_forecast_index,
    prepare_records,
    select_demo_scenario,
    summarize_observed_progression,
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
        cls.sample_records = load_csv_bytes(
            (ROOT / "data" / "sample" / "sample_network_traffic.csv").read_bytes()
        )

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

    def test_non_benign_index_is_one_minus_benign_probability(self):
        distribution = {
            "BENIGN": 0.25,
            "RECONNAISSANCE": 0.15,
            "INITIAL_ACCESS": 0.20,
            "LATERAL_MOVEMENT": 0.20,
            "COMMAND_AND_CONTROL": 0.20,
        }
        self.assertEqual(non_benign_forecast_index(distribution), 0.75)
        self.assertEqual(format_non_benign_forecast_index(0.75), "75.00%")
        self.assertEqual(format_non_benign_forecast_index(1.0 - 0.1234), "87.66%")
        with self.assertRaises(ValueError):
            non_benign_forecast_index({"BENIGN": 0.4})

    def test_timeline_uses_current_and_supplied_rollout_probabilities(self):
        class_names = (
            "BENIGN", "RECONNAISSANCE", "INITIAL_ACCESS", "LATERAL_MOVEMENT", "COMMAND_AND_CONTROL"
        )
        current = np.asarray([0.70, 0.10, 0.10, 0.05, 0.05])
        rollout = np.asarray([
            [0.60, 0.20, 0.10, 0.05, 0.05],
            [0.40, 0.35, 0.10, 0.10, 0.05],
            [0.20, 0.20, 0.40, 0.15, 0.05],
            [0.10, 0.10, 0.20, 0.50, 0.10],
            [0.05, 0.05, 0.10, 0.20, 0.60],
        ])
        timeline = build_forecast_probability_timeline(current, rollout, class_names)
        self.assertEqual([row["step"] for row in timeline], [
            "Current", "Step +1", "Step +2", "Step +3", "Step +4", "Step +5"
        ])
        for index, row in enumerate(timeline):
            original = current if index == 0 else rollout[index - 1]
            self.assertAlmostEqual(row["non_benign_forecast_index"], 1.0 - float(original[0]))
            self.assertAlmostEqual(row["confidence"], float(np.max(original)))
            self.assertEqual(row["predicted_stage"], class_names[int(np.argmax(original))])

    def test_forward_looking_alert_uses_supplied_model_outputs(self):
        alert = build_forward_looking_alert("LATERAL_MOVEMENT", "COMMAND_AND_CONTROL", 0.83, 5)
        self.assertEqual(alert["current_observed_stage"], "LATERAL_MOVEMENT")
        self.assertEqual(alert["forecasted_next_stage"], "COMMAND_AND_CONTROL")
        self.assertEqual(alert["model_confidence"], 0.83)
        self.assertEqual(alert["forecast_horizon"], 5)
        self.assertIn("temporal network-traffic patterns", alert["explanation"])
        self.assertIn("firewall, proxy, endpoint and authentication logs", alert["recommended_actions"][2])
        self.assertIn("does not automatically block traffic", alert["disclaimer"])

    def test_benign_forecast_uses_neutral_language(self):
        alert = build_forward_looking_alert("BENIGN", "BENIGN", 0.91, 5)
        self.assertTrue(alert["is_benign_forecast"])
        self.assertIn("Routine monitoring", alert["headline"])
        self.assertIn("No attack stage is forecast", alert["explanation"])
        self.assertNotIn("attack is occurring", alert["explanation"].lower())

    def test_low_confidence_alert_requests_more_telemetry(self):
        alert = build_forward_looking_alert("RECONNAISSANCE", "INITIAL_ACCESS", 0.59, 5)
        self.assertTrue(alert["low_confidence"])
        self.assertEqual(
            alert["low_confidence_message"],
            "LOW CONFIDENCE — additional telemetry recommended.",
        )
        accepted = build_forward_looking_alert("BENIGN", "BENIGN", 0.60, 5)
        self.assertFalse(accepted["low_confidence"])

    def test_forward_looking_alert_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            build_forward_looking_alert("BENIGN", "BENIGN", float("nan"))
        with self.assertRaises(ValueError):
            build_forward_looking_alert("BENIGN", "BENIGN", 0.8, 0)

    def test_window_stage_selection(self):
        labels = ["BENIGN", "BENIGN", "RECONNAISSANCE"]
        self.assertEqual(window_stage(labels, 0, 2), "BENIGN")
        with self.assertRaises(ValueError):
            window_stage(labels, 2, 2)

    def test_each_scenario_selects_an_existing_temporal_window(self):
        self.assertEqual(len(DEMO_SCENARIOS), 6)
        for scenario in DEMO_SCENARIOS:
            with self.subTest(scenario=scenario):
                selection = select_demo_scenario(self.sample_records, scenario)
                start, end = selection["window_start"], selection["window_end"]
                self.assertEqual(end - start + 1, 20)
                actual_labels = [
                    row["attack_stage"]
                    for row in self.sample_records[start : end + 1]
                ]
                self.assertEqual(selection["stage_counts"], {
                    stage: actual_labels.count(stage)
                    for stage in dict.fromkeys(actual_labels)
                })

    def test_scenario_windows_match_the_requested_stage(self):
        expected = {
            "Normal / Benign Traffic": "BENIGN",
            "Reconnaissance": "RECONNAISSANCE",
            "Initial Access": "INITIAL_ACCESS",
            "Lateral Movement": "LATERAL_MOVEMENT",
            "Command & Control": "COMMAND_AND_CONTROL",
        }
        for scenario, stage in expected.items():
            with self.subTest(scenario=scenario):
                selection = select_demo_scenario(self.sample_records, scenario)
                if stage == "BENIGN":
                    self.assertEqual(selection["stage_counts"][stage], 20)
                else:
                    self.assertGreater(selection["stage_counts"][stage], 0)

    def test_full_progression_uses_exact_existing_stage_boundaries(self):
        selection = select_demo_scenario(self.sample_records, "Full Attack Progression")
        expected = [
            "RECONNAISSANCE", "INITIAL_ACCESS",
            "LATERAL_MOVEMENT", "COMMAND_AND_CONTROL",
        ]
        runs = selection["progression_runs"]
        self.assertEqual([run["stage"] for run in runs], expected)
        for run in runs:
            actual = {
                row["attack_stage"]
                for row in self.sample_records[run["start"] : run["end"] + 1]
            }
            self.assertEqual(actual, {run["stage"]})

    def test_observed_progression_collapses_repeats_and_separates_benign(self):
        result = summarize_observed_progression([
            "BENIGN", "BENIGN", "RECONNAISSANCE", "RECONNAISSANCE",
            "INITIAL_ACCESS", "LATERAL_MOVEMENT", "COMMAND_AND_CONTROL", "BENIGN",
        ])
        self.assertEqual(result["observed_stages"], [
            "RECONNAISSANCE", "INITIAL_ACCESS", "LATERAL_MOVEMENT", "COMMAND_AND_CONTROL"
        ])
        self.assertTrue(result["benign_observed"])
        self.assertTrue(result["complete_attack_progression"])
        self.assertEqual(result["current_observed_stage"], "BENIGN")

    def test_partial_or_out_of_order_window_is_not_claimed_as_complete(self):
        partial = summarize_observed_progression(["BENIGN", "RECONNAISSANCE", "INITIAL_ACCESS"])
        self.assertEqual(partial["observed_stages"], ["RECONNAISSANCE", "INITIAL_ACCESS"])
        self.assertFalse(partial["complete_attack_progression"])

        out_of_order = summarize_observed_progression([
            "RECONNAISSANCE", "LATERAL_MOVEMENT", "INITIAL_ACCESS", "COMMAND_AND_CONTROL"
        ])
        self.assertEqual(out_of_order["observed_stages"], [
            "RECONNAISSANCE", "LATERAL_MOVEMENT", "INITIAL_ACCESS", "COMMAND_AND_CONTROL"
        ])
        self.assertFalse(out_of_order["complete_attack_progression"])

    def test_observed_progression_matches_the_actual_selected_demo_window(self):
        scenario = select_demo_scenario(self.sample_records, "Full Attack Progression")
        labels = [
            row["attack_stage"]
            for row in self.sample_records[scenario["window_start"] : scenario["window_end"] + 1]
        ]
        summary = summarize_observed_progression(labels)
        expected = []
        for label in labels:
            if label != "BENIGN" and (not expected or expected[-1] != label):
                expected.append(label)
        self.assertEqual(summary["observed_stages"], expected)
        self.assertTrue(summary["complete_attack_progression"])


if __name__ == "__main__":
    unittest.main()
