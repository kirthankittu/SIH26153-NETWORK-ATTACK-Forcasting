from __future__ import annotations

import csv
import io
from typing import Any, Mapping, Sequence

import numpy as np

from src.preprocessing.feature_engineering import (
    FeatureEncoder,
    extract_labels,
    load_network_csv,
    sort_records_by_timestamp,
)

CONFIDENCE_THRESHOLD = 0.60
DEFAULT_WINDOW_SIZE = 20
DEMO_SCENARIOS = (
    "Normal / Benign Traffic",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Full Attack Progression",
)
_SCENARIO_STAGES = {
    "Normal / Benign Traffic": "BENIGN",
    "Reconnaissance": "RECONNAISSANCE",
    "Initial Access": "INITIAL_ACCESS",
    "Lateral Movement": "LATERAL_MOVEMENT",
    "Command & Control": "COMMAND_AND_CONTROL",
}
_SCENARIO_DESCRIPTIONS = {
    "Normal / Benign Traffic": "A window of observations labeled BENIGN, used as a normal-traffic baseline context.",
    "Reconnaissance": "The context contains the strongest available concentration of rows labeled RECONNAISSANCE, representing probing-like traffic in this synthetic dataset.",
    "Initial Access": "The context contains the strongest available concentration of rows labeled INITIAL_ACCESS, representing unusual service-directed activity in this synthetic dataset.",
    "Lateral Movement": "The context contains the strongest available concentration of rows labeled LATERAL_MOVEMENT, representing changing internal host and service patterns in this synthetic dataset.",
    "Command & Control": "The context contains the strongest available concentration of rows labeled COMMAND_AND_CONTROL, representing periodic low-volume traffic in this synthetic dataset.",
}
_PROGRESSION_STAGES = (
    "RECONNAISSANCE", "INITIAL_ACCESS", "LATERAL_MOVEMENT", "COMMAND_AND_CONTROL"
)
ATTACK_STAGE_ORDER = _PROGRESSION_STAGES


def load_csv_bytes(contents: bytes) -> list[dict[str, str]]:
    """Load uploaded bytes through the same Phase 2 CSV validation pipeline."""
    import tempfile
    from pathlib import Path

    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must be UTF-8 encoded.") from exc
    # Inference does not require ground-truth labels. The Phase 2 training loader
    # requires attack_stage, so add an explicit UNKNOWN label for unlabeled flows.
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is not None and "attack_stage" not in {
        name.strip().lower() for name in reader.fieldnames if name
    }:
        rows = list(reader)
        output = io.StringIO()
        fieldnames = list(reader.fieldnames) + ["attack_stage"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["attack_stage"] = "UNKNOWN"
            writer.writerow(row)
        text = output.getvalue()
    # The established loader accepts paths. Keep the uploaded file temporary and
    # feed it to that shared parser so schema validation and ordering stay equal.
    with tempfile.TemporaryDirectory(prefix="sih26153_upload_") as temp_dir:
        path = Path(temp_dir) / "traffic.csv"
        path.write_text(text, encoding="utf-8", newline="")
        return load_network_csv(path)


def encoder_from_checkpoint(state: Mapping[str, Any]) -> FeatureEncoder:
    """Recreate the training-fitted encoder stored alongside the model."""
    required = {"vocabularies", "medians", "minima", "ranges", "feature_names"}
    missing = required.difference(state)
    if missing:
        raise ValueError("Checkpoint encoder is missing: " + ", ".join(sorted(missing)))
    return FeatureEncoder(
        vocabularies={k: dict(v) for k, v in state["vocabularies"].items()},
        medians=np.asarray(state["medians"], dtype=np.float64),
        minima=np.asarray(state["minima"], dtype=np.float64),
        ranges=np.asarray(state["ranges"], dtype=np.float64),
        feature_names=tuple(state["feature_names"]),
    )


def prepare_records(
    records: Sequence[Mapping[str, str]], encoder: FeatureEncoder, window_size: int = DEFAULT_WINDOW_SIZE
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    """Sort, validate labels, and transform using the checkpoint's fitted encoder."""
    if window_size < 1:
        raise ValueError("window_size must be positive.")
    ordered = sort_records_by_timestamp(records)
    if len(ordered) < window_size:
        raise ValueError(
            f"At least {window_size} observations are required; received {len(ordered)}."
        )
    # Missing labels are valid in production inference; the label is only shown
    # as context and is never fed into the model.
    ordered = [dict(row, attack_stage=(row.get("attack_stage", "").strip() or "UNKNOWN")) for row in ordered]
    labels = extract_labels(ordered)
    features = encoder.transform(ordered)
    if features.shape[1] != len(encoder.feature_names):
        raise ValueError("Transformed feature width does not match the checkpoint encoder.")
    if not np.isfinite(features).all():
        raise ValueError("CSV preprocessing produced non-finite feature values.")
    return ordered, features, labels


def validate_probability_matrix(probabilities: Any, class_count: int = 5) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != class_count:
        raise ValueError(f"Expected a 2D probability matrix with {class_count} columns.")
    if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Each probability row must sum to approximately 1.")
    return matrix


def non_benign_forecast_index(
    probabilities: Mapping[str, float] | Sequence[float] | np.ndarray,
    class_names: Sequence[str] = (
        "BENIGN", "RECONNAISSANCE", "INITIAL_ACCESS", "LATERAL_MOVEMENT", "COMMAND_AND_CONTROL"
    ),
) -> float:
    """Return 1 - P(BENIGN) from an existing normalized class distribution."""
    names = tuple(str(name).strip().upper() for name in class_names)
    if "BENIGN" not in names:
        raise ValueError("class_names must include BENIGN.")
    if isinstance(probabilities, Mapping):
        normalized_map = {str(name).strip().upper(): float(value) for name, value in probabilities.items()}
        if "BENIGN" not in normalized_map:
            raise ValueError("probabilities must include BENIGN.")
        values = np.asarray([normalized_map.get(name, 0.0) for name in names], dtype=np.float64)
    else:
        values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if values.size != len(names):
            raise ValueError("probabilities length must match class_names.")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be finite values between 0 and 1.")
    if not np.isclose(values.sum(), 1.0, atol=1e-4):
        raise ValueError("class probabilities must sum to approximately 1.")
    benign_probability = float(values[names.index("BENIGN")])
    return 1.0 - benign_probability


def format_non_benign_forecast_index(index: float) -> str:
    """Format a validated index as a percentage with two decimal places."""
    value = float(index)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("index must be a finite value between 0 and 1.")
    return f"{value:.2%}"


def build_forecast_probability_timeline(
    current_probabilities: Mapping[str, float] | Sequence[float] | np.ndarray,
    rollout_probabilities: Any,
    class_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Build Current plus autoregressive rows using only supplied model outputs."""
    names = tuple(str(name).strip().upper() for name in class_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("class_names must contain at least two unique classes.")
    current = np.asarray(
        [float(current_probabilities.get(name, current_probabilities.get(name.lower(), 0.0)))
         for name in names], dtype=np.float64
    ) if isinstance(current_probabilities, Mapping) else np.asarray(current_probabilities, dtype=np.float64).reshape(-1)
    if current.shape != (len(names),):
        raise ValueError("current_probabilities length must match class_names.")
    matrix = np.vstack((current, validate_probability_matrix(rollout_probabilities, len(names))))
    # Revalidate the current row alongside the rollout distributions.
    if not np.isfinite(matrix).all() or np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("All timeline probabilities must be finite values between 0 and 1.")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Each timeline probability row must sum to approximately 1.")

    result = []
    for index, row in enumerate(matrix):
        probability_map = {name: float(row[column]) for column, name in enumerate(names)}
        result.append({
            "step": "Current" if index == 0 else f"Step +{index}",
            "predicted_stage": names[int(np.argmax(row))],
            "confidence": float(np.max(row)),
            "non_benign_forecast_index": non_benign_forecast_index(probability_map, names),
            "probabilities": probability_map,
        })
    return result


def confidence_status(confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> str:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    return "LOW_CONFIDENCE" if float(confidence) < threshold else "OK"


def build_forward_looking_alert(
    current_observed_stage: str,
    forecasted_stage: str,
    model_confidence: float,
    forecast_horizon: int = 5,
) -> dict[str, Any]:
    """Build analyst-facing copy from existing observed and model outputs."""
    confidence = float(model_confidence)
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("model_confidence must be a finite value between 0 and 1.")
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be positive.")
    current = str(current_observed_stage).strip().upper() or "UNKNOWN"
    forecast = str(forecasted_stage).strip().upper() or "UNKNOWN"
    is_benign = forecast == "BENIGN"
    low_confidence = confidence_status(confidence) == "LOW_CONFIDENCE"
    if is_benign:
        headline = "Routine monitoring — BENIGN stage forecast"
        explanation = (
            "The model forecasts BENIGN as the next stage. No attack stage is forecast; "
            "continue routine monitoring and review any traffic that deviates from the expected baseline."
        )
        actions = (
            "Continue routine monitoring of the observed communication pattern.",
            "Review telemetry if behavior deviates from the expected baseline.",
            "Correlate with other logs if separate indicators warrant investigation.",
        )
    else:
        headline = "Analyst review recommended"
        explanation = (
            "The model forecasts the next stage from temporal network-traffic patterns, giving a security analyst "
            "additional information for investigation and response."
        )
        actions = (
            "Review the relevant source/destination communication pattern.",
            "Inspect unusual protocol/port behaviour.",
            "Correlate with firewall, proxy, endpoint and authentication logs.",
            "Investigate the predicted stage before taking containment action.",
        )
    return {
        "current_observed_stage": current,
        "forecasted_next_stage": forecast,
        "model_confidence": confidence,
        "forecast_horizon": int(forecast_horizon),
        "confidence_status": "LOW_CONFIDENCE" if low_confidence else "OK",
        "low_confidence": low_confidence,
        "is_benign_forecast": is_benign,
        "headline": headline,
        "explanation": explanation,
        "recommended_actions": actions,
        "low_confidence_message": "LOW CONFIDENCE — additional telemetry recommended.",
        "disclaimer": (
            "Forecast confidence is a model output, not a calibrated probability that an attack will occur. "
            "The system assists analysts and does not automatically block traffic."
        ),
    }


def window_stage(labels: Sequence[str], start: int, window_size: int) -> str:
    end = start + window_size
    if start < 0 or end > len(labels):
        raise ValueError("Selected sequence is outside the available observations.")
    return str(labels[end - 1])


def summarize_observed_progression(labels: Sequence[str]) -> dict[str, Any]:
    """Summarize stage changes exactly as they appear in a selected row window.

    Consecutive duplicate labels are collapsed to one stage entry. BENIGN is
    reported separately and never included in the attack-stage chain. Unknown
    labels are not mapped to an attack stage. A complete progression is true
    only when the four canonical attack-stage runs occur contiguously and in
    their prescribed order in the observed chain.
    """
    normalized = [str(label).strip().upper() or "UNKNOWN" for label in labels]
    observed: list[str] = []
    for label in normalized:
        if label in ATTACK_STAGE_ORDER and (not observed or observed[-1] != label):
            observed.append(label)
    complete = any(
        tuple(observed[start : start + len(ATTACK_STAGE_ORDER)]) == ATTACK_STAGE_ORDER
        for start in range(len(observed) - len(ATTACK_STAGE_ORDER) + 1)
    )
    return {
        "observed_stages": observed,
        "benign_observed": "BENIGN" in normalized,
        "current_observed_stage": normalized[-1] if normalized else "UNKNOWN",
        "complete_attack_progression": complete,
    }


def select_demo_scenario(
    records: Sequence[Mapping[str, str]],
    scenario: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> dict[str, Any]:
    """Choose a real contiguous demo window for a scenario using its CSV labels.

    Stage-specific scenarios maximize the count of matching labels in the
    selected model context. Full progression first finds exact contiguous runs
    of all four ordered attack stages, then chooses a 20-row context from that
    episode with the greatest stage diversity. No labels or outputs are made up.
    """
    if scenario not in DEMO_SCENARIOS:
        raise ValueError(f"Unknown demo scenario: {scenario!r}.")
    ordered = sort_records_by_timestamp(records)
    if window_size < 1 or len(ordered) < window_size:
        raise ValueError(f"At least {window_size} observations are required for this scenario.")
    labels = [str(row.get("attack_stage", "UNKNOWN")).strip().upper() or "UNKNOWN" for row in ordered]
    windows = [labels[start : start + window_size] for start in range(len(labels) - window_size + 1)]

    if scenario != "Full Attack Progression":
        target = _SCENARIO_STAGES[scenario]
        if target not in labels:
            raise ValueError(f"The development dataset contains no rows labeled {target}.")
        # Prefer a context with the largest target-stage share; ties choose the
        # earliest occurrence, so scenario selection is deterministic.
        start = max(range(len(windows)), key=lambda idx: (windows[idx].count(target), -idx))
        selected = windows[start]
        counts = {stage: selected.count(stage) for stage in dict.fromkeys(selected)}
        if scenario == "Normal / Benign Traffic" and counts.get("BENIGN", 0) != window_size:
            raise ValueError("No fully BENIGN model window exists in the development dataset.")
        return {
            "scenario": scenario,
            "description": _SCENARIO_DESCRIPTIONS[scenario],
            "window_start": start,
            "window_end": start + window_size - 1,
            "stage_counts": counts,
            "region_start": start,
            "region_end": start + window_size - 1,
            "progression_runs": [],
        }

    # Exact contiguous label runs identify complete attack episodes.
    runs: list[dict[str, Any]] = []
    run_start = 0
    while run_start < len(labels):
        run_end = run_start + 1
        while run_end < len(labels) and labels[run_end] == labels[run_start]:
            run_end += 1
        runs.append({"stage": labels[run_start], "start": run_start, "end": run_end - 1})
        run_start = run_end
    progressions = [
        runs[index : index + len(_PROGRESSION_STAGES)]
        for index in range(len(runs) - len(_PROGRESSION_STAGES) + 1)
        if tuple(run["stage"] for run in runs[index : index + len(_PROGRESSION_STAGES)]) == _PROGRESSION_STAGES
    ]
    if not progressions:
        raise ValueError("No complete ordered multi-stage progression exists in the development dataset.")

    # Choose the episode with the richest 20-observation attack-stage context.
    candidates = []
    for progression in progressions:
        region_start, region_end = progression[0]["start"], progression[-1]["end"]
        possible_starts = range(region_start, region_end - window_size + 2)
        best_start = max(
            possible_starts,
            key=lambda idx: (
                len(set(windows[idx]).intersection(_PROGRESSION_STAGES)),
                sum(label != "BENIGN" for label in windows[idx]),
                -idx,
            ),
        )
        selected = windows[best_start]
        score = (
            len(set(selected).intersection(_PROGRESSION_STAGES)),
            sum(label != "BENIGN" for label in selected),
            -best_start,
        )
        candidates.append((score, progression, best_start, selected))
    _, progression, start, selected = max(candidates, key=lambda item: item[0])
    region_start, region_end = progression[0]["start"], progression[-1]["end"]
    stage_counts = {stage: selected.count(stage) for stage in dict.fromkeys(selected)}
    progression_runs = [dict(run) for run in progression]
    exact_stages = " → ".join(run["stage"] for run in progression_runs)
    description = (
        f"Selected an existing episode whose exact contiguous stages are {exact_stages} "
        f"(rows {region_start + 1}–{region_end + 1}). The model still receives only its "
        f"{window_size}-observation context (rows {start + 1}–{start + window_size}); "
        "the complete episode is shown as scenario context, not passed as one oversized input."
    )
    return {
        "scenario": scenario,
        "description": description,
        "window_start": start,
        "window_end": start + window_size - 1,
        "stage_counts": stage_counts,
        "region_start": region_start,
        "region_end": region_end,
        "progression_runs": progression_runs,
    }


def raw_window_summary(records: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Summarize numeric flow telemetry without pandas or hidden coercions."""
    numeric = (
        "packet_count", "byte_count", "duration", "ttl", "tcp_window_size",
        "inter_arrival_mean", "payload_size_mean",
    )
    result: dict[str, Any] = {}
    for name in numeric:
        values = []
        for record in records:
            try:
                value = float(record.get(name, ""))
                if np.isfinite(value):
                    values.append(value)
            except (TypeError, ValueError):
                continue
        result[name] = (sum(values) if name in {"packet_count", "byte_count"} else float(np.mean(values))) if values else None
    result["src_ports"] = sorted({str(r.get("src_port", "")).strip() for r in records if str(r.get("src_port", "")).strip()})
    result["dst_ports"] = sorted({str(r.get("dst_port", "")).strip() for r in records if str(r.get("dst_port", "")).strip()})
    return result
