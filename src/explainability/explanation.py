"""Model-agnostic evidence summaries based on feature deviations.

These explanations identify input or forecast-state differences relative to a
reference. They are descriptive evidence, not causal attribution.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .feature_metadata import get_feature_metadata

LOW_CONFIDENCE_THRESHOLD = 0.60
VALID_DIRECTIONS = {"increased", "decreased", "stable", "changed"}


def build_reference_state(
    reference_vectors: np.ndarray | Sequence[Sequence[float]],
    feature_names: Sequence[str],
    method: str = "mean",
) -> np.ndarray:
    """Build a training-reference vector; categorical codes use their mode.

    Pass only training-set vectors when using this as a fitted baseline. For
    numeric features, ``method`` may be ``mean`` or ``median``. Encoded
    categorical features always use their observed mode so the reference remains
    a known category value rather than an uninterpretable average code.
    """
    matrix = np.asarray(reference_vectors, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("reference_vectors must be a non-empty 1D or 2D array.")
    if matrix.shape[1] != len(feature_names):
        raise ValueError("reference_vectors width must match feature_names.")
    if not np.isfinite(matrix).all():
        raise ValueError("reference_vectors must contain only finite values.")
    if method not in {"mean", "median"}:
        raise ValueError("method must be 'mean' or 'median'.")
    values = []
    for column, name in enumerate(feature_names):
        data = matrix[:, column]
        metadata = get_feature_metadata(name)
        if metadata["value_type"] == "categorical":
            unique, counts = np.unique(data, return_counts=True)
            value = unique[np.argmax(counts)]  # sorted unique makes ties deterministic
        elif method == "median":
            value = np.median(data)
        else:
            value = np.mean(data)
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def _probability_map(probabilities: Mapping[str, float] | Sequence[float] | np.ndarray) -> dict[str, float]:
    if isinstance(probabilities, Mapping):
        result = {str(name): float(value) for name, value in probabilities.items()}
    else:
        vector = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if len(vector) != 5:
            raise ValueError("probabilities must contain one value for each of the five stages.")
        stages = ("BENIGN", "RECONNAISSANCE", "INITIAL_ACCESS", "LATERAL_MOVEMENT", "COMMAND_AND_CONTROL")
        result = dict(zip(stages, (float(value) for value in vector)))
    if not result or not np.isfinite(list(result.values())).all():
        raise ValueError("probabilities must be non-empty and finite.")
    if any(value < 0.0 or value > 1.0 for value in result.values()):
        raise ValueError("probabilities must be between 0 and 1.")
    return result


def explain_forecast(
    sequence: np.ndarray | Sequence[Sequence[float]],
    predicted_stage: str,
    probabilities: Mapping[str, float] | Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
    reference_state: Sequence[float] | np.ndarray,
    top_n: int = 5,
    confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    stability_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Rank forecast feature evidence by deviation from a reference state.

    For temporal input the latest observation is compared. Input and reference
    values are expected in the same feature space (Phase 2 min-max vectors).
    ``importance`` is each returned feature's share of total ranked deviation;
    it is not a causal or model-gradient attribution.
    """
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim == 2:
        if values.shape[0] == 0:
            raise ValueError("sequence must contain at least one observation.")
        current = values[-1]
    elif values.ndim == 1:
        current = values
    else:
        raise ValueError("sequence must be a feature vector or a 2D temporal window.")
    reference = np.asarray(reference_state, dtype=np.float64).reshape(-1)
    if len(current) != len(feature_names) or len(reference) != len(feature_names):
        raise ValueError("Current values, reference_state, and feature_names must have equal widths.")
    if not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError("Current and reference feature values must be finite.")
    if not 1 <= top_n <= 5:
        raise ValueError("top_n must be between 1 and 5.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    if stability_tolerance < 0 or not np.isfinite(stability_tolerance):
        raise ValueError("stability_tolerance must be a finite non-negative value.")

    probability_by_stage = _probability_map(probabilities)
    confidence = max(probability_by_stage.values())
    deviations = []
    for index, name in enumerate(feature_names):
        metadata = get_feature_metadata(name)
        delta = float(current[index] - reference[index])
        if metadata["value_type"] == "categorical":
            deviation = 0.0 if abs(delta) <= stability_tolerance else 1.0
            direction = "stable" if deviation == 0.0 else "changed"
        else:
            deviation = abs(delta)
            if deviation <= stability_tolerance:
                direction = "stable"
            else:
                direction = "increased" if delta > 0 else "decreased"
        deviations.append((index, name, delta, deviation, direction, metadata))
    deviations.sort(key=lambda item: (-item[3], item[0]))
    selected = deviations[: min(top_n, len(deviations))]
    total_deviation = sum(item[3] for item in selected)
    evidence = []
    for index, name, delta, deviation, direction, metadata in selected:
        evidence.append(
            {
                "feature": name,
                "description": metadata["description"],
                "category": metadata["category"],
                "current_value": float(current[index]),
                "reference_value": float(reference[index]),
                "direction": direction,
                "normalized_deviation": float(deviation),
                "importance": float(deviation / total_deviation) if total_deviation else 0.0,
            }
        )
    return {
        "predicted_stage": str(predicted_stage),
        "confidence": float(confidence),
        "probabilities": probability_by_stage,
        "low_confidence": bool(confidence < confidence_threshold),
        "confidence_threshold": float(confidence_threshold),
        "evidence": evidence,
        "interpretation_note": "Forecast feature evidence is descriptive deviation from a reference, not causal attribution.",
    }


def explain_trajectory(
    rollout: Any,
    feature_names: Sequence[str],
    reference_state: Sequence[float] | np.ndarray,
    top_n: int = 5,
    confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    stability_tolerance: float = 0.02,
) -> list[dict[str, Any]]:
    """Explain each predicted state and stage in a rollout result."""
    if hasattr(rollout, "future_steps"):
        steps = rollout.future_steps
    elif isinstance(rollout, Mapping):
        steps = rollout.get("future_steps", [])
    else:
        raise TypeError("rollout must be a RolloutResult or mapping with future_steps.")
    explanations = []
    for index, item in enumerate(steps, start=1):
        state = item.get("predicted_feature_vector")
        if state is None and hasattr(rollout, "predicted_feature_vectors"):
            state = rollout.predicted_feature_vectors[index - 1]
        probability_row = item.get("stage_probabilities")
        if probability_row is None and hasattr(rollout, "stage_probabilities"):
            probability_row = rollout.stage_probabilities[index - 1]
        result = explain_forecast(
            sequence=np.asarray(state),
            predicted_stage=item["predicted_stage"],
            probabilities=probability_row,
            feature_names=feature_names,
            reference_state=reference_state,
            top_n=top_n,
            confidence_threshold=confidence_threshold,
            stability_tolerance=stability_tolerance,
        )
        explanations.append(
            {
                "step": int(item.get("step", index)),
                "predicted_stage": result["predicted_stage"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "top_evidence_features": result["evidence"],
                "low_confidence": result["low_confidence"],
                "interpretation_note": result["interpretation_note"],
            }
        )
    return explanations


def generate_forecast_summary(
    explanation: Mapping[str, Any],
    mitre_context: Mapping[str, Any] | None = None,
) -> str:
    """Create concise analyst-facing text from a forecast explanation."""
    stage = str(explanation["predicted_stage"])
    confidence = float(explanation["confidence"])
    lines = [f"Forecast: {stage}", f"Confidence: {confidence:.2f}"]
    if explanation.get("low_confidence", confidence < LOW_CONFIDENCE_THRESHOLD):
        lines.append("LOW_CONFIDENCE: treat this forecast as uncertain.")
    lines.append("\nDriving feature analysis (reference-state deviations):")
    evidence = explanation.get("evidence", explanation.get("top_evidence_features", []))
    if evidence:
        for item in evidence:
            name = item["feature"]
            direction = item["direction"]
            if direction == "changed":
                phrase = "changed relative to the reference state"
            elif direction == "stable":
                phrase = "was stable relative to the reference state"
            else:
                phrase = f"{direction} relative to the reference state"
            lines.append(f"- {name} {phrase}")
    else:
        lines.append("- No feature deviations exceeded the configured tolerance.")
    lines.append("\nPotentially relevant ATT&CK context:")
    if mitre_context is None:
        lines.append("- No ATT&CK mapping supplied.")
    elif mitre_context.get("tactic"):
        tactic_id = mitre_context.get("tactic_id")
        tactic = mitre_context["tactic"]
        lines.append(f"- {tactic}{f' ({tactic_id})' if tactic_id else ''} tactic")
        for technique in mitre_context.get("potential_techniques", []):
            lines.append(f"- {technique['name']} ({technique['id']})")
    else:
        lines.append("- No ATT&CK technique suggested for BENIGN.")
    lines.append(
        "\nThis model forecast and its feature evidence do not prove causation "
        "or confirm that an ATT&CK technique occurred."
    )
    return "\n".join(lines)
