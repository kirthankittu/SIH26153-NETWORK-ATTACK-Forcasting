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


def confidence_status(confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> str:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    return "LOW_CONFIDENCE" if float(confidence) < threshold else "OK"


def window_stage(labels: Sequence[str], start: int, window_size: int) -> str:
    end = start + window_size
    if start < 0 or end > len(labels):
        raise ValueError("Selected sequence is outside the available observations.")
    return str(labels[end - 1])


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
