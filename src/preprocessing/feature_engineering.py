"""CSV-based network traffic feature extraction and temporal windowing.

This module intentionally uses only Python's standard library and NumPy.
"""

from __future__ import annotations

import csv
import ipaddress
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

# Minimum schema for useful supervised temporal forecasting. Other supported
# feature columns may be absent; absent values are safely median-imputed.
REQUIRED_COLUMNS = ("timestamp", "protocol", "attack_stage")
NUMERIC_FEATURES = (
    "src_port",
    "dst_port",
    "packet_count",
    "byte_count",
    "duration",
    "ttl",
    "tcp_window_size",
    "inter_arrival_mean",
    "payload_size_mean",
)
SUPPORTED_COLUMNS = (
    "timestamp", "src_ip", "dst_ip", *NUMERIC_FEATURES[:2], "protocol",
    *NUMERIC_FEATURES[2:], "tcp_flags", "attack_stage",
)
CATEGORICAL_FEATURES = ("src_ip", "dst_ip", "protocol", "tcp_flags")


def _normalise_column_name(name: str) -> str:
    return name.strip().lower()


def load_network_csv(path: str | Path) -> list[dict[str, str]]:
    """Load and validate a traffic CSV with ``csv.DictReader``.

    The minimum required columns are timestamp, protocol, and attack_stage.
    Optional supported columns are retained when present.
    """
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV is empty or has no header row.")
        field_map = {name: _normalise_column_name(name) for name in reader.fieldnames if name}
        normalized_headers = set(field_map.values())
        missing = [column for column in REQUIRED_COLUMNS if column not in normalized_headers]
        if missing:
            raise ValueError(
                "CSV is missing required fields: " + ", ".join(missing)
                + ". Required fields are timestamp, protocol, and attack_stage."
            )
        records: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if row is None or all(value is None or not value.strip() for value in row.values()):
                continue
            normalized = {
                field_map[key]: (value.strip() if value is not None else "")
                for key, value in row.items()
                if key in field_map
            }
            normalized["_row_number"] = str(row_number)
            records.append(normalized)
    if not records:
        raise ValueError("CSV contains a header but no data rows.")
    return sort_records_by_timestamp(records)


def _timestamp_key(record: Mapping[str, str]) -> tuple[int, object]:
    value = record.get("timestamp", "").strip()
    if not value:
        return (1, "")
    try:
        # Accept ISO timestamps, including a trailing Z.
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (0, parsed.timestamp())
    except (ValueError, OverflowError):
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)


def sort_records_by_timestamp(records: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Return records ordered by timestamp; stable order breaks timestamp ties."""
    return [dict(row) for row in sorted(records, key=_timestamp_key)]


def extract_labels(
    records: Sequence[Mapping[str, str]], label_column: str = "attack_stage"
) -> np.ndarray:
    """Extract string stage labels, reporting any missing labels clearly."""
    labels = [str(row.get(label_column, "")).strip() for row in records]
    missing_rows = [index for index, label in enumerate(labels) if not label]
    if missing_rows:
        preview = ", ".join(str(index + 1) for index in missing_rows[:10])
        raise ValueError(f"Missing {label_column!r} labels at record indices: {preview}.")
    return np.asarray(labels, dtype=str)


def _numeric_value(value: object) -> float:
    if value is None or not str(value).strip():
        return math.nan
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _canonical_category(feature: str, value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return "<MISSING>"
    if feature == "protocol":
        return text.upper()
    if feature == "tcp_flags":
        return "+".join(sorted(part.strip().upper() for part in text.replace("|", "+").split("+") if part.strip()))
    if feature in ("src_ip", "dst_ip"):
        try:
            return str(ipaddress.ip_address(text))
        except ValueError:
            return text
    return text


@dataclass
class FeatureEncoder:
    """Fitted categorical encoder and min-max normalizer for traffic records.

    Fit this object on training records only, then reuse ``transform`` for
    validation, test, and inference data to prevent information leakage.
    """

    vocabularies: dict[str, dict[str, int]] = field(default_factory=dict)
    medians: np.ndarray | None = None
    minima: np.ndarray | None = None
    ranges: np.ndarray | None = None
    feature_names: tuple[str, ...] = ()

    def fit(self, records: Sequence[Mapping[str, str]]) -> "FeatureEncoder":
        if not records:
            raise ValueError("Cannot fit feature encoder on an empty record set.")
        sorted_records = sort_records_by_timestamp(records)
        self.vocabularies = {}
        for feature in CATEGORICAL_FEATURES:
            categories = sorted({_canonical_category(feature, row.get(feature, "")) for row in sorted_records})
            # Unknown categories map to 0; fitted categories start at 1.
            self.vocabularies[feature] = {category: index + 1 for index, category in enumerate(categories)}
        self.feature_names = tuple(
            [f"{name}_code" for name in CATEGORICAL_FEATURES] + list(NUMERIC_FEATURES)
        )
        raw = self._raw_matrix(sorted_records)
        medians = np.nanmedian(raw, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isfinite(raw), raw, medians)
        self.medians = medians.astype(np.float64)
        self.minima = filled.min(axis=0)
        maxima = filled.max(axis=0)
        self.ranges = maxima - self.minima
        self.ranges = np.where(self.ranges == 0.0, 1.0, self.ranges)
        return self

    def _raw_matrix(self, records: Sequence[Mapping[str, str]]) -> np.ndarray:
        rows = []
        for record in records:
            categories = [
                self.vocabularies[name].get(_canonical_category(name, record.get(name, "")), 0)
                for name in CATEGORICAL_FEATURES
            ]
            numeric = [_numeric_value(record.get(name, "")) for name in NUMERIC_FEATURES]
            rows.append(categories + numeric)
        return np.asarray(rows, dtype=np.float64)

    def transform(self, records: Sequence[Mapping[str, str]]) -> np.ndarray:
        """Convert records to finite min-max normalized feature vectors."""
        if self.medians is None or self.minima is None or self.ranges is None:
            raise RuntimeError("FeatureEncoder must be fit before calling transform().")
        if not records:
            return np.empty((0, len(self.feature_names)), dtype=np.float64)
        sorted_records = sort_records_by_timestamp(records)
        raw = self._raw_matrix(sorted_records)
        filled = np.where(np.isfinite(raw), raw, self.medians)
        normalized = (filled - self.minima) / self.ranges
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def fit_transform(self, records: Sequence[Mapping[str, str]]) -> np.ndarray:
        """Fit on records and return normalized vectors for those records."""
        self.fit(records)
        return self.transform(records)


def extract_features(
    records: Sequence[Mapping[str, str]], encoder: FeatureEncoder | None = None
) -> tuple[np.ndarray, tuple[str, ...], FeatureEncoder]:
    """Extract normalized vectors and names, fitting an encoder if omitted."""
    if not records:
        raise ValueError("Cannot extract features from an empty record set.")
    feature_encoder = encoder or FeatureEncoder()
    if encoder is None or feature_encoder.medians is None:
        feature_encoder.fit(records)
    vectors = feature_encoder.transform(records)
    return vectors, feature_encoder.feature_names, feature_encoder


def create_temporal_windows(
    features: np.ndarray,
    labels: Sequence[str] | np.ndarray,
    window_size: int,
    stride: int = 1,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build observation windows and the stage ``horizon`` steps ahead.

    For N observations, window size W, horizon H, and stride S, each X sample
    contains W consecutive feature vectors and y contains the stage after that
    window at offset H-1. The output X has shape (windows, W, features).
    """
    matrix = np.asarray(features, dtype=np.float32)
    target = np.asarray(labels, dtype=str)
    if matrix.ndim != 2:
        raise ValueError(f"features must be a 2D array; received shape {matrix.shape}.")
    if len(target) != len(matrix):
        raise ValueError("features and labels must contain the same number of observations.")
    if window_size < 1 or stride < 1 or horizon < 1:
        raise ValueError("window_size, stride, and horizon must all be positive integers.")
    last_start = len(matrix) - window_size - horizon
    if last_start < 0:
        raise ValueError(
            f"Not enough observations ({len(matrix)}) for window_size={window_size} and horizon={horizon}."
        )
    starts = range(0, last_start + 1, stride)
    windows = np.stack([matrix[start : start + window_size] for start in starts])
    targets = np.asarray([target[start + window_size + horizon - 1] for start in starts], dtype=str)
    return windows, targets


@dataclass(frozen=True)
class PreprocessingResult:
    """Prepared model inputs, targets, names, and fitted encoder."""

    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    encoder: FeatureEncoder
    labels: np.ndarray


def preprocess_records(
    records: Sequence[Mapping[str, str]],
    window_size: int,
    stride: int = 1,
    horizon: int = 1,
    label_column: str = "attack_stage",
    encoder: FeatureEncoder | None = None,
) -> PreprocessingResult:
    """Fit/reuse an encoder, extract labels, then produce model-ready windows."""
    ordered = sort_records_by_timestamp(records)
    if not ordered:
        raise ValueError("Cannot preprocess an empty record set.")
    labels = extract_labels(ordered, label_column=label_column)
    vectors, feature_names, fitted = extract_features(ordered, encoder=encoder)
    X, y = create_temporal_windows(vectors, labels, window_size, stride, horizon)
    return PreprocessingResult(X=X, y=y, feature_names=feature_names, encoder=fitted, labels=labels)


def preprocess_csv(
    path: str | Path,
    window_size: int,
    stride: int = 1,
    horizon: int = 1,
    label_column: str = "attack_stage",
    encoder: FeatureEncoder | None = None,
) -> PreprocessingResult:
    """Load CSV and return model-ready normalized temporal sequences."""
    return preprocess_records(
        load_network_csv(path), window_size, stride, horizon, label_column, encoder
    )


