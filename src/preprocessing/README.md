# Network traffic preprocessing

This module converts defensive network-flow CSV records into normalized temporal sequences for future LSTM/GRU training and inference. It deliberately does not import pandas. It uses Python's `csv` and `ipaddress` standard-library modules plus NumPy.

## Supported CSV schema

Minimum required header fields are `timestamp`, `protocol`, and `attack_stage`. Missing required headers produce a `ValueError` listing the missing names. Optional supported fields are:

- `src_ip`, `dst_ip`
- `src_port`, `dst_port`
- `packet_count`, `byte_count`, `duration`
- `tcp_flags`, `ttl`, `tcp_window_size`
- `inter_arrival_mean`, `payload_size_mean`

Optional fields with missing columns, blank values, non-numeric values, NaN, or infinity are handled safely. Numeric missing values are imputed with medians calculated during `FeatureEncoder.fit`; if an entire numeric feature is missing, its fitted median is zero. Absent categorical columns receive a missing-category code. Records are sorted stably by timestamp when loaded and whenever transformed.

## Preprocessing steps

1. `load_network_csv()` reads rows with `csv.DictReader`, normalizes header case/spacing, validates the minimum schema, and sorts by timestamp.
2. `FeatureEncoder.fit()` learns deterministic category vocabularies and per-feature numeric medians/minima/ranges from the supplied fit records.
3. `FeatureEncoder.transform()` converts IP addresses, protocol, and TCP flags to category codes; converts the remaining selected fields to numbers; imputes missing numbers; and min-max normalizes to finite float32 vectors clipped to [0, 1]. Unknown categories map to code zero.
4. `extract_labels()` reads string labels from `attack_stage` (or a selected label column).
5. `create_temporal_windows()` stacks N consecutive observations into each X sample and assigns the label `horizon` steps after the end of each window to y.

For sound evaluation, fit the encoder using training records only, then pass that same encoder to `preprocess_records()`/`preprocess_csv()` for validation, test, or inference records. The convenience path without an encoder fits on all records and is intended for exploration and the synthetic end-to-end demonstration only.

## Feature list

The stable output order is:

1. `src_ip_code`
2. `dst_ip_code`
3. `protocol_code`
4. `tcp_flags_code`
5. `src_port`
6. `dst_port`
7. `packet_count`
8. `byte_count`
9. `duration`
10. `ttl`
11. `tcp_window_size`
12. `inter_arrival_mean`
13. `payload_size_mean`

Categorical codes use deterministic sorted vocabularies. They are numeric identifiers, not ordinal measurements.

## Temporal windows and model handoff

```python
from src.preprocessing.feature_engineering import preprocess_csv

prepared = preprocess_csv("data/sample/sample_network_traffic.csv", window_size=20, stride=1, horizon=1)
# prepared.X: (number_of_windows, 20, number_of_features), float32
# prepared.y: (number_of_windows,), string stage labels for the next observation
# prepared.feature_names: feature column order
# prepared.encoder: fitted encoder to persist/reuse with the eventual model
```

The future LSTM/GRU consumes `X` as batches of `[time steps, features]`; `y` supplies the next-stage class labels. A later training layer should split chronologically before fitting the encoder, encode the label vocabulary for the model, and persist that label mapping alongside the fitted feature encoder and model. No model is trained here and no performance values are produced.
