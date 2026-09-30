from __future__ import annotations

import csv
import math
from pathlib import Path


REQUIRED_COLUMNS = {
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
}


def _number(value, default=0.0):
    try:
        if value is None or str(value).strip() == "":
            return float(default)

        value = str(value).strip().replace(",", "")
        number = float(value)

        if not math.isfinite(number):
            return float(default)

        return number
    except (ValueError, TypeError):
        return float(default)


def _stable_code(value):
    """Stable deterministic numeric code for categorical values."""
    text = str(value).strip()

    if not text:
        return 0.0

    # Deterministic hash independent of Python's randomized hash().
    code = 0
    for char in text:
        code = (code * 31 + ord(char)) % 10000

    return float(code)


def _first_available(row, names, default=0.0):
    for name in names:
        if name in row and str(row[name]).strip():
            return _number(row[name], default)

    return float(default)


def adapt_cic_csv(input_path, output_path):
    """
    Convert a CIC-IDS2017 GeneratedLabelledFlows CSV into
    the 13-column SIH26153 dashboard schema.

    This is a feature adapter, not a retraining step.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        # Remove accidental surrounding spaces from headers.
        reader.fieldnames = [
            name.strip() if name else name
            for name in reader.fieldnames
        ]

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing:
            raise ValueError(
                "This does not look like the expected CIC-IDS2017 "
                f"GeneratedLabelledFlows CSV. Missing columns: "
                f"{sorted(missing)}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_columns = [
            "timestamp",
            "src_ip",
            "dst_ip",
            "protocol",
            "tcp_flags",
            "src_port",
            "dst_port",
            "packet_count",
            "byte_count",
            "duration",
            "ttl",
            "tcp_window_size",
            "inter_arrival_mean",
            "payload_size_mean",
        ]

        with output_path.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as out:

            writer = csv.DictWriter(
                out,
                fieldnames=output_columns
            )
            writer.writeheader()

            written = 0

            for row in reader:
                src_ip = str(row.get("Source IP", "")).strip()
                dst_ip = str(row.get("Destination IP", "")).strip()

                protocol = _number(row.get("Protocol"), 0)

                src_port = _number(row.get("Source Port"), 0)
                dst_port = _number(row.get("Destination Port"), 0)

                fwd_packets = _number(
                    row.get("Total Fwd Packets"), 0
                )
                bwd_packets = _number(
                    row.get("Total Backward Packets"), 0
                )

                fwd_bytes = _number(
                    row.get("Total Length of Fwd Packets"), 0
                )
                bwd_bytes = _number(
                    row.get("Total Length of Bwd Packets"), 0
                )

                packet_count = fwd_packets + bwd_packets
                byte_count = fwd_bytes + bwd_bytes

                duration = _number(
                    row.get("Flow Duration"), 0
                )

                # CIC-IDS2017 does not provide one generic TTL
                # field in this flow-level CSV.
                ttl = 0.0

                # Use initial TCP window information when available.
                fwd_window = _first_available(
                    row,
                    [
                        "Init_Win_bytes_forward",
                        "Init Win bytes forward",
                    ],
                    0,
                )

                bwd_window = _first_available(
                    row,
                    [
                        "Init_Win_bytes_backward",
                        "Init Win bytes backward",
                    ],
                    0,
                )

                if fwd_window or bwd_window:
                    tcp_window_size = (
                        fwd_window + bwd_window
                    ) / 2.0
                else:
                    tcp_window_size = 0.0

                inter_arrival_mean = _first_available(
                    row,
                    [
                        "Flow IAT Mean",
                        "Flow IAT Mean ",
                    ],
                    0,
                )

                payload_size_mean = _first_available(
                    row,
                    [
                        "Average Packet Size",
                        "Avg Packet Size",
                    ],
                    0,
                )

                writer.writerow({
                    "timestamp": row.get("Timestamp", ""),
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "protocol": protocol,
                    "tcp_flags": 0.0,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "packet_count": packet_count,
                    "byte_count": byte_count,
                    "duration": duration,
                    "ttl": ttl,
                    "tcp_window_size": tcp_window_size,
                    "inter_arrival_mean": inter_arrival_mean,
                    "payload_size_mean": payload_size_mean,
                })

                written += 1

    return written