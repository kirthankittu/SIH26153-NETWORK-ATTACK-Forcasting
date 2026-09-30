"""Descriptions and interpretation guidance for the 13 model input features."""

from __future__ import annotations

FEATURE_METADATA: dict[str, dict[str, str]] = {
    "src_ip_code": {
        "name": "src_ip_code",
        "description": "Encoded category identifier for the source IP address.",
        "category": "network_identity",
        "interpretation": "A categorical code, not an ordered address or identity attribution.",
        "value_type": "categorical",
    },
    "dst_ip_code": {
        "name": "dst_ip_code",
        "description": "Encoded category identifier for the destination IP address.",
        "category": "network_identity",
        "interpretation": "A categorical code, not an ordered address or identity attribution.",
        "value_type": "categorical",
    },
    "protocol_code": {
        "name": "protocol_code",
        "description": "Encoded network protocol category for the flow.",
        "category": "protocol",
        "interpretation": "A categorical identifier; numeric size does not imply protocol severity.",
        "value_type": "categorical",
    },
    "tcp_flags_code": {
        "name": "tcp_flags_code",
        "description": "Encoded TCP flag combination observed for the flow.",
        "category": "transport_control",
        "interpretation": "A categorical identifier; numeric size does not imply flag severity.",
        "value_type": "categorical",
    },
    "src_port": {
        "name": "src_port",
        "description": "Source transport-layer port recorded for the flow.",
        "category": "network_service",
        "interpretation": "A port value provides context but does not identify an application by itself.",
        "value_type": "numeric",
    },
    "dst_port": {
        "name": "dst_port",
        "description": "Destination transport-layer port recorded for the flow.",
        "category": "network_service",
        "interpretation": "A port value provides context but does not establish service use or maliciousness.",
        "value_type": "numeric",
    },
    "packet_count": {
        "name": "packet_count",
        "description": "Number of packets observed in the flow record.",
        "category": "traffic_volume",
        "interpretation": "A change describes packet-count deviation from the configured reference.",
        "value_type": "numeric",
    },
    "byte_count": {
        "name": "byte_count",
        "description": "Total bytes observed in the flow record.",
        "category": "traffic_volume",
        "interpretation": "A change describes byte-count deviation from the configured reference.",
        "value_type": "numeric",
    },
    "duration": {
        "name": "duration",
        "description": "Observed duration of the network flow.",
        "category": "flow_timing",
        "interpretation": "A change describes duration deviation from the configured reference.",
        "value_type": "numeric",
    },
    "ttl": {
        "name": "ttl",
        "description": "Time-to-live value recorded for the network traffic.",
        "category": "packet_header",
        "interpretation": "A recorded header value; it does not identify an operating system with certainty.",
        "value_type": "numeric",
    },
    "tcp_window_size": {
        "name": "tcp_window_size",
        "description": "TCP window-size value recorded for the flow.",
        "category": "transport_configuration",
        "interpretation": "A transport setting that may vary by host, stack, and connection state.",
        "value_type": "numeric",
    },
    "inter_arrival_mean": {
        "name": "inter_arrival_mean",
        "description": "Mean time between packets in the flow record.",
        "category": "timing",
        "interpretation": "A timing deviation from the configured reference; it is not a beacon verdict.",
        "value_type": "numeric",
    },
    "payload_size_mean": {
        "name": "payload_size_mean",
        "description": "Mean observed payload size per packet in the flow record.",
        "category": "traffic_volume",
        "interpretation": "A payload-size deviation from the configured reference; content is not inspected here.",
        "value_type": "numeric",
    },
}


def get_feature_metadata(feature_name: str) -> dict[str, str]:
    """Return a copy of feature metadata or a neutral fallback for custom inputs."""
    if feature_name in FEATURE_METADATA:
        return dict(FEATURE_METADATA[feature_name])
    return {
        "name": feature_name,
        "description": f"Model input feature {feature_name}.",
        "category": "other",
        "interpretation": "Interpret according to the dataset's feature definition.",
        "value_type": "numeric",
    }
