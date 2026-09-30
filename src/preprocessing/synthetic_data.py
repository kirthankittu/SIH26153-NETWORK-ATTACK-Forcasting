"""Deterministic synthetic network-flow generator with repeated attack episodes.

All labels and traffic are fictional, benign demonstration data. No packets,
exploits, payloads, credentials, or external data are generated or downloaded.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

CSV_FIELDS = (
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "packet_count", "byte_count", "duration", "tcp_flags", "ttl",
    "tcp_window_size", "inter_arrival_mean", "payload_size_mean", "attack_stage",
)
STAGES = (
    "BENIGN", "RECONNAISSANCE", "INITIAL_ACCESS", "LATERAL_MOVEMENT", "COMMAND_AND_CONTROL"
)


def count_attack_episodes(records: Sequence[dict[str, object]]) -> int:
    """Count contiguous non-BENIGN periods in chronological labeled records."""
    count = 0
    was_benign = True
    for record in records:
        is_benign = str(record.get("attack_stage", "")).upper() == "BENIGN"
        if was_benign and not is_benign:
            count += 1
        was_benign = is_benign
    return count


def _stage_record(
    stage: str,
    episode: int,
    offset: int,
    tick: int,
    rng: random.Random,
    started: datetime,
) -> dict[str, object]:
    """Make a synthetic flow observation with stage-dependent traffic patterns."""
    source_host = 10 + ((episode * 3 + offset) % 40)
    destination_host = 60 + ((episode * 7 + offset * (episode % 5 + 1)) % 80)
    source_ip = f"192.168.{1 + episode % 4}.{source_host}"
    destination_ip = f"10.{episode % 8}.{(episode * 3 + offset) % 200}.{destination_host}"
    source_port = rng.randint(49152, 65535)
    protocol = rng.choice(("TCP", "TCP", "UDP"))
    flags = rng.choice(("ACK", "PSH+ACK", "ACK"))
    packet_count = rng.randint(1, 7)
    byte_count = rng.randint(120, 1100)
    duration = rng.uniform(0.03, 0.8)
    destination_port = rng.choice((53, 80, 123, 443, 8080))
    ttl = rng.choice((48, 56, 64, 64, 128))
    window = rng.choice((4096, 8192, 16384, 32768, 65535))
    inter_arrival = rng.uniform(0.08, 1.6)

    if stage == "RECONNAISSANCE":
        # Repeated short, low-byte observations with destinations/ports varying.
        destination_host = 80 + (offset * 7 + episode) % 60
        destination_ip = f"10.{episode % 8}.{offset % 200}.{destination_host}"
        destination_port = 20 + (offset * 13 + episode * 3) % 160
        packet_count = rng.randint(8, 30)
        byte_count = rng.randint(300, 1800)
        duration = rng.uniform(0.01, 0.20)
        flags = rng.choice(("SYN", "SYN", "ACK"))
        inter_arrival = rng.uniform(0.004, 0.09)
        protocol = "TCP"
    elif stage == "INITIAL_ACCESS":
        destination_port = rng.choice((443, 8443, 8080, 80))
        packet_count = rng.randint(6, 20)
        byte_count = rng.randint(1800, 8500)
        duration = rng.uniform(0.4, 5.0)
        flags = rng.choice(("SYN+ACK", "PSH+ACK", "ACK"))
        inter_arrival = rng.uniform(0.025, 0.30)
        protocol = rng.choice(("TCP", "TCP", "UDP"))
    elif stage == "LATERAL_MOVEMENT":
        source_host = 10 + (episode * 5 + offset // 2) % 40
        destination_host = 60 + (episode * 11 + offset) % 80
        source_ip = f"192.168.{1 + (episode + offset // 3) % 4}.{source_host}"
        destination_ip = f"192.168.{1 + (episode + offset // 2) % 4}.{destination_host}"
        destination_port = rng.choice((22, 445, 3389, 5985))
        packet_count = rng.randint(12, 44)
        byte_count = rng.randint(1400, 11000)
        duration = rng.uniform(0.2, 3.2)
        flags = rng.choice(("ACK", "PSH+ACK", "ACK"))
        inter_arrival = rng.uniform(0.015, 0.22)
        protocol = "TCP"
    elif stage == "COMMAND_AND_CONTROL":
        # Low-volume regular traffic to a stable fictional endpoint per episode.
        source_host = 10 + (episode * 3) % 40
        destination_host = 200 + episode % 40
        source_ip = f"192.168.{1 + episode % 4}.{source_host}"
        destination_ip = f"198.51.100.{destination_host}"
        destination_port = rng.choice((443, 8443))
        packet_count = rng.randint(2, 6)
        byte_count = rng.randint(90, 520)
        duration = rng.uniform(0.01, 0.16)
        flags = rng.choice(("PSH+ACK", "ACK"))
        inter_arrival = rng.uniform(18.0, 42.0)
        protocol = "TCP"

    byte_count += rng.randint(-min(byte_count // 8, 20), min(byte_count // 8, 20))
    payload_mean = max(0.0, byte_count / max(packet_count, 1) * rng.uniform(0.85, 1.15))
    return {
        "timestamp": (started + timedelta(seconds=2 * tick)).isoformat().replace("+00:00", "Z"),
        "src_ip": source_ip,
        "dst_ip": destination_ip,
        "src_port": source_port,
        "dst_port": destination_port,
        "protocol": protocol,
        "packet_count": packet_count,
        "byte_count": byte_count,
        "duration": round(duration, 5),
        "tcp_flags": flags,
        "ttl": ttl,
        "tcp_window_size": window,
        "inter_arrival_mean": round(inter_arrival, 5),
        "payload_size_mean": round(payload_mean, 5),
        "attack_stage": stage,
    }


def generate_synthetic_records(
    num_episodes: int = 20,
    seed: int = 26153,
    minimum_records: int = 1000,
) -> list[dict[str, object]]:
    """Generate varied, ordered benign periods and multiple attack episodes.

    Most episodes traverse the full staged progression. A deterministic subset
    terminate after reconnaissance, initial access, or lateral movement. The
    final episode reaches C2 so the file ends with an attack-stage observation.
    """
    if num_episodes < 3:
        raise ValueError("num_episodes must be at least 3 to distribute episodes across time.")
    if minimum_records < 1:
        raise ValueError("minimum_records must be positive.")
    rng = random.Random(seed)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    tick = 0
    for episode in range(num_episodes):
        # Variable background length and baseline intensity between incidents.
        benign_duration = rng.randint(28, 56)
        for offset in range(benign_duration):
            rows.append(_stage_record("BENIGN", episode, offset, tick, rng, now))
            tick += 1

        recon_duration = rng.randint(7, 15)
        initial_duration = rng.randint(6, 13)
        lateral_duration = rng.randint(7, 15)
        c2_duration = rng.randint(7, 16)
        if episode == num_episodes - 1:
            terminal_stage = "COMMAND_AND_CONTROL"
        elif episode % 9 == 4:
            terminal_stage = "RECONNAISSANCE"
        elif episode % 7 == 2:
            terminal_stage = "INITIAL_ACCESS"
        elif episode % 5 == 3:
            terminal_stage = "LATERAL_MOVEMENT"
        else:
            terminal_stage = "COMMAND_AND_CONTROL"

        phase_durations = [
            ("RECONNAISSANCE", recon_duration),
            ("INITIAL_ACCESS", initial_duration),
            ("LATERAL_MOVEMENT", lateral_duration),
            ("COMMAND_AND_CONTROL", c2_duration),
        ]
        stage_order = ["RECONNAISSANCE", "INITIAL_ACCESS", "LATERAL_MOVEMENT", "COMMAND_AND_CONTROL"]
        end_index = stage_order.index(terminal_stage)
        for stage, duration in phase_durations[: end_index + 1]:
            # Repeated observations within a stage include modest intensity drift.
            for offset in range(duration):
                rows.append(_stage_record(stage, episode, offset, tick, rng, now))
                tick += 1

    # Extend with ordinary baseline if unusually short, preserving chronology.
    filler_episode = num_episodes
    filler_offset = 0
    while len(rows) < minimum_records:
        rows.append(_stage_record("BENIGN", filler_episode, filler_offset, tick, rng, now))
        tick += 1
        filler_offset += 1
    return rows


def generate_synthetic_dataset(
    output_path: str | Path,
    num_episodes: int = 20,
    seed: int = 26153,
    minimum_records: int = 1000,
) -> list[dict[str, object]]:
    """Generate and write a synthetic CSV, returning its chronological rows."""
    rows = generate_synthetic_records(num_episodes, seed, minimum_records)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
