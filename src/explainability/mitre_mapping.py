"""Central stage-to-MITRE ATT&CK context mappings.

Mappings are potentially relevant context for a forecast stage, not evidence
that a tactic or technique occurred.
"""

from __future__ import annotations

from copy import deepcopy

_STAGE_MAPPINGS = {
    "BENIGN": {
        "stage": "BENIGN",
        "tactic_id": None,
        "tactic": None,
        "potential_techniques": [],
    },
    "RECONNAISSANCE": {
        "stage": "RECONNAISSANCE",
        "tactic_id": "TA0043",
        "tactic": "Reconnaissance",
        "potential_techniques": [
            {
                "id": "T1595",
                "name": "Active Scanning",
                "reason": "The synthetic stage includes probing-like changes in destination and port activity; flow metadata alone cannot establish scanning intent.",
                "url": "https://attack.mitre.org/techniques/T1595/",
            }
        ],
    },
    "INITIAL_ACCESS": {
        "stage": "INITIAL_ACCESS",
        "tactic_id": "TA0001",
        "tactic": "Initial Access",
        "potential_techniques": [
            {
                "id": "T1190",
                "name": "Exploit Public-Facing Application",
                "reason": "The prototype's synthetic initial-access phase models unusual traffic toward service ports; the CSV has no evidence of vulnerability exploitation.",
                "url": "https://attack.mitre.org/techniques/T1190/",
            }
        ],
    },
    "LATERAL_MOVEMENT": {
        "stage": "LATERAL_MOVEMENT",
        "tactic_id": "TA0008",
        "tactic": "Lateral Movement",
        "potential_techniques": [
            {
                "id": "T1021",
                "name": "Remote Services",
                "reason": "The synthetic stage varies internal source/destination relationships and remote-service ports; flow metadata alone cannot show remote logon or control.",
                "url": "https://attack.mitre.org/techniques/T1021/",
            }
        ],
    },
    "COMMAND_AND_CONTROL": {
        "stage": "COMMAND_AND_CONTROL",
        "tactic_id": "TA0011",
        "tactic": "Command and Control",
        "potential_techniques": [
            {
                "id": "T1071",
                "name": "Application Layer Protocol",
                "reason": "The synthetic stage models periodic low-volume traffic over common transport ports; flow features do not identify application payloads or prove command traffic.",
                "url": "https://attack.mitre.org/techniques/T1071/",
            }
        ],
    },
}
_DISCLAIMER = "A predicted stage only suggests contextual ATT&CK relevance; it does not establish that a tactic or technique occurred."


def get_mitre_mapping(stage: str) -> dict[str, object]:
    """Return a structured, independent mapping for a prototype attack stage."""
    normalized = str(stage).strip().upper()
    if normalized not in _STAGE_MAPPINGS:
        raise ValueError(f"Unsupported attack stage: {stage!r}")
    result = deepcopy(_STAGE_MAPPINGS[normalized])
    result["disclaimer"] = _DISCLAIMER
    return result


def map_attack_stage_to_mitre(stage: str) -> dict[str, object]:
    """Alias with an explicit name for downstream consumers."""
    return get_mitre_mapping(stage)


def all_mitre_mappings() -> dict[str, dict[str, object]]:
    """Return mappings for all prototype stages."""
    return {stage: get_mitre_mapping(stage) for stage in _STAGE_MAPPINGS}
