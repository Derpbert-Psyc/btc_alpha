"""Promotion reader — scans promotions dir, derives lifecycle state."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "research")

LIFECYCLE_TIERS = [
    "DRAFT",
    "COMPILED",
    "TRIAGE_PASSED",
    "BASELINE_PLUS_PASSED",
    "SHADOW_VALIDATED",
    "LIVE_APPROVED",
]

TIER_MAP = {
    "TRIAGE": "TRIAGE_PASSED",
    "BASELINE_PLUS": "BASELINE_PLUS_PASSED",
    "SHADOW": "SHADOW_VALIDATED",
    "LIVE": "LIVE_APPROVED",
}


def derive_lifecycle_state(
    composition_id: str,
    latest_compiled_hash: Optional[str],
) -> Tuple[str, Optional[str]]:
    """Derive lifecycle state from promotion artifacts.

    Returns (state, warning_message).
    """
    if not latest_compiled_hash:
        return "DRAFT", None

    hash_val = latest_compiled_hash
    if hash_val.startswith("sha256:"):
        hash_val = hash_val[7:]

    strategies_dir = os.path.join(RESEARCH_DIR, "strategies", hash_val)
    resolved_path = os.path.join(strategies_dir, "resolved.json")
    report_path = os.path.join(strategies_dir, "lowering_report.json")

    if not os.path.exists(resolved_path):
        if not os.path.exists(report_path):
            return "CORRUPTED", "Artifact incomplete"
        return "CORRUPTED", "Artifact missing — recompile required"

    promotions_dir = os.path.join(RESEARCH_DIR, "promotions", hash_val)
    if not os.path.isdir(promotions_dir):
        return "COMPILED", None

    highest_tier_idx = LIFECYCLE_TIERS.index("COMPILED")
    for filename in os.listdir(promotions_dir):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(promotions_dir, filename)) as f:
                artifact = json.load(f)
            if artifact.get("result") != "PASS":
                continue
            tier = artifact.get("tier", "")
            mapped = TIER_MAP.get(tier, "")
            if mapped in LIFECYCLE_TIERS:
                idx = LIFECYCLE_TIERS.index(mapped)
                if idx > highest_tier_idx:
                    highest_tier_idx = idx
        except (json.JSONDecodeError, IOError):
            continue

    return LIFECYCLE_TIERS[highest_tier_idx], None


def derive_binding_state(
    spec: Dict[str, Any],
    latest_compiled_hash: Optional[str],
) -> str:
    """Check if current spec matches the promoted config.

    Returns 'current' or 'stale'.
    """
    if not latest_compiled_hash:
        return "current"

    from strategy_framework_v1_8_0 import compute_config_hash
    current_hash = compute_config_hash(spec)

    hash_val = latest_compiled_hash
    if hash_val.startswith("sha256:"):
        hash_val = hash_val[7:]

    promotions_dir = os.path.join(RESEARCH_DIR, "promotions", hash_val)
    if not os.path.isdir(promotions_dir):
        return "current"

    for filename in os.listdir(promotions_dir):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(promotions_dir, filename)) as f:
                artifact = json.load(f)
            spec_hash = artifact.get("composition_spec_hash", "")
            if spec_hash and spec_hash != current_hash:
                return "stale"
        except (json.JSONDecodeError, IOError):
            continue

    return "current"
