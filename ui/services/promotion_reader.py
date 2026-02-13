"""Promotion reader — scans promotions dir, derives lifecycle state.

Lifecycle is derived from filesystem artifacts on every call.
No caching. Filenames are for human readability — lifecycle derivation
reads the 'tier' field inside each JSON file.
"""

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

PROMOTION_REQUIRED_FIELDS = {
    "strategy_config_hash", "composition_spec_hash", "dataset_prefix",
    "runner_economics", "triage_result_summary", "timestamp", "tier",
}


def derive_lifecycle_state(
    composition_id: str,
    latest_compiled_hash: Optional[str],
) -> Tuple[str, int, Optional[str]]:
    """Derive lifecycle state from promotion artifacts.

    Returns (state, dataset_count, warning_message).
    Lifecycle is always re-derived from filesystem — never cached.
    """
    if not latest_compiled_hash:
        return "DRAFT", 0, None

    hash_val = latest_compiled_hash
    if hash_val.startswith("sha256:"):
        hash_val = hash_val[7:]

    # Check resolved artifact exists
    strategies_dir = os.path.join(RESEARCH_DIR, "strategies", hash_val)
    resolved_path = os.path.join(strategies_dir, "resolved.json")

    if not os.path.exists(resolved_path):
        return "CORRUPTED", 0, "Resolved artifact missing — recompile required"

    # Scan promotion artifacts
    promotions_dir = os.path.join(RESEARCH_DIR, "promotions", hash_val)
    if not os.path.isdir(promotions_dir):
        return "COMPILED", 0, None

    highest_tier_idx = LIFECYCLE_TIERS.index("COMPILED")
    dataset_prefixes = set()

    for filename in os.listdir(promotions_dir):
        if not filename.endswith(".json"):
            continue
        try:
            filepath = os.path.join(promotions_dir, filename)
            with open(filepath) as f:
                artifact = json.load(f)

            # Validate required fields — skip invalid promotions
            if not PROMOTION_REQUIRED_FIELDS.issubset(artifact.keys()):
                continue

            # Read tier from JSON content, not filename
            tier = artifact.get("tier", "")
            if tier not in LIFECYCLE_TIERS:
                continue

            tier_idx = LIFECYCLE_TIERS.index(tier)
            if tier_idx > highest_tier_idx:
                highest_tier_idx = tier_idx

            # Track datasets
            ds = artifact.get("dataset_prefix", "")
            if ds:
                dataset_prefixes.add(ds)

        except (json.JSONDecodeError, IOError):
            continue

    return LIFECYCLE_TIERS[highest_tier_idx], len(dataset_prefixes), None


def get_promotion_details(
    latest_compiled_hash: str,
) -> List[Dict[str, Any]]:
    """Get all valid promotion artifacts for a hash."""
    hash_val = latest_compiled_hash
    if hash_val.startswith("sha256:"):
        hash_val = hash_val[7:]

    promotions_dir = os.path.join(RESEARCH_DIR, "promotions", hash_val)
    if not os.path.isdir(promotions_dir):
        return []

    promotions = []
    for filename in os.listdir(promotions_dir):
        if not filename.endswith(".json"):
            continue
        try:
            filepath = os.path.join(promotions_dir, filename)
            with open(filepath) as f:
                artifact = json.load(f)
            if PROMOTION_REQUIRED_FIELDS.issubset(artifact.keys()):
                promotions.append(artifact)
        except (json.JSONDecodeError, IOError):
            continue

    return promotions


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
