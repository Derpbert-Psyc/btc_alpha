"""Indicator catalog service — reads INDICATOR_OUTPUTS from framework."""

import json
import os
from typing import Any, Dict, List

from strategy_framework_v1_8_0 import (
    INDICATOR_ID_TO_NAME,
    INDICATOR_NAME_TO_ID,
    INDICATOR_OUTPUTS,
    compute_instance_warmup,
    get_warmup_bars_for_output,
)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "indicator_groups.json")


def load_indicator_groups() -> List[Dict[str, Any]]:
    """Load indicator group assignments from config JSON."""
    with open(_CONFIG_PATH) as f:
        data = json.load(f)
    return data["groups"]


def get_indicator_info(indicator_id: int) -> Dict[str, Any]:
    """Get full info for an indicator by ID."""
    name = INDICATOR_ID_TO_NAME.get(indicator_id, f"unknown_{indicator_id}")
    outputs = INDICATOR_OUTPUTS.get(indicator_id, {})
    return {
        "id": indicator_id,
        "name": name,
        "outputs": outputs,
    }


def get_all_indicators() -> List[Dict[str, Any]]:
    """Get all indicators with group assignments."""
    groups = load_indicator_groups()
    id_to_group = {}
    for g in groups:
        for iid in g["indicator_ids"]:
            id_to_group[iid] = g["name"]

    result = []
    for iid in sorted(INDICATOR_OUTPUTS.keys()):
        info = get_indicator_info(iid)
        info["group"] = id_to_group.get(iid, "Other")
        result.append(info)
    return result


def get_outputs_for_indicator(indicator_id: int) -> Dict[str, str]:
    """Get output name -> semantic type mapping for an indicator."""
    return dict(INDICATOR_OUTPUTS.get(indicator_id, {}))


def resolve_indicator_id(name_or_id) -> int:
    """Resolve string name or int ID to int ID."""
    if isinstance(name_or_id, int):
        return name_or_id
    return INDICATOR_NAME_TO_ID.get(name_or_id, -1)
