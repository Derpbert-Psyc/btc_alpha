"""
Composition Compiler v1.5.2

Lowering pipeline from Composition Spec (authoring form) to resolved
Strategy Framework v1.8.0 artifact.

Implements STRATEGY_COMPOSITION_CONTRACT_v1_5_2.md:
  - 10-step lowering pipeline (SS4.2)
  - Capability registry with version gating
  - Indicator name -> ID resolution
  - Role expansion, fallback expansion, unit lowering
  - Canonical ordering and hash computation
  - Resolved artifact + lowering report emission
  - Promotion artifact writing

The composition layer is compile-time only.  The engine loads only
resolved Strategy Framework configs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from strategy_framework_v1_8_0 import (
    INDICATOR_NAME_TO_ID,
    INDICATOR_OUTPUTS,
    DIAGNOSTIC_OUTPUTS,
    compute_instance_warmup,
    get_warmup_bars_for_output,
    canonical_json,
    compute_config_hash,
    compute_raw_hash,
    validate_schema_strict,
    version_gte,
)

# ---------------------------------------------------------------------------
# Compiler constants
# ---------------------------------------------------------------------------

COMPILER_VERSION = "1.0.0"
LOWERING_PIPELINE_VERSION = "1.0.0"
CAPABILITY_REGISTRY_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Capability registry  (Composition SS3.4, SS9)
# ---------------------------------------------------------------------------
# Maps capability name -> minimum engine version required.

CAPABILITY_REGISTRY: Dict[str, Dict[str, str]] = {
    # Operators
    "op_gt":             {"min_engine_version": "1.0.0"},
    "op_lt":             {"min_engine_version": "1.0.0"},
    "op_gte":            {"min_engine_version": "1.0.0"},
    "op_lte":            {"min_engine_version": "1.0.0"},
    "op_eq":             {"min_engine_version": "1.0.0"},
    "op_crosses_above":  {"min_engine_version": "1.0.0"},
    "op_crosses_below":  {"min_engine_version": "1.0.0"},
    "op_is_present":     {"min_engine_version": "1.8.0"},
    "op_is_absent":      {"min_engine_version": "1.8.0"},
    # Exit types
    "exit_signal":       {"min_engine_version": "1.0.0"},
    "exit_stop_loss":    {"min_engine_version": "1.0.0"},
    "exit_trailing_stop": {"min_engine_version": "1.0.0"},
    "exit_time_limit":   {"min_engine_version": "1.0.0"},
    "exit_gate_exit":    {"min_engine_version": "1.0.0"},
    "exit_mtm_drawdown": {"min_engine_version": "1.8.0"},
    # Gate policies
    "gate_exit_policy_force_flat":  {"min_engine_version": "1.0.0"},
    "gate_exit_policy_hold":        {"min_engine_version": "1.0.0"},
    "gate_exit_policy_handoff":     {"min_engine_version": "1.8.0"},
    # Features
    "cross_indicator_ref": {"min_engine_version": "1.8.0"},
    "per_output_warmup":   {"min_engine_version": "1.8.0"},
}


def _capability_registry_hash() -> str:
    """Compute SHA-256 of the canonical registry JSON."""
    return compute_raw_hash(canonical_json(CAPABILITY_REGISTRY))


# ---------------------------------------------------------------------------
# Framework schema definition (for target engine version)
# ---------------------------------------------------------------------------
# The schema version matches the target engine version.
# Schema hash is computed from the allowed field sets.

FRAMEWORK_SCHEMA_VERSION = "1.8.0"

_FRAMEWORK_SCHEMA_FIELDS = {
    "top": ["engine_version", "indicator_instances", "entry_rules",
            "exit_rules", "gate_rules", "execution_params", "archetype_tags"],
    "indicator_instance": ["label", "indicator_id", "timeframe", "parameters",
                           "outputs_used", "role", "data_source",
                           "bar_provider", "warmup_bars"],
    "entry_path": ["name", "direction", "evaluation_cadence", "conditions",
                   "condition_groups"],
    "exit_path": ["name", "applies_to", "evaluation_cadence", "type",
                  "conditions", "parameters"],
    "gate_rule": ["name", "conditions", "on_close_policy"],
    "condition": ["indicator", "output", "operator", "value",
                  "ref_indicator", "ref_output"],
}


def _framework_schema_hash() -> str:
    """Compute SHA-256 of the canonical schema JSON."""
    return compute_raw_hash(canonical_json(_FRAMEWORK_SCHEMA_FIELDS))


# ---------------------------------------------------------------------------
# Valid timeframes  (Framework SS6.1)
# ---------------------------------------------------------------------------

VALID_BAR_TIMEFRAMES = {
    "1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "3d",
}
VALID_SNAPSHOT_TIMEFRAMES_REGEX = re.compile(r'^[1-9][0-9]*(ms|s)$')

_VALID_BAR_TF_RE = re.compile(r'^([1-9][0-9]*)(m|h|d)$')


def _is_valid_bar_timeframe(tf: str) -> bool:
    """Validate custom bar timeframe: must divide a day or be whole days."""
    m = _VALID_BAR_TF_RE.match(tf)
    if not m:
        return False
    value, unit = int(m.group(1)), m.group(2)
    if unit == "m":
        minutes = value
    elif unit == "h":
        minutes = value * 60
    elif unit == "d":
        minutes = value * 1440
    else:
        return False
    if minutes <= 1440:
        return 1440 % minutes == 0
    return minutes % 1440 == 0


# Timeframe to seconds mapping (for warmup calculation)
TIMEFRAME_SECONDS: Dict[str, int] = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "12h": 43200, "1d": 86400, "3d": 259200,
}


def _parse_tf_seconds(tf: str) -> int:
    """Parse any valid bar timeframe to seconds."""
    if tf in TIMEFRAME_SECONDS:
        return TIMEFRAME_SECONDS[tf]
    m = _VALID_BAR_TF_RE.match(tf)
    if m:
        val, unit = int(m.group(1)), m.group(2)
        if unit == "m":
            return val * 60
        if unit == "h":
            return val * 3600
        if unit == "d":
            return val * 86400
    return 60


# ---------------------------------------------------------------------------
# Operator name mapping
# ---------------------------------------------------------------------------

OPERATOR_TO_CAPABILITY = {
    ">": "op_gt", "<": "op_lt", ">=": "op_gte", "<=": "op_lte",
    "==": "op_eq", "crosses_above": "op_crosses_above",
    "crosses_below": "op_crosses_below",
    "is_present": "op_is_present", "is_absent": "op_is_absent",
}

EXIT_TYPE_TO_CAPABILITY = {
    "SIGNAL": "exit_signal", "STOP_LOSS": "exit_stop_loss",
    "TRAILING_STOP": "exit_trailing_stop", "TIME_LIMIT": "exit_time_limit",
    "GATE_EXIT": "exit_gate_exit", "MTM_DRAWDOWN_EXIT": "exit_mtm_drawdown",
}

GATE_POLICY_TO_CAPABILITY = {
    "FORCE_FLAT": "gate_exit_policy_force_flat",
    "HOLD": "gate_exit_policy_hold",
    "HANDOFF": "gate_exit_policy_handoff",
}

# Gate exit_policy -> Framework on_close_policy mapping (Composition SS2.6)
GATE_POLICY_MAP = {
    "FORCE_FLAT": "FORCE_FLAT",
    "HOLD": "HOLD_CURRENT",
    "HANDOFF": "HANDOFF",
}

# Archetype tag regex (Composition SS2.2)
ARCHETYPE_TAG_REGEX = re.compile(r'^[a-z][a-z0-9_]*$')


# ---------------------------------------------------------------------------
# Compilation errors
# ---------------------------------------------------------------------------

class CompilationError(Exception):
    """Fatal compilation error."""


class CompilationWarning:
    """Non-fatal warning."""
    def __init__(self, message: str, category: str = "general"):
        self.message = message
        self.category = category

    def __repr__(self):
        return f"Warning({self.category}): {self.message}"


# ---------------------------------------------------------------------------
# Lowering pipeline
# ---------------------------------------------------------------------------

def compile_composition(
    spec: dict,
    validation_context: str = "RESEARCH",
) -> Dict[str, Any]:
    """Run the full lowering pipeline.

    Returns dict with:
      resolved_artifact: dict  -- the Strategy Framework config
      strategy_config_hash: str
      lowering_report: dict
      lowering_report_semantic_hash: str
      lowering_report_full_hash: str

    Raises CompilationError on fatal errors.
    """
    spec = copy.deepcopy(spec)
    target_ev = spec.get("target_engine_version", "1.8.0")
    transformations: List[dict] = []
    warnings: List[str] = []

    # Step 0: Load (already done -- spec is dict)

    # Step 1: Schema validation
    step1_validate(spec, target_ev, warnings)

    # Step 2: Indicator catalog validation
    warmup_info = step2_catalog_validate(spec, target_ev)

    # Step 3: Constraint validation
    step3_constraints(spec, validation_context, warnings)

    # Step 4: Role expansion
    step4_role_expand(spec, transformations)

    # Step 5: Fallback expansion
    step5_fallback_expand(spec, target_ev, transformations)

    # Step 6: Unit lowering
    step6_unit_lower(spec, transformations)

    # Step 7: Strip authoring-only fields
    step7_strip(spec, transformations)

    # Step 8: Canonicalize ordering
    resolved = step8_canonicalize(spec)

    # Step 9: Emit resolved config
    resolved["engine_version"] = target_ev
    _inject_nulls_and_defaults(resolved)

    # Step 10: Hash
    strategy_config_hash = compute_config_hash(resolved)

    # Build lowering report
    composition_spec_hash = compute_config_hash(spec)
    report = _build_lowering_report(
        spec, resolved, strategy_config_hash, composition_spec_hash,
        transformations, warnings, warmup_info)

    report_semantic_hash = _compute_semantic_hash(report)
    report_full_hash = compute_config_hash(report)

    return {
        "resolved_artifact": resolved,
        "strategy_config_hash": strategy_config_hash,
        "lowering_report": report,
        "lowering_report_semantic_hash": report_semantic_hash,
        "lowering_report_full_hash": report_full_hash,
    }


# ---------------------------------------------------------------------------
# Step 1: Schema validation  (Composition SS4.2 Step 1)
# ---------------------------------------------------------------------------

def step1_validate(spec: dict, target_ev: str,
                   warnings: List[str]) -> None:
    """Validate the composition spec schema."""
    errors: List[str] = []

    # Required top-level fields
    for field in ["target_engine_version", "indicator_instances"]:
        if field not in spec:
            errors.append(f"Missing required field: {field}")

    # Validate indicator instances
    labels_seen: set = set()
    for i, inst in enumerate(spec.get("indicator_instances", [])):
        label = inst.get("label", "")
        if not label:
            errors.append(f"indicator_instances[{i}]: missing label")
        if label in labels_seen:
            errors.append(
                f"indicator_instances[{i}]: duplicate label '{label}'")
        labels_seen.add(label)

        # Validate indicator_id exists
        ind_id = inst.get("indicator_id", "")
        if isinstance(ind_id, str):
            if ind_id not in INDICATOR_NAME_TO_ID:
                errors.append(
                    f"indicator_instances[{i}]: unknown indicator "
                    f"'{ind_id}'")
        elif isinstance(ind_id, int):
            if ind_id not in INDICATOR_OUTPUTS and ind_id not in DIAGNOSTIC_OUTPUTS:
                errors.append(
                    f"indicator_instances[{i}]: unknown indicator_id "
                    f"{ind_id}")

        # Validate timeframe
        tf = inst.get("timeframe", "")
        if tf and tf not in VALID_BAR_TIMEFRAMES:
            if not _is_valid_bar_timeframe(tf) and not VALID_SNAPSHOT_TIMEFRAMES_REGEX.match(tf):
                errors.append(
                    f"indicator_instances[{i}]: invalid timeframe "
                    f"'{tf}'")

        # Validate outputs_used reference valid outputs
        outputs_used = inst.get("outputs_used", [])
        int_id = _resolve_indicator_id(ind_id)
        if int_id is not None:
            known_outputs = set(INDICATOR_OUTPUTS.get(int_id, {}).keys()) | \
                            set(DIAGNOSTIC_OUTPUTS.get(int_id, {}).keys())
            for out in outputs_used:
                if out not in known_outputs:
                    errors.append(
                        f"indicator_instances[{i}]: output '{out}' not "
                        f"valid for indicator {ind_id}")

    # Validate operators in conditions
    _validate_operators_in_spec(spec, target_ev, errors)

    # Validate exit types
    for i, rule in enumerate(spec.get("exit_rules", [])):
        et = rule.get("exit_type", "SIGNAL")
        cap = EXIT_TYPE_TO_CAPABILITY.get(et)
        if cap is None:
            errors.append(f"exit_rules[{i}]: unknown exit_type '{et}'")
        elif cap in CAPABILITY_REGISTRY:
            min_ver = CAPABILITY_REGISTRY[cap]["min_engine_version"]
            if not version_gte(target_ev, min_ver):
                errors.append(
                    f"exit_rules[{i}]: exit type '{et}' requires "
                    f"engine_version >= {min_ver}, target is {target_ev}")

    # Validate gate exit policies
    for i, rule in enumerate(spec.get("gate_rules", [])):
        pol = rule.get("exit_policy", "HOLD")
        cap = GATE_POLICY_TO_CAPABILITY.get(pol)
        if cap is None:
            errors.append(f"gate_rules[{i}]: unknown exit_policy '{pol}'")
        elif cap in CAPABILITY_REGISTRY:
            min_ver = CAPABILITY_REGISTRY[cap]["min_engine_version"]
            if not version_gte(target_ev, min_ver):
                errors.append(
                    f"Capability '{cap}' ({pol}) requires framework "
                    f"v{min_ver} or later.")

    # Validate archetype tags
    for tag in spec.get("archetype_tags", []):
        if not ARCHETYPE_TAG_REGEX.match(tag):
            errors.append(
                f"Invalid archetype tag '{tag}': must match "
                f"^[a-z][a-z0-9_]*$")

    # Validate condition references point to existing labels
    for cond in _all_conditions_in_spec(spec):
        ind = cond.get("indicator", "")
        if ind and ind not in labels_seen:
            # Could be a role reference, skip for now
            pass

    if errors:
        raise CompilationError(
            "Schema validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors))


def _resolve_indicator_id(ind_id) -> Optional[int]:
    """Resolve string or int indicator ID to int."""
    if isinstance(ind_id, int):
        return ind_id
    if isinstance(ind_id, str):
        return INDICATOR_NAME_TO_ID.get(ind_id)
    return None


def _validate_operators_in_spec(spec: dict, target_ev: str,
                                errors: List[str]) -> None:
    """Validate all operators in conditions against target engine version."""
    for cond in _all_conditions_in_spec(spec):
        op = cond.get("operator", "")
        cap = OPERATOR_TO_CAPABILITY.get(op)
        if cap is None:
            errors.append(f"Unknown operator '{op}'")
        elif cap in CAPABILITY_REGISTRY:
            min_ver = CAPABILITY_REGISTRY[cap]["min_engine_version"]
            if not version_gte(target_ev, min_ver):
                errors.append(
                    f"Operator '{op}' requires engine_version >= "
                    f"{min_ver}, target is {target_ev}")


def _all_conditions_in_spec(spec: dict):
    """Yield all condition dicts from the composition spec."""
    for rule in spec.get("entry_rules", []):
        yield from rule.get("conditions", [])
        for grp in rule.get("condition_groups", []):
            if isinstance(grp, dict):
                rc = grp.get("role_condition")
                if rc:
                    yield rc
                yield from grp.get("conditions", [])
    for rule in spec.get("exit_rules", []):
        yield from rule.get("conditions", [])
    for rule in spec.get("gate_rules", []):
        yield from rule.get("conditions", [])


# ---------------------------------------------------------------------------
# Step 2: Indicator catalog validation  (Composition SS4.2 Step 2)
# ---------------------------------------------------------------------------

def step2_catalog_validate(spec: dict, target_ev: str) -> dict:
    """Validate indicators and compute warmup.

    Returns warmup_info dict.
    """
    warmup_info: Dict[str, dict] = {}
    max_warmup_seconds = 0
    dominating_instance = ""

    for inst in spec.get("indicator_instances", []):
        label = inst.get("label", "")
        ind_id = inst.get("indicator_id", "")
        int_id = _resolve_indicator_id(ind_id)
        if int_id is None:
            continue

        outputs_used = inst.get("outputs_used", [])
        params = inst.get("parameters", {})
        tf = inst.get("timeframe", "1m")
        tf_seconds = _parse_tf_seconds(tf)

        instance_warmup = compute_instance_warmup(int_id, outputs_used, params)
        warmup_seconds = instance_warmup * tf_seconds

        warmup_info[label] = {
            "warmup_bars": instance_warmup,
            "timeframe": tf,
            "warmup_seconds": warmup_seconds,
        }

        if (warmup_seconds > max_warmup_seconds or
            (warmup_seconds == max_warmup_seconds and
             instance_warmup > warmup_info.get(dominating_instance, {}).get("warmup_bars", 0)) or
            (warmup_seconds == max_warmup_seconds and
             instance_warmup == warmup_info.get(dominating_instance, {}).get("warmup_bars", 0) and
             label < dominating_instance)):
            max_warmup_seconds = warmup_seconds
            dominating_instance = label

    return {
        "effective_warmup_seconds": max_warmup_seconds,
        "dominating_instance": dominating_instance,
        "per_instance": warmup_info,
    }


# ---------------------------------------------------------------------------
# Step 3: Constraint validation  (Composition SS4.2 Step 3)
# ---------------------------------------------------------------------------

def step3_constraints(spec: dict, context: str,
                      warnings: List[str]) -> None:
    """Validate hard and soft constraints."""
    errors: List[str] = []

    # Hard: at least one exit per enabled direction
    entry_dirs = set()
    for rule in spec.get("entry_rules", []):
        d = rule.get("direction", "")
        if d:
            entry_dirs.add(d)

    exit_dirs = set()
    for rule in spec.get("exit_rules", []):
        applies = rule.get("applies_to", [])
        if isinstance(applies, str):
            applies = [applies]
        for d in applies:
            exit_dirs.add(d)
        if "ANY" in applies:
            exit_dirs.update(entry_dirs)

    # Check stop_loss as alternative exit
    has_stop = False
    ep = spec.get("execution_params", {})
    if isinstance(ep, dict):
        sl = ep.get("stop_loss")
        if sl and isinstance(sl, dict):
            has_stop = True

    for d in entry_dirs:
        if d not in exit_dirs and not has_stop:
            errors.append(
                f"No exit path for direction '{d}' and no stop_loss")

    # Soft: warmup > 90 days
    for inst in spec.get("indicator_instances", []):
        int_id = _resolve_indicator_id(inst.get("indicator_id", ""))
        if int_id is None:
            continue
        warmup = compute_instance_warmup(
            int_id, inst.get("outputs_used", []),
            inst.get("parameters", {}))
        tf_sec = _parse_tf_seconds(inst.get("timeframe", "1m"))
        warmup_days = (warmup * tf_sec) / 86400
        if warmup_days > 90:
            msg = (f"Instance '{inst.get('label', '')}' warmup is "
                   f"{warmup_days:.0f} days (> 90)")
            if context in ("SHADOW", "LIVE"):
                errors.append(msg)
            else:
                warnings.append(msg)

    if errors:
        raise CompilationError(
            "Constraint validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors))


# ---------------------------------------------------------------------------
# Step 4: Role expansion  (Composition SS4.2 Step 4)
# ---------------------------------------------------------------------------

def step4_role_expand(spec: dict, transformations: List[dict]) -> None:
    """Expand role references to concrete indicator labels."""
    # Build role -> instances mapping
    role_map: Dict[str, List[dict]] = {}
    group_map: Dict[Tuple[str, str], List[dict]] = {}

    for inst in spec.get("indicator_instances", []):
        role = inst.get("role", "")
        group = inst.get("group", "")
        if role:
            role_map.setdefault(role, []).append(inst)
            if group:
                group_map.setdefault((role, group), []).append(inst)

    # Expand entry rules
    for rule in spec.get("entry_rules", []):
        # Expand condition_groups with role_conditions
        new_groups = []
        for grp in rule.get("condition_groups", []):
            rc = grp.get("role_condition")
            if rc:
                expanded = _expand_role_condition(
                    rc, role_map, group_map, transformations)
                new_groups.append({
                    "name": grp.get("label", grp.get("name", "")),
                    "conditions": expanded,
                })
            else:
                new_groups.append({
                    "name": grp.get("label", grp.get("name", "")),
                    "conditions": grp.get("conditions", []),
                })
        rule["condition_groups"] = new_groups

        # Expand standalone conditions with role references
        new_conds = []
        for cond in rule.get("conditions", []):
            if "role" in cond:
                expanded = _expand_single_role_condition(
                    cond, role_map, group_map, transformations)
                new_conds.extend(expanded)
            else:
                new_conds.append(cond)
        rule["conditions"] = new_conds

    # Expand exit rule conditions
    for rule in spec.get("exit_rules", []):
        new_conds = []
        for cond in rule.get("conditions", []):
            if "role" in cond:
                expanded = _expand_single_role_condition(
                    cond, role_map, group_map, transformations)
                new_conds.extend(expanded)
            else:
                new_conds.append(cond)
        rule["conditions"] = new_conds


def _expand_role_condition(
    rc: dict,
    role_map: Dict[str, List[dict]],
    group_map: Dict[Tuple[str, str], List[dict]],
    transformations: List[dict],
) -> List[dict]:
    """Expand a role_condition to concrete conditions."""
    role = rc.get("role", "")
    filter_group = rc.get("filter_group", "")
    output = rc.get("output", "")
    operator = rc.get("operator", "")
    value = rc.get("value", "")
    quantifier = rc.get("quantifier", "ALL")

    if filter_group:
        instances = group_map.get((role, filter_group), [])
    else:
        instances = role_map.get(role, [])

    # Sort by (label ASC) for deterministic output
    instances = sorted(instances, key=lambda x: x.get("label", ""))

    conditions = []
    matched_labels = []
    for inst in instances:
        label = inst.get("label", "")
        matched_labels.append(label)
        conditions.append({
            "indicator": label,
            "output": output,
            "operator": operator,
            "value": value,
        })

    transformations.append({
        "step": "role_expansion",
        "role": f"{role}/{filter_group}" if filter_group else role,
        "instances_matched": matched_labels,
    })

    return conditions


def _expand_single_role_condition(
    cond: dict,
    role_map: Dict[str, List[dict]],
    group_map: Dict[Tuple[str, str], List[dict]],
    transformations: List[dict],
) -> List[dict]:
    """Expand a single condition with role reference."""
    role = cond.get("role", "")
    output = cond.get("output", "")
    operator = cond.get("operator", "")
    value = cond.get("value", "")

    instances = role_map.get(role, [])
    instances = sorted(instances, key=lambda x: x.get("label", ""))

    if len(instances) == 1:
        transformations.append({
            "step": "role_expansion",
            "role": role,
            "resolved_to": instances[0].get("label", ""),
        })
        return [{
            "indicator": instances[0].get("label", ""),
            "output": output,
            "operator": operator,
            "value": value,
        }]

    conditions = []
    matched_labels = []
    for inst in instances:
        label = inst.get("label", "")
        matched_labels.append(label)
        conditions.append({
            "indicator": label,
            "output": output,
            "operator": operator,
            "value": value,
        })

    transformations.append({
        "step": "role_expansion",
        "role": role,
        "instances_matched": matched_labels,
    })

    return conditions


# ---------------------------------------------------------------------------
# Step 5: Fallback expansion  (Composition SS4.2 Step 5)
# ---------------------------------------------------------------------------

def step5_fallback_expand(spec: dict, target_ev: str,
                          transformations: List[dict]) -> None:
    """Expand fallback_bindings into explicit exit paths."""
    bindings = spec.get("fallback_bindings", [])
    if not bindings:
        return

    # Check capability
    if not version_gte(target_ev, "1.8.0"):
        raise CompilationError(
            "Fallback bindings require is_present/is_absent operators "
            "which need framework v1.8.0 or later.")

    exit_rules = spec.get("exit_rules", [])

    for binding in bindings:
        primary = binding.get("primary_label", "")
        fallback = binding.get("fallback_label", "")

        # Find exit rules that reference the primary
        for orig_rule in list(exit_rules):
            if _rule_references_label(orig_rule, primary):
                # Generate primary path (is_present guard)
                primary_path = copy.deepcopy(orig_rule)
                primary_path["name"] = f"{orig_rule.get('name', '')}_primary"
                is_present_cond = {
                    "indicator": primary,
                    "output": _get_referenced_output(orig_rule, primary),
                    "operator": "is_present",
                }
                primary_path["conditions"] = [is_present_cond] + \
                    primary_path.get("conditions", [])

                # Generate fallback path (is_absent guard)
                fallback_path = copy.deepcopy(orig_rule)
                fallback_path["name"] = f"{orig_rule.get('name', '')}_fallback"
                is_absent_cond = {
                    "indicator": primary,
                    "output": _get_referenced_output(orig_rule, primary),
                    "operator": "is_absent",
                }
                # Replace primary references with fallback
                fb_conds = []
                for c in fallback_path.get("conditions", []):
                    if c.get("indicator") == primary:
                        c = copy.deepcopy(c)
                        c["indicator"] = fallback
                    fb_conds.append(c)
                # Add is_present check for fallback output
                fb_output = _get_referenced_output(orig_rule, primary)
                fallback_path["conditions"] = [
                    is_absent_cond,
                    {"indicator": fallback, "output": fb_output,
                     "operator": "is_present"},
                ] + fb_conds

                exit_rules.append(primary_path)
                exit_rules.append(fallback_path)

                transformations.append({
                    "step": "fallback_expansion",
                    "primary": primary,
                    "fallback": fallback,
                    "generated_paths": [primary_path["name"],
                                        fallback_path["name"]],
                })

    spec["exit_rules"] = exit_rules


def _rule_references_label(rule: dict, label: str) -> bool:
    """Check if any condition in the rule references the given label."""
    for cond in rule.get("conditions", []):
        if cond.get("indicator") == label:
            return True
    return False


def _get_referenced_output(rule: dict, label: str) -> str:
    """Get the output name referenced for a label in a rule."""
    for cond in rule.get("conditions", []):
        if cond.get("indicator") == label:
            return cond.get("output", "")
    return ""


# ---------------------------------------------------------------------------
# Step 6: Unit lowering  (Composition SS4.2 Step 6)
# ---------------------------------------------------------------------------

def step6_unit_lower(spec: dict, transformations: List[dict]) -> None:
    """Convert authoring-friendly units to framework-native formats."""
    # Convert exit rules to framework format
    new_exit_rules = []
    for rule in spec.get("exit_rules", []):
        et = rule.get("exit_type", "SIGNAL")
        fw_rule = _lower_exit_rule(rule, et, transformations)
        new_exit_rules.append(fw_rule)
    spec["exit_rules"] = new_exit_rules

    # Convert gate rules
    new_gates = []
    for rule in spec.get("gate_rules", []):
        fw_gate = _lower_gate_rule(rule, transformations)
        new_gates.append(fw_gate)
    spec["gate_rules"] = new_gates

    # Convert entry rules
    new_entries = []
    for rule in spec.get("entry_rules", []):
        fw_entry = _lower_entry_rule(rule, transformations)
        new_entries.append(fw_entry)
    spec["entry_rules"] = new_entries

    # Convert indicator instances (resolve string IDs to int)
    for inst in spec.get("indicator_instances", []):
        ind_id = inst.get("indicator_id", "")
        if isinstance(ind_id, str):
            int_id = INDICATOR_NAME_TO_ID.get(ind_id)
            if int_id is not None:
                inst["indicator_id"] = int_id

    # Convert execution_params
    ep = spec.get("execution_params", {})
    if isinstance(ep, dict):
        _lower_execution_params(ep, transformations)
    spec["execution_params"] = ep


def _lower_exit_rule(rule: dict, exit_type: str,
                     transformations: List[dict]) -> dict:
    """Lower an authoring-form exit rule to framework format."""
    fw = {}
    fw["name"] = rule.get("label", rule.get("name", ""))
    fw["type"] = exit_type

    applies_to = rule.get("applies_to", ["ANY"])
    if isinstance(applies_to, str):
        applies_to = [applies_to]
    fw["applies_to"] = sorted(applies_to)

    fw["evaluation_cadence"] = rule.get("evaluation_cadence", "1m")
    fw["conditions"] = rule.get("conditions", [])
    fw["parameters"] = {}

    if exit_type == "STOP_LOSS":
        params = {}
        if "mode" in rule:
            params["mode"] = rule["mode"]
        if "exchange_side" in rule:
            params["exchange_side"] = rule["exchange_side"]
        if "value_long_bps" in rule:
            params["percent_long"] = str(
                Decimal(str(rule["value_long_bps"])) / 10000)
            transformations.append({
                "step": "unit_lowering",
                "field": f"stop_loss.value_long_bps",
                "from": rule["value_long_bps"],
                "to": {"percent_long": params["percent_long"]},
            })
        if "value_short_bps" in rule:
            params["percent_short"] = str(
                Decimal(str(rule["value_short_bps"])) / 10000)
        fw["parameters"] = params

    elif exit_type == "MTM_DRAWDOWN_EXIT":
        params = {}
        if "drawdown_bps_long" in rule:
            params["drawdown_bps_long"] = rule["drawdown_bps_long"]
        if "drawdown_bps_short" in rule:
            params["drawdown_bps_short"] = rule["drawdown_bps_short"]
        if "enabled" in rule:
            params["enabled"] = rule["enabled"]
        fw["parameters"] = params
        fw["evaluation_cadence"] = rule.get("evaluation_cadence", "1m")

    elif exit_type == "TIME_LIMIT":
        params = {}
        if "time_limit_bars" in rule:
            params["time_limit_bars"] = rule["time_limit_bars"]
        if "time_limit_reference_cadence" in rule:
            params["time_limit_reference_cadence"] = \
                rule["time_limit_reference_cadence"]
        fw["parameters"] = params

    return fw


def _lower_gate_rule(rule: dict, transformations: List[dict]) -> dict:
    """Lower an authoring-form gate rule to framework format."""
    policy = rule.get("exit_policy", "HOLD")
    fw_policy = GATE_POLICY_MAP.get(policy, "HOLD_CURRENT")
    return {
        "name": rule.get("label", rule.get("name", "")),
        "conditions": rule.get("conditions", []),
        "on_close_policy": fw_policy,
    }


def _lower_entry_rule(rule: dict, transformations: List[dict]) -> dict:
    """Lower an authoring-form entry rule to framework format."""
    return {
        "name": rule.get("label", rule.get("name", "")),
        "direction": rule.get("direction", "LONG"),
        "evaluation_cadence": rule.get("evaluation_cadence", "1m"),
        "conditions": rule.get("conditions", []),
        "condition_groups": [
            {
                "name": g.get("label", g.get("name", "")),
                "conditions": g.get("conditions", []),
            }
            for g in rule.get("condition_groups", [])
        ],
    }


def _lower_execution_params(ep: dict,
                            transformations: List[dict]) -> None:
    """Lower execution params to framework format."""
    # Ensure all expected sub-objects exist
    if "position_sizing" not in ep:
        ep["position_sizing"] = None
    if "stop_loss" not in ep:
        ep["stop_loss"] = None
    if "entry_type" not in ep:
        ep["entry_type"] = "MARKET"
    if "flip_enabled" not in ep:
        ep["flip_enabled"] = False


# ---------------------------------------------------------------------------
# Step 7: Strip authoring-only fields  (Composition SS4.2 Step 7)
# ---------------------------------------------------------------------------

AUTHORING_ONLY_TOP = {
    "composition_id", "display_name", "description", "version",
    "target_engine_version", "min_engine_version", "target_instrument",
    "target_variant", "metadata", "fallback_bindings",
}

AUTHORING_ONLY_INSTANCE = {
    "role", "group", "filter_group", "description",
}

AUTHORING_ONLY_ENTRY = {
    "label", "entry_type", "description",
}

AUTHORING_ONLY_EXIT = {
    "label", "description", "exit_type", "enabled",
    "mode", "exchange_side", "value_long_bps", "value_short_bps",
    "drawdown_bps_long", "drawdown_bps_short", "time_limit_bars",
    "time_limit_reference_cadence",
}

AUTHORING_ONLY_GATE = {
    "label", "description", "exit_policy",
}

AUTHORING_ONLY_CONDITION = {
    "role", "filter_group", "quantifier",
}


def step7_strip(spec: dict, transformations: List[dict]) -> None:
    """Remove authoring-only fields from the spec."""
    stripped = []

    # Top-level
    for field in list(spec.keys()):
        if field in AUTHORING_ONLY_TOP:
            del spec[field]
            stripped.append(field)

    # Indicator instances
    for inst in spec.get("indicator_instances", []):
        for field in list(inst.keys()):
            if field in AUTHORING_ONLY_INSTANCE:
                del inst[field]
                if field not in stripped:
                    stripped.append(field)

    # Conditions
    for cond in _all_framework_conditions(spec):
        for field in list(cond.keys()):
            if field in AUTHORING_ONLY_CONDITION:
                del cond[field]

    if stripped:
        transformations.append({
            "step": "field_stripped",
            "fields": sorted(stripped),
        })


def _all_framework_conditions(spec: dict):
    """Yield all condition dicts from a partially-lowered spec."""
    for rule in spec.get("entry_rules", []):
        yield from rule.get("conditions", [])
        for grp in rule.get("condition_groups", []):
            yield from grp.get("conditions", [])
    for rule in spec.get("exit_rules", []):
        yield from rule.get("conditions", [])
    for rule in spec.get("gate_rules", []):
        yield from rule.get("conditions", [])


# ---------------------------------------------------------------------------
# Step 8: Canonicalize ordering  (Composition SS4.2 Step 8, NORMATIVE)
# ---------------------------------------------------------------------------

def step8_canonicalize(spec: dict) -> dict:
    """Produce deterministic canonical ordering.

    Semantic-order arrays (preserve declaration order):
      entry_rules, exit_rules, gate_rules,
      condition_groups within entry paths,
      conditions within any path/group

    Non-semantic arrays (sort deterministically):
      indicator_instances -> by label ASC
      applies_to -> alphabetical
      archetype_tags -> alphabetical
      outputs_used -> alphabetical
    """
    resolved: dict = {}

    # Indicator instances: sort by label ASC
    instances = spec.get("indicator_instances", [])
    resolved["indicator_instances"] = sorted(
        instances, key=lambda x: x.get("label", ""))

    # Sort outputs_used within each instance
    for inst in resolved["indicator_instances"]:
        if "outputs_used" in inst:
            inst["outputs_used"] = sorted(inst["outputs_used"])

    # Entry rules: preserve order (semantic)
    resolved["entry_rules"] = spec.get("entry_rules", [])

    # Exit rules: preserve order (semantic)
    resolved["exit_rules"] = spec.get("exit_rules", [])
    for rule in resolved["exit_rules"]:
        if "applies_to" in rule:
            rule["applies_to"] = sorted(rule["applies_to"])

    # Gate rules: preserve order (semantic)
    resolved["gate_rules"] = spec.get("gate_rules", [])

    # Archetype tags: sort alphabetically
    if "archetype_tags" in spec:
        resolved["archetype_tags"] = sorted(spec["archetype_tags"])
    else:
        resolved["archetype_tags"] = []

    # Execution params
    resolved["execution_params"] = spec.get("execution_params", {})

    # Resolve cross-indicator dot-notation values
    _resolve_cross_indicator_refs(resolved)

    return resolved


def _resolve_cross_indicator_refs(resolved: dict) -> None:
    """Resolve dot-notation value references to structured ref fields.

    Composition SS3.3: "value": "bb_4h.upper" becomes
    ref_indicator: "bb_4h", ref_output: "upper" (value field removed).
    """
    labels = {inst.get("label", "") for inst in
              resolved.get("indicator_instances", [])}

    for cond in _all_framework_conditions(resolved):
        val = cond.get("value")
        if isinstance(val, str) and "." in val:
            parts = val.split(".", 1)
            if len(parts) == 2 and parts[0] in labels:
                cond["ref_indicator"] = parts[0]
                cond["ref_output"] = parts[1]
                del cond["value"]


# ---------------------------------------------------------------------------
# Null/default injection  (Composition SS4.2 Step 8)
# ---------------------------------------------------------------------------

def _inject_nulls_and_defaults(resolved: dict) -> None:
    """Inject null for absent optional fields; populate defaults.

    Composition SS4.2 Step 8: all optional fields explicitly null.
    """
    # Top-level defaults
    if "archetype_tags" not in resolved:
        resolved["archetype_tags"] = []

    ep = resolved.get("execution_params", {})
    if not isinstance(ep, dict):
        ep = {}
        resolved["execution_params"] = ep

    # Execution params defaults
    _set_default(ep, "entry_type", "MARKET")
    _set_default(ep, "leverage", None)
    _set_default(ep, "position_sizing", None)
    _set_default(ep, "stop_loss", None)
    _set_default(ep, "take_profit", None)
    _set_default(ep, "trailing_stop", None)
    _set_default(ep, "time_limit_bars", None)
    _set_default(ep, "time_limit_reference_cadence", None)
    _set_default(ep, "time_limit_allows_flip", False)
    _set_default(ep, "flip_enabled", False)
    _set_default(ep, "scale_in", None)
    _set_default(ep, "funding_model", None)
    _set_default(ep, "trade_rate_limit", None)
    _set_default(ep, "slippage_budget", None)
    _set_default(ep, "warmup_restart_policy", None)
    _set_default(ep, "mtm_drawdown_exit", None)
    _set_default(ep, "strict_entry_paths", None)
    _set_default(ep, "strict_exit_paths", None)

    # Indicator instance defaults
    for inst in resolved.get("indicator_instances", []):
        _set_default(inst, "data_source", "BAR")
        _set_default(inst, "bar_provider", None)
        _set_default(inst, "warmup_bars", None)
        _set_default(inst, "role", None)

    # Condition defaults (null for absent optional fields)
    for cond in _all_framework_conditions(resolved):
        _set_default(cond, "value", None)
        _set_default(cond, "ref_indicator", None)
        _set_default(cond, "ref_output", None)

    # Entry path defaults
    for rule in resolved.get("entry_rules", []):
        _set_default(rule, "condition_groups", [])
        _set_default(rule, "conditions", [])

    # Exit path defaults
    for rule in resolved.get("exit_rules", []):
        _set_default(rule, "conditions", [])
        _set_default(rule, "parameters", {})

    # Gate rule defaults (no extra defaults needed beyond conditions)


def _set_default(d: dict, key: str, default: Any) -> None:
    """Set default value if key is absent."""
    if key not in d:
        d[key] = default


# ---------------------------------------------------------------------------
# Lowering report  (Composition SS6.3)
# ---------------------------------------------------------------------------

def _build_lowering_report(
    original_spec: dict,
    resolved: dict,
    strategy_config_hash: str,
    composition_spec_hash: str,
    transformations: List[dict],
    warnings: List[str],
    warmup_info: dict,
) -> dict:
    """Build the lowering report artifact."""
    dom = warmup_info.get("dominating_instance", "")
    dom_info = warmup_info.get("per_instance", {}).get(dom, {})

    return {
        "composition_id": original_spec.get("composition_id", ""),
        "composition_version": original_spec.get("version", "1.0.0"),
        "composition_spec_hash": composition_spec_hash,
        "strategy_config_hash": strategy_config_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "compiler_version": COMPILER_VERSION,
        "lowering_pipeline_version": LOWERING_PIPELINE_VERSION,
        "capability_registry_version": CAPABILITY_REGISTRY_VERSION,
        "capability_registry_hash": f"sha256:{_capability_registry_hash()}",
        "framework_schema_version": FRAMEWORK_SCHEMA_VERSION,
        "framework_schema_hash": f"sha256:{_framework_schema_hash()}",
        "transformations": transformations,
        "warnings": warnings,
        "effective_warmup": {
            "bars": dom_info.get("warmup_bars", 0),
            "timeframe": dom_info.get("timeframe", ""),
            "duration_days": warmup_info.get("effective_warmup_seconds", 0)
                             / 86400,
            "dominating_instance": dom,
        },
    }


def _compute_semantic_hash(report: dict) -> str:
    """Compute lowering_report_semantic_hash.

    Excludes: timestamp, environment, compiler_version,
              capability_registry_hash, framework_schema_hash
    (Composition SS6.3, NORMATIVE exclusion set)
    """
    excluded = {
        "timestamp", "environment", "compiler_version",
        "capability_registry_hash", "framework_schema_hash",
    }
    filtered = {k: v for k, v in report.items() if k not in excluded}
    return compute_config_hash(filtered)


# ---------------------------------------------------------------------------
# Promotion artifact  (Composition SS7.2)
# ---------------------------------------------------------------------------

def write_promotion_artifact(
    strategy_config_hash: str,
    composition_id: str,
    composition_spec_hash: str,
    tier: str,
    result: str,
    dataset_hash: str,
    runner_hash: str,
    lowering_report_semantic_hash: str,
    base_dir: str = "research/promotions",
) -> str:
    """Write a promotion artifact.

    Returns the file path written.
    """
    artifact = {
        "strategy_config_hash": strategy_config_hash,
        "composition_id": composition_id,
        "composition_spec_hash": composition_spec_hash,
        "tier": tier,
        "result": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": dataset_hash,
        "runner_hash": runner_hash,
        "lowering_report_semantic_hash": lowering_report_semantic_hash,
    }

    # Extract hash without prefix for path
    hash_val = strategy_config_hash
    if hash_val.startswith("sha256:"):
        hash_val = hash_val[7:]

    ds_prefix = dataset_hash[:12] if dataset_hash else "unknown"
    if ds_prefix.startswith("sha256:"):
        ds_prefix = ds_prefix[7:19]

    dir_path = os.path.join(base_dir, hash_val)
    file_name = f"{tier}_{ds_prefix}.json"
    file_path = os.path.join(dir_path, file_name)

    # Check existing
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            existing = json.load(f)
        # Validate semantic idempotency
        for key in artifact:
            if key == "timestamp":
                continue
            if artifact[key] != existing.get(key):
                raise CompilationError(
                    f"Determinism violation: field '{key}' differs in "
                    f"existing promotion artifact at {file_path}")
        return file_path  # Existing artifact is valid

    os.makedirs(dir_path, exist_ok=True)
    content = canonical_json(artifact)
    with open(file_path, "w") as f:
        f.write(content)

    return file_path


# ---------------------------------------------------------------------------
# Artifact writing  (Composition SS7.3)
# ---------------------------------------------------------------------------

def write_artifacts(
    compilation_result: dict,
    base_dir: str = "research/strategies",
) -> str:
    """Write resolved artifact and lowering report.

    Returns the artifact directory path.
    """
    hash_val = compilation_result["strategy_config_hash"]
    if hash_val.startswith("sha256:"):
        hash_val = hash_val[7:]

    final_dir = os.path.join(base_dir, hash_val)

    # Immutability check
    resolved_path = os.path.join(final_dir, "resolved.json")
    if os.path.exists(resolved_path):
        with open(resolved_path, "r") as f:
            existing_content = f.read()
        new_content = canonical_json(
            compilation_result["resolved_artifact"])
        existing_hash = compute_raw_hash(existing_content)
        new_hash = compute_raw_hash(new_content)
        if existing_hash == new_hash:
            return final_dir  # Already exists, identical
        raise CompilationError(
            f"Determinism violation: resolved artifact differs at "
            f"{resolved_path}")

    # Atomic write via temp directory
    tmp_dir = os.path.join(base_dir, ".tmp", hash_val)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # Write resolved artifact
        resolved_content = canonical_json(
            compilation_result["resolved_artifact"])
        tmp_resolved = os.path.join(tmp_dir, "resolved.json")
        with open(tmp_resolved, "w") as f:
            f.write(resolved_content)
            f.flush()
            os.fsync(f.fileno())

        # Write lowering report
        report_content = canonical_json(
            compilation_result["lowering_report"])
        tmp_report = os.path.join(tmp_dir, "lowering_report.json")
        with open(tmp_report, "w") as f:
            f.write(report_content)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        os.makedirs(os.path.dirname(final_dir), exist_ok=True)
        os.rename(tmp_dir, final_dir)
    except Exception:
        # Cleanup on failure
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        raise

    return final_dir
