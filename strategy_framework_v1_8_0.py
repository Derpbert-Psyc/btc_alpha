"""
Strategy Framework Engine v1.8.0

Implements the Strategy Framework Contract v1.8.0:
  - Schema definitions (enums, types, condition/path/gate schemas)
  - Per-output warmup API (Delta 3, Framework SS6.4.1)
  - Schema strictness: recursive unknown-field rejection (Delta 6, Framework SS7.1)
  - DSL evaluator with is_present/is_absent operators (Delta 1, Framework SS4.1)
  - Cross-indicator condition references (Delta 5, Framework SS4.1)
  - MTM_DRAWDOWN_EXIT exit type (Delta 2, Framework SS3.11)
  - HANDOFF gate policy (Delta 4, Framework SS5.2)
  - Config loader with version-gated validation
  - Canonical JSON serialization and config hashing

Frozen artefacts are consumed, never modified:
  - btc_alpha_phase4b_1_7_2.py (indicator engine)
  - SYSTEM_LAWS.md, PHASE4A_INDICATOR_CONTRACT.md, PHASE4B_CONTRACT_LOCKED.md
"""

from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal, ROUND_HALF_EVEN, getcontext, localcontext
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Decimal context -- fixed at module load (Framework SS4.1.1)
# ---------------------------------------------------------------------------
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ExitType(Enum):
    SIGNAL = "SIGNAL"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_LIMIT = "TIME_LIMIT"
    GATE_EXIT = "GATE_EXIT"
    MTM_DRAWDOWN_EXIT = "MTM_DRAWDOWN_EXIT"


class GateExitPolicy(Enum):
    FORCE_FLAT = "FORCE_FLAT"
    HOLD_CURRENT = "HOLD_CURRENT"
    HANDOFF = "HANDOFF"


class Operator(Enum):
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQ = "=="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    IS_PRESENT = "is_present"
    IS_ABSENT = "is_absent"


# Operator string -> enum mapping
OPERATOR_MAP: Dict[str, Operator] = {op.value: op for op in Operator}

# ---------------------------------------------------------------------------
# Operator compatibility table (Framework SS4.1, NORMATIVE)
# ---------------------------------------------------------------------------

OPERATOR_COMPAT: Dict[Operator, str] = {
    Operator.GT: "1.0.0",
    Operator.LT: "1.0.0",
    Operator.GTE: "1.0.0",
    Operator.LTE: "1.0.0",
    Operator.EQ: "1.0.0",
    Operator.CROSSES_ABOVE: "1.0.0",
    Operator.CROSSES_BELOW: "1.0.0",
    Operator.IS_PRESENT: "1.8.0",
    Operator.IS_ABSENT: "1.8.0",
}

# Exit type compatibility
EXIT_TYPE_COMPAT: Dict[ExitType, str] = {
    ExitType.SIGNAL: "1.0.0",
    ExitType.STOP_LOSS: "1.0.0",
    ExitType.TRAILING_STOP: "1.0.0",
    ExitType.TIME_LIMIT: "1.0.0",
    ExitType.GATE_EXIT: "1.0.0",
    ExitType.MTM_DRAWDOWN_EXIT: "1.8.0",
}

# Gate exit policy compatibility
GATE_POLICY_COMPAT: Dict[GateExitPolicy, str] = {
    GateExitPolicy.FORCE_FLAT: "1.0.0",
    GateExitPolicy.HOLD_CURRENT: "1.0.0",
    GateExitPolicy.HANDOFF: "1.8.0",
}

# Feature compatibility (cross-indicator references)
FEATURE_COMPAT: Dict[str, str] = {
    "cross_indicator_ref": "1.8.0",
}


def version_gte(v1: str, v2: str) -> bool:
    """Check if semantic version v1 >= v2."""
    def parse(v: str) -> Tuple[int, ...]:
        return tuple(int(x) for x in v.split("."))
    return parse(v1) >= parse(v2)


# ---------------------------------------------------------------------------
# Per-output warmup API  (Delta 3, Framework SS6.4.1)
# ---------------------------------------------------------------------------
# This wraps the frozen Phase 4B indicator catalog with per-output warmup
# knowledge.  Phase 4B is not modified -- this is framework-level metadata.
#
# MACD (indicator 7) outputs and warmup:
#   macd_line:   slow_length bars
#   signal_line: slow_length + signal_length - 1 bars
#   histogram:   slow_length + signal_length - 1 bars
#   slope_sign:         slow_length + 1 bars  (derivative of macd_line)
#   signal_slope_sign:  slow_length + signal_length bars  (derivative of signal_line)
#
# The slope_sign and signal_slope_sign outputs are computed by the framework
# as the sign of the bar-over-bar delta of macd_line and signal_line
# respectively.  They are not native Phase 4B outputs.

# Indicator name -> integer ID mapping (composition uses string names)
INDICATOR_NAME_TO_ID: Dict[str, int] = {
    "ema": 1, "rsi": 2, "atr": 3, "pivot_structure": 4, "avwap": 5,
    "dd_equity": 6, "macd": 7, "macd_tv": 7, "roc": 8, "adx": 9,
    "choppiness": 10, "bollinger": 11, "linreg": 12, "hv": 13,
    "donchian": 14, "floor_pivots": 15, "dynamic_sr": 16,
    "vol_targeting": 17, "vrvp": 18, "rs_ratio": 19,
    "correlation": 20, "beta": 21, "dd_price": 22,
    "dd_per_trade": 23, "dd_metrics": 24,
    "lmagr": 25,
}

# Reverse mapping
INDICATOR_ID_TO_NAME: Dict[int, str] = {}
for _name, _id in INDICATOR_NAME_TO_ID.items():
    if _id not in INDICATOR_ID_TO_NAME:
        INDICATOR_ID_TO_NAME[_id] = _name

# Per-indicator output sets with semantic types
INDICATOR_OUTPUTS: Dict[int, Dict[str, str]] = {
    1: {"ema": "PRICE"},
    2: {"rsi": "RATE"},
    3: {"atr": "PRICE"},
    4: {"pivot_high": "PRICE", "pivot_low": "PRICE"},
    5: {"avwap": "PRICE"},
    6: {"equity_dd": "RATE", "equity_dd_duration": "RATE"},
    7: {"macd_line": "PRICE", "signal_line": "PRICE", "histogram": "PRICE",
        "slope_sign": "RATE",
        "signal_slope_sign": "RATE"},  # slope_sign, signal_slope_sign = framework-computed derivatives
    8: {"roc": "RATE"},
    9: {"adx": "RATE", "plus_di": "RATE", "minus_di": "RATE"},
    10: {"choppiness": "RATE"},
    11: {"basis": "PRICE", "upper": "PRICE", "lower": "PRICE",
         "bandwidth": "RATE", "percent_b": "RATE"},
    12: {"slope": "RATE"},
    13: {"hv": "RATE"},
    14: {"upper": "PRICE", "lower": "PRICE", "basis": "PRICE"},
    15: {"pivot": "PRICE", "r1": "PRICE", "s1": "PRICE",
         "r2": "PRICE", "s2": "PRICE", "r3": "PRICE", "s3": "PRICE"},
    16: {"resistance": "PRICE", "support": "PRICE"},
    17: {"target_position": "RATE"},
    18: {"poc": "PRICE", "value_area_high": "PRICE", "value_area_low": "PRICE"},
    19: {"rs_ratio": "RATE"},
    20: {"correlation": "RATE"},
    21: {"beta": "RATE"},
    22: {"price_dd": "RATE", "price_dd_duration": "RATE"},
    23: {"per_trade_dd": "RATE"},
    24: {"max_dd": "RATE", "avg_dd": "RATE", "recovery_factor": "RATE"},
}

# Diagnostic probes (IDs 25-29) -- output sets
DIAGNOSTIC_OUTPUTS: Dict[int, Dict[str, str]] = {
    25: {"lmagr": "RATE", "lmagr_pct": "RATE"},
    26: {"funding_rate": "RATE"},
    27: {"oi_change": "RATE"},
    28: {"volume_profile": "RATE"},
    29: {"liquidation_intensity": "RATE"},
}


def _macd_output_warmup(output: str, params: dict) -> int:
    """Per-output warmup for MACD (indicator 7)."""
    slow = params.get("slow", params.get("slow_length", 26))
    signal = params.get("signal", params.get("signal_length", 9))
    if output == "macd_line":
        return slow
    elif output == "signal_line":
        return slow + signal - 1
    elif output == "histogram":
        return slow + signal - 1
    elif output == "slope_sign":
        # Derivative of macd_line: needs one extra bar
        return slow + 1
    elif output == "signal_slope_sign":
        # Derivative of signal_line: signal warmup + 1 extra bar
        return slow + signal
    else:
        raise ValueError(f"Unknown MACD output: {output!r}")


def _adx_output_warmup(output: str, params: dict) -> int:
    """Per-output warmup for ADX (indicator 9)."""
    length = params.get("length", 14)
    if output in ("plus_di", "minus_di"):
        return length
    elif output == "adx":
        return 2 * length
    else:
        raise ValueError(f"Unknown ADX output: {output!r}")


def _bollinger_output_warmup(output: str, params: dict) -> int:
    """Per-output warmup for Bollinger (indicator 11)."""
    length = params.get("length", params.get("period", 20))
    # All Bollinger outputs have the same warmup
    return length


def _lmagr_output_warmup(output: str, params: dict) -> int:
    """Per-output warmup for LMAGR (indicator 25)."""
    length = params.get("ma_length", params.get("length", params.get("period", 20)))
    if output in ("lmagr", "lmagr_pct"):
        return length
    else:
        raise ValueError(f"Unknown LMAGR output: {output!r}")


def _generic_warmup(indicator_id: int, params: dict) -> int:
    """Single warmup value for indicators without per-output differentiation."""
    if indicator_id == 1:  # EMA
        return params.get("length", params.get("period", 20))
    elif indicator_id == 2:  # RSI
        return params.get("length", params.get("period", 14)) + 1
    elif indicator_id == 3:  # ATR
        return params.get("length", params.get("period", 14))
    elif indicator_id == 4:  # Pivot Structure
        left = params.get("left_bars", 5)
        right = params.get("right_bars", 5)
        return left + right + 1
    elif indicator_id == 5:  # AVWAP
        return 1  # Activation-gated, not history-gated
    elif indicator_id == 6:  # DD Equity
        return 1
    elif indicator_id == 7:  # MACD (full indicator warmup)
        slow = params.get("slow", params.get("slow_length", 26))
        signal = params.get("signal", params.get("signal_length", 9))
        return slow + signal - 1
    elif indicator_id == 8:  # ROC
        return params.get("length", params.get("period", 9))
    elif indicator_id == 9:  # ADX
        return 2 * params.get("length", params.get("period", 14))
    elif indicator_id == 10:  # Choppiness
        return params.get("length", params.get("period", 14))
    elif indicator_id == 11:  # Bollinger
        return params.get("length", params.get("period", 20))
    elif indicator_id == 12:  # LinReg
        return params.get("length", params.get("period", 14))
    elif indicator_id == 13:  # HV
        return params.get("length", params.get("period", 20)) + 1
    elif indicator_id == 14:  # Donchian
        return params.get("length", params.get("period", 20))
    elif indicator_id == 15:  # Floor Pivots
        return 1
    elif indicator_id == 16:  # Dynamic SR
        left = params.get("left_bars", 5)
        right = params.get("right_bars", 5)
        atr_len = params.get("atr_length", 14)
        return max(left + right + 1, atr_len)
    elif indicator_id == 17:  # Vol Targeting
        return params.get("length", params.get("period", 20))
    elif indicator_id == 18:  # VRVP
        return params.get("length", params.get("period", 50))
    elif indicator_id == 19:  # RS Ratio
        return params.get("length", params.get("period", 14))
    elif indicator_id == 20:  # Correlation
        return params.get("length", params.get("period", 20))
    elif indicator_id == 21:  # Beta
        return params.get("length", params.get("period", 20))
    elif indicator_id == 22:  # DD Price
        return 1
    elif indicator_id == 23:  # DD Per-Trade
        return 1
    elif indicator_id == 24:  # DD Metrics
        return 1
    elif 25 <= indicator_id <= 29:  # Diagnostic probes
        return params.get("length", params.get("period", 1))
    else:
        raise ValueError(f"Unknown indicator_id: {indicator_id}")


# Indicators with per-output warmup support
_PER_OUTPUT_INDICATORS: Dict[int, Any] = {
    7: _macd_output_warmup,
    9: _adx_output_warmup,
    11: _bollinger_output_warmup,
    25: _lmagr_output_warmup,
}


def get_warmup_bars_for_output(indicator_id: int, output_name: str,
                               params: dict) -> int:
    """Return minimum warmup bars for a specific output with given params.

    Framework SS6.4.1: per-output warmup API.
    """
    # Resolve string indicator name to ID if needed
    if isinstance(indicator_id, str):
        if indicator_id in INDICATOR_NAME_TO_ID:
            indicator_id = INDICATOR_NAME_TO_ID[indicator_id]
        else:
            raise ValueError(f"Unknown indicator name: {indicator_id!r}")

    if indicator_id in _PER_OUTPUT_INDICATORS:
        return _PER_OUTPUT_INDICATORS[indicator_id](output_name, params)

    # Fallback: single warmup value for all outputs
    return _generic_warmup(indicator_id, params)


def compute_instance_warmup(indicator_id: int, outputs_used: List[str],
                            params: dict) -> int:
    """Compute instance warmup = max warmup across all outputs_used.

    Framework SS6.4.1: instance warmup computation.
    """
    if not outputs_used:
        return _generic_warmup(indicator_id if isinstance(indicator_id, int)
                               else INDICATOR_NAME_TO_ID.get(indicator_id, 0),
                               params)
    return max(
        get_warmup_bars_for_output(indicator_id, out, params)
        for out in outputs_used
    )


# ---------------------------------------------------------------------------
# Schema strictness  (Delta 6, Framework SS7.1)
# ---------------------------------------------------------------------------
# The engine rejects unknown fields at any depth in the config JSON.

# Allowed fields at each level of the config hierarchy for v1.8.0
_CONDITION_FIELDS = {
    "indicator", "output", "operator", "value",
    "ref_indicator", "ref_output",  # v1.8.0 cross-indicator
}

_CONDITION_GROUP_FIELDS = {
    "name", "conditions",
}

_ENTRY_PATH_FIELDS = {
    "name", "direction", "evaluation_cadence", "conditions",
    "condition_groups",
}

_EXIT_PATH_FIELDS = {
    "name", "applies_to", "evaluation_cadence", "type", "conditions",
    "parameters",
}

_GATE_RULE_FIELDS = {
    "name", "conditions", "on_close_policy",
}

_INDICATOR_INSTANCE_FIELDS = {
    "label", "indicator_id", "timeframe", "parameters", "outputs_used",
    "role", "data_source", "bar_provider", "warmup_bars",
}

_POSITION_SIZING_FIELDS = {
    "mode", "fraction_of_equity", "target_vol", "vol_indicator_label",
    "max_leverage", "min_leverage", "min_vol_threshold",
}

_STOP_LOSS_FIELDS = {
    "mode", "atr_multiple", "atr_indicator_label", "percent",
    "exchange_side",
}

_TRAILING_STOP_FIELDS = {
    "distance_atr_multiple", "atr_indicator_label",
    "activation_profit_atr", "tighten_condition",
}

_TAKE_PROFIT_FIELDS = {
    "mode", "legs",
}

_FUNDING_MODEL_FIELDS = {
    "enabled", "interval_hours", "rate_per_interval", "credit_allowed",
}

_TRADE_RATE_LIMIT_FIELDS = {
    "min_time_between_trades_ms", "max_trades_per_hour",
}

_SLIPPAGE_BUDGET_FIELDS = {
    "max_slippage_bps_per_hour",
}

_WARMUP_RESTART_FIELDS = {
    "mode", "hard_stop_percent",
}

_MTM_DRAWDOWN_FIELDS = {
    "enabled", "evaluation_cadence", "drawdown_bps_long",
    "drawdown_bps_short", "applies_to",
}

_EXECUTION_PARAMS_FIELDS = {
    "position_sizing", "entry_type", "leverage", "stop_loss",
    "take_profit", "trailing_stop", "time_limit_bars",
    "time_limit_reference_cadence", "time_limit_allows_flip",
    "flip_enabled", "scale_in", "funding_model",
    "trade_rate_limit", "slippage_budget", "warmup_restart_policy",
    "mtm_drawdown_exit",
    "strict_entry_paths", "strict_exit_paths",
}

_TOP_LEVEL_FIELDS = {
    "engine_version", "indicator_instances", "entry_rules", "exit_rules",
    "gate_rules", "execution_params", "archetype_tags",
}


class SchemaError(Exception):
    """Raised when schema validation fails."""


def _check_unknown_fields(obj: dict, allowed: Set[str], path: str) -> List[str]:
    """Return list of error messages for unknown fields."""
    errors = []
    for key in obj:
        if key not in allowed:
            errors.append(
                f"Unknown field '{key}' at {path or 'root'}"
            )
    return errors


def validate_schema_strict(config: dict) -> List[str]:
    """Recursively validate that config contains no unknown fields.

    Returns list of error messages.  Empty list = valid.
    """
    errors: List[str] = []

    # Top level
    errors.extend(_check_unknown_fields(config, _TOP_LEVEL_FIELDS, ""))

    # Indicator instances
    for i, inst in enumerate(config.get("indicator_instances", [])):
        if isinstance(inst, dict):
            errors.extend(_check_unknown_fields(
                inst, _INDICATOR_INSTANCE_FIELDS,
                f"indicator_instances[{i}]"))

    # Entry rules
    for i, rule in enumerate(config.get("entry_rules", [])):
        if isinstance(rule, dict):
            errors.extend(_check_unknown_fields(
                rule, _ENTRY_PATH_FIELDS, f"entry_rules[{i}]"))
            for j, grp in enumerate(rule.get("condition_groups", [])):
                if isinstance(grp, dict):
                    errors.extend(_check_unknown_fields(
                        grp, _CONDITION_GROUP_FIELDS,
                        f"entry_rules[{i}].condition_groups[{j}]"))
                    for k, cond in enumerate(grp.get("conditions", [])):
                        if isinstance(cond, dict):
                            errors.extend(_check_unknown_fields(
                                cond, _CONDITION_FIELDS,
                                f"entry_rules[{i}].condition_groups[{j}]"
                                f".conditions[{k}]"))
            for j, cond in enumerate(rule.get("conditions", [])):
                if isinstance(cond, dict):
                    errors.extend(_check_unknown_fields(
                        cond, _CONDITION_FIELDS,
                        f"entry_rules[{i}].conditions[{j}]"))

    # Exit rules
    for i, rule in enumerate(config.get("exit_rules", [])):
        if isinstance(rule, dict):
            errors.extend(_check_unknown_fields(
                rule, _EXIT_PATH_FIELDS, f"exit_rules[{i}]"))
            for j, cond in enumerate(rule.get("conditions", [])):
                if isinstance(cond, dict):
                    errors.extend(_check_unknown_fields(
                        cond, _CONDITION_FIELDS,
                        f"exit_rules[{i}].conditions[{j}]"))

    # Gate rules
    for i, rule in enumerate(config.get("gate_rules", [])):
        if isinstance(rule, dict):
            errors.extend(_check_unknown_fields(
                rule, _GATE_RULE_FIELDS, f"gate_rules[{i}]"))
            for j, cond in enumerate(rule.get("conditions", [])):
                if isinstance(cond, dict):
                    errors.extend(_check_unknown_fields(
                        cond, _CONDITION_FIELDS,
                        f"gate_rules[{i}].conditions[{j}]"))

    # Execution params
    ep = config.get("execution_params")
    if isinstance(ep, dict):
        errors.extend(_check_unknown_fields(
            ep, _EXECUTION_PARAMS_FIELDS, "execution_params"))
        for sub_key, sub_fields in [
            ("position_sizing", _POSITION_SIZING_FIELDS),
            ("stop_loss", _STOP_LOSS_FIELDS),
            ("trailing_stop", _TRAILING_STOP_FIELDS),
            ("take_profit", _TAKE_PROFIT_FIELDS),
            ("funding_model", _FUNDING_MODEL_FIELDS),
            ("trade_rate_limit", _TRADE_RATE_LIMIT_FIELDS),
            ("slippage_budget", _SLIPPAGE_BUDGET_FIELDS),
            ("warmup_restart_policy", _WARMUP_RESTART_FIELDS),
            ("mtm_drawdown_exit", _MTM_DRAWDOWN_FIELDS),
        ]:
            sub = ep.get(sub_key)
            if isinstance(sub, dict):
                errors.extend(_check_unknown_fields(
                    sub, sub_fields, f"execution_params.{sub_key}"))

    return errors


# ---------------------------------------------------------------------------
# Config validation  (version-gated)
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised on invalid strategy config."""


def validate_config(config: dict) -> List[str]:
    """Validate a strategy config.  Returns list of error messages."""
    errors: List[str] = []
    ev = config.get("engine_version", "")
    if not ev:
        errors.append("Missing required field 'engine_version'")
        return errors

    # 1. Schema strictness (Delta 6)
    errors.extend(validate_schema_strict(config))

    # 2. Validate operators against engine version
    for i, rule in enumerate(config.get("entry_rules", [])):
        for j, cond in enumerate(_iter_conditions(rule)):
            errors.extend(_validate_condition(cond, ev,
                                              f"entry_rules[{i}]"))
    for i, rule in enumerate(config.get("exit_rules", [])):
        for j, cond in enumerate(_iter_conditions(rule)):
            errors.extend(_validate_condition(cond, ev,
                                              f"exit_rules[{i}]"))
    for i, rule in enumerate(config.get("gate_rules", [])):
        for j, cond in enumerate(rule.get("conditions", [])):
            errors.extend(_validate_condition(cond, ev,
                                              f"gate_rules[{i}]"))

    # 3. Validate exit types
    for i, rule in enumerate(config.get("exit_rules", [])):
        et_str = rule.get("type", "")
        try:
            et = ExitType(et_str)
        except ValueError:
            errors.append(f"exit_rules[{i}]: unknown exit type '{et_str}'")
            continue
        min_ver = EXIT_TYPE_COMPAT.get(et, "1.0.0")
        if not version_gte(ev, min_ver):
            errors.append(
                f"exit_rules[{i}]: exit type '{et.value}' requires "
                f"engine_version >= {min_ver}, got {ev}")

    # 4. Validate gate policies
    for i, rule in enumerate(config.get("gate_rules", [])):
        pol_str = rule.get("on_close_policy", "")
        try:
            pol = GateExitPolicy(pol_str)
        except ValueError:
            errors.append(
                f"gate_rules[{i}]: unknown on_close_policy '{pol_str}'")
            continue
        min_ver = GATE_POLICY_COMPAT.get(pol, "1.0.0")
        if not version_gte(ev, min_ver):
            errors.append(
                f"gate_rules[{i}]: gate policy '{pol.value}' requires "
                f"engine_version >= {min_ver}, got {ev}")

    # 5. Validate per-output warmup (Delta 3)
    for i, inst in enumerate(config.get("indicator_instances", [])):
        ind_id = inst.get("indicator_id")
        if isinstance(ind_id, str):
            ind_id = INDICATOR_NAME_TO_ID.get(ind_id)
        if ind_id is None:
            continue
        outputs_used = inst.get("outputs_used", [])
        params = inst.get("parameters", {})
        explicit_warmup = inst.get("warmup_bars")
        if explicit_warmup is not None and outputs_used:
            required = compute_instance_warmup(ind_id, outputs_used, params)
            if explicit_warmup < required:
                # Find which output demands more
                for out in outputs_used:
                    out_warmup = get_warmup_bars_for_output(ind_id, out, params)
                    if explicit_warmup < out_warmup:
                        label = inst.get("label", f"instance[{i}]")
                        errors.append(
                            f"Instance '{label}' warmup_bars "
                            f"({explicit_warmup}) is less than required "
                            f"({out_warmup}) for output '{out}'.")
                        break

    # 6. Validate outputs_used for cross-indicator references
    label_to_outputs: Dict[str, List[str]] = {}
    for inst in config.get("indicator_instances", []):
        label_to_outputs[inst.get("label", "")] = inst.get("outputs_used", [])

    for i, rule in enumerate(config.get("entry_rules", [])):
        for cond in _iter_conditions(rule):
            errors.extend(_validate_outputs_used(cond, label_to_outputs,
                                                  f"entry_rules[{i}]"))
    for i, rule in enumerate(config.get("exit_rules", [])):
        for cond in _iter_conditions(rule):
            errors.extend(_validate_outputs_used(cond, label_to_outputs,
                                                  f"exit_rules[{i}]"))
    for i, rule in enumerate(config.get("gate_rules", [])):
        for cond in rule.get("conditions", []):
            errors.extend(_validate_outputs_used(cond, label_to_outputs,
                                                  f"gate_rules[{i}]"))

    return errors


def _iter_conditions(rule: dict):
    """Yield all conditions from a rule (entry or exit path)."""
    for cond in rule.get("conditions", []):
        yield cond
    for grp in rule.get("condition_groups", []):
        for cond in grp.get("conditions", []):
            yield cond


def _validate_condition(cond: dict, engine_version: str,
                        context: str) -> List[str]:
    """Validate a single condition against engine version."""
    errors: List[str] = []
    op_str = cond.get("operator", "")
    op = OPERATOR_MAP.get(op_str)
    if op is None:
        errors.append(f"{context}: unknown operator '{op_str}'")
        return errors

    min_ver = OPERATOR_COMPAT.get(op, "1.0.0")
    if not version_gte(engine_version, min_ver):
        errors.append(
            f"{context}: operator '{op.value}' requires "
            f"engine_version >= {min_ver}, got {engine_version}")

    # is_present / is_absent must not have a value field (Delta 1)
    if op in (Operator.IS_PRESENT, Operator.IS_ABSENT):
        if "value" in cond and cond["value"] is not None:
            errors.append(
                f"{context}: operator '{op.value}' must not have a "
                f"'value' field")

    # Cross-indicator ref validation (Delta 5)
    has_value = "value" in cond and cond.get("value") is not None
    has_ref = "ref_indicator" in cond and cond.get("ref_indicator") is not None
    if has_value and has_ref:
        errors.append(
            f"{context}: 'value' and 'ref_indicator' are mutually exclusive")
    if has_ref:
        if not version_gte(engine_version, FEATURE_COMPAT["cross_indicator_ref"]):
            errors.append(
                f"{context}: cross-indicator references require "
                f"engine_version >= {FEATURE_COMPAT['cross_indicator_ref']}, "
                f"got {engine_version}")
        if not cond.get("ref_output"):
            errors.append(
                f"{context}: 'ref_indicator' present but 'ref_output' missing")
        # is_present/is_absent cannot use ref_indicator
        if op in (Operator.IS_PRESENT, Operator.IS_ABSENT):
            errors.append(
                f"{context}: operator '{op.value}' must not use "
                f"'ref_indicator' (unary operator)")

    return errors


def _validate_outputs_used(cond: dict, label_to_outputs: dict,
                           context: str) -> List[str]:
    """Validate that referenced outputs are declared in outputs_used."""
    errors: List[str] = []
    ind = cond.get("indicator", "")
    out = cond.get("output", "")
    if ind and out and ind in label_to_outputs:
        if out not in label_to_outputs[ind]:
            errors.append(
                f"{context}: output '{out}' not in outputs_used "
                f"for instance '{ind}'")
    ref_ind = cond.get("ref_indicator", "")
    ref_out = cond.get("ref_output", "")
    if ref_ind and ref_out and ref_ind in label_to_outputs:
        if ref_out not in label_to_outputs[ref_ind]:
            errors.append(
                f"{context}: ref_output '{ref_out}' not in outputs_used "
                f"for instance '{ref_ind}'")
    return errors


def load_strategy_config(config: dict) -> dict:
    """Load and validate a strategy config.

    Raises ConfigError on invalid config.
    Returns the validated config dict.
    """
    errors = validate_config(config)
    if errors:
        raise ConfigError(
            f"Strategy config validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors))
    return config


# ---------------------------------------------------------------------------
# DSL evaluator  (Delta 1 + Delta 5)
# ---------------------------------------------------------------------------

def evaluate_condition(
    cond: dict,
    indicator_outputs: Dict[str, Dict[str, Any]],
    prev_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    """Evaluate a single DSL condition.

    indicator_outputs: {instance_label: {output_name: value_or_None}}
    prev_outputs: same shape, previous bar values (for crosses_*)

    Returns True if condition is satisfied, False otherwise.
    Framework SS4.1.1: None -> condition false (no exception).
    """
    op = OPERATOR_MAP.get(cond.get("operator", ""), None)
    if op is None:
        return False

    ind_label = cond.get("indicator", "")
    output_name = cond.get("output", "")

    # Get current value
    current = _get_output_value(indicator_outputs, ind_label, output_name)

    # is_present / is_absent (Delta 1)
    if op == Operator.IS_PRESENT:
        return current is not None
    if op == Operator.IS_ABSENT:
        return current is None

    # For all other operators, None -> false
    if current is None:
        return False

    # Resolve comparison value
    ref_ind = cond.get("ref_indicator")
    ref_out = cond.get("ref_output")

    if ref_ind and ref_out:
        # Cross-indicator reference (Delta 5)
        comp_value = _get_output_value(indicator_outputs, ref_ind, ref_out)
        if comp_value is None:
            return False
    else:
        comp_value = cond.get("value")
        if comp_value is None:
            return False
        # Convert to comparable type
        comp_value = _to_number(comp_value)

    current = _to_number(current)
    if current is None or comp_value is None:
        return False

    # Simple comparison operators
    if op == Operator.GT:
        return current > comp_value
    elif op == Operator.LT:
        return current < comp_value
    elif op == Operator.GTE:
        return current >= comp_value
    elif op == Operator.LTE:
        return current <= comp_value
    elif op == Operator.EQ:
        return current == comp_value

    # Crossing operators -- need previous values
    if prev_outputs is None:
        return False

    prev_current = _get_output_value(prev_outputs, ind_label, output_name)
    if prev_current is None:
        return False
    prev_current = _to_number(prev_current)
    if prev_current is None:
        return False

    if ref_ind and ref_out:
        prev_comp = _get_output_value(prev_outputs, ref_ind, ref_out)
        if prev_comp is None:
            return False
        prev_comp = _to_number(prev_comp)
    else:
        prev_comp = comp_value

    if prev_comp is None:
        return False

    if op == Operator.CROSSES_ABOVE:
        return prev_current <= prev_comp and current > comp_value
    elif op == Operator.CROSSES_BELOW:
        return prev_current >= prev_comp and current < comp_value

    return False


def _get_output_value(outputs: dict, label: str, output: str) -> Any:
    """Get an output value from the indicator outputs dict."""
    inst = outputs.get(label)
    if inst is None:
        return None
    return inst.get(output)


def _to_number(val: Any) -> Any:
    """Convert a value to a numeric type for comparison."""
    if val is None:
        return None
    if isinstance(val, (int, float, Decimal)):
        return val
    if isinstance(val, str):
        try:
            # Try int first
            if "." not in val:
                return int(val)
            return Decimal(val)
        except (ValueError, Exception):
            return None
    return val


def evaluate_all_conditions(
    conditions: List[dict],
    indicator_outputs: Dict[str, Dict[str, Any]],
    prev_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    """Evaluate a list of conditions with implicit AND.

    All conditions must be True for the result to be True.
    """
    for cond in conditions:
        if not evaluate_condition(cond, indicator_outputs, prev_outputs):
            return False
    return True


def evaluate_condition_groups(
    groups: List[dict],
    indicator_outputs: Dict[str, Dict[str, Any]],
    prev_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    """Evaluate condition groups.  All groups must pass (AND).

    Framework SS2.2.1: Groups evaluated first; if any group fails,
    standalone conditions not evaluated (short-circuit).
    """
    for grp in groups:
        conds = grp.get("conditions", [])
        if not evaluate_all_conditions(conds, indicator_outputs, prev_outputs):
            return False
    return True


def evaluate_entry_path(
    path: dict,
    indicator_outputs: Dict[str, Dict[str, Any]],
    prev_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    """Evaluate a single entry path.

    Groups first, then standalone conditions. All must pass.
    """
    groups = path.get("condition_groups", [])
    if groups:
        if not evaluate_condition_groups(groups, indicator_outputs, prev_outputs):
            return False
    conditions = path.get("conditions", [])
    if conditions:
        if not evaluate_all_conditions(conditions, indicator_outputs, prev_outputs):
            return False
    return True


def evaluate_exit_path(
    path: dict,
    indicator_outputs: Dict[str, Dict[str, Any]],
    prev_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    """Evaluate conditions for a SIGNAL-type exit path."""
    conditions = path.get("conditions", [])
    return evaluate_all_conditions(conditions, indicator_outputs, prev_outputs)


# ---------------------------------------------------------------------------
# MTM Drawdown tracker  (Delta 2, Framework SS3.11)
# ---------------------------------------------------------------------------

class MTMDrawdownTracker:
    """Tracks mark-to-market drawdown from peak for a single position.

    All arithmetic uses integer basis points (bps).
    P&L is gross mark-to-market (excludes fees, funding, slippage).
    """

    def __init__(self) -> None:
        self.peak_pnl_bps: int = 0
        self.current_pnl_bps: int = 0
        self._active: bool = False

    def open_position(self) -> None:
        """Reset peak tracking for a new position."""
        self.peak_pnl_bps = 0
        self.current_pnl_bps = 0
        self._active = True

    def close_position(self) -> None:
        """Discard peak state."""
        self.peak_pnl_bps = 0
        self.current_pnl_bps = 0
        self._active = False

    def update(self, entry_price: int, current_price: int,
               direction: str) -> int:
        """Update P&L and peak, return current drawdown in bps.

        entry_price and current_price are integer-scaled values.
        direction: "LONG" or "SHORT"

        Framework SS3.11 computation:
          For longs:  pnl_bps = (current - entry) * 10000 / entry
          For shorts: pnl_bps = (entry - current) * 10000 / entry
          peak = max(pnl_bps) since entry
          drawdown = peak - current_pnl_bps
        """
        if not self._active or entry_price == 0:
            return 0

        if direction == "LONG":
            self.current_pnl_bps = (
                (current_price - entry_price) * 10000 // entry_price
            )
        else:  # SHORT
            self.current_pnl_bps = (
                (entry_price - current_price) * 10000 // entry_price
            )

        if self.current_pnl_bps > self.peak_pnl_bps:
            self.peak_pnl_bps = self.current_pnl_bps

        return self.peak_pnl_bps - self.current_pnl_bps

    def check_trigger(self, threshold_bps: int) -> bool:
        """Check if drawdown >= threshold (deterministic >= comparison)."""
        if not self._active:
            return False
        drawdown = self.peak_pnl_bps - self.current_pnl_bps
        return drawdown >= threshold_bps

    @property
    def active(self) -> bool:
        return self._active

    def snapshot(self) -> dict:
        """Return state for determinism verification."""
        return {
            "peak_pnl_bps": self.peak_pnl_bps,
            "current_pnl_bps": self.current_pnl_bps,
            "active": self._active,
        }


# ---------------------------------------------------------------------------
# Gate evaluator  (Delta 4, Framework SS5.2)
# ---------------------------------------------------------------------------

def evaluate_gates(
    gate_rules: List[dict],
    indicator_outputs: Dict[str, Dict[str, Any]],
    prev_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[bool, Optional[GateExitPolicy], List[dict]]:
    """Evaluate all gate rules.

    Returns (all_gates_open, effective_policy, per_gate_status).

    Framework SS5.2.1: conflict resolution order:
      FORCE_FLAT > HANDOFF > HOLD_CURRENT
    """
    all_open = True
    closed_policies: List[GateExitPolicy] = []
    per_gate: List[dict] = []

    for rule in gate_rules:
        conditions = rule.get("conditions", [])
        is_open = evaluate_all_conditions(conditions, indicator_outputs,
                                          prev_outputs)
        policy_str = rule.get("on_close_policy", "HOLD_CURRENT")
        try:
            policy = GateExitPolicy(policy_str)
        except ValueError:
            policy = GateExitPolicy.HOLD_CURRENT

        per_gate.append({
            "name": rule.get("name", ""),
            "is_open": is_open,
            "on_close_policy": policy.value,
        })

        if not is_open:
            all_open = False
            closed_policies.append(policy)

    effective_policy: Optional[GateExitPolicy] = None
    if not all_open and closed_policies:
        # Most restrictive wins: FORCE_FLAT > HANDOFF > HOLD_CURRENT
        priority = {
            GateExitPolicy.FORCE_FLAT: 3,
            GateExitPolicy.HANDOFF: 2,
            GateExitPolicy.HOLD_CURRENT: 1,
        }
        effective_policy = max(closed_policies, key=lambda p: priority[p])

    return all_open, effective_policy, per_gate


def should_suppress_signal_exit(effective_policy: Optional[GateExitPolicy]) -> bool:
    """Check if SIGNAL exits should be suppressed (HANDOFF policy).

    Framework SS5.2: HANDOFF suppresses SIGNAL exits at step 6.
    Non-SIGNAL exits evaluate normally.
    """
    return effective_policy == GateExitPolicy.HANDOFF


def should_force_flat(effective_policy: Optional[GateExitPolicy]) -> bool:
    """Check if position should be force-flattened."""
    return effective_policy == GateExitPolicy.FORCE_FLAT


# ---------------------------------------------------------------------------
# Signal evaluation pipeline (Framework SS5.1)
# ---------------------------------------------------------------------------

class SignalResult:
    """Result of evaluating the full signal pipeline for one cycle."""

    def __init__(self) -> None:
        self.action: Optional[str] = None  # "ENTRY", "EXIT", "FLIP", "HOLD"
        self.direction: Optional[str] = None  # "LONG", "SHORT"
        self.exit_reason: Optional[str] = None
        self.exit_path_name: Optional[str] = None
        self.entry_path_name: Optional[str] = None
        self.gate_status: List[dict] = []
        self.diagnostics: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return (f"SignalResult(action={self.action}, dir={self.direction}, "
                f"exit_reason={self.exit_reason})")


def evaluate_signal_pipeline(
    config: dict,
    indicator_outputs: Dict[str, Dict[str, Any]],
    prev_outputs: Optional[Dict[str, Dict[str, Any]]],
    position: Optional[dict],  # {"direction": "LONG"/"SHORT", "entry_price": int}
    mtm_tracker: Optional[MTMDrawdownTracker] = None,
    current_price: Optional[int] = None,
    risk_override: Optional[str] = None,  # "EXIT_ALL" etc.
) -> SignalResult:
    """Evaluate the full signal pipeline for one cycle.

    Framework SS5.1 steps 0-7 (simplified for testing).
    """
    result = SignalResult()

    # Step 2: Risk overrides
    if risk_override == "EXIT_ALL" and position:
        result.action = "EXIT"
        result.exit_reason = "RISK_OVERRIDE"
        return result

    # Step 5: Gate evaluation
    gate_rules = config.get("gate_rules", [])
    all_open, effective_policy, gate_status = evaluate_gates(
        gate_rules, indicator_outputs, prev_outputs)
    result.gate_status = gate_status

    # Gate FORCE_FLAT
    if should_force_flat(effective_policy) and position:
        result.action = "EXIT"
        result.exit_reason = "GATE_FORCE_FLAT"
        return result

    # Step 6: Signal evaluation
    exit_rules = config.get("exit_rules", [])
    entry_rules = config.get("entry_rules", [])
    flip_enabled = config.get("execution_params", {}).get("flip_enabled", False)
    handoff_active = should_suppress_signal_exit(effective_policy)

    # Evaluate exits first
    exit_fired = False
    exit_path_name = None
    exit_type_str = None

    for exit_rule in exit_rules:
        et = exit_rule.get("type", "SIGNAL")

        # HANDOFF: suppress SIGNAL exits only (Delta 4)
        if handoff_active and et == "SIGNAL":
            continue

        if et == "SIGNAL":
            if position and evaluate_exit_path(exit_rule, indicator_outputs,
                                               prev_outputs):
                exit_fired = True
                exit_path_name = exit_rule.get("name", "")
                exit_type_str = "SIGNAL"
                break

        elif et == "MTM_DRAWDOWN_EXIT":
            if position and mtm_tracker and current_price is not None:
                entry_price = position.get("entry_price", 0)
                direction = position.get("direction", "LONG")
                mtm_tracker.update(entry_price, current_price, direction)
                threshold = (exit_rule.get("parameters", {})
                             .get(f"drawdown_bps_{direction.lower()}",
                                  exit_rule.get("parameters", {})
                                  .get("drawdown_bps", 0)))
                if mtm_tracker.check_trigger(threshold):
                    exit_fired = True
                    exit_path_name = exit_rule.get("name", "")
                    exit_type_str = "MTM_DRAWDOWN_EXIT"
                    break

        elif et == "STOP_LOSS":
            # Stop loss check (simplified)
            if position and exit_rule.get("conditions"):
                if evaluate_exit_path(exit_rule, indicator_outputs,
                                      prev_outputs):
                    exit_fired = True
                    exit_path_name = exit_rule.get("name", "")
                    exit_type_str = "STOP_LOSS"
                    break

    # Flip detection (Framework SS5.1.2)
    if exit_fired and flip_enabled and all_open and position:
        pos_dir = position.get("direction", "LONG")
        opp_dir = "SHORT" if pos_dir == "LONG" else "LONG"
        for entry_rule in entry_rules:
            if entry_rule.get("direction") == opp_dir:
                if evaluate_entry_path(entry_rule, indicator_outputs,
                                       prev_outputs):
                    result.action = "FLIP"
                    result.direction = opp_dir
                    result.exit_reason = exit_type_str
                    result.exit_path_name = exit_path_name
                    result.entry_path_name = entry_rule.get("name", "")
                    return result

    if exit_fired:
        result.action = "EXIT"
        result.exit_reason = exit_type_str
        result.exit_path_name = exit_path_name
        return result

    # Entry evaluation (only if no position and gates open)
    if not position and all_open:
        for entry_rule in entry_rules:
            if evaluate_entry_path(entry_rule, indicator_outputs, prev_outputs):
                result.action = "ENTRY"
                result.direction = entry_rule.get("direction", "LONG")
                result.entry_path_name = entry_rule.get("name", "")
                return result

    result.action = "HOLD"
    return result


# ---------------------------------------------------------------------------
# Warmup gating
# ---------------------------------------------------------------------------

def is_output_warmed_up(indicator_id: int, output_name: str,
                        params: dict, bar_index: int) -> bool:
    """Check if a specific output has completed warmup at bar_index.

    bar_index is 0-based.  Warmup requires bar_index+1 >= warmup_bars.
    """
    required = get_warmup_bars_for_output(indicator_id, output_name, params)
    return (bar_index + 1) >= required


# ---------------------------------------------------------------------------
# Canonical JSON serialisation  (Framework SS7.1)
# ---------------------------------------------------------------------------

def _canonical_value(val: Any) -> Any:
    """Convert a value for canonical JSON serialisation.

    Framework SS7.1:
    - Decimal values: JSON strings, normalised, no trailing zeros, no sci notation
    - Integer values: JSON numbers
    - None: JSON null
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        # Convert to Decimal for normalisation
        d = Decimal(str(val))
        return _canonical_decimal_str(d)
    if isinstance(val, Decimal):
        return _canonical_decimal_str(val)
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return [_canonical_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _canonical_value(v) for k, v in sorted(val.items())}
    return val


def _canonical_decimal_str(d: Decimal) -> str:
    """Normalise Decimal to canonical string form.

    No trailing zeros, no scientific notation.
    """
    # Normalise to remove trailing zeros
    norm = d.normalize()
    # Avoid scientific notation for very small/large numbers
    sign, digits, exponent = norm.as_tuple()
    if exponent > 0:
        # e.g. 5E+2 -> "500"
        s = str(int(norm))
    else:
        s = format(norm, 'f')
    # Remove trailing zeros after decimal point
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    # Handle -0
    if s == "-0":
        s = "0"
    return s


def canonical_json(obj: Any) -> str:
    """Produce canonical JSON string (sorted keys, no whitespace, UTF-8).

    Framework SS7.1: config hash = SHA-256 of this output.
    """
    converted = _canonical_value(obj)
    return json.dumps(converted, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False)


def compute_config_hash(config: dict) -> str:
    """Compute SHA-256 hex digest of canonical JSON config.

    Returns lowercase hex string prefixed with "sha256:".
    """
    cj = canonical_json(config)
    digest = hashlib.sha256(cj.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def compute_raw_hash(data: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
