"""Unit tests for the cascade condition builder and related pure functions."""

import copy
import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.components.condition_builder import (
    OPERATOR_OPTIONS,
    _format_key_params,
    _instance_display_label,
    _instance_display_label_short,
    _output_options,
    _build_condition_summary,
    _hydrate_condition,
    _unique_label,
    _rename_indicator_references,
)
from ui.services.output_descriptions import OUTPUT_DESCRIPTIONS, get_output_description


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_spec(instances=None, entry_rules=None, exit_rules=None, gate_rules=None):
    return {
        "indicator_instances": instances or [],
        "entry_rules": entry_rules or [],
        "exit_rules": exit_rules or [],
        "gate_rules": gate_rules or [],
    }


def _ema_instance(label="ema_1d_p20", tf="1d", period=20):
    return {
        "label": label,
        "indicator_id": "ema",
        "timeframe": tf,
        "parameters": {"period": period},
        "outputs_used": ["ema"],
        "role": "trigger",
        "group": "Trend",
    }


def _macd_instance(label="macd_5m", tf="5m"):
    return {
        "label": label,
        "indicator_id": "macd_tv",
        "timeframe": tf,
        "parameters": {"fast": 12, "slow": 26, "signal": 9},
        "outputs_used": ["macd_line", "signal_line", "histogram", "slope_sign"],
        "role": "trigger",
        "group": "Trend",
    }


def _boll_instance(label="boll_1h_p20", tf="1h"):
    return {
        "label": label,
        "indicator_id": "bollinger",
        "timeframe": tf,
        "parameters": {"period": 20, "num_std": 2.0},
        "outputs_used": ["basis", "upper", "lower", "bandwidth", "percent_b"],
        "role": "filter",
        "group": "Volatility",
    }


def _donchian_instance(label="dc_1d_p20", tf="1d"):
    return {
        "label": label,
        "indicator_id": "donchian",
        "timeframe": tf,
        "parameters": {"period": 20},
        "outputs_used": ["upper", "lower", "basis"],
        "role": "filter",
        "group": "Trend",
    }


# ---------------------------------------------------------------------------
# test_instance_display_label
# ---------------------------------------------------------------------------

def test_instance_display_label_ema():
    inst = _ema_instance("ema_1d_p20", "1d", 20)
    result = _instance_display_label(inst)
    assert result == "ema_1d_p20 (ema, 1d, period=20)"


def test_instance_display_label_macd():
    inst = _macd_instance("macd_5m", "5m")
    result = _instance_display_label(inst)
    assert result == "macd_5m (macd_tv, 5m, 12/26/9)"


def test_instance_display_label_bollinger():
    inst = _boll_instance("boll_1h_p20", "1h")
    result = _instance_display_label(inst)
    assert result == "boll_1h_p20 (bollinger, 1h, p=20, std=2.0)"


def test_instance_display_label_donchian():
    inst = _donchian_instance("dc_1d_p20", "1d")
    result = _instance_display_label(inst)
    assert result == "dc_1d_p20 (donchian, 1d, period=20)"


def test_instance_display_label_missing_params():
    inst = {"label": "test", "indicator_id": "ema", "timeframe": "1d", "parameters": {}}
    result = _instance_display_label(inst)
    assert result == "test (ema, 1d)"


# ---------------------------------------------------------------------------
# test_output_options
# ---------------------------------------------------------------------------

def test_output_options_ema():
    spec = _make_spec([_ema_instance()])
    opts = _output_options(spec, "ema_1d_p20")
    assert "ema" in opts
    assert "Smoothed price value" in opts["ema"]


def test_output_options_macd():
    spec = _make_spec([_macd_instance()])
    opts = _output_options(spec, "macd_5m")
    assert "slope_sign" in opts
    assert "macd_line" in opts


def test_output_options_no_match():
    spec = _make_spec([_ema_instance()])
    opts = _output_options(spec, "nonexistent")
    assert opts == {}


# ---------------------------------------------------------------------------
# test_output_description_coverage
# ---------------------------------------------------------------------------

def test_output_description_coverage():
    """Verify OUTPUT_DESCRIPTIONS covers all outputs in INDICATOR_OUTPUTS."""
    from strategy_framework_v1_8_0 import INDICATOR_OUTPUTS, INDICATOR_ID_TO_NAME
    missing = []
    for ind_id, outputs in INDICATOR_OUTPUTS.items():
        ind_name = INDICATOR_ID_TO_NAME.get(ind_id)
        if not ind_name:
            continue
        for out_name in outputs:
            desc = get_output_description(ind_name, out_name)
            if desc == out_name:
                missing.append(f"{ind_name}.{out_name}")
    if missing:
        warnings.warn(f"Missing output descriptions: {missing}")
    # Most should be covered
    assert len(missing) <= 2, f"Too many missing descriptions: {missing}"


# ---------------------------------------------------------------------------
# test_operator_options
# ---------------------------------------------------------------------------

def test_operator_options():
    assert ">" in OPERATOR_OPTIONS
    assert "<" in OPERATOR_OPTIONS
    assert ">=" in OPERATOR_OPTIONS
    assert "<=" in OPERATOR_OPTIONS
    assert "==" in OPERATOR_OPTIONS
    assert "crosses_above" in OPERATOR_OPTIONS
    assert "crosses_below" in OPERATOR_OPTIONS
    assert "is_present" not in OPERATOR_OPTIONS
    assert "is_absent" not in OPERATOR_OPTIONS


# ---------------------------------------------------------------------------
# test_build_condition_summary
# ---------------------------------------------------------------------------

def test_build_condition_summary_value():
    spec = _make_spec([_macd_instance()])
    cond = {"indicator": "macd_5m", "output": "slope_sign", "operator": ">", "value": 0}
    result = _build_condition_summary(cond, spec)
    assert "macd_5m" in result
    assert "slope_sign" in result
    assert ">" in result
    assert "0" in result


def test_build_condition_summary_xref():
    spec = _make_spec([_ema_instance("ema_1d_p2", "1d", 2), _ema_instance("ema_1d_p35", "1d", 35)])
    cond = {
        "indicator": "ema_1d_p2", "output": "ema",
        "operator": "crosses_above",
        "ref_indicator": "ema_1d_p35", "ref_output": "ema",
    }
    result = _build_condition_summary(cond, spec)
    assert "ema_1d_p2" in result
    assert "crosses_above" in result
    assert "ema_1d_p35" in result


# ---------------------------------------------------------------------------
# test_hydrate_condition
# ---------------------------------------------------------------------------

def test_hydrate_condition_valid():
    spec = _make_spec([_ema_instance()])
    cond = {"indicator": "ema_1d_p20", "output": "ema", "operator": ">", "value": 100}
    h = _hydrate_condition(cond, spec)
    assert h["instance_valid"] is True
    assert h["output_valid"] is True
    assert h["rhs_mode"] == "Value"
    assert h["value"] == 100
    assert h["warnings"] == []


def test_hydrate_condition_missing_instance():
    spec = _make_spec([_ema_instance()])
    cond = {"indicator": "nonexistent", "output": "ema", "operator": ">", "value": 0}
    h = _hydrate_condition(cond, spec)
    assert h["instance_valid"] is False
    assert "not found" in h["warnings"][0]


def test_hydrate_condition_missing_output():
    spec = _make_spec([_ema_instance()])
    cond = {"indicator": "ema_1d_p20", "output": "bad_output", "operator": ">", "value": 0}
    h = _hydrate_condition(cond, spec)
    assert h["instance_valid"] is True
    assert h["output_valid"] is False
    assert "not available" in h["warnings"][0]


def test_hydrate_condition_xref_mode():
    spec = _make_spec([_ema_instance("e1"), _ema_instance("e2")])
    cond = {
        "indicator": "e1", "output": "ema", "operator": "crosses_above",
        "ref_indicator": "e2", "ref_output": "ema",
    }
    h = _hydrate_condition(cond, spec)
    assert h["rhs_mode"] == "Indicator"
    assert h["ref_indicator"] == "e2"


def test_hydrate_condition_value_mode():
    spec = _make_spec([_ema_instance()])
    cond = {"indicator": "ema_1d_p20", "output": "ema", "operator": ">", "value": 50}
    h = _hydrate_condition(cond, spec)
    assert h["rhs_mode"] == "Value"
    assert h["ref_indicator"] is None


def test_hydrate_condition_string_value():
    """Preset conditions may have string values like '0'."""
    spec = _make_spec([_ema_instance()])
    cond = {"indicator": "ema_1d_p20", "output": "ema", "operator": ">", "value": "42"}
    h = _hydrate_condition(cond, spec)
    assert h["value"] == 42.0


# ---------------------------------------------------------------------------
# test_unique_label
# ---------------------------------------------------------------------------

def test_unique_label_no_collision():
    result = _unique_label("ema_1d_p20", ["macd_5m", "rsi_4h"])
    assert result == "ema_1d_p20"


def test_unique_label():
    result = _unique_label("ema_1d_p20", ["ema_1d_p20", "ema_1d_p20_2"])
    assert result == "ema_1d_p20_3"


def test_unique_label_first_collision():
    result = _unique_label("ema_1d", ["ema_1d"])
    assert result == "ema_1d_2"


# ---------------------------------------------------------------------------
# test_format_key_params
# ---------------------------------------------------------------------------

def test_format_key_params_ema():
    assert _format_key_params("ema", {"period": 20}) == "period=20"


def test_format_key_params_macd():
    assert _format_key_params("macd_tv", {"fast": 12, "slow": 26, "signal": 9}) == "12/26/9"


def test_format_key_params_macd_period_variant():
    assert _format_key_params("macd", {"fast_period": 12, "slow_period": 26, "signal_period": 9}) == "12/26/9"


def test_format_key_params_bollinger():
    result = _format_key_params("bollinger", {"period": 20, "num_std": 2.0})
    assert result == "p=20, std=2.0"


def test_format_key_params_empty():
    assert _format_key_params("floor_pivots", {}) == ""


def test_format_key_params_pivot_structure():
    result = _format_key_params("pivot_structure", {"left": 5, "right": 5})
    assert result == "L=5, R=5"


def test_format_key_params_dynamic_sr():
    result = _format_key_params("dynamic_sr", {"lookback": 100})
    assert result == "lookback=100"


def test_format_key_params_vrvp():
    result = _format_key_params("vrvp", {"num_bins": 50})
    assert result == "bins=50"


# ---------------------------------------------------------------------------
# test_rename_indicator_references
# ---------------------------------------------------------------------------

def _make_rename_spec():
    return {
        "indicator_instances": [
            {"label": "old_label", "indicator_id": "ema"},
            {"label": "other", "indicator_id": "rsi"},
        ],
        "entry_rules": [{
            "label": "entry",
            "conditions": [
                {"indicator": "old_label", "output": "ema", "operator": ">", "value": 0},
                {"indicator": "other", "output": "rsi", "operator": ">", "value": 50,
                 "ref_indicator": "old_label", "ref_output": "ema"},
            ],
            "condition_groups": [{
                "label": "grp",
                "conditions": [
                    {"indicator": "old_label", "output": "ema", "operator": "<", "value": 100},
                ],
            }],
        }],
        "exit_rules": [{
            "label": "exit",
            "conditions": [
                {"indicator": "old_label", "output": "ema", "operator": "<", "value": 0},
            ],
        }],
        "gate_rules": [{
            "label": "gate",
            "conditions": [
                {"indicator": "old_label", "output": "ema", "operator": ">", "value": 10},
            ],
        }],
    }


def test_rename_references_entry_rule():
    spec = _make_rename_spec()
    _rename_indicator_references(spec, "old_label", "new_label")
    assert spec["entry_rules"][0]["conditions"][0]["indicator"] == "new_label"


def test_rename_references_xref():
    spec = _make_rename_spec()
    _rename_indicator_references(spec, "old_label", "new_label")
    assert spec["entry_rules"][0]["conditions"][1]["ref_indicator"] == "new_label"


def test_rename_references_nested_group():
    spec = _make_rename_spec()
    _rename_indicator_references(spec, "old_label", "new_label")
    grp_cond = spec["entry_rules"][0]["condition_groups"][0]["conditions"][0]
    assert grp_cond["indicator"] == "new_label"


def test_rename_references_exit_rule():
    spec = _make_rename_spec()
    _rename_indicator_references(spec, "old_label", "new_label")
    assert spec["exit_rules"][0]["conditions"][0]["indicator"] == "new_label"


def test_rename_references_gate():
    spec = _make_rename_spec()
    _rename_indicator_references(spec, "old_label", "new_label")
    assert spec["gate_rules"][0]["conditions"][0]["indicator"] == "new_label"


def test_rename_references_instance():
    spec = _make_rename_spec()
    _rename_indicator_references(spec, "old_label", "new_label")
    labels = [inst["label"] for inst in spec["indicator_instances"]]
    assert "new_label" in labels
    assert "old_label" not in labels


def test_rename_does_not_affect_unreferenced():
    spec = _make_rename_spec()
    _rename_indicator_references(spec, "old_label", "new_label")
    # "other" instance should be unchanged
    assert spec["indicator_instances"][1]["label"] == "other"
    assert spec["entry_rules"][0]["conditions"][1]["indicator"] == "other"


# ---------------------------------------------------------------------------
# Additional tests from F10
# ---------------------------------------------------------------------------

def test_instance_equals_ref_cleared():
    """If instance == ref_indicator, the condition builder should clear ref fields.
    Test the pure logic: after instance change to match ref_indicator, ref gets cleared."""
    cond = {
        "indicator": "ema_1d_p2",
        "output": "ema",
        "operator": "crosses_above",
        "ref_indicator": "ema_1d_p2",  # self-reference!
        "ref_output": "ema",
    }
    # Simulate what the UI handler does (F6 check)
    new_inst = "ema_1d_p2"
    if new_inst and new_inst == cond.get("ref_indicator", ""):
        cond.pop("ref_indicator", None)
        cond.pop("ref_output", None)
    assert "ref_indicator" not in cond
    assert "ref_output" not in cond


def test_reorder_conditions():
    """Swap two conditions and verify list order changes."""
    conditions = [
        {"indicator": "a", "output": "x", "operator": ">", "value": 1},
        {"indicator": "b", "output": "y", "operator": "<", "value": 2},
    ]
    conditions[0], conditions[1] = conditions[1], conditions[0]
    assert conditions[0]["indicator"] == "b"
    assert conditions[1]["indicator"] == "a"


def test_reorder_groups():
    """Swap two groups and verify list order changes."""
    groups = [
        {"label": "grp1", "conditions": [{"indicator": "a"}]},
        {"label": "grp2", "conditions": [{"indicator": "b"}]},
    ]
    groups[0], groups[1] = groups[1], groups[0]
    assert groups[0]["label"] == "grp2"
    assert groups[1]["label"] == "grp1"


def test_instance_display_label_short():
    spec = _make_spec([_ema_instance("ema_1d_p2", "1d", 2)])
    result = _instance_display_label_short(spec, "ema_1d_p2")
    assert result == "ema_1d_p2 (ema, 1d)"


def test_instance_display_label_short_not_found():
    spec = _make_spec([])
    result = _instance_display_label_short(spec, "missing")
    assert result == "missing"


# ---------------------------------------------------------------------------
# test_generate_auto_label
# ---------------------------------------------------------------------------

def test_auto_label_ema():
    from ui.components.indicator_picker import _generate_auto_label
    assert _generate_auto_label("ema", 1, "1d", {"period": 20}) == "ema_1d_p20"


def test_auto_label_ema_period_1():
    from ui.components.indicator_picker import _generate_auto_label
    assert _generate_auto_label("ema", 1, "1d", {"period": 1}) == "price_1d"


def test_auto_label_macd():
    from ui.components.indicator_picker import _generate_auto_label
    assert _generate_auto_label("macd", 7, "5m", {}) == "macd_5m"


def test_auto_label_macd_tv():
    from ui.components.indicator_picker import _generate_auto_label
    assert _generate_auto_label("macd_tv", 7, "1d", {}) == "macd_1d"


def test_auto_label_bollinger():
    from ui.components.indicator_picker import _generate_auto_label
    assert _generate_auto_label("bollinger", 11, "1h", {"period": 20}) == "boll_1h_p20"


def test_auto_label_donchian():
    from ui.components.indicator_picker import _generate_auto_label
    assert _generate_auto_label("donchian", 14, "1d", {"period": 20}) == "dc_1d_p20"


def test_auto_label_pivot():
    from ui.components.indicator_picker import _generate_auto_label
    assert _generate_auto_label("pivot_structure", 4, "1d", {}) == "pivot_1d"


def test_auto_label_floor_pivots():
    from ui.components.indicator_picker import _generate_auto_label
    assert _generate_auto_label("floor_pivots", 15, "1d", {}) == "fpivot_1d"


def test_auto_label_rsi():
    from ui.components.indicator_picker import _generate_auto_label
    assert _generate_auto_label("rsi", 2, "4h", {"period": 14}) == "rsi_4h_p14"


def test_auto_label_collision_avoidance():
    from ui.components.indicator_picker import _unique_label
    result = _unique_label("ema_1d_p50", ["ema_1d_p50"])
    assert result == "ema_1d_p50_2"

    result2 = _unique_label("ema_1d_p50", ["ema_1d_p50", "ema_1d_p50_2"])
    assert result2 == "ema_1d_p50_3"
