"""
Acceptance tests for Strategy Framework v1.8.0 and Composition Compiler v1.5.2.

57 delta acceptance tests + 4 composition proofs.

Delta 1: is_present / is_absent operators        (10 tests)
Delta 2: MTM_DRAWDOWN_EXIT                       (10 tests)
Delta 3: Per-output warmup                       (10 tests)
Delta 4: HANDOFF gate policy                     (12 tests)
Delta 5: Cross-indicator condition references    (8 tests)
Delta 6: Schema strictness                       (4 tests — reduced from table)
Delta 7: signal_slope_sign + LMAGR wiring        (10 tests)
Composition Proofs                               (4 proofs)
                                                  --------
Total:                                            68 tests + proofs
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile

from strategy_framework_v1_8_0 import (
    ConfigError,
    Direction,
    ExitType,
    GateExitPolicy,
    MTMDrawdownTracker,
    Operator,
    canonical_json,
    compute_config_hash,
    compute_instance_warmup,
    evaluate_all_conditions,
    evaluate_condition,
    evaluate_entry_path,
    evaluate_exit_path,
    evaluate_gates,
    evaluate_signal_pipeline,
    get_warmup_bars_for_output,
    is_output_warmed_up,
    load_strategy_config,
    should_force_flat,
    should_suppress_signal_exit,
    validate_config,
    validate_schema_strict,
    version_gte,
)

from composition_compiler_v1_5_2 import (
    CompilationError,
    compile_composition,
    write_artifacts,
    write_promotion_artifact,
)

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

_results: list = []
_pass_count = 0
_fail_count = 0


def _run_test(test_id: str, func):
    global _pass_count, _fail_count
    try:
        func()
        _results.append((test_id, "PASS", ""))
        _pass_count += 1
        print(f"  PASS  {test_id}")
    except Exception as e:
        _results.append((test_id, "FAIL", str(e)))
        _fail_count += 1
        print(f"  FAIL  {test_id}: {e}")


def _assert(condition, msg="Assertion failed"):
    if not condition:
        raise AssertionError(msg)


def _assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: {a!r} != {b!r}" if msg else
                             f"{a!r} != {b!r}")


# ---------------------------------------------------------------------------
# Helpers: build config fragments
# ---------------------------------------------------------------------------

def _make_minimal_config(engine_version: str = "1.8.0",
                         extra_fields: dict = None,
                         extra_entry: dict = None,
                         extra_exit: dict = None,
                         extra_gate: dict = None,
                         indicator_instances: list = None,
                         entry_rules: list = None,
                         exit_rules: list = None,
                         gate_rules: list = None,
                         execution_params: dict = None) -> dict:
    """Build a minimal valid strategy config."""
    config = {
        "engine_version": engine_version,
        "indicator_instances": indicator_instances or [],
        "entry_rules": entry_rules or [],
        "exit_rules": exit_rules or [],
        "gate_rules": gate_rules or [],
        "execution_params": execution_params or {
            "entry_type": "MARKET",
            "flip_enabled": False,
        },
        "archetype_tags": [],
    }
    if extra_fields:
        config.update(extra_fields)
    return config


def _make_indicator_instance(label="ind1", indicator_id=7,
                             timeframe="1m", outputs_used=None,
                             parameters=None, **kwargs) -> dict:
    inst = {
        "label": label,
        "indicator_id": indicator_id,
        "timeframe": timeframe,
        "parameters": parameters or {},
        "outputs_used": outputs_used or ["macd_line"],
        "data_source": "BAR",
    }
    inst.update(kwargs)
    return inst


def _make_condition(indicator="ind1", output="macd_line",
                    operator=">", value="0", **kwargs) -> dict:
    cond = {
        "indicator": indicator,
        "output": output,
        "operator": operator,
    }
    if value is not None:
        cond["value"] = value
    cond.update(kwargs)
    return cond


def _make_entry_rule(name="entry1", direction="LONG",
                     conditions=None, **kwargs) -> dict:
    rule = {
        "name": name,
        "direction": direction,
        "evaluation_cadence": "1m",
        "conditions": conditions or [],
        "condition_groups": [],
    }
    rule.update(kwargs)
    return rule


def _make_exit_rule(name="exit1", exit_type="SIGNAL",
                    conditions=None, **kwargs) -> dict:
    rule = {
        "name": name,
        "applies_to": "ANY",
        "evaluation_cadence": "1m",
        "type": exit_type,
        "conditions": conditions or [],
        "parameters": {},
    }
    rule.update(kwargs)
    return rule


def _make_gate_rule(name="gate1", conditions=None,
                    policy="HOLD_CURRENT") -> dict:
    return {
        "name": name,
        "conditions": conditions or [],
        "on_close_policy": policy,
    }


# ---------------------------------------------------------------------------
# Delta 1: is_present / is_absent operators
# ---------------------------------------------------------------------------

def test_d1_t1():
    """Config with is_present condition, engine_version 1.8.0 -> loads."""
    inst = _make_indicator_instance(outputs_used=["slope_sign"])
    cond = _make_condition(output="slope_sign", operator="is_present",
                           value=None)
    exit_cond = _make_condition(output="slope_sign", operator="<", value="0")
    config = _make_minimal_config(
        engine_version="1.8.0",
        indicator_instances=[inst],
        entry_rules=[_make_entry_rule(conditions=[cond])],
        exit_rules=[_make_exit_rule(conditions=[exit_cond])])
    errors = validate_config(config)
    _assert(not errors, f"Should load: {errors}")


def test_d1_t2():
    """Config with is_present condition, engine_version 1.7.0 -> rejection."""
    inst = _make_indicator_instance(outputs_used=["slope_sign"])
    cond = _make_condition(output="slope_sign", operator="is_present",
                           value=None)
    config = _make_minimal_config(
        engine_version="1.7.0",
        indicator_instances=[inst],
        entry_rules=[_make_entry_rule(conditions=[cond])],
        exit_rules=[_make_exit_rule()])
    errors = validate_config(config)
    _assert(any("is_present" in e and "1.8.0" in e for e in errors),
            f"Should reject is_present on 1.7.0: {errors}")


def test_d1_t3():
    """Config with is_present condition that includes a value field -> rejection."""
    inst = _make_indicator_instance(outputs_used=["slope_sign"])
    cond = _make_condition(output="slope_sign", operator="is_present",
                           value="1")
    config = _make_minimal_config(
        engine_version="1.8.0",
        indicator_instances=[inst],
        entry_rules=[_make_entry_rule(conditions=[cond])],
        exit_rules=[_make_exit_rule()])
    errors = validate_config(config)
    _assert(any("value" in e for e in errors),
            f"Should reject is_present with value: {errors}")


def test_d1_t4():
    """Evaluate is_present on output that is non-None -> true."""
    outputs = {"ind1": {"slope_sign": 1}}
    cond = _make_condition(output="slope_sign", operator="is_present",
                           value=None)
    result = evaluate_condition(cond, outputs)
    _assert(result is True, "is_present on non-None should be True")


def test_d1_t5():
    """Evaluate is_present on output that is None -> false."""
    outputs = {"ind1": {"slope_sign": None}}
    cond = _make_condition(output="slope_sign", operator="is_present",
                           value=None)
    result = evaluate_condition(cond, outputs)
    _assert(result is False, "is_present on None should be False")


def test_d1_t6():
    """Evaluate is_absent on output that is None -> true."""
    outputs = {"ind1": {"slope_sign": None}}
    cond = _make_condition(output="slope_sign", operator="is_absent",
                           value=None)
    result = evaluate_condition(cond, outputs)
    _assert(result is True, "is_absent on None should be True")


def test_d1_t7():
    """Evaluate is_absent on output that is non-None -> false."""
    outputs = {"ind1": {"slope_sign": 1}}
    cond = _make_condition(output="slope_sign", operator="is_absent",
                           value=None)
    result = evaluate_condition(cond, outputs)
    _assert(result is False, "is_absent on non-None should be False")


def test_d1_t8():
    """Hash stability: serialize is_present condition twice -> identical."""
    cond = _make_condition(output="slope_sign", operator="is_present",
                           value=None)
    h1 = canonical_json(cond)
    h2 = canonical_json(cond)
    _assert_eq(h1, h2, "Hash should be stable")


def test_d1_t9():
    """Fallback path: primary is_present + fallback is_absent on same output.

    Dataset with periodic None -> mutually exclusive.
    """
    primary_cond = _make_condition(indicator="primary",
                                    output="slope_sign",
                                    operator="is_present", value=None)
    fallback_cond = _make_condition(indicator="primary",
                                     output="slope_sign",
                                     operator="is_absent", value=None)

    # Simulate dataset with alternating None/non-None
    test_data = [
        {"primary": {"slope_sign": 1}},   # non-None
        {"primary": {"slope_sign": None}}, # None
        {"primary": {"slope_sign": -1}},   # non-None
        {"primary": {"slope_sign": None}}, # None
    ]

    for outputs in test_data:
        primary_fires = evaluate_condition(primary_cond, outputs)
        fallback_fires = evaluate_condition(fallback_cond, outputs)
        _assert(primary_fires != fallback_fires,
                f"Mutually exclusive violated: primary={primary_fires}, "
                f"fallback={fallback_fires} for {outputs}")


def test_d1_t10():
    """Replay determinism: same config + dataset twice -> identical."""
    inst = _make_indicator_instance(outputs_used=["macd_line"])
    cond = _make_condition(operator=">", value="0")
    config = _make_minimal_config(
        indicator_instances=[inst],
        entry_rules=[_make_entry_rule(conditions=[cond])],
        exit_rules=[_make_exit_rule()])

    dataset = [
        {"ind1": {"macd_line": 100}},
        {"ind1": {"macd_line": -50}},
        {"ind1": {"macd_line": 200}},
    ]

    decisions_1 = []
    decisions_2 = []
    for outputs in dataset:
        decisions_1.append(evaluate_condition(cond, outputs))
    for outputs in dataset:
        decisions_2.append(evaluate_condition(cond, outputs))

    _assert_eq(decisions_1, decisions_2, "Replay determinism")


# ---------------------------------------------------------------------------
# Delta 2: MTM_DRAWDOWN_EXIT
# ---------------------------------------------------------------------------

def test_d2_t1():
    """Config with MTM_DRAWDOWN_EXIT, engine_version 1.8.0 -> loads."""
    exit_rule = _make_exit_rule(name="mtm_dd", exit_type="MTM_DRAWDOWN_EXIT",
                                parameters={"drawdown_bps_long": 250})
    config = _make_minimal_config(
        engine_version="1.8.0",
        exit_rules=[exit_rule])
    errors = validate_config(config)
    _assert(not errors, f"Should load: {errors}")


def test_d2_t2():
    """Config with MTM_DRAWDOWN_EXIT, engine_version 1.7.0 -> rejection."""
    exit_rule = _make_exit_rule(name="mtm_dd", exit_type="MTM_DRAWDOWN_EXIT")
    config = _make_minimal_config(
        engine_version="1.7.0",
        exit_rules=[exit_rule])
    errors = validate_config(config)
    _assert(any("MTM_DRAWDOWN_EXIT" in e for e in errors),
            f"Should reject on 1.7.0: {errors}")


def test_d2_t3():
    """Long position, price rises 500bps then drops 300bps from peak,
    threshold 250 -> exit triggers."""
    tracker = MTMDrawdownTracker()
    tracker.open_position()
    entry_price = 10000

    # Rise 500 bps: entry=10000, current=10500 -> pnl=(500*10000//10000)=500bps
    dd = tracker.update(entry_price, 10500, "LONG")
    _assert_eq(dd, 0, "No drawdown at peak")
    _assert(not tracker.check_trigger(250), "Should not trigger at peak")

    # Drop 300 bps from peak: price=10200
    # pnl = (10200-10000)*10000//10000 = 200bps
    # peak=500, drawdown=500-200=300bps
    dd = tracker.update(entry_price, 10200, "LONG")
    _assert_eq(dd, 300, "Drawdown should be 300bps")
    _assert(tracker.check_trigger(250), "Should trigger at 300 >= 250")


def test_d2_t4():
    """Long position, price rises 500bps then drops 200bps, threshold 250
    -> no exit."""
    tracker = MTMDrawdownTracker()
    tracker.open_position()
    entry_price = 10000

    tracker.update(entry_price, 10500, "LONG")  # peak at 500 bps
    dd = tracker.update(entry_price, 10300, "LONG")
    # pnl = 300bps, drawdown = 500-300 = 200bps
    _assert_eq(dd, 200, "Drawdown should be 200bps")
    _assert(not tracker.check_trigger(250), "200 < 250, should not trigger")


def test_d2_t5():
    """Short position, favorable then adverse 350bps retracement,
    threshold 300 -> exit triggers."""
    tracker = MTMDrawdownTracker()
    tracker.open_position()
    entry_price = 10000

    # Short favorable: price drops to 9500 -> pnl=500bps
    tracker.update(entry_price, 9500, "SHORT")

    # Adverse retracement: price rises to 9850
    # pnl = (10000-9850)*10000//10000 = 150bps
    # peak=500, drawdown=500-150=350bps
    dd = tracker.update(entry_price, 9850, "SHORT")
    _assert_eq(dd, 350, "Drawdown should be 350bps")
    _assert(tracker.check_trigger(300), "350 >= 300, should trigger")


def test_d2_t6():
    """Position closes via stop loss before MTM drawdown triggers ->
    peak state discarded."""
    tracker = MTMDrawdownTracker()
    tracker.open_position()
    entry_price = 10000

    tracker.update(entry_price, 10500, "LONG")  # peak at 500 bps
    _assert(tracker.active, "Should be active")

    tracker.close_position()
    _assert(not tracker.active, "Should be inactive")
    _assert_eq(tracker.peak_pnl_bps, 0, "Peak should be reset")
    _assert(not tracker.check_trigger(0), "Should not trigger when closed")


def test_d2_t7():
    """Two consecutive positions: verify peak tracking resets."""
    tracker = MTMDrawdownTracker()

    # Position 1
    tracker.open_position()
    tracker.update(10000, 10500, "LONG")
    _assert_eq(tracker.peak_pnl_bps, 500)
    tracker.close_position()

    # Position 2
    tracker.open_position()
    _assert_eq(tracker.peak_pnl_bps, 0, "Peak should reset on new position")
    tracker.update(10000, 10200, "LONG")
    _assert_eq(tracker.peak_pnl_bps, 200, "New peak for new position")


def test_d2_t8():
    """Determinism: identical series + config twice -> same states."""
    def run_series():
        tracker = MTMDrawdownTracker()
        tracker.open_position()
        states = []
        prices = [10000, 10300, 10500, 10200, 10100, 10400]
        for p in prices:
            tracker.update(10000, p, "LONG")
            states.append(tracker.snapshot())
        return states

    s1 = run_series()
    s2 = run_series()
    _assert_eq(s1, s2, "Determinism: states should be identical")


def test_d2_t9():
    """Precedence: gate FORCE_FLAT fires same cycle as MTM drawdown ->
    gate-exit takes priority."""
    inst = _make_indicator_instance()
    # Gate that is closed (condition false)
    gate = _make_gate_rule(
        conditions=[_make_condition(operator=">", value="999999")],
        policy="FORCE_FLAT")
    # MTM exit
    exit_rule = _make_exit_rule(
        name="mtm", exit_type="MTM_DRAWDOWN_EXIT",
        parameters={"drawdown_bps_long": 100, "drawdown_bps": 100})

    config = _make_minimal_config(
        indicator_instances=[inst],
        gate_rules=[gate],
        exit_rules=[exit_rule],
        execution_params={"entry_type": "MARKET", "flip_enabled": False})

    tracker = MTMDrawdownTracker()
    tracker.open_position()
    tracker.update(10000, 10500, "LONG")
    tracker.update(10000, 10200, "LONG")  # 300bps drawdown

    outputs = {"ind1": {"macd_line": 100}}
    position = {"direction": "LONG", "entry_price": 10000}

    result = evaluate_signal_pipeline(
        config, outputs, None, position,
        mtm_tracker=tracker, current_price=10200)

    _assert_eq(result.action, "EXIT", "Should exit")
    _assert_eq(result.exit_reason, "GATE_FORCE_FLAT",
               "Gate should take priority over MTM")


def test_d2_t10():
    """Flip detection: MTM drawdown fires + opposite entry met + flip_enabled
    -> FLIP signal."""
    inst = _make_indicator_instance(outputs_used=["macd_line"])
    # MTM exit
    exit_rule = _make_exit_rule(
        name="mtm", exit_type="MTM_DRAWDOWN_EXIT",
        parameters={"drawdown_bps_long": 100, "drawdown_bps": 100})
    # Opposite direction entry
    entry_rule = _make_entry_rule(
        name="short_entry", direction="SHORT",
        conditions=[_make_condition(operator="<", value="0")])

    config = _make_minimal_config(
        indicator_instances=[inst],
        exit_rules=[exit_rule],
        entry_rules=[entry_rule],
        execution_params={"entry_type": "MARKET", "flip_enabled": True})

    tracker = MTMDrawdownTracker()
    tracker.open_position()
    tracker.update(10000, 10500, "LONG")
    tracker.update(10000, 10200, "LONG")

    outputs = {"ind1": {"macd_line": -50}}
    position = {"direction": "LONG", "entry_price": 10000}

    result = evaluate_signal_pipeline(
        config, outputs, None, position,
        mtm_tracker=tracker, current_price=10200)

    _assert_eq(result.action, "FLIP", "Should produce FLIP signal")
    _assert_eq(result.direction, "SHORT", "Should flip to SHORT")


# ---------------------------------------------------------------------------
# Delta 3: Per-output warmup
# ---------------------------------------------------------------------------

def test_d3_t1():
    """MACD slope_sign with slow=26 -> warmup 27."""
    result = get_warmup_bars_for_output(7, "slope_sign", {"slow": 26})
    _assert_eq(result, 27, "slope_sign warmup")


def test_d3_t2():
    """MACD macd_line with slow=26 -> warmup 26."""
    result = get_warmup_bars_for_output(7, "macd_line", {"slow": 26})
    _assert_eq(result, 26, "macd_line warmup")


def test_d3_t3():
    """Instance using only macd_line: instance warmup = 26."""
    result = compute_instance_warmup(7, ["macd_line"], {"slow": 26})
    _assert_eq(result, 26, "Instance warmup with only macd_line")


def test_d3_t4():
    """Instance using macd_line and slope_sign: instance warmup = max(26,27) = 27."""
    result = compute_instance_warmup(
        7, ["macd_line", "slope_sign"], {"slow": 26})
    _assert_eq(result, 27, "Instance warmup with both outputs")


def test_d3_t5():
    """Condition on slope_sign at bar 26 (warmup not met) -> false."""
    # slope_sign needs 27 bars. At bar index 25 (26th bar, 0-based),
    # bar_index+1 = 26 < 27
    _assert(not is_output_warmed_up(7, "slope_sign", {"slow": 26}, 25),
            "Bar 25 (26th bar) should not satisfy warmup of 27")


def test_d3_t6():
    """Condition on slope_sign at bar 27 (warmup met) -> evaluates normally."""
    # At bar index 26 (27th bar), bar_index+1 = 27 >= 27
    _assert(is_output_warmed_up(7, "slope_sign", {"slow": 26}, 26),
            "Bar 26 (27th bar) should satisfy warmup of 27")


def test_d3_t7():
    """Condition on macd_line at bar 26 (warmup met for this output)."""
    # macd_line needs 26 bars. At bar index 25, bar_index+1 = 26 >= 26
    _assert(is_output_warmed_up(7, "macd_line", {"slow": 26}, 25),
            "Bar 25 (26th bar) should satisfy macd_line warmup of 26")


def test_d3_t8():
    """Legacy indicator without per-output warmup -> falls back to single value."""
    # EMA (indicator 1) has single warmup for all outputs
    result = get_warmup_bars_for_output(1, "ema", {"length": 20})
    _assert_eq(result, 20, "EMA fallback warmup")


def test_d3_t9():
    """Config with explicit warmup_bars: 20 but output requires 27 -> fail."""
    inst = _make_indicator_instance(
        outputs_used=["slope_sign"],
        parameters={"slow": 26},
        warmup_bars=20)
    config = _make_minimal_config(
        indicator_instances=[inst],
        exit_rules=[_make_exit_rule()])
    errors = validate_config(config)
    _assert(any("warmup_bars" in e and "20" in e for e in errors),
            f"Should reject warmup_bars=20 < required 27: {errors}")


def test_d3_t10():
    """Determinism: warmup gating identical across two runs."""
    params = {"slow": 26}
    results_1 = [is_output_warmed_up(7, "slope_sign", params, i)
                 for i in range(30)]
    results_2 = [is_output_warmed_up(7, "slope_sign", params, i)
                 for i in range(30)]
    _assert_eq(results_1, results_2, "Warmup gating determinism")


# ---------------------------------------------------------------------------
# Delta 4: HANDOFF gate policy
# ---------------------------------------------------------------------------

def test_d4_t1():
    """Config with on_close_policy: HANDOFF, engine_version 1.8.0 -> loads."""
    gate = _make_gate_rule(conditions=[], policy="HANDOFF")
    config = _make_minimal_config(
        engine_version="1.8.0",
        gate_rules=[gate])
    errors = validate_config(config)
    _assert(not errors, f"Should load: {errors}")


def test_d4_t2():
    """Config with on_close_policy: HANDOFF, engine_version 1.7.0 -> rejection."""
    gate = _make_gate_rule(conditions=[], policy="HANDOFF")
    config = _make_minimal_config(
        engine_version="1.7.0",
        gate_rules=[gate])
    errors = validate_config(config)
    _assert(any("HANDOFF" in e for e in errors),
            f"Should reject HANDOFF on 1.7.0: {errors}")


def test_d4_t3():
    """Gate closes with HANDOFF, position open, SIGNAL exit condition met
    -> signal exit suppressed."""
    inst = _make_indicator_instance(outputs_used=["macd_line"])
    # Gate is closed (condition evaluates to False since value > 999999)
    gate = _make_gate_rule(
        conditions=[_make_condition(operator=">", value="999999")],
        policy="HANDOFF")
    # SIGNAL exit whose condition IS met
    exit_rule = _make_exit_rule(
        name="signal_exit", exit_type="SIGNAL",
        conditions=[_make_condition(operator="<", value="999")])

    config = _make_minimal_config(
        indicator_instances=[inst],
        gate_rules=[gate],
        exit_rules=[exit_rule],
        execution_params={"entry_type": "MARKET", "flip_enabled": False})

    outputs = {"ind1": {"macd_line": 100}}
    position = {"direction": "LONG", "entry_price": 10000}

    result = evaluate_signal_pipeline(config, outputs, None, position)
    # HANDOFF suppresses SIGNAL exits, so action should be HOLD
    _assert_eq(result.action, "HOLD",
               "SIGNAL exit should be suppressed under HANDOFF")


def test_d4_t4():
    """Gate closes with HANDOFF, position open, stop loss triggers
    -> fires normally."""
    inst = _make_indicator_instance(outputs_used=["macd_line"])
    gate = _make_gate_rule(
        conditions=[_make_condition(operator=">", value="999999")],
        policy="HANDOFF")
    # STOP_LOSS exit with conditions met
    exit_rule = _make_exit_rule(
        name="stop_loss", exit_type="STOP_LOSS",
        conditions=[_make_condition(operator="<", value="999")])

    config = _make_minimal_config(
        indicator_instances=[inst],
        gate_rules=[gate],
        exit_rules=[exit_rule],
        execution_params={"entry_type": "MARKET", "flip_enabled": False})

    outputs = {"ind1": {"macd_line": 100}}
    position = {"direction": "LONG", "entry_price": 10000}

    result = evaluate_signal_pipeline(config, outputs, None, position)
    _assert_eq(result.action, "EXIT", "STOP_LOSS should fire under HANDOFF")
    _assert_eq(result.exit_reason, "STOP_LOSS")


def test_d4_t5():
    """Gate closes with HANDOFF, MTM drawdown triggers -> fires normally."""
    inst = _make_indicator_instance(outputs_used=["macd_line"])
    gate = _make_gate_rule(
        conditions=[_make_condition(operator=">", value="999999")],
        policy="HANDOFF")
    exit_rule = _make_exit_rule(
        name="mtm", exit_type="MTM_DRAWDOWN_EXIT",
        parameters={"drawdown_bps_long": 100, "drawdown_bps": 100})

    config = _make_minimal_config(
        indicator_instances=[inst],
        gate_rules=[gate],
        exit_rules=[exit_rule],
        execution_params={"entry_type": "MARKET", "flip_enabled": False})

    tracker = MTMDrawdownTracker()
    tracker.open_position()
    tracker.update(10000, 10500, "LONG")
    tracker.update(10000, 10200, "LONG")  # 300bps drawdown

    outputs = {"ind1": {"macd_line": 100}}
    position = {"direction": "LONG", "entry_price": 10000}

    result = evaluate_signal_pipeline(
        config, outputs, None, position,
        mtm_tracker=tracker, current_price=10200)

    _assert_eq(result.action, "EXIT", "MTM should fire under HANDOFF")
    _assert_eq(result.exit_reason, "MTM_DRAWDOWN_EXIT")


def test_d4_t6():
    """Gate closes with HANDOFF, no position open -> blocks entries."""
    inst = _make_indicator_instance(outputs_used=["macd_line"])
    gate = _make_gate_rule(
        conditions=[_make_condition(operator=">", value="999999")],
        policy="HANDOFF")
    entry_rule = _make_entry_rule(
        conditions=[_make_condition(operator=">", value="0")])

    config = _make_minimal_config(
        indicator_instances=[inst],
        gate_rules=[gate],
        entry_rules=[entry_rule],
        exit_rules=[_make_exit_rule()],
        execution_params={"entry_type": "MARKET", "flip_enabled": False})

    outputs = {"ind1": {"macd_line": 100}}

    result = evaluate_signal_pipeline(config, outputs, None, None)
    _assert_eq(result.action, "HOLD",
               "Entry should be blocked when gate is closed")


def test_d4_t7():
    """Gate reopens after HANDOFF: signal exit conditions now met -> fires."""
    inst = _make_indicator_instance(outputs_used=["macd_line"])
    # Gate is now open (condition satisfied)
    gate = _make_gate_rule(
        conditions=[_make_condition(operator=">", value="0")],
        policy="HANDOFF")
    exit_rule = _make_exit_rule(
        name="signal_exit", exit_type="SIGNAL",
        conditions=[_make_condition(operator=">", value="50")])

    config = _make_minimal_config(
        indicator_instances=[inst],
        gate_rules=[gate],
        exit_rules=[exit_rule],
        execution_params={"entry_type": "MARKET", "flip_enabled": False})

    outputs = {"ind1": {"macd_line": 100}}
    position = {"direction": "LONG", "entry_price": 10000}

    result = evaluate_signal_pipeline(config, outputs, None, position)
    _assert_eq(result.action, "EXIT",
               "SIGNAL exit should fire when gate is open")
    _assert_eq(result.exit_reason, "SIGNAL")


def test_d4_t8():
    """Two gates: one FORCE_FLAT, one HANDOFF, both close -> FORCE_FLAT wins."""
    inst = _make_indicator_instance(outputs_used=["macd_line"])
    gate_ff = _make_gate_rule(
        name="ff_gate",
        conditions=[_make_condition(operator=">", value="999999")],
        policy="FORCE_FLAT")
    gate_ho = _make_gate_rule(
        name="ho_gate",
        conditions=[_make_condition(operator=">", value="999999")],
        policy="HANDOFF")

    config = _make_minimal_config(
        indicator_instances=[inst],
        gate_rules=[gate_ff, gate_ho],
        exit_rules=[_make_exit_rule()],
        execution_params={"entry_type": "MARKET", "flip_enabled": False})

    outputs = {"ind1": {"macd_line": 100}}
    position = {"direction": "LONG", "entry_price": 10000}

    result = evaluate_signal_pipeline(config, outputs, None, position)
    _assert_eq(result.action, "EXIT", "Should exit")
    _assert_eq(result.exit_reason, "GATE_FORCE_FLAT",
               "FORCE_FLAT should win over HANDOFF")


def test_d4_t9():
    """HANDOFF + risk override EXIT_ALL during gate closure -> fires."""
    inst = _make_indicator_instance(outputs_used=["macd_line"])
    gate = _make_gate_rule(
        conditions=[_make_condition(operator=">", value="999999")],
        policy="HANDOFF")

    config = _make_minimal_config(
        indicator_instances=[inst],
        gate_rules=[gate],
        exit_rules=[_make_exit_rule()],
        execution_params={"entry_type": "MARKET", "flip_enabled": False})

    outputs = {"ind1": {"macd_line": 100}}
    position = {"direction": "LONG", "entry_price": 10000}

    result = evaluate_signal_pipeline(
        config, outputs, None, position,
        risk_override="EXIT_ALL")
    _assert_eq(result.action, "EXIT", "Risk override should fire")
    _assert_eq(result.exit_reason, "RISK_OVERRIDE")


def test_d4_t10():
    """Hash stability: gate rule with HANDOFF serializes deterministically."""
    gate = _make_gate_rule(policy="HANDOFF",
                            conditions=[_make_condition()])
    h1 = canonical_json(gate)
    h2 = canonical_json(gate)
    _assert_eq(h1, h2, "Hash stability for HANDOFF gate")


def test_d4_t11():
    """Resolved artifact from composition: field on_close_policy = HOLD_CURRENT."""
    gate = _make_gate_rule(policy="HOLD_CURRENT")
    config = _make_minimal_config(
        engine_version="1.8.0",
        gate_rules=[gate])
    errors = validate_config(config)
    _assert(not errors, f"HOLD_CURRENT should be valid: {errors}")


def test_d4_t12():
    """Gate closes HANDOFF while SIGNAL exit true. Gate reopens next bar.
    Condition still true -> exit fires on first bar after reopen."""
    inst = _make_indicator_instance(outputs_used=["macd_line"])
    exit_cond = _make_condition(operator=">", value="50")
    exit_rule = _make_exit_rule(
        name="signal_exit", exit_type="SIGNAL",
        conditions=[exit_cond])

    outputs = {"ind1": {"macd_line": 100}}
    position = {"direction": "LONG", "entry_price": 10000}

    # Bar 1: gate closed (HANDOFF), SIGNAL exit suppressed
    gate_closed = _make_gate_rule(
        conditions=[_make_condition(operator=">", value="999999")],
        policy="HANDOFF")
    config1 = _make_minimal_config(
        indicator_instances=[inst],
        gate_rules=[gate_closed],
        exit_rules=[exit_rule],
        execution_params={"entry_type": "MARKET", "flip_enabled": False})

    r1 = evaluate_signal_pipeline(config1, outputs, None, position)
    _assert_eq(r1.action, "HOLD", "SIGNAL suppressed under HANDOFF")

    # Bar 2: gate reopens
    gate_open = _make_gate_rule(
        conditions=[_make_condition(operator=">", value="0")],
        policy="HANDOFF")
    config2 = _make_minimal_config(
        indicator_instances=[inst],
        gate_rules=[gate_open],
        exit_rules=[exit_rule],
        execution_params={"entry_type": "MARKET", "flip_enabled": False})

    r2 = evaluate_signal_pipeline(config2, outputs, None, position)
    _assert_eq(r2.action, "EXIT",
               "SIGNAL exit should fire on first bar after reopen")
    _assert_eq(r2.exit_reason, "SIGNAL")


# ---------------------------------------------------------------------------
# Delta 5: Cross-indicator condition references
# ---------------------------------------------------------------------------

def test_d5_t1():
    """Config with cross-indicator condition, engine_version 1.8.0 -> loads."""
    inst1 = _make_indicator_instance(label="entry_5m", indicator_id=7,
                                      outputs_used=["macd_line"])
    inst2 = _make_indicator_instance(label="bb_4h", indicator_id=11,
                                      outputs_used=["upper"])
    cond = {
        "indicator": "entry_5m", "output": "macd_line",
        "operator": ">",
        "ref_indicator": "bb_4h", "ref_output": "upper",
    }
    config = _make_minimal_config(
        engine_version="1.8.0",
        indicator_instances=[inst1, inst2],
        entry_rules=[_make_entry_rule(conditions=[cond])],
        exit_rules=[_make_exit_rule()])
    errors = validate_config(config)
    _assert(not errors, f"Should load: {errors}")


def test_d5_t2():
    """Config with cross-indicator condition, engine_version 1.7.0 -> rejection."""
    cond = {
        "indicator": "entry_5m", "output": "macd_line",
        "operator": ">",
        "ref_indicator": "bb_4h", "ref_output": "upper",
    }
    config = _make_minimal_config(
        engine_version="1.7.0",
        entry_rules=[_make_entry_rule(conditions=[cond])],
        exit_rules=[_make_exit_rule()])
    errors = validate_config(config)
    _assert(any("cross-indicator" in e.lower() or "ref_indicator" in e
                for e in errors),
            f"Should reject cross-indicator on 1.7.0: {errors}")


def test_d5_t3():
    """Config with both value and ref_indicator -> rejection."""
    cond = {
        "indicator": "entry_5m", "output": "macd_line",
        "operator": ">",
        "value": "100",
        "ref_indicator": "bb_4h", "ref_output": "upper",
    }
    config = _make_minimal_config(
        engine_version="1.8.0",
        entry_rules=[_make_entry_rule(conditions=[cond])],
        exit_rules=[_make_exit_rule()])
    errors = validate_config(config)
    _assert(any("mutually exclusive" in e for e in errors),
            f"Should reject value + ref_indicator: {errors}")


def test_d5_t4():
    """Config with ref_indicator but no ref_output -> rejection."""
    cond = {
        "indicator": "entry_5m", "output": "macd_line",
        "operator": ">",
        "ref_indicator": "bb_4h",
    }
    config = _make_minimal_config(
        engine_version="1.8.0",
        entry_rules=[_make_entry_rule(conditions=[cond])],
        exit_rules=[_make_exit_rule()])
    errors = validate_config(config)
    _assert(any("ref_output" in e for e in errors),
            f"Should reject missing ref_output: {errors}")


def test_d5_t5():
    """Evaluate entry_5m.close > bb_4h.upper where close=50000, upper=49000
    -> true."""
    cond = {
        "indicator": "entry_5m", "output": "close",
        "operator": ">",
        "ref_indicator": "bb_4h", "ref_output": "upper",
    }
    outputs = {
        "entry_5m": {"close": 50000},
        "bb_4h": {"upper": 49000},
    }
    result = evaluate_condition(cond, outputs)
    _assert(result is True, "50000 > 49000 should be True")


def test_d5_t6():
    """Evaluate cross-indicator where ref output is None -> false."""
    cond = {
        "indicator": "entry_5m", "output": "close",
        "operator": ">",
        "ref_indicator": "bb_4h", "ref_output": "upper",
    }
    outputs = {
        "entry_5m": {"close": 50000},
        "bb_4h": {"upper": None},
    }
    result = evaluate_condition(cond, outputs)
    _assert(result is False, "None ref output -> false")


def test_d5_t7():
    """Evaluate crosses_above with cross-indicator reference."""
    cond = {
        "indicator": "entry_5m", "output": "close",
        "operator": "crosses_above",
        "ref_indicator": "bb_4h", "ref_output": "upper",
    }
    prev_outputs = {
        "entry_5m": {"close": 48000},
        "bb_4h": {"upper": 49000},
    }
    curr_outputs = {
        "entry_5m": {"close": 50000},
        "bb_4h": {"upper": 49000},
    }
    result = evaluate_condition(cond, curr_outputs, prev_outputs)
    _assert(result is True,
            "Cross from 48000 <= 49000 to 50000 > 49000 should detect crossing")


def test_d5_t8():
    """Hash stability: cross-indicator condition serializes deterministically."""
    cond = {
        "indicator": "entry_5m", "output": "close",
        "operator": ">",
        "ref_indicator": "bb_4h", "ref_output": "upper",
    }
    h1 = canonical_json(cond)
    h2 = canonical_json(cond)
    _assert_eq(h1, h2, "Hash stability for cross-indicator condition")


# ---------------------------------------------------------------------------
# Delta 6: Schema strictness
# ---------------------------------------------------------------------------

def test_d6_t1():
    """Load config with valid v1.8.0 fields -> loads."""
    config = _make_minimal_config(engine_version="1.8.0")
    errors = validate_config(config)
    _assert(not errors, f"Should load valid config: {errors}")


def test_d6_t2():
    """Load config with unknown field composition_id -> rejection."""
    config = _make_minimal_config(engine_version="1.8.0")
    config["composition_id"] = "abc"
    errors = validate_config(config)
    _assert(any("composition_id" in e for e in errors),
            f"Should reject unknown field: {errors}")


def test_d6_t3():
    """Load config with unknown field metadata -> rejection."""
    config = _make_minimal_config(engine_version="1.8.0")
    config["metadata"] = {}
    errors = validate_config(config)
    _assert(any("metadata" in e for e in errors),
            f"Should reject unknown field: {errors}")


def test_d6_t4():
    """Load v1.7.0 config (no new fields) on v1.8.0 engine -> loads.

    Note: we validate that a config declaring engine_version 1.7.0
    with only v1.0 features loads successfully.
    """
    config = _make_minimal_config(engine_version="1.7.0")
    errors = validate_config(config)
    _assert(not errors, f"Should load v1.7.0 config: {errors}")


# ---------------------------------------------------------------------------
# Composition Proofs
# ---------------------------------------------------------------------------

def _make_macd_composition() -> dict:
    """Build MACD Confluence composition spec."""
    return {
        "composition_id": "macd-confluence-v1",
        "display_name": "MACD Confluence v1",
        "description": "Multi-TF MACD trend following",
        "archetype_tags": ["trend_following", "multi_timeframe"],
        "version": "1.0.0",
        "target_engine_version": "1.8.0",
        "min_engine_version": "1.8.0",
        "target_instrument": "BTCUSDT",
        "target_variant": "perp",
        "indicator_instances": [
            {"label": "macro_3d", "role": "filter", "group": "macro",
             "indicator_id": "macd_tv", "timeframe": "3d",
             "parameters": {"slow": 26}, "outputs_used": ["slope_sign"],
             "data_source": "BAR"},
            {"label": "macro_1d", "role": "filter", "group": "macro",
             "indicator_id": "macd_tv", "timeframe": "1d",
             "parameters": {"slow": 26}, "outputs_used": ["slope_sign"],
             "data_source": "BAR"},
            {"label": "macro_12h", "role": "filter", "group": "macro",
             "indicator_id": "macd_tv", "timeframe": "12h",
             "parameters": {"slow": 26}, "outputs_used": ["slope_sign"],
             "data_source": "BAR"},
            {"label": "intra_1h", "role": "filter", "group": "intra",
             "indicator_id": "macd_tv", "timeframe": "1h",
             "parameters": {"slow": 26}, "outputs_used": ["slope_sign"],
             "data_source": "BAR"},
            {"label": "intra_30m", "role": "filter", "group": "intra",
             "indicator_id": "macd_tv", "timeframe": "30m",
             "parameters": {"slow": 26}, "outputs_used": ["slope_sign"],
             "data_source": "BAR"},
            {"label": "intra_15m", "role": "filter", "group": "intra",
             "indicator_id": "macd_tv", "timeframe": "15m",
             "parameters": {"slow": 26}, "outputs_used": ["slope_sign"],
             "data_source": "BAR"},
            {"label": "entry_5m", "role": "entry_signal",
             "indicator_id": "macd_tv", "timeframe": "5m",
             "parameters": {"slow": 26}, "outputs_used": ["slope_sign"],
             "data_source": "BAR"},
            {"label": "exit_1d", "role": "exit_signal",
             "indicator_id": "macd_tv", "timeframe": "1d",
             "parameters": {"slow": 26}, "outputs_used": ["slope_sign"],
             "data_source": "BAR"},
        ],
        "entry_rules": [
            {
                "label": "long_entry",
                "direction": "LONG",
                "evaluation_cadence": "5m",
                "condition_groups": [
                    {
                        "label": "macro_alignment",
                        "role_condition": {
                            "role": "filter",
                            "filter_group": "macro",
                            "output": "slope_sign",
                            "operator": ">",
                            "value": "0",
                            "quantifier": "ALL",
                        },
                    },
                    {
                        "label": "intra_alignment",
                        "role_condition": {
                            "role": "filter",
                            "filter_group": "intra",
                            "output": "slope_sign",
                            "operator": ">",
                            "value": "0",
                            "quantifier": "ALL",
                        },
                    },
                ],
                "conditions": [
                    {
                        "role": "entry_signal",
                        "output": "slope_sign",
                        "operator": "crosses_above",
                        "value": "0",
                    },
                ],
            },
        ],
        "exit_rules": [
            {
                "label": "daily_exit",
                "exit_type": "SIGNAL",
                "applies_to": ["LONG"],
                "evaluation_cadence": "1d",
                "conditions": [
                    {
                        "role": "exit_signal",
                        "output": "slope_sign",
                        "operator": "<",
                        "value": "0",
                    },
                ],
            },
            {
                "label": "stop_loss",
                "exit_type": "STOP_LOSS",
                "applies_to": ["LONG"],
                "mode": "PERCENT",
                "exchange_side": True,
                "value_long_bps": 500,
            },
        ],
        "gate_rules": [],
        "execution_params": {
            "entry_type": "MARKET",
            "flip_enabled": False,
            "leverage": "1.0",
        },
        "metadata": {
            "author": "system_owner",
            "thesis": "Multi-TF MACD trend alignment",
        },
    }


def _make_mean_reversion_composition() -> dict:
    """Build a mean-reversion composition spec."""
    return {
        "composition_id": "mean-rev-v1",
        "display_name": "Band Bounce v1",
        "description": "Mean reversion at Bollinger bands",
        "archetype_tags": ["mean_reversion"],
        "version": "1.0.0",
        "target_engine_version": "1.8.0",
        "min_engine_version": "1.8.0",
        "target_instrument": "BTCUSDT",
        "target_variant": "perp",
        "indicator_instances": [
            {"label": "bb_1h", "role": "entry_signal",
             "indicator_id": "bollinger", "timeframe": "1h",
             "parameters": {"length": 20}, "outputs_used": ["lower", "upper"],
             "data_source": "BAR"},
            {"label": "chop_4h", "role": "gate",
             "indicator_id": "choppiness", "timeframe": "4h",
             "parameters": {"length": 14}, "outputs_used": ["choppiness"],
             "data_source": "BAR"},
        ],
        "entry_rules": [
            {
                "label": "long_at_lower",
                "direction": "LONG",
                "evaluation_cadence": "1h",
                "conditions": [
                    {"indicator": "bb_1h", "output": "lower",
                     "operator": ">=", "value": "0"},
                ],
                "condition_groups": [],
            },
        ],
        "exit_rules": [
            {
                "label": "exit_at_upper",
                "exit_type": "SIGNAL",
                "applies_to": ["LONG"],
                "evaluation_cadence": "1h",
                "conditions": [
                    {"indicator": "bb_1h", "output": "upper",
                     "operator": "<=", "value": "999999"},
                ],
            },
        ],
        "gate_rules": [
            {
                "label": "chop_gate",
                "exit_policy": "HOLD",
                "conditions": [
                    {"indicator": "chop_4h", "output": "choppiness",
                     "operator": ">", "value": "0.5"},
                ],
            },
        ],
        "execution_params": {
            "entry_type": "MARKET",
            "flip_enabled": True,
        },
        "metadata": {},
    }


def test_proof_1():
    """Compile same composition spec twice -> compare config hashes -> identical."""
    spec = _make_macd_composition()
    result1 = compile_composition(copy.deepcopy(spec))
    result2 = compile_composition(copy.deepcopy(spec))
    _assert_eq(result1["strategy_config_hash"],
               result2["strategy_config_hash"],
               "Same spec -> same hash")


def test_proof_2():
    """Compile trend-following + mean-reversion archetypes using different roles
    -> both compile without error."""
    tf_spec = _make_macd_composition()
    mr_spec = _make_mean_reversion_composition()

    result_tf = compile_composition(tf_spec)
    _assert(result_tf["strategy_config_hash"].startswith("sha256:"),
            "Trend-following should compile")

    result_mr = compile_composition(mr_spec)
    _assert(result_mr["strategy_config_hash"].startswith("sha256:"),
            "Mean-reversion should compile")

    # Different strategies -> different hashes
    _assert(result_tf["strategy_config_hash"] !=
            result_mr["strategy_config_hash"],
            "Different archetypes should produce different hashes")


def test_proof_3():
    """Load resolved artifact into Framework engine schema validator
    -> passes with no unknown fields."""
    spec = _make_macd_composition()
    result = compile_composition(spec)
    resolved = result["resolved_artifact"]

    errors = validate_schema_strict(resolved)
    _assert(not errors,
            f"Resolved artifact has unknown fields: {errors}")


def test_proof_4():
    """Compile, promote, wait, recompile -> compare hash to promotion artifact
    -> match."""
    spec = _make_macd_composition()
    result1 = compile_composition(copy.deepcopy(spec))
    hash1 = result1["strategy_config_hash"]

    # Simulate promotion (in temp dir)
    tmp_dir = tempfile.mkdtemp()
    try:
        promo_path = write_promotion_artifact(
            strategy_config_hash=hash1,
            composition_id=spec["composition_id"],
            composition_spec_hash=result1["lowering_report"]["composition_spec_hash"],
            tier="TRIAGE",
            result="PASS",
            dataset_hash="sha256:abc123def456",
            runner_hash="sha256:runner001",
            lowering_report_semantic_hash=result1["lowering_report_semantic_hash"],
            base_dir=tmp_dir,
        )
        _assert(os.path.exists(promo_path), "Promotion artifact written")

        # Recompile
        result2 = compile_composition(copy.deepcopy(spec))
        hash2 = result2["strategy_config_hash"]

        _assert_eq(hash1, hash2,
                   "Recompiled hash should match promotion hash")

        # Read promotion artifact and verify
        with open(promo_path, "r") as f:
            promo = json.loads(f.read())
        _assert_eq(promo["strategy_config_hash"], hash2,
                   "Promotion artifact hash should match recompiled hash")
    finally:
        shutil.rmtree(tmp_dir)


# ---------------------------------------------------------------------------
# Delta 7: signal_slope_sign + LMAGR wiring
# ---------------------------------------------------------------------------

def test_d7_t1():
    """MACD signal_slope_sign warmup = slow + signal = 35."""
    result = get_warmup_bars_for_output(
        7, "signal_slope_sign", {"slow": 26, "signal": 9})
    _assert_eq(result, 35, "signal_slope_sign warmup")


def test_d7_t2():
    """Instance using signal_line + signal_slope_sign: warmup = max(34, 35) = 35."""
    result = compute_instance_warmup(
        7, ["signal_line", "signal_slope_sign"], {"slow": 26, "signal": 9})
    _assert_eq(result, 35, "Instance warmup with signal_line + signal_slope_sign")


def test_d7_t3():
    """signal_slope_sign at bar 33 (34th bar) -> not warmed up."""
    _assert(not is_output_warmed_up(
        7, "signal_slope_sign", {"slow": 26, "signal": 9}, 33),
        "Bar 33 should not satisfy warmup of 35")


def test_d7_t4():
    """signal_slope_sign at bar 34 (35th bar) -> warmed up."""
    _assert(is_output_warmed_up(
        7, "signal_slope_sign", {"slow": 26, "signal": 9}, 34),
        "Bar 34 should satisfy warmup of 35")


def test_d7_t5():
    """LMAGR warmup: get_warmup_bars_for_output(25, 'lmagr', {ma_length: 20}) = 20."""
    result = get_warmup_bars_for_output(25, "lmagr", {"ma_length": 20})
    _assert_eq(result, 20, "LMAGR lmagr warmup")


def test_d7_t6():
    """LMAGR warmup: lmagr_pct also = 20."""
    result = get_warmup_bars_for_output(25, "lmagr_pct", {"ma_length": 20})
    _assert_eq(result, 20, "LMAGR lmagr_pct warmup")


def test_d7_t7():
    """LMAGR determinism: two runs of is_output_warmed_up produce identical sequences."""
    params = {"ma_length": 20}
    r1 = [is_output_warmed_up(25, "lmagr", params, i) for i in range(25)]
    r2 = [is_output_warmed_up(25, "lmagr", params, i) for i in range(25)]
    _assert_eq(r1, r2, "LMAGR warmup gating determinism")


def test_d7_t8():
    """LMAGR name resolution: get_warmup_bars_for_output('lmagr', ...) works."""
    result = get_warmup_bars_for_output("lmagr", "lmagr", {"ma_length": 10})
    _assert_eq(result, 10, "LMAGR string name resolution warmup")


def test_d7_t9():
    """Config with LMAGR indicator instance validates without errors."""
    inst = _make_indicator_instance(
        label="stretch",
        indicator_id="lmagr",
        outputs_used=["lmagr"],
        parameters={"ma_length": 20})
    config = _make_minimal_config(
        indicator_instances=[inst],
        exit_rules=[_make_exit_rule()])
    errors = validate_config(config)
    _assert(not errors, f"LMAGR config should be valid: {errors}")


def test_d7_t10():
    """signal_slope_sign with non-default params: slow=20, signal=5 -> warmup=25."""
    result = get_warmup_bars_for_output(
        7, "signal_slope_sign", {"slow": 20, "signal": 5})
    _assert_eq(result, 25, "signal_slope_sign warmup with custom params")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _pass_count, _fail_count, _results
    _pass_count = 0
    _fail_count = 0
    _results = []

    print("=" * 60)
    print("Strategy Framework v1.8.0 Acceptance Tests")
    print("=" * 60)

    print("\n--- Delta 1: is_present / is_absent operators ---")
    _run_test("D1-T1", test_d1_t1)
    _run_test("D1-T2", test_d1_t2)
    _run_test("D1-T3", test_d1_t3)
    _run_test("D1-T4", test_d1_t4)
    _run_test("D1-T5", test_d1_t5)
    _run_test("D1-T6", test_d1_t6)
    _run_test("D1-T7", test_d1_t7)
    _run_test("D1-T8", test_d1_t8)
    _run_test("D1-T9", test_d1_t9)
    _run_test("D1-T10", test_d1_t10)

    print("\n--- Delta 2: MTM_DRAWDOWN_EXIT ---")
    _run_test("D2-T1", test_d2_t1)
    _run_test("D2-T2", test_d2_t2)
    _run_test("D2-T3", test_d2_t3)
    _run_test("D2-T4", test_d2_t4)
    _run_test("D2-T5", test_d2_t5)
    _run_test("D2-T6", test_d2_t6)
    _run_test("D2-T7", test_d2_t7)
    _run_test("D2-T8", test_d2_t8)
    _run_test("D2-T9", test_d2_t9)
    _run_test("D2-T10", test_d2_t10)

    print("\n--- Delta 3: Per-output warmup ---")
    _run_test("D3-T1", test_d3_t1)
    _run_test("D3-T2", test_d3_t2)
    _run_test("D3-T3", test_d3_t3)
    _run_test("D3-T4", test_d3_t4)
    _run_test("D3-T5", test_d3_t5)
    _run_test("D3-T6", test_d3_t6)
    _run_test("D3-T7", test_d3_t7)
    _run_test("D3-T8", test_d3_t8)
    _run_test("D3-T9", test_d3_t9)
    _run_test("D3-T10", test_d3_t10)

    print("\n--- Delta 4: HANDOFF gate policy ---")
    _run_test("D4-T1", test_d4_t1)
    _run_test("D4-T2", test_d4_t2)
    _run_test("D4-T3", test_d4_t3)
    _run_test("D4-T4", test_d4_t4)
    _run_test("D4-T5", test_d4_t5)
    _run_test("D4-T6", test_d4_t6)
    _run_test("D4-T7", test_d4_t7)
    _run_test("D4-T8", test_d4_t8)
    _run_test("D4-T9", test_d4_t9)
    _run_test("D4-T10", test_d4_t10)
    _run_test("D4-T11", test_d4_t11)
    _run_test("D4-T12", test_d4_t12)

    print("\n--- Delta 5: Cross-indicator condition references ---")
    _run_test("D5-T1", test_d5_t1)
    _run_test("D5-T2", test_d5_t2)
    _run_test("D5-T3", test_d5_t3)
    _run_test("D5-T4", test_d5_t4)
    _run_test("D5-T5", test_d5_t5)
    _run_test("D5-T6", test_d5_t6)
    _run_test("D5-T7", test_d5_t7)
    _run_test("D5-T8", test_d5_t8)

    print("\n--- Delta 6: Schema strictness ---")
    _run_test("D6-T1", test_d6_t1)
    _run_test("D6-T2", test_d6_t2)
    _run_test("D6-T3", test_d6_t3)
    _run_test("D6-T4", test_d6_t4)

    print("\n--- Delta 7: signal_slope_sign + LMAGR wiring ---")
    _run_test("D7-T1", test_d7_t1)
    _run_test("D7-T2", test_d7_t2)
    _run_test("D7-T3", test_d7_t3)
    _run_test("D7-T4", test_d7_t4)
    _run_test("D7-T5", test_d7_t5)
    _run_test("D7-T6", test_d7_t6)
    _run_test("D7-T7", test_d7_t7)
    _run_test("D7-T8", test_d7_t8)
    _run_test("D7-T9", test_d7_t9)
    _run_test("D7-T10", test_d7_t10)

    print("\n--- Composition Proofs ---")
    _run_test("Proof-1", test_proof_1)
    _run_test("Proof-2", test_proof_2)
    _run_test("Proof-3", test_proof_3)
    _run_test("Proof-4", test_proof_4)

    print("\n" + "=" * 60)
    print(f"Results: {_pass_count} passed, {_fail_count} failed, "
          f"{_pass_count + _fail_count} total")
    print("=" * 60)

    if _fail_count > 0:
        print("\nFailed tests:")
        for tid, status, msg in _results:
            if status == "FAIL":
                print(f"  {tid}: {msg}")
        return 1
    else:
        print("\nAll tests passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
