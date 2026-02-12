"""Gate 4: Framework semantic equivalence — runner decisions match evaluate_signal_pipeline.

This is the single best anti-drift test. For each evaluation bar:
  1. Extract indicator_outputs from runner
  2. Call evaluate_signal_pipeline() directly with those same outputs
  3. Compare signal type and position state
"""

import math
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ui.services.backtest_runner import (
    Bar, resample_bars, compute_indicator_outputs,
    parse_timeframe_seconds, compute_instance_warmup,
    run_backtest, MTMDrawdownTracker,
)
from strategy_framework_v1_8_0 import (
    evaluate_signal_pipeline,
    INDICATOR_NAME_TO_ID,
    SignalResult,
)


def _make_synthetic_bars(n_bars=2880, base_price=50000.0):
    bars = []
    start_ts = 1700000000
    for i in range(n_bars):
        phase = 2 * math.pi * i / 240
        trend = i * 0.5
        noise = math.sin(i * 7.13) * 50 + math.cos(i * 3.79) * 30
        price = base_price + math.sin(phase) * 500 + trend + noise
        spread = abs(math.sin(i * 1.37)) * 20 + 5
        o = price + math.sin(i * 2.71) * spread * 0.3
        h = max(price, o) + spread
        l = min(price, o) - spread
        c = price
        bars.append(Bar(ts=start_ts + i * 60, o=o, h=h, l=l, c=c, v=1000.0, index=i))
    return bars


def _make_ema_config():
    """Minimal: one EMA(1) passthrough, trivial entry, stop loss."""
    return {
        "engine_version": "1.8.0",
        "indicator_instances": [
            {
                "label": "price_1m",
                "indicator_id": 1,
                "timeframe": "1m",
                "parameters": {"period": 1},
                "outputs_used": ["ema"],
                "warmup_bars": 1,
            },
        ],
        "entry_rules": [
            {
                "name": "Long Entry",
                "direction": "LONG",
                "evaluation_cadence": "1m",
                "conditions": [
                    {"indicator": "price_1m", "output": "ema", "operator": ">", "value": 50200},
                ],
                "condition_groups": [],
            },
        ],
        "exit_rules": [
            {
                "name": "Signal Exit",
                "type": "SIGNAL",
                "applies_to": ["LONG"],
                "evaluation_cadence": "1m",
                "conditions": [
                    {"indicator": "price_1m", "output": "ema", "operator": "<", "value": 49800},
                ],
                "parameters": {},
            },
        ],
        "gate_rules": [],
        "execution_params": {"flip_enabled": False},
        "archetype_tags": [],
    }


def _make_macd_config():
    """MACD-based config for parity testing."""
    return {
        "engine_version": "1.8.0",
        "indicator_instances": [
            {
                "label": "macd_15m",
                "indicator_id": 7,
                "timeframe": "15m",
                "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "outputs_used": ["macd_line", "signal_line", "histogram"],
                "warmup_bars": 35,
            },
        ],
        "entry_rules": [
            {
                "name": "Long Entry",
                "direction": "LONG",
                "evaluation_cadence": "15m",
                "conditions": [
                    {"indicator": "macd_15m", "output": "histogram", "operator": ">", "value": 0},
                ],
                "condition_groups": [],
            },
        ],
        "exit_rules": [
            {
                "name": "Signal Exit",
                "type": "SIGNAL",
                "applies_to": ["LONG"],
                "evaluation_cadence": "15m",
                "conditions": [
                    {"indicator": "macd_15m", "output": "histogram", "operator": "<", "value": 0},
                ],
                "parameters": {},
            },
        ],
        "gate_rules": [],
        "execution_params": {"flip_enabled": False},
        "archetype_tags": [],
    }


def _make_bollinger_config():
    """Bollinger-based config for parity testing."""
    return {
        "engine_version": "1.8.0",
        "indicator_instances": [
            {
                "label": "bb_15m",
                "indicator_id": 11,
                "timeframe": "15m",
                "parameters": {"period": 20, "num_std": 2.0},
                "outputs_used": ["percent_b"],
                "warmup_bars": 20,
            },
        ],
        "entry_rules": [
            {
                "name": "Long Entry",
                "direction": "LONG",
                "evaluation_cadence": "15m",
                "conditions": [
                    {"indicator": "bb_15m", "output": "percent_b", "operator": "<", "value": 0.1},
                ],
                "condition_groups": [],
            },
        ],
        "exit_rules": [
            {
                "name": "Signal Exit",
                "type": "SIGNAL",
                "applies_to": ["LONG"],
                "evaluation_cadence": "15m",
                "conditions": [
                    {"indicator": "bb_15m", "output": "percent_b", "operator": ">", "value": 0.5},
                ],
                "parameters": {},
            },
        ],
        "gate_rules": [],
        "execution_params": {"flip_enabled": False},
        "archetype_tags": [],
    }


def _make_atr_config():
    """ATR-based config — entry when ATR is high (volatile market)."""
    return {
        "engine_version": "1.8.0",
        "indicator_instances": [
            {
                "label": "atr_15m",
                "indicator_id": 3,
                "timeframe": "15m",
                "parameters": {"period": 14},
                "outputs_used": ["atr"],
                "warmup_bars": 14,
            },
            {
                "label": "price_15m",
                "indicator_id": 1,
                "timeframe": "15m",
                "parameters": {"period": 1},
                "outputs_used": ["ema"],
                "warmup_bars": 1,
            },
        ],
        "entry_rules": [
            {
                "name": "Long Entry",
                "direction": "LONG",
                "evaluation_cadence": "15m",
                "conditions": [
                    {"indicator": "atr_15m", "output": "atr", "operator": ">", "value": 30},
                ],
                "condition_groups": [],
            },
        ],
        "exit_rules": [
            {
                "name": "Signal Exit",
                "type": "SIGNAL",
                "applies_to": ["LONG"],
                "evaluation_cadence": "15m",
                "conditions": [
                    {"indicator": "atr_15m", "output": "atr", "operator": "<", "value": 20},
                ],
                "parameters": {},
            },
        ],
        "gate_rules": [],
        "execution_params": {"flip_enabled": False},
        "archetype_tags": [],
    }


def _verify_parity(config, bars, config_name):
    """Core parity verification: replay runner decisions and compare with framework."""
    instances = config.get("indicator_instances", [])
    eval_cadence_sec = 60
    for rule in config.get("entry_rules", []):
        cad_sec = parse_timeframe_seconds(rule.get("evaluation_cadence", "1m"))
        eval_cadence_sec = max(eval_cadence_sec, cad_sec)

    # Precompute same way runner does
    instance_bars = {}
    instance_outputs = {}
    instance_tf_sec = {}
    instance_ts_to_rsidx = {}
    for inst in instances:
        label = inst["label"]
        tf = inst.get("timeframe", "1m")
        tf_sec = parse_timeframe_seconds(tf)
        instance_tf_sec[label] = tf_sec
        resampled = resample_bars(bars, tf_sec)
        instance_bars[label] = resampled
        instance_outputs[label] = compute_indicator_outputs(inst, resampled)
        # Build ts → resampled index mapping
        ts_map = {}
        rs_idx = 0
        for bar_1m in bars:
            while (rs_idx + 1 < len(resampled) and
                   resampled[rs_idx + 1].ts <= bar_1m.ts):
                rs_idx += 1
            if rs_idx < len(resampled) and resampled[rs_idx].ts <= bar_1m.ts:
                ts_map[bar_1m.ts] = rs_idx
        instance_ts_to_rsidx[label] = ts_map

    max_warmup_1m = 0
    for inst in instances:
        label = inst["label"]
        ind_id = inst["indicator_id"]
        if isinstance(ind_id, str):
            ind_id = INDICATOR_NAME_TO_ID.get(ind_id, 0)
        outputs_used = inst.get("outputs_used", [])
        warmup_bars_val = inst.get("warmup_bars")
        if warmup_bars_val is None:
            warmup_bars_val = compute_instance_warmup(ind_id, outputs_used, inst.get("parameters", {}))
        tf_sec = instance_tf_sec[label]
        warmup_1m = warmup_bars_val * (tf_sec // 60)
        max_warmup_1m = max(max_warmup_1m, warmup_1m)

    # Run the backtest
    trades, _, _ = run_backtest(config, bars, "parity_test")

    # Replay: manually evaluate signal pipeline at each evaluation bar
    position = None
    mtm_tracker = MTMDrawdownTracker()
    prev_outputs = None
    runner_actions = []
    framework_actions = []

    for bar_idx, bar in enumerate(bars):
        if bar_idx < max_warmup_1m:
            continue

        next_ts = bar.ts + 60
        current_bucket = bar.ts // eval_cadence_sec
        next_bucket = next_ts // eval_cadence_sec
        if current_bucket == next_bucket and eval_cadence_sec > 60:
            continue

        # Build indicator outputs using precomputed ts→rsidx mapping
        indicator_outs = {}
        for inst in instances:
            label = inst["label"]
            outputs = instance_outputs[label]
            ts_map = instance_ts_to_rsidx[label]
            rs_idx = ts_map.get(bar.ts)

            inst_outs = {}
            if rs_idx is not None:
                for out_name, out_series in outputs.items():
                    if rs_idx < len(out_series):
                        inst_outs[out_name] = out_series[rs_idx]
                    else:
                        inst_outs[out_name] = None
            else:
                for out_name in outputs:
                    inst_outs[out_name] = None
            indicator_outs[label] = inst_outs

        pos_dict = None
        if position:
            pos_dict = {
                "direction": position["direction"],
                "entry_price": round(position["entry_price"] * 100),
            }

        # Call framework directly
        signal = evaluate_signal_pipeline(
            config=config,
            indicator_outputs=indicator_outs,
            prev_outputs=prev_outputs,
            position=pos_dict,
            mtm_tracker=mtm_tracker if position else None,
            current_price=round(bar.close * 100) if position else None,
        )

        framework_actions.append((bar_idx, signal.action, signal.direction))

        # Track position state same way runner does
        if signal.action == "ENTRY" and position is None:
            position = {"direction": signal.direction, "entry_price": bar.close, "entry_idx": bar_idx}
            mtm_tracker.open_position()
        elif signal.action == "EXIT" and position is not None:
            mtm_tracker.close_position()
            position = None
        elif signal.action == "FLIP" and position is not None:
            mtm_tracker.close_position()
            position = {"direction": signal.direction, "entry_price": bar.close, "entry_idx": bar_idx}
            mtm_tracker.open_position()

        prev_outputs = indicator_outs

    # Now run the backtest and get its trade sequence
    trades2, _, _ = run_backtest(config, bars, "parity_test")

    # The trade entry/exit indices from both runs should match
    runner_trade_points = []
    for t in trades2:
        runner_trade_points.append(("ENTRY", t.entry_idx, t.side.upper()))
        runner_trade_points.append(("EXIT", t.exit_idx))

    framework_trade_points = []
    for bar_idx, action, direction in framework_actions:
        if action in ("ENTRY", "FLIP"):
            framework_trade_points.append(("ENTRY", bar_idx, direction))
        if action in ("EXIT", "FLIP"):
            framework_trade_points.append(("EXIT", bar_idx))

    # Trim trailing unpaired entry from framework (open position at backtest end)
    # Runner only emits completed trades, so an open position at the end has no
    # EXIT trade point. If framework has one more ENTRY than runner, that's the
    # open position — trim it.
    if (len(framework_trade_points) == len(runner_trade_points) + 1 and
            framework_trade_points[-1][0] == "ENTRY"):
        framework_trade_points = framework_trade_points[:-1]

    # Compare trade sequences
    assert len(runner_trade_points) == len(framework_trade_points), (
        f"{config_name}: Trade point count mismatch: "
        f"runner={len(runner_trade_points)}, framework={len(framework_trade_points)}\n"
        f"Runner: {runner_trade_points[:10]}\n"
        f"Framework: {framework_trade_points[:10]}"
    )

    for i, (rtp, ftp) in enumerate(zip(runner_trade_points, framework_trade_points)):
        assert rtp == ftp, (
            f"{config_name}: Trade point {i} mismatch: runner={rtp}, framework={ftp}"
        )


def test_ema_parity():
    """EMA(1) passthrough — simplest case."""
    bars = _make_synthetic_bars()
    config = _make_ema_config()
    _verify_parity(config, bars, "EMA(1)")


def test_macd_parity():
    """MACD on 15m — multi-output indicator."""
    bars = _make_synthetic_bars()
    config = _make_macd_config()
    _verify_parity(config, bars, "MACD")


def test_bollinger_parity():
    """Bollinger percent_b — derived output."""
    bars = _make_synthetic_bars()
    config = _make_bollinger_config()
    _verify_parity(config, bars, "Bollinger")


def test_atr_parity():
    """ATR — volatility indicator."""
    bars = _make_synthetic_bars()
    config = _make_atr_config()
    _verify_parity(config, bars, "ATR")
