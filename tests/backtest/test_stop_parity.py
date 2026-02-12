"""Gate 5: Stop-loss and trailing-stop parity — close-only evaluation.

Tests:
  - Stop loss triggers at correct bar
  - Trailing stop triggers at correct bar
  - Intrabar breach without close trigger does NOT fire (close-only)
  - Time limit exits at correct bar
"""

import math
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ui.services.backtest_runner import Bar, run_backtest


def _make_stop_loss_bars():
    """Create a price path that:
    1. Rises above entry threshold
    2. Falls below stop loss at a known bar

    Entry at price > 100, stop at 2% (price < 98).
    """
    bars = []
    start_ts = 1700000000

    prices = (
        [95] * 5 +           # Below entry threshold
        [102] * 1 +          # Entry triggers here (bar 5)
        [103, 104, 105] +    # Price rises
        [101, 100, 99] +     # Declining but above stop
        [97] +               # Bar 12: close below 98 (stop loss trigger)
        [96, 95, 94]         # Continues falling
    )

    for i, price in enumerate(prices):
        bars.append(Bar(
            ts=start_ts + i * 60,
            o=price, h=price + 1, l=price - 1, c=price,
            v=100.0, index=i,
        ))
    return bars


def _make_stop_loss_config():
    """Config: enter when price > 100, stop loss at 2% (200 bps)."""
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
                    {"indicator": "price_1m", "output": "ema", "operator": ">", "value": 100},
                ],
                "condition_groups": [],
            },
        ],
        "exit_rules": [
            {
                "name": "Stop Loss",
                "type": "STOP_LOSS",
                "applies_to": ["LONG"],
                "evaluation_cadence": "1m",
                "conditions": [],
                "parameters": {
                    "mode": "FIXED_PERCENT",
                    "percent_long": "0.02",
                    "percent_short": "0.02",
                },
            },
        ],
        "gate_rules": [],
        "execution_params": {"flip_enabled": False},
        "archetype_tags": [],
    }


def test_stop_loss_triggers_at_correct_bar():
    """Stop loss fires when close breaches stop price."""
    bars = _make_stop_loss_bars()
    config = _make_stop_loss_config()
    trades, _, _ = run_backtest(config, bars, "sl_test")

    assert len(trades) >= 1, f"Expected at least 1 trade, got {len(trades)}"
    t = trades[0]
    assert t.side == "long"
    # Entry at bar 5 (first bar where price > 100)
    assert t.entry_idx == 5
    # Exit at bar 12 (first bar where close < 98 = 100*(1-0.02))
    # Entry price is 102, so stop = 102 * 0.98 = 99.96
    # Bar 11 close=99 — below 99.96 → triggers
    assert t.exit_idx == 11, f"Expected exit at bar 11, got {t.exit_idx}"
    assert t.gross_return_bps < 0  # Loss


def _make_intrabar_breach_bars():
    """Price path where low breaches stop but close recovers.

    This tests close-only evaluation: stop should NOT fire.
    """
    bars = []
    start_ts = 1700000000

    prices_close = (
        [95] * 5 +    # Below entry
        [102] +        # Entry (bar 5)
        [103, 104, 105, 104, 103, 102, 101, 100, 99, 98, 97]  # Declining
    )
    # Bar 10: low dips below stop, but close is above
    prices_low = [p - 1 for p in prices_close]
    prices_low[8] = 95  # Low at bar 8 dips way below stop (entry=102, stop~99.96)
    prices_close[8] = 101  # But close recovers above stop

    for i in range(len(prices_close)):
        bars.append(Bar(
            ts=start_ts + i * 60,
            o=prices_close[i],
            h=prices_close[i] + 2,
            l=prices_low[i],
            c=prices_close[i],
            v=100.0,
            index=i,
        ))
    return bars


def test_close_only_evaluation():
    """Intrabar low breach does NOT trigger stop — close-only evaluation."""
    bars = _make_intrabar_breach_bars()
    config = _make_stop_loss_config()
    trades, _, _ = run_backtest(config, bars, "close_only_test")

    assert len(trades) >= 1
    t = trades[0]
    # Stop should NOT fire at bar 8 (low breaches but close=101 is above stop~99.96)
    # Stop should fire at a later bar where close actually breaches
    assert t.exit_idx != 8, "Stop fired on intrabar breach — should be close-only!"


def _make_trailing_stop_bars():
    """Price rises after entry, then retraces to trigger trailing stop.

    Entry at bar 5 (price=102), rises to 110, then trails back.
    Trailing stop at 3% (300 bps): peak=110, stop = 110*0.97 = 106.7
    """
    bars = []
    start_ts = 1700000000

    prices = (
        [95] * 5 +                    # Below entry
        [102] +                        # Entry (bar 5)
        [104, 106, 108, 110] +        # Rising (peak at bar 9: 110)
        [109, 108, 107, 106] +        # Declining, trail stop = 110*0.97=106.7
        # Bar 13: 106 < 106.7 → trailing stop fires
        [105, 104]                     # Continues falling
    )

    for i, price in enumerate(prices):
        bars.append(Bar(
            ts=start_ts + i * 60,
            o=price, h=price + 1, l=price - 1, c=price,
            v=100.0, index=i,
        ))
    return bars


def _make_trailing_stop_config():
    """Config with trailing stop at 3% (300 bps)."""
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
                    {"indicator": "price_1m", "output": "ema", "operator": ">", "value": 100},
                ],
                "condition_groups": [],
            },
        ],
        "exit_rules": [
            {
                "name": "Trailing Stop",
                "type": "TRAILING_STOP",
                "applies_to": ["LONG"],
                "evaluation_cadence": "1m",
                "conditions": [],
                "parameters": {
                    "mode": "FIXED_PERCENT",
                    "percent_long": "0.03",
                    "percent_short": "0.03",
                },
            },
        ],
        "gate_rules": [],
        "execution_params": {"flip_enabled": False},
        "archetype_tags": [],
    }


def test_trailing_stop_triggers():
    """Trailing stop fires after peak, when price retraces by stop %."""
    bars = _make_trailing_stop_bars()
    config = _make_trailing_stop_config()
    trades, _, _ = run_backtest(config, bars, "trail_test")

    assert len(trades) >= 1, f"Expected at least 1 trade, got {len(trades)}"
    t = trades[0]
    assert t.side == "long"
    assert t.entry_idx == 5
    # Peak at bar 9 (110), trail stop = 110 * 0.97 = 106.7
    # Bar 13 close=106 < 106.7 → trail stop fires
    assert t.exit_idx == 13, f"Expected trailing stop at bar 13, got {t.exit_idx}"
    assert t.gross_return_bps > 0  # Still profitable (entry 102, exit 106)


def _make_time_limit_bars():
    """Price stays flat to test time limit exit."""
    bars = []
    start_ts = 1700000000

    for i in range(100):
        if i < 5:
            price = 95  # Below entry
        else:
            price = 102  # Above entry, stays there
        bars.append(Bar(
            ts=start_ts + i * 60,
            o=price, h=price + 1, l=price - 1, c=price,
            v=100.0, index=i,
        ))
    return bars


def _make_time_limit_config(limit_bars=20):
    """Config with time limit."""
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
                    {"indicator": "price_1m", "output": "ema", "operator": ">", "value": 100},
                ],
                "condition_groups": [],
            },
        ],
        "exit_rules": [
            {
                "name": "Time Limit",
                "type": "TIME_LIMIT",
                "applies_to": ["LONG"],
                "evaluation_cadence": "1m",
                "conditions": [],
                "parameters": {
                    "time_limit_bars": limit_bars,
                    "time_limit_reference_cadence": "1m",
                },
            },
        ],
        "gate_rules": [],
        "execution_params": {"flip_enabled": False},
        "archetype_tags": [],
    }


def test_time_limit_exit():
    """Time limit fires after N evaluation bars."""
    bars = _make_time_limit_bars()
    config = _make_time_limit_config(limit_bars=20)
    trades, _, _ = run_backtest(config, bars, "time_test")

    assert len(trades) >= 1, f"Expected at least 1 trade, got {len(trades)}"
    t = trades[0]
    assert t.side == "long"
    assert t.entry_idx == 5  # First bar above 100
    # Time limit of 20 bars from entry → exit at bar 25
    assert t.exit_idx == 25, f"Expected time limit exit at bar 25, got {t.exit_idx}"
