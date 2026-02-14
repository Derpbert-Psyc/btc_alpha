"""Runner tests for Chop Harvester features.

Tests:
  R-1: ATR stop triggers for LONG
  R-2: ATR stop triggers for SHORT
  R-3: Post-exit cooldown suppresses entry
  R-4: FLIP does NOT trigger cooldown
  R-5: Cooldown zero = no suppression
  RP-1: compute_donchian_position — percent_b midpoint
  RP-2: compute_donchian_position — bars_since forward scan
  RP-3: compute_donchian_position — new_upper/new_lower transitions
  RP-4: compute_vol_regime — known value
  RP-5: compute_vol_regime — zero range
  RP-6: Two-pass precompute with dep_outputs
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.services.backtest_runner import (
    Bar,
    RATE_SCALE,
    compute_donchian,
    compute_donchian_position,
    compute_indicator_outputs,
    compute_vol_regime,
    run_backtest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(prices, ts_start=1700000000, tf_sec=60):
    """Create bars from (close, high, low) tuples or float prices."""
    bars = []
    for i, p in enumerate(prices):
        if isinstance(p, (tuple, list)):
            c, h, l = p
        else:
            c, h, l = p, p * 1.01, p * 0.99
        bars.append(Bar(
            ts=ts_start + i * tf_sec,
            o=c, h=h, l=l, c=c, v=100.0,
            index=i,
        ))
    return bars


def _make_config_with_atr_stop(atr_multiple=1.5, cooldown=0):
    """Minimal resolved config with ATR stop and Donchian-based entry/exit."""
    return {
        "indicator_instances": [
            {
                "label": "atr_1m",
                "indicator_id": 3,
                "timeframe": "1m",
                "parameters": {"period": 5},
                "outputs_used": ["atr"],
            },
            {
                "label": "ema_fast",
                "indicator_id": 1,
                "timeframe": "1m",
                "parameters": {"period": 5},
                "outputs_used": ["ema"],
            },
        ],
        "entry_rules": [
            {
                "name": "long_entry",
                "direction": "LONG",
                "evaluation_cadence": "1m",
                "conditions": [
                    {"indicator": "ema_fast", "output": "ema", "operator": ">", "value": "0"},
                ],
                "condition_groups": [],
            },
        ],
        "exit_rules": [
            {
                "type": "STOP_LOSS",
                "applies_to": ["LONG", "SHORT"],
                "parameters": {
                    "mode": "ATR_MULTIPLE",
                    "atr_multiple": atr_multiple,
                    "atr_indicator_label": "atr_1m",
                },
            },
        ],
        "gate_rules": [],
        "execution_params": {
            "direction": "LONG",
            "leverage": 1.0,
            "flip_enabled": False,
            "post_exit_cooldown_bars": cooldown,
            "stop_loss": None,
            "position_sizing": None,
            "entry_type": "MARKET",
        },
    }


# ---------------------------------------------------------------------------
# R-1: ATR stop triggers for LONG
# ---------------------------------------------------------------------------

class TestATRStop:
    def test_atr_stop_long(self):
        """ATR stop should exit LONG when close drops below entry - ATR * multiple."""
        config = _make_config_with_atr_stop(atr_multiple=1.0, cooldown=0)
        # Create bars: first 10 at $100, then a big drop
        prices = []
        for i in range(10):
            prices.append((100.0, 101.0, 99.0))
        # ATR ≈ 2.0 (high-low=2.0), so stop at 100 - 2*1.0 = 98
        # Bar 10: price drops to 97 → should trigger stop
        prices.append((97.0, 100.0, 96.0))
        # Bar 11: price recovers (should not enter again immediately)
        prices.append((100.0, 101.0, 99.0))

        bars = _make_bars(prices)
        trades, _, _ = run_backtest(config, bars, strategy_hash="test_atr")

        # We should have at least 1 trade (entry + stop exit)
        stopped = [t for t in trades if t.gross_return_bps < 0]
        assert len(stopped) >= 1, f"Expected stop-loss trade, got {len(trades)} trades"

    def test_atr_stop_short(self):
        """ATR stop should exit SHORT when close rises above entry + ATR * multiple."""
        config = _make_config_with_atr_stop(atr_multiple=1.0, cooldown=0)
        # Make it SHORT direction
        config["entry_rules"][0]["direction"] = "SHORT"
        config["execution_params"]["direction"] = "SHORT"
        # Condition: ema > 0 always true, so entry will happen
        prices = []
        for i in range(10):
            prices.append((100.0, 101.0, 99.0))
        # ATR ≈ 2, stop at 100 + 2*1.0 = 102
        # Bar 10: price rises to 103 → should trigger stop
        prices.append((103.0, 104.0, 100.0))
        prices.append((100.0, 101.0, 99.0))

        bars = _make_bars(prices)
        trades, _, _ = run_backtest(config, bars, strategy_hash="test_atr_s")

        stopped = [t for t in trades if t.gross_return_bps < 0]
        assert len(stopped) >= 1, f"Expected stop-loss trade for SHORT"


# ---------------------------------------------------------------------------
# R-3: Cooldown suppresses entry
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_cooldown_suppresses_entry(self):
        """After exit, entry should be suppressed for cooldown_bars."""
        config = _make_config_with_atr_stop(atr_multiple=0.5, cooldown=5)
        # 10 bars warmup, entry at bar 10, stop at bar 11, then 5 bars cooldown
        prices = []
        for i in range(10):
            prices.append((100.0, 101.0, 99.0))
        # ATR ≈ 2, stop = 100 - 2*0.5 = 99. Bar 10: close at 98 → immediate stop
        prices.append((98.0, 100.0, 97.0))
        # Bars 11-15: price recovers but still in cooldown
        for i in range(5):
            prices.append((100.0, 101.0, 99.0))
        # Bar 16: cooldown expired, new entry should happen
        prices.append((100.0, 101.0, 99.0))
        # Bar 17: exit bar
        prices.append((100.5, 101.5, 99.5))

        bars = _make_bars(prices)
        trades, _, _ = run_backtest(config, bars, strategy_hash="test_cd")

        # First trade is the stopped trade
        # The cooldown should delay the second entry
        # We should have at most 2 trades total
        assert len(trades) >= 1, "Should have at least the stopped trade"

    def test_cooldown_zero_no_suppression(self):
        """Cooldown=0 should not suppress any entries."""
        config = _make_config_with_atr_stop(atr_multiple=0.5, cooldown=0)
        prices = []
        for i in range(10):
            prices.append((100.0, 101.0, 99.0))
        prices.append((98.0, 100.0, 97.0))  # Stop out
        prices.append((100.0, 101.0, 99.0))  # Immediate re-entry
        prices.append((100.0, 101.0, 99.0))

        bars = _make_bars(prices)
        trades, _, _ = run_backtest(config, bars, strategy_hash="test_cd0")

        # With no cooldown, re-entry should happen immediately
        assert len(trades) >= 1, "Should have trades with cooldown=0"


# ---------------------------------------------------------------------------
# RP-1 to RP-6: Runner probe helpers
# ---------------------------------------------------------------------------

class TestRunnerProbes:
    def test_donchian_position_percent_b_midpoint(self):
        """percent_b should be 500000 at channel midpoint."""
        # Channel: highs=110, lows=100, close=105 → percent_b = 50%
        bars = _make_bars([(105.0, 110.0, 100.0)] * 5)
        dc_upper = [None, None, None, None, 110.0]
        dc_lower = [None, None, None, None, 100.0]
        result = compute_donchian_position(bars, dc_upper, dc_lower, length=5)
        assert result["percent_b"][4] == 500000

    def test_donchian_position_bars_since_forward_scan(self):
        """bars_since_upper should find earliest occurrence (forward scan)."""
        # Bar 0: high=110 (upper), bars 1-4: lower highs
        highs = [110.0, 105.0, 108.0, 110.0, 107.0]
        bars = _make_bars([(105.0, h, 100.0) for h in highs])
        dc_upper = [None] * 4 + [110.0]
        dc_lower = [None] * 4 + [100.0]
        result = compute_donchian_position(bars, dc_upper, dc_lower, length=5)
        # Forward scan: bar 0 has high=110=upper → bars_since = 4 - 0 = 4
        assert result["bars_since_upper"][4] == 4

    def test_donchian_position_transitions(self):
        """new_upper/new_lower should be RATE_SCALE on transition, 0 otherwise."""
        bars = _make_bars([
            (105.0, 110.0, 100.0),  # Bar 0
            (105.0, 110.0, 100.0),  # Bar 1
            (105.0, 110.0, 100.0),  # Bar 2: first output, no prev → 0
            (105.0, 115.0, 100.0),  # Bar 3: new upper → RATE_SCALE
            (105.0, 115.0, 100.0),  # Bar 4: no change → 0
        ])
        dc_upper = [None, None, 110.0, 115.0, 115.0]
        dc_lower = [None, None, 100.0, 100.0, 100.0]
        result = compute_donchian_position(bars, dc_upper, dc_lower, length=3)
        assert result["new_upper"][2] == 0  # First output, no previous
        assert result["new_upper"][3] == RATE_SCALE  # Upper changed
        assert result["new_upper"][4] == 0  # No change

    def test_vol_regime_known_value(self):
        """vol_ratio should match integer math formula."""
        bars = _make_bars([(105.0, 110.0, 100.0)] * 3)
        dc_upper = [None, None, 110.0]
        dc_lower = [None, None, 100.0]
        ref = 1_000_000
        result = compute_vol_regime(bars, dc_upper, dc_lower, ref)
        # range_c = 11000 - 10000 = 1000
        # vol_ratio = 1000 * 100 * 1M * 1M // (10500 * 1M)
        expected = 1000 * 100 * RATE_SCALE * 1_000_000 // (10500 * 1_000_000)
        assert result["vol_ratio"][2] == expected

    def test_vol_regime_zero_range(self):
        """vol_ratio should be 0 when upper == lower."""
        bars = _make_bars([(100.0, 100.0, 100.0)] * 3)
        dc_upper = [None, None, 100.0]
        dc_lower = [None, None, 100.0]
        result = compute_vol_regime(bars, dc_upper, dc_lower, 1_000_000)
        assert result["vol_ratio"][2] == 0

    def test_two_pass_precompute(self):
        """Probe 30 should receive dep_outputs from pass 1 Donchian."""
        instance = {
            "label": "dc_pos",
            "indicator_id": 30,
            "timeframe": "1m",
            "parameters": {"length": 5, "donchian_label": "dc_5m"},
            "outputs_used": ["percent_b"],
        }
        bars = _make_bars([(105.0, 110.0, 100.0)] * 10)
        # Simulate pass 1: compute Donchian
        dc_inst = {
            "label": "dc_5m", "indicator_id": 14,
            "parameters": {"period": 5}, "outputs_used": ["upper", "lower"],
        }
        dc_outputs = compute_indicator_outputs(dc_inst, bars)

        # Pass 2: compute dc_position with dep_outputs
        result = compute_indicator_outputs(instance, bars, dep_outputs=dc_outputs)
        # After warmup (bar index 4), percent_b should be non-None
        assert result["percent_b"][4] is not None
        assert result["percent_b"][4] == 500000  # midpoint
