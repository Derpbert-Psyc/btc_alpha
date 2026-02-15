"""Tests for per-exit-rule cadence and stop-loss fill semantics.

Tests:
  SC-1: Stop fires between entry cadence bars (5m entry, 1m stop)
  SC-2: Stop fills at stop price, not bar.close
  SC-3: Trailing stop fills at trail price
  SC-4: Signal exit still evaluates only at entry cadence
  SC-5: TIME_LIMIT counts eval bars, not 1m bars
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.services.backtest_runner import Bar, run_backtest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(prices, ts_start=1700006000, tf_sec=60):
    """Create bars from (close, high, low) tuples or float prices.

    ts_start is chosen to align with 5m boundaries (divisible by 300).
    """
    bars = []
    for i, p in enumerate(prices):
        if isinstance(p, (tuple, list)):
            c, h, l = p
        else:
            c, h, l = p, p * 1.001, p * 0.999
        bars.append(Bar(
            ts=ts_start + i * tf_sec,
            o=c, h=h, l=l, c=c, v=100.0,
            index=i,
        ))
    return bars


def _config_5m_entry_1m_stop(stop_pct=0.02):
    """Config: 5m entry cadence, 1m stop-loss (FIXED_PERCENT at stop_pct).

    Uses EMA > 0 as always-true entry condition.
    """
    return {
        "indicator_instances": [
            {
                "label": "ema_5m",
                "indicator_id": 1,
                "timeframe": "5m",
                "parameters": {"period": 3},
                "outputs_used": ["ema"],
            },
        ],
        "entry_rules": [
            {
                "name": "long_entry",
                "direction": "LONG",
                "evaluation_cadence": "5m",
                "conditions": [
                    {"indicator": "ema_5m", "output": "ema",
                     "operator": ">", "value": "0"},
                ],
                "condition_groups": [],
            },
        ],
        "exit_rules": [
            {
                "type": "STOP_LOSS",
                "evaluation_cadence": "1m",
                "applies_to": ["LONG"],
                "parameters": {
                    "mode": "FIXED_PERCENT",
                    "percent_long": str(stop_pct),
                    "percent_short": "0",
                },
            },
        ],
        "gate_rules": [],
        "execution_params": {
            "direction": "LONG",
            "leverage": 1.0,
            "flip_enabled": False,
            "post_exit_cooldown_bars": 0,
            "stop_loss": None,
            "position_sizing": None,
            "entry_type": "MARKET",
        },
    }


def _config_1m_entry_with_stop(stop_pct=0.02):
    """Config: 1m entry cadence, 1m stop-loss. For testing fill price semantics."""
    return {
        "indicator_instances": [
            {
                "label": "ema_1m",
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
                    {"indicator": "ema_1m", "output": "ema",
                     "operator": ">", "value": "0"},
                ],
                "condition_groups": [],
            },
        ],
        "exit_rules": [
            {
                "type": "STOP_LOSS",
                "evaluation_cadence": "1m",
                "applies_to": ["LONG"],
                "parameters": {
                    "mode": "FIXED_PERCENT",
                    "percent_long": str(stop_pct),
                    "percent_short": "0",
                },
            },
        ],
        "gate_rules": [],
        "execution_params": {
            "direction": "LONG",
            "leverage": 1.0,
            "flip_enabled": False,
            "post_exit_cooldown_bars": 0,
            "stop_loss": None,
            "position_sizing": None,
            "entry_type": "MARKET",
        },
    }


# ---------------------------------------------------------------------------
# SC-1: Stop fires between entry cadence bars
# ---------------------------------------------------------------------------

class TestStopFiresBetweenCadenceBars:
    def test_stop_fires_at_minute_2_of_5m_bar(self):
        """With 5m entry cadence, a 1m stop should fire at minute 2 when
        price crashes — not wait until minute 5."""
        # 15 bars warmup (period=3 on 5m = 3*5=15 bars)
        # Then entry at bar 15 (5m boundary), price crash at bar 17 (minute 2)
        prices = []
        # Bars 0-14: stable at $100 (warmup)
        for _ in range(15):
            prices.append((100.0, 100.5, 99.5))
        # Bar 15: 5m boundary — entry at $100 (close=100, high=100.5, low=99.5)
        prices.append((100.0, 100.5, 99.5))
        # Bar 16: still OK (close=99.5, low=99.0 > stop at 98.0)
        prices.append((99.5, 100.0, 99.0))
        # Bar 17: crash! low=$97 breaches 2% stop at $98
        prices.append((97.5, 99.0, 97.0))
        # Bar 18: further down (should NOT be reached — stop already fired)
        prices.append((96.0, 97.0, 95.0))
        # Bar 19: more bars to fill out 5m bucket
        prices.append((95.0, 96.0, 94.0))

        bars = _make_bars(prices)
        config = _config_5m_entry_1m_stop(stop_pct=0.02)  # 2% stop → $98
        trades, _, _ = run_backtest(config, bars, strategy_hash="sc1")

        assert len(trades) >= 1, f"Expected at least 1 trade, got {len(trades)}"
        # The stop should fire at bar 17 (index 17), not bar 19 (next 5m boundary)
        stopped = trades[0]
        assert stopped.exit_idx == 17, (
            f"Stop should fire at bar 17 (minute 2), not bar {stopped.exit_idx}")


# ---------------------------------------------------------------------------
# SC-2: Stop fills at stop price, not bar.close
# ---------------------------------------------------------------------------

class TestStopFillAtStopPrice:
    def test_long_stop_fill_at_stop_price(self):
        """LONG stop at $98 should fill at $98, not bar.close=$97."""
        prices = []
        # 5 bars warmup (EMA period=5 at 1m)
        for _ in range(5):
            prices.append((100.0, 100.5, 99.5))
        # Bar 5: entry at $100
        prices.append((100.0, 100.5, 99.5))
        # Bar 6: crash — low=$96 breaches 2% stop at $98, close=$97
        prices.append((97.0, 99.5, 96.0))
        # Extra bars
        prices.append((97.0, 98.0, 96.0))

        bars = _make_bars(prices)
        config = _config_1m_entry_with_stop(stop_pct=0.02)  # 2% stop → $98
        trades, _, _ = run_backtest(config, bars, strategy_hash="sc2")

        assert len(trades) >= 1, f"Expected at least 1 trade, got {len(trades)}"
        trade = trades[0]
        # Exit price should be the stop price ($98), not bar.close ($97)
        exit_price_float = trade.exit_price.value / 100.0  # Fixed PRICE → float
        assert abs(exit_price_float - 98.0) < 0.01, (
            f"Exit price should be ~$98 (stop price), got ${exit_price_float:.2f}")

    def test_short_stop_fill_at_stop_price(self):
        """SHORT stop at $102 should fill at $102, not bar.close=$103."""
        prices = []
        for _ in range(5):
            prices.append((100.0, 100.5, 99.5))
        # Bar 5: entry SHORT at $100
        prices.append((100.0, 100.5, 99.5))
        # Bar 6: spike — high=$104 breaches 2% stop at $102, close=$103
        prices.append((103.0, 104.0, 100.5))
        prices.append((103.0, 104.0, 102.0))

        bars = _make_bars(prices)
        config = _config_1m_entry_with_stop(stop_pct=0.02)
        config["entry_rules"][0]["direction"] = "SHORT"
        config["execution_params"]["direction"] = "SHORT"
        config["exit_rules"][0]["applies_to"] = ["SHORT"]
        config["exit_rules"][0]["parameters"]["percent_short"] = "0.02"
        trades, _, _ = run_backtest(config, bars, strategy_hash="sc2s")

        assert len(trades) >= 1, f"Expected at least 1 trade, got {len(trades)}"
        trade = trades[0]
        exit_price_float = trade.exit_price.value / 100.0
        assert abs(exit_price_float - 102.0) < 0.01, (
            f"Exit price should be ~$102 (stop price), got ${exit_price_float:.2f}")


# ---------------------------------------------------------------------------
# SC-3: Trailing stop fills at trail price
# ---------------------------------------------------------------------------

class TestTrailingStopFill:
    def test_trailing_stop_fill_at_trail_price(self):
        """Trailing stop should fill at computed trail price, not bar.close."""
        prices = []
        # 15 bars warmup
        for _ in range(15):
            prices.append((100.0, 100.5, 99.5))
        # Bar 15: entry at $100
        prices.append((100.0, 100.5, 99.5))
        # Bars 16-19: price rises to $105 (trailing peak tracks highs)
        prices.append((102.0, 103.0, 101.0))
        prices.append((104.0, 105.0, 103.0))
        prices.append((104.5, 105.0, 103.5))
        prices.append((104.0, 105.0, 103.0))
        # Bar 20 (next 5m boundary): still at $104, no trail trigger yet
        # trail_stop = 105.0 * 0.97 = 101.85 — low=103 > 101.85
        prices.append((104.0, 104.5, 103.0))
        # Bar 21: drop — low=$101 < trail_stop=101.85, close=$101.5
        prices.append((101.5, 103.0, 101.0))
        prices.append((101.0, 102.0, 100.0))
        prices.append((101.0, 102.0, 100.0))
        prices.append((101.0, 102.0, 100.0))

        bars = _make_bars(prices)
        config = _config_5m_entry_1m_stop(stop_pct=0.02)
        # Replace STOP_LOSS with TRAILING_STOP
        config["exit_rules"] = [
            {
                "type": "TRAILING_STOP",
                "evaluation_cadence": "1m",
                "applies_to": ["LONG"],
                "parameters": {
                    "percent_long": "0.03",  # 3% trail
                    "percent_short": "0",
                },
            },
        ]
        trades, _, _ = run_backtest(config, bars, strategy_hash="sc3")

        assert len(trades) >= 1
        trade = trades[0]
        exit_price_float = trade.exit_price.value / 100.0
        # Trail peak = 105.0 (bar 17-18 high), trail_stop = 105 * 0.97 = 101.85
        expected_trail = 105.0 * 0.97
        assert abs(exit_price_float - expected_trail) < 0.01, (
            f"Exit price should be ~${expected_trail:.2f} (trail price), "
            f"got ${exit_price_float:.2f}")


# ---------------------------------------------------------------------------
# SC-4: Signal exit still evaluates only at entry cadence
# ---------------------------------------------------------------------------

class TestSignalExitAtCadence:
    def test_signal_exit_only_at_5m_boundary(self):
        """A SIGNAL exit condition should only evaluate at 5m boundaries,
        not on intermediate 1m bars."""
        prices = []
        # 15 bars warmup
        for _ in range(15):
            prices.append((100.0, 100.5, 99.5))
        # Bar 15: entry at $100
        prices.append((100.0, 100.5, 99.5))
        # Bars 16-19: EMA still > 0 so no signal exit
        for _ in range(4):
            prices.append((100.0, 100.5, 99.5))
        # Bar 20 (5m boundary): still fine
        prices.append((100.0, 100.5, 99.5))
        # Bars 21-24: more bars
        for _ in range(4):
            prices.append((100.0, 100.5, 99.5))
        # Bar 25 (5m boundary)
        prices.append((100.0, 100.5, 99.5))

        bars = _make_bars(prices)
        config = _config_5m_entry_1m_stop(stop_pct=0.02)
        # Add a signal exit that triggers when ema < 999999 (always true)
        config["exit_rules"].append({
            "type": "SIGNAL",
            "evaluation_cadence": "5m",
            "applies_to": ["LONG"],
            "conditions": [
                {"indicator": "ema_5m", "output": "ema",
                 "operator": "<", "value": "999999"},
            ],
            "condition_groups": [],
        })
        trades, _, _ = run_backtest(config, bars, strategy_hash="sc4")

        # Signal exit should fire at bar 20 (next 5m boundary after entry)
        # NOT at bars 16-19
        if trades:
            trade = trades[0]
            # Exit should be at a 5m boundary (bar 20 or later)
            assert trade.exit_idx >= 20, (
                f"Signal exit should fire at 5m boundary (>=20), "
                f"got bar {trade.exit_idx}")


# ---------------------------------------------------------------------------
# SC-5: TIME_LIMIT counts eval bars, not 1m bars
# ---------------------------------------------------------------------------

class TestTimeLimitCountsEvalBars:
    def test_time_limit_counts_eval_bars(self):
        """TIME_LIMIT with time_limit_bars=2 on 5m cadence should wait
        2 eval bars (10 minutes), not 2 minutes."""
        prices = []
        # 15 bars warmup
        for _ in range(15):
            prices.append((100.0, 100.5, 99.5))
        # Bar 15: entry
        prices.append((100.0, 100.5, 99.5))
        # Bars 16-24: 9 more bars (fills out 5m buckets)
        for _ in range(9):
            prices.append((100.0, 100.5, 99.5))
        # Bar 25: 2nd eval bar after entry → TIME_LIMIT should fire
        prices.append((100.0, 100.5, 99.5))
        # Extra bars
        for _ in range(5):
            prices.append((100.0, 100.5, 99.5))

        bars = _make_bars(prices)
        config = _config_5m_entry_1m_stop(stop_pct=0.02)
        # Remove stop loss, add TIME_LIMIT
        config["exit_rules"] = [
            {
                "type": "TIME_LIMIT",
                "evaluation_cadence": "5m",
                "applies_to": ["LONG"],
                "parameters": {
                    "time_limit_bars": 2,
                },
            },
        ]
        trades, _, _ = run_backtest(config, bars, strategy_hash="sc5")

        assert len(trades) >= 1
        trade = trades[0]
        # Should exit at bar 25 (2 eval bars × 5 minutes each = 10 minutes)
        # Not at bar 17 (2 × 1m bars)
        assert trade.exit_idx >= 24, (
            f"TIME_LIMIT should count eval bars (exit at >=24), "
            f"got bar {trade.exit_idx}")
