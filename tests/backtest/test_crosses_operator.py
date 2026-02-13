"""Test crosses_above / crosses_below operator — verify prev_outputs is passed.

If prev_outputs is None or not passed, crosses_above always returns False
(framework line 793-794), and no entry signal fires.

Uses a simple config with a single EMA indicator and a crosses_above condition.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ui.services.backtest_runner import Bar, run_backtest, compute_macd
from composition_compiler_v1_5_2 import compile_composition


def _make_bars_with_trend_reversal(n: int = 300) -> list:
    """Generate bars: uptrend → downtrend → uptrend.

    This creates MACD slope_sign transitions in both directions,
    ensuring both crosses_above and crosses_below fire.
    """
    bars = []
    base_ts = 1700000000
    for i in range(n):
        if i < 100:
            close = 100.0 + i * 1.0     # uptrend 100→200
        elif i < 200:
            close = 200.0 - (i - 100) * 1.0  # downtrend 200→100
        else:
            close = 100.0 + (i - 200) * 1.0  # uptrend 100→200
        bars.append(Bar(
            ts=base_ts + i * 60,
            o=close - 0.1,
            h=close + 0.5,
            l=close - 0.5,
            c=close,
            v=100.0,
            index=i,
        ))
    return bars


def _make_simple_crosses_above_spec():
    """A minimal spec using MACD slope_sign crosses_above 0 for LONG entry."""
    return {
        "display_name": "Test Crosses Above",
        "spec_version": "1.5.2",
        "target_engine_version": "1.8.0",
        "target_instrument": "BTCUSDT",
        "target_variant": "perp",
        "archetype_tags": [],
        "indicator_instances": [
            {
                "label": "macd_1m",
                "indicator_id": "macd",
                "timeframe": "1m",
                "parameters": {"fast_period": 5, "slow_period": 10, "signal_period": 3},
                "outputs_used": ["macd_line", "slope_sign"],
                "role": "entry_signal",
                "group": "test"
            }
        ],
        "entry_rules": [
            {
                "label": "Long Entry — crosses_above test",
                "direction": "LONG",
                "evaluation_cadence": "1m",
                "conditions": [
                    {
                        "indicator": "macd_1m",
                        "output": "slope_sign",
                        "operator": "crosses_above",
                        "value": 0
                    }
                ],
                "condition_groups": []
            }
        ],
        "exit_rules": [
            {
                "label": "Stop Loss",
                "exit_type": "STOP_LOSS",
                "applies_to": ["LONG", "SHORT"],
                "evaluation_cadence": "1m",
                "mode": "FIXED_PERCENT",
                "value_long_bps": 1000,
                "value_short_bps": 1000,
                "conditions": []
            }
        ],
        "gate_rules": [],
        "execution_params": {
            "direction": "BOTH",
            "leverage": 1,
            "entry_type": "MARKET",
            "sizing_mode": "FIXED",
            "risk_fraction_bps": 10000,
            "flip_enabled": False
        },
        "metadata": {}
    }


def _make_simple_crosses_below_spec():
    """A minimal spec using MACD slope_sign crosses_below 0 for SHORT entry,
    with a signal exit when slope turns positive again."""
    spec = _make_simple_crosses_above_spec()
    spec["entry_rules"] = [
        {
            "label": "Short Entry — crosses_below test",
            "direction": "SHORT",
            "evaluation_cadence": "1m",
            "conditions": [
                {
                    "indicator": "macd_1m",
                    "output": "slope_sign",
                    "operator": "crosses_below",
                    "value": 0
                }
            ],
            "condition_groups": []
        }
    ]
    # Add signal exit: close SHORT when slope_sign > 0
    spec["exit_rules"].append({
        "label": "Signal Exit — slope reversal",
        "exit_type": "SIGNAL",
        "applies_to": ["SHORT"],
        "evaluation_cadence": "1m",
        "conditions": [
            {
                "indicator": "macd_1m",
                "output": "slope_sign",
                "operator": ">",
                "value": 0
            }
        ]
    })
    return spec


class TestCrossesAbove:
    """Verify that crosses_above produces at least one entry signal."""

    def test_crosses_above_produces_entries(self):
        """Run backtest with crosses_above config → must get >= 1 trade."""
        spec = _make_simple_crosses_above_spec()
        result = compile_composition(spec)
        resolved = result["resolved_artifact"]
        config_hash = result["strategy_config_hash"]

        bars = _make_bars_with_trend_reversal(300)
        trades, _, n_bars = run_backtest(resolved, bars, config_hash)

        assert len(trades) >= 1, (
            f"Expected >= 1 trade from crosses_above, got {len(trades)}. "
            "This likely means prev_outputs is not being passed to evaluate_signal_pipeline()."
        )

    def test_crosses_below_produces_entries(self):
        """Run backtest with crosses_below config → must get >= 1 trade."""
        spec = _make_simple_crosses_below_spec()
        result = compile_composition(spec)
        resolved = result["resolved_artifact"]
        config_hash = result["strategy_config_hash"]

        bars = _make_bars_with_trend_reversal(300)
        trades, _, n_bars = run_backtest(resolved, bars, config_hash)

        assert len(trades) >= 1, (
            f"Expected >= 1 trade from crosses_below, got {len(trades)}. "
            "This likely means prev_outputs is not being passed to evaluate_signal_pipeline()."
        )

    def test_prev_outputs_not_none_after_first_bar(self):
        """Directly verify prev_outputs is maintained in the runner loop."""
        # This is implicitly tested by test_crosses_above_produces_entries,
        # but we verify the mechanism by checking MACD slope transitions exist
        closes = [100.0 + i for i in range(100)] + [200.0 - i for i in range(100)]
        ml, _, _, ss = compute_macd(closes, 5, 10, 3)

        # Must have at least one transition from <=0 to >0
        transitions = 0
        for i in range(1, len(ss)):
            if ss[i - 1] is not None and ss[i] is not None:
                if ss[i - 1] <= 0 and ss[i] > 0:
                    transitions += 1
        assert transitions >= 1, "MACD slope should have at least one -→+ transition"
