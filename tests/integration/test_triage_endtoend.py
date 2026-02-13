"""Integration test: compile → backtest → triage end-to-end.

CRITICAL: This test calls run_triage_for_composition() — the same
service-layer function that the UI [Run Triage] button handler calls.
This ensures the test exercises the same code path as the UI.

Uses embedded synthetic data — no file dependencies outside the test.
"""

import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from composition_compiler_v1_5_2 import compile_composition
from ui.services.backtest_runner import Bar
from ui.services.research_services import run_triage_for_composition


def _make_synthetic_parquet(tmpdir: str, n_bars: int = 5000) -> str:
    """Generate synthetic 1m candle data as a parquet file."""
    import pandas as pd
    import math

    base_ts = 1700000000
    rows = []
    for i in range(n_bars):
        ts = base_ts + i * 60
        # Oscillating price to generate trades
        trend = 100.0 + 0.01 * i
        cycle = 5.0 * math.sin(i * 0.02)
        noise = 0.5 * math.sin(i * 0.13)
        close = trend + cycle + noise
        o = close - 0.1
        h = close + abs(noise) + 0.3
        l = close - abs(noise) - 0.3
        rows.append({
            "timestamp": ts,
            "open": o,
            "high": h,
            "low": l,
            "close": close,
            "volume": 100.0 + i % 50,
        })

    df = pd.DataFrame(rows)
    path = os.path.join(tmpdir, "synthetic_test_data.parquet")
    df.to_parquet(path, index=False)
    return path


def _make_simple_spec():
    """A simple spec with MACD + crosses_above that will produce trades."""
    return {
        "display_name": "Test Strategy",
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
                "label": "Long Entry",
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
                "value_long_bps": 200,
                "value_short_bps": 200,
                "conditions": []
            },
            {
                "label": "Signal Exit",
                "exit_type": "SIGNAL",
                "applies_to": ["LONG"],
                "evaluation_cadence": "1m",
                "conditions": [
                    {
                        "indicator": "macd_1m",
                        "output": "slope_sign",
                        "operator": "<",
                        "value": 0
                    }
                ]
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
        "metadata": {
            "triage_sensitive_params": [
                {"param": "macd_1m.fast_period", "default": 5, "min": 3, "max": 8},
                {"param": "macd_1m.slow_period", "default": 10, "min": 7, "max": 15},
                {"param": "macd_1m.signal_period", "default": 3, "min": 2, "max": 5},
            ]
        }
    }


@pytest.fixture
def tmp_research(tmp_path, monkeypatch):
    """Set up temporary research dir for the test."""
    research_dir = str(tmp_path / "research")
    os.makedirs(os.path.join(research_dir, "strategies"), exist_ok=True)
    os.makedirs(os.path.join(research_dir, "triage_results"), exist_ok=True)
    os.makedirs(os.path.join(research_dir, "promotions"), exist_ok=True)
    # Patch RESEARCH_DIR in the services
    monkeypatch.setattr("ui.services.research_services.RESEARCH_DIR", research_dir)
    return research_dir


class TestTriageEndToEnd:

    def test_triage_produces_result(self, tmp_path, tmp_research):
        """Compile → backtest → triage → assert TriageResult is non-None.

        Calls run_triage_for_composition() — the same function the UI button calls.
        """
        spec = _make_simple_spec()
        compilation = compile_composition(spec)
        resolved = compilation["resolved_artifact"]
        config_hash = compilation["strategy_config_hash"]

        # Write resolved artifact (required by triage_results save)
        from composition_compiler_v1_5_2 import write_artifacts
        write_artifacts(compilation, base_dir=os.path.join(tmp_research, "strategies"))

        # Generate synthetic data
        dataset_path = _make_synthetic_parquet(str(tmp_path))

        # Call the service function — same as UI
        result = run_triage_for_composition(
            resolved_config=resolved,
            strategy_config_hash=config_hash,
            dataset_path=dataset_path,
            spec=spec,
        )

        # Verify result
        assert result is not None
        assert result.bar_count > 0
        assert result.runtime_seconds > 0
        assert result.dataset_filename == "synthetic_test_data.parquet"

        if result.zero_trades:
            # If no trades, ensure warning message is populated
            assert len(result.zero_trade_message) > 0
        else:
            # If trades, verify triage v2 ran
            assert result.triage_v2 is not None
            assert result.triage_v2.tier in ("S", "A", "B", "C", "F")
            assert result.triage_v2.tier_action is not None
            assert result.saved_path != ""
            assert os.path.exists(result.saved_path)

            # Verify saved JSON has required fields
            with open(result.saved_path) as f:
                saved = json.load(f)
            assert "strategy_config_hash" in saved
            assert "runner_economics" in saved
            assert "triage_v2" in saved
            assert saved["trade_count"] == result.trade_count

    def test_triage_runner_economics_recorded(self, tmp_path, tmp_research):
        """Runner economics must be present in result and saved file."""
        spec = _make_simple_spec()
        compilation = compile_composition(spec)
        resolved = compilation["resolved_artifact"]
        config_hash = compilation["strategy_config_hash"]

        from composition_compiler_v1_5_2 import write_artifacts
        write_artifacts(compilation, base_dir=os.path.join(tmp_research, "strategies"))

        dataset_path = _make_synthetic_parquet(str(tmp_path))

        result = run_triage_for_composition(
            resolved_config=resolved,
            strategy_config_hash=config_hash,
            dataset_path=dataset_path,
            fee_rate=0.0006,
            slippage_bps=10,
            starting_capital=10000,
        )

        assert result.runner_economics["fee_rate"] == 0.0006
        assert result.runner_economics["slippage_bps"] == 10
        assert result.runner_economics["starting_capital"] == 10000
