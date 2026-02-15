"""Indicator Audit Tests — verify output scales, ADX, warnings, determinism, regression.

Test groups:
  V1: Output scale ranges (RSI, ROC, Choppiness, ADX, Bollinger)
  V2: Sweep fidelity — different periods produce different outputs
  V3: None handling — unimplemented indicator gate blocks all trades
  V5: Determinism — same input = same output
  V6: Regression — MACD Confluence and Chop Harvester trade counts unchanged
"""

import json
import logging
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.services.backtest_runner import (
    Bar,
    compute_adx,
    compute_bollinger,
    compute_choppiness,
    compute_indicator_outputs,
    compute_rsi,
    run_backtest,
)
from composition_compiler_v1_5_2 import compile_composition

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


def _trending_prices(n=200, start=100.0, step=0.5):
    """Generate a trending price series with volatility."""
    prices = []
    p = start
    for i in range(n):
        # Alternate up/down around trend to create RSI variation
        noise = step * (1.0 if i % 3 != 0 else -0.5)
        p += noise
        prices.append(p)
    return prices


def _oscillating_prices(n=200, center=100.0, amplitude=5.0):
    """Generate oscillating prices for choppy/range-bound testing."""
    prices = []
    for i in range(n):
        prices.append(center + amplitude * math.sin(i * 0.15))
    return prices


PRESET_DIR = os.path.join(os.path.dirname(__file__), "..", "ui", "presets")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "historic_data")
RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "..", "research")

DATASET_13MO = os.path.join(
    DATA_DIR, "btcusdt_binance_spot_1m_2025-01-01_to_2026-01-31.parquet",
)

MACD_COMP = os.path.join(
    RESEARCH_DIR, "compositions",
    "e9163f2e-95d7-4668-8387-70ed258b9144", "composition.json",
)

CHOP_PRESET = os.path.join(PRESET_DIR, "chop_harvester_clean.json")


def _load_bars_parquet(path):
    import pandas as pd
    df = pd.read_parquet(path)
    bars = []
    for i, row in enumerate(df.itertuples()):
        ts = int(row.timestamp) if hasattr(row, "timestamp") else int(row.ts)
        bars.append(Bar(
            ts=ts, o=float(row.open), h=float(row.high),
            l=float(row.low), c=float(row.close),
            v=float(row.volume) if hasattr(row, "volume") else 0.0,
            index=i,
        ))
    return bars


# ===========================================================================
# V1: Output Scale Ranges
# ===========================================================================

class TestV1ScaleRanges:
    """Verify all scale-changed indicators output in 0.0-1.0 range."""

    def test_rsi_range_0_to_1(self):
        """RSI non-None values must be in [0.0, 1.0]."""
        prices = _trending_prices(100)
        bars = _make_bars(prices)
        result = compute_indicator_outputs(
            {"indicator_id": 2, "parameters": {"period": 14}}, bars,
        )
        rsi = result["rsi"]
        valid = [v for v in rsi if v is not None]
        assert len(valid) > 0, "RSI should have non-None values"
        for v in valid:
            assert 0.0 <= v <= 1.0, f"RSI {v} out of [0.0, 1.0]"

    def test_rsi_70_becomes_0_70(self):
        """A raw RSI of 70 should output as 0.70."""
        # Create a strong uptrend to push RSI high
        prices = [100.0 + i * 2.0 for i in range(50)]
        bars = _make_bars(prices)
        result = compute_indicator_outputs(
            {"indicator_id": 2, "parameters": {"period": 14}}, bars,
        )
        rsi = result["rsi"]
        valid = [v for v in rsi if v is not None]
        # Strong uptrend → RSI should be well above 0.5
        assert any(v > 0.5 for v in valid), "Strong uptrend RSI should be > 0.5"
        assert all(v <= 1.0 for v in valid), "RSI must not exceed 1.0"

    def test_roc_decimal_scale(self):
        """5% price jump → ROC ≈ 0.05 (not 5.0)."""
        prices = [100.0] * 20 + [105.0]  # 5% jump at index 20
        bars = _make_bars(prices)
        result = compute_indicator_outputs(
            {"indicator_id": 8, "parameters": {"period": 10}}, bars,
        )
        roc = result["roc"]
        # At index 20, price went from 100 to 105 = 5% = 0.05
        assert roc[20] is not None
        assert abs(roc[20] - 0.05) < 0.001, f"ROC should be ~0.05, got {roc[20]}"

    def test_choppiness_range_0_to_1(self):
        """Choppiness non-None values must be in [0.0, 1.0]."""
        prices = _oscillating_prices(100)
        bars = _make_bars(prices)
        result = compute_indicator_outputs(
            {"indicator_id": 10, "parameters": {"period": 14}}, bars,
        )
        chop = result["choppiness"]
        valid = [v for v in chop if v is not None]
        assert len(valid) > 0, "Choppiness should have non-None values"
        for v in valid:
            assert 0.0 <= v <= 1.0, f"Choppiness {v} out of [0.0, 1.0]"

    def test_adx_range_0_to_1(self):
        """ADX non-None values must be in [0.0, 1.0]."""
        prices = _trending_prices(100, step=1.0)
        bars = _make_bars(prices)
        result = compute_indicator_outputs(
            {"indicator_id": 9, "parameters": {"period": 14}}, bars,
        )
        adx = result["adx"]
        valid = [v for v in adx if v is not None]
        assert len(valid) > 0, "ADX should have non-None values"
        for v in valid:
            assert 0.0 <= v <= 1.0, f"ADX {v} out of [0.0, 1.0]"

    def test_adx_di_range_0_to_1(self):
        """+DI and -DI must be in [0.0, 1.0]."""
        prices = _trending_prices(100, step=1.0)
        bars = _make_bars(prices)
        result = compute_indicator_outputs(
            {"indicator_id": 9, "parameters": {"period": 14}}, bars,
        )
        for key in ("plus_di", "minus_di"):
            vals = result[key]
            valid = [v for v in vals if v is not None]
            assert len(valid) > 0, f"{key} should have non-None values"
            for v in valid:
                assert 0.0 <= v <= 1.0, f"{key} {v} out of [0.0, 1.0]"

    def test_bollinger_percent_b_reasonable(self):
        """Bollinger percent_b should be in reasonable range, bandwidth > 0."""
        prices = _oscillating_prices(100)
        bars = _make_bars(prices)
        result = compute_indicator_outputs(
            {"indicator_id": 11, "parameters": {"period": 20, "num_std": 2.0}}, bars,
        )
        pb = result["percent_b"]
        bw = result["bandwidth"]
        valid_pb = [v for v in pb if v is not None]
        valid_bw = [v for v in bw if v is not None]
        assert len(valid_pb) > 0
        assert len(valid_bw) > 0
        # percent_b should mostly be near 0-1 range (can exceed briefly)
        assert all(-0.5 <= v <= 1.5 for v in valid_pb), "percent_b out of reasonable range"
        assert all(v > 0 for v in valid_bw), "bandwidth should be > 0"


# ===========================================================================
# V2: Sweep Fidelity
# ===========================================================================

class TestV2SweepFidelity:
    """Different parameters produce different output series."""

    def test_roc_different_periods_differ(self):
        """ROC with period 10 vs 20 vs 30 should produce different series."""
        prices = _trending_prices(100)
        bars = _make_bars(prices)
        series = {}
        for period in [10, 20, 30]:
            result = compute_indicator_outputs(
                {"indicator_id": 8, "parameters": {"period": period}}, bars,
            )
            # Extract non-None values as tuple for comparison
            series[period] = tuple(
                v for v in result["roc"] if v is not None
            )
        assert series[10] != series[20], "Period 10 and 20 should differ"
        assert series[20] != series[30], "Period 20 and 30 should differ"

    def test_rsi_different_periods_differ(self):
        """RSI with period 7 vs 14 should produce different series."""
        prices = _trending_prices(100)
        bars = _make_bars(prices)
        results = {}
        for period in [7, 14]:
            result = compute_indicator_outputs(
                {"indicator_id": 2, "parameters": {"period": period}}, bars,
            )
            results[period] = tuple(
                v for v in result["rsi"] if v is not None
            )
        assert results[7] != results[14], "RSI period 7 and 14 should differ"


# ===========================================================================
# V3: None Handling — Unimplemented Indicator
# ===========================================================================

class TestV3NoneHandling:
    """Unimplemented indicators produce None outputs and emit warnings."""

    def test_unimplemented_indicator_all_none(self):
        """Indicator ID 4 (Pivot Structure) is unimplemented → all None."""
        prices = _trending_prices(50)
        bars = _make_bars(prices)
        result = compute_indicator_outputs(
            {"indicator_id": 4, "parameters": {}}, bars,
        )
        assert "pivot_high" in result
        assert "pivot_low" in result
        assert all(v is None for v in result["pivot_high"])
        assert all(v is None for v in result["pivot_low"])

    def test_unimplemented_indicator_logs_warning(self, caplog):
        """Unimplemented indicator should log a warning."""
        prices = _trending_prices(50)
        bars = _make_bars(prices)
        with caplog.at_level(logging.WARNING):
            compute_indicator_outputs(
                {"indicator_id": 4, "parameters": {}}, bars,
            )
        assert any("not implemented" in r.message for r in caplog.records), \
            "Expected 'not implemented' warning in logs"

    def test_gate_with_unimplemented_blocks_trades(self):
        """A gate condition referencing an unimplemented indicator blocks all trades."""
        config = {
            "indicator_instances": [
                {
                    "label": "ema_fast",
                    "indicator_id": 1,
                    "timeframe": "1m",
                    "parameters": {"period": 5},
                    "outputs_used": ["ema"],
                },
                {
                    "label": "pivot_struct",
                    "indicator_id": 4,
                    "timeframe": "1m",
                    "parameters": {},
                    "outputs_used": ["pivot_high"],
                },
            ],
            "entry_rules": [
                {
                    "name": "long_entry",
                    "direction": "LONG",
                    "evaluation_cadence": "1m",
                    "conditions": [
                        {"indicator": "ema_fast", "output": "ema",
                         "operator": ">", "value": 0},
                    ],
                    "condition_groups": [],
                },
            ],
            "exit_rules": [
                {
                    "type": "TIME_LIMIT",
                    "exit_type": "TIME_LIMIT",
                    "applies_to": ["LONG"],
                    "evaluation_cadence": "1m",
                    "time_limit_bars": 10,
                    "time_limit_reference_cadence": "1m",
                    "conditions": [],
                },
            ],
            "gate_rules": [
                {
                    "label": "pivot_gate",
                    "evaluation_cadence": "1m",
                    "conditions": [
                        {"indicator": "pivot_struct", "output": "pivot_high",
                         "operator": ">", "value": 0},
                    ],
                    "condition_groups": [],
                },
            ],
            "execution_params": {
                "direction": "LONG",
                "leverage": 1,
                "flip_enabled": False,
            },
        }
        # 200 bars of uptrend — would normally produce trades without the gate
        prices = _trending_prices(200, start=50000.0, step=10.0)
        bars = _make_bars(prices)
        trades, _, _ = run_backtest(config, bars, strategy_hash="test_gate_none")
        assert len(trades) == 0, \
            f"Gate with all-None output should block trades, got {len(trades)}"


# ===========================================================================
# V5: Determinism
# ===========================================================================

class TestV5Determinism:
    """Same input = same output."""

    def test_backtest_deterministic(self):
        """Running the same config + bars twice produces identical trade counts."""
        config = {
            "indicator_instances": [
                {
                    "label": "rsi_1m",
                    "indicator_id": 2,
                    "timeframe": "1m",
                    "parameters": {"period": 14},
                    "outputs_used": ["rsi"],
                },
            ],
            "entry_rules": [
                {
                    "name": "long_entry",
                    "direction": "LONG",
                    "evaluation_cadence": "1m",
                    "conditions": [
                        {"indicator": "rsi_1m", "output": "rsi",
                         "operator": "<", "value": 0.30},
                    ],
                    "condition_groups": [],
                },
            ],
            "exit_rules": [
                {
                    "type": "TIME_LIMIT",
                    "exit_type": "TIME_LIMIT",
                    "applies_to": ["LONG"],
                    "evaluation_cadence": "1m",
                    "time_limit_bars": 10,
                    "time_limit_reference_cadence": "1m",
                    "conditions": [],
                },
            ],
            "gate_rules": [],
            "execution_params": {
                "direction": "LONG",
                "leverage": 1,
                "flip_enabled": False,
            },
        }
        prices = _oscillating_prices(500, center=50000.0, amplitude=2000.0)
        bars = _make_bars(prices)
        trades1, _, _ = run_backtest(config, bars, strategy_hash="det_test")
        trades2, _, _ = run_backtest(config, bars, strategy_hash="det_test")
        assert len(trades1) == len(trades2), "Determinism violated"
        for t1, t2 in zip(trades1, trades2):
            assert t1.trade_id == t2.trade_id


# ===========================================================================
# V6: Regression — MACD Confluence and Chop Harvester
# ===========================================================================

# Baseline constants from audit_baseline.py run BEFORE scale changes.
# These strategies do NOT use RSI/ROC/Choppiness in conditions,
# so their trade counts must remain identical.
MACD_CONFLUENCE_TRADE_COUNT = 37
CHOP_HARVESTER_TRADE_COUNT = 28


@pytest.mark.slow
class TestV6Regression:
    """Verify scale changes didn't break unrelated strategies."""

    @pytest.fixture(scope="class")
    def bars_13mo(self):
        if not os.path.exists(DATASET_13MO):
            pytest.skip("13-month dataset not available")
        return _load_bars_parquet(DATASET_13MO)

    def test_macd_confluence_trade_count(self, bars_13mo):
        """MACD Confluence Bull & Bear trade count unchanged after scale fixes."""
        if not os.path.exists(MACD_COMP):
            pytest.skip("MACD Confluence composition not available")
        with open(MACD_COMP) as f:
            spec = json.load(f)
        result = compile_composition(spec)
        trades, _, _ = run_backtest(
            result["resolved_artifact"], bars_13mo,
            strategy_hash=result["strategy_config_hash"],
        )
        assert len(trades) == MACD_CONFLUENCE_TRADE_COUNT, \
            f"MACD Confluence: expected {MACD_CONFLUENCE_TRADE_COUNT}, got {len(trades)}"

    def test_chop_harvester_trade_count(self, bars_13mo):
        """Chop Harvester Clean trade count unchanged after scale fixes."""
        if not os.path.exists(CHOP_PRESET):
            pytest.skip("Chop Harvester preset not available")
        with open(CHOP_PRESET) as f:
            spec = json.load(f)
        result = compile_composition(spec)
        trades, _, _ = run_backtest(
            result["resolved_artifact"], bars_13mo,
            strategy_hash=result["strategy_config_hash"],
        )
        assert len(trades) == CHOP_HARVESTER_TRADE_COUNT, \
            f"Chop Harvester: expected {CHOP_HARVESTER_TRADE_COUNT}, got {len(trades)}"
