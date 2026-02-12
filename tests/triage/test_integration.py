"""Gate 6: Triage integration — TradeEvents → run_triage() → result."""

import math
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ui.services.backtest_runner import Bar, run_backtest
from phase5_triage_types import (
    TradeEvent, TriageConfig, TriageResult, StrategyMetadata,
)
from phase5_triage import run_triage
from btc_alpha_v3_final import Fixed, SemanticType


def _make_synthetic_bars(n_bars=5000, base_price=50000.0):
    """Create deterministic bars with enough data for triage."""
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


def _make_config():
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


def _make_metadata(strategy_hash="test_hash"):
    return StrategyMetadata(
        strategy_id=strategy_hash,
        strategy_version_hash=strategy_hash,
        param_defaults={"fast_period": 12, "slow_period": 26, "signal_period": 9},
        param_bounds={"fast_period": (8, 16), "slow_period": (20, 32), "signal_period": (5, 13)},
        triage_sensitive_params=("fast_period", "slow_period", "signal_period"),
    )


def test_triage_runs_end_to_end():
    """run_triage() completes and returns a TriageResult."""
    bars = _make_synthetic_bars()
    config = _make_config()

    trades, close_prices_float, n_bars = run_backtest(config, bars, "triage_test")
    close_prices_fixed = [Fixed(value=round(p * 100), sem=SemanticType.PRICE)
                          for p in close_prices_float]

    metadata = _make_metadata()
    triage_config = TriageConfig()

    split_idx = int(n_bars * triage_config.train_fraction)
    split_timestamp = str(bars[split_idx].ts if split_idx < len(bars) else 0)

    def evaluate_fn(param_overrides):
        t, _, _ = run_backtest(config, bars, "triage_test")
        return t

    result = run_triage(
        strategy_id="triage_test",
        trades=trades,
        close_prices=close_prices_fixed,
        n_bars=n_bars,
        metadata=metadata,
        config=triage_config,
        dataset_hash="test_dataset_hash",
        split_timestamp=split_timestamp,
        evaluate_fn=evaluate_fn,
    )

    assert isinstance(result, TriageResult)
    assert isinstance(result.passed, bool)
    assert result.reason is not None
    assert "test_1" in result.test_results


def test_triage_result_stable():
    """Same inputs → same PASS/FAIL result."""
    bars = _make_synthetic_bars()
    config = _make_config()

    trades, close_prices_float, n_bars = run_backtest(config, bars, "stable_test")
    close_prices_fixed = [Fixed(value=round(p * 100), sem=SemanticType.PRICE)
                          for p in close_prices_float]

    metadata = _make_metadata("stable_test")
    triage_config = TriageConfig()
    split_idx = int(n_bars * triage_config.train_fraction)
    split_timestamp = str(bars[split_idx].ts)

    def evaluate_fn(param_overrides):
        t, _, _ = run_backtest(config, bars, "stable_test")
        return t

    r1 = run_triage("stable_test", trades, close_prices_fixed, n_bars,
                    metadata, triage_config, "ds_hash", split_timestamp, evaluate_fn)
    r2 = run_triage("stable_test", trades, close_prices_fixed, n_bars,
                    metadata, triage_config, "ds_hash", split_timestamp, evaluate_fn)

    assert r1.passed == r2.passed
    assert r1.reason == r2.reason


def test_triage_result_save(tmp_path, monkeypatch):
    """Triage result saves to research/triage_results/ with correct path."""
    import ui.services.triage_bridge as tb
    monkeypatch.setattr(tb, "RESEARCH_DIR", str(tmp_path / "research"))

    bars = _make_synthetic_bars()
    config = _make_config()

    trades, close_prices_float, n_bars = run_backtest(config, bars, "save_test")
    close_prices_fixed = [Fixed(value=round(p * 100), sem=SemanticType.PRICE)
                          for p in close_prices_float]

    metadata = _make_metadata("save_test")
    triage_config = TriageConfig()
    split_idx = int(n_bars * triage_config.train_fraction)
    split_timestamp = str(bars[split_idx].ts)

    def evaluate_fn(param_overrides):
        t, _, _ = run_backtest(config, bars, "save_test")
        return t

    result = run_triage("save_test", trades, close_prices_fixed, n_bars,
                        metadata, triage_config, "ds_hash_abc", split_timestamp, evaluate_fn)

    filepath = tb.save_triage_result("sha256:abc123", result, "ds_hash_abc")
    assert os.path.exists(filepath)
    assert "ds_hash_" in filepath
    assert "_triage.json" in filepath
