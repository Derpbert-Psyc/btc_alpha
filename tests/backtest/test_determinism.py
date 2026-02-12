"""Gate 3: Backtest runner determinism — same inputs = same outputs."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ui.services.backtest_runner import Bar, run_backtest


def _make_synthetic_bars(n_bars=2880, base_price=50000.0, seed=42):
    """Create deterministic synthetic 2-day 1m candles (2880 bars).

    Uses a simple deterministic price walk — no randomness.
    """
    import math
    bars = []
    price = base_price
    start_ts = 1700000000  # Fixed epoch

    for i in range(n_bars):
        # Deterministic price movement: sine wave + trend
        phase = 2 * math.pi * i / 240  # 4-hour cycle
        trend = i * 0.5  # Slow upward trend
        noise = math.sin(i * 7.13) * 50 + math.cos(i * 3.79) * 30  # Deterministic "noise"
        price = base_price + math.sin(phase) * 500 + trend + noise

        # OHLC with deterministic spread
        spread = abs(math.sin(i * 1.37)) * 20 + 5
        o = price + math.sin(i * 2.71) * spread * 0.3
        h = max(price, o) + spread
        l = min(price, o) - spread
        c = price

        bars.append(Bar(
            ts=start_ts + i * 60,
            o=o, h=h, l=l, c=c,
            v=1000.0 + abs(math.sin(i * 0.53)) * 500,
            index=i,
        ))

    return bars


def _make_simple_config():
    """Minimal resolved config: one EMA(1) passthrough + trivial entry/exit."""
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
        "execution_params": {
            "flip_enabled": False,
        },
        "archetype_tags": [],
    }


def test_determinism_same_inputs():
    """Run twice with identical inputs → identical TradeEvents."""
    bars = _make_synthetic_bars()
    config = _make_simple_config()

    trades1, prices1, count1 = run_backtest(config, bars, "test_hash")
    trades2, prices2, count2 = run_backtest(config, bars, "test_hash")

    assert count1 == count2
    assert len(trades1) == len(trades2)

    for t1, t2 in zip(trades1, trades2):
        assert t1.trade_id == t2.trade_id, f"Trade ID mismatch: {t1.trade_id} != {t2.trade_id}"
        assert t1.entry_idx == t2.entry_idx
        assert t1.exit_idx == t2.exit_idx
        assert t1.side == t2.side
        assert t1.gross_return_bps == t2.gross_return_bps
        assert t1.entry_price.value == t2.entry_price.value
        assert t1.exit_price.value == t2.exit_price.value


def test_trade_ids_are_deterministic():
    """Trade IDs are content-derived, not random."""
    bars = _make_synthetic_bars()
    config = _make_simple_config()

    trades, _, _ = run_backtest(config, bars, "hash_a")

    if trades:
        # Same content hash → same trade IDs
        trades2, _, _ = run_backtest(config, bars, "hash_a")
        assert trades[0].trade_id == trades2[0].trade_id

        # Different strategy hash → different trade IDs
        trades3, _, _ = run_backtest(config, bars, "hash_b")
        if trades3:
            assert trades[0].trade_id != trades3[0].trade_id


def test_produces_trades():
    """The backtest actually produces trades (not an empty list)."""
    bars = _make_synthetic_bars()
    config = _make_simple_config()
    trades, _, _ = run_backtest(config, bars, "test")
    assert len(trades) > 0, "Backtest produced no trades — check config/data"


def test_trade_event_schema():
    """TradeEvents have correct field types."""
    bars = _make_synthetic_bars()
    config = _make_simple_config()
    trades, _, _ = run_backtest(config, bars, "test")

    for t in trades:
        assert isinstance(t.trade_id, str)
        assert isinstance(t.entry_idx, int)
        assert isinstance(t.exit_idx, int)
        assert t.exit_idx > t.entry_idx
        assert t.side in ("long", "short")
        assert isinstance(t.gross_return_bps, int)
        assert t.entry_price.value > 0
        assert t.exit_price.value > 0
        assert t.qty.value > 0
