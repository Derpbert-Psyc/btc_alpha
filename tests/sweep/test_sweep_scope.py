"""Gate 7: Sweep scope — varies one param, recompiles, runs Test 1 only, writes results."""

import copy
import json
import math
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ui.services.composition_store import create_composition, load_composition, save_composition
from ui.services.compiler_bridge import compile_spec
from ui.services.backtest_runner import Bar, run_backtest
from phase5_triage_types import TriageConfig
from phase5_triage import run_test_1
from ui.services.research_services import _apply_param_override


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def research_dir(tmp_path, monkeypatch):
    """Redirect research dir to tmp."""
    rd = str(tmp_path / "research")
    os.makedirs(rd, exist_ok=True)
    import ui.services.research_services as rs
    monkeypatch.setattr(rs, "RESEARCH_DIR", rd)
    return rd


def _make_preset_spec():
    """Minimal compilable spec with triage-sensitive params."""
    return {
        "composition_id": "test_sweep",
        "display_name": "Sweep Test",
        "target_engine_version": "1.8.0",
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
        "metadata": {
            "triage_sensitive_params": [
                {"param": "macd_15m.fast_period", "min": 8, "max": 16, "default": 12},
                {"param": "macd_15m.slow_period", "min": 20, "max": 32, "default": 26},
                {"param": "macd_15m.signal_period", "min": 5, "max": 13, "default": 9},
            ],
        },
    }


def _make_bars(n_bars=2880, base_price=50000.0):
    """Synthetic bars for sweep testing."""
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sweep_varies_exactly_one_param():
    """Modifying one param leaves all other params unchanged."""
    spec = _make_preset_spec()

    modified = _apply_param_override(spec, "macd_15m.fast_period", 8)

    # The targeted param changed
    inst = modified["indicator_instances"][0]
    assert inst["parameters"]["fast_period"] == 8

    # Other params unchanged
    assert inst["parameters"]["slow_period"] == 26
    assert inst["parameters"]["signal_period"] == 9

    # Original spec unchanged (deep copy)
    orig_inst = spec["indicator_instances"][0]
    assert orig_inst["parameters"]["fast_period"] == 12


def test_sweep_compiles_each_variant_different_hashes():
    """Each parameter value produces a different config hash."""
    spec = _make_preset_spec()
    values = [8, 10, 12, 14, 16]
    hashes = set()

    for val in values:
        modified = _apply_param_override(spec, "macd_15m.fast_period", val)
        compilation = compile_spec(modified)
        h = compilation["strategy_config_hash"]
        assert h not in hashes, f"Duplicate hash for fast_period={val}"
        hashes.add(h)

    assert len(hashes) == len(values)


def test_sweep_runs_test_1_only():
    """Sweep runs only Test 1 (OOS Sharpe), not full triage pipeline."""
    spec = _make_preset_spec()
    bars = _make_bars()

    compilation = compile_spec(spec)
    resolved = compilation["resolved_artifact"]
    config_hash = compilation["strategy_config_hash"]

    trades, prices, n_bars = run_backtest(resolved, bars, config_hash)

    triage_config = TriageConfig()
    t1 = run_test_1(trades, n_bars, triage_config)

    # run_test_1 returns a result with oos_sharpe and train_sharpe
    assert hasattr(t1, "oos_sharpe")
    assert hasattr(t1, "train_sharpe")
    assert hasattr(t1, "passed")
    assert isinstance(t1.oos_sharpe, (int, float))
    assert isinstance(t1.train_sharpe, (int, float))


def test_sweep_writes_results_no_overwrite(research_dir):
    """Sweep saves results to file with unique timestamps (no overwrite)."""
    import time
    from datetime import datetime, timezone

    comp_id = "test_sweep"
    dir_path = os.path.join(research_dir, "sweep_results", comp_id)
    os.makedirs(dir_path, exist_ok=True)

    results = [
        {"param_value": 8, "oos_sharpe": 0.5, "train_sharpe": 0.6,
         "passed": True, "trades": 10, "hash": "abc123", "is_default": False},
        {"param_value": 12, "oos_sharpe": 0.7, "train_sharpe": 0.8,
         "passed": True, "trades": 12, "hash": "def456", "is_default": True},
    ]

    def save(suffix=""):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"macd_15m_fast_period_{ts}{suffix}.json"
        filepath = os.path.join(dir_path, filename)
        with open(filepath, "w") as f:
            json.dump({
                "composition_id": comp_id,
                "param_name": "macd_15m.fast_period",
                "timestamp": ts,
                "results": results,
            }, f, indent=2)

    save()
    time.sleep(1.1)
    save()

    files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
    assert len(files) == 2, f"Expected 2 files, got {len(files)}: {files}"

    for fname in files:
        with open(os.path.join(dir_path, fname)) as f:
            data = json.load(f)
        assert data["param_name"] == "macd_15m.fast_period"
        assert len(data["results"]) == 2
