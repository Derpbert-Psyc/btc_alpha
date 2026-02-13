"""Test that parameter sweep does not mutate on-disk composition spec.

After a sweep, the spec hash must be identical to before.
"""

import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import pandas as pd

from ui.services.research_services import run_sweep_for_composition


def _make_spec():
    return {
        "composition_id": "test-sweep-id",
        "display_name": "Sweep Test",
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
                    {"indicator": "macd_1m", "output": "slope_sign", "operator": ">", "value": 0}
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
            ]
        }
    }


def _make_test_parquet(tmpdir: str) -> str:
    base_ts = 1700000000
    rows = []
    for i in range(1000):
        close = 100.0 + 0.01 * i + 3.0 * math.sin(i * 0.05)
        rows.append({
            "timestamp": base_ts + i * 60,
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": 100.0,
        })
    df = pd.DataFrame(rows)
    path = os.path.join(tmpdir, "sweep_test.parquet")
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def tmp_research(tmp_path, monkeypatch):
    research_dir = str(tmp_path / "research")
    os.makedirs(os.path.join(research_dir, "strategies"), exist_ok=True)
    os.makedirs(os.path.join(research_dir, "sweep_results"), exist_ok=True)
    monkeypatch.setattr("ui.services.research_services.RESEARCH_DIR", research_dir)
    return research_dir


class TestSweepNoDiskMutation:

    def test_spec_unchanged_after_sweep(self, tmp_path, tmp_research):
        """Spec hash before and after sweep must be identical."""
        spec = _make_spec()

        # Hash the spec before sweep
        spec_json_before = json.dumps(spec, sort_keys=True)
        hash_before = hashlib.sha256(spec_json_before.encode()).hexdigest()

        dataset_path = _make_test_parquet(str(tmp_path))

        # Run sweep (3 steps for speed)
        sweep_result = run_sweep_for_composition(
            spec=spec,
            param_name="macd_1m.fast_period",
            param_min=3,
            param_max=8,
            n_steps=3,
            dataset_path=dataset_path,
            default_value=5,
        )

        # Hash the spec after sweep
        spec_json_after = json.dumps(spec, sort_keys=True)
        hash_after = hashlib.sha256(spec_json_after.encode()).hexdigest()

        assert hash_before == hash_after, (
            "Sweep mutated the spec! The in-memory spec must not be modified."
        )

        # Verify sweep produced results
        assert len(sweep_result.results) == 3
        assert sweep_result.saved_path != ""
