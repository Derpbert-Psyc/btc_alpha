"""Preset, compiler, and integration tests for Chop Harvester.

Tests:
  P-1: Clean preset loads as valid JSON
  P-2: Tight stop preset has atr_multiple = 1.3
  P-3: Conservative preset has cooldown = 288
  P-4: All presets have required top-level keys
  P-5: All presets have 4 indicator instances
  P-6: All presets have vol_regime gate
  CC-1: Compiler lowers ATR stop correctly
  I-1: Clean preset compiles without error
  I-2: All three presets compile and produce indicator_instances
  BF-1: Compiled preset produces valid resolved config for runner
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PRESET_DIR = os.path.join(os.path.dirname(__file__), "..", "ui", "presets")

CLEAN = os.path.join(PRESET_DIR, "chop_harvester_clean.json")
TIGHT = os.path.join(PRESET_DIR, "chop_harvester_tight_stop.json")
CONSERVATIVE = os.path.join(PRESET_DIR, "chop_harvester_conservative.json")

ALL_PRESETS = [CLEAN, TIGHT, CONSERVATIVE]


def _load(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# P-1 to P-6: Preset validation
# ---------------------------------------------------------------------------

class TestPresetStructure:
    def test_clean_loads(self):
        """P-1: Clean preset loads as valid JSON."""
        data = _load(CLEAN)
        assert data["composition_id"] == "chop_harvester_clean_v1"

    def test_tight_stop_atr_multiple(self):
        """P-2: Tight stop preset has atr_multiple = 1.3."""
        data = _load(TIGHT)
        stop_rule = [r for r in data["exit_rules"] if r.get("exit_type") == "STOP_LOSS"][0]
        assert stop_rule["atr_multiple"] == 1.3

    def test_conservative_cooldown(self):
        """P-3: Conservative preset has cooldown = 288."""
        data = _load(CONSERVATIVE)
        assert data["execution_params"]["post_exit_cooldown_bars"] == 288

    @pytest.mark.parametrize("path", ALL_PRESETS)
    def test_required_keys(self, path):
        """P-4: All presets have required top-level keys."""
        data = _load(path)
        for key in ["composition_id", "indicator_instances", "entry_rules",
                     "exit_rules", "gate_rules", "execution_params"]:
            assert key in data, f"Missing key '{key}' in {os.path.basename(path)}"

    @pytest.mark.parametrize("path", ALL_PRESETS)
    def test_four_instances(self, path):
        """P-5: All presets have exactly 4 indicator instances."""
        data = _load(path)
        assert len(data["indicator_instances"]) == 4

    @pytest.mark.parametrize("path", ALL_PRESETS)
    def test_vol_regime_gate(self, path):
        """P-6: All presets have a vol_regime_gate in gate_rules."""
        data = _load(path)
        gate_labels = [g["label"] for g in data["gate_rules"]]
        assert "vol_regime_gate" in gate_labels


# ---------------------------------------------------------------------------
# CC-1: Compiler ATR lowering
# ---------------------------------------------------------------------------

class TestCompilerATRLowering:
    def test_compiler_lowers_atr_stop(self):
        """CC-1: Compiler should lower atr_multiple into exit rule params."""
        from composition_compiler_v1_5_2 import compile_composition
        data = _load(CLEAN)
        result = compile_composition(data)
        resolved = result["resolved_artifact"]

        # Find STOP_LOSS exit rule
        stop_rules = [r for r in resolved["exit_rules"] if r.get("type") == "STOP_LOSS"]
        assert len(stop_rules) >= 1, "No STOP_LOSS exit rule in resolved config"

        params = stop_rules[0].get("parameters", {})
        assert params.get("atr_multiple") == 1.5
        assert params.get("atr_indicator_label") == "atr_4h"
        assert params.get("mode") == "ATR_MULTIPLE"


# ---------------------------------------------------------------------------
# I-1, I-2: Integration smoke tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_clean_compiles(self):
        """I-1: Clean preset compiles without error."""
        from composition_compiler_v1_5_2 import compile_composition
        data = _load(CLEAN)
        result = compile_composition(data)
        assert "resolved_artifact" in result
        assert "strategy_config_hash" in result

    @pytest.mark.parametrize("path", ALL_PRESETS)
    def test_all_compile(self, path):
        """I-2: All presets compile and produce indicator_instances."""
        from composition_compiler_v1_5_2 import compile_composition
        data = _load(path)
        result = compile_composition(data)
        assert "resolved_artifact" in result, f"{os.path.basename(path)} failed to compile"
        resolved = result["resolved_artifact"]
        assert len(resolved["indicator_instances"]) == 4


# ---------------------------------------------------------------------------
# BF-1: Backtest fidelity
# ---------------------------------------------------------------------------

class TestBacktestFidelity:
    def test_compiled_preset_runs(self):
        """BF-1: Compiled clean preset produces valid config for runner."""
        from composition_compiler_v1_5_2 import compile_composition
        data = _load(CLEAN)
        result = compile_composition(data)
        resolved = result["resolved_artifact"]

        # Verify the resolved config has the expected structure
        assert "indicator_instances" in resolved
        assert "entry_rules" in resolved
        assert "exit_rules" in resolved
        assert "execution_params" in resolved

        # Check execution_params has cooldown
        ep = resolved["execution_params"]
        assert ep.get("post_exit_cooldown_bars") == 144
        assert ep.get("flip_enabled") is True
