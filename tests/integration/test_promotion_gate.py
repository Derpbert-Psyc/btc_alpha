"""Test promotion service-level gate: FAIL triage → exception, no file written.

Calls write_promotion_for_composition() — the same service-layer function
that the UI [Write Promotion Artifact] button handler calls.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from ui.services.research_services import write_promotion_for_composition


@pytest.fixture
def tmp_research(tmp_path, monkeypatch):
    research_dir = str(tmp_path / "research")
    os.makedirs(os.path.join(research_dir, "promotions"), exist_ok=True)
    monkeypatch.setattr("ui.services.research_services.RESEARCH_DIR", research_dir)
    return research_dir


class TestPromotionGate:

    def test_fail_triage_blocks_promotion(self, tmp_research):
        """Force a FAIL triage result → service must raise, no file written."""
        fail_summary = {
            "tier": "F",
            "tier_action": "Reject — strategy is not viable",
            "test_results": [],
            "trade_count": 50,
        }

        with pytest.raises(ValueError, match="not promotable"):
            write_promotion_for_composition(
                strategy_config_hash="sha256:abcdef1234567890",
                composition_spec_hash="sha256:spec123",
                dataset_prefix="test_dataset",
                runner_economics={"fee_rate": 0.0006, "slippage_bps": 10, "starting_capital": 10000},
                triage_result_summary=fail_summary,
            )

        # Verify no promotion file was written
        promo_dir = os.path.join(tmp_research, "promotions")
        for root, dirs, files in os.walk(promo_dir):
            for f in files:
                if f.endswith(".json"):
                    pytest.fail(f"Promotion file should not exist but found: {f}")

    def test_pass_triage_allows_promotion(self, tmp_research):
        """PASS triage result → promotion artifact written with all fields."""
        pass_summary = {
            "tier": "A",
            "tier_action": "Promote — strong strategy",
            "test_results": [{"name": "test_1", "status": "PASS"}],
            "trade_count": 100,
        }

        filepath = write_promotion_for_composition(
            strategy_config_hash="sha256:abcdef1234567890",
            composition_spec_hash="sha256:spec123",
            dataset_prefix="test_dataset",
            runner_economics={"fee_rate": 0.0006, "slippage_bps": 10, "starting_capital": 10000},
            triage_result_summary=pass_summary,
        )

        assert os.path.exists(filepath)

        import json
        with open(filepath) as f:
            artifact = json.load(f)

        # Verify all 7 required fields
        required = ["tier", "strategy_config_hash", "composition_spec_hash",
                     "dataset_prefix", "runner_economics", "triage_result_summary",
                     "timestamp"]
        for field in required:
            assert field in artifact, f"Missing required field: {field}"

        assert artifact["tier"] == "TRIAGE_PASSED"
        assert artifact["strategy_config_hash"] == "sha256:abcdef1234567890"
        assert artifact["dataset_prefix"] == "test_dataset"

    def test_missing_strategy_hash_raises(self, tmp_research):
        """Empty strategy_config_hash must raise."""
        with pytest.raises(ValueError, match="strategy_config_hash is required"):
            write_promotion_for_composition(
                strategy_config_hash="",
                composition_spec_hash="sha256:spec123",
                dataset_prefix="test",
                runner_economics={},
                triage_result_summary={"tier": "A"},
            )
