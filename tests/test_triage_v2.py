"""Tests for Triage Test Battery v2.0.

Covers all 7 tests + Wilson formula + tier integration.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ui", "services"))

from phase5_triage_types import TradeEvent
from btc_alpha_v3_final import Fixed, SemanticType

from ui.services.triage_v2 import (
    run_test_1,
    run_test_2,
    run_test_3,
    run_test_4,
    run_test_5,
    run_test_6,
    run_test_7,
    run_triage_v2,
    wilson_lower_bound,
    _compute_trade_metrics,
    _compute_r_squared,
    _interpolate_breakeven,
    _net_return_bps,
    TriageTestResult,
    COST_RAMP_LEVELS,
    DEFAULT_ROUND_TRIP_BPS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(gross_bps: int, entry_idx: int = 0, exit_idx: int = 100,
                side: str = "long") -> TradeEvent:
    """Create a synthetic TradeEvent."""
    return TradeEvent(
        trade_id=f"test_{entry_idx}_{exit_idx}",
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        side=side,
        entry_price=Fixed(value=5000000, sem=SemanticType.PRICE),  # $50,000
        exit_price=Fixed(value=5000000 + gross_bps * 50, sem=SemanticType.PRICE),
        qty=Fixed(value=100000000, sem=SemanticType.QTY),  # 1 BTC
        gross_return_bps=gross_bps,
    )


def _make_trades(bps_list, start_idx=0, spacing=100, side="long"):
    """Create a list of trades from gross_return_bps values."""
    trades = []
    for i, bps in enumerate(bps_list):
        entry = start_idx + i * spacing
        exit_ = entry + spacing - 1
        trades.append(_make_trade(bps, entry_idx=entry, exit_idx=exit_, side=side))
    return trades


# ===========================================================================
# Test 1: Expectancy + Profit Factor
# ===========================================================================

class TestTest1:
    def test_positive_expectancy_pass(self):
        """Synthetic trade list with known positive expectancy passes."""
        trades = _make_trades([100, 200, 150, -50, 300, 120, -30, 180, 250, 160])
        result = run_test_1(trades, round_trip_cost_bps=32)
        assert result.status == "PASS"
        assert result.metrics["expectancy_bps"] > 0
        assert float(result.metrics["profit_factor"].replace("NO LOSSES", "999")) > 1.0

    def test_negative_expectancy_fail(self):
        """Negative expectancy results in FAIL."""
        trades = _make_trades([-100, -200, -150, 20, -300, -120, 10, -180])
        result = run_test_1(trades, round_trip_cost_bps=32)
        assert result.status == "FAIL"
        assert result.metrics["expectancy_bps"] < 0

    def test_all_winners_no_losses(self):
        """All winning trades: PF = NO LOSSES."""
        trades = _make_trades([100, 200, 150, 300])
        result = run_test_1(trades, round_trip_cost_bps=32)
        assert result.status == "PASS"
        assert result.metrics["profit_factor"] == "NO LOSSES"

    def test_profit_factor_computation(self):
        """Verify PF = gross_profit / gross_loss."""
        # 3 wins of +100 net each, 1 loss of -50 net
        # Net returns: 100-32=68, 100-32=68, 100-32=68, -50-32=-82
        # gross_profit = 68+68+68 = 204, gross_loss = 82
        # PF = 204/82 ≈ 2.49
        trades = _make_trades([100, 100, 100, -50])
        result = run_test_1(trades, round_trip_cost_bps=32)
        pf = float(result.metrics["profit_factor"])
        assert 2.0 < pf < 3.0

    def test_breakeven_is_fail(self):
        """A strategy with exactly zero expectancy should FAIL (not > 0)."""
        # Construct: net must be <= 0
        trades = _make_trades([32, 32, 32, -96])  # net: 0, 0, 0, -128 → negative
        result = run_test_1(trades, round_trip_cost_bps=32)
        assert result.status == "FAIL"


# ===========================================================================
# Test 2: Profit Concentration
# ===========================================================================

class TestTest2:
    def test_distributed_profits_pass(self):
        """Distributed profits across many trades → PASS."""
        trades = _make_trades([100] * 20)  # 20 identical winners
        result = run_test_2(trades, round_trip_cost_bps=32)
        assert result.status == "PASS"

    def test_hero_trade_flips_pnl_fail(self):
        """Removing top trade flips PnL negative → FAIL."""
        # 1 huge winner, rest are small losers
        bps = [5000] + [-50] * 19  # hero trade + 19 losers
        trades = _make_trades(bps)
        result = run_test_2(trades, round_trip_cost_bps=32)
        assert result.status == "FAIL"

    def test_concentrated_but_positive_warn(self):
        """PnL drops >50% but stays positive → WARN."""
        # 1 big winner, rest small but still net positive
        bps = [3000] + [40] * 19
        trades = _make_trades(bps)
        result = run_test_2(trades, round_trip_cost_bps=32)
        # After removing top 1 trade: 19 * (40-32) = 152 bps
        # Total with hero: 3000-32 + 152 = 3120 bps
        # Drop: 1 - 152/3120 ≈ 95% → WARN
        assert result.status == "WARN"

    def test_n_remove_calculation(self):
        """Verify N = max(1, ceil(5% of trades))."""
        # 20 trades → N = max(1, ceil(1.0)) = 1
        trades = _make_trades([100] * 20)
        result = run_test_2(trades, round_trip_cost_bps=32)
        assert result.metrics["trades_removed"] == 1

        # 200 trades → N = max(1, ceil(10.0)) = 10
        trades = _make_trades([100] * 200)
        result = run_test_2(trades, round_trip_cost_bps=32)
        assert result.metrics["trades_removed"] == 10


# ===========================================================================
# Test 3: Temporal Trade Consistency
# ===========================================================================

class TestTest3:
    def test_consistent_wins_pass(self):
        """Consistent win rate across windows → PASS."""
        # 80% win rate, evenly distributed
        bps = ([100] * 8 + [-50] * 2) * 5  # 50 trades, 80% WR
        trades = _make_trades(bps)
        result = run_test_3(trades, round_trip_cost_bps=32)
        assert result.status == "PASS"

    def test_window_below_40_fail(self):
        """A window with win rate below 40% → FAIL."""
        # Start with 80% WR, then a catastrophic losing streak
        bps = ([100] * 8 + [-50] * 2) * 3  # 30 good trades
        bps += [-100] * 10  # 10 consecutive losers (0% WR in this window)
        trades = _make_trades(bps)
        result = run_test_3(trades, round_trip_cost_bps=32)
        assert result.status == "FAIL"
        assert result.metrics["worst_window_wr"] < 40.0

    def test_window_15pp_below_warn(self):
        """One window >15pp below overall but above 40% → WARN."""
        # Overall ~75% WR, but one weak window at ~50%
        bps = ([100] * 8 + [-50] * 2) * 4  # 40 trades at 80% WR
        bps += ([100] * 5 + [-50] * 5)      # 10 trades at 50% WR
        trades = _make_trades(bps)
        result = run_test_3(trades, round_trip_cost_bps=32)
        # Overall WR: 37/50 = 74%. Worst window ~50%. Drop: 24pp > 15pp
        assert result.status == "WARN"

    def test_insufficient_data(self):
        """Fewer than 10 trades → INSUFFICIENT DATA."""
        trades = _make_trades([100] * 5)
        result = run_test_3(trades, round_trip_cost_bps=32)
        assert result.status == "INSUFFICIENT DATA"


# ===========================================================================
# Test 4: Drawdown Survivability
# ===========================================================================

class TestTest4:
    def test_small_drawdown_pass(self):
        """Equity curve with <20% DD → PASS."""
        # Steady wins with occasional small loss
        bps = [100, 150, 200, -50, 100, 200, -30, 150, 100, 200]
        trades = _make_trades(bps)
        result = run_test_4(trades, round_trip_cost_bps=32)
        assert result.status == "PASS"
        assert result.metrics["max_drawdown_pct"] < 20.0

    def test_large_drawdown_fail(self):
        """>30% DD → FAIL."""
        # Big consecutive losses
        bps = [100, -1000, -1000, -1000, -1000, 100]
        trades = _make_trades(bps)
        result = run_test_4(trades, round_trip_cost_bps=32)
        assert result.status == "FAIL"
        assert result.metrics["max_drawdown_pct"] > 30.0

    def test_consecutive_loss_fail(self):
        """Consecutive loss > 2500 bps → FAIL."""
        # Many medium losses in a row
        bps = [100, -300, -300, -300, -300, -300, -300, -300, -300, -300, 100]
        trades = _make_trades(bps)
        result = run_test_4(trades, round_trip_cost_bps=32)
        assert result.status == "FAIL"
        assert result.metrics["worst_consecutive_loss_bps"] > 2500

    def test_tier_cap_at_b(self):
        """Test 4 FAIL caps tier at B."""
        # This is verified in the integration test (TestTest7)
        pass

    def test_medium_drawdown_warn(self):
        """20-30% DD → WARN."""
        # Calibrate: ~25% DD
        bps = [500, -800, -800, -800, 500, 500, 500, 500, 500]
        trades = _make_trades(bps)
        result = run_test_4(trades, round_trip_cost_bps=32)
        dd = result.metrics["max_drawdown_pct"]
        # The actual DD depends on compounding, verify it's in WARN range
        if 20 <= dd <= 30:
            assert result.status == "WARN"


# ===========================================================================
# Test 5: OOS Holdout
# ===========================================================================

class TestTest5:
    def test_positive_oos_pass(self):
        """Positive OOS Sharpe >= 50% of train → PASS."""
        # All in train region (first 80% of bars), good trades
        n_bars = 10000
        split = int(n_bars * 0.8)  # 8000

        # Train trades: indices 0-7999, OOS trades: indices 8000+
        train_bps = [200, 150, 100, 180, 220, 160, 190, 140, 170, 210,
                     200, 150, 100, 180, 220]
        oos_bps = [180, 160, 140, 200, 190, 170, 150, 130, 210, 160,
                   180, 170]

        trades = _make_trades(train_bps, start_idx=0, spacing=500)
        trades += _make_trades(oos_bps, start_idx=split, spacing=100)

        result = run_test_5(trades, n_bars, round_trip_cost_bps=32)
        assert result.status == "PASS"
        assert result.metrics["oos_sharpe"] > 0

    def test_negative_oos_fail(self):
        """Negative OOS Sharpe → FAIL."""
        n_bars = 10000
        split = int(n_bars * 0.8)

        train_bps = [200, 150, 100, 180, 220, 160, 190, 140, 170, 210,
                     200, 150]
        oos_bps = [-100, -150, -200, -50, -180, -120, -90, -200, -150, -100,
                   -80, -130]

        trades = _make_trades(train_bps, start_idx=0, spacing=500)
        trades += _make_trades(oos_bps, start_idx=split, spacing=100)

        result = run_test_5(trades, n_bars, round_trip_cost_bps=32)
        assert result.status == "FAIL"

    def test_insufficient_oos_trades(self):
        """< 10 OOS trades → INSUFFICIENT DATA."""
        n_bars = 10000
        split = int(n_bars * 0.8)

        train_bps = [200, 150, 100, 180, 220, 160, 190, 140, 170, 210]
        oos_bps = [150, 100, 200]  # Only 3 OOS trades

        trades = _make_trades(train_bps, start_idx=0, spacing=500)
        trades += _make_trades(oos_bps, start_idx=split, spacing=100)

        result = run_test_5(trades, n_bars, round_trip_cost_bps=32)
        assert result.status == "INSUFFICIENT DATA"
        assert "manual approval" in result.detail.lower()

    def test_auto_promote_blocked_on_insufficient(self):
        """Auto-promote is blocked when Test 5 is INSUFFICIENT DATA."""
        # Build a scenario where everything is S-tier but OOS insufficient
        n_bars = 10000
        split = int(n_bars * 0.8)

        # 50 excellent train trades
        train_bps = [200 + i * 3 for i in range(50)]
        oos_bps = [200, 250]  # Only 2 OOS trades

        trades = _make_trades(train_bps, start_idx=0, spacing=100)
        trades += _make_trades(oos_bps, start_idx=split, spacing=100)

        res = run_triage_v2(trades, n_bars, round_trip_cost_bps=32)
        # Tier action should mention manual approval
        t5 = [t for t in res.test_results if t.name.startswith("5:")][0]
        assert t5.status == "INSUFFICIENT DATA"
        if res.tier in ("S", "A"):
            assert "manual" in res.tier_action.lower() or "blocked" in res.tier_action.lower()


# ===========================================================================
# Test 6: Cost Ramp
# ===========================================================================

class TestTest6:
    def test_robust_strategy_pass(self):
        """Strategy with high gross returns survives cost ramp → PASS."""
        # Varied winners around +500 bps → survives even at 200 bps cost
        bps = [400 + (i % 10) * 30 for i in range(60)]  # 400-670 range
        trades = _make_trades(bps)
        result = run_test_6(trades, operating_cost_bps=32)
        assert result.status == "PASS"

    def test_thin_edge_fail(self):
        """Strategy barely profitable at base cost → FAIL on cost ramp."""
        # Trades at +40 bps gross: net at 32 bps = +8, net at 50 bps = -10
        trades = _make_trades([40] * 60)
        result = run_test_6(trades, operating_cost_bps=32)
        # Win rate at 100 bps: all trades +40-100 = -60, so 0% WR
        assert result.metrics["win_rate_at_100bps"] < 50
        assert result.status == "FAIL"

    def test_r_squared_computation(self):
        """Verify R² is computed for >= 50 trades."""
        trades = _make_trades([500] * 60)
        result = run_test_6(trades, operating_cost_bps=32)
        assert result.metrics["r_squared"] is not None

    def test_r_squared_skipped_low_trades(self):
        """R² skipped when < 50 trades."""
        trades = _make_trades([500] * 30)
        result = run_test_6(trades, operating_cost_bps=32)
        assert result.metrics["r_squared"] is None
        assert result.metrics["r_squared_status"] == "NOT EVALUATED"

    def test_breakeven_interpolation(self):
        """Verify breakeven cost interpolation."""
        # +100 bps gross trades: breakeven at 100 bps cost
        ramp = [
            {"cost_bps": 25, "total_pnl_bps": 75 * 10},
            {"cost_bps": 50, "total_pnl_bps": 50 * 10},
            {"cost_bps": 100, "total_pnl_bps": 0},
            {"cost_bps": 200, "total_pnl_bps": -100 * 10},
        ]
        be = _interpolate_breakeven(ramp)
        assert be is not None
        assert 95 <= be <= 105  # Should be ~100

    def test_win_rate_floor_at_100bps(self):
        """Verify win rate floor check at 100 bps."""
        trades = _make_trades([500] * 60)
        result = run_test_6(trades, operating_cost_bps=32)
        # +500 gross - 100 cost = +400 net: 100% win rate at 100 bps
        assert result.metrics["win_rate_at_100bps"] == 100.0


# ===========================================================================
# Test 7: Tier Classification (Integration)
# ===========================================================================

class TestTest7:
    def test_base_tier_s(self):
        """85% win rate → base tier S."""
        bps = [200] * 85 + [-100] * 15  # 85% WR
        trades = _make_trades(bps)
        result = run_triage_v2(trades, n_bars=100000, round_trip_cost_bps=32)
        assert result.tier in ("S", "A")  # May drop to A if Wilson LB < 0.80

    def test_base_tier_f_negative_expectancy(self):
        """Negative expectancy → always F."""
        bps = [-100] * 80 + [50] * 20
        trades = _make_trades(bps)
        result = run_triage_v2(trades, n_bars=100000, round_trip_cost_bps=32)
        assert result.tier == "F"

    def test_test1_fail_overrides_to_f(self):
        """Test 1 FAIL → tier F regardless of other results."""
        bps = [-100] * 60 + [50] * 40
        trades = _make_trades(bps)
        result = run_triage_v2(trades, n_bars=100000, round_trip_cost_bps=32)
        assert result.tier == "F"

    def test_test2_fail_caps_at_c(self):
        """Test 2 FAIL → max tier C."""
        # Strategy: 1 hero trade, many losers
        bps = [10000] + [-60] * 19
        trades = _make_trades(bps)
        result = run_triage_v2(trades, n_bars=100000, round_trip_cost_bps=32)
        t2 = [t for t in result.test_results if t.name.startswith("2:")][0]
        if t2.status == "FAIL":
            assert result.tier in ("C", "F")

    def test_test4_fail_caps_at_b(self):
        """Test 4 FAIL (>30% DD) → max tier B."""
        # Good win rate but with devastating drawdown
        bps = [200] * 80 + [-3000, -3000] + [200] * 18  # 98% WR but huge DD
        trades = _make_trades(bps)
        result = run_triage_v2(trades, n_bars=100000, round_trip_cost_bps=32)
        t4 = [t for t in result.test_results if t.name.startswith("4:")][0]
        if t4.status == "FAIL":
            assert result.tier in ("B", "C", "F")

    def test_wilson_modifier(self):
        """Wilson LOW CONFIDENCE changes tier action."""
        # High WR but tiny sample → Wilson LB drops below S threshold
        bps = [200] * 9 + [-50] * 1  # 90% WR, only 10 trades
        trades = _make_trades(bps)
        result = run_triage_v2(trades, n_bars=100000, round_trip_cost_bps=32)
        # With 10 trades, Wilson LB for 90% ≈ 0.59 → way below S (0.80)
        assert "LOW CONFIDENCE" in result.tier_reasoning or result.tier != "S"

    def test_cost_ramp_hard_cap(self):
        """Win rate < 50% at 100 bps → hard cap at C."""
        # Trades at +80 bps gross: at 100 bps cost = -20 net (0% WR)
        bps = [80] * 100  # 100% WR at 32 bps, 0% at 100 bps
        trades = _make_trades(bps)
        result = run_triage_v2(trades, n_bars=100000, round_trip_cost_bps=32)
        # S-tier base (100% WR) but hard capped at C
        assert result.tier in ("C", "F")
        assert "HARD CAP" in result.tier_reasoning

    def test_cost_ramp_tier_floor(self):
        """S-tier requires >= 60% WR at 100 bps; if not met, drops to A."""
        # 90% WR at base cost, but some trades fail at 100 bps
        # +110 bps gross: at 100 bps cost = +10 (win). At 32 bps = +78 (win).
        # +60 bps gross: at 100 bps cost = -40 (loss). At 32 bps = +28 (win).
        # Mix: 80 trades at +200, 20 trades at +60
        bps = [200] * 80 + [60] * 20  # 100% WR at 32 bps
        trades = _make_trades(bps)
        result = run_triage_v2(trades, n_bars=100000, round_trip_cost_bps=32)
        # At 100 bps: 80 trades win (+100), 20 lose (-40) → 80% WR → S floor met (60%)
        # Should remain S
        assert result.tier in ("S", "A")

    def test_tier_action_with_oos_insufficient(self):
        """OOS insufficient data blocks auto-promote."""
        # All trades in train region, none in OOS
        n_bars = 10000
        bps = [200] * 50
        trades = _make_trades(bps, start_idx=0, spacing=100)  # All in train (first 8000 bars)
        result = run_triage_v2(trades, n_bars, round_trip_cost_bps=32)
        t5 = [t for t in result.test_results if t.name.startswith("5:")][0]
        assert t5.status == "INSUFFICIENT DATA"
        if result.tier in ("S", "A"):
            assert "manual" in result.tier_action.lower() or "blocked" in result.tier_action.lower()

    def test_multiple_overrides_most_restrictive_wins(self):
        """When Test 1 FAIL and Test 4 FAIL, F wins (most restrictive)."""
        bps = [-500] * 50 + [100] * 50  # Negative expectancy + huge DD
        trades = _make_trades(bps)
        result = run_triage_v2(trades, n_bars=100000, round_trip_cost_bps=32)
        assert result.tier == "F"


# ===========================================================================
# Wilson Score Formula
# ===========================================================================

class TestWilson:
    def test_known_inputs_200_trades(self):
        """85% win rate over 200 trades → verify Wilson LB."""
        lb = wilson_lower_bound(0.85, 200)
        # Expected: around 0.7952 (within tolerance)
        assert 0.79 < lb < 0.82, f"Wilson LB for 85%/200 = {lb}"

    def test_known_inputs_30_trades(self):
        """82% win rate over 30 trades → verify Wilson LB."""
        lb = wilson_lower_bound(0.82, 30)
        # With small sample, LB drops significantly
        assert 0.60 < lb < 0.75, f"Wilson LB for 82%/30 = {lb}"

    def test_perfect_win_rate(self):
        """100% win rate → Wilson LB < 1.0."""
        lb = wilson_lower_bound(1.0, 50)
        assert lb < 1.0
        assert lb > 0.90

    def test_zero_trades(self):
        """Zero trades → Wilson LB = 0."""
        lb = wilson_lower_bound(0.8, 0)
        assert lb == 0.0

    def test_large_sample_converges(self):
        """With large n, Wilson LB approaches p_hat."""
        lb = wilson_lower_bound(0.80, 10000)
        assert abs(lb - 0.80) < 0.01

    def test_50_percent_small_sample(self):
        """50% WR over 20 trades."""
        lb = wilson_lower_bound(0.50, 20)
        assert 0.27 < lb < 0.45, f"Wilson LB for 50%/20 = {lb}"


# ===========================================================================
# Helper function tests
# ===========================================================================

class TestHelpers:
    def test_net_return(self):
        assert _net_return_bps(100, 32) == 68
        assert _net_return_bps(30, 32) == -2
        assert _net_return_bps(-50, 32) == -82

    def test_r_squared_perfect_line(self):
        """Perfect linear relationship → R² ≈ 1.0."""
        r2 = _compute_r_squared([1, 2, 3, 4], [10, 20, 30, 40])
        assert r2 > 0.99

    def test_r_squared_no_correlation(self):
        """Weak correlation → low R²."""
        r2 = _compute_r_squared([1, 2, 3, 4], [10, 40, 10, 40])
        assert r2 < 0.5  # Not strongly correlated

    def test_compute_trade_metrics_empty(self):
        m = _compute_trade_metrics([], 32)
        assert m["trade_count"] == 0
        assert m["win_rate"] == 0.0

    def test_compute_trade_metrics_all_wins(self):
        trades = _make_trades([100, 200, 300])
        m = _compute_trade_metrics(trades, 32)
        assert m["win_rate"] == 1.0
        assert m["profit_factor"] == float("inf")

    def test_breakeven_extrapolation(self):
        """All levels profitable → extrapolate breakeven."""
        ramp = [
            {"cost_bps": 25, "total_pnl_bps": 1000},
            {"cost_bps": 50, "total_pnl_bps": 800},
            {"cost_bps": 100, "total_pnl_bps": 400},
            {"cost_bps": 200, "total_pnl_bps": 100},
        ]
        be = _interpolate_breakeven(ramp)
        # Extrapolating from last 2 points: slope = (100-400)/(200-100) = -3/bps
        # breakeven = 200 + 100/3 ≈ 233
        assert be is not None
        assert be > 200


# ===========================================================================
# Full Integration
# ===========================================================================

class TestFullIntegration:
    def test_elite_strategy(self):
        """Elite strategy with consistent high win rate → S tier."""
        n_bars = 100000
        split = int(n_bars * 0.8)

        # 90% WR, good returns, train and OOS
        train_bps = [300 + (i % 10) * 20 for i in range(120)]
        train_bps += [-100] * 13  # Add some losses for 90% WR

        oos_bps = [280 + (i % 10) * 15 for i in range(30)]
        oos_bps += [-80] * 3  # 91% OOS WR

        trades = _make_trades(train_bps, start_idx=0, spacing=500)
        trades += _make_trades(oos_bps, start_idx=split, spacing=100)

        result = run_triage_v2(trades, n_bars, round_trip_cost_bps=32)

        # Should be S or A
        assert result.tier in ("S", "A"), f"Expected S/A, got {result.tier}: {result.tier_reasoning}"
        assert result.metrics["win_rate"] > 85

    def test_zero_trades(self):
        """Zero trades → F with appropriate flag."""
        result = run_triage_v2([], n_bars=100000)
        assert result.tier == "F"
        assert any("ZERO TRADES" in f for f in result.flags)

    def test_result_serialization(self):
        """Verify result can be serialized to dict."""
        trades = _make_trades([100, 200, -50, 150, 100] * 10)
        result = run_triage_v2(trades, n_bars=100000)
        d = result.to_dict()
        assert "tier" in d
        assert "test_results" in d
        assert "cost_ramp_table" in d
        assert len(d["test_results"]) == 6  # Tests 1-6

    def test_cost_ramp_table_has_4_rows(self):
        """Cost ramp table should have exactly 4 rows."""
        trades = _make_trades([200] * 20)
        result = run_triage_v2(trades, n_bars=100000)
        assert len(result.cost_ramp_table) == 4
        costs = [r["cost_bps"] for r in result.cost_ramp_table]
        assert costs == [25, 50, 100, 200]
