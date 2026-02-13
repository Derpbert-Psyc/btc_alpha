"""Triage Test Battery v2.0 — 7-test automated strategy evaluation.

Tests:
    1. Expectancy Per Trade + Profit Factor
    2. Profit Concentration (remove top 5%)
    3. Temporal Trade Consistency (rolling windows)
    4. Drawdown Survivability (max DD, consecutive loss)
    5. OOS Holdout (80/20 split, Sharpe comparison)
    6. Cost Ramp (4 cost levels, R², breakeven, win rate floor)
    7. Tier Classification (S/A/B/C/F, Wilson, modifiers)

Reference: TRIAGE_TEST_BATTERY_v2.md
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from phase5_triage_types import TradeEvent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default operating costs (round trip = 2 * (fee + slippage) = 2 * (6 + 10) = 32 bps)
DEFAULT_FEE_PER_FILL_BPS = 6
DEFAULT_SLIPPAGE_PER_FILL_BPS = 10
DEFAULT_ROUND_TRIP_BPS = 2 * (DEFAULT_FEE_PER_FILL_BPS + DEFAULT_SLIPPAGE_PER_FILL_BPS)

DEFAULT_STARTING_CAPITAL = 10_000  # USD

COST_RAMP_LEVELS = [25, 50, 100, 200]  # Round-trip bps

# Tier definitions
TIERS = ["S", "A", "B", "C", "F"]

TIER_ACTIONS = {
    "S": "Auto-promote to shadow",
    "A": "Promote to shadow with monitoring",
    "B": "Manual review required",
    "C": "Do not deploy. Investigate only.",
    "F": "Discard. No viable edge.",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class TriageTestResult:
    """Result from a single triage test."""
    def __init__(self, name: str, status: str, detail: str = "",
                 metrics: Optional[Dict[str, Any]] = None):
        self.name = name
        self.status = status  # PASS, WARN, FAIL, INSUFFICIENT DATA
        self.detail = detail
        self.metrics = metrics or {}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "metrics": self.metrics,
        }


class TriageV2Result:
    """Complete triage battery result."""
    def __init__(self):
        self.test_results: List[TriageTestResult] = []
        self.tier: str = "F"
        self.tier_action: str = TIER_ACTIONS["F"]
        self.flags: List[str] = []
        self.warnings: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.cost_ramp_table: List[Dict[str, Any]] = []
        self.tier_reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "test_results": [t.to_dict() for t in self.test_results],
            "tier": self.tier,
            "tier_action": self.tier_action,
            "flags": self.flags,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "cost_ramp_table": self.cost_ramp_table,
            "tier_reasoning": self.tier_reasoning,
        }


# ---------------------------------------------------------------------------
# Helper: apply costs to trade gross returns
# ---------------------------------------------------------------------------

def _net_return_bps(gross_bps: int, round_trip_cost_bps: int) -> float:
    """Compute net return after round-trip cost."""
    return gross_bps - round_trip_cost_bps


def _compute_trade_metrics(trades: List[TradeEvent],
                           round_trip_cost_bps: int) -> Dict[str, Any]:
    """Compute standard metrics on a trade list at a given cost level."""
    if not trades:
        return {
            "trade_count": 0, "win_rate": 0.0, "expectancy_bps": 0.0,
            "profit_factor": 0.0, "total_pnl_bps": 0.0,
            "avg_win_bps": 0.0, "avg_loss_bps": 0.0,
            "gross_profit_bps": 0.0, "gross_loss_bps": 0.0,
        }

    net_returns = [_net_return_bps(t.gross_return_bps, round_trip_cost_bps)
                   for t in trades]

    wins = [r for r in net_returns if r > 0]
    losses = [r for r in net_returns if r <= 0]

    win_rate = len(wins) / len(net_returns) if net_returns else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(abs(r) for r in losses) / len(losses) if losses else 0.0
    total_pnl = sum(net_returns)

    gross_profit = sum(wins)
    gross_loss = sum(abs(r) for r in losses)

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf")

    loss_rate = 1 - win_rate
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    return {
        "trade_count": len(trades),
        "win_rate": win_rate,
        "expectancy_bps": expectancy,
        "profit_factor": profit_factor,
        "total_pnl_bps": total_pnl,
        "avg_win_bps": avg_win,
        "avg_loss_bps": avg_loss,
        "gross_profit_bps": gross_profit,
        "gross_loss_bps": gross_loss,
    }


def _compute_sharpe(returns: List[float]) -> float:
    """Compute per-trade Sharpe ratio (ddof=1)."""
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std < 1e-10:
        return 0.0
    return mean_r / std


# ---------------------------------------------------------------------------
# Test 1: Expectancy Per Trade + Profit Factor
# ---------------------------------------------------------------------------

def run_test_1(trades: List[TradeEvent],
               round_trip_cost_bps: int) -> TriageTestResult:
    """Test 1: Expectancy Per Trade + Profit Factor."""
    m = _compute_trade_metrics(trades, round_trip_cost_bps)

    pf_display = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "NO LOSSES"

    if m["profit_factor"] == float("inf"):
        pf_label = "elite (no losses)"
    elif m["profit_factor"] > 3.0:
        pf_label = "elite"
    elif m["profit_factor"] > 2.0:
        pf_label = "strong"
    elif m["profit_factor"] > 1.5:
        pf_label = "solid"
    elif m["profit_factor"] > 1.0:
        pf_label = "positive but thin"
    else:
        pf_label = "negative"

    metrics = {
        "expectancy_bps": round(m["expectancy_bps"], 2),
        "profit_factor": pf_display,
        "profit_factor_label": pf_label,
        "win_rate": round(m["win_rate"] * 100, 2),
        "avg_win_bps": round(m["avg_win_bps"], 2),
        "avg_loss_bps": round(m["avg_loss_bps"], 2),
        "total_pnl_bps": round(m["total_pnl_bps"], 2),
    }

    if m["expectancy_bps"] > 0:
        status = "PASS"
        detail = (f"Expectancy: {m['expectancy_bps']:+.1f} bps/trade | "
                  f"PF: {pf_display} ({pf_label}) | "
                  f"Win rate: {m['win_rate']*100:.1f}%")
    else:
        status = "FAIL"
        detail = (f"Negative expectancy: {m['expectancy_bps']:+.1f} bps/trade | "
                  f"PF: {pf_display}")

    return TriageTestResult("1: Expectancy + PF", status, detail, metrics)


# ---------------------------------------------------------------------------
# Test 2: Profit Concentration
# ---------------------------------------------------------------------------

def run_test_2(trades: List[TradeEvent],
               round_trip_cost_bps: int) -> TriageTestResult:
    """Test 2: Profit Concentration — remove top 5% trades, check profitability."""
    if len(trades) < 2:
        return TriageTestResult("2: Profit Concentration", "INSUFFICIENT DATA",
                          "Need at least 2 trades")

    net_returns = [_net_return_bps(t.gross_return_bps, round_trip_cost_bps)
                   for t in trades]
    total_pnl = sum(net_returns)

    # Sort descending and remove top N
    n_remove = max(1, math.ceil(0.05 * len(trades)))
    sorted_returns = sorted(net_returns, reverse=True)
    remaining = sorted_returns[n_remove:]
    remaining_pnl = sum(remaining)

    metrics = {
        "total_pnl_bps": round(total_pnl, 2),
        "trades_removed": n_remove,
        "remaining_pnl_bps": round(remaining_pnl, 2),
        "pnl_drop_pct": round((1 - remaining_pnl / total_pnl) * 100, 1) if total_pnl > 0 else 0,
    }

    if remaining_pnl <= 0:
        status = "FAIL"
        detail = (f"Removing top {n_remove} trade(s) flips PnL negative: "
                  f"{remaining_pnl:+.0f} bps (was {total_pnl:+.0f} bps)")
    elif total_pnl > 0 and remaining_pnl < total_pnl * 0.5:
        status = "WARN"
        detail = (f"PnL drops {metrics['pnl_drop_pct']:.0f}% after removing "
                  f"top {n_remove} trade(s): {remaining_pnl:+.0f} bps "
                  f"(was {total_pnl:+.0f} bps)")
    else:
        status = "PASS"
        detail = (f"PnL still positive after removing top {n_remove} trade(s): "
                  f"{remaining_pnl:+.0f} bps (was {total_pnl:+.0f} bps)")

    return TriageTestResult("2: Profit Concentration", status, detail, metrics)


# ---------------------------------------------------------------------------
# Test 3: Temporal Trade Consistency
# ---------------------------------------------------------------------------

def run_test_3(trades: List[TradeEvent],
               round_trip_cost_bps: int) -> TriageTestResult:
    """Test 3: Temporal Trade Consistency — rolling windows, 15pp threshold."""
    if len(trades) < 10:
        return TriageTestResult("3: Temporal Consistency", "INSUFFICIENT DATA",
                          f"Need at least 10 trades, got {len(trades)}")

    net_returns = [_net_return_bps(t.gross_return_bps, round_trip_cost_bps)
                   for t in trades]
    overall_wr = sum(1 for r in net_returns if r > 0) / len(net_returns)

    # Window size
    w = max(10, math.ceil(0.15 * len(trades)))
    step = max(1, w // 2)

    windows = []
    i = 0
    while i + w <= len(net_returns):
        window = net_returns[i:i + w]
        wr = sum(1 for r in window if r > 0) / len(window)
        windows.append({"start_idx": i, "end_idx": i + w, "win_rate": wr})
        i += step

    # If no complete windows (shouldn't happen if len >= 10 and w >= 10)
    if not windows:
        return TriageTestResult("3: Temporal Consistency", "PASS",
                          "Too few trades for windowed analysis")

    worst_wr = min(w_["win_rate"] for w_ in windows)
    worst_drop = overall_wr - worst_wr

    metrics = {
        "overall_win_rate": round(overall_wr * 100, 2),
        "window_size": w,
        "num_windows": len(windows),
        "worst_window_wr": round(worst_wr * 100, 2),
        "max_drop_pp": round(worst_drop * 100, 1),
    }

    if worst_wr < 0.40:
        status = "FAIL"
        detail = (f"Window with {worst_wr*100:.0f}% win rate (below 40% floor). "
                  f"Overall: {overall_wr*100:.0f}%, drop: {worst_drop*100:.0f}pp")
    elif worst_drop > 0.25:
        status = "WARN"
        detail = (f"Window drops {worst_drop*100:.0f}pp below overall "
                  f"({worst_wr*100:.0f}% vs {overall_wr*100:.0f}%)")
    elif worst_drop > 0.15:
        status = "WARN"
        detail = (f"Window drops {worst_drop*100:.0f}pp below overall "
                  f"({worst_wr*100:.0f}% vs {overall_wr*100:.0f}%)")
    else:
        status = "PASS"
        detail = (f"All {len(windows)} windows within 15pp of overall "
                  f"{overall_wr*100:.0f}% win rate "
                  f"(worst: {worst_wr*100:.0f}%, drop: {worst_drop*100:.0f}pp)")

    return TriageTestResult("3: Temporal Consistency", status, detail, metrics)


# ---------------------------------------------------------------------------
# Test 4: Drawdown Survivability
# ---------------------------------------------------------------------------

def run_test_4(trades: List[TradeEvent],
               round_trip_cost_bps: int,
               starting_capital: float = DEFAULT_STARTING_CAPITAL) -> TriageTestResult:
    """Test 4: Drawdown Survivability — max DD, consecutive loss."""
    if not trades:
        return TriageTestResult("4: Drawdown", "INSUFFICIENT DATA", "No trades")

    net_returns = [_net_return_bps(t.gross_return_bps, round_trip_cost_bps)
                   for t in trades]

    # Build equity curve
    equity = [starting_capital]
    for r in net_returns:
        new_eq = equity[-1] * (1 + r / 10000)
        equity.append(new_eq)

    # Max drawdown
    peak = equity[0]
    max_dd_pct = 0.0
    max_dd_dollar = 0.0
    for eq in equity:
        if eq > peak:
            peak = eq
        dd_dollar = peak - eq
        dd_pct = dd_dollar / peak if peak > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_dollar = dd_dollar

    # Consecutive losses
    max_consec_loss = 0
    worst_consec_loss_bps = 0.0
    current_streak = 0
    current_streak_bps = 0.0

    for r in net_returns:
        if r <= 0:
            current_streak += 1
            current_streak_bps += abs(r)
            if current_streak > max_consec_loss:
                max_consec_loss = current_streak
            if current_streak_bps > worst_consec_loss_bps:
                worst_consec_loss_bps = current_streak_bps
        else:
            current_streak = 0
            current_streak_bps = 0.0

    metrics = {
        "max_drawdown_pct": round(max_dd_pct * 100, 2),
        "max_drawdown_dollar": round(max_dd_dollar, 2),
        "max_consecutive_losses": max_consec_loss,
        "worst_consecutive_loss_bps": round(worst_consec_loss_bps, 1),
        "final_equity": round(equity[-1], 2),
    }

    dd_fail = max_dd_pct > 0.30
    cl_fail = worst_consec_loss_bps > 2500
    dd_warn = 0.20 <= max_dd_pct <= 0.30
    cl_warn = 1500 <= worst_consec_loss_bps <= 2500

    if dd_fail or cl_fail:
        status = "FAIL"
        parts = []
        if dd_fail:
            parts.append(f"max DD {max_dd_pct*100:.1f}% > 30%")
        if cl_fail:
            parts.append(f"consecutive loss {worst_consec_loss_bps:.0f} bps > 2500")
        detail = "FAIL: " + " AND ".join(parts)
    elif dd_warn or cl_warn:
        status = "WARN"
        parts = []
        if dd_warn:
            parts.append(f"max DD {max_dd_pct*100:.1f}% (20-30% range)")
        if cl_warn:
            parts.append(f"consecutive loss {worst_consec_loss_bps:.0f} bps (1500-2500 range)")
        detail = "WARN: " + " AND ".join(parts)
    else:
        status = "PASS"
        detail = (f"Max DD: {max_dd_pct*100:.1f}% | "
                  f"Max consecutive losses: {max_consec_loss} "
                  f"({worst_consec_loss_bps:.0f} bps)")

    return TriageTestResult("4: Drawdown", status, detail, metrics)


# ---------------------------------------------------------------------------
# Test 5: OOS Holdout
# ---------------------------------------------------------------------------

def run_test_5(trades: List[TradeEvent],
               n_bars: int,
               round_trip_cost_bps: int,
               train_fraction: float = 0.8) -> TriageTestResult:
    """Test 5: OOS Holdout — 80/20 split, Sharpe comparison."""
    if not trades:
        return TriageTestResult("5: OOS Holdout", "INSUFFICIENT DATA", "No trades")

    split_idx = int(n_bars * train_fraction)

    train_trades = [t for t in trades if t.entry_idx < split_idx]
    oos_trades = [t for t in trades if t.entry_idx >= split_idx]

    train_returns = [_net_return_bps(t.gross_return_bps, round_trip_cost_bps)
                     for t in train_trades]
    oos_returns = [_net_return_bps(t.gross_return_bps, round_trip_cost_bps)
                   for t in oos_trades]

    metrics = {
        "train_trades": len(train_trades),
        "oos_trades": len(oos_trades),
        "split_bar_idx": split_idx,
    }

    if len(oos_trades) < 10:
        metrics["oos_sharpe"] = None
        metrics["train_sharpe"] = _compute_sharpe(train_returns) if len(train_returns) >= 2 else None
        return TriageTestResult(
            "5: OOS Holdout", "INSUFFICIENT DATA",
            f"Only {len(oos_trades)} OOS trades (need 10). "
            "Manual approval required for shadow promotion.",
            metrics,
        )

    train_sharpe = _compute_sharpe(train_returns) if len(train_returns) >= 2 else 0.0
    oos_sharpe = _compute_sharpe(oos_returns)

    metrics["train_sharpe"] = round(train_sharpe, 4)
    metrics["oos_sharpe"] = round(oos_sharpe, 4)
    metrics["oos_to_train_ratio"] = round(oos_sharpe / train_sharpe, 4) if train_sharpe != 0 else None

    if oos_sharpe <= 0:
        status = "FAIL"
        detail = (f"OOS Sharpe {oos_sharpe:.4f} <= 0 "
                  f"(Train: {train_sharpe:.4f}, {len(oos_trades)} OOS trades)")
    elif train_sharpe > 0 and oos_sharpe < 0.5 * train_sharpe:
        status = "WARN"
        ratio = oos_sharpe / train_sharpe
        detail = (f"OOS Sharpe {oos_sharpe:.4f} < 50% of Train {train_sharpe:.4f} "
                  f"(ratio: {ratio:.2f}, {len(oos_trades)} OOS trades)")
    else:
        status = "PASS"
        ratio_str = f"{oos_sharpe/train_sharpe:.2f}" if train_sharpe > 0 else "N/A"
        detail = (f"OOS Sharpe {oos_sharpe:.4f}, Train {train_sharpe:.4f}, "
                  f"ratio {ratio_str} ({len(oos_trades)} OOS trades)")

    return TriageTestResult("5: OOS Holdout", status, detail, metrics)


# ---------------------------------------------------------------------------
# Test 6: Cost Ramp (Friction Robustness)
# ---------------------------------------------------------------------------

def run_test_6(trades: List[TradeEvent],
               operating_cost_bps: int = DEFAULT_ROUND_TRIP_BPS) -> TriageTestResult:
    """Test 6: Cost Ramp — 4 cost levels, R², breakeven, win rate floor."""
    if not trades:
        return TriageTestResult("6: Cost Ramp", "INSUFFICIENT DATA", "No trades")

    cost_levels = COST_RAMP_LEVELS
    ramp_data = []

    for cost in cost_levels:
        net_returns = [_net_return_bps(t.gross_return_bps, cost) for t in trades]
        m = _compute_trade_metrics(trades, cost)

        sharpe = _compute_sharpe(net_returns) if len(net_returns) >= 2 else 0.0

        ramp_data.append({
            "cost_bps": cost,
            "win_rate": m["win_rate"],
            "sharpe": sharpe,
            "total_pnl_bps": m["total_pnl_bps"],
            "expectancy_bps": m["expectancy_bps"],
            "profit_factor": m["profit_factor"] if m["profit_factor"] != float("inf") else 999.99,
        })

    # Breakeven interpolation: find where total_pnl crosses zero
    breakeven_cost = _interpolate_breakeven(ramp_data)

    # R² of Sharpe across cost levels (only if >= 50 trades)
    r_squared = None
    r_squared_status = "NOT EVALUATED"
    if len(trades) >= 50:
        costs = [d["cost_bps"] for d in ramp_data]
        sharpes = [d["sharpe"] for d in ramp_data]
        r_squared = _compute_r_squared(costs, sharpes)
        if r_squared >= 0.85:
            r_squared_status = "PASS"
        elif r_squared >= 0.70:
            r_squared_status = "WARN"
        else:
            r_squared_status = "FAIL"

    # Win rate at 100 bps
    wr_100 = next((d["win_rate"] for d in ramp_data if d["cost_bps"] == 100), 0)

    # Breakeven headroom
    if breakeven_cost is not None:
        headroom_ratio = breakeven_cost / operating_cost_bps if operating_cost_bps > 0 else float("inf")
    else:
        # All levels profitable → breakeven > 200 bps
        headroom_ratio = 200.0 / operating_cost_bps if operating_cost_bps > 0 else float("inf")

    # Determine overall status
    statuses = []
    status_details = []

    # Linearity
    if r_squared is not None:
        statuses.append(r_squared_status)
        status_details.append(f"R²: {r_squared:.3f} ({r_squared_status})")
    else:
        status_details.append(f"R²: NOT EVALUATED (< 50 trades)")

    # Breakeven headroom
    if breakeven_cost is not None:
        if headroom_ratio > 3:
            statuses.append("PASS")
            status_details.append(f"Breakeven: {breakeven_cost:.0f} bps ({headroom_ratio:.1f}x operating)")
        elif headroom_ratio >= 2:
            statuses.append("WARN")
            status_details.append(f"Breakeven: {breakeven_cost:.0f} bps ({headroom_ratio:.1f}x operating, WARN)")
        else:
            statuses.append("FAIL")
            status_details.append(f"Breakeven: {breakeven_cost:.0f} bps ({headroom_ratio:.1f}x operating, FAIL)")
    else:
        statuses.append("PASS")
        status_details.append(f"Breakeven: >200 bps (>{200/operating_cost_bps:.1f}x operating)")

    # Win rate floor at 100 bps
    if wr_100 >= 0.60:
        statuses.append("PASS")
        status_details.append(f"WR@100bps: {wr_100*100:.0f}% (PASS)")
    elif wr_100 >= 0.50:
        statuses.append("WARN")
        status_details.append(f"WR@100bps: {wr_100*100:.0f}% (WARN)")
    else:
        statuses.append("FAIL")
        status_details.append(f"WR@100bps: {wr_100*100:.0f}% (FAIL)")

    # Overall: worst of sub-tests
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses:
        overall = "WARN"
    else:
        overall = "PASS"

    metrics = {
        "breakeven_cost_bps": breakeven_cost,
        "breakeven_headroom_ratio": round(headroom_ratio, 2) if headroom_ratio != float("inf") else ">6x",
        "r_squared": round(r_squared, 4) if r_squared is not None else None,
        "r_squared_status": r_squared_status,
        "win_rate_at_100bps": round(wr_100 * 100, 1),
    }

    return TriageTestResult("6: Cost Ramp", overall,
                      " | ".join(status_details), metrics)


def _interpolate_breakeven(ramp_data: List[Dict]) -> Optional[float]:
    """Interpolate the cost level where PnL crosses zero."""
    # If all levels positive, breakeven is above max tested
    if all(d["total_pnl_bps"] > 0 for d in ramp_data):
        # Extrapolate from last two points
        if len(ramp_data) >= 2:
            d1 = ramp_data[-2]
            d2 = ramp_data[-1]
            pnl1, pnl2 = d1["total_pnl_bps"], d2["total_pnl_bps"]
            c1, c2 = d1["cost_bps"], d2["cost_bps"]
            if pnl1 != pnl2:
                slope = (pnl2 - pnl1) / (c2 - c1)
                if slope < 0:
                    breakeven = c2 + (-pnl2 / slope)
                    return round(breakeven, 1)
        return None  # Can't determine

    # If first level already negative, breakeven is below minimum
    if ramp_data[0]["total_pnl_bps"] <= 0:
        return 0.0

    # Find the crossing point
    for i in range(1, len(ramp_data)):
        if ramp_data[i]["total_pnl_bps"] <= 0:
            d1 = ramp_data[i - 1]
            d2 = ramp_data[i]
            pnl1, pnl2 = d1["total_pnl_bps"], d2["total_pnl_bps"]
            c1, c2 = d1["cost_bps"], d2["cost_bps"]
            if pnl1 == pnl2:
                return float(c1)
            # Linear interpolation
            frac = pnl1 / (pnl1 - pnl2)
            return round(c1 + frac * (c2 - c1), 1)

    return None


def _compute_r_squared(x: List[float], y: List[float]) -> float:
    """Compute R² for a simple linear regression."""
    n = len(x)
    if n < 2:
        return 0.0
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    ss_xx = sum((xi - x_mean) ** 2 for xi in x)
    ss_yy = sum((yi - y_mean) ** 2 for yi in y)
    if ss_xx == 0 or ss_yy == 0:
        return 0.0
    r = ss_xy / math.sqrt(ss_xx * ss_yy)
    return r * r


# ---------------------------------------------------------------------------
# Wilson Score Lower Bound
# ---------------------------------------------------------------------------

def wilson_lower_bound(p_hat: float, n: int, z: float = 1.96) -> float:
    """Wilson score 95% confidence interval lower bound.

    Formula:
        (1 / (1 + z²/n)) × (p̂ + z²/(2n) - z × √(p̂(1-p̂)/n + z²/(4n²)))
    """
    if n <= 0:
        return 0.0
    z2 = z * z
    denominator = 1 + z2 / n
    center = p_hat + z2 / (2 * n)
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    return (center - spread) / denominator


# ---------------------------------------------------------------------------
# Test 7: Tier Classification
# ---------------------------------------------------------------------------

def _base_tier_from_win_rate(win_rate: float) -> str:
    """Step 1: Base tier from win rate."""
    if win_rate >= 0.80:
        return "S"
    elif win_rate >= 0.70:
        return "A"
    elif win_rate >= 0.60:
        return "B"
    elif win_rate >= 0.50:
        return "C"
    else:
        return "F"


def _tier_floor(tier: str) -> float:
    """Return win rate floor for a tier."""
    return {"S": 0.80, "A": 0.70, "B": 0.60, "C": 0.50, "F": 0.0}[tier]


def _tier_index(tier: str) -> int:
    """Numeric index for tier comparison (higher = better)."""
    return {"S": 4, "A": 3, "B": 2, "C": 1, "F": 0}[tier]


def _tier_from_index(idx: int) -> str:
    return {4: "S", 3: "A", 2: "B", 1: "C", 0: "F"}[max(0, min(4, idx))]


def _cap_tier(current: str, max_tier: str) -> str:
    """Cap a tier at max_tier (e.g., cap S at B → B)."""
    if _tier_index(current) > _tier_index(max_tier):
        return max_tier
    return current


def run_test_7(trades: List[TradeEvent],
               round_trip_cost_bps: int,
               test_results: List[TriageTestResult],
               cost_ramp_data: List[Dict[str, Any]]) -> Tuple[str, str, str, List[str]]:
    """Test 7: Tier Classification.

    Returns: (tier, tier_action, reasoning, flags)
    """
    m = _compute_trade_metrics(trades, round_trip_cost_bps)
    win_rate = m["win_rate"]
    n = len(trades)

    reasoning_parts = []
    flags = []

    # Step 1: Base tier from win rate
    tier = _base_tier_from_win_rate(win_rate)
    reasoning_parts.append(f"Base win rate: {win_rate*100:.1f}% ({tier})")

    # Step 2: Wilson confidence modifier
    wilson_lb = wilson_lower_bound(win_rate, n) if n > 0 else 0.0
    wilson_tier = _base_tier_from_win_rate(wilson_lb)
    low_confidence = _tier_index(wilson_tier) < _tier_index(tier)

    if low_confidence:
        reasoning_parts.append(
            f"Wilson 95% LB: {wilson_lb*100:.1f}% ({wilson_tier}) — LOW CONFIDENCE")
        flags.append(f"LOW CONFIDENCE — Wilson 95% LB: {wilson_lb*100:.1f}%")
    else:
        reasoning_parts.append(
            f"Wilson 95% LB: {wilson_lb*100:.1f}% ({wilson_tier}) — CONFIRMED")

    # Step 3: Cost ramp modifier
    wr_100 = 0.0
    for d in cost_ramp_data:
        if d["cost_bps"] == 100:
            wr_100 = d["win_rate"]
            break

    # 3a: Hard cap (applied first)
    cost_ramp_applied = False
    if wr_100 < 0.50:
        tier = _cap_tier(tier, "C")
        reasoning_parts.append(
            f"Cost ramp modifier: HARD CAP (win rate {wr_100*100:.0f}% at 100bps < 50%)")
        cost_ramp_applied = True
    else:
        # 3b: Tier-specific floor (applied second)
        required_floors = {"S": 0.60, "A": 0.55, "B": 0.50}
        floor = required_floors.get(tier)
        if floor is not None and wr_100 < floor:
            old_tier = tier
            tier = _tier_from_index(_tier_index(tier) - 1)
            reasoning_parts.append(
                f"Cost ramp modifier: dropped {old_tier}→{tier} "
                f"(WR@100bps {wr_100*100:.0f}% < {floor*100:.0f}% floor)")
            cost_ramp_applied = True

    if not cost_ramp_applied:
        reasoning_parts.append(
            f"Cost ramp modifier: PASS ({wr_100*100:.0f}% at 100bps, "
            f"floor {required_floors.get(tier, 'N/A')})")

    # Step 4: Test override rules (applied last)
    # Collect test statuses by name prefix
    test_1_result = None
    test_2_result = None
    test_4_result = None

    for tr in test_results:
        if tr.name.startswith("1:"):
            test_1_result = tr
        elif tr.name.startswith("2:"):
            test_2_result = tr
        elif tr.name.startswith("4:"):
            test_4_result = tr

    overrides = []
    if test_1_result and test_1_result.status == "FAIL":
        overrides.append(("F", "Test 1 FAIL (negative expectancy) → F"))
    if test_2_result and test_2_result.status == "FAIL":
        overrides.append(("C", "Test 2 FAIL (profit concentration) → max C"))
    if test_4_result and test_4_result.status == "FAIL":
        overrides.append(("B", "Test 4 FAIL (drawdown) → max B"))

    if overrides:
        # Most restrictive cap wins
        most_restrictive = min(overrides, key=lambda o: _tier_index(o[0]))
        tier = _cap_tier(tier, most_restrictive[0])
        reasoning_parts.append(f"Test overrides: {'; '.join(o[1] for o in overrides)}")
    else:
        reasoning_parts.append("Test overrides: None")

    # Collect flags from WARN tests
    for tr in test_results:
        if tr.status == "WARN":
            flags.append(f"{tr.name} WARN — {tr.detail}")

    # Step 5: Tier action
    tier_action = TIER_ACTIONS[tier]

    # Check for OOS insufficient data
    test_5_result = None
    for tr in test_results:
        if tr.name.startswith("5:"):
            test_5_result = tr
            break

    oos_insufficient = test_5_result and test_5_result.status == "INSUFFICIENT DATA"

    action_modifiers = []
    if oos_insufficient:
        action_modifiers.append("OOS insufficient data")
        reasoning_parts.append(
            f"OOS status: INSUFFICIENT DATA ({test_5_result.metrics.get('oos_trades', 0)} OOS trades)")
    elif test_5_result:
        reasoning_parts.append(
            f"OOS status: {test_5_result.status} "
            f"({test_5_result.metrics.get('oos_trades', '?')} OOS trades"
            f"{', OOS Sharpe ' + str(test_5_result.metrics.get('oos_sharpe', '?')) if test_5_result.metrics.get('oos_sharpe') is not None else ''})")

    if low_confidence:
        action_modifiers.append("LOW CONFIDENCE")

    # Apply action modifiers
    if tier == "S" and action_modifiers:
        tier_action = f"Manual approval required ({' + '.join(action_modifiers)})"
    elif tier == "A" and low_confidence:
        tier_action = "Manual review required (LOW CONFIDENCE downgrades A action to B)"
    elif tier in ("S", "A") and oos_insufficient:
        tier_action = f"{TIER_ACTIONS[tier]} — BLOCKED: manual approval required (OOS insufficient data)"

    reasoning = "\n".join(reasoning_parts)
    return tier, tier_action, reasoning, flags


# ---------------------------------------------------------------------------
# Main entry point: run full triage battery
# ---------------------------------------------------------------------------

def run_triage_v2(
    trades: List[TradeEvent],
    n_bars: int,
    round_trip_cost_bps: int = DEFAULT_ROUND_TRIP_BPS,
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
) -> TriageV2Result:
    """Run the complete 7-test triage battery.

    Args:
        trades: Trade list from a single backtest run.
        n_bars: Total number of 1m bars in the dataset.
        round_trip_cost_bps: Operating round-trip cost in bps (default 32).
        starting_capital: Starting capital in USD.

    Returns:
        TriageV2Result with all test results, tier, and action.
    """
    result = TriageV2Result()

    if not trades:
        result.tier = "F"
        result.tier_action = "Discard. No trades produced."
        result.flags.append("ZERO TRADES — backtest produced no trades")
        return result

    # Test 1: Expectancy + PF
    t1 = run_test_1(trades, round_trip_cost_bps)
    result.test_results.append(t1)

    # Test 2: Profit Concentration
    t2 = run_test_2(trades, round_trip_cost_bps)
    result.test_results.append(t2)

    # Test 3: Temporal Consistency
    t3 = run_test_3(trades, round_trip_cost_bps)
    result.test_results.append(t3)

    # Test 4: Drawdown
    t4 = run_test_4(trades, round_trip_cost_bps, starting_capital)
    result.test_results.append(t4)

    # Test 5: OOS Holdout
    t5 = run_test_5(trades, n_bars, round_trip_cost_bps)
    result.test_results.append(t5)

    # Test 6: Cost Ramp
    t6 = run_test_6(trades, round_trip_cost_bps)
    result.test_results.append(t6)

    # Build cost ramp table for display
    for cost in COST_RAMP_LEVELS:
        m = _compute_trade_metrics(trades, cost)
        net_returns = [_net_return_bps(t.gross_return_bps, cost) for t in trades]
        sharpe = _compute_sharpe(net_returns) if len(net_returns) >= 2 else 0.0
        result.cost_ramp_table.append({
            "cost_bps": cost,
            "win_rate": round(m["win_rate"] * 100, 1),
            "sharpe": round(sharpe, 4),
            "total_pnl_bps": round(m["total_pnl_bps"], 1),
            "profit_factor": (f"{m['profit_factor']:.2f}"
                              if m["profit_factor"] != float("inf") else "NO LOSSES"),
        })

    # Test 7: Tier Classification
    tier, tier_action, reasoning, tier_flags = run_test_7(
        trades, round_trip_cost_bps, result.test_results, result.cost_ramp_table)
    result.tier = tier
    result.tier_action = tier_action
    result.tier_reasoning = reasoning
    result.flags.extend(tier_flags)

    # Collect warnings from individual tests
    for tr in result.test_results:
        if tr.status == "WARN":
            result.warnings.append(f"{tr.name}: {tr.detail}")

    # Aggregate metrics
    base_metrics = _compute_trade_metrics(trades, round_trip_cost_bps)
    result.metrics = {
        "trade_count": len(trades),
        "win_rate": round(base_metrics["win_rate"] * 100, 2),
        "expectancy_bps": round(base_metrics["expectancy_bps"], 2),
        "profit_factor": (f"{base_metrics['profit_factor']:.2f}"
                          if base_metrics["profit_factor"] != float("inf") else "NO LOSSES"),
        "total_pnl_bps": round(base_metrics["total_pnl_bps"], 1),
        "breakeven_cost_bps": t6.metrics.get("breakeven_cost_bps"),
        "wilson_lb": round(wilson_lower_bound(base_metrics["win_rate"], len(trades)) * 100, 2),
        "max_drawdown_pct": t4.metrics.get("max_drawdown_pct"),
        "max_consecutive_losses": t4.metrics.get("max_consecutive_losses"),
        "worst_consecutive_loss_bps": t4.metrics.get("worst_consecutive_loss_bps"),
    }

    return result
