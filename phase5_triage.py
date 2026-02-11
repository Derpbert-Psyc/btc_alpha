"""
Phase 5 — Triage Filter (Tier 1)

Implements the 4-test sequential pipeline from PHASE5_ROBUSTNESS_CONTRACT:
    Test 1: Simple OOS Holdout (hard gate)
    Test 1.5: Cost Sanity Check (diagnostic only)
    Test 2: Monte Carlo Date-Shift (hard gate)
    Test 3: 3-Parameter Sensitivity (hard gate)
    Test 4: Quick Correlation Check (soft gate, log only)

References:
    PHASE5_ROBUSTNESS_CONTRACT_v1_2_0.md — Sections 3-9
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from btc_alpha_v3_final import Fixed, SemanticType

from phase5_triage_types import (
    InsufficientDataError,
    NumericalError,
    PromotionArtifact,
    StrategyMetadata,
    StrategyState,
    TradeEvent,
    TriageConfig,
    TriageRejectionError,
    TriageResult,
    ZeroVarianceError,
    canonical_data_hash,
    compute_sharpe_fixed,
    derive_master_seed,
    derive_subseed,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Split computation
# ---------------------------------------------------------------------------

def compute_split_index(
    n_bars: int,
    train_fraction: float = 0.8,
) -> int:
    """Compute the index that separates train from OOS data.

    Returns the first OOS index (train = [0, split), oos = [split, n_bars)).
    """
    split = int(n_bars * train_fraction)
    return max(1, min(split, n_bars - 1))


# ---------------------------------------------------------------------------
# Trade repricing for Monte Carlo
# ---------------------------------------------------------------------------

def _reprice_trade(
    trade: TradeEvent,
    offset: int,
    close_prices: Sequence[Fixed],
    n_bars: int,
) -> Optional[TradeEvent]:
    """Shift a trade by offset bars, reprice from close_prices.

    Returns None if shifted indices are out of bounds.
    """
    shifted_entry = trade.entry_idx + offset
    shifted_exit = trade.exit_idx + offset

    if shifted_entry < 0 or shifted_exit >= n_bars:
        return None

    new_entry_price = close_prices[shifted_entry]
    new_exit_price = close_prices[shifted_exit]

    # Recompute gross_return_bps (integer arithmetic)
    if new_entry_price.value == 0:
        return None

    if trade.side == "long":
        gross_return_bps = (
            (new_exit_price.value - new_entry_price.value) * 10000
            // new_entry_price.value
        )
    else:  # short
        gross_return_bps = (
            (new_entry_price.value - new_exit_price.value) * 10000
            // new_entry_price.value
        )

    return TradeEvent(
        trade_id=f"{trade.trade_id}_shifted_{offset}",
        entry_idx=shifted_entry,
        exit_idx=shifted_exit,
        side=trade.side,
        entry_price=new_entry_price,
        exit_price=new_exit_price,
        qty=trade.qty,
        gross_return_bps=gross_return_bps,
    )


# ---------------------------------------------------------------------------
# Test 1: Simple OOS Holdout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Test1Result:
    passed: bool
    reason: str
    train_sharpe: int       # RATE scale
    oos_sharpe: int         # RATE scale
    train_trades: int
    oos_trades: int
    split_index: int


def run_test_1(
    trades: List[TradeEvent],
    n_bars: int,
    config: TriageConfig,
) -> Test1Result:
    """Test 1: Simple OOS holdout.  Hard gate."""
    split_idx = compute_split_index(n_bars, config.train_fraction)

    train_trades = [t for t in trades if t.exit_idx < split_idx]
    oos_trades = [t for t in trades if t.entry_idx >= split_idx]

    # Pre-flight checks
    if len(oos_trades) < config.min_oos_trades:
        return Test1Result(
            passed=False,
            reason=f"Insufficient OOS trades: {len(oos_trades)} < {config.min_oos_trades}",
            train_sharpe=0, oos_sharpe=0,
            train_trades=len(train_trades), oos_trades=len(oos_trades),
            split_index=split_idx,
        )

    if n_bars - split_idx < config.min_oos_bars:
        return Test1Result(
            passed=False,
            reason=f"Insufficient OOS bars: {n_bars - split_idx} < {config.min_oos_bars}",
            train_sharpe=0, oos_sharpe=0,
            train_trades=len(train_trades), oos_trades=len(oos_trades),
            split_index=split_idx,
        )

    # Compute Sharpe ratios
    try:
        train_returns = [t.gross_return_bps for t in train_trades]
        oos_returns = [t.gross_return_bps for t in oos_trades]

        train_sharpe_fixed = compute_sharpe_fixed(train_returns)
        oos_sharpe_fixed = compute_sharpe_fixed(oos_returns)

        train_sharpe_val = train_sharpe_fixed.value
        oos_sharpe_val = oos_sharpe_fixed.value
    except (InsufficientDataError, ZeroVarianceError, NumericalError) as e:
        return Test1Result(
            passed=False, reason=f"Sharpe computation failed: {e}",
            train_sharpe=0, oos_sharpe=0,
            train_trades=len(train_trades), oos_trades=len(oos_trades),
            split_index=split_idx,
        )

    # Pass criteria: oos_sharpe >= 0.3 AND oos_sharpe >= 0.5 * train_sharpe
    oos_abs_pass = oos_sharpe_val >= config.oos_sharpe_min
    oos_rel_pass = oos_sharpe_val >= (train_sharpe_val * config.oos_degradation_ratio // 1_000_000)

    if oos_abs_pass and oos_rel_pass:
        reason = "PASS"
        passed = True
    else:
        reasons = []
        if not oos_abs_pass:
            reasons.append(
                f"oos_sharpe={oos_sharpe_val} < min={config.oos_sharpe_min}"
            )
        if not oos_rel_pass:
            threshold = train_sharpe_val * config.oos_degradation_ratio // 1_000_000
            reasons.append(
                f"oos_sharpe={oos_sharpe_val} < 0.5*train={threshold}"
            )
        reason = "; ".join(reasons)
        passed = False

    return Test1Result(
        passed=passed, reason=reason,
        train_sharpe=train_sharpe_val, oos_sharpe=oos_sharpe_val,
        train_trades=len(train_trades), oos_trades=len(oos_trades),
        split_index=split_idx,
    )


# ---------------------------------------------------------------------------
# Test 1.5: Cost Sanity Check (diagnostic only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Test15Result:
    cost_adjusted_sharpe: int   # RATE scale
    warning: Optional[str]


def run_test_15(
    oos_trades: List[TradeEvent],
    config: TriageConfig,
) -> Test15Result:
    """Test 1.5: Cost sanity check.  Diagnostic only — never rejects."""
    if len(oos_trades) < 2:
        return Test15Result(cost_adjusted_sharpe=0, warning="Insufficient OOS trades")

    cost_adjusted = [
        t.gross_return_bps - config.cost_per_trade_bps for t in oos_trades
    ]
    try:
        sharpe = compute_sharpe_fixed(cost_adjusted)
        warning = None
        # Warning threshold: cost-adjusted Sharpe < 0.2
        if sharpe.value < 200_000:
            warning = (
                f"Cost-adjusted Sharpe {sharpe.value} < 200000 (0.2). "
                "Strategy may not survive real-world friction."
            )
        return Test15Result(cost_adjusted_sharpe=sharpe.value, warning=warning)
    except (InsufficientDataError, ZeroVarianceError, NumericalError) as e:
        return Test15Result(cost_adjusted_sharpe=0, warning=str(e))


# ---------------------------------------------------------------------------
# Test 2: Monte Carlo Date-Shift
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Test2Result:
    passed: bool
    reason: str
    p_value: int                 # RATE scale
    real_sharpe: int             # RATE scale
    randomized_sharpes: List[int]  # RATE scale
    valid_iterations: int
    total_iterations: int


def run_test_2(
    oos_trades: List[TradeEvent],
    close_prices: Sequence[Fixed],
    n_bars: int,
    mc_seed: int,
    config: TriageConfig,
) -> Test2Result:
    """Test 2: Monte Carlo date-shift.  Hard gate."""
    # Pre-flight: minimum trade count
    if len(oos_trades) < config.mc_min_baseline_trades:
        return Test2Result(
            passed=False,
            reason=f"INSUFFICIENT_TRADES_FOR_MC: {len(oos_trades)} < {config.mc_min_baseline_trades}",
            p_value=1_000_000, real_sharpe=0,
            randomized_sharpes=[], valid_iterations=0,
            total_iterations=config.mc_iterations,
        )

    # Real Sharpe
    try:
        real_returns = [t.gross_return_bps for t in oos_trades]
        real_sharpe_fixed = compute_sharpe_fixed(real_returns)
        real_sharpe = real_sharpe_fixed.value
    except (InsufficientDataError, ZeroVarianceError, NumericalError) as e:
        return Test2Result(
            passed=False, reason=f"Real Sharpe computation failed: {e}",
            p_value=1_000_000, real_sharpe=0,
            randomized_sharpes=[], valid_iterations=0,
            total_iterations=config.mc_iterations,
        )

    rng = np.random.RandomState(mc_seed)
    randomized_sharpes: List[int] = []
    valid_iterations = 0
    baseline_count = len(oos_trades)

    for _ in range(config.mc_iterations):
        # Generate random offset, skip k=0
        k = int(rng.randint(-config.mc_shift_range, config.mc_shift_range + 1))
        while k == 0:
            k = int(rng.randint(-config.mc_shift_range, config.mc_shift_range + 1))

        # Shift all trades
        shifted = []
        for t in oos_trades:
            repriced = _reprice_trade(t, k, close_prices, n_bars)
            if repriced is not None:
                shifted.append(repriced)

        # Check survival threshold
        survival_rate = len(shifted) / baseline_count
        if survival_rate < 0.5:
            continue

        # Compute Sharpe for shifted trades
        try:
            shifted_returns = [t.gross_return_bps for t in shifted]
            shifted_sharpe = compute_sharpe_fixed(shifted_returns)
            randomized_sharpes.append(shifted_sharpe.value)
            valid_iterations += 1
        except (InsufficientDataError, ZeroVarianceError, NumericalError):
            continue

    # Minimum valid iterations check
    if valid_iterations < config.mc_min_valid_iterations:
        return Test2Result(
            passed=False,
            reason=(
                f"MC_TEST_INVALID: only {valid_iterations}/{config.mc_iterations} "
                "iterations had 50% trade survival"
            ),
            p_value=1_000_000, real_sharpe=real_sharpe,
            randomized_sharpes=randomized_sharpes,
            valid_iterations=valid_iterations,
            total_iterations=config.mc_iterations,
        )

    # p-value: fraction of randomized Sharpes >= real Sharpe
    count_ge = sum(1 for s in randomized_sharpes if s >= real_sharpe)
    p_value_float = count_ge / valid_iterations
    p_value = int(p_value_float * 1_000_000)  # Scale to RATE

    # Pass criteria: real > 95th percentile AND p < 0.05
    percentile_95 = int(np.percentile(randomized_sharpes, 95))
    real_gt_95 = real_sharpe > percentile_95
    p_pass = p_value < config.mc_p_threshold

    if real_gt_95 and p_pass:
        passed = True
        reason = "PASS"
    else:
        reasons = []
        if not real_gt_95:
            reasons.append(
                f"real_sharpe={real_sharpe} <= p95={percentile_95}"
            )
        if not p_pass:
            reasons.append(f"p_value={p_value} >= threshold={config.mc_p_threshold}")
        reason = "; ".join(reasons)
        passed = False

    return Test2Result(
        passed=passed, reason=reason,
        p_value=p_value, real_sharpe=real_sharpe,
        randomized_sharpes=randomized_sharpes,
        valid_iterations=valid_iterations,
        total_iterations=config.mc_iterations,
    )


# ---------------------------------------------------------------------------
# Test 3: 3-Parameter Sensitivity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Test3Result:
    passed: bool
    reason: str
    pass_rate: int              # RATE scale
    num_passing: int
    total_variations: int
    variation_sharpes: List[int]  # RATE scale


def run_test_3(
    oos_trades: List[TradeEvent],
    close_prices: Sequence[Fixed],
    n_bars: int,
    metadata: StrategyMetadata,
    param_seed: int,
    config: TriageConfig,
    evaluate_fn: Callable[[Dict[str, int]], List[TradeEvent]],
) -> Test3Result:
    """Test 3: 3-parameter sensitivity.  Hard gate.

    evaluate_fn(params) -> trades: re-evaluate strategy with modified params
    on OOS data, returning trade events.
    """
    # Build parameter grid (3^3 = 27 variations)
    params_names = list(metadata.triage_sensitive_params)
    defaults = metadata.param_defaults
    bounds = metadata.param_bounds

    axes: List[List[int]] = []
    for pname in params_names:
        default_val = defaults[pname]
        variations = []
        for mult in config.param_multipliers:
            scaled = int(round(default_val * mult))
            lo, hi = bounds[pname]
            scaled = max(lo, min(hi, scaled))
            if default_val >= 1 and scaled < 1:
                scaled = 1
            variations.append(scaled)
        axes.append(sorted(set(variations)))

    grid = list(itertools.product(*axes))

    # Degenerate case: grid collapsed
    if len(grid) < 20:
        return Test3Result(
            passed=False,
            reason=f"PARAM_GRID_COLLAPSED: {len(grid)} < 20 unique variations",
            pass_rate=0, num_passing=0,
            total_variations=len(grid), variation_sharpes=[],
        )

    total_variations = len(grid)
    variation_sharpes: List[int] = []
    num_passing = 0

    for combo in grid:
        params = dict(defaults)  # Start from defaults
        for i, pname in enumerate(params_names):
            params[pname] = combo[i]

        try:
            variant_trades = evaluate_fn(params)
            if len(variant_trades) < 2:
                variation_sharpes.append(0)
                continue
            returns = [t.gross_return_bps for t in variant_trades]
            sharpe = compute_sharpe_fixed(returns)
            variation_sharpes.append(sharpe.value)
            if sharpe.value > config.param_pass_sharpe_min:
                num_passing += 1
        except (InsufficientDataError, ZeroVarianceError, NumericalError):
            variation_sharpes.append(0)

    # Pass criteria: pass_rate >= 0.6
    pass_rate_float = num_passing / total_variations if total_variations > 0 else 0.0
    pass_rate = int(pass_rate_float * 1_000_000)

    if pass_rate >= config.param_pass_rate_min:
        passed = True
        reason = "PASS"
    else:
        passed = False
        reason = (
            f"pass_rate={num_passing}/{total_variations} "
            f"({pass_rate}) < threshold={config.param_pass_rate_min}"
        )

    return Test3Result(
        passed=passed, reason=reason,
        pass_rate=pass_rate, num_passing=num_passing,
        total_variations=total_variations,
        variation_sharpes=variation_sharpes,
    )


# ---------------------------------------------------------------------------
# Test 4: Quick Correlation Check (soft gate)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Test4Result:
    warnings: List[str]
    correlation_drift: Optional[float]


def run_test_4(
    train_returns: List[int],
    oos_returns: List[int],
    config: TriageConfig,
) -> Test4Result:
    """Test 4: Correlation check.  Diagnostic only — never rejects."""
    warnings: List[str] = []

    if len(train_returns) < 5 or len(oos_returns) < 5:
        return Test4Result(
            warnings=["Insufficient data for correlation analysis"],
            correlation_drift=None,
        )

    # Compute mean returns for drift estimation
    train_mean = np.mean(train_returns)
    oos_mean = np.mean(oos_returns)
    train_std = np.std(train_returns, ddof=1) if len(train_returns) > 1 else 1.0
    oos_std = np.std(oos_returns, ddof=1) if len(oos_returns) > 1 else 1.0

    # Normalised drift = |train_mean/train_std - oos_mean/oos_std|
    train_norm = train_mean / train_std if train_std > 1e-10 else 0.0
    oos_norm = oos_mean / oos_std if oos_std > 1e-10 else 0.0
    drift = abs(train_norm - oos_norm)

    if drift > config.correlation_drift_threshold:
        warnings.append(
            f"Indicator correlation unstable between train/OOS: drift={drift:.4f}"
        )

    # Sign flip check
    if train_mean * oos_mean < 0:
        warnings.append("Indicator reversed direction in OOS period")

    for w in warnings:
        _log.warning(w)

    return Test4Result(warnings=warnings, correlation_drift=drift)


# ---------------------------------------------------------------------------
# Hash triage config
# ---------------------------------------------------------------------------

def hash_triage_config(config: TriageConfig) -> str:
    """Compute deterministic hash of TriageConfig."""
    d = {
        "train_fraction": str(config.train_fraction),
        "oos_sharpe_min": config.oos_sharpe_min,
        "oos_degradation_ratio": config.oos_degradation_ratio,
        "min_oos_bars": config.min_oos_bars,
        "min_oos_trades": config.min_oos_trades,
        "cost_per_trade_bps": config.cost_per_trade_bps,
        "mc_iterations": config.mc_iterations,
        "mc_p_threshold": config.mc_p_threshold,
        "mc_shift_range": config.mc_shift_range,
        "mc_survival_threshold": config.mc_survival_threshold,
        "mc_min_valid_iterations": config.mc_min_valid_iterations,
        "mc_min_baseline_trades": config.mc_min_baseline_trades,
        "param_multipliers": [str(m) for m in config.param_multipliers],
        "param_pass_sharpe_min": config.param_pass_sharpe_min,
        "param_pass_rate_min": config.param_pass_rate_min,
        "correlation_drift_threshold": str(config.correlation_drift_threshold),
    }
    js = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(js.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Full triage pipeline
# ---------------------------------------------------------------------------

def run_triage(
    strategy_id: str,
    trades: List[TradeEvent],
    close_prices: Sequence[Fixed],
    n_bars: int,
    metadata: StrategyMetadata,
    config: TriageConfig,
    dataset_hash: str,
    split_timestamp: str,
    evaluate_fn: Callable[[Dict[str, int]], List[TradeEvent]],
) -> TriageResult:
    """
    Run complete Tier 1 triage pipeline.

    Sequential with early-exit: Tests 1 → 1.5 → 2 → 3 → 4.
    """
    config_hash = hash_triage_config(config)
    test_results: Dict[str, Any] = {}

    # Seed cascade
    master_seed = derive_master_seed(
        strategy_id, dataset_hash, split_timestamp, config_hash
    )
    mc_seed = derive_subseed(master_seed, "monte_carlo")
    param_seed = derive_subseed(master_seed, "param_sweep")

    # Split
    split_idx = compute_split_index(n_bars, config.train_fraction)
    oos_trades = [t for t in trades if t.entry_idx >= split_idx]
    train_trades = [t for t in trades if t.exit_idx < split_idx]

    # ------------------------------------------------------------------
    # Test 1: OOS Holdout
    # ------------------------------------------------------------------
    _log.info("Running Test 1: OOS Holdout...")
    t1 = run_test_1(trades, n_bars, config)
    test_results["test_1"] = {
        "passed": t1.passed, "reason": t1.reason,
        "train_sharpe": t1.train_sharpe, "oos_sharpe": t1.oos_sharpe,
        "train_trades": t1.train_trades, "oos_trades": t1.oos_trades,
    }
    if not t1.passed:
        return TriageResult(
            strategy_id=strategy_id, dataset_hash=dataset_hash,
            config_hash=config_hash, passed=False,
            reason=f"Test 1 FAIL: {t1.reason}",
            test_results=test_results,
            train_sharpe=t1.train_sharpe, oos_sharpe=t1.oos_sharpe,
            mc_p_value=None, param_pass_rate=None,
            promotion_artifact=None,
        )

    # ------------------------------------------------------------------
    # Test 1.5: Cost Sanity (diagnostic only)
    # ------------------------------------------------------------------
    _log.info("Running Test 1.5: Cost Sanity...")
    t15 = run_test_15(oos_trades, config)
    test_results["test_1_5"] = {
        "cost_adjusted_sharpe": t15.cost_adjusted_sharpe,
        "warning": t15.warning,
    }
    if t15.warning:
        _log.warning("Test 1.5: %s", t15.warning)

    # ------------------------------------------------------------------
    # Test 2: Monte Carlo Date-Shift
    # ------------------------------------------------------------------
    _log.info("Running Test 2: Monte Carlo Date-Shift...")
    t2 = run_test_2(oos_trades, close_prices, n_bars, mc_seed, config)
    test_results["test_2"] = {
        "passed": t2.passed, "reason": t2.reason,
        "p_value": t2.p_value, "real_sharpe": t2.real_sharpe,
        "valid_iterations": t2.valid_iterations,
    }
    if not t2.passed:
        return TriageResult(
            strategy_id=strategy_id, dataset_hash=dataset_hash,
            config_hash=config_hash, passed=False,
            reason=f"Test 2 FAIL: {t2.reason}",
            test_results=test_results,
            train_sharpe=t1.train_sharpe, oos_sharpe=t1.oos_sharpe,
            mc_p_value=t2.p_value, param_pass_rate=None,
            promotion_artifact=None,
        )

    # ------------------------------------------------------------------
    # Test 3: Parameter Sensitivity
    # ------------------------------------------------------------------
    _log.info("Running Test 3: Parameter Sensitivity...")
    t3 = run_test_3(
        oos_trades, close_prices, n_bars, metadata, param_seed,
        config, evaluate_fn,
    )
    test_results["test_3"] = {
        "passed": t3.passed, "reason": t3.reason,
        "pass_rate": t3.pass_rate,
        "num_passing": t3.num_passing,
        "total_variations": t3.total_variations,
    }
    if not t3.passed:
        return TriageResult(
            strategy_id=strategy_id, dataset_hash=dataset_hash,
            config_hash=config_hash, passed=False,
            reason=f"Test 3 FAIL: {t3.reason}",
            test_results=test_results,
            train_sharpe=t1.train_sharpe, oos_sharpe=t1.oos_sharpe,
            mc_p_value=t2.p_value, param_pass_rate=t3.pass_rate,
            promotion_artifact=None,
        )

    # ------------------------------------------------------------------
    # Test 4: Correlation Check (soft gate)
    # ------------------------------------------------------------------
    _log.info("Running Test 4: Correlation Check...")
    train_returns = [t.gross_return_bps for t in train_trades]
    oos_returns = [t.gross_return_bps for t in oos_trades]
    t4 = run_test_4(train_returns, oos_returns, config)
    test_results["test_4"] = {
        "warnings": t4.warnings,
        "correlation_drift": t4.correlation_drift,
    }

    # ------------------------------------------------------------------
    # All tests passed — create promotion artifact
    # ------------------------------------------------------------------
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    run_id = f"triage_{now.replace(':', '').replace('-', '').replace('.', '_')}"

    sig_input = (
        f"{metadata.strategy_version_hash}|1|PASS|{now}|"
        f"{dataset_hash}|{config_hash}"
    )
    signature = hashlib.sha256(sig_input.encode("utf-8")).hexdigest()

    artifact = PromotionArtifact(
        strategy_id=strategy_id,
        strategy_version_hash=metadata.strategy_version_hash,
        tier=1,
        result="PASS",
        timestamp=now,
        triage_run_id=run_id,
        dataset_hash=dataset_hash,
        config_hash=config_hash,
        train_sharpe=t1.train_sharpe,
        oos_sharpe=t1.oos_sharpe,
        signature=signature,
    )

    return TriageResult(
        strategy_id=strategy_id, dataset_hash=dataset_hash,
        config_hash=config_hash, passed=True,
        reason="ALL TESTS PASSED",
        test_results=test_results,
        train_sharpe=t1.train_sharpe, oos_sharpe=t1.oos_sharpe,
        mc_p_value=t2.p_value, param_pass_rate=t3.pass_rate,
        promotion_artifact=artifact,
    )
