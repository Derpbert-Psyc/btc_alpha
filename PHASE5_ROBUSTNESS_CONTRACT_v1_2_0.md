# PHASE 5 ROBUSTNESS CONTRACT

## Research Validation Layer for BTC Alpha

**Version:** 1.2.0  
**Status:** AUTHORITATIVE  
**Authority Class:** Same as PHASE5_INTEGRATION_SEAMS.md

**PSEUDOCODE DISCLAIMER**: All code blocks in this document are illustrative pseudocode only and are not normative. Implementation details belong in implementation guides.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | TBD | Initial specification |
| 1.1.0 | TBD | Critical fixes: deterministic data hashing, split timestamp derivation, metadata pre-flight validation, MC trade survival thresholds, seed-history boundary discovery, no assumed dataset dates |
| 1.2.0 | TBD | Added: OS-agnostic runtime budget enforcement, split warmup rules, canonical trade event schema, promotion state machine, cost sanity check (Test 1.5), format-agnostic data hashing |

---

## 1. Purpose and Scope

This document defines the **robustness validation contract** for strategy candidates in the BTC Alpha Research Engine.

### 1.1 Design Philosophy

The robustness layer implements a **tiered validation pipeline**:

1. **Triage Filter** (2-3 minutes): Aggressive early rejection of overfit/noise
2. **Baseline-Plus Validation** (15-20 minutes): Comprehensive validation for promoted candidates
3. **Research-Grade Suite** (30-40 minutes): Pre-deployment validation
4. **Paranoid Academic Suite** (45-60 minutes): Publication/audit-grade validation

This document specifies the **Triage Filter** (Tier 1) and references the higher tiers.

### 1.2 Authority and Change Control

- This contract is **delegable for implementation** but **not for threshold tuning**
- Implementers build the test harness; system owner tunes acceptance criteria
- Changes to test definitions or thresholds require system owner approval per Section 8

---

## 2. Triage Filter (Tier 1): Ultra-Fast Plausibility Gate

### 2.1 Purpose

**Goal**: Reject noise/overfit in 2-3 minutes with near-zero false negatives while accepting moderate false positives.

**Mental Model**: Smoke detector, not fire investigation. Answer: "Is this plausibly real, or almost certainly noise?"

### 2.2 Target Specifications

| Metric | Target | Absolute Maximum |
|--------|--------|------------------|
| **Wall-clock runtime** | 2-3 minutes | 5 minutes |
| **Rejection rate** | 70-75% | N/A |
| **False negative rate** | <5% | 10% |
| **Early exit on failure** | Required | Required |

### 2.3 Test Suite Definition

The Triage Filter consists of **4 sequential tests** with early-exit semantics:

| # | Test Name | Runtime | Exit on Fail? |
|---|-----------|---------|---------------|
| 1 | Simple OOS Holdout | 30s | **Yes** |
| 2 | Monte Carlo Date-Shift | 90s | **Yes** |
| 3 | 3-Parameter Sensitivity | 60s | **Yes** |
| 4 | Quick Correlation Check | 20s | No (log only) |

**Total Expected Runtime**:
- Best case (fails Test 1): 30 seconds
- Worst case (passes all): 3.5 minutes
- Average case (50% fail early): ~90 seconds

### 2.4 Runtime Budget Enforcement (OS-Agnostic)

**Target Runtime**: 2-3 minutes  
**Maximum Runtime**: 5 minutes (hard deadline)

**Per-Test Budgets**:
- Test 1 (OOS Holdout + Cost Sanity): 40s target, 70s max
- Test 2 (Monte Carlo): 90s target, 180s max
- Test 3 (Parameter Sweep): 60s target, 120s max
- Test 4 (Correlation): 20s target, 40s max
- **Total**: 210s target, 410s max

**Enforcement Mechanism**: Cooperative deadline checking (OS-agnostic)

**Implementation Contract**:
Tests must accept a `check_deadline` callback and call it periodically:

```python
def run_test(strategy, check_deadline):
    """Test function with cooperative deadline checking."""
    for iteration in range(n):
        check_deadline()  # Check before expensive operation
        # ... perform test iteration
```

**On Deadline Exceeded**:
- Result: REJECT
- Reason: "TIME_BUDGET_EXCEEDED: {actual}s > {max}s"
- Partial results: Discarded (not used for pass/fail decision)
- No process kill, no OS-specific signals

**Rationale**: Cooperative checking works on all platforms (Windows, Linux, macOS).

---

## 3. Test 1: Simple OOS Holdout (30 seconds)

### 3.1 What It Does

- Split data 80/20 chronologically (train on first 80%, test on last 20%)
- Train strategy once, evaluate on holdout
- Compare in-sample (IS) vs out-of-sample (OOS) Sharpe ratio

### 3.2 What It Catches

**Primary Failure Mode**: Blatant in-sample overfitting (strategy memorizes training data)

**Why This Test**: Fastest possible temporal validation; empirically rejects 40-60% of naive strategies

### 3.2.1 Degenerate Case Handling (Pre-Flight Checks)

**Run BEFORE computing Sharpe ratios**:

```python
def validate_oos_data_quality(
    train_data: pd.DataFrame,
    oos_data: pd.DataFrame,
    train_trades: List[Trade],
    oos_trades: List[Trade]
) -> None:
    """
    Check for degenerate cases that invalidate Sharpe calculation.
    
    Raises: DataQualityError with diagnostic message
    """
    # Minimum data length
    if len(oos_data) < 100:
        raise DataQualityError(
            f"INSUFFICIENT_OOS_DATA: {len(oos_data)} bars, minimum 100 required "
            f"for statistical validity"
        )
    
    # Minimum trade count
    if len(oos_trades) < 10:
        raise DataQualityError(
            f"INSUFFICIENT_OOS_TRADES: {len(oos_trades)} trades, minimum 10 required"
        )
    
    # Check OOS trade count vs train (sanity check)
    if len(oos_trades) < 0.2 * len(train_trades):
        log.warning(
            f"OOS trade count ({len(oos_trades)}) is <20% of train count "
            f"({len(train_trades)}) - may indicate regime shift or overfitting"
        )
    
    # Compute return series
    train_returns = compute_returns(train_trades, train_data)
    oos_returns = compute_returns(oos_trades, oos_data)
    
    # Zero variance check (BEFORE Sharpe calculation)
    if np.std(oos_returns) < 1e-6:
        raise DataQualityError(
            f"ZERO_OOS_VARIANCE: std(returns) = {np.std(oos_returns):.2e}, "
            f"cannot compute meaningful Sharpe (all trades identical)"
        )
    
    if np.std(train_returns) < 1e-6:
        raise DataQualityError(
            f"ZERO_TRAIN_VARIANCE: std(returns) = {np.std(train_returns):.2e}, "
            f"strategy produces no variation in returns"
        )

def compute_sharpe_with_validation(returns: np.ndarray) -> float:
    """
    Compute Sharpe ratio with NaN/Inf protection.
    
    Returns: Finite Sharpe ratio
    Raises: NumericalError if result is NaN or Inf
    """
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        raise NumericalError(
            f"ZERO_VARIANCE: std(returns) = 0, Sharpe undefined"
        )
    
    sharpe = mean_return / std_return
    
    # Validate result is finite
    if not math.isfinite(sharpe):
        raise NumericalError(
            f"INVALID_SHARPE: {sharpe} (likely numerical instability), "
            f"mean={mean_return:.6f}, std={std_return:.6f}"
        )
    
    return sharpe
```

**Failure Behavior**:

- DataQualityError or NumericalError  REJECT with diagnostic message
- Never silently pass/fail on NaN or Inf
- Never compare NaN to thresholds

### 3.2.2 Split Boundary Warmup Rules

**Warmup Source**: Last N bars of train set (where N = strategy's max indicator lookback)

**Evaluation Window**: Entire test set (no warmup region carved out of test set)

**Algorithm**:
```python
def compute_oos_sharpe_with_warmup(
    data: pd.DataFrame,
    train_end_idx: int,
    test_start_idx: int,
    strategy: Strategy
) -> float:
    """
    Compute OOS Sharpe with explicit warmup handling.
    
    Warmup uses train data (no future leakage).
    Test set remains full size.
    """
    # Determine strategy's max lookback
    max_lookback = strategy.get_max_lookback()
    
    # Warmup region: last max_lookback bars of train set
    warmup_start_idx = train_end_idx - max_lookback
    warmup_end_idx = train_end_idx
    
    # Full evaluation data: warmup + test
    eval_data = data[warmup_start_idx:]
    
    # Run strategy (indicators warm up on [warmup_start:warmup_end])
    trades = strategy.run(eval_data)
    
    # Filter trades: only those with entry >= test_start_idx
    # (trades in warmup region are excluded)
    oos_trades = [
        t for t in trades
        if t.entry_idx >= (test_start_idx - warmup_start_idx)
    ]
    
    # Compute Sharpe on OOS trades only
    return compute_sharpe_with_validation(oos_trades)
```

**Insufficient Warmup Handling**:
```python
if max_lookback > train_end_idx:
    raise ValidationError(
        f"Insufficient train data for warmup: need {max_lookback} bars, "
        f"have {train_end_idx} bars"
    )
```

**Rationale**: 
- Warmup uses train data  no future leakage
- Test set remains full size  maximizes evaluation data
- Indicators reach steady state before test set begins

### 3.3 Pass Criteria (Hard Gate)

```python
PASS if BOTH conditions hold:
  1. oos_sharpe >= 0.3
  2. oos_sharpe >= 0.5 * train_sharpe
```

**Rationale**:
- `0.3` = barely above noise floor for crypto (Sharpe ~0.2 is random walk territory)
- `0.5` degradation = tolerate some overfitting but not collapse
- Both absolute AND relative checks required

### 3.4 Early Exit Behavior

 **If fails**  REJECT immediately, skip Tests 2-4, log reason

### 3.5 Example Outcomes

```
Strategy A: train_sharpe=1.2, oos_sharpe=0.1   FAIL (collapse, 0.08 ratio)
Strategy B: train_sharpe=0.6, oos_sharpe=0.25  FAIL (below 0.3 absolute threshold)
Strategy C: train_sharpe=0.8, oos_sharpe=0.45  PASS (0.56 ratio, above 0.3)
```

### 3.6 Split Timestamp Derivation (No Hardcoded Dates)

**CRITICAL**: All time bounds must be discovered from data content, never from filenames, comments, or assumed dates.

**Canonical Process**:

```python
def compute_split_timestamp(
    data: pd.DataFrame,
    split_ratio: float = 0.8
) -> Tuple[pd.Timestamp, int, int]:
    """
    Derive split timestamp from actual data bounds.
    
    Args:
        data: OHLCV data (already loaded and validated)
        split_ratio: Train fraction (e.g., 0.8 for 80/20 split)
    
    Returns:
        split_timestamp: The temporal boundary
        train_end_idx: Last row index of train set
        test_start_idx: First row index of test set
    """
    # Discover dataset bounds from data itself (never assume dates)
    dataset_start = data['timestamp'].min()
    dataset_end = data['timestamp'].max()
    dataset_duration = dataset_end - dataset_start
    
    # Compute split timestamp (temporal split, not row percentage)
    split_timestamp = dataset_start + (dataset_duration * split_ratio)
    
    # Find closest actual timestamp in data
    train_end_idx = (data['timestamp'] < split_timestamp).sum() - 1
    test_start_idx = train_end_idx + 1
    
    # Verify split is valid
    assert train_end_idx >= 100, f"Train set too small: {train_end_idx} rows"
    assert test_start_idx < len(data) - 100, f"Test set too small"
    
    # Log discovered split (not assumed)
    logger.info(
        f"Split derived from data: "
        f"train=[{dataset_start}, {data.iloc[train_end_idx]['timestamp']}], "
        f"test=[{data.iloc[test_start_idx]['timestamp']}, {dataset_end}]"
    )
    
    return split_timestamp, train_end_idx, test_start_idx
```

**Determinism Contract**:

A valid split requires THREE logged values for reproducibility:

1. **`split_timestamp`**: The temporal boundary (e.g., `"2024-10-21T07:12:00.000Z"`)
2. **`dataset_hash`**: Content hash of the exact dataset used
3. **`split_indices`**: Exact `(train_end_idx, test_start_idx)` computed from that dataset

**Replay Verification**:

```python
def verify_split_reproducibility(config: dict, data: pd.DataFrame) -> bool:
    """
    Verify split can be reproduced exactly.
    
    Returns: True if reproducible, raises error otherwise
    """
    # Verify dataset hasn't changed
    current_hash = canonical_data_hash(data)
    if current_hash != config['dataset_hash']:
        raise ReproducibilityError(
            f"DATASET_CHANGED: expected {config['dataset_hash']}, "
            f"got {current_hash}"
        )
    
    # Use logged indices (fast path, guaranteed exact)
    train_end_idx = config['split_indices']['train_end_idx']
    test_start_idx = config['split_indices']['test_start_idx']
    
    # Verify indices still match the logged timestamp
    actual_split_ts = data.iloc[test_start_idx]['timestamp']
    logged_split_ts = pd.Timestamp(config['split_timestamp'])
    
    assert actual_split_ts == logged_split_ts, "Split index mismatch"
    
    return True
```

**Logged Artifacts**:

```json
{
  "dataset_discovered_bounds": {
    "start": "2020-01-01T00:00:00.000Z",  // From min(timestamp)
    "end": "2025-12-31T23:59:00.000Z"     // From max(timestamp)
  },
  "split_derived": {
    "split_timestamp": "2024-10-21T07:12:00.000Z",  // Computed
    "split_ratio": 0.8,
    "train_end_idx": 2100000,
    "test_start_idx": 2100001,
    "train_date_range": ["2020-01-01T00:00:00.000Z", "2024-10-21T07:11:00.000Z"],
    "test_date_range": ["2024-10-21T07:12:00.000Z", "2025-12-31T23:59:00.000Z"]
  },
  "dataset_hash": "sha256:abc123...",
  "train_sharpe": 0.82,
  "oos_sharpe": 0.47,
  "cost_adjusted_sharpe": 0.41,
  "cost_sanity_pass": true
}
```

**Rationale**:
- Timestamp alone: breaks under prepend (dataset extended backward)
- Index alone: breaks under append (dataset extended forward)
- **All three together**: survives both prepend and append

### 3.7 Test 1.5: Cost Sanity Check (Diagnostic Only in Tier 1)

**Purpose**: Provide early signal for transaction cost sensitivity without blocking strategies at triage

**Tier 1 Principle**: Tier 1 exists to filter structural overfitting, not execution realism.

**What It Does**:
- Apply fixed per-trade cost model to OOS returns
- Log cost-adjusted Sharpe for future reference
- **Does NOT fail strategies** based on cost sensitivity in Tier 1

**What It Catches** (for Tier 2+ enforcement):
- Strategies that only work under zero-cost assumptions
- High-frequency churn that cannot overcome realistic friction

**Implementation** (runs AFTER Test 1 passes, shares same data):

```python
def cost_sanity_check(
    oos_trades: List[TradeEvent],
    base_oos_sharpe: float
) -> Tuple[float, bool]:
    """
    Apply fixed cost model, verify strategy survives.
    
    Returns: (cost_adjusted_sharpe, diagnostic_flag)
    """
    # Fixed cost model (conservative estimate)
    COST_PER_TRADE = 0.001  # 10 bps (0.1%) per round trip
    
    # Adjust gross returns
    cost_adjusted_returns = [
        trade.gross_return - COST_PER_TRADE
        for trade in oos_trades
    ]
    
    # Recompute Sharpe
    cost_adjusted_sharpe = compute_sharpe_from_returns(cost_adjusted_returns)
    
    # Diagnostic only - always returns True (no rejection)
    return cost_adjusted_sharpe, True
```

**Tier 1 Behavior** (Diagnostic Only):
- Cost-adjusted Sharpe is **logged** but does NOT cause rejection
- If cost-adjusted Sharpe < 0.2, log **WARNING** but still PASS Test 1
- Rationale: Triage goal is speed; full cost analysis belongs in Tier 2

**Tier 2 Enforcement** (Baseline-Plus):
- Cost-adjusted Sharpe must be  0.5  base Sharpe
- Cost-adjusted Sharpe must be  0.2 (absolute)
- If fails: strategy rejected at Tier 2 with reason "COST_SENSITIVITY"

**Runtime**: ~5-10 seconds (simple calculation on existing trades)

---

## 3.8 Canonical Trade Event Schema

**All robustness tests operate on a standardized trade representation** to ensure determinism and comparability.

**Required Fields**:
```python
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

@dataclass(frozen=True)
class TradeEvent:
    """
    Canonical trade event for robustness testing.
    
    All fields immutable and logged for audit.
    """
    # Identity
    trade_id: str  # Unique identifier
    
    # Timing (bar indices, not timestamps, for determinism)
    entry_idx: int  # Bar index where trade entered (0-indexed)
    exit_idx: int   # Bar index where trade exited
    
    # Direction
    side: Literal["long", "short"]
    
    # Execution prices (gross, before fees/slippage)
    entry_price: Decimal
    exit_price: Decimal
    
    # Position size
    qty: Decimal  # e.g., 1.0 BTC
    
    # Return (for Sharpe calculation)
    gross_return: Decimal  # (exit - entry) / entry for long
                           # (entry - exit) / entry for short
    
    def __post_init__(self):
        assert self.exit_idx > self.entry_idx, "Exit must be after entry"
        assert self.qty > 0, "Qty must be positive"
        assert self.side in ["long", "short"], f"Invalid side: {self.side}"
```

**Sharpe Calculation from Trade Events**:
```python
def compute_sharpe_from_returns(returns: List[Decimal]) -> float:
    """
    Compute Sharpe ratio from trade returns.
    
    Tier 1: Uses gross returns (no fees/slippage)
    Tier 2+: May use net returns (after costs)
    """
    if len(returns) == 0:
        raise InsufficientDataError("No trades to compute Sharpe")
    
    returns_float = [float(r) for r in returns]
    mean_return = np.mean(returns_float)
    std_return = np.std(returns_float, ddof=1)  # Sample std
    
    if std_return < 1e-10:
        raise ZeroVarianceError("Trade returns have zero variance")
    
    sharpe = mean_return / std_return
    
    if not math.isfinite(sharpe):
        raise NumericalError(f"Sharpe is {sharpe}, not finite")
    
    return sharpe
```

**Open Trades Handling**:
- If strategy has open trade at end of evaluation period: **EXCLUDE** from Sharpe
- Rationale: Cannot compute realized return on unrealized trade
- Note: May penalize strategies with long holding periods (acceptable for triage)

**Fees/Slippage**:
- **Tier 1 (Triage)**: Gross returns only (no transaction costs)
- **Tier 2+ (Baseline-Plus)**: Add transaction cost sensitivity tests
- Cost sanity check (Test 1.5) uses fixed 10 bps per-trade estimate

---

## 4. Test 2: Monte Carlo Date-Shift (90 seconds)

### 4.1 What It Does (Canonical Semantics)

**What is shifted**: Discrete trade event timestamps (entry and exit bar indices), NOT indicator outputs

**Algorithm**:

```python
def monte_carlo_date_shift(
    strategy_trades: List[Trade],  # From OOS evaluation
    oos_data: pd.DataFrame,
    n_iterations: int = 50,
    seed: int = None
) -> Tuple[float, List[float]]:
    """
    Test if strategy performance is due to timing luck.
    
    Args:
        strategy_trades: [(entry_idx, exit_idx, direction), ...]
        oos_data: Out-of-sample OHLCV data
        n_iterations: Number of random shifts (default 50)
        seed: Random seed for reproducibility
    
    Returns:
        p_value: Two-tailed statistical test result
        randomized_sharpes: Distribution of Sharpe ratios from shifted trades
    """
    rng = np.random.RandomState(seed)
    real_sharpe = compute_sharpe(strategy_trades, oos_data)
    randomized_sharpes = []
    
    for i in range(n_iterations):
        # Generate random shift: 5 to 30 bars
        k = rng.randint(-30, 31)
        while k == 0:  # Exclude zero shift
            k = rng.randint(-30, 31)
        
        shifted_trades = []
        for entry_idx, exit_idx, direction in strategy_trades:
            # Shift both entry and exit by same offset
            shifted_entry = entry_idx + k
            shifted_exit = exit_idx + k
            
            # Discard if out of bounds
            if shifted_entry < 0 or shifted_exit >= len(oos_data):
                continue
            
            # Keep trade with original direction at shifted timestamps
            shifted_trades.append((shifted_entry, shifted_exit, direction))
        
        # Check trade survival threshold (see Section 4.3.1)
        survival_rate = len(shifted_trades) / len(strategy_trades)
        if survival_rate < 0.5:
            # Iteration invalid, skip
            continue
        
        # Compute Sharpe for shifted trades
        sharpe_i = compute_sharpe(shifted_trades, oos_data)
        randomized_sharpes.append(sharpe_i)
    
    # Statistical test
    p_value = compute_permutation_pvalue(real_sharpe, randomized_sharpes)
    
    return p_value, randomized_sharpes
```

**Why this definition**:
- Preserves strategy logic (same entry/exit rules applied at different times)
- Breaks spurious timing correlations (if strategy only works at specific dates, it fails)
- Deterministic (seeded random offsets)
- Strategy-agnostic (works for any discrete signal generator)
- Does NOT shift indicators (would violate indicator warmup and dependency contracts)

### 4.2 What It Catches

**Primary Failure Mode**: Strategies that work purely by lucky alignment with price moves

**Why Date-Shift (not label-shuffling)**: 
- 2-3 faster (no indicator recomputation)
- Preserves autocorrelation structure (more realistic null)
- Breaks causal timing relationships

### 4.3 Pass Criteria (Hard Gate)

```python
PASS if BOTH conditions hold:
  1. real_sharpe > percentile_95(randomized_sharpes)
  2. p_value < 0.05  (two-tailed test)
```

**Rationale**:
- Real strategy must beat 95% of random timings
- p<0.05 ensures statistical significance at triage stage

### 4.3.1 Trade Survival Requirements

**Pre-Flight Checks** (run BEFORE Monte Carlo iterations):

```python
# Minimum baseline trade count
if len(oos_trades) < 30:
    return REJECT("INSUFFICIENT_TRADES_FOR_MC: {count} trades, minimum 30 required")

# Check if strategy has boundary issues
boundary_sensitive_count = sum(
    1 for entry, exit, _ in oos_trades
    if entry < 30 or exit >= len(oos_data) - 30
)
boundary_sensitivity = boundary_sensitive_count / len(oos_trades)

if boundary_sensitivity > 0.4:
    log.warning(
        f"Strategy has {boundary_sensitivity:.1%} trades near boundaries - "
        f"MC may have low valid iteration count"
    )
```

**Per-Iteration Survival Threshold**:
- Each MC iteration must retain 50% of baseline trade count
- If iteration has <50% survival  discard that iteration, continue with others
- Minimum valid iterations: 40 out of 50 (80% success rate)

**Test Invalidation**:

```python
valid_iterations = len([s for s in randomized_sharpes if s is not None])

if valid_iterations < 40:
    return REJECT(
        f"MC_TEST_INVALID: only {valid_iterations}/50 iterations had 50% "
        f"trade survival. Strategy may be too boundary-sensitive for MC validation."
    )
```

**Rationale**:
- 30 trades minimum: ensures meaningful statistical distribution
- 50% per-iteration survival: avoids comparing fundamentally different trade sets
- 40/50 valid iterations (80%): test is robust enough to be meaningful

### 4.4 Early Exit Behavior

 **If fails**  REJECT immediately, skip Tests 3-4, log reason and p-value

### 4.5 Example Outcomes

```
Real Sharpe: 0.65
Random distribution: [0.1, 0.3, ..., 0.72, 0.81]  (95th %ile = 0.70)
 FAIL: real doesn't exceed 95th percentile (lucky timing)

Real Sharpe: 0.55
Random distribution: [-0.1, 0.0, ..., 0.45, 0.52]  (95th %ile = 0.50)
 PASS: real clearly exceeds randomized baseline, p=0.003
```

### 4.6 Determinism Requirements (Collision-Resistant Seed Cascade)

**CRITICAL**: Seeds must be globally unique across all strategies and runs to prevent adversarial seed selection.

**Seed Derivation** (SHA256-based, domain-separated):

```python
import hashlib

def derive_master_seed(
    strategy_id: str,
    dataset_hash: str,
    split_timestamp: str,
    config_hash: str
) -> int:
    """
    Derive collision-resistant master seed.
    
    Args:
        strategy_id: Unique strategy identifier
        dataset_hash: SHA256 hash of canonical dataset
        split_timestamp: ISO 8601 timestamp of train/test split
        config_hash: SHA256 hash of triage configuration
    
    Returns:
        32-bit unsigned integer seed
    """
    # Concatenate inputs with delimiters
    seed_input = f"{strategy_id}|{dataset_hash}|{split_timestamp}|{config_hash}"
    
    # SHA256 hash
    h = hashlib.sha256(seed_input.encode('utf-8'))
    
    # Extract first 4 bytes as uint32
    master_seed = int.from_bytes(h.digest()[:4], byteorder='big')
    
    return master_seed

def derive_subseed(master_seed: int, domain: str) -> int:
    """
    Derive domain-separated sub-seed.
    
    Args:
        master_seed: Master seed from derive_master_seed()
        domain: Domain separator (e.g., "monte_carlo", "param_sweep")
    
    Returns:
        32-bit unsigned integer sub-seed
    """
    seed_bytes = master_seed.to_bytes(4, byteorder='big')
    domain_bytes = domain.encode('utf-8')
    
    # SHA256 hash with domain separation
    h = hashlib.sha256(seed_bytes + b"|" + domain_bytes)
    
    return int.from_bytes(h.digest()[:4], byteorder='big')
```

**Usage**:

```python
# At triage start
master_seed = derive_master_seed(
    strategy_id="momentum_hybrid_v3",
    dataset_hash=canonical_data_hash(data),
    split_timestamp="2024-10-21T07:12:00.000Z",
    config_hash=hash_triage_config(config)
)

# For Monte Carlo test
mc_seed = derive_subseed(master_seed, "monte_carlo")

# For parameter sweep
param_seed = derive_subseed(master_seed, "param_sweep")
```

**Logged Artifacts**:

```json
{
  "strategy_id": "momentum_hybrid_v3",
  "dataset_hash": "sha256:abc123...",
  "split_timestamp": "2024-10-21T07:12:00.000Z",
  "config_hash": "sha256:def456...",
  "master_seed": 2847561234,
  "mc_seed": 3928471923,
  "param_seed": 1847293847,
  "real_sharpe": 0.65,
  "randomized_sharpes": [...],
  "p_value": 0.003,
  "valid_iterations": 48
}
```

**Why SHA256-based derivation**:
- Collision-resistant for this use case
- Deterministic and reproducible across platforms
- No secret key management required
- Domain separation prevents seed reuse across contexts

### 4.7 Configuration Options

**Current (Option A)**: 50 runs, 90s, p<0.05  Conservative, lower false negative rate  
**Aggressive (Option B)**: 25 runs, 45s, p<0.10  Faster, slightly higher false negative risk  
**Paranoid (Option C)**: 100 runs, 120s, p<0.01  Slower, near-zero false negatives  

**Default**: Option A. System owner may tune after empirical validation.

---

## 5. Test 3: 3-Parameter Sensitivity (60 seconds)

### 5.1 What It Does

- Identify top 3 most influential strategy parameters (e.g., `lookback_period`, `threshold`, `stop_loss`)
- Test at 15% from optimal values
- Generate ~20 parameter combinations (coarse grid)
- Evaluate on OOS data only (reuse cached indicators from Test 1)

### 5.2 What It Catches

**Primary Failure Mode**: Knife-edge tuned strategies ("magic number" syndrome)

**Why Only 3 Parameters**: Time constraint; empirically top 3 account for 70-80% of variance

### 5.3 Parameter Grid Definition (Canonical - Exactly 27 Variations)

**Grid Structure** (locks combinatorial explosion):

```python
def generate_param_grid(
    optimal_params: Dict[str, float],
    triage_sensitive_params: List[str]
) -> List[Dict[str, float]]:
    """
    Generate exact 3^3 = 27 parameter variations.
    
    Args:
        optimal_params: Baseline parameter values
        triage_sensitive_params: Exactly 3 parameter names to vary
    
    Returns:
        List of 27 parameter dictionaries
    """
    assert len(triage_sensitive_params) == 3, "Must have exactly 3 params"
    
    # Multipliers: [0.85, 1.0, 1.15] for each param
    multipliers = [0.85, 1.0, 1.15]
    
    variations = []
    for m1 in multipliers:
        for m2 in multipliers:
            for m3 in multipliers:
                variant = optimal_params.copy()
                
                # Apply multipliers to sensitive params
                variant[triage_sensitive_params[0]] = _scale_param(
                    optimal_params[triage_sensitive_params[0]], m1
                )
                variant[triage_sensitive_params[1]] = _scale_param(
                    optimal_params[triage_sensitive_params[1]], m2
                )
                variant[triage_sensitive_params[2]] = _scale_param(
                    optimal_params[triage_sensitive_params[2]], m3
                )
                
                variations.append(variant)
    
    assert len(variations) == 27, "Grid must contain exactly 27 variations"
    return variations

def _scale_param(value: float, multiplier: float) -> float:
    """
    Scale parameter with discrete constraints.
    
    Handles integer params, minimum bounds, etc.
    """
    scaled = value * multiplier
    
    # If original was integer, round scaled value
    if isinstance(value, int) or value == int(value):
        scaled = int(round(scaled))
    
    # Ensure minimum value (e.g., lookback >= 1)
    if scaled < 1 and value >= 1:
        scaled = 1
    
    return scaled
```

**Degenerate Case Handling**:

```python
# Check for duplicate variations after rounding
unique_variations = {frozenset(v.items()) for v in variations}

if len(unique_variations) < 20:  # Significant collapse
    return REJECT(
        f"PARAM_GRID_COLLAPSED: parameter values too small, "
        f"rounding produced only {len(unique_variations)} unique variations "
        f"(expected 27). Increase optimal parameter values."
    )
```

**Runtime Guarantee**: 27 variations  ~2s = 54 seconds (fits in 60s budget)

### 5.4 Pass Criteria (Hard Gate)

```python
pass_rate = (num_variations_with_sharpe > 0.2) / 27
PASS if pass_rate >= 0.6  # At least 17 out of 27 variations survive
```

**Rationale**:
- Threshold `0.2` is extremely lenient (just "not terrible")
- But `60%` pass rate (17/27 variations) ensures strategy isn't hyper-sensitive
- Allows degradation while filtering extreme brittleness

### 5.5 Early Exit Behavior

 **If fails**  REJECT immediately, skip Test 4, log pass_rate and variation grid

### 5.6 Example Outcomes

```
27 parameter variations tested:
- 11 variations: Sharpe > 0.2  pass_rate = 41% (11/27)  FAIL
- 20 variations: Sharpe > 0.2  pass_rate = 74% (20/27)  PASS
```

### 5.7 Determinism Requirements

- Parameter grid must be **explicitly computed** from logged optimal values
- Grid generation is **deterministic** (always same 27 variations for same inputs)
- Indicators are **cached** from Test 1 (no recomputation)
- Logged artifacts: `param_grid.csv` (all 27 variations + outcomes), `pass_rate`

### 5.8 Strategy Metadata Requirement (Pre-Flight Validation)

**CRITICAL**: Strategies MUST declare `triage_sensitive_params` before triage runs.

**Required Metadata** (enforced at triage start, not during test):

```python
class StrategyMetadata:
    name: str
    version: str  # Semantic versioning (e.g., "1.2.3")
    triage_sensitive_params: List[str]  # Exactly 3 parameter names
    param_defaults: Dict[str, Any]      # Default/optimal values
    param_bounds: Dict[str, Tuple[Any, Any]]  # (min, max) for each param
```

**Pre-Flight Validation** (before ANY triage tests run):

```python
def validate_strategy_metadata(strategy: Strategy) -> None:
    """
    Validate metadata before triage. Raises exception if invalid.
    
    Raises: MetadataValidationError with diagnostic message
    """
    metadata = strategy.metadata
    
    # Check triage_sensitive_params exists
    if not hasattr(metadata, 'triage_sensitive_params'):
        raise MetadataValidationError(
            f"Strategy '{strategy.name}': MISSING_METADATA - "
            f"triage_sensitive_params required (list of 3 parameter names)"
        )
    
    # Check exactly 3 params
    if len(metadata.triage_sensitive_params) != 3:
        raise MetadataValidationError(
            f"Strategy '{strategy.name}': INVALID_METADATA - "
            f"triage_sensitive_params must contain exactly 3 parameter names, "
            f"got {len(metadata.triage_sensitive_params)}"
        )
    
    # Check params exist in strategy
    for param_name in metadata.triage_sensitive_params:
        if param_name not in metadata.param_defaults:
            raise MetadataValidationError(
                f"Strategy '{strategy.name}': INVALID_METADATA - "
                f"parameter '{param_name}' declared in triage_sensitive_params "
                f"but not found in param_defaults"
            )
        
        if param_name not in metadata.param_bounds:
            raise MetadataValidationError(
                f"Strategy '{strategy.name}': INVALID_METADATA - "
                f"parameter '{param_name}' declared in triage_sensitive_params "
                f"but not found in param_bounds"
            )
```

**Batch Processing** (Research/Triage Only):

```python
def run_triage_batch(strategies: List[Strategy]) -> List[TriageResult]:
    """
    Run triage on multiple strategies.
    
    Metadata validation errors cause affected strategies to be REJECTED.
    Batch continues for all valid strategies.
    """
    results = []
    metadata_errors = []
    
    # Process each strategy independently
    for strategy in strategies:
        try:
            # Validate metadata
            validate_strategy_metadata(strategy)
            
            # Run triage if valid
            result = run_triage_filter(strategy)
            results.append(result)
            
        except MetadataValidationError as e:
            # REJECT this strategy only, continue batch
            metadata_errors.append((strategy.name, str(e)))
            results.append(TriageResult(
                strategy_id=strategy.id,
                decision="REJECT",
                reason=f"METADATA_ERROR: {str(e)}"
            ))
            log.error(f"Strategy {strategy.name} rejected: {e}")
    
    # Emit consolidated error report at batch end
    if metadata_errors:
        error_report = "Metadata validation errors:\n" + "\n".join(
            f"  - {name}: {error}" for name, error in metadata_errors
        )
        log.warning(f"\n{error_report}\n")
    
    return results
```

**Behavior**:
- Metadata validation errors cause the affected strategy to be REJECTED and excluded from triage
- Batch execution continues for all other valid strategies
- All metadata errors are reported together at batch end
- Researcher receives: (1) triage results for valid strategies, (2) rejection list for invalid metadata

**Rationale**:
- Maintains high research velocity (don't block entire batch for one typo)
- Errors are visible in consolidated report (not silent)
- Allows iteration on multiple strategies in parallel

**Note**: This behavior applies to **research/triage only**. Live systems use stricter validation gates.

---

## 6. Test 4: Quick Correlation Check (20 seconds, diagnostic only)

### 6.1 What It Does

- Compute Spearman correlation between:
  - Key indicators and returns (train vs OOS)
  - Indicator pairs (train vs OOS)
- Flag if correlations flip sign or drift > 0.4

### 6.2 What It Catches

**Primary Failure Mode**: Feature regime shifts (indicator loses predictive power in OOS)

**Why Diagnostic Only**: 
- Some strategies are regime-specific by design
- Hard to set universal thresholds quickly
- High false positive rate for hard gates

### 6.3 Pass Criteria (Soft Gate - Log Warnings Only)

```python
# No hard rejection, but log warnings:
if correlation_drift > 0.4:
    log.warning("Indicator correlation unstable between train/OOS")
if sign_flip_detected:
    log.warning("Indicator reversed direction in OOS period")
```

### 6.4 No Early Exit

 Test always runs if Tests 1-3 pass  
 Warnings logged but do not block promotion

### 6.5 Determinism Requirements

- Logged artifacts: `correlation_drift.json` (train vs OOS correlations, drift magnitudes)

---

## 7. Risks Intentionally Left Uncovered (By Design)

The Triage Filter is a **plausibility filter**, not a comprehensive validator. The following risks are **explicitly accepted**:

| Risk Category | Why Not Covered | Mitigation Path |
|--------------|-----------------|-----------------|
| **Temporal overfitting across regimes** | No walk-forward; single train/test split | Promote to Baseline-Plus (3-fold WF) |
| **Transaction cost sensitivity** | No slippage/fee modeling | Add in Baseline-Plus |
| **Multi-regime stress testing** | No bear market / crash scenarios | Suite 2+ for promoted strategies |
| **Cross-exchange generalization** | Single data source | Research-Grade suite for finalists |
| **Multiple hypothesis correction** | No adjustment for # strategies tested | Audit finalists with Suite 4 |
| **Subtle look-ahead bias** | Feature engineering not validated | Code review + later validation |
| **Long-term regime shifts** | Only tests last 20% of data | Accept; real edges may be time-limited |

**Estimated False Positive Rate**: 10-20% of strategies passing Triage are still overfit  
**Acceptable**: This is a triage filter; Baseline-Plus catches remaining overfit

**Estimated False Negative Rate**: <5% of truly robust strategies wrongly rejected  
**Acceptable**: Conservative thresholds minimize this; manual override available for strong theoretical justification

---

## 8. Determinism and Reproducibility Contract

### 8.1 Seed Cascade and Canonical Data Hashing

**CRITICAL**: All stochastic procedures must use collision-resistant, deterministic seeds derived from canonical inputs.

**Seed Derivation** (SHA256-based, collision-resistant, no secret key):

```python
# Master seed derived from:
master_seed = derive_master_seed(
    strategy_id="momentum_hybrid_v3",
    dataset_hash=canonical_data_hash(data),  # See below
    triage_timestamp="2026-02-08T12:34:56.789Z"
)

# Sub-seeds for each test
mc_seed = derive_subseed(master_seed, "monte_carlo")
param_seed = derive_subseed(master_seed, "param_sweep")
```

**Canonical Data Hashing** (Deterministic Across Platforms):

```python
import hashlib
import numpy as np

def canonical_data_hash(df: pd.DataFrame) -> str:
    """
    Hash OHLCV data deterministically across Python/Pandas versions.
    
    CRITICAL: This is the ONLY approved data hashing implementation.
    Do NOT substitute alternative algorithms without system owner approval.
    
    Args:
        df: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
    
    Returns:
        SHA256 hex digest of canonical data representation
    """
    # Sort by timestamp (deterministic ordering)
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    # Extract canonical columns as numpy arrays
    timestamps = df_sorted['timestamp'].astype('int64').values  # Unix epoch ms
    opens = df_sorted['open'].values.astype('float64')
    highs = df_sorted['high'].values.astype('float64')
    lows = df_sorted['low'].values.astype('float64')
    closes = df_sorted['close'].values.astype('float64')
    volumes = df_sorted['volume'].values.astype('float64')
    
    # Convert to fixed-precision integers (avoids float representation issues)
    # 8 decimal places for prices, 2 for volume
    opens_int = (opens * 1e8).astype('int64')
    highs_int = (highs * 1e8).astype('int64')
    lows_int = (lows * 1e8).astype('int64')
    closes_int = (closes * 1e8).astype('int64')
    volumes_int = (volumes * 1e2).astype('int64')
    
    # Concatenate all as raw bytes (no CSV/JSON formatting)
    data_bytes = b''.join([
        timestamps.tobytes(),
        opens_int.tobytes(),
        highs_int.tobytes(),
        lows_int.tobytes(),
        closes_int.tobytes(),
        volumes_int.tobytes()
    ])
    
    # SHA256 hash
    return hashlib.sha256(data_bytes).hexdigest()
```

**Why this algorithm**:
- No CSV/JSON formatting  no pandas/python version dependency
- No float-to-string conversion  no trailing zero issues
- Fixed precision integers  deterministic across platforms
- Raw bytes  no encoding ambiguity
- Survives file format changes (Parquet metadata, compression)

**Test Vector** (Mandatory Validation):

```python
# Implementers MUST produce this exact hash for this test data
test_data = pd.DataFrame({
    'timestamp': [1609459200000, 1609459260000, 1609459320000],  # 2021-01-01 00:00:00 UTC
    'open':  [29000.00, 29100.50, 29050.25],
    'high':  [29150.00, 29200.00, 29175.75],
    'low':   [28950.00, 29050.00, 29000.50],
    'close': [29100.00, 29080.00, 29150.00],
    'volume': [123.45, 234.56, 345.67]
})

# System owner must provide expected hash after canonical implementation
EXPECTED_HASH = "SYSTEM_OWNER_PROVIDED_EXPECTED_HASH"

assert canonical_data_hash(test_data) == EXPECTED_HASH, "Hash implementation mismatch"
```

**Note**: The expected hash value will be computed and frozen by system owner after canonical implementation is validated.

**Config Structure**:

```python
robustness_config = {
    "strategy_id": "momentum_hybrid_v3",
    "dataset_hash": "sha256:abc123...",  # From canonical_data_hash()
    "triage_timestamp": "2026-02-08T12:34:56.789Z",
    "master_seed": 2847561234,  # Derived via SHA256
    "mc_seed": 3928471923,
    "param_seed": 1847293847,
    "split_timestamp": "2024-10-21T07:12:00.000Z",  # Derived from data
    "split_indices": {"train_end_idx": 2100000, "test_start_idx": 2100001}
}
```

### 8.2 Logged Artifacts (Per Test Run)

**Mandatory artifacts** for audit and replay:

```
triage_results/<run_id>/
 config.json                    # Seeds, thresholds, split dates
 data_hash.txt                  # SHA256 of input OHLCV data
 test1_oos_metrics.json         # train_sharpe, oos_sharpe, split_date
 test2_mc_distribution.npy      # 50 randomized Sharpes + p-value
 test3_param_grid.csv           # 20 variations + pass/fail
 test4_correlation_drift.json   # Diagnostic warnings
 decision.txt                   # PASS/FAIL + reason
```

**Storage**: ~500 KB per run (lightweight)

### 8.3 Reproducibility Verification

Implementers must provide:

```bash
# Reproduce any historical run
$ btc_alpha_research verify-triage \
    --run-id <run_id> \
    --config triage_results/<run_id>/config.json \
    --verify-only  # Check results match, don't re-run
```

Exit codes:
- `0`: Results match exactly
- `1`: Results diverge (reproducibility violation)
- `2`: Missing artifacts (audit trail incomplete)

---

## 9. Integration with Research Workflow

### 9.1 Usage Pattern

```python
def research_iteration(strategy):
    # Step 1: Baseline backtest (outside scope of this contract)
    baseline = run_baseline_backtest(strategy)
    
    if baseline.sharpe < 0.5:
        return "REJECT: Baseline too weak"
    
    # Step 2: Triage Filter (THIS CONTRACT)
    triage = run_triage_filter(strategy, max_time=180)  # 3 min timeout
    
    if triage.decision == "FAIL":
        log_rejection(strategy, triage.reason)
        return f"REJECT: {triage.reason}"
    
    # Step 3: Promote to Baseline-Plus validation
    log_promotion(strategy, "Triage passed  Queue for Baseline-Plus")
    schedule_full_validation(strategy, suite="Baseline-Plus")
    
    return "PASS: Ready for full validation"
```

### 9.2 Expected Iteration Speed

- **10 candidates/hour** (6 min each: 3 min backtest + 3 min triage)
- **~70% rejected at triage**  ~3 survivors promoted
- **3 survivors  20 min Baseline-Plus**  1 hour validation batch
- **Total**: Test 10 ideas in 2 hours (vs 4+ hours without triage)

### 9.3 Manual Override Path

If a strategy fails Triage but has strong theoretical justification:

```bash
$ btc_alpha_research override-triage \
    --strategy-id <id> \
    --reason "Strong theoretical foundation: XYZ" \
    --promote-to baseline-plus \
    --require-approval  # System owner approval required
```

### 9.4 Promotion State Machine and Enforcement

**Strategy Lifecycle States**:
```python
class StrategyState(Enum):
    UNTESTED = "untested"                    # Never run through triage
    TRIAGE_FAILED = "triage_failed"          # Failed Tier 1
    TRIAGE_PASSED = "triage_passed"          # Passed Tier 1, eligible for Tier 2
    BASELINE_PLUS_FAILED = "baseline_failed" # Failed Tier 2
    BASELINE_PLUS_PASSED = "baseline_passed" # Passed Tier 2, eligible for Tier 3
    # ... (higher tiers)
```

**Transition Rules**:
1. UNTESTED  TRIAGE_PASSED: Only via successful Tier 1 run
2. TRIAGE_PASSED  BASELINE_PLUS: Manual promotion by system owner
3. **No skipping tiers**: Cannot run Tier 2 without Tier 1 PASS artifact

**Promotion Artifact** (immutable, logged):
```json
{
  "strategy_id": "momentum_hybrid_v3",
  "strategy_version_hash": "sha256:abc123...",
  "tier": 1,
  "result": "PASS",
  "timestamp": "2026-02-09T12:34:56.789Z",
  "triage_run_id": "triage_20260209_123456",
  "dataset_hash": "sha256:def456...",
  "config_hash": "sha256:ghi789...",
  "artifacts_path": "/path/to/triage_results/triage_20260209_123456/",
  "signature": "sha256(strategy_version_hash + tier + result + timestamp + ...)"
}
```

**Single-Run Enforcement** (Anti-Pattern: Re-Running with Different Seeds):

Each strategy version may run triage ONCE per dataset version:
```python
def run_triage_filter(strategy: Strategy, dataset: Dataset) -> TriageResult:
    """Run triage with single-run enforcement."""
    
    # Check if already run
    strategy_hash = compute_strategy_hash(strategy)
    dataset_hash = canonical_data_hash(dataset)
    
    existing_run = find_triage_run(strategy_hash, dataset_hash)
    
    if existing_run:
        raise ValidationError(
            f"Strategy {strategy.id} already ran triage on this dataset. "
            f"Result: {existing_run.result}. "
            f"To re-run, modify strategy (new version hash) or use different dataset."
        )
    
    # Proceed with single triage run
    # ...
```

**Tier 2 Pre-Flight Check** (Enforce Tier 1 PASS):
```python
def run_baseline_plus(strategy: Strategy) -> None:
    """
    Run Tier 2 validation (Baseline-Plus).
    
    Requires: Valid Tier 1 PASS artifact
    """
    # Load promotion artifact
    artifact = load_promotion_artifact(strategy.id, tier=1)
    
    if not artifact:
        raise ValidationError(
            f"Strategy {strategy.id} has no Tier 1 PASS artifact. "
            f"Run triage first."
        )
    
    # Verify strategy hasn't changed since Tier 1
    current_hash = compute_strategy_hash(strategy)
    if current_hash != artifact['strategy_version_hash']:
        raise ValidationError(
            f"Strategy {strategy.id} modified since Tier 1 pass. "
            f"Tier 1 hash: {artifact['strategy_version_hash']}, "
            f"current hash: {current_hash}. "
            f"Re-run triage on modified strategy."
        )
    
    # Proceed with Tier 2 validation
    # ...
```

---

## 10. Delegation Contract for Implementers

### 10.1 What Is Delegable (Infrastructure)

Implementers (Claude Code, Ralph) are **authorized to build**:

 Test harness infrastructure (Test 1-4 execution logic)  
 Seed management and deterministic RNG  
 Indicator caching and reuse logic  
 Artifact logging and storage  
 Reproducibility verification tools  
 Parallelization and early-exit optimizations  

### 10.2 What Is NOT Delegable (Thresholds)

Implementers **may not tune** without system owner approval:

 Pass/fail thresholds (e.g., `oos_sharpe >= 0.3`, `p < 0.05`)  
 Test selection (adding/removing tests from the suite)  
 Early-exit logic (changing which tests trigger rejection)  
 Correlation drift thresholds (Test 4 warning levels)  

### 10.3 Framing for Delegation

**Handoff instruction to implementers**:

> "Implement the Triage Filter test harness per PHASE5_ROBUSTNESS_CONTRACT.md.  
> Input: strategy object + OHLCV data  
> Output: PASS/FAIL decision + logged artifacts  
> Hard constraints: 3-minute max runtime, deterministic, early-exit on failure  
> Verification: System owner will provide 3 test strategies (known overfit, known robust, borderline) for validation."

### 10.4 Threshold Tuning Process

After initial implementation:

1. Implementers run Triage Filter on **10 known-good strategies** (system owner provides)
2. Log false negative rate (how many good strategies were rejected)
3. Run on **10 known-bad strategies** (overfit/noise)
4. Log false positive rate (how many bad strategies passed)
5. Submit tuning report to system owner with empirical data
6. System owner adjusts thresholds if needed (explicit approval required)

---

## 11. Higher-Tier Validation Suites (Reference)

Strategies that **pass Triage** are promoted to higher tiers for comprehensive validation.

### 11.1 Baseline-Plus Validation (Tier 2, 15-20 minutes)

**Purpose**: Production-viable validation for promoted strategies

**Tests**:
- Walk-Forward (3 folds, 60/20/20 train/val/test)
- Parameter Sensitivity (10%, 20% on top 5 params)
- Stress Test: Longest Drawdown Period (algorithmic detection, 180 days if available)
- Monte Carlo Permutation (200 runs, label shuffling)
- Correlation Stability Check

**Stress Test Details**:
- Identify worst drawdown period in available historical data
- Duration: 180 days (if dataset contains such periods)
- Test criteria: max DD < 35%, recovery within 180 days
- Log period: `{dd_start_timestamp, dd_end_timestamp, dd_magnitude}`
- If no 180-day drawdown exists: skip test, log WARNING

**Pass Criteria**: See full Baseline-Plus specification (separate document)

### 11.2 Research-Grade Suite (Tier 3, 30-40 minutes)

**Purpose**: Pre-deployment validation for live candidates

**Adds**:
- 5-fold Walk-Forward
- Multi-Regime Stress Tests: Detected regime shifts in available data
  - Algorithmically identify: high volatility periods, rapid drawdowns (>30% in <90 days), low volatility grinds
  - Test strategy across 3 distinct detected regimes
  - Log regime boundaries: `{regime_id, start_ts, end_ts, characteristics}`
  - If <3 regimes detectable: reduce requirement to available count, log WARNING
- Transaction Cost Sensitivity (0.5, 1, 2 slippage)
- Monte Carlo Bootstrap (500 runs)
- Cross-Exchange Validation (train Coinbase, test Binance)

### 11.3 Paranoid Academic Suite (Tier 4, 45-60 minutes)

**Purpose**: Publication/audit-grade validation for finalists

**Adds**:
- 10-fold Nested Cross-Validation
- White Reality Check / SPA Test (multiple hypothesis correction)
- Synthetic Data Validation (GAN-generated price paths)
- Live-Delayed OOS (30-day embargo holdout)

---

## 12. Change Control

Any modification to:

- Test definitions (Tests 1-4)
- Pass/fail thresholds
- Seed cascade logic
- Artifact logging requirements
- This document

Requires:

1. Explicit written proposal with rationale
2. Empirical data supporting the change (false positive/negative rates)
3. Explicit review by system owner
4. Explicit written approval by system owner
5. Version increment of this document

**Emergency Changes**: Not applicable. Robustness validation is a research-time process; no emergency modifications are ever justified.

---

## 13. Document Status

This document is **authoritative** for Tier 1 (Triage Filter) validation.

**Version**: 1.0.0  
**Status**: APPROVED  
**Next Review**: After 100 strategy evaluations (empirical validation of thresholds)

---

## End of PHASE5_ROBUSTNESS_CONTRACT.md
