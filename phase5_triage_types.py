"""
Phase 5 — Triage Filter Types

TradeEvent, TriageConfig, TriageResult, PromotionArtifact,
StrategyMetadata, StrategyState — all Fixed/int based.

References:
    PHASE5_ROBUSTNESS_CONTRACT_v1_2_0.md — Sections 3-9
    SYSTEM_LAWS.md — Law I.1 (integer authority)
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from btc_alpha_v3_final import Fixed, SemanticType


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InsufficientDataError(Exception):
    """Raised when data is insufficient for a statistical test."""


class ZeroVarianceError(Exception):
    """Raised when standard deviation is below threshold."""


class NumericalError(Exception):
    """Raised when a numerical result is non-finite."""


class TriageRejectionError(Exception):
    """Raised when a triage test rejects a strategy."""

    def __init__(self, reason: str, test_name: str):
        self.reason = reason
        self.test_name = test_name
        super().__init__(f"{test_name}: {reason}")


# ---------------------------------------------------------------------------
# Strategy state machine
# ---------------------------------------------------------------------------

class StrategyState(Enum):
    UNTESTED = "untested"
    TRIAGE_FAILED = "triage_failed"
    TRIAGE_PASSED = "triage_passed"
    BASELINE_PLUS_FAILED = "baseline_failed"
    BASELINE_PLUS_PASSED = "baseline_passed"


# ---------------------------------------------------------------------------
# TradeEvent (Fixed-based)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeEvent:
    """Single trade for triage evaluation.  All monetary values Fixed or int."""
    trade_id: str
    entry_idx: int
    exit_idx: int
    side: Literal["long", "short"]
    entry_price: Fixed          # PRICE
    exit_price: Fixed           # PRICE
    qty: Fixed                  # QTY
    gross_return_bps: int       # Integer basis points.  100 = 1%.

    def __post_init__(self):
        assert self.exit_idx > self.entry_idx, (
            f"exit_idx ({self.exit_idx}) must be > entry_idx ({self.entry_idx})"
        )
        assert self.qty.value > 0, "qty must be positive"
        assert self.side in ("long", "short"), f"invalid side: {self.side}"


# ---------------------------------------------------------------------------
# TriageConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TriageConfig:
    """Configuration for Tier 1 triage filter."""
    # Test 1 — OOS holdout
    train_fraction: float = 0.8       # 80/20 split
    oos_sharpe_min: int = 300_000     # 0.3 in RATE scale (scale=6)
    oos_degradation_ratio: int = 500_000  # 0.5 in RATE scale
    min_oos_bars: int = 100
    min_oos_trades: int = 10

    # Test 1.5 — Cost sanity (diagnostic only)
    cost_per_trade_bps: int = 10      # 10 bps fixed cost

    # Test 2 — Monte Carlo
    mc_iterations: int = 50
    mc_p_threshold: int = 50_000      # 0.05 in RATE scale (scale=6)
    mc_shift_range: int = 30          # -30 to +30
    mc_survival_threshold: int = 500_000  # 0.5 in RATE scale
    mc_min_valid_iterations: int = 40
    mc_min_baseline_trades: int = 30

    # Test 3 — Parameter sensitivity
    param_multipliers: Tuple[float, ...] = (0.85, 1.0, 1.15)
    param_pass_sharpe_min: int = 200_000  # 0.2 in RATE scale
    param_pass_rate_min: int = 600_000    # 0.6 in RATE scale

    # Test 4 — Correlation check (soft gate)
    correlation_drift_threshold: float = 0.4


# ---------------------------------------------------------------------------
# TriageResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TriageResult:
    """Result from a complete Tier 1 triage run."""
    strategy_id: str
    dataset_hash: str
    config_hash: str
    passed: bool
    reason: str
    test_results: Dict[str, Any]     # Per-test outcomes
    train_sharpe: int                # Scaled int (RATE scale=6)
    oos_sharpe: int                  # Scaled int
    mc_p_value: Optional[int]        # Scaled int or None if not reached
    param_pass_rate: Optional[int]   # Scaled int or None if not reached
    promotion_artifact: Optional[PromotionArtifact]


# ---------------------------------------------------------------------------
# PromotionArtifact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromotionArtifact:
    """Immutable promotion record.  All numeric fields int."""
    strategy_id: str
    strategy_version_hash: str
    tier: int
    result: Literal["PASS", "FAIL"]
    timestamp: str                    # ISO 8601
    triage_run_id: str
    dataset_hash: str
    config_hash: str
    train_sharpe: int
    oos_sharpe: int
    signature: str                    # SHA256 of canonical fields


# ---------------------------------------------------------------------------
# StrategyMetadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyMetadata:
    """Metadata a strategy must declare for triage."""
    strategy_id: str
    strategy_version_hash: str
    param_defaults: Dict[str, int]
    param_bounds: Dict[str, Tuple[int, int]]
    triage_sensitive_params: Tuple[str, str, str]    # Exactly 3

    def __post_init__(self):
        assert len(self.triage_sensitive_params) == 3, (
            f"Exactly 3 sensitive params required, got {len(self.triage_sensitive_params)}"
        )
        for p in self.triage_sensitive_params:
            assert p in self.param_defaults, f"Param '{p}' not in param_defaults"
            assert p in self.param_bounds, f"Param '{p}' not in param_bounds"


# ---------------------------------------------------------------------------
# Sharpe computation — pinned numeric rules
# ---------------------------------------------------------------------------

def compute_sharpe_fixed(returns_bps: List[int]) -> Fixed:
    """
    Compute Sharpe ratio from integer basis-point returns.

    PINNED RULES:
        - ddof=1 (sample standard deviation, Bessel's correction)
        - NOT annualized (per-trade Sharpe)
        - Minimum sample size: >= 2 (required for ddof=1)
        - Zero variance threshold: std < 1e-10
        - Float→Fixed: multiply by 1_000_000, truncate toward zero (int())
        - Returns Fixed(RATE) with scale=6
    """
    if len(returns_bps) < 2:
        raise InsufficientDataError(
            f"Need >= 2 trades for Sharpe (ddof=1), got {len(returns_bps)}"
        )
    arr = np.array(returns_bps, dtype=np.float64)
    mean_r = float(np.mean(arr))
    std_r = float(np.std(arr, ddof=1))
    if std_r < 1e-10:
        raise ZeroVarianceError(f"std={std_r:.2e}, below 1e-10 threshold")
    sharpe_float = mean_r / std_r
    if not math.isfinite(sharpe_float):
        raise NumericalError(f"Sharpe is {sharpe_float}")
    # TRUNCATE toward zero (deterministic)
    sharpe_scaled = int(sharpe_float * 1_000_000)
    return Fixed(value=sharpe_scaled, sem=SemanticType.RATE)


# ---------------------------------------------------------------------------
# Canonical data hash
# ---------------------------------------------------------------------------

def canonical_data_hash(
    timestamps: List[int],
    opens: List[int],
    highs: List[int],
    lows: List[int],
    closes: List[int],
    volumes: List[int],
) -> str:
    """
    Canonical data hash using raw integer bytes.

    All inputs must be pre-scaled integers (e.g. prices * 10^8, volumes * 10^2).
    Concatenates raw bytes in order: timestamps, opens, highs, lows, closes, volumes.
    """
    import struct
    buf = bytearray()
    for series in [timestamps, opens, highs, lows, closes, volumes]:
        for v in series:
            buf.extend(struct.pack(">q", v))  # big-endian int64
    return hashlib.sha256(bytes(buf)).hexdigest()


# ---------------------------------------------------------------------------
# Seed cascade
# ---------------------------------------------------------------------------

def derive_master_seed(
    strategy_id: str,
    dataset_hash: str,
    split_timestamp: str,
    config_hash: str,
) -> int:
    """Derive collision-resistant 32-bit master seed."""
    seed_input = f"{strategy_id}|{dataset_hash}|{split_timestamp}|{config_hash}"
    h = hashlib.sha256(seed_input.encode("utf-8"))
    return int.from_bytes(h.digest()[:4], byteorder="big")


def derive_subseed(master_seed: int, domain: str) -> int:
    """Derive domain-separated 32-bit sub-seed."""
    seed_bytes = master_seed.to_bytes(4, byteorder="big")
    domain_bytes = domain.encode("utf-8")
    h = hashlib.sha256(seed_bytes + b"|" + domain_bytes)
    return int.from_bytes(h.digest()[:4], byteorder="big")
