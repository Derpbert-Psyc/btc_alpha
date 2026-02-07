"""
BTC ALPHA SYSTEM - PHASE 2 (v4 - CANONICAL FINAL)
===================================================

Strategy Research Kernel and Baseline Alpha Candidate

FIXES FROM v3:
1. Semantic consistency validation in all indicator kernels
2. Candle __post_init__ enforces semantic types
3. HTF lag precondition validation (prevents lookahead)
4. Demo cases corrected to show real HALF_EVEN tie cases
5. Warmup explicitly tied to max_lookback (dual check)

Author: Claude (Anthropic)
Phase: 2
Version: 4 (Canonical Final)
Depends on: btc_alpha_v3_final.py (Phase 1 - frozen, canonical)
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Final,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

# =============================================================================
# PHASE 1 IMPORTS — DENY BY DEFAULT
# =============================================================================

_DEMO_MODE_ENV_VAR: Final[str] = "BTC_ALPHA_DEMO_MODE"
_DEMO_MODE_ENABLED: Final[bool] = os.environ.get(_DEMO_MODE_ENV_VAR, "0") == "1"

_PHASE1_IMPORT_SUCCESS: bool = False

try:
    from btc_alpha_v3_final import (
        Fixed,
        SemanticType,
        RoundingMode,
        SEMANTIC_SCALES,
        FixedError,
        TypeMismatchError,
    )
    from btc_alpha_v3_final import _integer_rescale
    _PHASE1_IMPORT_SUCCESS = True
except ImportError:
    if not _DEMO_MODE_ENABLED:
        raise ImportError(
            "Phase 1 module (btc_alpha_v3_final) not found. "
            "Phase 2 requires Phase 1 imports for numeric determinism. "
            f"To run in demo mode, set environment variable: {_DEMO_MODE_ENV_VAR}=1"
        )
    _PHASE1_IMPORT_SUCCESS = False

if not _PHASE1_IMPORT_SUCCESS:
    # Demo fallback — FORBIDDEN in production
    from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_EVEN, getcontext
    getcontext().prec = 50
    
    class SemanticType(Enum):
        PRICE = auto()
        QTY = auto()
        USD = auto()
        RATE = auto()
    
    SEMANTIC_SCALES: Final[dict[SemanticType, int]] = {
        SemanticType.PRICE: 2,
        SemanticType.QTY: 8,
        SemanticType.USD: 2,
        SemanticType.RATE: 6,
    }
    
    class RoundingMode(Enum):
        TRUNCATE = auto()
        AWAY_FROM_ZERO = auto()
        HALF_EVEN = auto()
    
    class FixedError(Exception):
        pass
    
    class TypeMismatchError(FixedError):
        pass
    
    @dataclass(frozen=True, slots=True)
    class Fixed:
        value: int
        sem: SemanticType
        
        @property
        def scale(self) -> int:
            return SEMANTIC_SCALES[self.sem]
        
        def to_decimal(self) -> Decimal:
            return Decimal(self.value) / Decimal(10 ** self.scale)
        
        @classmethod
        def from_decimal(cls, x: Decimal, sem: SemanticType, rounding: RoundingMode = RoundingMode.TRUNCATE) -> Fixed:
            scale = SEMANTIC_SCALES[sem]
            scaled = x * Decimal(10 ** scale)
            if rounding == RoundingMode.TRUNCATE:
                int_val = int(scaled.to_integral_value(rounding=ROUND_DOWN))
            elif rounding == RoundingMode.AWAY_FROM_ZERO:
                int_val = int(scaled.to_integral_value(rounding=ROUND_UP)) if scaled >= 0 else int(scaled.to_integral_value(rounding=ROUND_DOWN))
            else:
                int_val = int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))
            return cls(value=int_val, sem=sem)
        
        @classmethod
        def from_str(cls, s: str, sem: SemanticType, rounding: RoundingMode = RoundingMode.TRUNCATE) -> Fixed:
            return cls.from_decimal(Decimal(s), sem, rounding)
        
        @classmethod
        def zero(cls, sem: SemanticType) -> Fixed:
            return cls(value=0, sem=sem)
        
        def __add__(self, other: Fixed) -> Fixed:
            if self.sem != other.sem:
                raise TypeMismatchError(f"Cannot add {self.sem.name} to {other.sem.name}")
            return Fixed(value=self.value + other.value, sem=self.sem)
        
        def __sub__(self, other: Fixed) -> Fixed:
            if self.sem != other.sem:
                raise TypeMismatchError(f"Cannot subtract {other.sem.name} from {self.sem.name}")
            return Fixed(value=self.value - other.value, sem=self.sem)
        
        def __neg__(self) -> Fixed:
            return Fixed(value=-self.value, sem=self.sem)
        
        def abs(self) -> Fixed:
            return Fixed(value=abs(self.value), sem=self.sem)
        
        def __lt__(self, other: Fixed) -> bool:
            if self.sem != other.sem:
                raise TypeMismatchError(f"Cannot compare {self.sem.name} to {other.sem.name}")
            return self.value < other.value
        
        def __le__(self, other: Fixed) -> bool:
            if self.sem != other.sem:
                raise TypeMismatchError(f"Cannot compare {self.sem.name} to {other.sem.name}")
            return self.value <= other.value
        
        def __gt__(self, other: Fixed) -> bool:
            if self.sem != other.sem:
                raise TypeMismatchError(f"Cannot compare {self.sem.name} to {other.sem.name}")
            return self.value > other.value
        
        def __ge__(self, other: Fixed) -> bool:
            if self.sem != other.sem:
                raise TypeMismatchError(f"Cannot compare {self.sem.name} to {other.sem.name}")
            return self.value >= other.value
        
        def __eq__(self, other: object) -> bool:
            if not isinstance(other, Fixed):
                return NotImplemented
            if self.sem != other.sem:
                return NotImplemented
            return self.value == other.value
        
        def __hash__(self) -> int:
            return hash((self.value, self.sem))
        
        def is_zero(self) -> bool:
            return self.value == 0
        
        def is_positive(self) -> bool:
            return self.value > 0
        
        def is_negative(self) -> bool:
            return self.value < 0
        
        def __repr__(self) -> str:
            return f"Fixed({self.to_decimal()}, {self.sem.name})"
        
        def to_canonical(self) -> dict:
            return {"v": self.value, "t": self.sem.name}


# =============================================================================
# PHASE 2 SPECIFIC ERRORS
# =============================================================================

class DeterminismError(Exception):
    """Raised when a determinism invariant is violated."""
    pass


class WarmupIncompleteError(DeterminismError):
    """Raised when warmup check passes but indicators are still None."""
    pass


class SeriesOrderingError(DeterminismError):
    """Raised when time series is not properly ordered."""
    pass


class SemanticConsistencyError(DeterminismError):
    """Raised when inputs have inconsistent semantic types."""
    pass


class LookaheadError(DeterminismError):
    """Raised when HTF data violates lag requirements (potential lookahead)."""
    pass


# =============================================================================
# INTEGER DIVISION WITH ROUNDING
# =============================================================================

def _integer_divide_with_rounding(numerator: int, divisor: int, rounding: RoundingMode) -> int:
    """
    Integer division with explicit rounding mode.
    
    Handles all divisor parities correctly for HALF_EVEN.
    """
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    
    if rounding == RoundingMode.TRUNCATE:
        if numerator >= 0:
            return numerator // divisor
        else:
            return -((-numerator) // divisor)
    
    elif rounding == RoundingMode.AWAY_FROM_ZERO:
        if numerator >= 0:
            return (numerator + divisor - 1) // divisor
        else:
            return -((-numerator + divisor - 1) // divisor)
    
    else:  # HALF_EVEN
        if numerator >= 0:
            q = numerator // divisor
            r = numerator % divisor
        else:
            q = -((-numerator) // divisor)
            r = (-numerator) % divisor
        
        twice_remainder = 2 * r
        
        if twice_remainder < divisor:
            return q
        elif twice_remainder > divisor:
            if numerator >= 0:
                return q + 1
            else:
                return q - 1
        else:
            # Exact tie: round to even
            if q % 2 == 0:
                return q
            else:
                if numerator >= 0:
                    return q + 1
                else:
                    return q - 1


# =============================================================================
# TIMESTAMP CONVENTION
# =============================================================================

TIMESTAMP_UNIT: Final[str] = "epoch_seconds"
SECONDS_PER_MINUTE: Final[int] = 60
SECONDS_PER_4H: Final[int] = 4 * 60 * 60  # 14400 seconds


def ts_to_minutes(ts: int) -> int:
    """Convert epoch seconds to minute index."""
    return ts // SECONDS_PER_MINUTE


def holding_time_minutes(entry_ts: int, exit_ts: int) -> int:
    """Compute holding time in minutes from epoch second timestamps."""
    return (exit_ts - entry_ts) // SECONDS_PER_MINUTE


def is_4h_boundary(ts: int) -> bool:
    """Check if timestamp is aligned to a 4H boundary."""
    return ts % SECONDS_PER_4H == 0


# =============================================================================
# SEMANTIC CONSISTENCY VALIDATION (FIX #1)
# =============================================================================

def _require_same_sem(values: Sequence[Fixed], context: str) -> SemanticType:
    """
    Validate that all Fixed values share the same semantic type.
    
    Raises SemanticConsistencyError if any element differs.
    Returns the common semantic type.
    
    FIX #1: Prevents silent divergence from mixed-type inputs.
    """
    if not values:
        raise SemanticConsistencyError(f"{context}: empty sequence")
    
    expected_sem = values[0].sem
    
    for i, v in enumerate(values[1:], start=1):
        if v.sem != expected_sem:
            raise SemanticConsistencyError(
                f"{context}: semantic type mismatch at index {i}. "
                f"Expected {expected_sem.name}, got {v.sem.name}"
            )
    
    return expected_sem


# =============================================================================
# CANDLE WITH SEMANTIC VALIDATION (FIX #2)
# =============================================================================

@dataclass(frozen=True)
class Candle:
    """
    A single OHLCV candle.
    
    ts: Close timestamp in EPOCH SECONDS
    
    SEMANTIC REQUIREMENTS (enforced in __post_init__):
    - open, high, low, close: must be PRICE type
    - volume: must be QTY type
    """
    ts: int
    open: Fixed
    high: Fixed
    low: Fixed
    close: Fixed
    volume: Fixed
    
    def __post_init__(self) -> None:
        """
        FIX #2: Enforce semantic types for all Candle fields.
        """
        # Validate OHLC are PRICE
        for name, field_val in [
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
        ]:
            if field_val.sem != SemanticType.PRICE:
                raise SemanticConsistencyError(
                    f"Candle.{name} must be PRICE, got {field_val.sem.name}"
                )
        
        # Validate volume is QTY
        if self.volume.sem != SemanticType.QTY:
            raise SemanticConsistencyError(
                f"Candle.volume must be QTY, got {self.volume.sem.name}"
            )
        
        # Validate OHLC consistency (high >= max(open, close), low <= min(open, close))
        if self.high.value < self.open.value or self.high.value < self.close.value:
            raise DeterminismError(
                f"Candle high ({self.high.value}) < open ({self.open.value}) or close ({self.close.value})"
            )
        if self.low.value > self.open.value or self.low.value > self.close.value:
            raise DeterminismError(
                f"Candle low ({self.low.value}) > open ({self.open.value}) or close ({self.close.value})"
            )


# =============================================================================
# HISTORICAL VIEWS WITH LAG VALIDATION (FIX #3)
# =============================================================================

@dataclass(frozen=True)
class HistoricalView1m:
    """Committed 1-minute history for indicator computation."""
    candles: Tuple[Candle, ...]
    current_ts: int  # Current timestamp in epoch seconds
    
    def __len__(self) -> int:
        return len(self.candles)
    
    def latest(self, n: int = 1) -> Tuple[Candle, ...]:
        if n > len(self.candles):
            return self.candles
        return self.candles[-n:]
    
    def closes(self, n: Optional[int] = None) -> Tuple[Fixed, ...]:
        candles = self.candles if n is None else self.latest(n)
        return tuple(c.close for c in candles)
    
    def highs(self, n: Optional[int] = None) -> Tuple[Fixed, ...]:
        candles = self.candles if n is None else self.latest(n)
        return tuple(c.high for c in candles)
    
    def lows(self, n: Optional[int] = None) -> Tuple[Fixed, ...]:
        candles = self.candles if n is None else self.latest(n)
        return tuple(c.low for c in candles)


@dataclass(frozen=True)
class HistoricalView4H:
    """
    Committed 4-hour history for HTF regime.
    
    INVARIANT: All candles must be FULLY CLOSED relative to current_ts.
    This is validated in __post_init__.
    
    FIX #3: Explicit lag validation prevents lookahead contamination.
    """
    candles: Tuple[Candle, ...]
    current_ts: int
    
    def __post_init__(self) -> None:
        """
        Validate that all 4H candles are fully closed (no lookahead).
        
        A 4H candle is "closed" if its close timestamp (candle.ts) is
        at least SECONDS_PER_4H before current_ts.
        
        More precisely: candle.ts + SECONDS_PER_4H <= current_ts
        (The candle closed at candle.ts, so next candle opens at candle.ts,
        and we need the full 4H to have passed)
        
        Actually, candle.ts IS the close time. So we just need:
        candle.ts <= current_ts - SECONDS_PER_4H + SECONDS_PER_MINUTE
        (allow 1 minute tolerance for the current minute processing)
        
        Simplified: all candle.ts must be < current_ts aligned to 4H boundary
        """
        if not self.candles:
            return  # Empty is valid (warmup)
        
        for i, candle in enumerate(self.candles):
            # The candle closed at candle.ts
            # We are processing at current_ts
            # The candle is "safe" (fully closed) if candle.ts <= current_ts
            # But we also need to ensure it's a COMPLETE 4H bar
            
            # Strictest check: candle.ts must be a 4H boundary
            if not is_4h_boundary(candle.ts):
                raise LookaheadError(
                    f"4H candle at index {i} has non-aligned close time {candle.ts}. "
                    f"Expected 4H boundary (divisible by {SECONDS_PER_4H})"
                )
            
            # The candle must have closed BEFORE current_ts
            # (closed at candle.ts, so candle.ts < current_ts)
            if candle.ts >= self.current_ts:
                raise LookaheadError(
                    f"4H candle at index {i} with ts={candle.ts} is not closed "
                    f"relative to current_ts={self.current_ts}. Potential lookahead!"
                )
    
    def __len__(self) -> int:
        return len(self.candles)
    
    def latest(self, n: int = 1) -> Tuple[Candle, ...]:
        if n > len(self.candles):
            return self.candles
        return self.candles[-n:]
    
    def closes(self, n: Optional[int] = None) -> Tuple[Fixed, ...]:
        candles = self.candles if n is None else self.latest(n)
        return tuple(c.close for c in candles)


@dataclass(frozen=True)
class CommittedHistory:
    """Complete committed history passed to indicators."""
    view_1m: HistoricalView1m
    view_4h: HistoricalView4H
    
    def __post_init__(self) -> None:
        """Validate consistency between views."""
        # Both views must refer to the same current time
        if self.view_1m.current_ts != self.view_4h.current_ts:
            raise DeterminismError(
                f"Timestamp mismatch: 1m view has current_ts={self.view_1m.current_ts}, "
                f"4H view has current_ts={self.view_4h.current_ts}"
            )


# =============================================================================
# INDICATOR OUTPUT WITH EXPLICIT WARMUP LINKAGE (FIX #5)
# =============================================================================

@dataclass(frozen=True)
class IndicatorOutput:
    """
    Output from indicator computation.
    
    FIX #5: Warmup check is now explicit about which indicators are required
    for the baseline strategy. This list MUST be updated if strategy dependencies change.
    """
    sma_fast: Optional[Fixed]
    sma_slow: Optional[Fixed]
    ema: Optional[Fixed]
    donchian_high: Optional[Fixed]
    donchian_low: Optional[Fixed]
    donchian_mid: Optional[Fixed]
    atr: Optional[Fixed]
    htf_close: Optional[Fixed]
    htf_sma: Optional[Fixed]
    htf_trend_up: Optional[bool]
    pct_change_bps: Optional[int]
    
    # Metadata for warmup validation
    history_length_1m: int = 0
    history_length_4h: int = 0
    required_lookback_1m: int = 0
    required_lookback_4h: int = 0
    
    def has_warmup_complete(self) -> bool:
        """
        Check if warmup is complete.
        
        FIX #5: Dual check:
        1. History length >= required lookbacks
        2. All required indicator fields are not None
        
        This ensures warmup failure is detected both by insufficient data
        AND by indicator computation failures.
        """
        # Check 1: History length sufficient
        if self.history_length_1m < self.required_lookback_1m:
            return False
        if self.history_length_4h < self.required_lookback_4h:
            return False
        
        # Check 2: Required fields not None
        # BASELINE STRATEGY DEPENDENCIES (update if strategy changes):
        required_fields = [
            self.sma_fast,       # For trend filter
            self.donchian_high,  # For breakout entry
            self.donchian_low,   # For breakout entry and exit
            self.atr,            # For volatility filter
            self.htf_sma,        # For HTF trend
            self.htf_trend_up,   # For HTF agreement
        ]
        
        return all(v is not None for v in required_fields)
    
    def get_required_or_raise(self) -> Tuple[Fixed, Fixed, Fixed, Fixed, Fixed, bool]:
        """Get required indicators, raising DeterminismError if any are None."""
        if self.sma_fast is None:
            raise WarmupIncompleteError("sma_fast is None")
        if self.donchian_high is None:
            raise WarmupIncompleteError("donchian_high is None")
        if self.donchian_low is None:
            raise WarmupIncompleteError("donchian_low is None")
        if self.atr is None:
            raise WarmupIncompleteError("atr is None")
        if self.htf_sma is None:
            raise WarmupIncompleteError("htf_sma is None")
        if self.htf_trend_up is None:
            raise WarmupIncompleteError("htf_trend_up is None")
        
        return (
            self.sma_fast,
            self.donchian_high,
            self.donchian_low,
            self.atr,
            self.htf_sma,
            self.htf_trend_up,
        )


class IndicatorModule(Protocol):
    def compute(self, history: CommittedHistory) -> IndicatorOutput: ...
    @property
    def max_lookback_1m(self) -> int: ...
    @property
    def max_lookback_4h(self) -> int: ...


# =============================================================================
# INDICATOR KERNEL WITH SEMANTIC VALIDATION (FIX #1)
# =============================================================================

def sma(prices: Tuple[Fixed, ...], period: int) -> Optional[Fixed]:
    """
    Simple Moving Average using integer arithmetic.
    
    FIX #1: Validates semantic consistency of all inputs.
    """
    if len(prices) < period:
        return None
    
    recent = prices[-period:]
    
    # Validate semantic consistency
    sem = _require_same_sem(recent, f"sma(period={period})")
    
    total = sum(p.value for p in recent)
    avg_value = total // period
    
    return Fixed(value=avg_value, sem=sem)


def ema(
    prices: Tuple[Fixed, ...],
    period: int,
    rounding: RoundingMode = RoundingMode.TRUNCATE
) -> Optional[Fixed]:
    """
    Exponential Moving Average using Fixed arithmetic.
    
    FIX #1: Validates semantic consistency of all inputs.
    """
    if len(prices) < period:
        return None
    
    # Validate semantic consistency
    sem = _require_same_sem(prices, f"ema(period={period})")
    
    initial_prices = prices[:period]
    ema_value = sum(p.value for p in initial_prices) // period
    
    multiplier_new = 2
    multiplier_old = period - 1
    divisor = period + 1
    
    for price in prices[period:]:
        numerator = multiplier_new * price.value + multiplier_old * ema_value
        ema_value = _integer_divide_with_rounding(numerator, divisor, rounding)
    
    return Fixed(value=ema_value, sem=sem)


def rolling_high(prices: Tuple[Fixed, ...], period: int) -> Optional[Fixed]:
    """
    Highest value over the last N periods.
    
    FIX #1: Validates semantic consistency.
    """
    if len(prices) < period:
        return None
    
    recent = prices[-period:]
    _require_same_sem(recent, f"rolling_high(period={period})")
    
    return max(recent, key=lambda p: p.value)


def rolling_low(prices: Tuple[Fixed, ...], period: int) -> Optional[Fixed]:
    """
    Lowest value over the last N periods.
    
    FIX #1: Validates semantic consistency.
    """
    if len(prices) < period:
        return None
    
    recent = prices[-period:]
    _require_same_sem(recent, f"rolling_low(period={period})")
    
    return min(recent, key=lambda p: p.value)


def donchian_channel(
    highs: Tuple[Fixed, ...],
    lows: Tuple[Fixed, ...],
    period: int
) -> Tuple[Optional[Fixed], Optional[Fixed], Optional[Fixed]]:
    """
    Donchian Channel: (high, low, mid).
    
    FIX #1: Validates semantic consistency of highs and lows.
    """
    if len(highs) < period or len(lows) < period:
        return (None, None, None)
    
    # Validate each series
    _require_same_sem(highs[-period:], f"donchian_channel highs")
    _require_same_sem(lows[-period:], f"donchian_channel lows")
    
    # Validate highs and lows have same sem
    if highs[0].sem != lows[0].sem:
        raise SemanticConsistencyError(
            f"donchian_channel: highs are {highs[0].sem.name} but lows are {lows[0].sem.name}"
        )
    
    high = rolling_high(highs, period)
    low = rolling_low(lows, period)
    
    if high is None or low is None:
        return (None, None, None)
    
    mid_value = (high.value + low.value) // 2
    mid = Fixed(value=mid_value, sem=high.sem)
    
    return (high, low, mid)


def true_range(candle: Candle, prev_close: Optional[Fixed]) -> Fixed:
    """True Range for a single candle."""
    hl = candle.high.value - candle.low.value
    
    if prev_close is None:
        return Fixed(value=hl, sem=candle.high.sem)
    
    # Validate prev_close semantic
    if prev_close.sem != SemanticType.PRICE:
        raise SemanticConsistencyError(
            f"true_range: prev_close must be PRICE, got {prev_close.sem.name}"
        )
    
    hc = abs(candle.high.value - prev_close.value)
    lc = abs(candle.low.value - prev_close.value)
    
    tr_value = max(hl, hc, lc)
    return Fixed(value=tr_value, sem=candle.high.sem)


def atr(
    candles: Tuple[Candle, ...],
    period: int,
    rounding: RoundingMode = RoundingMode.TRUNCATE
) -> Optional[Fixed]:
    """Average True Range using Wilder smoothing."""
    if len(candles) < period + 1:
        return None
    
    trs: list[Fixed] = []
    for i, candle in enumerate(candles):
        prev_close = candles[i - 1].close if i > 0 else None
        trs.append(true_range(candle, prev_close))
    
    if len(trs) < period + 1:
        return None
    
    # Validate TR semantic consistency
    _require_same_sem(trs, f"atr(period={period}) true_ranges")
    
    initial_trs = trs[1:period + 1]
    atr_value = sum(tr.value for tr in initial_trs) // period
    sem = trs[0].sem
    
    for tr in trs[period + 1:]:
        numerator = (period - 1) * atr_value + tr.value
        atr_value = _integer_divide_with_rounding(numerator, period, rounding)
    
    return Fixed(value=atr_value, sem=sem)


def pct_change_bps(old: Fixed, new: Fixed) -> Optional[int]:
    """
    Percent change as integer basis points.
    
    Already validates semantic consistency (old.sem == new.sem).
    """
    if old.sem != new.sem:
        raise TypeMismatchError(
            f"pct_change_bps: semantic mismatch {old.sem.name} vs {new.sem.name}"
        )
    
    if old.value == 0:
        return None
    
    diff = new.value - old.value
    scaled_diff = diff * 10000
    
    if scaled_diff >= 0:
        return scaled_diff // old.value
    else:
        return -((-scaled_diff) // old.value)


# =============================================================================
# BASELINE INDICATOR MODULE
# =============================================================================

@dataclass(frozen=True)
class BaselineIndicatorConfig:
    sma_fast_period: int = 20
    sma_slow_period: int = 50
    ema_period: int = 20
    donchian_period: int = 20
    atr_period: int = 14
    htf_sma_period: int = 20


class BaselineIndicatorModule:
    """
    Baseline indicator module.
    
    FIX #5: max_lookback values are explicitly used in IndicatorOutput
    to enable dual warmup checking.
    """
    
    def __init__(self, config: BaselineIndicatorConfig):
        self.config = config
    
    @property
    def max_lookback_1m(self) -> int:
        return max(
            self.config.sma_fast_period,
            self.config.sma_slow_period,
            self.config.ema_period,
            self.config.donchian_period,
            self.config.atr_period + 1,
        )
    
    @property
    def max_lookback_4h(self) -> int:
        return self.config.htf_sma_period
    
    def compute(self, history: CommittedHistory) -> IndicatorOutput:
        """
        Compute all indicators.
        
        FIX #3: History views already validated for lookahead in __post_init__.
        FIX #5: Returns lookback metadata for dual warmup check.
        """
        closes_1m = history.view_1m.closes()
        highs_1m = history.view_1m.highs()
        lows_1m = history.view_1m.lows()
        
        sma_fast_val = sma(closes_1m, self.config.sma_fast_period)
        sma_slow_val = sma(closes_1m, self.config.sma_slow_period)
        ema_val = ema(closes_1m, self.config.ema_period)
        
        don_high, don_low, don_mid = donchian_channel(
            highs_1m, lows_1m, self.config.donchian_period
        )
        
        atr_val = atr(history.view_1m.candles, self.config.atr_period)
        
        pct_bps = None
        if len(closes_1m) >= 2:
            pct_bps = pct_change_bps(closes_1m[-2], closes_1m[-1])
        
        closes_4h = history.view_4h.closes()
        
        htf_close_val = closes_4h[-1] if closes_4h else None
        htf_sma_val = sma(closes_4h, self.config.htf_sma_period)
        
        htf_trend_up_val = None
        if htf_close_val is not None and htf_sma_val is not None:
            htf_trend_up_val = htf_close_val.value > htf_sma_val.value
        
        return IndicatorOutput(
            sma_fast=sma_fast_val,
            sma_slow=sma_slow_val,
            ema=ema_val,
            donchian_high=don_high,
            donchian_low=don_low,
            donchian_mid=don_mid,
            atr=atr_val,
            htf_close=htf_close_val,
            htf_sma=htf_sma_val,
            htf_trend_up=htf_trend_up_val,
            pct_change_bps=pct_bps,
            # FIX #5: Include metadata for dual warmup check
            history_length_1m=len(history.view_1m),
            history_length_4h=len(history.view_4h),
            required_lookback_1m=self.max_lookback_1m,
            required_lookback_4h=self.max_lookback_4h,
        )


# =============================================================================
# SIGNAL MODULE
# =============================================================================

class Signal(Enum):
    HOLD = auto()
    ENTER_LONG = auto()
    ENTER_SHORT = auto()
    EXIT = auto()


@dataclass(frozen=True)
class PositionView:
    position_qty: Fixed
    avg_entry_price: Optional[Fixed]
    
    def __post_init__(self) -> None:
        if self.position_qty.sem != SemanticType.QTY:
            raise SemanticConsistencyError(
                f"PositionView.position_qty must be QTY, got {self.position_qty.sem.name}"
            )
        if self.avg_entry_price is not None and self.avg_entry_price.sem != SemanticType.PRICE:
            raise SemanticConsistencyError(
                f"PositionView.avg_entry_price must be PRICE, got {self.avg_entry_price.sem.name}"
            )
    
    def is_flat(self) -> bool:
        return self.position_qty.is_zero()
    
    def is_long(self) -> bool:
        return self.position_qty.is_positive()
    
    def is_short(self) -> bool:
        return self.position_qty.is_negative()


@dataclass(frozen=True)
class SignalOutput:
    signal: Signal
    reason: str


@dataclass(frozen=True)
class BaselineSignalConfig:
    require_htf_agreement: bool = True


class BaselineSignalModule:
    def __init__(self, config: BaselineSignalConfig):
        self.config = config
    
    def compute(
        self,
        indicators: IndicatorOutput,
        position: PositionView,
        current_price: Fixed,
    ) -> SignalOutput:
        if not indicators.has_warmup_complete():
            return SignalOutput(signal=Signal.HOLD, reason="warmup_incomplete")
        
        try:
            (sma_fast, don_high, don_low, atr_val, htf_sma, htf_trend_up) = \
                indicators.get_required_or_raise()
        except WarmupIncompleteError as e:
            return SignalOutput(signal=Signal.HOLD, reason=f"warmup_error:{e}")
        
        if position.is_flat():
            return self._check_entry(current_price, don_high, don_low, atr_val, htf_trend_up)
        else:
            return self._check_exit(position, current_price, don_high, don_low)
    
    def _check_entry(
        self,
        price: Fixed,
        don_high: Fixed,
        don_low: Fixed,
        atr_val: Fixed,
        htf_trend_up: bool,
    ) -> SignalOutput:
        if atr_val.value <= 0:
            return SignalOutput(signal=Signal.HOLD, reason="atr_zero")
        
        if price.value > don_high.value:
            if self.config.require_htf_agreement and not htf_trend_up:
                return SignalOutput(signal=Signal.HOLD, reason="htf_disagrees_long")
            return SignalOutput(signal=Signal.ENTER_LONG, reason="breakout_high")
        
        if price.value < don_low.value:
            if self.config.require_htf_agreement and htf_trend_up:
                return SignalOutput(signal=Signal.HOLD, reason="htf_disagrees_short")
            return SignalOutput(signal=Signal.ENTER_SHORT, reason="breakout_low")
        
        return SignalOutput(signal=Signal.HOLD, reason="no_breakout")
    
    def _check_exit(
        self,
        position: PositionView,
        price: Fixed,
        don_high: Fixed,
        don_low: Fixed,
    ) -> SignalOutput:
        if position.is_long():
            if price.value < don_low.value:
                return SignalOutput(signal=Signal.EXIT, reason="exit_long_breakdown")
        elif position.is_short():
            if price.value > don_high.value:
                return SignalOutput(signal=Signal.EXIT, reason="exit_short_breakout")
        
        return SignalOutput(signal=Signal.HOLD, reason="hold_position")


# =============================================================================
# GATE VIEW
# =============================================================================

@dataclass(frozen=True)
class GateView:
    diagnostics_ok: bool
    diagnostics_veto: bool
    regime_permissive: bool
    warmup_complete: bool
    
    def can_trade(self) -> bool:
        return (
            self.diagnostics_ok and 
            not self.diagnostics_veto and 
            self.regime_permissive and
            self.warmup_complete
        )
    
    def blocked_reason(self) -> Optional[str]:
        if not self.warmup_complete:
            return "warmup_incomplete"
        if not self.diagnostics_ok:
            return "diagnostics_not_ok"
        if self.diagnostics_veto:
            return "diagnostics_veto"
        if not self.regime_permissive:
            return "regime_non_permissive"
        return None


def create_gate_view(
    diagnostics_ok: bool,
    diagnostics_veto: bool,
    regime_permissive: bool,
    indicators: IndicatorOutput,
) -> GateView:
    warmup_complete = indicators.has_warmup_complete()
    return GateView(
        diagnostics_ok=diagnostics_ok,
        diagnostics_veto=diagnostics_veto,
        regime_permissive=regime_permissive,
        warmup_complete=warmup_complete,
    )


# =============================================================================
# EXECUTION LOGIC
# =============================================================================

@dataclass(frozen=True)
class LedgerView:
    equity: Fixed
    position_qty: Fixed
    avg_entry_price: Optional[Fixed]
    
    def __post_init__(self) -> None:
        if self.equity.sem != SemanticType.USD:
            raise SemanticConsistencyError(
                f"LedgerView.equity must be USD, got {self.equity.sem.name}"
            )
        if self.position_qty.sem != SemanticType.QTY:
            raise SemanticConsistencyError(
                f"LedgerView.position_qty must be QTY, got {self.position_qty.sem.name}"
            )
        if self.avg_entry_price is not None and self.avg_entry_price.sem != SemanticType.PRICE:
            raise SemanticConsistencyError(
                f"LedgerView.avg_entry_price must be PRICE, got {self.avg_entry_price.sem.name}"
            )
    
    def is_flat(self) -> bool:
        return self.position_qty.is_zero()


@dataclass(frozen=True)
class OrderIntent:
    side: Literal["BUY", "SELL"]
    qty: Fixed
    reason: str
    
    def __post_init__(self):
        if self.qty.sem != SemanticType.QTY:
            raise TypeMismatchError("OrderIntent qty must be QTY type")
        if not self.qty.is_positive():
            raise ValueError("OrderIntent qty must be positive")


@dataclass(frozen=True)
class ExecutionOutput:
    action: Literal["HOLD", "ENTER", "EXIT"]
    orders: Tuple[OrderIntent, ...]
    reason: str


@dataclass(frozen=True)
class BaselineExecutionConfig:
    risk_fraction: Fixed
    min_qty: Fixed
    max_qty: Fixed
    
    def __post_init__(self):
        if self.risk_fraction.sem != SemanticType.RATE:
            raise TypeMismatchError("risk_fraction must be RATE type")
        if self.min_qty.sem != SemanticType.QTY:
            raise TypeMismatchError("min_qty must be QTY type")
        if self.max_qty.sem != SemanticType.QTY:
            raise TypeMismatchError("max_qty must be QTY type")


class BaselineExecutionLogic:
    def __init__(self, config: BaselineExecutionConfig):
        self.config = config
    
    def compute(
        self,
        signal: SignalOutput,
        gates: GateView,
        ledger: LedgerView,
        current_price: Fixed,
    ) -> ExecutionOutput:
        if not gates.can_trade():
            reason = gates.blocked_reason() or "gates_blocked"
            return ExecutionOutput(action="HOLD", orders=(), reason=reason)
        
        if signal.signal == Signal.HOLD:
            return ExecutionOutput(action="HOLD", orders=(), reason=signal.reason)
        
        if signal.signal == Signal.EXIT:
            return self._execute_exit(ledger, signal.reason)
        
        if signal.signal in (Signal.ENTER_LONG, Signal.ENTER_SHORT):
            return self._execute_entry(signal, ledger, current_price)
        
        return ExecutionOutput(action="HOLD", orders=(), reason="unknown_signal")
    
    def _compute_position_size(self, equity: Fixed, price: Fixed) -> Optional[Fixed]:
        if equity.value <= 0 or price.value <= 0:
            return None
        
        intermediate = equity.value * self.config.risk_fraction.value
        intermediate_scale = equity.scale + self.config.risk_fraction.scale
        
        usd_scale = SEMANTIC_SCALES[SemanticType.USD]
        if intermediate_scale > usd_scale:
            risk_amount = _integer_divide_with_rounding(
                intermediate, 
                10 ** (intermediate_scale - usd_scale),
                RoundingMode.TRUNCATE
            )
        else:
            risk_amount = intermediate * (10 ** (usd_scale - intermediate_scale))
        
        qty_scale = SEMANTIC_SCALES[SemanticType.QTY]
        scale_factor = qty_scale + price.scale - usd_scale
        scaled_risk = risk_amount * (10 ** scale_factor)
        
        qty_value = _integer_divide_with_rounding(scaled_risk, price.value, RoundingMode.TRUNCATE)
        
        if qty_value <= 0:
            return None
        
        qty_value = max(qty_value, self.config.min_qty.value)
        qty_value = min(qty_value, self.config.max_qty.value)
        
        return Fixed(value=qty_value, sem=SemanticType.QTY)
    
    def _execute_entry(
        self,
        signal: SignalOutput,
        ledger: LedgerView,
        price: Fixed,
    ) -> ExecutionOutput:
        if not ledger.is_flat():
            return ExecutionOutput(action="HOLD", orders=(), reason="already_in_position")
        
        qty = self._compute_position_size(ledger.equity, price)
        if qty is None:
            return ExecutionOutput(action="HOLD", orders=(), reason="insufficient_equity")
        
        side = "BUY" if signal.signal == Signal.ENTER_LONG else "SELL"
        order = OrderIntent(side=side, qty=qty, reason=signal.reason)
        
        return ExecutionOutput(action="ENTER", orders=(order,), reason=signal.reason)
    
    def _execute_exit(self, ledger: LedgerView, reason: str) -> ExecutionOutput:
        if ledger.is_flat():
            return ExecutionOutput(action="HOLD", orders=(), reason="already_flat")
        
        qty = ledger.position_qty.abs()
        side = "SELL" if ledger.position_qty.is_positive() else "BUY"
        order = OrderIntent(side=side, qty=qty, reason=reason)
        
        return ExecutionOutput(action="EXIT", orders=(order,), reason=reason)


# =============================================================================
# TRADE RECORD
# =============================================================================

@dataclass(frozen=True)
class TradeRecord:
    """All timestamps in EPOCH SECONDS."""
    entry_ts: int
    exit_ts: int
    side: Literal["LONG", "SHORT"]
    entry_price: Fixed
    exit_price: Fixed
    qty: Fixed
    gross_pnl: Fixed
    fees: Fixed
    slippage: Fixed
    net_pnl: Fixed
    
    @property
    def holding_time_minutes(self) -> int:
        return holding_time_minutes(self.entry_ts, self.exit_ts)
    
    def to_canonical(self) -> dict:
        return {
            "entry_ts": self.entry_ts,
            "exit_ts": self.exit_ts,
            "side": self.side,
            "entry_price": self.entry_price.to_canonical(),
            "exit_price": self.exit_price.to_canonical(),
            "qty": self.qty.to_canonical(),
            "gross_pnl": self.gross_pnl.to_canonical(),
            "fees": self.fees.to_canonical(),
            "slippage": self.slippage.to_canonical(),
            "net_pnl": self.net_pnl.to_canonical(),
        }


# =============================================================================
# METRICS
# =============================================================================

def _validate_series_ordering(
    series: Sequence[Tuple[int, Any]],
    series_name: str
) -> None:
    if len(series) <= 1:
        return
    
    for i in range(1, len(series)):
        prev_ts = series[i - 1][0]
        curr_ts = series[i][0]
        
        if curr_ts < prev_ts:
            raise SeriesOrderingError(
                f"{series_name} is not sorted: timestamp {curr_ts} at index {i} "
                f"is less than {prev_ts} at index {i-1}"
            )


@dataclass(frozen=True)
class EvaluationMetrics:
    net_pnl: Fixed
    gross_pnl: Fixed
    total_fees: Fixed
    total_slippage: Fixed
    trade_count: int
    max_drawdown: Fixed
    max_drawdown_pct_bps: int
    total_holding_minutes: int
    avg_holding_minutes: int
    max_leverage_bps: int
    avg_leverage_bps: int
    win_count: int
    loss_count: int
    win_rate_bps: int
    
    def to_canonical(self) -> dict:
        return {
            "net_pnl": self.net_pnl.to_canonical(),
            "gross_pnl": self.gross_pnl.to_canonical(),
            "total_fees": self.total_fees.to_canonical(),
            "total_slippage": self.total_slippage.to_canonical(),
            "trade_count": self.trade_count,
            "max_drawdown": self.max_drawdown.to_canonical(),
            "max_drawdown_pct_bps": self.max_drawdown_pct_bps,
            "total_holding_minutes": self.total_holding_minutes,
            "avg_holding_minutes": self.avg_holding_minutes,
            "max_leverage_bps": self.max_leverage_bps,
            "avg_leverage_bps": self.avg_leverage_bps,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate_bps": self.win_rate_bps,
        }


def compute_metrics(
    trades: Sequence[TradeRecord],
    equity_series: Sequence[Tuple[int, Fixed]],
    leverage_series: Sequence[Tuple[int, int]],
) -> EvaluationMetrics:
    _validate_series_ordering(equity_series, "equity_series")
    _validate_series_ordering(leverage_series, "leverage_series")
    
    net_pnl_value = sum(t.net_pnl.value for t in trades)
    gross_pnl_value = sum(t.gross_pnl.value for t in trades)
    fees_value = sum(t.fees.value for t in trades)
    slippage_value = sum(t.slippage.value for t in trades)
    
    trade_count = len(trades)
    
    total_holding = sum(t.holding_time_minutes for t in trades)
    avg_holding = total_holding // trade_count if trade_count > 0 else 0
    
    win_count = sum(1 for t in trades if t.net_pnl.value > 0)
    loss_count = sum(1 for t in trades if t.net_pnl.value <= 0)
    win_rate_bps = (win_count * 10000) // trade_count if trade_count > 0 else 0
    
    max_drawdown_value = 0
    max_drawdown_pct_bps = 0
    peak_value = 0
    
    for ts, equity in equity_series:
        if equity.value > peak_value:
            peak_value = equity.value
        
        drawdown = peak_value - equity.value
        if drawdown > max_drawdown_value:
            max_drawdown_value = drawdown
            if peak_value > 0:
                max_drawdown_pct_bps = (drawdown * 10000) // peak_value
    
    max_leverage_bps = max((lev for ts, lev in leverage_series), default=0)
    
    if len(leverage_series) > 1:
        total_weighted = 0
        total_time = 0
        for i in range(len(leverage_series) - 1):
            ts1, lev1 = leverage_series[i]
            ts2, _ = leverage_series[i + 1]
            duration = ts2 - ts1
            if duration < 0:
                raise SeriesOrderingError(f"Negative duration between {ts1} and {ts2}")
            total_weighted += lev1 * duration
            total_time += duration
        avg_leverage_bps = total_weighted // total_time if total_time > 0 else 0
    else:
        avg_leverage_bps = leverage_series[0][1] if leverage_series else 0
    
    return EvaluationMetrics(
        net_pnl=Fixed(value=net_pnl_value, sem=SemanticType.USD),
        gross_pnl=Fixed(value=gross_pnl_value, sem=SemanticType.USD),
        total_fees=Fixed(value=fees_value, sem=SemanticType.USD),
        total_slippage=Fixed(value=slippage_value, sem=SemanticType.USD),
        trade_count=trade_count,
        max_drawdown=Fixed(value=max_drawdown_value, sem=SemanticType.USD),
        max_drawdown_pct_bps=max_drawdown_pct_bps,
        total_holding_minutes=total_holding,
        avg_holding_minutes=avg_holding,
        max_leverage_bps=max_leverage_bps,
        avg_leverage_bps=avg_leverage_bps,
        win_count=win_count,
        loss_count=loss_count,
        win_rate_bps=win_rate_bps,
    )


# =============================================================================
# HASHING
# =============================================================================

def hash_trades(trades: Sequence[TradeRecord]) -> str:
    canonical = [t.to_canonical() for t in trades]
    json_bytes = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()


def hash_metrics(metrics: EvaluationMetrics) -> str:
    canonical = metrics.to_canonical()
    json_bytes = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate():
    """Demonstrate Phase 2 v4 components."""
    print("=== BTC Alpha System Phase 2 v4 (Canonical Final) Demo ===\n")
    
    print(f"Phase 1 import success: {_PHASE1_IMPORT_SUCCESS}")
    print(f"Demo mode enabled: {_DEMO_MODE_ENABLED}")
    if not _PHASE1_IMPORT_SUCCESS:
        print("   Running with demo fallback types")
    print()
    
    # FIX #4: Corrected HALF_EVEN demo with REAL tie cases
    print("1. Corrected HALF_EVEN Rounding (real tie cases):")
    
    test_cases = [
        # (numerator, divisor, description)
        (5, 2, "2.5 → 2 (round to even)"),
        (7, 2, "3.5 → 4 (round to even)"),
        (15, 6, "2.5 → 2 (round to even)"),
        (21, 6, "3.5 → 4 (round to even)"),
        (10, 3, "3.33... → 3 (truncate, not tie)"),
        (11, 3, "3.67... → 4 (round up, not tie)"),
    ]
    
    for num, div, desc in test_cases:
        result = _integer_divide_with_rounding(num, div, RoundingMode.HALF_EVEN)
        print(f"   {num} / {div} = {result}  ({desc})")
    
    # FIX #1: Semantic consistency validation
    print("\n2. Semantic Consistency Validation:")
    
    prices = tuple(
        Fixed.from_str(str(45000 + i * 10), SemanticType.PRICE)
        for i in range(25)
    )
    
    sma_val = sma(prices, 20)
    print(f"   SMA(20) with consistent PRICE inputs: {sma_val}")
    
    # Test mixed types (should fail)
    mixed = (
        Fixed.from_str("45000", SemanticType.PRICE),
        Fixed.from_str("100", SemanticType.USD),  # Wrong type!
    )
    try:
        sma(mixed, 2)
        print("   ERROR: Mixed types should have failed!")
    except SemanticConsistencyError as e:
        print(f"   Mixed types correctly rejected: {type(e).__name__}")
    
    # FIX #2: Candle semantic validation
    print("\n3. Candle Semantic Validation:")
    
    valid_candle = Candle(
        ts=1700000000,
        open=Fixed.from_str("45000", SemanticType.PRICE),
        high=Fixed.from_str("45100", SemanticType.PRICE),
        low=Fixed.from_str("44900", SemanticType.PRICE),
        close=Fixed.from_str("45050", SemanticType.PRICE),
        volume=Fixed.from_str("100", SemanticType.QTY),
    )
    print(f"   Valid candle created: ts={valid_candle.ts}")
    
    # Test invalid volume type
    try:
        Candle(
            ts=1700000000,
            open=Fixed.from_str("45000", SemanticType.PRICE),
            high=Fixed.from_str("45100", SemanticType.PRICE),
            low=Fixed.from_str("44900", SemanticType.PRICE),
            close=Fixed.from_str("45050", SemanticType.PRICE),
            volume=Fixed.from_str("100", SemanticType.USD),  # Wrong!
        )
        print("   ERROR: Invalid volume type should have failed!")
    except SemanticConsistencyError as e:
        print(f"   Invalid volume type correctly rejected: {type(e).__name__}")
    
    # FIX #3: HTF lag validation
    print("\n4. HTF Lag Validation (Lookahead Prevention):")
    
    current_ts = 1700014400  # Some timestamp
    
    # Valid: 4H candle closed before current_ts
    valid_4h_ts = 1700006400  # Aligned to 4H boundary, before current_ts
    valid_4h_candle = Candle(
        ts=valid_4h_ts,
        open=Fixed.from_str("45000", SemanticType.PRICE),
        high=Fixed.from_str("45500", SemanticType.PRICE),
        low=Fixed.from_str("44500", SemanticType.PRICE),
        close=Fixed.from_str("45200", SemanticType.PRICE),
        volume=Fixed.from_str("1000", SemanticType.QTY),
    )
    
    try:
        valid_view = HistoricalView4H(
            candles=(valid_4h_candle,),
            current_ts=current_ts,
        )
        print(f"   Valid 4H view created: candle.ts={valid_4h_ts}, current_ts={current_ts}")
    except LookaheadError as e:
        print(f"   ERROR: Valid view should have succeeded: {e}")
    
    # Invalid: 4H candle at or after current_ts (lookahead!)
    future_4h_ts = current_ts + SECONDS_PER_4H
    future_4h_candle = Candle(
        ts=future_4h_ts,
        open=Fixed.from_str("45000", SemanticType.PRICE),
        high=Fixed.from_str("45500", SemanticType.PRICE),
        low=Fixed.from_str("44500", SemanticType.PRICE),
        close=Fixed.from_str("45200", SemanticType.PRICE),
        volume=Fixed.from_str("1000", SemanticType.QTY),
    )
    
    try:
        HistoricalView4H(
            candles=(future_4h_candle,),
            current_ts=current_ts,
        )
        print("   ERROR: Lookahead should have been detected!")
    except LookaheadError as e:
        print(f"   Lookahead correctly detected: {type(e).__name__}")
    
    # FIX #5: Dual warmup check
    print("\n5. Dual Warmup Check (history length + required fields):")
    
    # Insufficient history
    indicators_short = IndicatorOutput(
        sma_fast=sma_val,
        sma_slow=None,
        ema=None,
        donchian_high=Fixed.from_str("45100", SemanticType.PRICE),
        donchian_low=Fixed.from_str("44900", SemanticType.PRICE),
        donchian_mid=Fixed.from_str("45000", SemanticType.PRICE),
        atr=Fixed.from_str("100", SemanticType.PRICE),
        htf_close=Fixed.from_str("45000", SemanticType.PRICE),
        htf_sma=Fixed.from_str("44900", SemanticType.PRICE),
        htf_trend_up=True,
        pct_change_bps=10,
        history_length_1m=10,   # Too short!
        history_length_4h=20,
        required_lookback_1m=50,
        required_lookback_4h=20,
    )
    print(f"   Insufficient 1m history (10 < 50): warmup_complete={indicators_short.has_warmup_complete()}")
    
    # Sufficient history, all fields present
    indicators_ok = IndicatorOutput(
        sma_fast=sma_val,
        sma_slow=sma_val,
        ema=sma_val,
        donchian_high=Fixed.from_str("45100", SemanticType.PRICE),
        donchian_low=Fixed.from_str("44900", SemanticType.PRICE),
        donchian_mid=Fixed.from_str("45000", SemanticType.PRICE),
        atr=Fixed.from_str("100", SemanticType.PRICE),
        htf_close=Fixed.from_str("45000", SemanticType.PRICE),
        htf_sma=Fixed.from_str("44900", SemanticType.PRICE),
        htf_trend_up=True,
        pct_change_bps=10,
        history_length_1m=100,
        history_length_4h=30,
        required_lookback_1m=50,
        required_lookback_4h=20,
    )
    print(f"   Sufficient history (100 >= 50, 30 >= 20): warmup_complete={indicators_ok.has_warmup_complete()}")
    
    # 6. Complete workflow
    print("\n6. Complete Workflow:")
    
    gate = create_gate_view(
        diagnostics_ok=True,
        diagnostics_veto=False,
        regime_permissive=True,
        indicators=indicators_ok,
    )
    
    position = PositionView(
        position_qty=Fixed.zero(SemanticType.QTY),
        avg_entry_price=None
    )
    
    signal_module = BaselineSignalModule(BaselineSignalConfig())
    current_price = Fixed.from_str("45150", SemanticType.PRICE)
    
    signal_out = signal_module.compute(indicators_ok, position, current_price)
    print(f"   Signal: {signal_out.signal.name} ({signal_out.reason})")
    
    exec_config = BaselineExecutionConfig(
        risk_fraction=Fixed.from_str("0.02", SemanticType.RATE),
        min_qty=Fixed.from_str("0.001", SemanticType.QTY),
        max_qty=Fixed.from_str("10", SemanticType.QTY),
    )
    
    exec_logic = BaselineExecutionLogic(exec_config)
    ledger = LedgerView(
        equity=Fixed.from_str("10000", SemanticType.USD),
        position_qty=Fixed.zero(SemanticType.QTY),
        avg_entry_price=None,
    )
    
    exec_out = exec_logic.compute(signal_out, gate, ledger, current_price)
    print(f"   Action: {exec_out.action}")
    if exec_out.orders:
        print(f"   Order: {exec_out.orders[0].side} {exec_out.orders[0].qty}")
    
    print("\n=== Phase 2 v4 Demo Complete ===")


if __name__ == "__main__":
    demonstrate()
