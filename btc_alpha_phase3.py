"""
BTC ALPHA SYSTEM - PHASE 3
===========================

Real-Data Pipeline Validation

This module implements:
1. Data Adapter (CSV/Parquet → HistoricalViews)
2. HTF Aggregation (1m → 4H with lag enforcement)
3. Friction Model (slippage + fees)
4. Backtest Runner (minute-by-minute simulation)
5. Evaluation Artifacts (deterministic hashing)

LOCKED DECISIONS:
- Fill timing: minute t+1 open + slippage
- Friction: 10 bps fee + 10 bps slippage per side (CONSERVATIVE)
- Friction order: slippage first, then fees
- Gap policy: HALT by default
- Regime: always permissive (test simplification)
- Warmup: enforced

Author: Claude (Anthropic)
Phase: 3
Version: 1
Depends on: btc_alpha_phase2_v4.py (Phase 2), btc_alpha_v3_final.py (Phase 1)
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    Final,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

# =============================================================================
# PHASE 1/2 IMPORTS
# =============================================================================

_DEMO_MODE_ENV_VAR: Final[str] = "BTC_ALPHA_DEMO_MODE"
_DEMO_MODE_ENABLED: Final[bool] = os.environ.get(_DEMO_MODE_ENV_VAR, "0") == "1"

try:
    from btc_alpha_phase2_v4 import (
        # Types
        Fixed,
        SemanticType,
        RoundingMode,
        SEMANTIC_SCALES,
        # Errors
        DeterminismError,
        SemanticConsistencyError,
        LookaheadError,
        SeriesOrderingError,
        # Data structures
        Candle,
        HistoricalView1m,
        HistoricalView4H,
        CommittedHistory,
        IndicatorOutput,
        PositionView,
        SignalOutput,
        Signal,
        GateView,
        LedgerView,
        OrderIntent,
        ExecutionOutput,
        TradeRecord,
        EvaluationMetrics,
        # Modules
        BaselineIndicatorConfig,
        BaselineIndicatorModule,
        BaselineSignalConfig,
        BaselineSignalModule,
        BaselineExecutionConfig,
        BaselineExecutionLogic,
        # Functions
        create_gate_view,
        compute_metrics,
        hash_trades,
        hash_metrics,
        _integer_divide_with_rounding,
        SECONDS_PER_4H,
        is_4h_boundary,
        holding_time_minutes,
    )
    _PHASE2_IMPORT_SUCCESS = True
except ImportError as e:
    if not _DEMO_MODE_ENABLED:
        raise ImportError(
            f"Phase 2 module not found: {e}. "
            f"Phase 3 requires Phase 2 imports. "
            f"Set {_DEMO_MODE_ENV_VAR}=1 for demo mode."
        )
    _PHASE2_IMPORT_SUCCESS = False
    # Demo fallback would go here, but we'll require imports for now


# =============================================================================
# PHASE 3 ERRORS
# =============================================================================

class DataIntegrityError(DeterminismError):
    """Raised when data validation fails."""
    pass


class GapError(DataIntegrityError):
    """Raised when a gap is detected in data and gap policy is HALT."""
    pass


class OHLCError(DataIntegrityError):
    """Raised when OHLC consistency is violated."""
    pass


class BacktestError(DeterminismError):
    """Raised when backtest encounters an unrecoverable error."""
    pass


# =============================================================================
# CONSTANTS
# =============================================================================

SECONDS_PER_MINUTE: Final[int] = 60
MINUTES_PER_4H: Final[int] = 240

# Price bounds for BTC (sanity check)
MIN_VALID_PRICE_USD: Final[int] = 10000       # $100.00 in cents
MAX_VALID_PRICE_USD: Final[int] = 100000000   # $1,000,000.00 in cents


# =============================================================================
# CONFIGURATION
# =============================================================================

class GapPolicy(Enum):
    HALT = auto()
    FILL_FORWARD = auto()
    SKIP = auto()


class FrictionPreset(Enum):
    DEFENSIBLE = auto()   # 6+5 bps = ~22 bps RT
    CONSERVATIVE = auto() # 10+10 bps = 40 bps RT
    PUNITIVE = auto()     # 25+25 bps = 100 bps RT
    CRITICAL = auto()     # 50+50 bps = 200 bps RT


FRICTION_PRESETS: Final[Dict[FrictionPreset, Tuple[int, int]]] = {
    FrictionPreset.DEFENSIBLE: (6, 5),
    FrictionPreset.CONSERVATIVE: (10, 10),
    FrictionPreset.PUNITIVE: (25, 25),
    FrictionPreset.CRITICAL: (50, 50),
}


@dataclass(frozen=True)
class Phase3Config:
    """Phase 3 backtest configuration."""
    
    # Identity
    config_version: str = "phase3_v1"
    
    # Capital (in USD cents for Fixed compatibility)
    starting_capital_cents: int = 1000000  # $10,000.00
    
    # Friction preset (linked fee + slippage)
    friction_preset: FrictionPreset = FrictionPreset.CONSERVATIVE
    
    # Strategy sizing
    risk_fraction_bps: int = 200  # 2%
    min_qty_satoshis: int = 100000       # 0.001 BTC
    max_qty_satoshis: int = 1000000000   # 10 BTC
    
    # Indicators
    sma_fast_period: int = 20
    sma_slow_period: int = 50
    ema_period: int = 20
    donchian_period: int = 20
    atr_period: int = 14
    htf_sma_period: int = 20
    
    # Data handling
    gap_policy: GapPolicy = GapPolicy.HALT
    
    # Regime
    regime_always_permissive: bool = True
    
    @property
    def fee_rate_bps(self) -> int:
        return FRICTION_PRESETS[self.friction_preset][0]
    
    @property
    def slippage_rate_bps(self) -> int:
        return FRICTION_PRESETS[self.friction_preset][1]
    
    @property
    def starting_capital(self) -> Fixed:
        return Fixed(value=self.starting_capital_cents, sem=SemanticType.USD)
    
    @property
    def risk_fraction(self) -> Fixed:
        # RATE has scale 6, so 200 bps = 0.02 = 20000 in RATE scale
        return Fixed(value=self.risk_fraction_bps * 100, sem=SemanticType.RATE)
    
    @property
    def min_qty(self) -> Fixed:
        return Fixed(value=self.min_qty_satoshis, sem=SemanticType.QTY)
    
    @property
    def max_qty(self) -> Fixed:
        return Fixed(value=self.max_qty_satoshis, sem=SemanticType.QTY)
    
    def to_canonical(self) -> dict:
        """Canonical representation for hashing."""
        return {
            "config_version": self.config_version,
            "starting_capital_cents": self.starting_capital_cents,
            "friction_preset": self.friction_preset.name,
            "fee_rate_bps": self.fee_rate_bps,
            "slippage_rate_bps": self.slippage_rate_bps,
            "risk_fraction_bps": self.risk_fraction_bps,
            "min_qty_satoshis": self.min_qty_satoshis,
            "max_qty_satoshis": self.max_qty_satoshis,
            "sma_fast_period": self.sma_fast_period,
            "sma_slow_period": self.sma_slow_period,
            "ema_period": self.ema_period,
            "donchian_period": self.donchian_period,
            "atr_period": self.atr_period,
            "htf_sma_period": self.htf_sma_period,
            "gap_policy": self.gap_policy.name,
            "regime_always_permissive": self.regime_always_permissive,
        }


def hash_config(config: Phase3Config) -> str:
    """Compute deterministic hash of configuration."""
    canonical = config.to_canonical()
    json_bytes = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()


# =============================================================================
# DATA ADAPTER
# =============================================================================

# =============================================================================
# TIMESTAMP CONVENTION (CRITICAL FOR NO-LOOKAHEAD)
# =============================================================================
#
# LOCKED DECISION: Timestamps represent CANDLE CLOSE TIME
#
# A candle with timestamp T contains price action from [T - 60, T).
# At real-time moment T, the candle at T is JUST NOW closed and observable.
# The candle at T + 60 is still forming and NOT observable.
#
# For HTF (4H) aggregation:
# - A 4H candle closing at T is observable only when current_ts >= T
# - The 4H candle is built from 1m candles with timestamps in (T - 14400, T]
#
# This convention aligns with:
# - System Law: No Lookahead
# - Phase 2 Invariant: HTF lag enforcement
# =============================================================================

TIMESTAMP_CONVENTION: Final[str] = "CLOSE_TIME"


class ParquetValidationError(DataIntegrityError):
    """Raised when Parquet schema or data validation fails."""
    pass


@dataclass
class RawCandle:
    """Raw candle from CSV/Parquet before validation."""
    timestamp: int
    open_str: str
    high_str: str
    low_str: str
    close_str: str
    volume_str: str


def load_parquet(path: Path) -> List[RawCandle]:
    """
    Load raw candles from Parquet file.
    
    ADAPTER LAYER ONLY: Translates bytes to canonical structure.
    No inference, no indicator math.
    
    STRICT SCHEMA VALIDATION:
    - Required columns: timestamp (or open_time/close_time), open, high, low, close, volume
    - Timestamps must be integers (epoch seconds or milliseconds)
    - Prices and volumes must be numeric
    
    TIMESTAMP CONVENTION:
    - If 'close_time' column exists, use it
    - If 'open_time' column exists, add 60 seconds to get close time
    - If 'timestamp' column exists, assume it is close time
    - All timestamps converted to epoch seconds
    """
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet support. "
            "Install with: pip install pyarrow"
        )
    
    # Read Parquet file
    try:
        table = pq.read_table(str(path))
        df_columns = table.column_names
    except Exception as e:
        raise ParquetValidationError(f"Failed to read Parquet file: {e}")
    
    # Normalize column names to lowercase
    col_map = {c.lower(): c for c in df_columns}
    
    # Identify timestamp column
    ts_col = None
    ts_is_open_time = False
    
    if 'close_time' in col_map:
        ts_col = col_map['close_time']
        ts_is_open_time = False
    elif 'closetime' in col_map:
        ts_col = col_map['closetime']
        ts_is_open_time = False
    elif 'open_time' in col_map:
        ts_col = col_map['open_time']
        ts_is_open_time = True
    elif 'opentime' in col_map:
        ts_col = col_map['opentime']
        ts_is_open_time = True
    elif 'timestamp' in col_map:
        ts_col = col_map['timestamp']
        ts_is_open_time = False  # Assume close time
    elif 'time' in col_map:
        ts_col = col_map['time']
        ts_is_open_time = False
    elif 'ts' in col_map:
        ts_col = col_map['ts']
        ts_is_open_time = False
    else:
        raise ParquetValidationError(
            f"No timestamp column found. Available columns: {df_columns}"
        )
    
    # Identify OHLCV columns
    def find_col(names: List[str]) -> Optional[str]:
        for n in names:
            if n in col_map:
                return col_map[n]
        return None
    
    open_col = find_col(['open', 'o'])
    high_col = find_col(['high', 'h'])
    low_col = find_col(['low', 'l'])
    close_col = find_col(['close', 'c'])
    volume_col = find_col(['volume', 'vol', 'v', 'quote_volume', 'base_volume'])
    
    missing = []
    if open_col is None:
        missing.append('open')
    if high_col is None:
        missing.append('high')
    if low_col is None:
        missing.append('low')
    if close_col is None:
        missing.append('close')
    if volume_col is None:
        missing.append('volume')
    
    if missing:
        raise ParquetValidationError(
            f"Missing required columns: {missing}. Available: {df_columns}"
        )
    
    # Convert to Python for processing
    # (Using to_pydict for memory efficiency with large files)
    data = table.to_pydict()
    
    timestamps = data[ts_col]
    opens = data[open_col]
    highs = data[high_col]
    lows = data[low_col]
    closes = data[close_col]
    volumes = data[volume_col]
    
    n_rows = len(timestamps)
    if n_rows == 0:
        raise ParquetValidationError("Parquet file contains no data")
    
    candles: List[RawCandle] = []
    
    for i in range(n_rows):
        # Parse timestamp
        ts_raw = timestamps[i]
        
        # Handle various timestamp formats
        if isinstance(ts_raw, (int, float)):
            ts_int = int(ts_raw)
        else:
            # Try to parse as string
            try:
                ts_int = int(ts_raw)
            except (ValueError, TypeError):
                raise ParquetValidationError(
                    f"Invalid timestamp at row {i}: {ts_raw}"
                )
        
        # Detect milliseconds vs seconds
        # Timestamps > 1e12 are likely milliseconds
        if ts_int > 1_000_000_000_000:
            ts_int = ts_int // 1000  # Convert ms to seconds
        
        # If this is open_time, add 60 seconds to get close_time
        if ts_is_open_time:
            ts_int = ts_int + SECONDS_PER_MINUTE
        
        # Parse OHLCV as strings (validation happens later)
        candle = RawCandle(
            timestamp=ts_int,
            open_str=str(opens[i]),
            high_str=str(highs[i]),
            low_str=str(lows[i]),
            close_str=str(closes[i]),
            volume_str=str(volumes[i]),
        )
        candles.append(candle)
    
    return candles


@dataclass
class ValidationReport:
    """Report from data validation."""
    total_records: int
    valid_records: int
    gaps_detected: int
    gap_minutes: List[Tuple[int, int]]  # (expected_ts, actual_ts)
    ohlc_errors: int
    price_bound_errors: int
    is_valid: bool
    error_message: Optional[str]


def load_csv(path: Path) -> List[RawCandle]:
    """
    Load raw candles from CSV file.
    
    Expected columns: timestamp, open, high, low, close, volume
    Timestamp should be Unix epoch seconds (close time per convention).
    """
    candles: List[RawCandle] = []
    
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Handle various column naming conventions
            ts_key = None
            ts_is_open_time = False
            
            for key in ["close_time", "closetime", "timestamp", "time", "ts"]:
                if key in row:
                    ts_key = key
                    ts_is_open_time = False
                    break
            
            if ts_key is None:
                for key in ["open_time", "opentime"]:
                    if key in row:
                        ts_key = key
                        ts_is_open_time = True
                        break
            
            if ts_key is None:
                raise DataIntegrityError("No timestamp column found in CSV")
            
            ts_raw = int(row[ts_key])
            
            # Detect milliseconds vs seconds
            if ts_raw > 1_000_000_000_000:
                ts_raw = ts_raw // 1000
            
            # Convert open_time to close_time if needed
            if ts_is_open_time:
                ts_raw = ts_raw + SECONDS_PER_MINUTE
            
            candle = RawCandle(
                timestamp=ts_raw,
                open_str=row.get("open", row.get("o", "")),
                high_str=row.get("high", row.get("h", "")),
                low_str=row.get("low", row.get("l", "")),
                close_str=row.get("close", row.get("c", "")),
                volume_str=row.get("volume", row.get("v", row.get("vol", "0"))),
            )
            candles.append(candle)
    
    return candles


def load_data(path: Path) -> List[RawCandle]:
    """
    Unified data loader: detects format from file extension.
    
    Supports:
    - .csv: CSV format
    - .parquet, .pq: Parquet format
    
    All formats normalized to the same RawCandle structure.
    Timestamp convention: CLOSE_TIME (epoch seconds).
    """
    suffix = path.suffix.lower()
    
    if suffix == '.csv':
        return load_csv(path)
    elif suffix in ('.parquet', '.pq'):
        return load_parquet(path)
    else:
        # Try CSV as fallback
        try:
            return load_csv(path)
        except Exception:
            raise DataIntegrityError(
                f"Unknown file format: {suffix}. Supported: .csv, .parquet"
            )


def validate_candles(
    raw_candles: List[RawCandle],
    gap_policy: GapPolicy,
) -> Tuple[List[Candle], ValidationReport]:
    """
    Validate raw candles and convert to typed Candles.
    
    Checks:
    - Monotonic timestamps
    - No gaps (or handle per policy)
    - OHLC consistency
    - Price bounds
    """
    if not raw_candles:
        return [], ValidationReport(
            total_records=0,
            valid_records=0,
            gaps_detected=0,
            gap_minutes=[],
            ohlc_errors=0,
            price_bound_errors=0,
            is_valid=False,
            error_message="No candles to validate",
        )
    
    # FIX #1: DO NOT SORT - enforce strict non-decreasing timestamps as read
    # Sorting masks upstream ordering faults and violates deny-by-default
    
    valid_candles: List[Candle] = []
    gaps: List[Tuple[int, int]] = []
    ohlc_errors = 0
    price_errors = 0
    duplicate_errors = 0
    ordering_errors = 0
    prev_ts: Optional[int] = None
    seen_timestamps: set[int] = set()
    
    for raw in raw_candles:
        # Check for duplicates
        if raw.timestamp in seen_timestamps:
            duplicate_errors += 1
            if gap_policy == GapPolicy.HALT:
                raise DataIntegrityError(
                    f"Duplicate timestamp detected: {raw.timestamp}"
                )
            continue  # Skip duplicate in non-HALT modes
        seen_timestamps.add(raw.timestamp)
        
        # Check strict ordering (non-decreasing)
        if prev_ts is not None and raw.timestamp < prev_ts:
            ordering_errors += 1
            if gap_policy == GapPolicy.HALT:
                raise DataIntegrityError(
                    f"Out-of-order timestamp: {raw.timestamp} after {prev_ts}. "
                    f"Input data must be pre-sorted."
                )
            # In non-HALT modes, we cannot safely continue with unordered data
            raise DataIntegrityError(
                f"Out-of-order data cannot be processed even in non-HALT mode. "
                f"Pre-sort your data before ingestion."
            )
        # Check timestamp alignment
        if raw.timestamp % SECONDS_PER_MINUTE != 0:
            raise DataIntegrityError(
                f"Timestamp {raw.timestamp} not aligned to minute boundary"
            )
        
        # Check for gaps
        if prev_ts is not None:
            expected_ts = prev_ts + SECONDS_PER_MINUTE
            if raw.timestamp != expected_ts:
                gap = (expected_ts, raw.timestamp)
                gaps.append(gap)
                
                if gap_policy == GapPolicy.HALT:
                    raise GapError(
                        f"Gap detected: expected {expected_ts}, got {raw.timestamp}"
                    )
                elif gap_policy == GapPolicy.FILL_FORWARD:
                    # Fill gap with previous close
                    prev_candle = valid_candles[-1] if valid_candles else None
                    if prev_candle:
                        fill_ts = expected_ts
                        while fill_ts < raw.timestamp:
                            fill_candle = Candle(
                                ts=fill_ts,
                                open=prev_candle.close,
                                high=prev_candle.close,
                                low=prev_candle.close,
                                close=prev_candle.close,
                                volume=Fixed.zero(SemanticType.QTY),
                            )
                            valid_candles.append(fill_candle)
                            fill_ts += SECONDS_PER_MINUTE
                # SKIP policy: just continue, gap remains
        
        # Parse prices
        try:
            open_price = Fixed.from_str(raw.open_str, SemanticType.PRICE)
            high_price = Fixed.from_str(raw.high_str, SemanticType.PRICE)
            low_price = Fixed.from_str(raw.low_str, SemanticType.PRICE)
            close_price = Fixed.from_str(raw.close_str, SemanticType.PRICE)
            volume = Fixed.from_str(raw.volume_str, SemanticType.QTY)
        except Exception as e:
            raise DataIntegrityError(f"Failed to parse candle at {raw.timestamp}: {e}")
        
        # OHLC consistency
        if high_price.value < open_price.value or high_price.value < close_price.value:
            ohlc_errors += 1
            if gap_policy == GapPolicy.HALT:
                raise OHLCError(
                    f"OHLC error at {raw.timestamp}: high < open or close"
                )
        
        if low_price.value > open_price.value or low_price.value > close_price.value:
            ohlc_errors += 1
            if gap_policy == GapPolicy.HALT:
                raise OHLCError(
                    f"OHLC error at {raw.timestamp}: low > open or close"
                )
        
        # Price bounds
        for price, name in [
            (open_price, "open"),
            (high_price, "high"),
            (low_price, "low"),
            (close_price, "close"),
        ]:
            if price.value < MIN_VALID_PRICE_USD or price.value > MAX_VALID_PRICE_USD:
                price_errors += 1
                if gap_policy == GapPolicy.HALT:
                    raise DataIntegrityError(
                        f"Price out of bounds at {raw.timestamp}: {name}={price}"
                    )
        
        # Create validated candle
        candle = Candle(
            ts=raw.timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )
        valid_candles.append(candle)
        prev_ts = raw.timestamp
    
    report = ValidationReport(
        total_records=len(raw_candles),
        valid_records=len(valid_candles),
        gaps_detected=len(gaps),
        gap_minutes=gaps,
        ohlc_errors=ohlc_errors,
        price_bound_errors=price_errors,
        is_valid=(len(gaps) == 0 or gap_policy != GapPolicy.HALT) and ohlc_errors == 0 and duplicate_errors == 0 and ordering_errors == 0,
        error_message=None if (ohlc_errors == 0 and duplicate_errors == 0) else f"{ohlc_errors} OHLC errors, {duplicate_errors} duplicates",
    )
    
    return valid_candles, report


def build_historical_view_1m(
    all_candles: Tuple[Candle, ...],
    current_ts: int,
    max_lookback: int,
) -> HistoricalView1m:
    """
    Build 1m historical view up to (but not including) current minute.
    
    Returns candles from [current_ts - max_lookback*60, current_ts - 60].
    """
    # Find candles that closed before current_ts
    visible = [c for c in all_candles if c.ts < current_ts]
    
    # Limit to max_lookback
    if len(visible) > max_lookback:
        visible = visible[-max_lookback:]
    
    return HistoricalView1m(
        candles=tuple(visible),
        current_ts=current_ts,
    )


def aggregate_to_4h(
    candles_1m: Tuple[Candle, ...],
    current_ts: int,
) -> HistoricalView4H:
    """
    Aggregate 1m candles into 4H candles.
    
    Only FULLY CLOSED 4H bars are included.
    A 4H bar closing at T is visible only when current_ts > T.
    
    NOTE: This function is O(n) and should not be called in a loop.
    For backtest simulation, use precompute_4h_index() instead.
    """
    if not candles_1m:
        return HistoricalView4H(candles=(), current_ts=current_ts)
    
    # Group 1m candles by 4H period
    # 4H period for a candle: floor(ts / SECONDS_PER_4H) * SECONDS_PER_4H
    periods: Dict[int, List[Candle]] = {}
    
    for c in candles_1m:
        period_start = (c.ts // SECONDS_PER_4H) * SECONDS_PER_4H
        period_end = period_start + SECONDS_PER_4H  # Close time of this 4H bar
        
        if period_end not in periods:
            periods[period_end] = []
        periods[period_end].append(c)
    
    # Build 4H candles only for completed periods
    candles_4h: List[Candle] = []
    
    for period_close_ts in sorted(periods.keys()):
        # Only include if fully closed (period_close_ts < current_ts)
        if period_close_ts >= current_ts:
            continue
        
        period_candles = periods[period_close_ts]
        if not period_candles:
            continue
        
        # Check we have all 240 minutes (or close to it for the period)
        # For now, aggregate what we have
        
        open_price = period_candles[0].open
        high_price = max(period_candles, key=lambda c: c.high.value).high
        low_price = min(period_candles, key=lambda c: c.low.value).low
        close_price = period_candles[-1].close
        total_volume = sum(c.volume.value for c in period_candles)
        
        candle_4h = Candle(
            ts=period_close_ts,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Fixed(value=total_volume, sem=SemanticType.QTY),
        )
        candles_4h.append(candle_4h)
    
    return HistoricalView4H(
        candles=tuple(candles_4h),
        current_ts=current_ts,
    )


# =============================================================================
# OPTIMIZED PRE-COMPUTATION FOR BACKTEST
# =============================================================================

@dataclass
class PrecomputedHTF:
    """
    Pre-computed 4H candles with index for O(1) lookups.
    
    This avoids recomputing 4H aggregation every minute during simulation.
    """
    # All 4H candles, sorted by close time
    candles_4h: Tuple[Candle, ...]
    
    # Index: 4H close time -> index in candles_4h tuple
    # Allows binary search for "all 4H candles closed before current_ts"
    close_times: Tuple[int, ...]
    
    def get_visible_4h(self, current_ts: int, max_lookback: int = 100) -> HistoricalView4H:
        """
        Get 4H candles visible at current_ts (O(log n) lookup).
        
        A 4H candle is visible if its close_time < current_ts.
        """
        if not self.close_times:
            return HistoricalView4H(candles=(), current_ts=current_ts)
        
        # Binary search: find rightmost index where close_time < current_ts
        # All candles at indices [0, right_idx] are visible
        left, right = 0, len(self.close_times)
        while left < right:
            mid = (left + right) // 2
            if self.close_times[mid] < current_ts:
                left = mid + 1
            else:
                right = mid
        
        # left is now the first index where close_time >= current_ts
        # So visible candles are [0, left)
        if left == 0:
            return HistoricalView4H(candles=(), current_ts=current_ts)
        
        # Limit to max_lookback
        start_idx = max(0, left - max_lookback)
        visible = self.candles_4h[start_idx:left]
        
        return HistoricalView4H(candles=visible, current_ts=current_ts)


def precompute_4h_index(candles_1m: Tuple[Candle, ...]) -> PrecomputedHTF:
    """
    Pre-compute all 4H candles from 1m data.
    
    Call once before simulation, then use get_visible_4h() for O(1) lookups.
    """
    if not candles_1m:
        return PrecomputedHTF(candles_4h=(), close_times=())
    
    # Group 1m candles by 4H period
    periods: Dict[int, List[Candle]] = {}
    
    for c in candles_1m:
        period_start = (c.ts // SECONDS_PER_4H) * SECONDS_PER_4H
        period_end = period_start + SECONDS_PER_4H
        
        if period_end not in periods:
            periods[period_end] = []
        periods[period_end].append(c)
    
    # Build all 4H candles (including incomplete final period)
    candles_4h: List[Candle] = []
    
    for period_close_ts in sorted(periods.keys()):
        period_candles = periods[period_close_ts]
        if not period_candles:
            continue
        
        # Sort by timestamp to ensure correct OHLC
        period_candles.sort(key=lambda c: c.ts)
        
        open_price = period_candles[0].open
        high_price = max(period_candles, key=lambda c: c.high.value).high
        low_price = min(period_candles, key=lambda c: c.low.value).low
        close_price = period_candles[-1].close
        total_volume = sum(c.volume.value for c in period_candles)
        
        candle_4h = Candle(
            ts=period_close_ts,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Fixed(value=total_volume, sem=SemanticType.QTY),
        )
        candles_4h.append(candle_4h)
    
    close_times = tuple(c.ts for c in candles_4h)
    
    return PrecomputedHTF(
        candles_4h=tuple(candles_4h),
        close_times=close_times,
    )


# =============================================================================
# FRICTION MODEL
# =============================================================================

def apply_slippage(
    price: Fixed,
    side: Literal["BUY", "SELL"],
    slippage_bps: int,
) -> Fixed:
    """
    Apply slippage to price.
    
    BUY: price increases (worse for buyer)
    SELL: price decreases (worse for seller)
    
    slippage_bps is per-side basis points.
    """
    if price.sem != SemanticType.PRICE:
        raise SemanticConsistencyError(f"apply_slippage expects PRICE, got {price.sem}")
    
    # slippage_rate = slippage_bps / 10000
    # For BUY: new_price = price * (1 + slippage_rate)
    #        = price + price * slippage_bps / 10000
    # In integer: (price.value * slippage_bps) / 10000
    
    adjustment = _integer_divide_with_rounding(
        price.value * slippage_bps,
        10000,
        RoundingMode.AWAY_FROM_ZERO,  # Always punitive
    )
    
    if side == "BUY":
        new_value = price.value + adjustment
    else:  # SELL
        new_value = price.value - adjustment
    
    return Fixed(value=new_value, sem=SemanticType.PRICE)


def compute_fee(
    price_after_slippage: Fixed,
    qty: Fixed,
    fee_bps: int,
) -> Fixed:
    """
    Compute fee based on post-slippage notional.
    
    fee = qty * price_after_slippage * fee_rate
    
    Returns fee in USD.
    """
    if price_after_slippage.sem != SemanticType.PRICE:
        raise SemanticConsistencyError("compute_fee expects PRICE for price")
    if qty.sem != SemanticType.QTY:
        raise SemanticConsistencyError("compute_fee expects QTY for qty")
    
    # notional = qty * price (QTY * PRICE → USD)
    # QTY scale = 8, PRICE scale = 2, USD scale = 2
    # notional_raw = qty.value * price.value (scale = 10)
    # notional_usd = notional_raw / 10^8 (to get scale 2)
    
    notional_raw = qty.value * price_after_slippage.value
    notional_scale = SEMANTIC_SCALES[SemanticType.QTY] + SEMANTIC_SCALES[SemanticType.PRICE]
    usd_scale = SEMANTIC_SCALES[SemanticType.USD]
    
    notional_usd = _integer_divide_with_rounding(
        notional_raw,
        10 ** (notional_scale - usd_scale),
        RoundingMode.TRUNCATE,
    )
    
    # fee = notional_usd * fee_bps / 10000
    fee_value = _integer_divide_with_rounding(
        notional_usd * fee_bps,
        10000,
        RoundingMode.AWAY_FROM_ZERO,  # Always punitive (more fees)
    )
    
    return Fixed(value=fee_value, sem=SemanticType.USD)


# =============================================================================
# SIMULATED LEDGER
# =============================================================================

@dataclass
class SimulatedLedger:
    """
    Simulated ledger for backtesting.
    
    Tracks cash, position, and equity.
    """
    cash: Fixed  # USD
    position_qty: Fixed  # QTY (positive = long, negative = short)
    avg_entry_price: Optional[Fixed]  # PRICE
    total_fees_paid: Fixed  # USD
    total_slippage_cost: Fixed  # USD
    realized_pnl: Fixed  # USD
    
    def equity(self, mark_price: Fixed) -> Fixed:
        """Compute equity at mark price."""
        if self.position_qty.is_zero():
            return self.cash
        
        # unrealized = qty * (mark - entry) for long
        # unrealized = qty * (entry - mark) for short (qty is negative)
        # Simplified: unrealized = qty * (mark - entry)
        
        if self.avg_entry_price is None:
            return self.cash
        
        # PnL per unit = mark - entry
        pnl_per_unit = mark_price.value - self.avg_entry_price.value
        
        # Total unrealized = qty * pnl_per_unit
        # QTY scale = 8, PRICE diff scale = 2
        # Result needs USD scale = 2
        # unrealized_raw = qty.value * pnl_per_unit (scale = 8 + 2 = 10)
        unrealized_raw = self.position_qty.value * pnl_per_unit
        
        unrealized_usd = _integer_divide_with_rounding(
            unrealized_raw,
            10 ** (SEMANTIC_SCALES[SemanticType.QTY]),  # Scale down by QTY scale
            RoundingMode.TRUNCATE,
        )
        
        return Fixed(value=self.cash.value + unrealized_usd, sem=SemanticType.USD)
    
    def leverage_bps(self, mark_price: Fixed) -> int:
        """Compute leverage in basis points (10000 = 1x)."""
        eq = self.equity(mark_price)
        if eq.value <= 0:
            return 0
        
        if self.position_qty.is_zero():
            return 0
        
        # notional = abs(qty) * mark_price
        notional_raw = abs(self.position_qty.value) * mark_price.value
        notional_usd = _integer_divide_with_rounding(
            notional_raw,
            10 ** SEMANTIC_SCALES[SemanticType.QTY],
            RoundingMode.TRUNCATE,
        )
        
        # leverage = notional / equity
        # leverage_bps = notional * 10000 / equity
        leverage_bps = _integer_divide_with_rounding(
            notional_usd * 10000,
            eq.value,
            RoundingMode.TRUNCATE,
        )
        
        return leverage_bps


def apply_fill(
    ledger: SimulatedLedger,
    side: Literal["BUY", "SELL"],
    qty: Fixed,
    fill_price: Fixed,
    fee: Fixed,
    slippage_cost: Fixed,
) -> SimulatedLedger:
    """
    Apply a fill to the ledger.
    
    Returns new ledger state.
    """
    # Compute notional
    notional_raw = qty.value * fill_price.value
    notional_usd = _integer_divide_with_rounding(
        notional_raw,
        10 ** SEMANTIC_SCALES[SemanticType.QTY],
        RoundingMode.TRUNCATE,
    )
    
    # Update cash
    if side == "BUY":
        # Cash decreases by notional + fee
        new_cash = ledger.cash.value - notional_usd - fee.value
        new_qty = ledger.position_qty.value + qty.value
    else:  # SELL
        # Cash increases by notional - fee
        new_cash = ledger.cash.value + notional_usd - fee.value
        new_qty = ledger.position_qty.value - qty.value
    
    # Update average entry price
    if new_qty == 0:
        new_avg_entry = None
    elif ledger.position_qty.is_zero():
        # Fresh entry
        new_avg_entry = fill_price
    elif (ledger.position_qty.value > 0 and side == "BUY") or \
         (ledger.position_qty.value < 0 and side == "SELL"):
        # Adding to position - compute weighted average
        # old_notional = old_qty * old_avg_entry (in QTY scale * PRICE scale)
        # new_notional = new_qty * fill_price (in QTY scale * PRICE scale)
        # avg = (old_notional + new_notional) / total_qty
        # Result is in PRICE scale (QTY cancels out)
        
        old_notional = abs(ledger.position_qty.value) * ledger.avg_entry_price.value
        new_notional = qty.value * fill_price.value
        total_qty = abs(ledger.position_qty.value) + qty.value
        
        # FIX #2: The result is already in PRICE scale after dividing by QTY
        # No second rescale needed!
        avg_value = _integer_divide_with_rounding(
            old_notional + new_notional,
            total_qty,
            RoundingMode.TRUNCATE,
        )
        new_avg_entry = Fixed(value=avg_value, sem=SemanticType.PRICE)
    else:
        # Reducing position - keep existing average for remaining
        new_avg_entry = ledger.avg_entry_price
    
    # Compute realized PnL if closing/reducing
    realized_delta = Fixed.zero(SemanticType.USD)
    if ledger.avg_entry_price is not None:
        if (ledger.position_qty.value > 0 and side == "SELL") or \
           (ledger.position_qty.value < 0 and side == "BUY"):
            # Closing/reducing
            close_qty = min(abs(ledger.position_qty.value), qty.value)
            pnl_per_unit = fill_price.value - ledger.avg_entry_price.value
            if ledger.position_qty.value < 0:
                pnl_per_unit = -pnl_per_unit  # Short position
            
            realized_raw = close_qty * pnl_per_unit
            realized_delta_value = _integer_divide_with_rounding(
                realized_raw,
                10 ** SEMANTIC_SCALES[SemanticType.QTY],
                RoundingMode.TRUNCATE,
            )
            realized_delta = Fixed(value=realized_delta_value, sem=SemanticType.USD)
    
    return SimulatedLedger(
        cash=Fixed(value=new_cash, sem=SemanticType.USD),
        position_qty=Fixed(value=new_qty, sem=SemanticType.QTY),
        avg_entry_price=new_avg_entry,
        total_fees_paid=Fixed(
            value=ledger.total_fees_paid.value + fee.value,
            sem=SemanticType.USD,
        ),
        total_slippage_cost=Fixed(
            value=ledger.total_slippage_cost.value + slippage_cost.value,
            sem=SemanticType.USD,
        ),
        realized_pnl=Fixed(
            value=ledger.realized_pnl.value + realized_delta.value,
            sem=SemanticType.USD,
        ),
    )


# =============================================================================
# BACKTEST RESULT
# =============================================================================

@dataclass(frozen=True)
class BacktestResult:
    """Result of a backtest run."""
    trades: Tuple[TradeRecord, ...]
    equity_series: Tuple[Tuple[int, Fixed], ...]
    leverage_series: Tuple[Tuple[int, int], ...]
    metrics: EvaluationMetrics
    config: Phase3Config
    
    # Hashes
    trades_hash: str
    equity_hash: str
    metrics_hash: str
    config_hash: str
    
    @classmethod
    def compute_hashes(
        cls,
        trades: Sequence[TradeRecord],
        equity_series: Sequence[Tuple[int, Fixed]],
        metrics: EvaluationMetrics,
        config: Phase3Config,
    ) -> Tuple[str, str, str, str]:
        """Compute all hashes."""
        trades_hash = hash_trades(trades)
        
        # Equity series hash
        equity_canonical = [
            {"ts": ts, "equity": eq.to_canonical()}
            for ts, eq in equity_series
        ]
        equity_json = json.dumps(equity_canonical, sort_keys=True, separators=(",", ":"))
        equity_hash = hashlib.sha256(equity_json.encode()).hexdigest()
        
        metrics_hash = hash_metrics(metrics)
        config_hash = hash_config(config)
        
        return trades_hash, equity_hash, metrics_hash, config_hash


# =============================================================================
# BACKTEST RUNNER
# =============================================================================

class BacktestRunner:
    """
    Minute-by-minute backtest runner.
    """
    
    def __init__(self, config: Phase3Config):
        self.config = config
        
        # Initialize modules from Phase 2
        self.indicator_config = BaselineIndicatorConfig(
            sma_fast_period=config.sma_fast_period,
            sma_slow_period=config.sma_slow_period,
            ema_period=config.ema_period,
            donchian_period=config.donchian_period,
            atr_period=config.atr_period,
            htf_sma_period=config.htf_sma_period,
        )
        self.indicator_module = BaselineIndicatorModule(self.indicator_config)
        
        self.signal_module = BaselineSignalModule(BaselineSignalConfig())
        
        self.execution_config = BaselineExecutionConfig(
            risk_fraction=config.risk_fraction,
            min_qty=config.min_qty,
            max_qty=config.max_qty,
        )
        self.execution_logic = BaselineExecutionLogic(self.execution_config)
        
        self.max_lookback_1m = self.indicator_module.max_lookback_1m
        self.max_lookback_4h = self.indicator_module.max_lookback_4h
    
    def run(
        self,
        candles_1m: Tuple[Candle, ...],
        start_ts: int,
        end_ts: int,
    ) -> BacktestResult:
        """
        Run backtest from start_ts to end_ts.
        
        start_ts and end_ts are epoch seconds.
        
        OPTIMIZATION: Uses pre-computed indices for O(n) total complexity
        instead of O(n²) from recomputing aggregations each minute.
        """
        # Initialize ledger
        ledger = SimulatedLedger(
            cash=self.config.starting_capital,
            position_qty=Fixed.zero(SemanticType.QTY),
            avg_entry_price=None,
            total_fees_paid=Fixed.zero(SemanticType.USD),
            total_slippage_cost=Fixed.zero(SemanticType.USD),
            realized_pnl=Fixed.zero(SemanticType.USD),
        )
        
        trades: List[TradeRecord] = []
        equity_series: List[Tuple[int, Fixed]] = []
        leverage_series: List[Tuple[int, int]] = []
        
        # Track open trade for trade record creation
        entry_ts: Optional[int] = None
        entry_price: Optional[Fixed] = None
        entry_qty: Optional[Fixed] = None
        entry_side: Optional[Literal["LONG", "SHORT"]] = None
        trade_fees: int = 0
        trade_slippage: int = 0
        
        # =====================================================================
        # PRE-COMPUTATION PHASE (O(n) once, not O(n) per minute)
        # =====================================================================
        
        # Build candle index for fast lookup
        candle_by_ts: Dict[int, Candle] = {c.ts: c for c in candles_1m}
        
        # Pre-compute 4H aggregation
        htf_index = precompute_4h_index(candles_1m)
        
        # Build sorted list of 1m candle timestamps for efficient slicing
        sorted_1m_candles = sorted(candles_1m, key=lambda c: c.ts)
        candle_index_by_ts: Dict[int, int] = {
            c.ts: i for i, c in enumerate(sorted_1m_candles)
        }
        
        # =====================================================================
        # SIMULATION LOOP (O(1) per minute)
        # =====================================================================
        
        current_ts = start_ts
        total_minutes = (end_ts - start_ts) // SECONDS_PER_MINUTE + 1
        processed = 0
        
        while current_ts <= end_ts:
            # Progress logging (every 100k minutes = ~70 days)
            processed += 1
            if processed % 10000 == 0:
                pct = (processed / total_minutes) * 100
                print(f"  Progress: {processed}/{total_minutes} ({pct:.1f}%)")
            
            # FIX #4: Under HALT policy, missing candles in run loop should raise GapError
            if current_ts not in candle_by_ts:
                if self.config.gap_policy == GapPolicy.HALT:
                    raise GapError(
                        f"Missing candle at {current_ts} during backtest run. "
                        f"Gap policy is HALT - cannot continue."
                    )
                current_ts += SECONDS_PER_MINUTE
                continue
            
            current_candle = candle_by_ts[current_ts]
            
            # Build 1m historical view (O(1) slice using pre-built index)
            if current_ts in candle_index_by_ts:
                current_idx = candle_index_by_ts[current_ts]
                # Get candles BEFORE current_ts (not including current)
                start_idx = max(0, current_idx - self.max_lookback_1m - 100)
                visible_1m = tuple(sorted_1m_candles[start_idx:current_idx])
            else:
                visible_1m = ()
            
            view_1m = HistoricalView1m(candles=visible_1m, current_ts=current_ts)
            
            # Build 4H historical view (O(log n) lookup using pre-computed index)
            view_4h = htf_index.get_visible_4h(current_ts, max_lookback=self.max_lookback_4h + 10)
            
            history = CommittedHistory(view_1m=view_1m, view_4h=view_4h)
            
            # Compute indicators
            indicators = self.indicator_module.compute(history)
            
            # Build gates
            gates = create_gate_view(
                diagnostics_ok=True,
                diagnostics_veto=False,
                regime_permissive=self.config.regime_always_permissive,
                indicators=indicators,
            )
            
            # Current price for decision (minute t close)
            decision_price = current_candle.close
            
            # Build position view
            position = PositionView(
                position_qty=ledger.position_qty,
                avg_entry_price=ledger.avg_entry_price,
            )
            
            # Generate signal
            signal = self.signal_module.compute(indicators, position, decision_price)
            
            # Generate execution intent
            ledger_view = LedgerView(
                equity=ledger.equity(decision_price),
                position_qty=ledger.position_qty,
                avg_entry_price=ledger.avg_entry_price,
            )
            
            execution = self.execution_logic.compute(
                signal, gates, ledger_view, decision_price
            )
            
            # Execute orders at next minute open
            if execution.orders:
                # Get next minute for fill
                next_ts = current_ts + SECONDS_PER_MINUTE
                if next_ts in candle_by_ts:
                    next_candle = candle_by_ts[next_ts]
                    fill_reference_price = next_candle.open
                    
                    for order in execution.orders:
                        # Apply slippage
                        fill_price = apply_slippage(
                            fill_reference_price,
                            order.side,
                            self.config.slippage_rate_bps,
                        )
                        
                        # Compute slippage cost
                        slippage_cost_value = abs(fill_price.value - fill_reference_price.value)
                        # FIX #3: Use AWAY_FROM_ZERO for slippage cost (punitive)
                        slippage_notional = _integer_divide_with_rounding(
                            slippage_cost_value * order.qty.value,
                            10 ** SEMANTIC_SCALES[SemanticType.QTY],
                            RoundingMode.AWAY_FROM_ZERO,  # Punitive
                        )
                        slippage_cost = Fixed(value=slippage_notional, sem=SemanticType.USD)
                        
                        # Compute fee
                        fee = compute_fee(
                            fill_price,
                            order.qty,
                            self.config.fee_rate_bps,
                        )
                        
                        # Track for trade record
                        was_flat = ledger.position_qty.is_zero()
                        prev_side = "LONG" if ledger.position_qty.is_positive() else "SHORT"
                        
                        # Apply fill
                        ledger = apply_fill(
                            ledger,
                            order.side,
                            order.qty,
                            fill_price,
                            fee,
                            slippage_cost,
                        )
                        
                        # Trade record management
                        if was_flat and execution.action == "ENTER":
                            # Opening trade
                            entry_ts = next_ts
                            entry_price = fill_price
                            entry_qty = order.qty
                            entry_side = "LONG" if order.side == "BUY" else "SHORT"
                            trade_fees = fee.value
                            trade_slippage = slippage_cost.value
                        
                        elif execution.action == "EXIT" and entry_ts is not None:
                            # Closing trade
                            trade_fees += fee.value
                            trade_slippage += slippage_cost.value
                            
                            # Compute PnL
                            if entry_side == "LONG":
                                gross_pnl_per_unit = fill_price.value - entry_price.value
                            else:
                                gross_pnl_per_unit = entry_price.value - fill_price.value
                            
                            gross_pnl_raw = entry_qty.value * gross_pnl_per_unit
                            gross_pnl_value = _integer_divide_with_rounding(
                                gross_pnl_raw,
                                10 ** SEMANTIC_SCALES[SemanticType.QTY],
                                RoundingMode.TRUNCATE,
                            )
                            
                            net_pnl_value = gross_pnl_value - trade_fees - trade_slippage
                            
                            trade = TradeRecord(
                                entry_ts=entry_ts,
                                exit_ts=next_ts,
                                side=entry_side,
                                entry_price=entry_price,
                                exit_price=fill_price,
                                qty=entry_qty,
                                gross_pnl=Fixed(value=gross_pnl_value, sem=SemanticType.USD),
                                fees=Fixed(value=trade_fees, sem=SemanticType.USD),
                                slippage=Fixed(value=trade_slippage, sem=SemanticType.USD),
                                net_pnl=Fixed(value=net_pnl_value, sem=SemanticType.USD),
                            )
                            trades.append(trade)
                            
                            # Reset
                            entry_ts = None
                            entry_price = None
                            entry_qty = None
                            entry_side = None
                            trade_fees = 0
                            trade_slippage = 0
            
            # Record equity and leverage
            mark_price = current_candle.close
            equity_series.append((current_ts, ledger.equity(mark_price)))
            leverage_series.append((current_ts, ledger.leverage_bps(mark_price)))
            
            current_ts += SECONDS_PER_MINUTE
        
        # Compute metrics
        metrics = compute_metrics(
            trades,
            equity_series,
            leverage_series,
        )
        
        # Compute hashes
        trades_hash, equity_hash, metrics_hash, config_hash = BacktestResult.compute_hashes(
            trades, equity_series, metrics, self.config
        )
        
        return BacktestResult(
            trades=tuple(trades),
            equity_series=tuple(equity_series),
            leverage_series=tuple(leverage_series),
            metrics=metrics,
            config=self.config,
            trades_hash=trades_hash,
            equity_hash=equity_hash,
            metrics_hash=metrics_hash,
            config_hash=config_hash,
        )


# =============================================================================
# DEMONSTRATION
# =============================================================================

def create_demo_data() -> Tuple[Candle, ...]:
    """Create synthetic demo data for testing."""
    import random
    random.seed(42)  # Deterministic
    
    candles: List[Candle] = []
    base_price = 4500000  # $45,000.00 in cents
    
    # Generate 1 week of 1m data
    start_ts = 1672531200  # 2023-01-01 00:00:00 UTC
    num_minutes = 7 * 24 * 60  # 1 week
    
    price = base_price
    
    for i in range(num_minutes):
        ts = start_ts + i * 60
        
        # Random walk
        change = random.randint(-5000, 5000)  # $50 max change
        price = max(100000, price + change)  # Min $1000
        
        # OHLC
        high = price + random.randint(0, 2000)
        low = price - random.randint(0, 2000)
        open_p = price + random.randint(-1000, 1000)
        open_p = max(low, min(high, open_p))
        close_p = price
        close_p = max(low, min(high, close_p))
        
        volume = random.randint(100000, 10000000)  # 0.001 to 0.1 BTC
        
        candle = Candle(
            ts=ts,
            open=Fixed(value=open_p, sem=SemanticType.PRICE),
            high=Fixed(value=high, sem=SemanticType.PRICE),
            low=Fixed(value=low, sem=SemanticType.PRICE),
            close=Fixed(value=close_p, sem=SemanticType.PRICE),
            volume=Fixed(value=volume, sem=SemanticType.QTY),
        )
        candles.append(candle)
    
    return tuple(candles)


def demonstrate():
    """Demonstrate Phase 3 components."""
    print("=== BTC Alpha System Phase 3 Demo ===\n")
    
    print(f"Phase 2 import success: {_PHASE2_IMPORT_SUCCESS}")
    print()
    
    # FIX #2 verification: Test average entry price calculation
    print("0. Average Entry Price Calculation Test:")
    
    # Start with empty ledger
    test_ledger = SimulatedLedger(
        cash=Fixed(value=10000000, sem=SemanticType.USD),  # $100,000
        position_qty=Fixed.zero(SemanticType.QTY),
        avg_entry_price=None,
        total_fees_paid=Fixed.zero(SemanticType.USD),
        total_slippage_cost=Fixed.zero(SemanticType.USD),
        realized_pnl=Fixed.zero(SemanticType.USD),
    )
    
    # Buy 1 BTC at $40,000
    price_40k = Fixed(value=4000000, sem=SemanticType.PRICE)  # $40,000.00
    qty_1btc = Fixed(value=100000000, sem=SemanticType.QTY)   # 1.0 BTC
    
    test_ledger = apply_fill(
        test_ledger, "BUY", qty_1btc, price_40k,
        Fixed.zero(SemanticType.USD), Fixed.zero(SemanticType.USD)
    )
    print(f"   After buying 1 BTC @ $40,000:")
    print(f"   Position: {test_ledger.position_qty.value / 100000000:.8f} BTC")
    print(f"   Avg Entry: ${test_ledger.avg_entry_price.value / 100:,.2f}")
    
    # Buy 1 more BTC at $60,000
    price_60k = Fixed(value=6000000, sem=SemanticType.PRICE)  # $60,000.00
    
    test_ledger = apply_fill(
        test_ledger, "BUY", qty_1btc, price_60k,
        Fixed.zero(SemanticType.USD), Fixed.zero(SemanticType.USD)
    )
    print(f"   After buying 1 more BTC @ $60,000:")
    print(f"   Position: {test_ledger.position_qty.value / 100000000:.8f} BTC")
    print(f"   Avg Entry: ${test_ledger.avg_entry_price.value / 100:,.2f}")
    
    # Verify: average should be $50,000
    expected_avg = 5000000  # $50,000.00 in cents
    if test_ledger.avg_entry_price.value == expected_avg:
        print(f"   ✓ CORRECT: Average entry is $50,000 as expected")
    else:
        print(f"   ✗ ERROR: Expected ${expected_avg/100:,.2f}, got ${test_ledger.avg_entry_price.value/100:,.2f}")
    
    print()
    
    # Create config
    config = Phase3Config()
    print("1. Configuration:")
    print(f"   Friction preset: {config.friction_preset.name}")
    print(f"   Fee rate: {config.fee_rate_bps} bps per side")
    print(f"   Slippage rate: {config.slippage_rate_bps} bps per side")
    print(f"   Total round-trip: {2 * (config.fee_rate_bps + config.slippage_rate_bps)} bps")
    print(f"   Starting capital: ${config.starting_capital_cents / 100:,.2f}")
    print(f"   Config hash: {hash_config(config)[:16]}...")
    
    # Demo data
    print("\n2. Demo Data Generation:")
    candles = create_demo_data()
    print(f"   Generated {len(candles)} 1m candles")
    print(f"   Range: {candles[0].ts} to {candles[-1].ts}")
    
    # HTF aggregation
    print("\n3. HTF Aggregation:")
    current_ts = candles[-1].ts + 60
    view_4h = aggregate_to_4h(candles, current_ts)
    print(f"   4H bars created: {len(view_4h.candles)}")
    if view_4h.candles:
        print(f"   Latest 4H close: {view_4h.candles[-1].close}")
    
    # Friction model
    print("\n4. Friction Model:")
    test_price = Fixed(value=4500000, sem=SemanticType.PRICE)  # $45,000
    
    slipped_buy = apply_slippage(test_price, "BUY", config.slippage_rate_bps)
    slipped_sell = apply_slippage(test_price, "SELL", config.slippage_rate_bps)
    print(f"   Reference price: ${test_price.value / 100:,.2f}")
    print(f"   After slippage (BUY):  ${slipped_buy.value / 100:,.2f}")
    print(f"   After slippage (SELL): ${slipped_sell.value / 100:,.2f}")
    
    test_qty = Fixed(value=10000000, sem=SemanticType.QTY)  # 0.1 BTC
    fee = compute_fee(slipped_buy, test_qty, config.fee_rate_bps)
    print(f"   Fee on 0.1 BTC @ slipped price: ${fee.value / 100:,.2f}")
    
    # Run backtest
    print("\n5. Backtest Run:")
    runner = BacktestRunner(config)
    
    start_ts = candles[0].ts + runner.max_lookback_1m * 60  # After warmup
    end_ts = candles[-1].ts
    
    print(f"   Running from {start_ts} to {end_ts}...")
    result = runner.run(candles, start_ts, end_ts)
    
    print(f"\n6. Results:")
    print(f"   Trades: {result.metrics.trade_count}")
    print(f"   Net PnL: ${result.metrics.net_pnl.value / 100:,.2f}")
    print(f"   Max Drawdown: ${result.metrics.max_drawdown.value / 100:,.2f}")
    print(f"   Max Drawdown %: {result.metrics.max_drawdown_pct_bps / 100:.2f}%")
    print(f"   Win Rate: {result.metrics.win_rate_bps / 100:.1f}%")
    print(f"   Avg Holding: {result.metrics.avg_holding_minutes} minutes")
    
    print(f"\n7. Determinism Hashes:")
    print(f"   Trades:  {result.trades_hash[:16]}...")
    print(f"   Equity:  {result.equity_hash[:16]}...")
    print(f"   Metrics: {result.metrics_hash[:16]}...")
    print(f"   Config:  {result.config_hash[:16]}...")
    
    # Verify determinism
    print("\n8. Determinism Verification:")
    result2 = runner.run(candles, start_ts, end_ts)
    
    hashes_match = (
        result.trades_hash == result2.trades_hash and
        result.equity_hash == result2.equity_hash and
        result.metrics_hash == result2.metrics_hash
    )
    print(f"   Run 1 vs Run 2 hashes match: {hashes_match}")
    
    print("\n=== Phase 3 Demo Complete ===")


if __name__ == "__main__":
    demonstrate()
