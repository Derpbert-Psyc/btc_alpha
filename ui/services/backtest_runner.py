"""Backtest Runner — resolved config + 1m OHLCV data → List[TradeEvent].

Bridges the gap between the compiler output (resolved Strategy Framework v1.8.0 config)
and the triage input (List[TradeEvent]).

Key design:
  - Calls evaluate_signal_pipeline() as THE decision function
  - Stop-loss, trailing, time-limit evaluated at bar close only (not intrabar)
  - Deterministic: same config + same data = same trades
  - Trade IDs derived from content (hash), not random
"""

import hashlib
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from strategy_framework_v1_8_0 import (
    INDICATOR_ID_TO_NAME,
    INDICATOR_NAME_TO_ID,
    INDICATOR_OUTPUTS,
    MTMDrawdownTracker,
    SignalResult,
    compute_instance_warmup,
    evaluate_signal_pipeline,
    is_output_warmed_up,
)
from phase5_triage_types import TradeEvent
from btc_alpha_v3_final import Fixed, SemanticType

# ---------------------------------------------------------------------------
# OHLCV Bar
# ---------------------------------------------------------------------------

class Bar:
    """Simple OHLCV bar."""
    __slots__ = ("ts", "open", "high", "low", "close", "volume", "index")

    def __init__(self, ts: int, o: float, h: float, l: float, c: float,
                 v: float = 0.0, index: int = 0):
        self.ts = ts
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v
        self.index = index


# ---------------------------------------------------------------------------
# Timeframe resampling
# ---------------------------------------------------------------------------

TIMEFRAME_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "12h": 43200, "1d": 86400, "3d": 259200,
    "1w": 604800,
}


def parse_timeframe_seconds(tf: str) -> int:
    """Parse timeframe string to seconds."""
    if tf in TIMEFRAME_SECONDS:
        return TIMEFRAME_SECONDS[tf]
    # Custom format: "15min", "4hr", etc.
    import re
    m = re.match(r'^(\d+)(m|min|h|hr|d|day|w|wk)$', tf)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        if unit in ("m", "min"):
            return val * 60
        elif unit in ("h", "hr"):
            return val * 3600
        elif unit in ("d", "day"):
            return val * 86400
        elif unit in ("w", "wk"):
            return val * 604800
    raise ValueError(f"Unknown timeframe: {tf}")


def resample_bars(bars_1m: List[Bar], tf_seconds: int) -> List[Bar]:
    """Resample 1m bars to higher timeframe using standard OHLC aggregation.

    Groups 1m bars into tf_seconds intervals:
    - Open: first open in group
    - High: max high in group
    - Low: min low in group
    - Close: last close in group
    - Volume: sum of volumes
    """
    if tf_seconds <= 60:
        return list(bars_1m)

    if not bars_1m:
        return []

    result = []
    group_start = None
    group_o = group_h = group_l = group_c = group_v = 0.0
    group_count = 0
    bar_idx = 0

    for bar in bars_1m:
        # Determine which group this bar belongs to
        bucket = (bar.ts // tf_seconds) * tf_seconds
        if group_start is None or bucket != group_start:
            # Emit previous group
            if group_start is not None and group_count > 0:
                result.append(Bar(
                    ts=group_start, o=group_o, h=group_h,
                    l=group_l, c=group_c, v=group_v,
                    index=bar_idx,
                ))
                bar_idx += 1
            group_start = bucket
            group_o = bar.open
            group_h = bar.high
            group_l = bar.low
            group_c = bar.close
            group_v = bar.volume
            group_count = 1
        else:
            group_h = max(group_h, bar.high)
            group_l = min(group_l, bar.low)
            group_c = bar.close
            group_v += bar.volume
            group_count += 1

    # Emit last group
    if group_start is not None and group_count > 0:
        result.append(Bar(
            ts=group_start, o=group_o, h=group_h,
            l=group_l, c=group_c, v=group_v,
            index=bar_idx,
        ))

    return result


# ---------------------------------------------------------------------------
# Indicator Implementations
# ---------------------------------------------------------------------------

def compute_ema(closes: List[float], period: int) -> List[Optional[float]]:
    """Compute EMA. Returns list parallel to closes."""
    if period < 1:
        return [None] * len(closes)
    if period == 1:
        return list(closes)  # Passthrough

    result = [None] * len(closes)
    if len(closes) < period:
        return result

    # SMA seed
    sma = sum(closes[:period]) / period
    result[period - 1] = sma
    k = 2.0 / (period + 1)
    prev = sma
    for i in range(period, len(closes)):
        val = closes[i] * k + prev * (1 - k)
        result[i] = val
        prev = val
    return result


def compute_sma(closes: List[float], period: int) -> List[Optional[float]]:
    """Compute SMA."""
    result = [None] * len(closes)
    if period < 1 or len(closes) < period:
        return result
    window_sum = sum(closes[:period])
    result[period - 1] = window_sum / period
    for i in range(period, len(closes)):
        window_sum += closes[i] - closes[i - period]
        result[i] = window_sum / period
    return result


def compute_rsi(closes: List[float], period: int = 14) -> List[Optional[float]]:
    """Compute RSI using Wilder's smoothing."""
    result = [None] * len(closes)
    if len(closes) < period + 1:
        return result

    gains = []
    losses = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    # Initial average
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return result


def compute_atr(highs: List[float], lows: List[float], closes: List[float],
                period: int = 14) -> List[Optional[float]]:
    """Compute ATR using Wilder's smoothing."""
    result = [None] * len(closes)
    if len(closes) < 2:
        return result

    true_ranges = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return result

    # SMA seed
    atr_val = sum(true_ranges[:period]) / period
    result[period - 1] = atr_val
    for i in range(period, len(true_ranges)):
        atr_val = (atr_val * (period - 1) + true_ranges[i]) / period
        result[i] = atr_val

    return result


def compute_macd(closes: List[float], fast: int = 12, slow: int = 26,
                 signal: int = 9) -> Tuple[
                     List[Optional[float]], List[Optional[float]],
                     List[Optional[float]], List[Optional[float]],
                     List[Optional[float]]]:
    """Compute MACD: macd_line, signal_line, histogram, slope_sign, signal_slope_sign."""
    fast_ema = compute_ema(closes, fast)
    slow_ema = compute_ema(closes, slow)

    n = len(closes)
    macd_line = [None] * n
    for i in range(n):
        if fast_ema[i] is not None and slow_ema[i] is not None:
            macd_line[i] = fast_ema[i] - slow_ema[i]

    # Signal line = EMA of MACD line
    macd_values = [v for v in macd_line if v is not None]
    sig_ema = compute_ema(macd_values, signal) if len(macd_values) >= signal else [None] * len(macd_values)

    signal_line = [None] * n
    histogram = [None] * n
    slope_sign = [None] * n

    macd_idx = 0
    for i in range(n):
        if macd_line[i] is not None:
            if macd_idx < len(sig_ema) and sig_ema[macd_idx] is not None:
                signal_line[i] = sig_ema[macd_idx]
                histogram[i] = macd_line[i] - sig_ema[macd_idx]
            macd_idx += 1

    # Slope sign: 1 if macd rising, -1 if falling, 0 if flat
    for i in range(1, n):
        if macd_line[i] is not None and macd_line[i - 1] is not None:
            diff = macd_line[i] - macd_line[i - 1]
            slope_sign[i] = 1 if diff > 0 else (-1 if diff < 0 else 0)

    # Signal slope sign: 1 if signal rising, -1 if falling, 0 if flat
    signal_slope_sign = [None] * n
    for i in range(1, n):
        if signal_line[i] is not None and signal_line[i - 1] is not None:
            diff = signal_line[i] - signal_line[i - 1]
            signal_slope_sign[i] = 1 if diff > 0 else (-1 if diff < 0 else 0)

    return macd_line, signal_line, histogram, slope_sign, signal_slope_sign


def compute_lmagr(closes: List[float], ma_length: int = 20) -> Tuple[
                      List[Optional[int]], List[Optional[int]]]:
    """Compute LMAGR (Log MA Gap Ratio): lmagr, lmagr_pct.

    Mirrors Phase 4B LMAGRIndicator (indicator 25) math exactly:
    - First-price-seeded EMA (NOT SMA-seeded; do NOT call compute_ema()).
    - Integer fixed-point EMA with SCALE_FACTOR = 10^10.
    - PRICE-scaled inputs (close in cents).
    - Outputs are integer-scaled by RATE_SCALE = 1,000,000.
    - int() truncation, not round().

    Returns: (lmagr_list, lmagr_pct_list) — parallel to closes.
    """
    n = len(closes)
    lmagr_list: List[Optional[int]] = [None] * n
    lmagr_pct_list: List[Optional[int]] = [None] * n

    SCALE_FACTOR = 10_000_000_000
    RATE_SCALE = 1_000_000
    k_scaled = (2 * SCALE_FACTOR) // (ma_length + 1)

    ema_value: Optional[int] = None
    bars_seen = 0

    for i in range(n):
        close_int = int(round(closes[i] * 100))  # PRICE-scaled (cents)

        # Gate: invalid close does not mutate state
        if close_int <= 0:
            continue

        bars_seen += 1

        # Update EMA — first-price-seeded (NOT SMA-seeded)
        if ema_value is None:
            ema_value = close_int
        else:
            ema_value = (
                close_int * k_scaled + ema_value * (SCALE_FACTOR - k_scaled)
            ) // SCALE_FACTOR

        # Warmup check
        if bars_seen < ma_length:
            continue

        if ema_value <= 0:
            continue

        # Compute LMAGR = ln(close / EMA)
        ratio = close_int / ema_value
        if ratio <= 0:
            continue

        lmagr_float = math.log(ratio)
        lmagr_list[i] = int(lmagr_float * RATE_SCALE)

        # Percentage form: (close/EMA - 1)
        lmagr_pct_list[i] = int((ratio - 1.0) * RATE_SCALE)

    return lmagr_list, lmagr_pct_list


def compute_bollinger(closes: List[float], period: int = 20,
                      num_std: float = 2.0) -> Tuple[
                          List[Optional[float]], List[Optional[float]],
                          List[Optional[float]], List[Optional[float]],
                          List[Optional[float]]]:
    """Compute Bollinger Bands: basis, upper, lower, bandwidth, percent_b."""
    n = len(closes)
    basis = [None] * n
    upper = [None] * n
    lower = [None] * n
    bandwidth = [None] * n
    percent_b = [None] * n

    sma = compute_sma(closes, period)

    for i in range(period - 1, n):
        if sma[i] is None:
            continue
        window = closes[i - period + 1:i + 1]
        mean = sma[i]
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)

        basis[i] = mean
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std
        if mean != 0:
            bandwidth[i] = (upper[i] - lower[i]) / mean
        else:
            bandwidth[i] = 0.0
        band_width = upper[i] - lower[i]
        if band_width != 0:
            percent_b[i] = (closes[i] - lower[i]) / band_width
        else:
            percent_b[i] = 0.5

    return basis, upper, lower, bandwidth, percent_b


def compute_choppiness(highs: List[float], lows: List[float],
                       closes: List[float], period: int = 14) -> List[Optional[float]]:
    """Compute Choppiness Index."""
    n = len(closes)
    result = [None] * n
    atr_values = compute_atr(highs, lows, closes, 1)  # 1-period ATR = true range

    for i in range(period - 1, n):
        # Sum of true ranges over period
        atr_sum = 0.0
        valid = True
        for j in range(i - period + 1, i + 1):
            if atr_values[j] is None:
                valid = False
                break
            atr_sum += atr_values[j]
        if not valid:
            continue

        # Highest high - lowest low over period
        hh = max(highs[i - period + 1:i + 1])
        ll = min(lows[i - period + 1:i + 1])
        hl_range = hh - ll
        if hl_range <= 0:
            result[i] = 100.0
            continue

        ratio = atr_sum / hl_range
        if ratio <= 0:
            result[i] = 0.0
        else:
            result[i] = 100.0 * math.log10(ratio) / math.log10(period)

    return result


def compute_donchian(highs: List[float], lows: List[float],
                     period: int = 20) -> Tuple[
                         List[Optional[float]], List[Optional[float]],
                         List[Optional[float]]]:
    """Compute Donchian Channels: upper, lower, basis."""
    n = len(highs)
    upper_ch = [None] * n
    lower_ch = [None] * n
    basis_ch = [None] * n

    for i in range(period - 1, n):
        hh = max(highs[i - period + 1:i + 1])
        ll = min(lows[i - period + 1:i + 1])
        upper_ch[i] = hh
        lower_ch[i] = ll
        basis_ch[i] = (hh + ll) / 2.0

    return upper_ch, lower_ch, basis_ch


def compute_linreg_slope(closes: List[float], period: int = 14) -> List[Optional[float]]:
    """Compute linear regression slope."""
    n = len(closes)
    result = [None] * n

    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        # Linear regression slope
        x_mean = (period - 1) / 2.0
        y_mean = sum(window) / period
        num = 0.0
        den = 0.0
        for j, y in enumerate(window):
            num += (j - x_mean) * (y - y_mean)
            den += (j - x_mean) ** 2
        if den != 0:
            result[i] = num / den
        else:
            result[i] = 0.0

    return result


# ---------------------------------------------------------------------------
# Diagnostic probe helpers (RATE_SCALE integer outputs)
# ---------------------------------------------------------------------------

RATE_SCALE = 1_000_000


def compute_donchian_position(
    bars: List[Bar],
    dc_upper: List[Optional[float]],
    dc_lower: List[Optional[float]],
    length: int = 288,
) -> Dict[str, List[Optional[int]]]:
    """Compute Donchian Position outputs: 7 RATE_SCALE integer lists parallel to bars.

    Uses pre-computed Donchian upper/lower from the dependency indicator.
    Mirrors Phase 4B DonchianPositionIndicator (probe 30) math.
    """
    n = len(bars)
    percent_b: List[Optional[int]] = [None] * n
    bars_since_upper: List[Optional[int]] = [None] * n
    bars_since_lower: List[Optional[int]] = [None] * n
    retrace_from_lower: List[Optional[int]] = [None] * n
    retrace_from_upper: List[Optional[int]] = [None] * n
    new_upper: List[Optional[int]] = [None] * n
    new_lower: List[Optional[int]] = [None] * n

    prev_upper_c: Optional[int] = None
    prev_lower_c: Optional[int] = None

    for i in range(n):
        if dc_upper[i] is None or dc_lower[i] is None:
            continue

        upper = dc_upper[i]
        lower = dc_lower[i]
        close = bars[i].close

        # Convert to integer cents for truncate-division matching engine
        close_c = int(round(close * 100))
        upper_c = int(round(upper * 100))
        lower_c = int(round(lower * 100))
        range_c = upper_c - lower_c

        # percent_b
        if range_c > 0:
            percent_b[i] = (close_c - lower_c) * RATE_SCALE // range_c
        else:
            percent_b[i] = RATE_SCALE // 2

        # bars_since — forward scan oldest→newest, first match = earliest occurrence
        start = max(0, i - length + 1)
        bsu = i - start
        for j in range(start, i + 1):
            if bars[j].high == upper:  # exact float equality OK (max of same values)
                bsu = i - j
                break
        bars_since_upper[i] = bsu

        bsl = i - start
        for j in range(start, i + 1):
            if bars[j].low == lower:
                bsl = i - j
                break
        bars_since_lower[i] = bsl

        # retrace from lower/upper
        retrace_from_lower[i] = (close_c - lower_c) * RATE_SCALE // lower_c if lower_c > 0 else 0
        retrace_from_upper[i] = (upper_c - close_c) * RATE_SCALE // upper_c if upper_c > 0 else 0

        # new_upper / new_lower transitions
        new_upper[i] = RATE_SCALE if (prev_upper_c is not None and upper_c != prev_upper_c) else 0
        new_lower[i] = RATE_SCALE if (prev_lower_c is not None and lower_c != prev_lower_c) else 0

        prev_upper_c = upper_c
        prev_lower_c = lower_c

    return {
        "percent_b": percent_b,
        "bars_since_upper": bars_since_upper,
        "bars_since_lower": bars_since_lower,
        "retrace_from_lower": retrace_from_lower,
        "retrace_from_upper": retrace_from_upper,
        "new_upper": new_upper,
        "new_lower": new_lower,
    }


def compute_vol_regime(
    bars: List[Bar],
    dc_upper: List[Optional[float]],
    dc_lower: List[Optional[float]],
    reference_vol_microbps: int = 3_333_365,
) -> Dict[str, List[Optional[int]]]:
    """Compute Volatility Regime: vol_ratio as RATE_SCALE integer list.

    Mirrors Phase 4B VolRegimeIndicator (probe 31) math:
    vol_ratio = dc_range * 100 * RATE_SCALE * 1_000_000 // (close * reference_vol_microbps)
    """
    n = len(bars)
    vol_ratio: List[Optional[int]] = [None] * n

    for i in range(n):
        if dc_upper[i] is None or dc_lower[i] is None:
            continue
        close = bars[i].close
        if close <= 0:
            continue

        upper_c = int(round(dc_upper[i] * 100))
        lower_c = int(round(dc_lower[i] * 100))
        close_c = int(round(close * 100))
        range_c = upper_c - lower_c

        vol_ratio[i] = range_c * 100 * RATE_SCALE * 1_000_000 // (close_c * reference_vol_microbps)

    return {"vol_ratio": vol_ratio}


# ---------------------------------------------------------------------------
# Indicator computation for a resolved config
# ---------------------------------------------------------------------------

def compute_indicator_outputs(
    instance: dict,
    bars: List[Bar],
    dep_outputs: Optional[Dict[str, List]] = None,
) -> Dict[str, List[Optional[float]]]:
    """Compute all outputs for an indicator instance on resampled bars."""
    ind_id = instance["indicator_id"]
    if isinstance(ind_id, str):
        ind_id = INDICATOR_NAME_TO_ID.get(ind_id, -1)

    params = instance.get("parameters", {})
    outputs_used = instance.get("outputs_used", [])

    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]

    result = {}

    if ind_id == 1:  # EMA
        period = int(params.get("period", params.get("length", 20)))
        ema_vals = compute_ema(closes, period)
        result["ema"] = ema_vals

    elif ind_id == 2:  # RSI
        period = int(params.get("period", params.get("length", 14)))
        result["rsi"] = compute_rsi(closes, period)

    elif ind_id == 3:  # ATR
        period = int(params.get("period", params.get("length", 14)))
        result["atr"] = compute_atr(highs, lows, closes, period)

    elif ind_id == 7:  # MACD
        fast = int(params.get("fast_period", params.get("fast", 12)))
        slow = int(params.get("slow_period", params.get("slow", 26)))
        sig = int(params.get("signal_period", params.get("signal", 9)))
        ml, sl, hist, ss, sss = compute_macd(closes, fast, slow, sig)
        result["macd_line"] = ml
        result["signal_line"] = sl
        result["histogram"] = hist
        result["slope_sign"] = ss
        result["signal_slope_sign"] = sss

    elif ind_id == 8:  # ROC
        period = int(params.get("period", params.get("length", 14)))
        roc_vals = [None] * len(closes)
        for i in range(period, len(closes)):
            if closes[i - period] != 0:
                roc_vals[i] = ((closes[i] - closes[i - period]) / closes[i - period]) * 100
        result["roc"] = roc_vals

    elif ind_id == 10:  # Choppiness
        period = int(params.get("period", params.get("length", 14)))
        result["choppiness"] = compute_choppiness(highs, lows, closes, period)

    elif ind_id == 11:  # Bollinger
        period = int(params.get("period", params.get("length", 20)))
        num_std = params.get("num_std", 2.0)
        basis, upper, lower, bw, pb = compute_bollinger(closes, period, num_std)
        result["basis"] = basis
        result["upper"] = upper
        result["lower"] = lower
        result["bandwidth"] = bw
        result["percent_b"] = pb

    elif ind_id == 12:  # LinReg
        period = int(params.get("period", params.get("length", 14)))
        result["slope"] = compute_linreg_slope(closes, period)

    elif ind_id == 14:  # Donchian
        period = int(params.get("period", params.get("length", 20)))
        upper, lower, basis = compute_donchian(highs, lows, period)
        result["upper"] = upper
        result["lower"] = lower
        result["basis"] = basis

    elif ind_id == 25:  # LMAGR (diagnostic probe)
        ma_length = params.get("ma_length", params.get("period", params.get("length", 20)))
        lmagr, lmagr_pct = compute_lmagr(closes, ma_length)
        result["lmagr"] = lmagr
        result["lmagr_pct"] = lmagr_pct

    elif ind_id == 30:  # Donchian Position (diagnostic probe, depends on 14)
        length = params.get("length", params.get("period", 288))
        if dep_outputs is not None:
            dc_upper = dep_outputs.get("upper", [None] * len(bars))
            dc_lower = dep_outputs.get("lower", [None] * len(bars))
        else:
            # Fallback: compute Donchian ourselves
            dc_upper, dc_lower, _ = compute_donchian(highs, lows, length)
        result = compute_donchian_position(bars, dc_upper, dc_lower, length)

    elif ind_id == 31:  # Volatility Regime (diagnostic probe, depends on 14)
        ref = params.get("reference_vol_microbps", 3_333_365)
        if dep_outputs is not None:
            dc_upper = dep_outputs.get("upper", [None] * len(bars))
            dc_lower = dep_outputs.get("lower", [None] * len(bars))
        else:
            dc_period = params.get("length", params.get("period", 288))
            dc_upper, dc_lower, _ = compute_donchian(highs, lows, dc_period)
        result = compute_vol_regime(bars, dc_upper, dc_lower, ref)

    else:
        # Stub: return None for all outputs
        known_outputs = INDICATOR_OUTPUTS.get(ind_id, {})
        for out_name in known_outputs:
            result[out_name] = [None] * len(bars)

    return result


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def _make_trade_id(entry_idx: int, exit_idx: int, side: str,
                   strategy_hash: str) -> str:
    """Deterministic trade ID from content."""
    content = f"{entry_idx}:{exit_idx}:{side}:{strategy_hash}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _to_fixed_price(value: float) -> Fixed:
    """Convert float to Fixed PRICE (scale=2, cents)."""
    return Fixed(value=round(value * 100), sem=SemanticType.PRICE)


def _to_fixed_qty(value: float) -> Fixed:
    """Convert float to Fixed QTY (scale=8, satoshis)."""
    return Fixed(value=round(value * 1e8), sem=SemanticType.QTY)


def compute_gross_return_bps(entry_price: float, exit_price: float,
                             side: str) -> int:
    """Compute gross return in basis points."""
    if entry_price == 0:
        return 0
    if side == "long":
        ret = (exit_price - entry_price) / entry_price
    else:
        ret = (entry_price - exit_price) / entry_price
    return round(ret * 10000)


def run_backtest(
    resolved_config: dict,
    bars_1m: List[Bar],
    strategy_hash: str = "",
    fee_override_bps: Optional[int] = None,
    slippage_override_bps: Optional[int] = None,
) -> Tuple[List[TradeEvent], List[float], int]:
    """Run backtest on resolved config with 1m bar data.

    Args:
        fee_override_bps: If provided, overrides configured fee (per fill, in bps).
        slippage_override_bps: If provided, overrides configured slippage (per fill, in bps).

    Returns:
        trades: List[TradeEvent]
        close_prices: List[float] (one per 1m bar)
        bar_count: int
    """
    instances = resolved_config.get("indicator_instances", [])
    entry_rules = resolved_config.get("entry_rules", [])
    exit_rules = resolved_config.get("exit_rules", [])
    gate_rules = resolved_config.get("gate_rules", [])
    exec_params = resolved_config.get("execution_params", {})
    flip_enabled = exec_params.get("flip_enabled", False)

    # ATR / execution_params stop-loss config
    sl_config = exec_params.get("stop_loss") or {}
    sl_mode = sl_config.get("mode")  # "ATR_MULTIPLE" or "FIXED_PERCENT" or None
    sl_atr_multiple = float(sl_config.get("atr_multiple", 1.5))
    sl_atr_label = sl_config.get("atr_indicator_label")
    sl_fixed_percent = float(sl_config.get("fixed_percent", 0.0))

    # Post-exit cooldown
    cooldown_bars = exec_params.get("post_exit_cooldown_bars", 0)
    last_exit_eval_bar = -999999

    # Determine evaluation cadence (from entry rules, default 1m)
    eval_cadence_sec = 60
    for rule in entry_rules:
        cad = rule.get("evaluation_cadence", "1m")
        cad_sec = parse_timeframe_seconds(cad)
        eval_cadence_sec = max(eval_cadence_sec, cad_sec)
    # Number of 1m bars per evaluation cycle
    eval_cadence_bars = eval_cadence_sec // 60

    # Resample bars for each instance's timeframe
    instance_bars: Dict[str, List[Bar]] = {}
    instance_outputs: Dict[str, Dict[str, List[Optional[float]]]] = {}
    instance_tf_sec: Dict[str, int] = {}

    # Precompute a mapping from 1m bar timestamp → resampled bar index for each instance.
    # This avoids the bucket alignment issues with non-minute-aligned start times.
    instance_ts_to_rsidx: Dict[str, Dict[int, int]] = {}

    def _resample_and_map(inst: dict) -> None:
        """Resample bars and build ts→index map for an indicator instance."""
        label = inst["label"]
        tf = inst.get("timeframe", "1m")
        tf_sec = parse_timeframe_seconds(tf)
        instance_tf_sec[label] = tf_sec
        resampled = resample_bars(bars_1m, tf_sec)
        instance_bars[label] = resampled
        # Build ts → resampled index
        ts_map: Dict[int, int] = {}
        rs_idx = 0
        for bar_1m in bars_1m:
            while (rs_idx + 1 < len(resampled) and
                   resampled[rs_idx + 1].ts <= bar_1m.ts):
                rs_idx += 1
            if rs_idx < len(resampled) and resampled[rs_idx].ts <= bar_1m.ts:
                ts_map[bar_1m.ts] = rs_idx
        instance_ts_to_rsidx[label] = ts_map

    # Pass 1: base indicators (ind_id < 26)
    for inst in instances:
        ind_id = inst["indicator_id"]
        if isinstance(ind_id, str):
            ind_id = INDICATOR_NAME_TO_ID.get(ind_id, -1)
        if ind_id >= 26:
            continue  # Defer to pass 2
        _resample_and_map(inst)
        instance_outputs[inst["label"]] = compute_indicator_outputs(
            inst, instance_bars[inst["label"]])

    # Pass 2: dependent probes (ind_id >= 26) — resolve donchian_label dependency
    for inst in instances:
        ind_id = inst["indicator_id"]
        if isinstance(ind_id, str):
            ind_id = INDICATOR_NAME_TO_ID.get(ind_id, -1)
        if ind_id < 26:
            continue  # Already done
        _resample_and_map(inst)
        label = inst["label"]
        params = inst.get("parameters", {})
        dep_out = None
        if ind_id in (30, 31):
            dc_label = params.get("donchian_label")
            if dc_label and dc_label in instance_outputs:
                dep_out = instance_outputs[dc_label]
        instance_outputs[label] = compute_indicator_outputs(
            inst, instance_bars[label], dep_outputs=dep_out)

    # Compute effective warmup in 1m bars
    max_warmup_1m = 0
    for inst in instances:
        label = inst["label"]
        ind_id = inst["indicator_id"]
        if isinstance(ind_id, str):
            ind_id = INDICATOR_NAME_TO_ID.get(ind_id, 0)
        outputs_used = inst.get("outputs_used", [])
        warmup_bars = inst.get("warmup_bars")
        if warmup_bars is None:
            warmup_bars = compute_instance_warmup(ind_id, outputs_used,
                                                  inst.get("parameters", {}))
        tf_sec = instance_tf_sec[label]
        warmup_1m = warmup_bars * (tf_sec // 60) if tf_sec >= 60 else warmup_bars
        max_warmup_1m = max(max_warmup_1m, warmup_1m)

    # Position state
    position = None  # {"direction": "LONG"/"SHORT", "entry_price": float, "entry_idx": int}
    mtm_tracker = MTMDrawdownTracker()
    trades: List[TradeEvent] = []
    close_prices = [b.close for b in bars_1m]
    prev_outputs: Optional[Dict[str, Dict[str, Any]]] = None

    # Time limit tracking
    position_entry_eval_bar = 0
    eval_bar_count = 0

    # Trailing stop tracking
    trailing_peak = 0.0

    # Execution-params stop price (computed at entry from ATR or fixed percent)
    ep_stop_price: Optional[float] = None

    for bar_1m_idx, bar in enumerate(bars_1m):
        # Skip warmup
        if bar_1m_idx < max_warmup_1m:
            continue

        # Only evaluate at evaluation cadence: last 1m bar of each bucket
        # A cadence bucket completes when the NEXT bar would be in a new bucket
        next_ts = bar.ts + 60
        current_bucket = bar.ts // eval_cadence_sec
        next_bucket = next_ts // eval_cadence_sec
        if current_bucket == next_bucket and eval_cadence_sec > 60:
            continue  # Not the last bar of this cadence bucket

        eval_bar_count += 1

        # Assemble indicator_outputs for this bar
        indicator_outputs: Dict[str, Dict[str, Any]] = {}
        for inst in instances:
            label = inst["label"]
            outputs = instance_outputs[label]
            ts_map = instance_ts_to_rsidx[label]

            # Look up the resampled bar index for the current 1m bar
            rs_idx = ts_map.get(bar.ts)

            inst_outputs: Dict[str, Any] = {}
            if rs_idx is not None:
                for out_name, out_series in outputs.items():
                    if rs_idx < len(out_series):
                        inst_outputs[out_name] = out_series[rs_idx]
                    else:
                        inst_outputs[out_name] = None
            else:
                for out_name in outputs:
                    inst_outputs[out_name] = None

            indicator_outputs[label] = inst_outputs

        # Build position dict for framework
        pos_dict = None
        if position is not None:
            pos_dict = {
                "direction": position["direction"],
                "entry_price": round(position["entry_price"] * 100),  # To cents for framework
            }

        # Check price-based exits BEFORE signal pipeline
        # (stop loss, trailing stop, time limit — evaluated at close)
        exit_override_reason = None

        if position is not None:
            current_close = bar.close
            entry_price = position["entry_price"]
            direction = position["direction"]

            # Update trailing peak
            if direction == "LONG":
                trailing_peak = max(trailing_peak, current_close)
            else:
                trailing_peak = min(trailing_peak, current_close) if trailing_peak > 0 else current_close

            # Execution-params ATR/fixed stop (computed at entry, checked every bar)
            if ep_stop_price is not None:
                if direction == "LONG" and current_close <= ep_stop_price:
                    exit_override_reason = "STOP_LOSS"
                elif direction == "SHORT" and current_close >= ep_stop_price:
                    exit_override_reason = "STOP_LOSS"

            for exit_rule in exit_rules:
                if exit_override_reason is not None:
                    break
                et = exit_rule.get("type", "SIGNAL")

                if et == "STOP_LOSS":
                    params = exit_rule.get("parameters", {})
                    mode = params.get("mode", "FIXED_PERCENT")
                    if direction == "LONG":
                        pct_str = params.get("percent_long", "0")
                        pct = float(pct_str) if pct_str else 0
                        stop_price = entry_price * (1 - pct)
                        if current_close <= stop_price and pct > 0:
                            exit_override_reason = "STOP_LOSS"
                            break
                    else:
                        pct_str = params.get("percent_short", "0")
                        pct = float(pct_str) if pct_str else 0
                        stop_price = entry_price * (1 + pct)
                        if current_close >= stop_price and pct > 0:
                            exit_override_reason = "STOP_LOSS"
                            break

                elif et == "TRAILING_STOP":
                    params = exit_rule.get("parameters", {})
                    if direction == "LONG":
                        pct_str = params.get("percent_long", "0")
                        pct = float(pct_str) if pct_str else 0
                        trail_stop = trailing_peak * (1 - pct)
                        if current_close <= trail_stop and pct > 0 and trailing_peak > entry_price:
                            exit_override_reason = "TRAILING_STOP"
                            break
                    else:
                        pct_str = params.get("percent_short", "0")
                        pct = float(pct_str) if pct_str else 0
                        if trailing_peak > 0:
                            trail_stop = trailing_peak * (1 + pct)
                            if current_close >= trail_stop and pct > 0 and trailing_peak < entry_price:
                                exit_override_reason = "TRAILING_STOP"
                                break

                elif et == "TIME_LIMIT":
                    params = exit_rule.get("parameters", {})
                    limit_bars = params.get("time_limit_bars", 0)
                    if limit_bars > 0:
                        bars_in_pos = eval_bar_count - position_entry_eval_bar
                        if bars_in_pos >= limit_bars:
                            exit_override_reason = "TIME_LIMIT"
                            break

        # Call the framework's decision function
        if exit_override_reason:
            signal = SignalResult()
            signal.action = "EXIT"
            signal.exit_reason = exit_override_reason
        else:
            # Filter exit rules by applies_to for the current position direction.
            # The framework does not enforce applies_to, so the runner must.
            if position is not None:
                pos_dir = position["direction"]
                filtered_exits = [
                    r for r in exit_rules
                    if not r.get("applies_to") or pos_dir in r["applies_to"]
                ]
                eval_config = dict(resolved_config, exit_rules=filtered_exits)
            else:
                eval_config = resolved_config

            signal = evaluate_signal_pipeline(
                config=eval_config,
                indicator_outputs=indicator_outputs,
                prev_outputs=prev_outputs,
                position=pos_dict,
                mtm_tracker=mtm_tracker if position else None,
                current_price=round(bar.close * 100) if position else None,
            )

        # Helper: compute ATR-based stop price at entry
        def _compute_atr_stop(direction: str, entry_px: float) -> Optional[float]:
            # Check execution_params stop_loss
            if sl_mode == "ATR_MULTIPLE" and sl_atr_label:
                atr_out = indicator_outputs.get(sl_atr_label, {})
                atr_val = atr_out.get("atr")
                if atr_val is not None:
                    if direction == "LONG":
                        return entry_px - atr_val * sl_atr_multiple
                    else:
                        return entry_px + atr_val * sl_atr_multiple
            elif sl_mode == "FIXED_PERCENT" and sl_fixed_percent > 0:
                if direction == "LONG":
                    return entry_px * (1 - sl_fixed_percent)
                else:
                    return entry_px * (1 + sl_fixed_percent)
            # Check exit rules for ATR_MULTIPLE mode
            for rule in exit_rules:
                if rule.get("type") != "STOP_LOSS":
                    continue
                rp = rule.get("parameters", {})
                if rp.get("mode") == "ATR_MULTIPLE":
                    lbl = rp.get("atr_indicator_label")
                    mult = float(rp.get("atr_multiple", 1.5))
                    if lbl and lbl in indicator_outputs:
                        atr_val = indicator_outputs[lbl].get("atr")
                        if atr_val is not None:
                            if direction == "LONG":
                                return entry_px - atr_val * mult
                            else:
                                return entry_px + atr_val * mult
            return None

        # Process signal
        if signal.action == "ENTRY" and position is None:
            # Cooldown check: skip entry if within cooldown period after last exit
            if cooldown_bars > 0 and (eval_bar_count - last_exit_eval_bar) < cooldown_bars:
                pass  # Suppress entry due to cooldown
            else:
                position = {
                    "direction": signal.direction,
                    "entry_price": bar.close,
                    "entry_idx": bar_1m_idx,
                }
                position_entry_eval_bar = eval_bar_count
                trailing_peak = bar.close
                ep_stop_price = _compute_atr_stop(signal.direction, bar.close)
                mtm_tracker.open_position()

        elif signal.action == "EXIT" and position is not None:
            # Close trade
            trade = _make_trade_event(
                position, bar_1m_idx, bar.close, strategy_hash)
            trades.append(trade)
            mtm_tracker.close_position()
            last_exit_eval_bar = eval_bar_count
            position = None
            trailing_peak = 0.0
            ep_stop_price = None

        elif signal.action == "FLIP" and position is not None:
            # Close current, open opposite
            trade = _make_trade_event(
                position, bar_1m_idx, bar.close, strategy_hash)
            trades.append(trade)
            mtm_tracker.close_position()
            # FLIP does NOT trigger cooldown

            # Open opposite
            position = {
                "direction": signal.direction,
                "entry_price": bar.close,
                "entry_idx": bar_1m_idx,
            }
            position_entry_eval_bar = eval_bar_count
            trailing_peak = bar.close
            ep_stop_price = _compute_atr_stop(signal.direction, bar.close)
            mtm_tracker.open_position()

        prev_outputs = indicator_outputs

    return trades, close_prices, len(bars_1m)


def _make_trade_event(position: dict, exit_idx: int, exit_price: float,
                      strategy_hash: str) -> TradeEvent:
    """Create a TradeEvent from position state."""
    side = position["direction"].lower()
    entry_price = position["entry_price"]
    entry_idx = position["entry_idx"]

    trade_id = _make_trade_id(entry_idx, exit_idx, side, strategy_hash)
    gross_bps = compute_gross_return_bps(entry_price, exit_price, side)

    return TradeEvent(
        trade_id=trade_id,
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        side=side,
        entry_price=_to_fixed_price(entry_price),
        exit_price=_to_fixed_price(exit_price),
        qty=_to_fixed_qty(1.0),  # Normalized to 1 BTC for triage
        gross_return_bps=gross_bps,
    )
