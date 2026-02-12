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
                     List[Optional[float]], List[Optional[float]]]:
    """Compute MACD: macd_line, signal_line, histogram, slope_sign."""
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

    return macd_line, signal_line, histogram, slope_sign


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
# Indicator computation for a resolved config
# ---------------------------------------------------------------------------

def compute_indicator_outputs(
    instance: dict,
    bars: List[Bar],
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
        period = params.get("period", 20)
        ema_vals = compute_ema(closes, period)
        result["ema"] = ema_vals

    elif ind_id == 2:  # RSI
        period = params.get("period", 14)
        result["rsi"] = compute_rsi(closes, period)

    elif ind_id == 3:  # ATR
        period = params.get("period", 14)
        result["atr"] = compute_atr(highs, lows, closes, period)

    elif ind_id == 7:  # MACD
        fast = params.get("fast_period", 12)
        slow = params.get("slow_period", 26)
        sig = params.get("signal_period", 9)
        ml, sl, hist, ss = compute_macd(closes, fast, slow, sig)
        result["macd_line"] = ml
        result["signal_line"] = sl
        result["histogram"] = hist
        result["slope_sign"] = ss

    elif ind_id == 8:  # ROC
        period = params.get("period", 14)
        roc_vals = [None] * len(closes)
        for i in range(period, len(closes)):
            if closes[i - period] != 0:
                roc_vals[i] = ((closes[i] - closes[i - period]) / closes[i - period]) * 100
        result["roc"] = roc_vals

    elif ind_id == 10:  # Choppiness
        period = params.get("period", 14)
        result["choppiness"] = compute_choppiness(highs, lows, closes, period)

    elif ind_id == 11:  # Bollinger
        period = params.get("period", 20)
        num_std = params.get("num_std", 2.0)
        basis, upper, lower, bw, pb = compute_bollinger(closes, period, num_std)
        result["basis"] = basis
        result["upper"] = upper
        result["lower"] = lower
        result["bandwidth"] = bw
        result["percent_b"] = pb

    elif ind_id == 12:  # LinReg
        period = params.get("period", 14)
        result["slope"] = compute_linreg_slope(closes, period)

    elif ind_id == 14:  # Donchian
        period = params.get("period", 20)
        upper, lower, basis = compute_donchian(highs, lows, period)
        result["upper"] = upper
        result["lower"] = lower
        result["basis"] = basis

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
) -> Tuple[List[TradeEvent], List[float], int]:
    """Run backtest on resolved config with 1m bar data.

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

    for inst in instances:
        label = inst["label"]
        tf = inst.get("timeframe", "1m")
        tf_sec = parse_timeframe_seconds(tf)
        instance_tf_sec[label] = tf_sec
        resampled = resample_bars(bars_1m, tf_sec)
        instance_bars[label] = resampled
        instance_outputs[label] = compute_indicator_outputs(inst, resampled)

        # Build ts → resampled index: for each 1m bar, find the latest resampled
        # bar whose ts <= 1m bar's ts.
        ts_map: Dict[int, int] = {}
        rs_idx = 0
        for bar_1m in bars_1m:
            while (rs_idx + 1 < len(resampled) and
                   resampled[rs_idx + 1].ts <= bar_1m.ts):
                rs_idx += 1
            if rs_idx < len(resampled) and resampled[rs_idx].ts <= bar_1m.ts:
                ts_map[bar_1m.ts] = rs_idx
        instance_ts_to_rsidx[label] = ts_map

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

            for exit_rule in exit_rules:
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
            signal = evaluate_signal_pipeline(
                config=resolved_config,
                indicator_outputs=indicator_outputs,
                prev_outputs=prev_outputs,
                position=pos_dict,
                mtm_tracker=mtm_tracker if position else None,
                current_price=round(bar.close * 100) if position else None,
            )

        # Process signal
        if signal.action == "ENTRY" and position is None:
            position = {
                "direction": signal.direction,
                "entry_price": bar.close,
                "entry_idx": bar_1m_idx,
            }
            position_entry_eval_bar = eval_bar_count
            trailing_peak = bar.close
            mtm_tracker.open_position()

        elif signal.action == "EXIT" and position is not None:
            # Close trade
            trade = _make_trade_event(
                position, bar_1m_idx, bar.close, strategy_hash)
            trades.append(trade)
            mtm_tracker.close_position()
            position = None
            trailing_peak = 0.0

        elif signal.action == "FLIP" and position is not None:
            # Close current, open opposite
            trade = _make_trade_event(
                position, bar_1m_idx, bar.close, strategy_hash)
            trades.append(trade)
            mtm_tracker.close_position()

            # Open opposite
            position = {
                "direction": signal.direction,
                "entry_price": bar.close,
                "entry_idx": bar_1m_idx,
            }
            position_entry_eval_bar = eval_bar_count
            trailing_peak = bar.close
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
