"""Test MACD implementation correctness — SMA seed validation against reference.

The reference implementation is self-contained (no external TA libs).
Verifies that the backtest runner's MACD matches the TradingView-style
EMA-with-SMA-seed specification from MACD_CONFLUENCE_STRATEGY_CONTRACT v1.7.0 §2.1.

Includes signal_slope_sign tests (framework-derived output, same precedent as slope_sign).
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ui.services.backtest_runner import compute_ema, compute_macd


# ---------------------------------------------------------------------------
# Reference implementation — embedded, no external deps
# ---------------------------------------------------------------------------

def ref_ema(closes: list, period: int) -> list:
    """EMA with SMA seed — reference implementation."""
    n = len(closes)
    result = [None] * n
    if period < 1 or n < period:
        return result
    if period == 1:
        return list(closes)

    sma_seed = sum(closes[:period]) / period
    result[period - 1] = sma_seed
    k = 2.0 / (period + 1)
    prev = sma_seed
    for i in range(period, n):
        val = closes[i] * k + prev * (1 - k)
        result[i] = val
        prev = val
    return result


def ref_macd(closes: list, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD with SMA-seeded EMA — reference implementation.

    Returns: (macd_line, signal_line, histogram, slope_sign, signal_slope_sign).
    """
    fast_ema = ref_ema(closes, fast)
    slow_ema = ref_ema(closes, slow)

    n = len(closes)
    macd_line = [None] * n
    for i in range(n):
        if fast_ema[i] is not None and slow_ema[i] is not None:
            macd_line[i] = fast_ema[i] - slow_ema[i]

    macd_vals = [v for v in macd_line if v is not None]
    sig_ema = ref_ema(macd_vals, signal) if len(macd_vals) >= signal else [None] * len(macd_vals)

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

    for i in range(1, n):
        if macd_line[i] is not None and macd_line[i - 1] is not None:
            diff = macd_line[i] - macd_line[i - 1]
            slope_sign[i] = 1 if diff > 0 else (-1 if diff < 0 else 0)

    signal_slope_sign = [None] * n
    for i in range(1, n):
        if signal_line[i] is not None and signal_line[i - 1] is not None:
            diff = signal_line[i] - signal_line[i - 1]
            signal_slope_sign[i] = 1 if diff > 0 else (-1 if diff < 0 else 0)

    return macd_line, signal_line, histogram, slope_sign, signal_slope_sign


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEMASMASeed:
    """Verify EMA uses SMA seed."""

    def test_ema_sma_seed(self):
        closes = [float(i + 1) for i in range(50)]
        period = 10
        ema = compute_ema(closes, period)
        ref = ref_ema(closes, period)
        expected_sma = sum(closes[:10]) / 10
        assert ema[9] is not None
        assert abs(ema[9] - expected_sma) < 1e-10, f"EMA seed {ema[9]} != SMA {expected_sma}"
        for i in range(10, 50):
            assert abs(ema[i] - ref[i]) < 1e-10, f"EMA mismatch at {i}: {ema[i]} vs {ref[i]}"

    def test_ema_none_before_warmup(self):
        closes = [float(i) for i in range(20)]
        period = 10
        ema = compute_ema(closes, period)
        for i in range(9):
            assert ema[i] is None, f"EMA[{i}] should be None during warmup"
        assert ema[9] is not None


class TestMACDCorrectness:
    """Verify MACD values match reference implementation within 1e-10."""

    def test_macd_matches_reference_100_bars(self):
        closes = [100.0 + 0.5 * i + 3.0 * math.sin(i * 0.3) for i in range(100)]
        ml, sl, hist, ss, sss = compute_macd(closes, 12, 26, 9)
        rml, rsl, rhist, rss, rsss = ref_macd(closes, 12, 26, 9)
        for i in range(100):
            if rml[i] is not None:
                assert ml[i] is not None, f"MACD line None at {i} but ref is {rml[i]}"
                assert abs(ml[i] - rml[i]) < 1e-10
            if rsl[i] is not None:
                assert sl[i] is not None
                assert abs(sl[i] - rsl[i]) < 1e-10
            if rhist[i] is not None:
                assert hist[i] is not None
                assert abs(hist[i] - rhist[i]) < 1e-10
            if rss[i] is not None:
                assert ss[i] == rss[i], f"slope_sign mismatch at {i}"
            if rsss[i] is not None:
                assert sss[i] == rsss[i], f"signal_slope_sign mismatch at {i}"

    def test_macd_slope_sign_transitions(self):
        closes = [200.0 - 2.0 * i for i in range(60)] + [80.0 + 2.0 * i for i in range(60)]
        ml, _, _, ss, _ = compute_macd(closes, 12, 26, 9)
        found = any(
            ss[i - 1] is not None and ss[i] is not None and ss[i - 1] <= 0 and ss[i] == 1
            for i in range(1, len(ss))
        )
        assert found, "Expected slope_sign <=0 -> +1 transition"

    def test_macd_values_with_constant_price(self):
        closes = [100.0] * 100
        ml, sl, hist, ss, sss = compute_macd(closes, 12, 26, 9)
        for i in range(26, 100):
            if ml[i] is not None:
                assert abs(ml[i]) < 1e-10, f"MACD line should be 0 at {i}, got {ml[i]}"


class TestSignalSlopeSign:
    """Verify signal_slope_sign (framework-derived output)."""

    def test_signal_slope_sign_matches_reference(self):
        closes = [100.0 + 0.5 * i + 3.0 * math.sin(i * 0.3) for i in range(100)]
        _, _, _, _, sss = compute_macd(closes, 12, 26, 9)
        _, _, _, _, rsss = ref_macd(closes, 12, 26, 9)
        for i in range(100):
            if rsss[i] is not None:
                assert sss[i] == rsss[i], f"signal_slope_sign mismatch at {i}"

    def test_signal_slope_sign_warmup_suppression(self):
        closes = [100.0 + 0.5 * i for i in range(100)]
        _, _, _, _, sss = compute_macd(closes, 12, 26, 9)
        for i in range(34):
            assert sss[i] is None, f"signal_slope_sign[{i}] should be None during warmup"
        assert sss[34] is not None, "signal_slope_sign[34] should be non-None"

    def test_signal_slope_sign_constant_price(self):
        closes = [100.0] * 100
        _, _, _, _, sss = compute_macd(closes, 12, 26, 9)
        for i in range(100):
            if sss[i] is not None:
                assert sss[i] == 0, f"signal_slope_sign should be 0 at {i}, got {sss[i]}"

    def test_signal_slope_sign_v_shape_transition(self):
        closes = [200.0 - 2.0 * i for i in range(60)] + [80.0 + 2.0 * i for i in range(60)]
        _, _, _, _, sss = compute_macd(closes, 12, 26, 9)
        found = any(
            sss[i - 1] is not None and sss[i] is not None and sss[i - 1] <= 0 and sss[i] == 1
            for i in range(1, len(sss))
        )
        assert found, "Expected signal_slope_sign <=0 -> +1 transition"

    def test_existing_outputs_unchanged(self):
        closes = [100.0 + 0.5 * i + 3.0 * math.sin(i * 0.3) for i in range(100)]
        ml, sl, hist, ss, _ = compute_macd(closes, 12, 26, 9)
        rml, rsl, rhist, rss, _ = ref_macd(closes, 12, 26, 9)
        for i in range(100):
            if rml[i] is not None:
                assert abs(ml[i] - rml[i]) < 1e-10, f"macd_line changed at {i}"
            if rsl[i] is not None:
                assert abs(sl[i] - rsl[i]) < 1e-10, f"signal_line changed at {i}"
            if rhist[i] is not None:
                assert abs(hist[i] - rhist[i]) < 1e-10, f"histogram changed at {i}"
            if rss[i] is not None:
                assert ss[i] == rss[i], f"slope_sign changed at {i}"
