"""Test MACD implementation correctness — SMA seed validation against reference.

The reference implementation is self-contained (no external TA libs).
Verifies that the backtest runner's MACD matches the TradingView-style
EMA-with-SMA-seed specification from MACD_CONFLUENCE_STRATEGY_CONTRACT v1.7.0 §2.1.
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

    Returns: (macd_line, signal_line, histogram, slope_sign) — all lists.
    """
    fast_ema = ref_ema(closes, fast)
    slow_ema = ref_ema(closes, slow)

    n = len(closes)
    macd_line = [None] * n
    for i in range(n):
        if fast_ema[i] is not None and slow_ema[i] is not None:
            macd_line[i] = fast_ema[i] - slow_ema[i]

    # Signal = EMA of non-None MACD values
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

    return macd_line, signal_line, histogram, slope_sign


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

        # First non-None value should be SMA of first 10 values
        expected_sma = sum(closes[:10]) / 10  # = 5.5
        assert ema[9] is not None
        assert abs(ema[9] - expected_sma) < 1e-10, f"EMA seed {ema[9]} != SMA {expected_sma}"

        # All subsequent values should match reference
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
        """100-bar synthetic series — MACD(12,26,9) must match reference."""
        # Synthetic price: oscillating + trend
        closes = [100.0 + 0.5 * i + 3.0 * math.sin(i * 0.3) for i in range(100)]

        ml, sl, hist, ss = compute_macd(closes, 12, 26, 9)
        rml, rsl, rhist, rss = ref_macd(closes, 12, 26, 9)

        for i in range(100):
            # MACD line
            if rml[i] is not None:
                assert ml[i] is not None, f"MACD line None at {i} but ref is {rml[i]}"
                assert abs(ml[i] - rml[i]) < 1e-10, f"MACD line mismatch at {i}: {ml[i]} vs {rml[i]}"
            # Signal line
            if rsl[i] is not None:
                assert sl[i] is not None, f"Signal line None at {i} but ref is {rsl[i]}"
                assert abs(sl[i] - rsl[i]) < 1e-10, f"Signal line mismatch at {i}: {sl[i]} vs {rsl[i]}"
            # Histogram
            if rhist[i] is not None:
                assert hist[i] is not None
                assert abs(hist[i] - rhist[i]) < 1e-10
            # Slope sign
            if rss[i] is not None:
                assert ss[i] == rss[i], f"slope_sign mismatch at {i}: {ss[i]} vs {rss[i]}"

    def test_macd_slope_sign_transitions(self):
        """Verify slope_sign transitions at inflection points."""
        # Strong downtrend then strong uptrend — sharp V-shape
        closes = [200.0 - 2.0 * i for i in range(60)] + [80.0 + 2.0 * i for i in range(60)]

        ml, _, _, ss = compute_macd(closes, 12, 26, 9)

        # Find any transition from negative/zero to positive slope
        found_neg_to_pos = False
        for i in range(1, len(ss)):
            if ss[i - 1] is not None and ss[i] is not None:
                if ss[i - 1] <= 0 and ss[i] == 1:
                    found_neg_to_pos = True
                    break
        assert found_neg_to_pos, (
            "Expected a slope_sign <=0 → +1 transition. "
            f"Slope values: {[s for s in ss if s is not None]}"
        )

    def test_macd_values_with_constant_price(self):
        """Constant price → MACD should converge to 0."""
        closes = [100.0] * 100
        ml, sl, hist, ss = compute_macd(closes, 12, 26, 9)

        # After warmup, MACD line should be exactly 0 (fast EMA == slow EMA)
        for i in range(26, 100):
            if ml[i] is not None:
                assert abs(ml[i]) < 1e-10, f"MACD line should be 0 at {i}, got {ml[i]}"
