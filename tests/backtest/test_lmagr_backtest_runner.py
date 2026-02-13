"""Test LMAGR (indicator 25) backtest runner implementation."""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ui.services.backtest_runner import compute_lmagr


def ref_lmagr(closes: list, ma_length: int = 20):
    """LMAGR reference matching Phase 4B LMAGRIndicator."""
    n = len(closes)
    lmagr_list = [None] * n
    lmagr_pct_list = [None] * n
    SCALE_FACTOR = 10_000_000_000
    RATE_SCALE = 1_000_000
    k_scaled = (2 * SCALE_FACTOR) // (ma_length + 1)
    ema_value = None
    bars_seen = 0
    for i in range(n):
        close_int = int(round(closes[i] * 100))
        if close_int <= 0:
            continue
        bars_seen += 1
        if ema_value is None:
            ema_value = close_int
        else:
            ema_value = (
                close_int * k_scaled + ema_value * (SCALE_FACTOR - k_scaled)
            ) // SCALE_FACTOR
        if bars_seen < ma_length:
            continue
        if ema_value <= 0:
            continue
        ratio = close_int / ema_value
        if ratio <= 0:
            continue
        lmagr_list[i] = int(math.log(ratio) * RATE_SCALE)
        lmagr_pct_list[i] = int((ratio - 1.0) * RATE_SCALE)
    return lmagr_list, lmagr_pct_list


class TestLMAGRCorrectness:

    def test_matches_reference_50_bars(self):
        closes = [100.0 + 0.5 * i + 3.0 * math.sin(i * 0.3) for i in range(50)]
        lmagr, lmagr_pct = compute_lmagr(closes, 20)
        rl, rp = ref_lmagr(closes, 20)
        for i in range(50):
            assert lmagr[i] == rl[i], f"lmagr mismatch at {i}: {lmagr[i]} vs {rl[i]}"
            assert lmagr_pct[i] == rp[i], f"lmagr_pct mismatch at {i}"

    def test_warmup_behavior(self):
        closes = [100.0 + i for i in range(50)]
        lmagr, _ = compute_lmagr(closes, 20)
        for i in range(19):
            assert lmagr[i] is None, f"lmagr[{i}] should be None during warmup"
        assert lmagr[19] is not None, "lmagr[19] should be non-None"

    def test_constant_price(self):
        closes = [100.0] * 50
        lmagr, lmagr_pct = compute_lmagr(closes, 10)
        for i in range(50):
            if lmagr[i] is not None:
                assert lmagr[i] == 0, f"lmagr should be 0 at {i}, got {lmagr[i]}"
                assert lmagr_pct[i] == 0

    def test_scale_invariance(self):
        base = [100.0] * 20 + [110.0] * 10
        high = [100000.0] * 20 + [110000.0] * 10
        l_lo, _ = compute_lmagr(base, 20)
        l_hi, _ = compute_lmagr(high, 20)
        for i in range(20, 30):
            if l_lo[i] is not None and l_hi[i] is not None:
                max_val = max(abs(l_lo[i]), 1)
                diff_pct = abs(l_lo[i] - l_hi[i]) / max_val * 100
                assert diff_pct < 1.0, f"Scale invariance failed at {i}: {diff_pct:.2f}%"

    def test_invalid_close_no_state_mutation(self):
        closes = [100.0 + i for i in range(30)]
        closes_bad = list(closes)
        closes_bad[10] = 0.0
        lmagr_bad, _ = compute_lmagr(closes_bad, 5)
        ref_bad, _ = ref_lmagr(closes_bad, 5)
        assert lmagr_bad[10] is None
        for i in range(30):
            assert lmagr_bad[i] == ref_bad[i], f"Mismatch at {i} after invalid close"

    def test_short_ma_length(self):
        closes = [100.0, 110.0, 90.0]
        lmagr, _ = compute_lmagr(closes, 1)
        assert lmagr[0] is not None
        assert lmagr[0] == 0
