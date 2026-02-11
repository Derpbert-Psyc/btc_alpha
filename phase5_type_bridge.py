"""
Phase 5 — Type Bridge: Fixed <-> TypedValue conversion

Provides a closed, hand-verified mapping between Phase 1 (Fixed/SemanticType)
and Phase 4B (TypedValue/SemanticType).  Fail-closed on unmapped types.

References:
    btc_alpha_v3_final.py:62-80 — Phase 1 SemanticType
    btc_alpha_phase4b_1_7_2.py:226-306 — Phase 4B SemanticType + TypedValue
"""

from __future__ import annotations

from typing import Dict, Optional

from btc_alpha_v3_final import SemanticType as P1Sem, Fixed
from btc_alpha_phase2_v4 import Candle
from btc_alpha_phase4b_1_7_2 import (
    SemanticType as P4Sem,
    TypedValue,
    SystemInputs,
)

# ---------------------------------------------------------------------------
# Closed mapping table — explicit, hand-verified
# ---------------------------------------------------------------------------

_P1_TO_P4: Dict[P1Sem, P4Sem] = {
    P1Sem.PRICE: P4Sem.PRICE,
    P1Sem.QTY:   P4Sem.QTY,
    P1Sem.USD:   P4Sem.USD,
    P1Sem.RATE:  P4Sem.RATE,
}

_P4_TO_P1: Dict[P4Sem, P1Sem] = {v: k for k, v in _P1_TO_P4.items()}

# ---------------------------------------------------------------------------
# Completeness assertions (import-time)
# ---------------------------------------------------------------------------

# HARD: Every Phase 1 SemanticType MUST be mapped
assert set(_P1_TO_P4.keys()) == set(P1Sem), (
    f"Phase 1 SemanticType has unmapped values: {set(P1Sem) - set(_P1_TO_P4.keys())}"
)

# SOFT: Phase 4B may have types not in Phase 1.
# We only assert coverage of the subset we actually map.
# Unmapped Phase 4B types will fail closed at conversion time.


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------

def fixed_to_typed(f: Fixed) -> TypedValue:
    """Convert Phase 1 Fixed to Phase 4B TypedValue.

    Raises KeyError if f.sem is not in the mapping (fail-closed).
    """
    try:
        p4_sem = _P1_TO_P4[f.sem]
    except KeyError:
        raise KeyError(f"Unmapped Phase 1 SemanticType: {f.sem}")
    return TypedValue.create(f.value, p4_sem)


def typed_to_fixed(tv: TypedValue) -> Fixed:
    """Convert Phase 4B TypedValue to Phase 1 Fixed.

    Raises KeyError if tv.sem is not in the mapping (fail-closed).
    """
    try:
        p1_sem = _P4_TO_P1[tv.sem]
    except KeyError:
        raise KeyError(f"Unmapped Phase 4B SemanticType: {tv.sem}")
    return Fixed(value=tv.value, sem=p1_sem)


# ---------------------------------------------------------------------------
# Candle → candle_inputs for IndicatorEngine.compute_all()
# ---------------------------------------------------------------------------

def candle_to_candle_inputs(
    candle: Candle,
) -> Dict[str, Optional[TypedValue]]:
    """Convert a Phase 2 Candle to the candle_inputs dict expected by compute_all().

    Keys: "open", "high", "low", "close", "volume" — all as TypedValue.
    """
    return {
        "open":   fixed_to_typed(candle.open),
        "high":   fixed_to_typed(candle.high),
        "low":    fixed_to_typed(candle.low),
        "close":  fixed_to_typed(candle.close),
        "volume": fixed_to_typed(candle.volume),
    }


# ---------------------------------------------------------------------------
# Build SystemInputs for compute_all()
# ---------------------------------------------------------------------------

def build_system_inputs(
    *,
    equity: Optional[Fixed] = None,
    position_side: int = 0,
    entry_index: Optional[int] = None,
    anchor_index: Optional[int] = None,
    realized_vol: Optional[Fixed] = None,
    benchmark_close: Optional[Fixed] = None,
) -> SystemInputs:
    """Build a Phase 4B SystemInputs from Phase 1 Fixed values."""
    return SystemInputs(
        equity=fixed_to_typed(equity) if equity is not None else None,
        position_side=position_side,
        entry_index=entry_index,
        anchor_index=anchor_index,
        realized_vol=fixed_to_typed(realized_vol) if realized_vol is not None else None,
        benchmark_close=fixed_to_typed(benchmark_close) if benchmark_close is not None else None,
    )


# ---------------------------------------------------------------------------
# Build period_data for compute_all()
# ---------------------------------------------------------------------------

def build_period_data(
    prev_4h: Optional[Candle],
) -> Optional[Dict[str, Optional[TypedValue]]]:
    """Build period_data dict from a previous 4H candle.

    Returns None if prev_4h is None (no 4H data available yet).
    """
    if prev_4h is None:
        return None
    return {
        "high":  fixed_to_typed(prev_4h.high),
        "low":   fixed_to_typed(prev_4h.low),
        "close": fixed_to_typed(prev_4h.close),
    }
