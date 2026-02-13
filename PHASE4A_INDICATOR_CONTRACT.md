# PHASE 4A: INDICATOR CONTRACT

**Version**: 1.2.1  
**Status**: ✅ COMPLETE — Phase 4A.2 Contract Consistency Audit PASSED  
**Date**: 2026-02-02

---

## Document Purpose

Phase 4A locks the observation-space contract implied by the **24-indicator** state set.

**All 24 indicators in this contract are mandatory components of the observation space. Implementations may not omit indicators or substitute proxy logic without violating the contract.**

**Indicator Count Declaration**: This contract defines exactly 24 indicators, numbered 1–24. There is no indicator 25. The observation space cardinality is fixed at 24. Future additions require a new contract version.

This document does NOT:
- Re-evaluate indicator importance (settled in state-space reduction)
- Prescribe implementation algorithms (that is Phase 4B)
- Define trading signals or strategies (that is Phase 5)

This document DOES:
- Formalize what state is observable
- Specify constraints on observation (timing, warmup, types)
- Lock ambiguity resolutions where specifications are underspecified
- Establish the interface between raw data and strategy logic

**Phase 4B Gate Rule**: Phase 4B may only begin when ALL ambiguity resolutions in this contract are marked RESOLVED. No indicator may be implemented while any ambiguity in its section remains PENDING.

---

## Global Invariants

All indicators in this contract are bound by the following invariants. Violations are implementation bugs.

### Invariant 1: Observation Contract (No Lookahead)

All indicators are functions of past data only. An indicator value at time T may only depend on data with timestamps < T (for lagged) or ≤ T (for current bar close).

No indicator may internally reintroduce lookahead even if downstream logic would gate it.

### Invariant 2: Evaluation Granularity (Bar Close Only)

Indicator outputs are valid only at bar close. There are no intrabar semantics, no partial updates, no streaming intermediate values.

If a bar is not yet closed, its indicator values do not exist.

### Invariant 3: State Ownership

Each indicator owns exactly one state contribution to the observation space. Derived indicators may summarize or transform state but may not redefine the state dimension of their dependencies.

### Invariant 4: Determinism

Given identical inputs (same candle sequence, same parameters, same timestamp), an indicator produces identical outputs across:
- Multiple runs
- Different platforms
- Different times of execution

Non-determinism is a bug, not a feature.

### Invariant 5: Semantic Types

All inputs and outputs use Phase 1 semantic types:
- `PRICE`: 2 decimal places (cents)
- `QTY`: 8 decimal places (satoshis)
- `USD`: 2 decimal places (cents)
- `RATE`: 6 decimal places (proportions/ratios)

**RATE Normalization Rule**: All RATE values are stored as decimals in the range that makes sense for the indicator:
- Percentages: 0.0 to 1.0 (e.g., RSI 70 = 0.70, drawdown -5% = -0.05)
- Unbounded ratios: Any real number (e.g., beta = 1.2, correlation = -0.3)
- Bounded indicators like RSI, ADX, Choppiness: 0.0 to 1.0 (not 0-100)

Raw floats are prohibited in indicator specifications. Implementation may use floats internally but must round to semantic type precision at output boundaries.

### Invariant 6: Warmup Behavior

During warmup (insufficient history for computation), indicators return `None`.

Not zero. Not NaN. Not a guess. `None`.

**Two Distinct Concepts**:

1. **Historical Warmup**: The number of bars of historical data required before computation is mathematically possible. This is statically determinable from parameters alone.
   - Example: EMA(20) requires 20 bars → historical warmup = 20

2. **Activation Condition**: A runtime condition that must be satisfied before the indicator produces output, independent of historical data sufficiency.
   - Example: AVWAP requires anchor_index to be set
   - Example: Per-Trade Drawdown requires an open position (entry_index defined)
   - Example: Relative Strength requires benchmark data to be present

**Contract Rule**: 
- Historical warmup is always statically determinable from parameters
- Activation conditions are documented per-indicator in the "Warmup" section
- An indicator outputs None if EITHER historical warmup is unsatisfied OR activation condition is unsatisfied
- **State Preservation Rule**: When an indicator returns None (for any reason), it must NOT update any internal state. Recurrence relations are not applied. The indicator's state remains frozen at its last valid computation. This prevents None inputs from corrupting future outputs.

### Invariant 7: Observability vs Implementation

This contract specifies *what* state is observable and *under what constraints*.

It does NOT specify *how* an implementation must compute values internally, except where specific computation steps are necessary for determinism (e.g., SMA seed for EMA, Wilder smoothing for ATR).

Implementations may optimize freely as long as outputs match the contract.

### Invariant 8: Cross-Asset Timestamp Alignment (Strict)

For indicators requiring multiple input series (Input Class X: cross-asset), the following alignment policy applies:

**STRICT ALIGNMENT**: If any required input series is missing data at timestamp t, the indicator output at t is `None`.

Specifically:
- All input series must have data at exactly the same timestamps
- No carry-forward of stale values is permitted
- No interpolation is permitted
- Gaps in any input series propagate as gaps in output

This policy is chosen over intersection-alignment or carry-forward because:
1. It preserves determinism (no hidden state from carry-forward)
2. It makes data quality issues visible rather than masked
3. It prevents spurious correlation/beta values from misaligned data

**Implication for data ingestion**: Cross-asset analysis requires pre-aligned datasets. Alignment is an ingestion-layer responsibility, not an indicator-layer responsibility.

**Consequences for Strategy Design**: The Strict Alignment policy has intentional downstream effects that strategies must accommodate:
1. **Cross-asset indicators will be sparse** — Any gap in benchmark data produces None output for that timestamp
2. **This sparsity is intentional** — It makes data quality issues visible rather than masked by stale values
3. **Strategies must handle missing state** — Strategy logic cannot assume cross-asset indicators (19, 20, 21) are always available. Strategies must either degrade gracefully when these indicators return None, or explicitly gate on their availability before executing cross-asset-dependent logic.

---

## Default Rounding Conventions

Unless explicitly overridden in an indicator's "Determinism and Rounding" section:

| Operation | Default Mode | Rationale |
|-----------|--------------|-----------|
| Division | TRUNCATE (toward zero) | Deterministic, no bias |
| Intermediate scaling | No clamping | Allow full precision |
| Final output | Round to semantic type scale | Match declared output type |
| Ratio calculations | Scale to common precision before division | Avoid precision loss |

Per-indicator sections document only DEVIATIONS from these defaults.

---

## Dependency Graph

Derived indicators depend on other indicators computed at the same timestamp. This graph defines:
1. Required computation order
2. Effective warmup (max of own warmup and dependency warmups)
3. Implementation validation (dependencies must be computed first)

### Current Dependencies

```
PRIMITIVE INDICATORS (no dependencies on other indicators):
  1 (EMA), 2 (RSI), 3 (ATR), 4 (Pivot Structure), 5 (AVWAP), 
  6 (DD Equity), 7 (MACD), 8 (ROC), 9 (ADX), 10 (Choppiness),
  11 (Bollinger), 12 (LinReg), 13 (HV), 14 (Donchian), 15 (Floor Pivots),
  17 (Vol Targeting), 18 (VRVP), 19 (RS Ratio), 20 (Correlation),
  21 (Beta), 22 (DD Price), 23 (DD Per-Trade)

DERIVED INDICATORS:
  16 (Dynamic SR) ────────► 4 (Pivot Structure)
  24 (Drawdown Metrics) ──► 6 (Drawdown State — Equity)
```

---

## Authoring Status

| Batch | Indicators | Status |
|-------|------------|--------|
| 1 | EMA, RSI, ATR, Pivot Structure, AVWAP, Drawdown State — Equity | ✅ COMPLETE |
| 2 | MACD, ROC, ADX/DMI, Choppiness Index, Bollinger Bands, Linear Regression Slope | ✅ COMPLETE |
| 3 | Historical Volatility, Donchian Channels, Floor Pivots, Dynamic SR, Volatility Targeting | ✅ COMPLETE |
| 4 | VRVP, Relative Strength, Rolling Correlation, Rolling Beta, Drawdown (Price, Per-Trade), Drawdown Metrics | ✅ COMPLETE |

**ALL AMBIGUITIES RESOLVED. PHASE 4B MAY PROCEED.**

---

# BATCH 1: COMPLETE SPECIFICATIONS

---

## 1. EMA (Exponential Moving Average)

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Trend Structure and Acceleration
- **Input Class**: P (Price)
- **Transform Class**: LP (Low-pass smoothing)
- **Role**: Conditioning
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| source | PRICE | Price series (typically close) |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| length | int | 20 | ≥ 1 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| ema | PRICE | Exponentially weighted moving average |

### Max Lookback
`length` bars

### Mathematical Definition

**Smoothing factor**:
```
α = 2 / (length + 1)
```

**Initialization (SMA seed)**:
```
EMA[length-1] = (1/length) × Σ(source[i]) for i in [0, length-1]
```

**Recurrence (t ≥ length)**:
```
EMA[t] = α × source[t] + (1 - α) × EMA[t-1]
```

Equivalently:
```
EMA[t] = EMA[t-1] + α × (source[t] - EMA[t-1])
```

### Warmup
- **Length**: `length` bars
- **Pre-warmup output**: None
- **First valid index**: t = length - 1

### Time Alignment
EMA[t] is valid at the close of bar t. It incorporates source[t] and all prior values with exponential decay.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| length ≤ 0 | Return None for all t |
| length = 1 | EMA equals source (α = 1) |
| Constant source | EMA equals that constant after seeding |
| Insufficient history | Return None |

### Determinism and Rounding
Uses default conventions. SMA seed computation uses TRUNCATE for division.

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale |
|-----------|------------|-----------|
| Seed method | SMA seed over first N bars | Matches TradingView; provides stable initialization |
| First-bar seeding | Not used | Would cause early-bar divergence |

### Subsumption Guard
Cannot be removed because: EMA encodes multi-timescale trend commitment. The relative position of EMAs at different horizons (EMA stacking) is a distinct structural state not available from raw price or other smoothing methods with equivalent stability.

### TradingView Parity Notes
TradingView `ta.ema()` uses identical SMA-seed initialization. Parity expected for all bars after warmup.

---

## 2. RSI (Relative Strength Index)

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Momentum Saturation and Exhaustion
- **Input Class**: P (Price)
- **Transform Class**: NORM + LP (Normalization with smoothing)
- **Role**: Gate
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| source | PRICE | Price series (typically close) |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| length | int | 14 | ≥ 1 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| rsi | RATE | RSI value as decimal (0.0 to 1.0, where 0.5 = RSI 50) |

### Max Lookback
`length + 1` bars (need length changes, which requires length+1 prices)

### Mathematical Definition

**Price changes**:
```
change[t] = source[t] - source[t-1]  for t ≥ 1
```

**Gains and losses**:
```
gain[t] = max(change[t], 0)
loss[t] = max(-change[t], 0)
```

**Initial averages (SMA over first `length` changes)**:
```
avg_gain[length] = (1/length) × Σ(gain[i]) for i in [1, length]
avg_loss[length] = (1/length) × Σ(loss[i]) for i in [1, length]
```

**Smoothed averages (Wilder's RMA, t > length)**:
```
avg_gain[t] = (avg_gain[t-1] × (length-1) + gain[t]) / length
avg_loss[t] = (avg_loss[t-1] × (length-1) + loss[t]) / length
```

**Relative Strength**:
```
RS[t] = avg_gain[t] / avg_loss[t]  (if avg_loss[t] ≠ 0)
```

**RSI** (output as decimal 0.0 to 1.0):
```
RSI[t] = 1 - (1 / (1 + RS[t]))

Equivalently: RSI[t] = RS[t] / (1 + RS[t])
```

If avg_loss[t] = 0 and avg_gain[t] > 0: RSI = 1.0
If avg_loss[t] = 0 and avg_gain[t] = 0: RSI = 0.5

### Warmup
- **Length**: `length + 1` bars
- **Pre-warmup output**: None
- **First valid index**: t = length

### Time Alignment
RSI[t] is valid at the close of bar t. It reflects momentum character up to and including bar t.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| length ≤ 0 | Return None for all t |
| All gains (no losses) | RSI = 1.0 |
| All losses (no gains) | RSI = 0.0 |
| No movement | RSI = 0.5 |
| Insufficient history | Return None |

### Determinism and Rounding
- Division in RS calculation: TRUNCATE
- Division in RMA smoothing: TRUNCATE
- Final RSI bounded to [0.0, 1.0] after computation

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale |
|-----------|------------|-----------|
| Smoothing method | Wilder's RMA (α = 1/N) | Original Wilder specification; matches TradingView |
| Zero-loss handling | RSI = 1.0 if gains exist, 0.5 if flat | Avoids division by zero; sensible limits |

### Subsumption Guard
Cannot be removed because: RSI provides bounded, nonlinear momentum exhaustion state. ROC provides magnitude but not saturation dynamics. The overbought/oversold regime classification is unique to RSI's bounded structure.

### TradingView Parity Notes
TradingView `ta.rsi()` uses identical Wilder RMA smoothing. Parity expected.

---

## 3. ATR (Average True Range)

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Volatility Regime and Risk Normalization
- **Input Class**: R (Range-aware price)
- **Transform Class**: STAT (Range aggregation)
- **Role**: Conditioning
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| high | PRICE | High price series |
| low | PRICE | Low price series |
| close | PRICE | Close price series |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| length | int | 14 | ≥ 1 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| atr | PRICE | Average True Range in price units |

### Max Lookback
`length` bars

### Mathematical Definition

**True Range**:
```
TR[0] = high[0] - low[0]

TR[t] = max(
    high[t] - low[t],
    abs(high[t] - close[t-1]),
    abs(low[t] - close[t-1])
)  for t ≥ 1
```

**Initial ATR (SMA seed)**:
```
ATR[length-1] = (1/length) × Σ(TR[i]) for i in [0, length-1]
```

**Smoothed ATR (Wilder's RMA, t ≥ length)**:
```
ATR[t] = (ATR[t-1] × (length-1) + TR[t]) / length
```

### Warmup
- **Length**: `length` bars
- **Pre-warmup output**: None
- **First valid index**: t = length - 1

### Time Alignment
ATR[t] is valid at the close of bar t. It reflects volatility up to and including bar t.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| length ≤ 0 | Return None for all t |
| Flat market (H=L=C) | TR = 0, ATR decays toward 0 |
| Gap up/down | TR captures gap via close[t-1] terms |
| First bar | TR = high - low only (no prior close) |
| Insufficient history | Return None |

### Determinism and Rounding
- Division in SMA seed: TRUNCATE
- Division in RMA smoothing: TRUNCATE
- All intermediate TR values computed in PRICE scale

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale |
|-----------|------------|-----------|
| Smoothing method | Wilder's RMA (α = 1/N) | Original Wilder specification; matches TradingView |
| First bar TR | high - low (no gap term) | No prior close exists |

### Subsumption Guard
Cannot be removed because: ATR captures range-based volatility including gaps. Historical Volatility uses close-to-close returns only. ATR is the only indicator encoding true range with gap sensitivity.

### TradingView Parity Notes
TradingView `ta.atr()` uses identical Wilder RMA smoothing and SMA seed. Parity expected.

---

## 4. Pivot-based Market Structure

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Market Structure and Key Levels
- **Input Class**: P (Price)
- **Transform Class**: STR (Structural detection)
- **Role**: State
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| high | PRICE | High price series |
| low | PRICE | Low price series |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| left_bars | int | 5 | ≥ 1 |
| right_bars | int | 5 | ≥ 1 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| pivot_high | PRICE or None | Confirmed pivot high value (at confirmation bar) |
| pivot_high_index | int or None | Bar index where the pivot high occurred |
| pivot_low | PRICE or None | Confirmed pivot low value (at confirmation bar) |
| pivot_low_index | int or None | Bar index where the pivot low occurred |

### Max Lookback
`left_bars + right_bars + 1` bars

### Mathematical Definition

**Pivot High at bar p** (confirmed at bar t = p + right_bars):
```
pivot_high[p] exists if and only if:
  high[p] > high[p-i] for all i in [1, left_bars]
  AND
  high[p] > high[p+j] for all j in [1, right_bars]
```

**Pivot Low at bar p** (confirmed at bar t = p + right_bars):
```
pivot_low[p] exists if and only if:
  low[p] < low[p-i] for all i in [1, left_bars]
  AND
  low[p] < low[p+j] for all j in [1, right_bars]
```

**Output semantics**:
- At bar t, if a pivot was confirmed (t = p + right_bars), output the pivot value and index p
- Otherwise, output None

**Confirmation delay**: A pivot at index p is only known/observable at time t = p + right_bars. This is NOT lookahead; it is structural confirmation delay.

### Warmup
- **Length**: `left_bars + right_bars + 1` bars
- **Pre-warmup output**: None for all outputs
- **First valid index**: t = left_bars + right_bars

### Time Alignment
Pivot outputs at bar t reflect a pivot that occurred at bar t - right_bars, confirmed by observing right_bars of subsequent price action. The pivot value is historical; the confirmation is current.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| left_bars ≤ 0 or right_bars ≤ 0 | Return None for all t |
| Tie (equal highs/lows) | Strict inequality required; ties are NOT pivots |
| Multiple pivots same bar | Both high and low pivots can exist at same bar |
| Insufficient history | Return None |

### Determinism and Rounding
No division or scaling. Comparisons only. Fully deterministic by construction.

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale |
|-----------|------------|-----------|
| Comparison strictness | Strict inequality (> and <) | Matches TradingView; avoids ambiguous ties |
| Output timing | At confirmation bar, not at pivot bar | No lookahead; pivot is only "known" after right_bars |
| Both pivot types | Independent detection | A bar can be both pivot high and pivot low |

### Subsumption Guard
Cannot be removed because: Pivot Structure provides confirmed structural extremes with explicit confirmation delay. Donchian provides rolling extremes without confirmation. The "confirmed swing high/low" state is unique.

### TradingView Parity Notes
TradingView `ta.pivothigh()` and `ta.pivotlow()` use identical strict-inequality logic and confirmation delay. Parity expected.

---

## 5. Anchored VWAP (AVWAP)

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Volume-Weighted Context
- **Input Class**: V (Volume-aware)
- **Transform Class**: STAT (Volume-weighted mean)
- **Role**: Conditioning
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| high | PRICE | High price series |
| low | PRICE | Low price series |
| close | PRICE | Close price series |
| volume | QTY | Volume series |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| anchor_index | int | (required) | ≥ 0, < series length |
| price_source | enum | HLC3 | HLC3, CLOSE, HL2, OHLC4 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| avwap | PRICE | Volume-weighted average price from anchor |
| cum_volume | QTY | Cumulative volume from anchor (diagnostic) |

### Max Lookback
Depends on anchor_index. For bars before anchor: not applicable (output is None).

### Mathematical Definition

**Typical price (default HLC3)**:
```
tp[t] = (high[t] + low[t] + close[t]) / 3
```

**Cumulative sums from anchor**:
```
cum_pv[t] = Σ(tp[i] × volume[i]) for i in [anchor_index, t]
cum_v[t] = Σ(volume[i]) for i in [anchor_index, t]
```

**AVWAP**:
```
AVWAP[t] = cum_pv[t] / cum_v[t]  if cum_v[t] > 0
AVWAP[t] = None                   if cum_v[t] = 0
```

**Recurrence form (for streaming)**:
```
At t = anchor_index:
  cum_pv = tp[anchor_index] × volume[anchor_index]
  cum_v = volume[anchor_index]
  AVWAP = cum_pv / cum_v (if cum_v > 0)

For t > anchor_index:
  cum_pv += tp[t] × volume[t]
  cum_v += volume[t]
  AVWAP = cum_pv / cum_v (if cum_v > 0)
```

### Warmup
- **Length**: 1 bar from anchor (anchor_index itself is first valid)
- **Pre-warmup output**: None for t < anchor_index
- **First valid index**: t = anchor_index

### Time Alignment
AVWAP[t] is valid at the close of bar t. It reflects volume-weighted average from anchor through bar t inclusive.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| t < anchor_index | Return None |
| Zero cumulative volume | Return None |
| Negative volume | Invalid input; reject at ingestion |
| anchor_index out of bounds | Return None for all t |

### Determinism and Rounding
- **Typical price division**: TRUNCATE (tp = (H+L+C) / 3)
- **AVWAP division**: TRUNCATE
- **Cumulative products**: Computed in extended precision (PRICE × QTY scale), then divided

**Scaling note**: 
- tp × volume has scale PRICE + QTY = 2 + 8 = 10
- cum_v has scale QTY = 8
- AVWAP = cum_pv / cum_v rescales to PRICE (scale 2)

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale |
|-----------|------------|-----------|
| Price source | HLC3 (typical price) default | Most common; matches TradingView default |
| Anchor semantics | Explicit index parameter | No implicit session/event anchors; anchor must be specified |
| Zero volume handling | Return None | Undefined average; do not invent a value |

### Subsumption Guard
Cannot be removed because: AVWAP provides volume-weighted fair value from a reference point. No other indicator encodes volume-weighted price with anchor semantics. VRVP provides distribution; AVWAP provides a single level.

### TradingView Parity Notes
TradingView Anchored VWAP uses identical HLC3 default and cumulative computation. Anchor selection is manual in TradingView; our anchor_index parameter replicates this. Parity expected when anchor matches.

---

## 6. Drawdown State — Equity Curve

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Path-Dependent Risk and System State
- **Input Class**: S (System/path-dependent)
- **Transform Class**: PATH (Path-dependent accumulation)
- **Role**: State
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| equity | USD | Equity series (mark-to-market or realized) |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| lookback_bars | int or None | None | None = inception-to-date; int ≥ 1 = rolling |
| recovery_rule | enum | GEQ_PEAK | GEQ_PEAK, GT_PEAK |
| equity_min | USD | 0 | Minimum valid equity for state update |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| equity_peak | USD | Running or rolling peak equity |
| drawdown_frac | RATE | Fractional drawdown (≤ 0) |
| drawdown_pct | RATE | Percentage drawdown (≤ 0, in percent units) |
| drawdown_abs | USD | Absolute drawdown (≤ 0) |
| in_drawdown | int | 1 if in drawdown, 0 if at peak |
| drawdown_duration | int | Consecutive bars in drawdown |

### Max Lookback
- If `lookback_bars` is None: 1 bar (inception mode, no lookback limit)
- If `lookback_bars` is int: `lookback_bars` bars

### Mathematical Definition

**Running peak (lookback_bars = None)**:
```
peak[t] = max(equity[0], equity[1], ..., equity[t])

Recurrence:
  peak[0] = equity[0]
  peak[t] = max(peak[t-1], equity[t])
```

**Rolling peak (lookback_bars = L)**:
```
peak[t] = max(equity[t-L+1], ..., equity[t])
```

**Drawdown calculations**:
```
drawdown_abs[t] = equity[t] - peak[t]           (≤ 0)
drawdown_frac[t] = drawdown_abs[t] / peak[t]    (≤ 0, if peak > 0)
drawdown_pct[t] = 100 × drawdown_frac[t]        (≤ 0)
```

**In-drawdown flag**:
```
in_drawdown[t] = 1 if equity[t] < peak[t] else 0
```

**Duration tracking**:
```
If in_drawdown[t] = 1:
  duration[t] = duration[t-1] + 1
Else (recovery):
  If recovery_rule = GEQ_PEAK and equity[t] >= peak[t]:
    duration[t] = 0
  If recovery_rule = GT_PEAK and equity[t] > peak[t]:
    duration[t] = 0
```

### Warmup
- **Length (inception mode)**: 1 bar
- **Length (rolling mode)**: `lookback_bars` bars
- **Pre-warmup output**: None for all outputs
- **First valid index**: t = 0 (inception) or t = lookback_bars - 1 (rolling)

### Time Alignment
All outputs at bar t reflect equity state through bar t inclusive. Peak is updated before drawdown is computed.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| equity ≤ equity_min | Return None, do not update state |
| peak ≤ 0 | Return None (cannot compute percentage) |
| equity = peak (exactly) | in_drawdown = 0, drawdown = 0 |
| Rolling window contains None | Return None for that bar |

### Determinism and Rounding
- **Drawdown fraction division**: TRUNCATE
- **Duration**: Integer counter, no rounding
- **Peak comparison**: Exact equality check (no tolerance)

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale |
|-----------|------------|-----------|
| Recovery rule | GEQ_PEAK default | More robust to float jitter; recovery at peak touch |
| Equity source | Mark-to-market (unrealized included) | Reflects true account state |
| Zero/negative equity | Return None, freeze state | Undefined drawdown; do not fabricate |

### Subsumption Guard
Cannot be removed because: Equity drawdown is path-dependent system state not derivable from price, volatility, or momentum. It encodes portfolio risk trajectory, which is independent of asset behavior.

### TradingView Parity Notes
TradingView does not have a built-in equity drawdown indicator. This is a custom state primitive. No parity target.

---

# BATCH 2: DRAFT PLACEHOLDERS

---

## 7. MACD (Moving Average Convergence Divergence)

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Trend Structure and Acceleration
- **Input Class**: P (Price)
- **Transform Class**: BP (Band-pass / multi-timescale separation)
- **Role**: State
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| source | PRICE | Price series (typically close) |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| fast_length | int | 12 | ≥ 1 |
| slow_length | int | 26 | > fast_length |
| signal_length | int | 9 | ≥ 1 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| macd_line | PRICE | EMA(fast) - EMA(slow) |
| signal_line | PRICE | EMA of macd_line |
| histogram | PRICE | macd_line - signal_line |
| slope_sign | RATE | Sign of bar-over-bar delta of macd_line (framework-derived) |
| signal_slope_sign | RATE | Sign of bar-over-bar delta of signal_line (framework-derived) |

### Max Lookback
**Exact formula**: `slow_length + signal_length - 1` bars

Derivation:
- EMA(slow) first valid at t = slow_length - 1
- MACD line first valid at t = slow_length - 1
- Signal line (EMA of MACD) first valid at t = (slow_length - 1) + (signal_length - 1) = slow_length + signal_length - 2
- First valid index: t = slow_length + signal_length - 2
- Max lookback = slow_length + signal_length - 1 bars

### Mathematical Definition

**Fast EMA** (using EMA definition from indicator #1):
```
EMA_fast[t] = EMA(source, fast_length)[t]
```

**Slow EMA**:
```
EMA_slow[t] = EMA(source, slow_length)[t]
```

**MACD Line**:
```
MACD_line[t] = EMA_fast[t] - EMA_slow[t]

First valid: t = slow_length - 1 (when both EMAs are valid)
```

**Signal Line** (EMA of MACD line):
```
Signal[t] = EMA(MACD_line, signal_length)[t]

First valid: t = (slow_length - 1) + (signal_length - 1) = slow_length + signal_length - 2
```

**Histogram**:
```
Histogram[t] = MACD_line[t] - Signal[t]

First valid: same as Signal line
```

### Warmup
- **Length**: `slow_length + signal_length - 1` bars
- **Pre-warmup output**: None for all outputs
- **First valid index**: t = slow_length + signal_length - 2

**Partial validity**: MACD_line is valid before Signal_line. However, for contract simplicity, all outputs return None until all are valid.

**Framework-derived output warmup** (computed by framework, not Phase 4B):
- `slope_sign`: `slow_length + 1` bars
- `signal_slope_sign`: `slow_length + signal_length` bars

### Time Alignment
All outputs at bar t reflect price data through bar t inclusive. MACD_line incorporates the current bar's close.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| fast_length ≥ slow_length | Invalid parameters; return None for all t |
| Any length ≤ 0 | Invalid parameters; return None for all t |
| Insufficient history | Return None |
| Constant source | MACD_line = 0, Signal = 0, Histogram = 0 after warmup |

### Determinism and Rounding
Uses default conventions. EMA computations follow indicator #1 specification exactly.

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| EMA seeding | SMA seed (per indicator #1) | Consistency with EMA spec | ✅ RESOLVED |
| Partial output validity | All None until all valid | Simpler contract; no partial states | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: MACD encodes multi-scale trend separation and acceleration. ROC measures point-to-point change; MACD measures the dynamics of trend convergence/divergence across timescales.

### TradingView Parity Notes
TradingView `ta.macd()` uses identical EMA-based computation. Parity expected.

---

## 8. ROC (Rate of Change)

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Momentum Saturation and Exhaustion
- **Input Class**: P (Price)
- **Transform Class**: DIFF (Differencing)
- **Role**: Conditioning
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| source | PRICE | Price series |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| length | int | 9 | ≥ 1 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| roc | RATE | Percentage change over length bars (as decimal, e.g., 0.05 = 5%) |

### Max Lookback
**Exact formula**: `length` bars

### Mathematical Definition

```
ROC[t] = (source[t] - source[t - length]) / source[t - length]

Equivalently:
ROC[t] = (source[t] / source[t - length]) - 1
```

### Warmup
- **Length**: `length` bars
- **Pre-warmup output**: None
- **First valid index**: t = length

### Time Alignment
ROC[t] compares the current bar's close to the close `length` bars ago.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| length ≤ 0 | Invalid parameter; return None for all t |
| source[t - length] = 0 | Return None (division by zero) |
| Insufficient history | Return None |

### Determinism and Rounding
Uses default conventions. Division uses TRUNCATE.

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| Output scale | Decimal (0.05 = 5%) | RATE semantic type convention | ✅ RESOLVED |
| Zero denominator | Return None | Undefined; do not fabricate | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: ROC provides unbounded momentum magnitude. RSI is bounded; ROC captures raw momentum scale.

### TradingView Parity Notes
TradingView `ta.roc()` returns percentage (5 = 5%). Our output is decimal (0.05 = 5%). Conversion: TV_ROC / 100 = our_ROC.

---

## 9. ADX / DMI (Average Directional Index)

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Trend Existence vs Chop Regime
- **Input Class**: R (Range-aware)
- **Transform Class**: LP + NORM
- **Role**: Gate
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| high | PRICE | High price series |
| low | PRICE | Low price series |
| close | PRICE | Close price series |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| length | int | 14 | ≥ 1 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| adx | RATE | Average Directional Index (0.0 to 1.0, where 0.25 = ADX 25) |
| plus_di | RATE | Positive Directional Indicator (0.0 to 1.0) |
| minus_di | RATE | Negative Directional Indicator (0.0 to 1.0) |

### Max Lookback
**Exact formula**: `2 * length` bars

Derivation:
- TR, +DM, -DM require 1 bar of history (first computed at t=1)
- Smoothed TR, +DM, -DM (Wilder RMA) first valid at t = length
- +DI, -DI first valid at t = length
- DX first valid at t = length
- ADX (Wilder RMA of DX) first valid at t = 2 * length - 1
- Max lookback = 2 * length bars

### Mathematical Definition

**True Range** (same as ATR indicator #3):
```
TR[0] = high[0] - low[0]
TR[t] = max(high[t] - low[t], abs(high[t] - close[t-1]), abs(low[t] - close[t-1]))
```

**Directional Movement**:
```
up_move[t] = high[t] - high[t-1]
down_move[t] = low[t-1] - low[t]

+DM[t] = up_move[t]   if up_move[t] > down_move[t] AND up_move[t] > 0
       = 0            otherwise

-DM[t] = down_move[t] if down_move[t] > up_move[t] AND down_move[t] > 0
       = 0            otherwise
```

**Smoothed values (Wilder RMA)**:
```
ATR[t] = RMA(TR, length)[t]        (same as indicator #3)
smooth_plus_DM[t] = RMA(+DM, length)[t]
smooth_minus_DM[t] = RMA(-DM, length)[t]

Where RMA uses SMA seed and α = 1/length (Wilder smoothing)
```

**Directional Indicators** (output as 0.0 to 1.0):
```
+DI[t] = smooth_plus_DM[t] / ATR[t]   (if ATR[t] > 0)
-DI[t] = smooth_minus_DM[t] / ATR[t]  (if ATR[t] > 0)
```

**Directional Index**:
```
DX[t] = abs(+DI[t] - -DI[t]) / (+DI[t] + -DI[t])   (if denominator > 0)
```

**Average Directional Index**:
```
ADX[t] = RMA(DX, length)[t]

First valid at t = 2 * length - 1
```

### Warmup
- **Length**: `2 * length` bars
- **Pre-warmup output**: None for all outputs
- **First valid index**: t = 2 * length - 1

**Partial validity**: +DI and -DI are valid before ADX. For contract simplicity, all outputs return None until ADX is valid.

### Time Alignment
All outputs at bar t reflect price data through bar t inclusive.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| length ≤ 0 | Invalid parameter; return None for all t |
| ATR = 0 | +DI = -DI = 0 (no range, no direction) |
| +DI + -DI = 0 | DX = 0 |
| Flat market | ADX decays toward 0 |
| Insufficient history | Return None |

### Determinism and Rounding
- All RMA computations use Wilder smoothing (α = 1/N) with SMA seed
- Division uses TRUNCATE
- Output values clamped to [0.0, 1.0]

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| Smoothing method | Wilder RMA | Original Wilder specification | ✅ RESOLVED |
| DM tie-breaking | Both zero if up_move = down_move | Standard convention | ✅ RESOLVED |
| Zero ATR handling | DI = 0 | No range means no directional information | ✅ RESOLVED |
| Partial output validity | All None until ADX valid | Simpler contract | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: ADX quantifies trend existence independent of direction. No other indicator measures trend strength as a regime state.

### TradingView Parity Notes
TradingView `ta.adx()` uses identical Wilder smoothing. Minor differences possible in early bars due to seeding. Parity expected after full warmup.

---

## 10. Choppiness Index

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Trend Existence vs Chop Regime
- **Input Class**: R (Range-aware)
- **Transform Class**: STAT + NORM
- **Role**: Gate
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| high | PRICE | High price series |
| low | PRICE | Low price series |
| close | PRICE | Close price series |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| length | int | 14 | ≥ 1 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| chop | RATE | Choppiness Index (0.0 to 1.0, where 0.618 = CHOP 61.8) |

### Max Lookback
**Exact formula**: `length` bars

### Mathematical Definition

**True Range** (same as ATR):
```
TR[t] = max(high[t] - low[t], abs(high[t] - close[t-1]), abs(low[t] - close[t-1]))
```

**Sum of True Ranges over window**:
```
sum_TR[t] = Σ(TR[i]) for i in [t - length + 1, t]
```

**Range over window**:
```
highest_high[t] = max(high[i]) for i in [t - length + 1, t]
lowest_low[t] = min(low[i]) for i in [t - length + 1, t]
range[t] = highest_high[t] - lowest_low[t]
```

**Choppiness Index** (output as 0.0 to 1.0):
```
CHOP[t] = log10(sum_TR[t] / range[t]) / log10(length)

If range[t] = 0: CHOP[t] = 1.0 (maximally choppy, no net direction)
```

**Interpretation**:
- CHOP near 1.0: Market is choppy/consolidating
- CHOP near 0.0: Market is trending strongly
- Typical threshold: CHOP > 0.618 = choppy, CHOP < 0.382 = trending

### Warmup
- **Length**: `length` bars
- **Pre-warmup output**: None
- **First valid index**: t = length - 1

### Time Alignment
CHOP[t] reflects price action over the window ending at bar t.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| length ≤ 0 | Invalid parameter; return None for all t |
| range = 0 | CHOP = 1.0 (perfectly choppy) |
| sum_TR = 0 | CHOP = 0.0 (impossible in practice) |
| Insufficient history | Return None |

### Determinism and Rounding
- Log10 computed in floating point, result rounded to RATE precision
- Division uses TRUNCATE before log

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| Zero range handling | CHOP = 1.0 | No range = no trend | ✅ RESOLVED |
| Log base | 10 | Standard Choppiness formula | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: Choppiness Index specifically identifies range-bound vs directional regimes. ADX measures trend strength when trending; Choppiness identifies the choppy state.

### TradingView Parity Notes
TradingView Choppiness Index uses identical formula. Parity expected.

---

## 11. Bollinger Bands

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Mean-Reversion vs Breakout Context
- **Input Class**: P (Price)
- **Transform Class**: ENV (Envelope)
- **Role**: Conditioning
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| source | PRICE | Price series (typically close) |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| length | int | 20 | ≥ 2 |
| mult | float | 2.0 | > 0 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| basis | PRICE | Middle band (SMA) |
| upper | PRICE | Upper band (basis + mult * stdev) |
| lower | PRICE | Lower band (basis - mult * stdev) |
| bandwidth | RATE | (upper - lower) / basis |
| percent_b | RATE | (source - lower) / (upper - lower) |

### Max Lookback
**Exact formula**: `length` bars

### Mathematical Definition

**Basis (SMA)**:
```
basis[t] = (1/length) * Σ(source[i]) for i in [t - length + 1, t]
```

**Standard Deviation (population, ddof=0)**:
```
variance[t] = (1/length) * Σ((source[i] - basis[t])²) for i in [t - length + 1, t]
stdev[t] = sqrt(variance[t])
```

**Bands**:
```
upper[t] = basis[t] + mult * stdev[t]
lower[t] = basis[t] - mult * stdev[t]
```

**Derived metrics**:
```
bandwidth[t] = (upper[t] - lower[t]) / basis[t]   (if basis[t] > 0)
percent_b[t] = (source[t] - lower[t]) / (upper[t] - lower[t])   (if upper[t] ≠ lower[t])
```

### Warmup
- **Length**: `length` bars
- **Pre-warmup output**: None for all outputs
- **First valid index**: t = length - 1

### Time Alignment
All outputs at bar t reflect source values over the window ending at bar t.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| length ≤ 1 | Invalid (need ≥ 2 for stdev); return None for all t |
| mult ≤ 0 | Invalid parameter; return None for all t |
| Constant source | stdev = 0, upper = lower = basis, bandwidth = 0, percent_b = None |
| basis = 0 | bandwidth = None |
| upper = lower | percent_b = None |
| Insufficient history | Return None |

### Determinism and Rounding
- **ddof for variance**: 0 (population standard deviation)
- Square root computed in floating point, result rounded to PRICE precision
- Division uses TRUNCATE

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| ddof for stdev | 0 (population) | Matches TradingView; simpler | ✅ RESOLVED |
| Zero bandwidth handling | percent_b = None | Undefined position | ✅ RESOLVED |
| SMA vs EMA basis | SMA | Standard Bollinger definition | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: Bollinger Bands encode relative price position within a volatility-adjusted equilibrium envelope. This spatial context is distinct from raw volatility (ATR, HV).

### TradingView Parity Notes
TradingView `ta.bb()` uses identical population stdev (ddof=0). Parity expected.

---

## 12. Linear Regression Slope

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Trend Structure and Acceleration
- **Input Class**: P (Price)
- **Transform Class**: REG (Regression)
- **Role**: Conditioning
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| source | PRICE | Price series |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| length | int | 14 | ≥ 2 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| slope | RATE | Slope of least-squares fit line (price change per bar) |

### Max Lookback
**Exact formula**: `length` bars

### Mathematical Definition

**Least-squares slope over window [t - length + 1, t]**:

Let n = length, and index the window as x = 0, 1, ..., n-1 with corresponding y = source values.

```
x̄ = (n - 1) / 2
ȳ = (1/n) * Σ(source[t - length + 1 + i]) for i in [0, n-1]

Numerator = Σ((i - x̄) * (source[t - length + 1 + i] - ȳ)) for i in [0, n-1]
Denominator = Σ((i - x̄)²) for i in [0, n-1]
            = n * (n² - 1) / 12    (formula for sum of squared deviations from mean)

slope[t] = Numerator / Denominator
```

**Simplified formula** (equivalent, more efficient):
```
sum_x = n * (n - 1) / 2
sum_x2 = n * (n - 1) * (2*n - 1) / 6
sum_y = Σ(source[t - length + 1 + i]) for i in [0, n-1]
sum_xy = Σ(i * source[t - length + 1 + i]) for i in [0, n-1]

slope[t] = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
```

### Warmup
- **Length**: `length` bars
- **Pre-warmup output**: None
- **First valid index**: t = length - 1

### Time Alignment
Slope[t] reflects the trend over the window ending at bar t. Positive slope indicates upward trend.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| length ≤ 1 | Invalid (need ≥ 2 points for regression); return None |
| Constant source | slope = 0 |
| Insufficient history | Return None |

### Determinism and Rounding
- Division uses TRUNCATE
- Intermediate sums computed in extended precision to avoid overflow

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| Index convention | 0-based within window | Standard convention | ✅ RESOLVED |
| Output units | Price per bar | Natural interpretation | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: Linear Regression Slope provides statistically smoothed trend rate. ROC is point-to-point; MACD is EMA-based. Least-squares slope is distinct.

### TradingView Parity Notes
TradingView `ta.linreg()` can return slope via the `ta.change(ta.linreg())` approach, or using custom calculation. Direct parity requires matching the formula above.

---

# BATCH 3: DRAFT PLACEHOLDERS

---

## 13. Historical Volatility (HV)

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Volatility Regime and Risk Normalization
- **Input Class**: P (Price)
- **Transform Class**: STAT (Return dispersion)
- **Role**: Conditioning
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| close | PRICE | Close price series |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| length | int | 20 | ≥ 2 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| hv | RATE | Annualized historical volatility (decimal, e.g., 0.50 = 50%) |
| hv_raw | RATE | Non-annualized volatility (stdev of returns) |

### Max Lookback
**Exact formula**: `length + 1` bars (need length returns, which requires length+1 prices)

### Mathematical Definition

**Log returns**:
```
return[t] = ln(close[t] / close[t-1])

First return valid at t = 1
```

**Standard deviation of returns (sample, ddof=1)**:
```
mean_return[t] = (1/length) * Σ(return[i]) for i in [t - length + 1, t]

variance[t] = (1/(length-1)) * Σ((return[i] - mean_return[t])²) for i in [t - length + 1, t]

hv_raw[t] = sqrt(variance[t])
```

**Annualization**:
```
For 1-minute bars:
  bars_per_year = 525600  (365.25 days * 24 hours * 60 minutes)
  
hv[t] = hv_raw[t] * sqrt(bars_per_year)
      = hv_raw[t] * sqrt(525600)
      = hv_raw[t] * 724.98
```

### Warmup
- **Length**: `length + 1` bars
- **Pre-warmup output**: None
- **First valid index**: t = length

### Time Alignment
HV[t] reflects return volatility over the window ending at bar t.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| length ≤ 1 | Invalid (need ≥ 2 for stdev); return None |
| close[t-1] ≤ 0 | Return None (invalid log) |
| Constant close | hv = 0 (no returns) |
| Insufficient history | Return None |

### Determinism and Rounding
- **ddof for variance**: 1 (sample standard deviation)
- Log and sqrt computed in floating point, result rounded to RATE precision
- Division uses TRUNCATE

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| ddof for stdev | 1 (sample) | Unbiased estimator; standard practice | ✅ RESOLVED |
| Return type | Log returns | Symmetric, additive properties | ✅ RESOLVED |
| Annualization factor | 525600 (1m bars) | Correct for minute data | ✅ RESOLVED |
| Mean subtraction | Yes (proper stdev) | Standard volatility definition | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: HV measures return-based volatility. ATR measures range-based volatility. These are distinct dimensions.

### TradingView Parity Notes
TradingView HV may use different annualization (often 252 for daily). Our factor is for 1-minute data. Parity requires matching the timeframe assumption.

---

## 14. Donchian Channels

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Market Structure and Key Levels
- **Input Class**: P (Price)
- **Transform Class**: ENV (Range extremes)
- **Role**: Conditioning
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| high | PRICE | High price series |
| low | PRICE | Low price series |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| length | int | 20 | ≥ 1 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| upper | PRICE | Highest high over length bars |
| lower | PRICE | Lowest low over length bars |
| basis | PRICE | Midpoint of channel |

### Max Lookback
**Exact formula**: `length` bars

### Mathematical Definition

```
upper[t] = max(high[i]) for i in [t - length + 1, t]
lower[t] = min(low[i]) for i in [t - length + 1, t]
basis[t] = (upper[t] + lower[t]) / 2
```

### Warmup
- **Length**: `length` bars
- **Pre-warmup output**: None for all outputs
- **First valid index**: t = length - 1

### Time Alignment
All outputs at bar t reflect the extremes over the window ending at bar t, including bar t.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| length ≤ 0 | Invalid parameter; return None for all t |
| Insufficient history | Return None |

### Determinism and Rounding
- Min/max are exact (no rounding)
- Basis division uses TRUNCATE

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| Inclusive/exclusive window | Inclusive (includes current bar) | Standard convention | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: Donchian provides rolling extremes with known, immediate latency. Pivot Structure provides confirmed extremes with confirmation delay. Different timing properties serve different purposes.

### TradingView Parity Notes
TradingView `ta.highest()` and `ta.lowest()` use identical inclusive window. Parity expected.

---

## 15. Floor Pivots (Traditional)

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Market Structure and Key Levels
- **Input Class**: P (Price)
- **Transform Class**: STAT (Prior-period projection)
- **Role**: State
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| high_prev | PRICE | Prior period high |
| low_prev | PRICE | Prior period low |
| close_prev | PRICE | Prior period close |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| levels | int | 3 | 1-4 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| pp | PRICE | Pivot point |
| r1 | PRICE | Resistance 1 |
| s1 | PRICE | Support 1 |
| r2 | PRICE | Resistance 2 |
| s2 | PRICE | Support 2 |
| r3 | PRICE | Resistance 3 |
| s3 | PRICE | Support 3 |

### Max Lookback
**Exact formula**: 1 prior period (period definition is external)

**Note**: This indicator requires prior-period OHLC as input. The period aggregation (daily, weekly, etc.) is performed upstream. The indicator itself has no bar-based lookback.

### Mathematical Definition

**Pivot Point**:
```
PP = (high_prev + low_prev + close_prev) / 3
```

**Level 1**:
```
R1 = 2 * PP - low_prev
S1 = 2 * PP - high_prev
```

**Level 2**:
```
R2 = PP + (high_prev - low_prev)
S2 = PP - (high_prev - low_prev)
```

**Level 3**:
```
R3 = high_prev + 2 * (PP - low_prev)
S3 = low_prev - 2 * (high_prev - PP)
```

### Warmup
- **Length**: 1 complete prior period
- **Pre-warmup output**: None (until first prior period available)
- **First valid index**: First bar of second period

### Time Alignment
Pivot levels for period K are computed from period K-1's OHLC and are constant throughout period K. They become valid at the first bar of period K.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| high_prev = low_prev | Range = 0; R2=S2=PP, R1≠S1 still valid |
| No prior period | Return None |
| levels < 1 or > 4 | Clamp to [1, 4] |

### Determinism and Rounding
- All divisions use TRUNCATE
- Outputs are constant within a period (no per-bar computation)

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| Period definition | External (not in indicator) | Separation of concerns | ✅ RESOLVED |
| Activation timing | First bar of new period | Standard convention | ✅ RESOLVED |
| Formula variant | Traditional (not Woodie, Camarilla) | Most common | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: Floor Pivots provide time-anchored horizontal levels. Dynamic SR provides structure-anchored levels. Both are needed for complete S/R context.

### TradingView Parity Notes
TradingView Pivot Points Standard with Type="Traditional" uses identical formulas. Parity depends on matching the period (daily, weekly, etc.).

---

## 16. Dynamic SR from Structure

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Market Structure and Key Levels
- **Input Class**: P (Price)
- **Transform Class**: STR (Structural)
- **Role**: State
- **Primitive/Derived**: Derived from Pivot-based Market Structure (ID: 4)

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| pivot_high_values | PRICE[] | Confirmed pivot high values from indicator #4 |
| pivot_high_indices | int[] | Bar indices of pivot highs |
| pivot_low_values | PRICE[] | Confirmed pivot low values from indicator #4 |
| pivot_low_indices | int[] | Bar indices of pivot lows |
| current_price | PRICE | Current close price (for proximity filtering) |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| max_levels | int | 3 | ≥ 1 |
| proximity_atr_mult | float | 0.5 | > 0 |
| atr_value | PRICE | (required) | From indicator #3 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| resistance_levels | PRICE[] | Up to max_levels active resistance levels (sorted descending) |
| support_levels | PRICE[] | Up to max_levels active support levels (sorted ascending) |
| nearest_resistance | PRICE or None | Closest resistance above current price |
| nearest_support | PRICE or None | Closest support below current price |

### Max Lookback
**Inherited**: Same as Pivot-based Market Structure (indicator #4) = `left_bars + right_bars + 1`

### Mathematical Definition

**Level Collection**:
```
All confirmed pivot highs become candidate resistance levels.
All confirmed pivot lows become candidate support levels.
```

**Proximity Filtering** (optional, removes stale levels):
```
A level is "active" if:
  - For resistance: level > current_price
  - For support: level < current_price

Levels too close together (within proximity_atr_mult * ATR) are merged:
  - Keep the level with more touches (recency tie-breaker)
```

**Level Selection**:
```
resistance_levels = top max_levels resistance levels above current_price, sorted descending
support_levels = top max_levels support levels below current_price, sorted ascending

nearest_resistance = min(level for level in resistance_levels if level > current_price)
nearest_support = max(level for level in support_levels if level < current_price)
```

### Warmup
- **Length**: Inherited from indicator #4
- **Pre-warmup output**: Empty arrays, None for nearest
- **First valid index**: When first pivots are confirmed

### Time Alignment
Levels are updated whenever new pivots are confirmed. Between confirmations, levels are static.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| No pivots confirmed yet | Empty arrays, None for nearest |
| Price above all pivots | Empty resistance, None for nearest_resistance |
| Price below all pivots | Empty support, None for nearest_support |
| ATR = 0 | No proximity filtering (all levels kept) |

### Determinism and Rounding
- Level comparison is exact (no tolerance beyond proximity filter)
- Sorting is deterministic (price, then recency for ties)

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| Level merge policy | ATR-based proximity | Prevents cluttered levels | ✅ RESOLVED |
| Recency vs price priority | Price primary, recency secondary | More intuitive | ✅ RESOLVED |
| Include touched levels | Yes (until broken) | Standard S/R behavior | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: Dynamic SR derives actionable levels from confirmed structure. Floor Pivots are time-based; Dynamic SR is structure-based.

### TradingView Parity Notes
No direct TradingView equivalent. This is a custom structural indicator.

---

## 17. Volatility Targeting Inputs

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Volatility Regime and Risk Normalization
- **Input Class**: R + P (Range and Price)
- **Transform Class**: STAT + NORM
- **Role**: Conditioning
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| realized_vol | RATE | Current realized volatility (from HV or ATR-derived) |
| price | PRICE | Current price (for notional calculation) |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| target_volatility | RATE | 0.10 | > 0 (e.g., 0.10 = 10% annual vol) |
| max_leverage | float | 3.0 | ≥ 1.0 |
| min_leverage | float | 0.1 | > 0, ≤ max_leverage |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| vol_scalar | RATE | target_volatility / realized_vol (clamped) |
| target_position_frac | RATE | Fraction of capital for vol-targeted position |
| realized_vol_annualized | RATE | Input vol passed through (for diagnostics) |

### Max Lookback
**Inherited**: From volatility source indicator (HV or ATR)

### Mathematical Definition

**Raw volatility scalar**:
```
raw_scalar = target_volatility / realized_vol   (if realized_vol > 0)
```

**Clamped scalar**:
```
vol_scalar = clamp(raw_scalar, min_leverage, max_leverage)

Where clamp(x, lo, hi) = max(lo, min(x, hi))
```

**Target position fraction**:
```
target_position_frac = vol_scalar

(This represents: if realized vol = target vol, position = 100% of capital)
(If realized vol < target vol, scale up; if higher, scale down)
```

### Warmup
- **Length**: Inherited from volatility source
- **Pre-warmup output**: None (until realized_vol is available)

### Time Alignment
Outputs at bar t use the realized_vol computed through bar t.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| realized_vol = 0 | vol_scalar = max_leverage (cap the scaling) |
| realized_vol very small | vol_scalar clamped to max_leverage |
| realized_vol very large | vol_scalar clamped to min_leverage |
| target_volatility ≤ 0 | Invalid; return None |

### Determinism and Rounding
- Division uses TRUNCATE
- Clamping is exact

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| Vol source | HV preferred (return-based) | More theoretically grounded for sizing | ✅ RESOLVED |
| Clamping behavior | Hard clamp at min/max | Prevent extreme positions | ✅ RESOLVED |
| Zero vol handling | Use max_leverage | Conservative cap | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: Provides the normalized sizing context for volatility targeting directly. While derivable from HV/ATR, having it as a first-class output prevents re-derivation and ensures consistent application.

### TradingView Parity Notes
No direct TradingView equivalent. This is a position-sizing utility indicator.

---

# BATCH 4: DRAFT PLACEHOLDERS

---

## 18. Volume Profile (VRVP)

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Volume-Weighted Context
- **Input Class**: V (Volume-aware)
- **Transform Class**: DIST (Volume-at-price distribution)
- **Role**: State
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| high | PRICE | High price series |
| low | PRICE | Low price series |
| close | PRICE | Close price series |
| volume | QTY | Volume series |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| row_count | int | 24 | ≥ 1 |
| value_area_pct | RATE | 0.70 | 0 < x ≤ 1 |
| lookback_bars | int | 240 | ≥ 1 (defines visible range) |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| poc | PRICE | Point of Control (price level with highest volume) |
| vah | PRICE | Value Area High |
| val | PRICE | Value Area Low |
| profile_high | PRICE | Highest price in profile range |
| profile_low | PRICE | Lowest price in profile range |

### Max Lookback
**Exact formula**: `lookback_bars` bars

### Mathematical Definition

**Profile Range**:
```
profile_high[t] = max(high[i]) for i in [t - lookback_bars + 1, t]
profile_low[t] = min(low[i]) for i in [t - lookback_bars + 1, t]
profile_range = profile_high - profile_low
```

**Row Construction**:
```
row_height = profile_range / row_count

For row r in [0, row_count - 1]:
  row_low[r] = profile_low + r * row_height
  row_high[r] = profile_low + (r + 1) * row_height
```

**Volume Allocation (RESOLVED AMBIGUITY)**:
```
For each bar i in lookback window:
  bar_range = high[i] - low[i]
  
  If bar_range = 0:
    # Allocate all volume to the row containing close[i]
    row_index = floor((close[i] - profile_low) / row_height)
    volume_by_row[row_index] += volume[i]
  Else:
    # Proportional allocation across overlapping rows
    For each row r:
      overlap_low = max(low[i], row_low[r])
      overlap_high = min(high[i], row_high[r])
      
      If overlap_high > overlap_low:
        overlap_frac = (overlap_high - overlap_low) / bar_range
        volume_by_row[r] += volume[i] * overlap_frac
```

**Point of Control**:
```
poc_row = argmax(volume_by_row)
poc = (row_low[poc_row] + row_high[poc_row]) / 2   # Row midpoint

Tie-breaking: If multiple rows have same volume, choose lowest row index (lowest price).
```

**Value Area Calculation**:
```
total_volume = sum(volume_by_row)
target_volume = total_volume * value_area_pct

Start with POC row included.
va_volume = volume_by_row[poc_row]
va_low_idx = poc_row
va_high_idx = poc_row

While va_volume < target_volume:
  # Compare adding row above vs row below
  above_idx = va_high_idx + 1
  below_idx = va_low_idx - 1
  
  above_vol = volume_by_row[above_idx] if above_idx < row_count else 0
  below_vol = volume_by_row[below_idx] if below_idx >= 0 else 0
  
  If above_vol >= below_vol and above_idx < row_count:
    va_high_idx = above_idx
    va_volume += above_vol
  Elif below_idx >= 0:
    va_low_idx = below_idx
    va_volume += below_vol
  Else:
    break  # No more rows to add

vah = row_high[va_high_idx]
val = row_low[va_low_idx]
```

### Warmup
- **Length**: `lookback_bars` bars
- **Pre-warmup output**: None for all outputs
- **First valid index**: t = lookback_bars - 1

### Time Alignment
Profile at bar t reflects the volume distribution over the lookback window ending at bar t.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| lookback_bars ≤ 0 | Invalid; return None |
| row_count ≤ 0 | Invalid; return None |
| profile_range = 0 | All volume in single row; POC = profile_low; VAH = VAL = profile_low |
| Zero total volume | POC = profile midpoint; VAH/VAL = profile bounds |
| Insufficient history | Return None |

### Determinism and Rounding
- Row boundaries use TRUNCATE for division
- Volume allocation uses TRUNCATE for fractional volume
- POC row midpoint uses TRUNCATE

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| Volume allocation | Proportional overlap | Most accurate distribution | ✅ RESOLVED |
| Zero-range bar | Allocate to close price row | Sensible default | ✅ RESOLVED |
| POC tie-breaking | Lowest row index | Deterministic | ✅ RESOLVED |
| VA expansion | Prefer higher volume side | Standard algorithm | ✅ RESOLVED |
| VA tie-breaking | Prefer above (higher price) | TradingView convention | ✅ RESOLVED |
| POC output | Row midpoint | Most representative | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: VRVP provides volume distribution across price levels. AVWAP provides a single level. Distribution is a distinct informational dimension.

### TradingView Parity Notes
TradingView VRVP uses similar proportional allocation. Minor differences may exist in edge cases. Our POC tie-breaking (lowest price) may differ from TradingView.

---

## 19. Relative Strength (Ratio-based)

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Cross-Asset and Macro Context
- **Input Class**: X (Cross-asset)
- **Transform Class**: DIFF (Relative return)
- **Role**: State
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| asset_close | PRICE | Asset price series (e.g., BTC) |
| benchmark_close | PRICE | Benchmark price series (e.g., ETH, SPY) |

### Parameters
None (ratio is parameter-free)

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| rs_ratio | RATE | asset_close / benchmark_close |
| rs_indexed | RATE | Ratio rebased to 100 at first valid point |

### Max Lookback
**Exact formula**: 1 bar (ratio is point-in-time)

For rs_indexed: Requires first valid bar as reference.

### Mathematical Definition

**Raw Ratio**:
```
rs_ratio[t] = asset_close[t] / benchmark_close[t]   (if benchmark_close[t] > 0)
rs_ratio[t] = None                                   (if benchmark_close[t] = 0)
```

**Indexed Ratio** (rebased to 100):
```
Let t0 = first index where rs_ratio[t0] is valid

rs_indexed[t] = 100 * rs_ratio[t] / rs_ratio[t0]   (for t >= t0)
rs_indexed[t] = None                                (for t < t0)
```

### Warmup
- **Length**: 1 bar
- **Pre-warmup output**: None
- **First valid index**: t = 0 (if both series have data)

**Cross-asset alignment**: Per Invariant 8, if benchmark_close is missing at t, rs_ratio[t] = None.

### Time Alignment
rs_ratio[t] reflects the ratio at the close of bar t.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| benchmark_close = 0 | rs_ratio = None |
| benchmark missing | rs_ratio = None (Invariant 8) |
| asset_close = 0 | rs_ratio = 0 |

### Determinism and Rounding
- Division uses TRUNCATE
- No special handling needed

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| Zero benchmark | Return None | Undefined ratio | ✅ RESOLVED |
| Missing benchmark data | Return None | Invariant 8 (strict alignment) | ✅ RESOLVED |
| Index base | 100 | Standard convention | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: Relative Strength measures cross-asset performance ratio. No other indicator provides relative performance context.

### TradingView Parity Notes
TradingView ratio charts compute identically. Parity expected when series are aligned.

---

## 20. Rolling Correlation

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Cross-Asset and Macro Context
- **Input Class**: X (Cross-asset)
- **Transform Class**: STAT
- **Role**: State
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| series_a | PRICE | First price series (e.g., BTC close) |
| series_b | PRICE | Second price series (e.g., ETH close) |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| length | int | 20 | ≥ 2 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| correlation | RATE | Pearson correlation coefficient (-1 to +1) |

### Max Lookback
**Exact formula**: `length` bars

### Mathematical Definition

**Returns** (for correlation of returns, not levels):
```
return_a[t] = (series_a[t] - series_a[t-1]) / series_a[t-1]   (if series_a[t-1] > 0)
return_b[t] = (series_b[t] - series_b[t-1]) / series_b[t-1]   (if series_b[t-1] > 0)
```

**Means over window**:
```
mean_a = (1/length) * Σ(return_a[i]) for i in [t - length + 1, t]
mean_b = (1/length) * Σ(return_b[i]) for i in [t - length + 1, t]
```

**Covariance and Variances**:
```
cov_ab = (1/length) * Σ((return_a[i] - mean_a) * (return_b[i] - mean_b))
var_a = (1/length) * Σ((return_a[i] - mean_a)²)
var_b = (1/length) * Σ((return_b[i] - mean_b)²)
```

**Pearson Correlation**:
```
If var_a > 0 and var_b > 0:
  correlation[t] = cov_ab / sqrt(var_a * var_b)
Else:
  correlation[t] = None (undefined for constant series)
```

### Warmup
- **Length**: `length + 1` bars (need length returns, which requires length+1 prices)
- **Pre-warmup output**: None
- **First valid index**: t = length

**Cross-asset alignment**: Per Invariant 8, if any value in either series is missing within the window, correlation = None.

### Time Alignment
correlation[t] reflects co-movement over the window of returns ending at bar t.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| length ≤ 1 | Invalid; return None |
| Either series constant | Variance = 0; correlation = None |
| Any missing data in window | correlation = None (Invariant 8) |
| Perfect correlation | correlation = 1.0 or -1.0 |

### Determinism and Rounding
- **ddof for variance/covariance**: 0 (population)
- Division uses TRUNCATE
- Output clamped to [-1, +1] to handle floating-point edge cases

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| Correlation of levels vs returns | Returns | Standard practice; removes trend bias | ✅ RESOLVED |
| ddof | 0 (population) | Simpler; matches typical trading use | ✅ RESOLVED |
| Zero variance | Return None | Undefined correlation | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: Correlation measures co-movement direction and strength. Beta measures sensitivity magnitude. Both are needed.

### TradingView Parity Notes
TradingView `ta.correlation()` computes on the raw series (levels). Our implementation uses returns. This is a deliberate divergence for better analytical properties.

---

## 21. Rolling Beta

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Cross-Asset and Macro Context
- **Input Class**: X (Cross-asset)
- **Transform Class**: STAT (Covariance scaling)
- **Role**: State
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| asset_close | PRICE | Asset price series |
| benchmark_close | PRICE | Benchmark price series |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| length | int | 20 | ≥ 2 |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| beta | RATE | Cov(asset_returns, benchmark_returns) / Var(benchmark_returns) |

### Max Lookback
**Exact formula**: `length + 1` bars (need length returns)

### Mathematical Definition

**Returns**:
```
asset_return[t] = (asset_close[t] - asset_close[t-1]) / asset_close[t-1]
benchmark_return[t] = (benchmark_close[t] - benchmark_close[t-1]) / benchmark_close[t-1]
```

**Means over window**:
```
mean_asset = (1/length) * Σ(asset_return[i]) for i in [t - length + 1, t]
mean_benchmark = (1/length) * Σ(benchmark_return[i]) for i in [t - length + 1, t]
```

**Covariance and Benchmark Variance**:
```
cov = (1/length) * Σ((asset_return[i] - mean_asset) * (benchmark_return[i] - mean_benchmark))
var_benchmark = (1/length) * Σ((benchmark_return[i] - mean_benchmark)²)
```

**Beta**:
```
If var_benchmark > 0:
  beta[t] = cov / var_benchmark
Else:
  beta[t] = None (undefined when benchmark has no variance)
```

### Warmup
- **Length**: `length + 1` bars
- **Pre-warmup output**: None
- **First valid index**: t = length

**Cross-asset alignment**: Per Invariant 8, if benchmark data is missing, beta = None.

### Time Alignment
beta[t] reflects sensitivity over the window ending at bar t.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| length ≤ 1 | Invalid; return None |
| Benchmark constant | var_benchmark = 0; beta = None |
| Asset constant | beta = 0 (no co-movement) |
| Missing benchmark data | beta = None (Invariant 8) |

### Determinism and Rounding
- **ddof for variance/covariance**: 0 (population)
- Division uses TRUNCATE

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| ddof | 0 (population) | Consistent with correlation; simpler | ✅ RESOLVED |
| Return type | Simple returns | Consistent with correlation | ✅ RESOLVED |
| Zero benchmark variance | Return None | Undefined beta | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: Beta measures sensitivity magnitude to benchmark. Correlation measures co-movement direction/strength. Different dimensions.

### TradingView Parity Notes
No direct TradingView built-in. Custom script would use similar calculation.

---

## 22. Drawdown State — Price-Based

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Path-Dependent Risk and System State
- **Input Class**: S (System/path-dependent)
- **Transform Class**: PATH
- **Role**: State
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| price | PRICE | Asset price series (typically close) |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| lookback_bars | int or None | None | None = inception; int ≥ 1 = rolling |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| price_peak | PRICE | Running or rolling peak price |
| price_drawdown_frac | RATE | Fractional drawdown from peak (≤ 0) |
| price_drawdown_abs | PRICE | Absolute drawdown from peak (≤ 0) |
| price_drawdown_pct | RATE | Percentage drawdown (≤ 0) |

### Max Lookback
- **Inception mode**: 1 bar
- **Rolling mode**: `lookback_bars` bars

### Mathematical Definition

**Running peak (lookback_bars = None)**:
```
peak[t] = max(price[0], price[1], ..., price[t])

Recurrence:
  peak[0] = price[0]
  peak[t] = max(peak[t-1], price[t])
```

**Rolling peak (lookback_bars = L)**:
```
peak[t] = max(price[i]) for i in [t - L + 1, t]
```

**Drawdown calculations**:
```
price_drawdown_abs[t] = price[t] - peak[t]              (≤ 0)
price_drawdown_frac[t] = price_drawdown_abs[t] / peak[t]  (≤ 0, if peak > 0)
price_drawdown_pct[t] = 100 * price_drawdown_frac[t]      (≤ 0)
```

### Warmup
- **Length (inception mode)**: 1 bar
- **Length (rolling mode)**: `lookback_bars` bars
- **Pre-warmup output**: None for all outputs
- **First valid index**: t = 0 (inception) or t = lookback_bars - 1 (rolling)

### Time Alignment
All outputs at bar t reflect price state through bar t inclusive.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| price ≤ 0 | Return None (invalid price) |
| peak = 0 | Return None (cannot compute fraction) |
| price = peak | drawdown = 0 |

### Determinism and Rounding
- Division uses TRUNCATE
- Peak comparison is exact

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| Price source | Close | Most common; represents settled price | ✅ RESOLVED |
| Peak update timing | Include current bar | Standard convention | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: Price drawdown is asset-level state independent of strategy equity. "Is BTC in a drawdown?" is different from "Is my portfolio in a drawdown?"

### TradingView Parity Notes
No direct TradingView equivalent. Custom script would match.

---

## 23. Drawdown State — Per-Trade

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Path-Dependent Risk and System State
- **Input Class**: S (System/path-dependent)
- **Transform Class**: PATH
- **Role**: State
- **Primitive/Derived**: Primitive

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| high | PRICE | High price series |
| low | PRICE | Low price series |
| close | PRICE | Close price series |
| position_side | enum | LONG, SHORT, or FLAT |
| entry_index | int or None | Bar index of trade entry (None if FLAT) |

### Parameters
| Name | Type | Default | Constraints |
|------|------|---------|-------------|
| excursion_basis | enum | HIGH_LOW | HIGH_LOW or CLOSE_ONLY |

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| favorable_excursion | PRICE or None | Best price since entry |
| adverse_excursion | PRICE or None | Worst price since entry |
| trade_drawdown_abs | PRICE or None | Drawdown from favorable (≤ 0) |
| trade_drawdown_frac | RATE or None | Fractional drawdown from favorable (≤ 0) |
| bars_since_entry | int or None | Number of bars since entry |

### Max Lookback
**Dynamic**: From `entry_index` to current bar

### Mathematical Definition

**When position_side = FLAT**: All outputs = None

**When position_side = LONG** (profit when price rises):
```
favorable_excursion[t] = max(high[i]) for i in [entry_index, t]
adverse_excursion[t] = min(low[i]) for i in [entry_index, t]
trade_drawdown_abs[t] = low[t] - favorable_excursion[t]   (≤ 0)
trade_drawdown_frac[t] = trade_drawdown_abs[t] / favorable_excursion[t]
```

**When position_side = SHORT** (profit when price falls):
```
favorable_excursion[t] = min(low[i]) for i in [entry_index, t]
adverse_excursion[t] = max(high[i]) for i in [entry_index, t]
trade_drawdown_abs[t] = favorable_excursion[t] - high[t]   (≤ 0)
trade_drawdown_frac[t] = trade_drawdown_abs[t] / favorable_excursion[t]
```

### Warmup
- **Length**: 1 bar from entry
- **Pre-warmup output**: None when FLAT
- **First valid index**: entry_index

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| position_side = FLAT | All outputs = None |
| entry_index = None | All outputs = None |
| favorable_excursion = 0 | trade_drawdown_frac = None |

### Determinism and Rounding
- Max/min are exact; Division uses TRUNCATE

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| Excursion basis | HIGH_LOW default | Captures intrabar extremes | ✅ RESOLVED |
| Drawdown sign | Always ≤ 0 | Consistent convention | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: Per-trade drawdown requires entry reference and direction. Not derivable from price or equity drawdown.

---

## 24. Drawdown Metrics

**Status**: ✅ COMPLETE

### Classification
- **State Dimension**: Path-Dependent Risk and System State
- **Input Class**: S (System/path-dependent)
- **Transform Class**: STAT(PATH)
- **Role**: Conditioning
- **Primitive/Derived**: Derived from Drawdown State — Equity Curve (ID: 6)

### Inputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| drawdown_frac | RATE | From Drawdown State — Equity Curve (indicator #6) |
| drawdown_duration | int | From Drawdown State — Equity Curve (indicator #6) |
| in_drawdown | int | From Drawdown State — Equity Curve (indicator #6) |

### Parameters
None (aggregates are parameter-free)

### Outputs
| Name | Semantic Type | Description |
|------|---------------|-------------|
| max_drawdown | RATE | Maximum drawdown observed to date (most negative) |
| max_duration | int | Maximum duration in drawdown to date |
| current_drawdown | RATE | Current drawdown (pass-through for convenience) |
| current_duration | int | Current duration in drawdown |
| drawdown_count | int | Number of drawdown episodes completed |

### Max Lookback
**Inherited**: From Drawdown State — Equity Curve (indicator #6)

### Mathematical Definition

**Running maximum drawdown**:
```
max_drawdown[t] = min(drawdown_frac[0], drawdown_frac[1], ..., drawdown_frac[t])

Note: min because drawdowns are negative; "max" refers to magnitude.

Recurrence:
  max_drawdown[0] = drawdown_frac[0]
  max_drawdown[t] = min(max_drawdown[t-1], drawdown_frac[t])
```

**Running maximum duration**:
```
max_duration[t] = max(all completed drawdown durations up to t, current_duration[t])
```

**Drawdown episode counting**:
```
A drawdown episode ends when in_drawdown transitions from 1 to 0.

drawdown_count[t] = number of such transitions in [0, t]
```

**Pass-through values**:
```
current_drawdown[t] = drawdown_frac[t]
current_duration[t] = drawdown_duration[t]
```

### Warmup
- **Length**: Inherited from indicator #6 (1 bar for inception mode)
- **Pre-warmup output**: None for all outputs
- **First valid index**: Same as indicator #6

### Time Alignment
All outputs at bar t reflect metrics computed through bar t.

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| No drawdowns yet | max_drawdown = 0, max_duration = 0, count = 0 |
| Always in drawdown | count = 0 (no completed episodes) |
| Dependency returns None | All outputs = None |

### Determinism and Rounding
- Min/max are exact (no rounding)
- Counts are integer

### Ambiguity Resolutions
| Ambiguity | Our Choice | Rationale | Status |
|-----------|------------|-----------|--------|
| Episode definition | Ends on recovery | Standard convention | ✅ RESOLVED |
| max_drawdown semantics | Most negative value | Magnitude interpretation | ✅ RESOLVED |

### Subsumption Guard
Cannot be removed because: While derived from Equity Drawdown State, these running aggregates are frequently needed for gating and reporting.

### TradingView Parity Notes
No direct TradingView equivalent. Custom strategy reporting would use similar metrics.

---

# APPENDICES

---

## Appendix A: Indicator ID Registry

| ID | Indicator | Status | Batch | Primitive/Derived |
|----|-----------|--------|-------|-------------------|
| 1 | EMA | ✅ COMPLETE | 1 | Primitive |
| 2 | RSI | ✅ COMPLETE | 1 | Primitive |
| 3 | ATR | ✅ COMPLETE | 1 | Primitive |
| 4 | Pivot-based Market Structure | ✅ COMPLETE | 1 | Primitive |
| 5 | Anchored VWAP | ✅ COMPLETE | 1 | Primitive |
| 6 | Drawdown State — Equity Curve | ✅ COMPLETE | 1 | Primitive |
| 7 | MACD | ✅ COMPLETE | 2 | Primitive |
| 8 | ROC | ✅ COMPLETE | 2 | Primitive |
| 9 | ADX / DMI | ✅ COMPLETE | 2 | Primitive |
| 10 | Choppiness Index | ✅ COMPLETE | 2 | Primitive |
| 11 | Bollinger Bands | ✅ COMPLETE | 2 | Primitive |
| 12 | Linear Regression Slope | ✅ COMPLETE | 2 | Primitive |
| 13 | Historical Volatility | ✅ COMPLETE | 3 | Primitive |
| 14 | Donchian Channels | ✅ COMPLETE | 3 | Primitive |
| 15 | Floor Pivots | ✅ COMPLETE | 3 | Primitive |
| 16 | Dynamic SR from Structure | ✅ COMPLETE | 3 | Derived (4) |
| 17 | Volatility Targeting Inputs | ✅ COMPLETE | 3 | Primitive |
| 18 | Volume Profile (VRVP) | ✅ COMPLETE | 4 | Primitive |
| 19 | Relative Strength | ✅ COMPLETE | 4 | Primitive |
| 20 | Rolling Correlation | ✅ COMPLETE | 4 | Primitive |
| 21 | Rolling Beta | ✅ COMPLETE | 4 | Primitive |
| 22 | Drawdown State — Price | ✅ COMPLETE | 4 | Primitive |
| 23 | Drawdown State — Per-Trade | ✅ COMPLETE | 4 | Primitive |
| 24 | Drawdown Metrics | ✅ COMPLETE | 4 | Derived (6) |

**Total**: 24 indicators fully specified. Observation space cardinality is FIXED at 24.

---

## Appendix B: Dependency Graph and Computation Order

### Dependency Classes

Indicators fall into three classes based on their input requirements:

**Class A: Candle-Pure Primitives** (depend only on OHLCV data)
```
1 (EMA), 2 (RSI), 3 (ATR), 4 (Pivot Structure), 7 (MACD), 8 (ROC), 
9 (ADX), 10 (Choppiness), 11 (Bollinger), 12 (LinReg), 13 (HV), 
14 (Donchian), 15 (Floor Pivots), 18 (VRVP)
```

**Class B: Cross-Asset Primitives** (require external benchmark series)
```
19 (Relative Strength), 20 (Rolling Correlation), 21 (Rolling Beta)
```

**Class C: System-State Primitives** (require runtime state: equity, position, entry)
```
5 (AVWAP - requires anchor_index)
6 (DD Equity - requires equity series)
17 (Vol Targeting - requires realized_vol from Class A)
22 (DD Price - candle-pure but conceptually system-state)
23 (DD Per-Trade - requires position_side, entry_index)
```

**Class D: Derived Indicators** (depend on other indicator outputs)
```
16 (Dynamic SR) ────────► 4 (Pivot Structure), 3 (ATR)
24 (Drawdown Metrics) ──► 6 (Drawdown State — Equity)
```

### Computation Order

```
Phase 1: Class A indicators (candle-pure, may be computed in parallel)
Phase 2: Class B indicators (cross-asset, may be computed in parallel if benchmark data available)
Phase 3: Class C indicators (system-state, require ordered evaluation based on runtime state)
Phase 4: Class D indicators (derived, must wait for dependencies)
```

### Effective Warmup for Derived
```
16: max(own_warmup, warmup_of_4, warmup_of_3)
24: max(own_warmup, warmup_of_6)
```

### Note on Parallelism
The statement "all primitives may be computed in parallel" applies ONLY to Class A indicators. Class B requires aligned external data. Class C requires runtime state that may not be available at indicator computation time. Class D requires prior indicator outputs.

---

## Appendix C: State Dimension Coverage Matrix

| State Dimension | Indicator IDs | Coverage Status |
|-----------------|---------------|-----------------|
| Trend Structure and Acceleration | 1, 7, 12 | ✅ Covered |
| Momentum Saturation and Exhaustion | 2, 8 | ✅ Covered |
| Trend Existence vs Chop Regime | 9, 10 | ✅ Covered |
| Volatility Regime and Risk Normalization | 3, 13, 17 | ✅ Covered |
| Mean-Reversion vs Breakout Context | 11 | ✅ Covered |
| Market Structure and Key Levels | 4, 14, 15, 16 | ✅ Covered |
| Volume-Weighted Context | 5, 18 | ✅ Covered |
| Cross-Asset and Macro Context | 19, 20, 21 | ✅ Covered |
| Path-Dependent Risk and System State | 6, 22, 23, 24 | ✅ Covered |

**All 9 state dimensions have at least one indicator. No blind spots.**

---

## Appendix D: Ambiguity Resolution Ledger (Summary)

**ALL AMBIGUITIES RESOLVED**

| Indicator | Ambiguity | Resolution | Status |
|-----------|-----------|------------|--------|
| EMA (1) | Seed method | SMA over first N bars | ✅ |
| RSI (2) | Smoothing | Wilder RMA (α=1/N) | ✅ |
| RSI (2) | Zero-loss | RSI=100 if gains, 50 if flat | ✅ |
| ATR (3) | Smoothing | Wilder RMA (α=1/N) | ✅ |
| ATR (3) | First bar TR | high-low only | ✅ |
| Pivot (4) | Comparison | Strict inequality | ✅ |
| Pivot (4) | Output timing | At confirmation bar | ✅ |
| AVWAP (5) | Price source | HLC3 default | ✅ |
| AVWAP (5) | Zero volume | Return None | ✅ |
| DD Equity (6) | Recovery rule | GEQ_PEAK | ✅ |
| DD Equity (6) | Equity source | Mark-to-market | ✅ |
| MACD (7) | Partial output | All None until all valid | ✅ |
| ROC (8) | Output scale | Decimal (0.05 = 5%) | ✅ |
| ADX (9) | DM tie-breaking | Both zero if equal | ✅ |
| ADX (9) | Zero ATR | DI = 0 | ✅ |
| Choppiness (10) | Zero range | CHOP = 1.0 | ✅ |
| Bollinger (11) | ddof for stdev | 0 (population) | ✅ |
| LinReg (12) | Index convention | 0-based within window | ✅ |
| HV (13) | ddof for stdev | 1 (sample) | ✅ |
| HV (13) | Annualization | 525600 (1m bars/year) | ✅ |
| HV (13) | Return type | Log returns | ✅ |
| Donchian (14) | Window inclusive | Yes (includes current bar) | ✅ |
| Floor Pivots (15) | Formula variant | Traditional | ✅ |
| Dynamic SR (16) | Level merge | ATR-based proximity | ✅ |
| Vol Target (17) | Zero vol handling | Use max_leverage | ✅ |
| VRVP (18) | Volume allocation | Proportional overlap | ✅ |
| VRVP (18) | POC tie-breaking | Lowest row index | ✅ |
| VRVP (18) | VA expansion tie | Prefer above | ✅ |
| RS (19) | Zero benchmark | Return None | ✅ |
| Correlation (20) | Level vs returns | Returns | ✅ |
| Correlation (20) | ddof | 0 (population) | ✅ |
| Beta (21) | ddof | 0 (population) | ✅ |
| DD Price (22) | Price source | Close | ✅ |
| DD Trade (23) | Excursion basis | HIGH_LOW default | ✅ |
| DD Metrics (24) | Episode definition | Ends on recovery | ✅ |

---

## Appendix E: Transform Class Legend

| Code | Name | Description |
|------|------|-------------|
| LP | Low-pass | Smoothing (removes high-frequency noise) |
| BP | Band-pass | Multi-timescale separation |
| DIFF | Differencing | Rate of change, derivatives |
| NORM | Normalization | Bounding, scaling to fixed range |
| REG | Regression | Curve fitting, least squares |
| ENV | Envelope | Price bands, channels |
| STR | Structural | Pattern detection, pivots |
| DIST | Distribution | Histograms, profiles |
| STAT | Statistical | Aggregations, moments |
| PATH | Path-dependent | Cumulative, history-dependent state |

---

**END OF PHASE 4A INDICATOR CONTRACT v1.0.0**
