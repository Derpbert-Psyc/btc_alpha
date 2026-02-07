# Phase 4B Contract and Non-Goals (LOCKED)

**Status**: LOCKED  
**Lock Date**: 2026-02-04  
**Changes After Lock**: Bug fixes only, or Phase 4.1

---

## 1. Indicator Class Definitions

### Class A: Candle-Pure Continuous
**Examples**: EMA, RSI, ATR, Pivot Structure, VRVP, Bollinger, Donchian, MACD, ROC, ADX, Choppiness, LinReg Slope, Historical Volatility

- **Activation**: Always active when candle data is present
- **Warmup**: Bar count since stream start (`bar_index + 1 >= warmup`)
- **State**: Updates every bar with valid candle inputs
- **Lifecycle**: `inactive (no data) → warming → eligible`
- **Constraint**: Class A indicators must be continuously computable from the first candle. Any indicator with optional dependencies, activation gating, or that may start late is not Class A.

### Class B: Candle-Pure Activation-Dependent
**Examples**: Relative Strength (19), Rolling Correlation (20), Rolling Beta (21)

- **Activation**: Requires external data (benchmark_close) to be present
- **Warmup**: Count of computed bars since activation, not global bar_index
- **State**: Updates only when activated AND inputs valid
- **Lifecycle**: `inactive (no benchmark) → activated/warming → eligible → inactive (benchmark disappears)`
- **Key invariant**: Warmup counter is only meaningful while activation is true. During inactive periods, the stored value has no semantic meaning and is reset on next activation start.
- **Activation flicker rule**: If activation toggles off for any number of bars and returns, it is treated as a new activation window and warmup restarts. Tolerance for brief dropouts is out of scope for Phase 4B.

### Class C: System-State Dependent
**Examples**: AVWAP (5), DD Equity (6), Floor Pivots (17), DD Price (22), DD Per-Trade (23)

- **Activation**: Requires system event (position entry, anchor set, period boundary)
- **Warmup**: Event-relative (e.g., bars since entry, bars since anchor)
- **State**: Updates only during active window; resets when activation ends
- **Lifecycle**: `inactive → event triggers activation → warming → eligible → event ends → inactive`
- **Key invariant**: Each activation window is independent; warmup and state reset per window
- **Scope constraint**: Class C indicators with persistent cross-window state are explicitly out of scope for Phase 4B.

### Class D: Derived
**Examples**: Dynamic SR (16), DD Metrics (24)

- **Activation**: All upstream dependencies must be eligible
- **Warmup**: None independent; inherits from upstream eligibility
- **State**: Updates only when all dependencies are eligible
- **Lifecycle**: Mirrors upstream; eligible when all dependencies eligible

---

## 2. Output Flag Semantics

### `computed: bool`
**Definition**: Was `_compute_impl()` called this bar?

| Value | Meaning | State Mutation Allowed? |
|-------|---------|------------------------|
| `True` | Compute was called, state may have updated | Yes |
| `False` | Compute was skipped | No |

### `eligible: bool`
**Definition**: Can downstream consumers (derived indicators, strategies) use this output?

| Value | Meaning |
|-------|---------|
| `True` | Output is valid for consumption |
| `False` | Output should not be consumed |

### Flag Combinations

| Scenario | `computed` | `eligible` | State Updates? | Values |
|----------|-----------|------------|----------------|--------|
| Warming (any class) | True | False | Yes | All-None (suppressed) |
| Invalid input | False | False | No | All-None |
| Activation failed | False | False | No | All-None |
| Dependency not eligible | False | False | No | All-None |
| Event-sparse valid | True | True | Yes | All-None (by design) |
| Normal valid | True | True | Yes | Has values |

**Invariant**: `eligible=True` requires `computed=True`. The reverse is not true.

---

## 3. Engine Behavior Per Class

### Class A Engine Flow
```
1. Check inputs valid
   → if invalid: computed=False, eligible=False, skip compute
2. Call compute (state updates)
3. Check warmup: bar_index + 1 >= warmup
   → if not satisfied: eligible=False, values=all-None
   → if satisfied: eligible=True
```

### Class B Engine Flow
```
1. Check activation (benchmark present)
   → if not active: computed=False, eligible=False, skip compute
   → if activation just started: reset warmup_counter
2. Check inputs valid
   → if invalid: computed=False, eligible=False, skip compute
3. Call compute (state updates)
4. Increment warmup_counter
5. Check warmup: warmup_counter >= warmup
   → if not satisfied: eligible=False, values=all-None
   → if satisfied: eligible=True
```

### Class C Engine Flow
```
1. Check activation (system event active)
   → if not active: computed=False, eligible=False, skip compute
   → if activation just started: reset warmup_counter AND indicator state
2. Check inputs valid
   → if invalid: computed=False, eligible=False, skip compute
3. Call compute (state updates)
4. Increment warmup_counter
5. Check warmup: warmup_counter >= warmup
   → if not satisfied: eligible=False, values=all-None
   → if satisfied: eligible=True
```

### Class D Engine Flow
```
1. Check all dependencies eligible
   → if any not eligible: computed=False, eligible=False, skip compute
2. Check inputs valid (if any direct inputs)
   → if invalid: computed=False, eligible=False, skip compute
3. Call compute (state updates)
4. eligible=True (inherits from dependencies)
```

---

## 4. Gate Definitions

### Gate 1: Warmup Formula Parity
Each indicator's `compute_warmup(params)` returns correct value for all parameter combinations.

### Gate 2: Input Mapping Audit
Each indicator's input mapping covers all declared inputs with correct semantic types.

### Gate 3: Derived Activation
Derived indicators check `dependency.eligible`, not `dependency.computed` or output values.

### Gate 4: Invalid-Input State Preservation
When `computed=False`, state must not change.  
**Scope**: Any scenario where the engine produces `computed=False`.

---

## 5. Explicit Non-Goals (Phase 4.1)

The following are **out of scope** for Phase 4B:

1. **Mid-bar recomputation**: Indicators compute once per bar, not on tick updates
2. **Partial activation recovery**: If Class B/C loses activation, warmup fully resets; no "resume from where we left off"
3. **Activation flicker tolerance**: Brief activation dropouts are not tolerated; any off→on transition restarts warmup
4. **Cross-indicator state sharing**: Each indicator's state is fully isolated
5. **Dynamic warmup**: Warmup period is fixed at indicator creation, not adaptive
6. **Lookback beyond warmup**: Indicators cannot request historical bars before their warmup window
7. **Multi-timeframe**: All indicators operate on a single timeframe per engine instance
8. **Warmup carryover across resets**: `engine.reset_all()` clears all state and warmup counters
9. **Persistent cross-window state for Class C**: Each activation window is independent

---

## 6. Lock Criteria

Phase 4B is **locked** when:

1. ✓ Contract page exists and is accepted
2. ✓ Engine implements class-specific flows with warmup counters
3. ✓ Three stress tests pass:
   - **Class B**: Rolling Correlation with late benchmark (warmup counts from activation)
   - **Class C**: DD Per-Trade with entry/exit cycles (warmup resets per trade)
   - **Class D**: Derived indicator that waits for upstream eligibility
4. ✓ All existing tests (Pivot Structure) still pass
5. ✓ "No new semantics" gate enforced

---

## 7. Change Control (Post-Lock)

After lock, any discovered issue must be classified as:

| Classification | Action |
|----------------|--------|
| **Bug**: Violates locked contract | Fix required |
| **Gap**: Ambiguity in contract | Clarify without changing semantics |
| **New requirement** | Phase 4.1 |

Changes to `computed`/`eligible` semantics, class definitions, or warmup rules are **not permitted** under Phase 4B.

---

## 8. Validation Results

### Stress Tests (2026-02-04)

```
CLASS B STRESS TEST: PASSED ✓
  - 1000 bars without benchmark: all computed=False
  - Bars 1000-1019: warming (computed=True, eligible=False)
  - Bars 1020+: eligible=True
  - Activation flicker: warmup correctly restarted

CLASS C STRESS TEST: PASSED ✓
  - 100 bars no position: all computed=False
  - Entry at bar 100: computed=True, eligible=True
  - Exit at bar 110: computed=False
  - New entry at bar 120: new window, computed=True, eligible=True

CLASS D STRESS TEST: PASSED ✓
  - Bars 0-9: Pivot warming, ATR warming, DSR not computed
  - Bars 10-12: Pivot eligible, ATR warming, DSR not computed
  - Bars 13+: Both deps eligible, DSR computed and eligible

ALL PHASE 4B STRESS TESTS PASSED ✓
Phase 4B contract is validated. Proceeding to VRVP is safe.
```

---

**Contract locked. Proceeding to VRVP implementation.**
