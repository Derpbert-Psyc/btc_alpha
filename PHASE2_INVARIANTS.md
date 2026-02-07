# PHASE 2 INVARIANTS

**Version**: 1.0 (Canonical)  
**Status**: FROZEN — Strategy layer code must obey these invariants.  
**Depends on**: SYSTEM_LAWS.md (Phase 1)

---

## I. PHASE 1 DEPENDENCY

1. **Single Source of Truth**: Phase 2 imports `Fixed`, `SemanticType`, `RoundingMode`, and `SEMANTIC_SCALES` from Phase 1. No redefinitions permitted.

2. **Deny-by-Default Import**: Phase 2 fails at startup if Phase 1 import fails. Demo mode requires explicit `BTC_ALPHA_DEMO_MODE=1` environment variable.

3. **Demo Mode is Forbidden in Production**: Any run with demo fallback types is non-authoritative and must not be used for trading decisions or performance evaluation.

---

## II. SEMANTIC CONSISTENCY

4. **Homogeneous Inputs**: All indicator functions validate that input sequences have consistent semantic types. Mixed types raise `SemanticConsistencyError`.

5. **Candle Validation**: `Candle.__post_init__` enforces:
   - `open`, `high`, `low`, `close` are `PRICE`
   - `volume` is `QTY`
   - `high >= max(open, close)`
   - `low <= min(open, close)`

6. **View Validation**: `PositionView`, `LedgerView`, and `OrderIntent` enforce semantic types in `__post_init__`.

---

## III. LOOKAHEAD PREVENTION

7. **4H Lag Enforcement**: `HistoricalView4H.__post_init__` validates:
   - All candle timestamps are 4H-aligned (divisible by 14400)
   - All candle timestamps are strictly less than `current_ts`
   - Violations raise `LookaheadError`

8. **Committed History Consistency**: `CommittedHistory.__post_init__` validates that `view_1m.current_ts == view_4h.current_ts`.

9. **No Partial Bars**: HTF indicators use only fully closed bars. The latest 4H bar visible at time T closed at or before the most recent 4H boundary before T.

---

## IV. WARMUP

10. **Dual Warmup Check**: `IndicatorOutput.has_warmup_complete()` requires BOTH:
    - `history_length_1m >= required_lookback_1m`
    - `history_length_4h >= required_lookback_4h`
    - All required indicator fields are not `None`

11. **Warmup Blocks Trading**: `GateView.warmup_complete` is derived from indicators. If warmup is incomplete, `can_trade()` returns `False` regardless of other gates.

12. **Required Fields Documentation**: The list of required indicator fields for warmup is explicitly documented in `IndicatorOutput.has_warmup_complete()` and must be updated if strategy dependencies change.

---

## V. STATELESS MODULES

13. **Indicators are Pure**: `IndicatorModule.compute()` is a pure function of `CommittedHistory`. No internal state.

14. **Signals are Pure**: `SignalModule.compute()` is a pure function of `IndicatorOutput`, `PositionView`, and `current_price`. No internal state.

15. **Execution is Pure**: `ExecutionLogic.compute()` is a pure function of `SignalOutput`, `GateView`, `LedgerView`, and `current_price`. No internal state.

16. **No Hidden Accumulators**: Strategy modules do not maintain private mutable state between invocations. All "memory" is either recomputed from committed history or explicitly stored in Phase 1 artifacts.

---

## VI. INTEGER ARITHMETIC

17. **No Decimal Past Ingress**: All indicator and signal computations use `Fixed`-backed integer arithmetic. `Decimal` is permitted only at ingress (parsing) and display boundaries.

18. **Explicit Rounding**: All division operations use `_integer_divide_with_rounding()` with an explicit `RoundingMode`. No implicit rounding.

19. **Correct Half-Even**: Half-even rounding uses `2 * remainder` vs `divisor` comparison, correct for all divisor parities.

---

## VII. POSITION SIZING

20. **Deterministic Formula**: `target_qty = floor(equity × risk_fraction / price)`

21. **Truncate Toward Zero**: Position sizing uses `RoundingMode.TRUNCATE`.

22. **Clamped to Limits**: Result is clamped to `[min_qty, max_qty]` from configuration.

23. **Zero Equity Guard**: If `equity <= 0`, sizing returns `None` and execution holds.

---

## VIII. ORDERS

24. **Market Orders Only**: Phase 2 produces `MARKET` order intents only. Limit order logic is deferred to Phase 3.

25. **Orders are Intents**: `OrderIntent` is transmitted AFTER packet commit per System Law 17.

26. **Positive Quantity**: `OrderIntent.qty` must be positive. Validated in `__post_init__`.

---

## IX. TIMESTAMPS

27. **Epoch Seconds**: All timestamps are Unix epoch seconds. Constant `TIMESTAMP_UNIT = "epoch_seconds"` documents this.

28. **Holding Time**: Computed as `(exit_ts - entry_ts) // 60` minutes.

29. **4H Alignment**: 4H boundaries are timestamps divisible by `SECONDS_PER_4H = 14400`.

---

## X. METRICS

30. **Validated Series**: `compute_metrics()` validates that `equity_series` and `leverage_series` are sorted by timestamp. Unsorted series raise `SeriesOrderingError`.

31. **Deterministic Aggregation**: All metric computations use integer arithmetic with explicit formulas.

32. **Canonical Hashing**: `hash_trades()` and `hash_metrics()` use JSON canonical serialization with `sort_keys=True`.

---

## XI. ERROR TAXONOMY

33. **DeterminismError**: Base class for all determinism violations.

34. **WarmupIncompleteError**: Raised when `get_required_or_raise()` finds `None` after warmup check passed.

35. **SemanticConsistencyError**: Raised when inputs have mismatched semantic types.

36. **LookaheadError**: Raised when HTF data violates lag requirements.

37. **SeriesOrderingError**: Raised when time series is not sorted.

---

## XII. SCOPE FREEZE

38. **No New Semantic Types**: Phase 2 uses only `PRICE`, `QTY`, `USD`, `RATE` from Phase 1. No new types introduced.

39. **No State Expansion**: Strategy state does not expand `StatePacket`. If future phases need state (debounce, hysteresis), it must be explicitly added with schema and hash coverage.

40. **Market Orders Only**: Limit orders, stop orders, and time-in-force logic are out of scope for Phase 2.

41. **Single Strategy**: Phase 2 implements exactly one baseline strategy (trend-following breakout with HTF agreement).

---

## XIII. ACCEPTANCE CRITERIA

Phase 2 is complete when:

1. ✅ Baseline strategy runs end-to-end without violating any System Law or Phase 2 Invariant.
2. ✅ Determinism: identical inputs + config produce identical trade hash and metrics hash across two runs.
3. ✅ Warmup enforced: no trades occur before `max_lookback` is satisfied.
4. ✅ No lookahead: HTF values are from closed bars only (validated at construction).
5. ✅ No Phase 1 modifications required.

---

**END OF PHASE 2 INVARIANTS**

*Any strategy code that violates these invariants is invalid and must be corrected.*
