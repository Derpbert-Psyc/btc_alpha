# BTC ALPHA SYSTEM LAWS

**Version**: 1.0 (Frozen)  
**Status**: CANONICAL — All code must obey these laws.

---

## I. NUMERIC DETERMINISM

1. **Integer Authority**: All authoritative arithmetic is integer math. No floating point. No Decimal past ingress.

2. **Decimal Boundaries**: `Decimal` may be used ONLY for:
   - Ingress (parsing external data via `Fixed.from_decimal()`)
   - Display (human output via `Fixed.to_decimal()`)

3. **Explicit Rounding**: Every cross-type operation declares its rounding mode. No silent defaults except where intentionally punitive (fees, slippage → `AWAY_FROM_ZERO`).

4. **Overflow Safety**: Guaranteed by pre-validation of inputs, not by intermediate bounds checking. Operational limits are documented and enforced at construction.

5. **Semantic Types**: Exactly four:
   - `PRICE` — USD per BTC (scale=2)
   - `QTY` — BTC quantity (scale=8)  
   - `USD` — All USD-valued quantities (scale=2)
   - `RATE` — Dimensionless rates (scale=6)

6. **USD Unification**: `NOTIONAL`, `PNL`, `FEE`, `FUNDING`, `BALANCE` are all `USD` type and freely addable. Financial interpretation is enforced by field-level invariants, not type separation.

---

## II. TEMPORAL INTEGRITY

7. **1-Minute Atom**: The 1-minute candle is the indivisible unit of time. All higher timeframes are derived by upsampling from committed 1m history.

8. **No Lookahead**: Higher-timeframe bars are visible only after their close. The latest 5m bar at 10:04 is the one that closed at 10:00.

9. **Watermark Authority**: Watermark is derived ONLY from committed bundles. No partially staged minute may advance time.

---

## III. ATOMIC COMMITMENT

10. **Single Artifact**: A `MinuteBundle` contains facts, fills, and packet. It commits atomically or not at all.

11. **Staging Key**: `stage_bundle()` produces an immutable `StagingKey`. `commit_bundle()` requires this key. No other path to commitment exists.

12. **Crash Recovery**: On boot:
    - Scan staged bundles
    - Delete any without corresponding committed bundle
    - Watermark resumes from last committed bundle
    - Hash chain resumes from last committed packet hash

13. **Immutability**: A committed bundle is immutable forever. Recovery may delete staged artifacts but never rewrites committed bundles.

---

## IV. ASYNC EXCHANGE REALITY

14. **Orders Are Intents**: `ExecutionIntent` is transmitted AFTER packet commit. It expresses desire, not execution.

15. **Fills Are Facts**: `Fill` events are ingested as facts in minute `t+1` or later. They are inputs, not outputs.

16. **Deterministic Fill Ordering**: Fills are sorted by `(exchange_ts_ns, fill_id)` before ledger application. Exchange ordering is not trusted.

17. **Post-Commit Transmission**: Broker transmit occurs strictly after `commit_bundle()` succeeds.

---

## V. SEPARATION OF POWERS

18. **Senses (Data)**: Pure translation. No indicators, no inference, no opinion.

19. **Context (Regime)**: Permissive-only gate. Says YES or NO to trading. Forbidden from suggesting BUY or SELL.

20. **Safety (Diagnostics)**: Evaluates system health. Can VETO any trade. Cannot suggest direction.

21. **Will (Execution)**: The ONLY component that expresses trading intent. Acts only if Regime=YES AND Diagnostics≠VETO.

22. **Gate vs Audit**: 
    - Gate view: minimal, stable, crosses module boundary at runtime
    - Audit payload: rich, persisted in StatePacket, never crosses boundary

---

## VI. LEDGER SOVEREIGNTY

23. **Adapters Are Bytes Only**: Storage adapters translate and persist. They do not compute.

24. **Modules Compute**: Any function that produces `Fixed` values from other `Fixed` values lives in a module.

25. **Ledger Owns Money**: All accounting math (equity, PnL, margin, leverage) lives in `LedgerModule`. No exceptions.

---

## VII. ASYMMETRIC TRANSITIONS

26. **Fast Exit**: Any crisis signal forces immediate transition to CRISIS state. Score resets to zero.

27. **Slow Entry**: Transition to PERMISSIVE requires `slow_rebuild_score >= threshold`. No shortcuts.

28. **Engine Enforcement**: Regime invariants are enforced by the engine AFTER `RegimeModule.step()`. The module cannot violate them.

---

## VIII. SOVEREIGN PODS

29. **1:1 Binding**: One pod maps to one exchange sub-account. Separate API keys per pod.

30. **No Fleet Coordination**: No cross-pod aggregation or coordination exists inside the trading system.

31. **Sub-Account Verification**: Broker adapter must verify sub-account binding at startup before any operations.

32. **Evidence Binding**: `sub_account_id` is recorded in `EvidenceManifest` and redundantly in each `MinuteBundle` header.

---

## IX. EVIDENCE & ANCHORING

33. **Internal Hash Chain**: Every `StatePacket` links to its predecessor via `prev_packet_hash`. Chain is append-only.

34. **Evidence Manifest**: Stored once per run. Contains: `run_id`, `pod_id`, `sub_account_id`, code/config/schema/runtime hashes.

35. **RunType-Dependent Anchoring**:
    - `RESEARCH`: Internal hash chain only (reproducibility)
    - `SHADOW`/`LIVE`: Internal chain + external anchoring (tamper resistance)

36. **Anchoring Is Observational**: External anchoring is post-commit, best-effort, and never influences state hashes.

37. **LIVE Anchoring Failure Policy**: Force regime non-permissive (soft halt). Continue run. Record diagnostics flag. Resume when anchoring recovers.

---

## X. DENY BY DEFAULT

38. **Explicit Allowlist**: If a behavior, data source, or permission is not explicitly allowed, it is forbidden.

39. **RunType Permissions**:
    - `RESEARCH`: `allow_optimize=True`, `allow_place_orders=False`
    - `SHADOW`: `allow_optimize=False`, `allow_place_orders=False`
    - `LIVE`: `allow_optimize=False`, `allow_place_orders=True`

40. **Defense in Depth**: Permissions are checked at engine level AND at adapter level.

---

## XI. ERROR POLICY

41. **Explicit Table**: Every error code maps to exactly one action:
    - `HALT_RUN` — Invalid run, stop immediately
    - `VETO_AND_HOLD` — Continue but no orders this minute
    - `ENTER_CRISIS` — Force regime non-permissive
    - `RETRY_N` — Bounded retries with deterministic schedule

42. **No Implicit Behavior**: Unhandled errors are forbidden. All failure modes are enumerated.

---

## XII. OPERATIONAL LIMITS

43. **Per-Pod Bounds** (documented, enforced at construction):
    - Max BTC quantity: 1,000 BTC
    - Max price: $10,000,000 / BTC
    - Max notional: $10,000,000,000
    - Max cumulative fees/funding: $100,000,000

44. **64-Bit Safety**: All intermediate products fit within signed 64-bit integers given validated inputs.

---

## XIII. SCOPE FREEZE

45. **No Expansion Without Operational Need**: The following are frozen and shall not be expanded:
    - Semantic types (PRICE, QTY, USD, RATE only)
    - Evidence scope
    - Anchoring mechanisms  
    - Cross-pod coordination

46. **Strategy Work Proceeds**: All further development is in:
    - Strategy logic (inside `ExecutionModule`)
    - Regime models (inside `RegimeModule`)
    - Diagnostics thresholds
    - Research tooling

---

**END OF SYSTEM LAWS**

*Any code that violates these laws is invalid and must be corrected.*
