# STRATEGY FRAMEWORK CONTRACT

**Version**: 1.8.0
**Status**: DRAFT — Awaiting system owner approval
**Date**: 2026-02-12
**Authority Class**: Same as PHASE5_INTEGRATION_SEAMS.md, PHASE4A_INDICATOR_CONTRACT.md
**Changelog**: v1.8.0 applies 6 deltas required by the Strategy Composition Contract v1.5.2. Adds: `is_present`/`is_absent` operators, `MTM_DRAWDOWN_EXIT` exit type, per-output warmup, HANDOFF gate policy (unlocked), cross-indicator condition references, schema strictness. See Amendment Log at end of document.

---

## 1. Purpose and Scope

This contract defines the **general strategy composition framework** for the BTC Alpha system. It specifies how strategies are defined, configured, validated, and executed — independent of any specific strategy.

**This document DOES:**
- Define the strategy composition model (indicator instances, signal rules, execution parameters)
- Define the signal DSL grammar
- Define execution primitives (entry, exit, flip, trailing stop, scale-in)
- Define the multi-timeframe aggregation contract
- Define regime gating semantics and gate-exit policies
- Define the strategy artifact format (serialisation, hashing, promotion)
- Define delegation boundaries (what Claude Code may build vs what requires owner approval)

**This document DOES NOT:**
- Define specific strategies (those are strategy configs, not framework)
- Modify any frozen artifact (SYSTEM_LAWS, PHASE4A, PHASE4B, PHASE5_INTEGRATION_SEAMS)
- Replace or conflict with the existing indicator contract (Phase 4A/4B indicators are consumed, not redefined)
- Replace or conflict with the OMS contract (BYBIT_OMS_CONTRACT)
- Replace or conflict with the Snapshot Subsystem (LOW_LATENCY_SNAPSHOT_SUBSYSTEM)

**Relationship to existing contracts:**
- SYSTEM_LAWS.md → This contract operates within System Laws. All strategy computation obeys integer arithmetic, determinism, separation of powers, and evidence chain requirements.
- PHASE4A/4B → Indicators are consumed by this framework. The indicator registry is the observation space; strategies select from it.
- PHASE5_INTEGRATION_SEAMS → Strategies are deployed via CLI, run in pods, and produce evidence chains per the integration seams contract.
- BYBIT_OMS_CONTRACT → Execution intents produced by strategies are transmitted via the OMS. The OMS contract is authoritative for order execution semantics.
- LOW_LATENCY_SNAPSHOT_SUBSYSTEM → Strategies that require sub-minute cadence use the Snapshot subsystem as their data path. This framework supports both bar-based (1m aggregation) and snapshot-based (100ms-1s) data paths.

---

## 2. Strategy Composition Model

A strategy is composed of three layers. Each layer is independently configurable and the combination is serialised as a single strategy artifact.

### 2.1 Layer 1: Indicator Instances

An indicator instance binds a specific indicator from the Phase 4A/4B registry to a specific timeframe and parameter set.

**Schema (NORMATIVE):**
```
IndicatorInstance:
  label: str              # Human-readable, unique within strategy (e.g. "trigger", "filter_4h")
  indicator_id: int       # Phase 4A indicator ID (1-24) or diagnostic probe ID (25-29)
  timeframe: str          # One of: "100ms", "1s", "5s", "1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "3d"
  parameters: dict        # Indicator-specific parameters (per Phase 4A contract)
  outputs_used: list[str] # Which outputs this instance exposes to signal rules
  role: str               # Human-readable role description (documentation only, not evaluated)
  data_source: str        # "BAR" (≥1m, from candle pipeline) or "SNAPSHOT" (<1m, from snapshot subsystem)
  bar_provider: str       # For BAR instances: "EXCHANGE_CANDLES" (external feed) or "SNAPSHOT_BARS" (derived from snapshot stream)
```

**Rules:**
1. A strategy MAY have multiple instances of the same indicator at different timeframes.
2. A strategy MAY have multiple instances of the same indicator at the same timeframe with different parameters.
3. Each instance MUST have a unique label within the strategy.
4. The `indicator_id` MUST refer to a registered indicator in the Phase 4B engine.
5. The `outputs_used` field MUST reference valid output names for the given indicator.
6. The `parameters` MUST satisfy the parameter constraints defined in Phase 4A for the given indicator.
7. Instances with `data_source: "SNAPSHOT"` MUST only be used in strategies that declare a snapshot-based evaluation cadence for at least one path.
8. **Instance count limits (NORMATIVE):** Soft limit: 64 indicator instances per strategy (validation warning). Hard limit: 128 indicator instances per strategy (validation error, requires system owner override flag `allow_extended_instances: true` in config). This prevents O(n²) evaluation cost from unbounded instance counts.
9. **Path count limits (NORMATIVE):** Soft limit: 32 entry paths and 32 exit paths per strategy (validation warning). Hard limit: 64 entry paths and 64 exit paths (validation error). This prevents O(n) explosion per evaluation cycle, especially in snapshot-cadence strategies.
10. **Condition count limits (NORMATIVE):** Soft limit: 32 conditions per path (validation warning). Hard limit: 64 conditions per path (validation error). This applies to the total of standalone conditions plus all conditions within condition_groups for that path.
11. **bar_provider validation (NORMATIVE):** For instances with `data_source: "BAR"`, `bar_provider` MUST be set. If `bar_provider` is `"SNAPSHOT_BARS"` and any bar-based evaluation occurs, `persistence_mode` MUST be `"full"` (all base snapshots persisted for replay). If `bar_provider` is `"EXCHANGE_CANDLES"`, the candle data MUST be persisted separately with its own hash for replay determinism.

### 2.2 Layer 2: Signal Rules

Signal rules define the conditions under which the strategy enters, exits, or modifies positions. Entry and exit are **completely independent** — they may use different indicators, different timeframes, different logic, and different evaluation cadences.

**Structure (NORMATIVE):**
```
SignalRules:
  entry_rules: list[EntryPath]     # One or more entry paths (first matching fires)
  exit_rules: list[ExitPath]       # One or more exit paths (first matching fires)
  gate_rules: list[GateRule]       # Pre-conditions that must ALL be true for any entry signal
```

#### 2.2.1 Entry Paths

An EntryPath defines one way to enter a position. A strategy may have multiple entry paths (e.g. one for long, one for short). The first entry path whose conditions are all satisfied fires.

```
EntryPath:
  name: str                        # Human-readable (e.g. "long_trend_confluence")
  direction: "LONG" | "SHORT"
  evaluation_cadence: str          # When to evaluate (e.g. "5m", "1m", "100ms")
  conditions: list[Condition]      # ALL must be true (implicit AND)
  condition_groups: list[ConditionGroup]  # Named groups, each group is AND internally
```

**Condition Groups** allow logical structuring of conditions. All groups must pass (groups are AND'd together). Within a group, all conditions must pass (also AND). This maps to the MACD backtester's `macro_tfs` and `intra_tfs` grouping.

**Evaluation order (NORMATIVE):** Condition groups are evaluated first, in list order. If all groups pass, standalone conditions (the `conditions` field on the path) are then evaluated. If any group fails, standalone conditions are not evaluated (short-circuit). All must be true for the path to fire.

```
ConditionGroup:
  name: str                        # e.g. "macro_regime", "intraday_regime"
  conditions: list[Condition]      # ALL must be true within group
```

#### 2.2.2 Exit Paths

An ExitPath defines one way to exit a position. Multiple exit paths may exist (e.g. signal-based exit, trailing stop, time limit, gate-exit). The **first** exit path whose conditions are satisfied fires. Order matters — earlier paths have priority.

```
ExitPath:
  name: str                        # Human-readable (e.g. "daily_macd_turndown", "trailing_stop")
  applies_to: "LONG" | "SHORT" | "ANY"
  evaluation_cadence: str          # When to evaluate (may differ from entry)
  type: ExitType                   # SIGNAL | STOP_LOSS | TRAILING_STOP | TIME_LIMIT | GATE_EXIT
  conditions: list[Condition]      # For SIGNAL type: all must be true
  parameters: dict                 # Type-specific parameters (see Section 3)
```

**ExitType enumeration:**
- `SIGNAL`: Exit when conditions are met (e.g. daily MACD slope < 0)
- `STOP_LOSS`: Exit when price crosses a fixed level (exchange-side hard SL via OMS)
- `TRAILING_STOP`: Exit when price retraces N ATR from peak (adaptive)
- `TIME_LIMIT`: Exit after N bars of holding (regardless of conditions)
- `GATE_EXIT`: Exit when a regime gate closes (configurable policy, see Section 5)
- `MTM_DRAWDOWN_EXIT`: Exit when mark-to-market drawdown from peak exceeds threshold (v1.8.0, see §3.11)

**Minimum exit requirement (NORMATIVE):** Every strategy MUST define at least one exit path in `exit_rules` OR a `stop_loss` in execution params (or both). A strategy with no exit mechanism is a validation error.

#### 2.2.3 Gate Rules

Gate rules are pre-conditions that must ALL be true before any **entry** signal is evaluated. They are NOT signals — they are hard preconditions (System Law §19: Context is a permissive-only gate). Exit paths continue to evaluate regardless of gate state (except where gate-exit policy applies).

```
GateRule:
  name: str                        # e.g. "chop_regime_gate", "data_quality_gate"
  conditions: list[Condition]      # ALL must be true for gate to be OPEN
  on_close_policy: GateExitPolicy  # What happens when gate closes while position is open
```

**GateExitPolicy (NORMATIVE enumeration):**
- `FORCE_FLAT`: Immediately close any open position when gate closes.
- `HOLD_CURRENT`: Hold current position; block new entries. Exit only via exit paths.
- `HANDOFF` (v1.8.0): When the gate closes, the position is handed off to risk-management exits only. Signal-based exits (type `SIGNAL`) are suppressed. The position will only close via `STOP_LOSS`, `TRAILING_STOP`, `MTM_DRAWDOWN_EXIT`, `TIME_LIMIT`, or risk overrides (§5.1 step 2). Gate re-opening resumes normal signal exit evaluation. Available from engine_version 1.8.0. Configs specifying HANDOFF with engine_version < 1.8.0 MUST be rejected.

The gate-exit policy is a **per-strategy, per-gate configuration**. It is a philosophical risk decision that the system owner must approve for each strategy.

#### 2.2.4 List Ordering Semantics (NORMATIVE)

The ordering of entry paths, exit paths, and gate rules within their respective lists is **semantically binding**. List order determines evaluation priority (first match fires) and is included in the config hash computation. Any reordering changes behavior and produces a different config hash.

Implementations and tooling MUST preserve list ordering. Auto-formatting or sorting tools MUST NOT reorder these lists.

### 2.3 Layer 3: Execution Parameters

Execution parameters define how the strategy sizes, enters, and manages trades.

**Schema (NORMATIVE):**
```
ExecutionParams:
  position_sizing: PositionSizing
  entry_type: "MARKET"             # v1: MARKET only. LIMIT requires contract amendment.
  leverage: Decimal                # 1.0 for spot-equivalent, >1.0 for leveraged
```

**Leverage validation (NORMATIVE):** `leverage` MUST be > 0. In SHADOW and LIVE modes, `leverage` MUST NOT exceed the exchange maximum for the instrument (validated at config load against OMS-reported exchange limits). A config with leverage ≤ 0 is a validation error in all modes.
  stop_loss: StopLossConfig
  take_profit: Optional[TakeProfitConfig]
  trailing_stop: Optional[TrailingStopConfig]
  time_limit_bars: Optional[int]   # Exit after N bars (see time_limit_reference_cadence)
  time_limit_reference_cadence: Optional[str]  # Which cadence defines bar count (mandatory if time_limit_bars set)
  time_limit_allows_flip: bool     # Whether TIME_LIMIT exit participates in flip detection (default false)
```

**time_limit_reference_cadence validation (NORMATIVE):** If `time_limit_bars` is set, `time_limit_reference_cadence` MUST be set and MUST match either an entry path evaluation_cadence, an exit path evaluation_cadence, or an indicator instance timeframe declared in the same strategy config. Referencing a cadence not present in the strategy is a validation error.

**TIME_LIMIT counter semantics (NORMATIVE):** The TIME_LIMIT holding bar counter increments strictly at the close of the `time_limit_reference_cadence` timeframe, independent of entry or exit evaluation cadence. The counter starts at 0 when a position is opened and increments by 1 at each subsequent close of the reference cadence bar. The counter increments only on usable bars — if the reference cadence bar is unusable (gap_detected, missing, or stale), the counter does not increment for that bar.
  flip_enabled: bool               # Whether this strategy uses position flips
  scale_in: Optional[ScaleInConfig]
  funding_model: Optional[FundingModelConfig]  # For leveraged strategies (backtest cost model)
  trade_rate_limit: TradeRateLimitConfig        # MANDATORY (per Snapshot Subsystem §12.3)
  slippage_budget: SlippageBudgetConfig          # MANDATORY (per Snapshot Subsystem §12.4)
  warmup_restart_policy: WarmupRestartPolicy     # MANDATORY (per Snapshot Subsystem §7.3.1)
```

---

## 3. Execution Primitive Specifications

### 3.1 Position Sizing

```
PositionSizing:
  mode: "FIXED_FRACTION" | "VOL_TARGETED"
  fraction_of_equity: Decimal      # For FIXED_FRACTION: e.g. 0.5 (50%)
  target_vol: Decimal              # For VOL_TARGETED: e.g. 0.10 (10% annual)
  vol_indicator_label: str         # For VOL_TARGETED: reference to indicator instance
  max_leverage: Decimal            # For VOL_TARGETED: cap on sizing leverage
  min_leverage: Decimal            # For VOL_TARGETED: floor on sizing leverage
  min_vol_threshold: Decimal       # For VOL_TARGETED: minimum vol below which exposure = 0
```

**VOL_TARGETED safety guard (NORMATIVE):** If the volatility indicator output is zero, negative, or below `min_vol_threshold`, position sizing MUST return zero exposure (no position). Division by zero or near-zero volatility MUST NOT produce infinite or unbounded leverage. The `min_leverage` floor applies only when volatility is above `min_vol_threshold` and positive.

### 3.2 Stop Loss

```
StopLossConfig:
  mode: "FIXED_PRICE" | "ATR_MULTIPLE" | "PERCENT"
  atr_multiple: Optional[Decimal]
  atr_indicator_label: Optional[str]
  percent: Optional[Decimal]
  exchange_side: bool              # MUST be true for LIVE mode (per OMS contract)
```

**Exchange-side SL interaction (NORMATIVE):** Exchange-side stop losses execute asynchronously on the exchange. If the exchange fills a stop loss AND the framework simultaneously evaluates a SIGNAL exit at the same bar, the exchange fill takes precedence. The decision log MUST record `exit_reason: "EXCHANGE_STOP"` when the OMS reports an exchange-initiated fill, regardless of what the framework's signal evaluation produced. The framework MUST NOT emit a second exit order if the OMS reports the position is already closed by exchange-side SL. When EXCHANGE_STOP closes a position, flip detection MUST NOT be evaluated in the same cycle. Exit reason hierarchy for logging: `EXCHANGE_STOP > RISK_OVERRIDE > GATE_EXIT > TRAILING_STOP > TIME_LIMIT > SIGNAL`.

**Post-EXCHANGE_STOP behavior (NORMATIVE):** If the position is FLAT due to EXCHANGE_STOP before signal evaluation begins in a cycle, flip detection is not applicable (there is no position to flip from). Any opposite-direction entry condition is treated as a fresh entry subject to all normal checks (gates, rate limiter, slippage budget). Same-direction re-entry is also permitted in a subsequent cycle (not the same cycle — per §5.1 step 0 no-re-entry rule).

### 3.3 Trailing Stop

```
TrailingStopConfig:
  distance_atr_multiple: Decimal
  atr_indicator_label: str
  activation_profit_atr: Optional[Decimal]
  tighten_condition: Optional[TightenCondition]
```

```
TightenCondition:
  indicator_label: str
  output: str
  threshold: Decimal
  tightened_distance_atr: Decimal
```

### 3.4 Take Profit

```
TakeProfitConfig:
  mode: "FIXED" | "ATR_MULTIPLE" | "LADDER"
  legs: list[TPLeg]                # For LADDER mode (per OMS contract: client-side reduce-only)
```

```
TPLeg:
  fraction: Decimal
  target_atr_multiple: Decimal
```

### 3.5 Position Flip

When `flip_enabled: true`, the strategy may produce FLIP signals. The FLIP is a logically atomic two-step operation via the OMS (close current + open opposite). The OMS `incomplete_flip` state (per LOW_LATENCY_SNAPSHOT_SUBSYSTEM §5.3 and BYBIT_OMS_CONTRACT) MUST be checked before any new signal is processed. See §5.1.2 for flip signal generation rules.

### 3.6 Scale-In (EXTENSION — v2)

```
ScaleInConfig:
  mode: "TIME_DISTRIBUTED" | "CONDITION_TRIGGERED"
  num_bars: int
  fraction_per_bar: Decimal
  confirmation_conditions: list[Condition]
  max_additions: int
  add_fraction: Decimal
```

Architecture MUST NOT prevent future implementation.

**v1 validation guard (NORMATIVE):** If a strategy config includes a non-null `scale_in` field and engine_version is v1.x, the config MUST be rejected with an explicit validation error: "Scale-in is not available in engine v1.x. Requires contract amendment."

### 3.7 Funding Model (Backtest Cost Model)

```
FundingModelConfig:
  enabled: bool
  interval_hours: int
  rate_per_interval: Decimal
  credit_allowed: bool             # false = always-pay (conservative)
```

**Scope restriction (NORMATIVE):** FundingModelConfig affects RESEARCH/backtest cost modeling only. In SHADOW and LIVE modes, PnL accounting MUST rely solely on exchange-reported funding debits/credits. The funding model config MUST NOT influence live or shadow trading decisions, position sizing, or risk calculations. SHADOW mode uses real exchange funding events, not the simulated model.

### 3.8 Trade Rate Limiting (MANDATORY)

Per LOW_LATENCY_SNAPSHOT_SUBSYSTEM §12.3. Prevents runaway overtrading from gate flicker or strategy instability.

```
TradeRateLimitConfig:
  min_time_between_trades_ms: int  # e.g. 10000 (10 seconds)
  max_trades_per_hour: int         # e.g. 20
```

Rate limiting applies ONLY to entry and flip signals. Exit signals (including FORCE_FLAT, EXIT_ALL, and signal-based exits) MUST bypass rate limiting unconditionally. See §5.1 step 3.

### 3.9 Slippage Budget (MANDATORY)

Per LOW_LATENCY_SNAPSHOT_SUBSYSTEM §12.4. Prevents death by accumulated slippage.

```
SlippageBudgetConfig:
  max_slippage_bps_per_hour: int   # e.g. 50 (0.5%)
```

Slippage budget applies ONLY to entry and flip signals. Exit signals MUST bypass slippage budget unconditionally. HALT_TRADING means "halt new entries," not "halt all order transmission." See §5.1 step 4.

**Catastrophic slippage cooldown (NORMATIVE):** If any single exit trade incurs slippage exceeding 50% of the hourly slippage budget (e.g. 25 bps if budget is 50 bps/hour), the framework MUST trigger an automatic entry halt for the remainder of the hourly window. This does not block further exits — only new entries and flips. The event is logged as `catastrophic_slippage_cooldown` in the decision log.

**Slippage arithmetic (NORMATIVE):** Accumulated slippage computation MUST use Decimal arithmetic quantised to 2 decimal places (bps) via `ROUND_HALF_EVEN` before comparison to `max_slippage_bps_per_hour`. This prevents threshold boundary divergence across Decimal contexts.

### 3.10 Warmup Restart Policy (MANDATORY)

Per LOW_LATENCY_SNAPSHOT_SUBSYSTEM §7.3.1. Defines behavior when a strategy restarts with an open position.

```
WarmupRestartPolicy:
  mode: "EMERGENCY_FLATTEN" | "HARD_STOPS"
  hard_stop_percent: Optional[Decimal]  # For HARD_STOPS mode (e.g. 0.03 = 3%)
```

**EMERGENCY_FLATTEN:** Close position immediately before warmup begins.
**HARD_STOPS:** Set price-based emergency stops (no indicators required) and monitor during warmup.

Positions MUST NOT be held unmonitored during indicator warmup. This is non-negotiable.

### 3.11 MTM Drawdown Exit (v1.8.0)

```
MTMDrawdownExitConfig:
  enabled: bool
  evaluation_cadence: str           # When to check (e.g. "1m")
  drawdown_bps_long: int            # Max drawdown from peak in basis points for long positions
  drawdown_bps_short: int           # Max drawdown from peak in basis points for short positions
  applies_to: list["LONG" | "SHORT"]  # Which directions this exit applies to
```

**Drawdown computation (NORMATIVE):**
```
For longs:  current_pnl_bps = (current_price - entry_price) * 10000 / entry_price
For shorts: current_pnl_bps = (entry_price - current_price) * 10000 / entry_price
peak_pnl_bps = max(current_pnl_bps) since entry   (tracked incrementally)
current_drawdown_bps = peak_pnl_bps - current_pnl_bps
```

All arithmetic uses Fixed-point integer bps (not floating point). P&L is **gross mark-to-market** — excludes fees, funding, and slippage. These are accounted for separately via the funding model and risk accounting. MTM drawdown measures price movement from peak, not realized economic P&L. This is the same P&L basis used for STOP_LOSS percent calculations (§3.2). `entry_price` is the fill price at position open. `current_price` is the decision price at the evaluation bar (§4.1.1).

**Trigger:** If `current_drawdown_bps >= drawdown_bps_{direction}`, the exit path fires. The comparison is `>=` (deterministic, no floating-point boundary issues).

**Evaluation cadence:** Checked at each bar close of the specified cadence. Not checked between bars.

**Peak reset:** Peak tracking resets when a new position is opened. On position close, peak state is discarded.

**Flip interaction:** If `flip_enabled: true` and MTM drawdown fires as the exit leg, flip detection (§5.1.2) applies normally. The MTM drawdown exit participates in flip detection like `SIGNAL` type exits.

**Precedence:** MTM_DRAWDOWN_EXIT follows standard exit path precedence (§5.1.1 priority 4). It is a strategy-defined exit, not a risk override. Risk overrides (priority 1) and gate-exit (priority 2) take precedence.

**Interaction with STOP_LOSS:** MTM drawdown and stop loss are independent. Stop loss is exchange-side and price-based. MTM drawdown is engine-side and P&L-based. Both may coexist. Whichever triggers first fires.

**Version gating (NORMATIVE):** Configs using `MTM_DRAWDOWN_EXIT` require `engine_version >= 1.8.0`. Configs with this exit type and engine_version < 1.8.0 MUST be rejected at validation.

---

## 4. Signal DSL Grammar

### 4.1 Condition Primitives (NORMATIVE)

Every condition: `<source> <operator> <value>` or `<source> <operator>` (for unary operators) or `<source> <operator> <ref_source>` (for cross-indicator comparisons).

**Source** is one of:
- `{label}.{output}` — indicator instance output
- `{label}.{output}.slope` — first derivative (bar-over-bar delta sign)
- `price` — current decision price (see §4.1.1)

**Core Operators (v1.0 — engine_version >= 1.0.0):**

| Operator | Meaning | Formal Definition | Example |
|----------|---------|-------------------|---------|
| `>` | Greater than | `current > value` | `filter_4h.macd_line.slope > 0` |
| `<` | Less than | `current < value` | `daily_macd.macd_line.slope < 0` |
| `>=` | Greater or equal | `current >= value` | `adx.adx_value >= 0.25` |
| `<=` | Less or equal | `current <= value` | `chop.choppiness <= 0.618` |
| `==` | Equals | `current == value` | `trigger.slope_sign == 1` |
| `crosses_above` | Upward crossing | `previous <= value AND current > value` | `trigger.macd_line.slope crosses_above 0` |
| `crosses_below` | Downward crossing | `previous >= value AND current < value` | `trigger.macd_line.slope crosses_below 0` |

**v1.8.0 Operators (engine_version >= 1.8.0):**

| Operator | Meaning | Formal Definition | Example |
|----------|---------|-------------------|---------|
| `is_present` | Output has value | `current IS NOT None` | `exit_3d.slope_sign is_present` |
| `is_absent` | Output is missing | `current IS None` | `exit_3d.slope_sign is_absent` |

**`is_present` / `is_absent` semantics (NORMATIVE):**
- These are **unary operators** — they reference an indicator label and output name but take no comparison value. The condition object has `indicator`, `output`, and `operator` fields only. No `value` field.
- `is_present` returns `true` if the indicator output is non-None at the current evaluation bar. Returns `false` if None (during warmup, data gaps, or HTF mid-bar).
- `is_absent` returns `true` if the indicator output is None. Returns `false` if non-None.
- These are **stateless operators** — they do not require previous values and do not appear in §4.5 (operator state persistence).
- If a `value` field is present in a condition using `is_present` or `is_absent`, the engine MUST reject at validation.
- These operators are consistent with the existing None handling rule (§4.1.1): the difference is that `is_present`/`is_absent` explicitly test for None, whereas existing comparison operators implicitly treat None as "condition false."

**Cross-indicator condition references (v1.8.0 — engine_version >= 1.8.0):**

In addition to comparing an indicator output against a literal `value`, conditions may compare one indicator's output against another indicator's output:

```
Condition (extended format):
  indicator: str         # label of primary indicator
  output: str            # output name on primary indicator
  operator: str          # comparison operator
  value: Decimal         # literal comparison value — MUTUALLY EXCLUSIVE with ref_indicator
  ref_indicator: str     # label of second indicator — MUTUALLY EXCLUSIVE with value
  ref_output: str        # output name on second indicator — required if ref_indicator present
```

**Cross-indicator rules (NORMATIVE):**
- `value` and `ref_indicator` are mutually exclusive. If both are present, validation MUST fail.
- If `ref_indicator` is present, `ref_output` MUST also be present. If `ref_indicator` is present but `ref_output` is absent, validation MUST fail.
- **outputs_used enforcement:** Any output referenced in a condition — whether via `indicator.output` or `ref_indicator.ref_output` — MUST appear in the corresponding instance's `outputs_used` array. Violation is a validation error at config load time.
- Cross-indicator comparison: `indicator.output <operator> ref_indicator.ref_output`
- Both values must be non-None at evaluation time. If either is None, the condition evaluates to `false` (consistent with §4.1.1 None handling).
- Both indicators must have completed warmup for the referenced outputs. If either output is in warmup, the condition evaluates to `false`.
- Price-denominated outputs on both sides are normalised to 8 decimal places per §4.1.1. Non-price outputs compared directly.
- `crosses_above` and `crosses_below` work with cross-indicator references: "previous" on both sides refers to each indicator's own previous completed bar value.
- **Cross-timeframe crossing:** Each side uses its own indicator's most recently completed bar at its own timeframe as of the evaluation timestamp. "Previous" for each side refers to the value at the **prior completed bar of that indicator's own timeframe** (consistent with the cross-timeframe `previous` definition below). If either indicator's timeframe bar has not updated since the last evaluation, `previous == current` for that side, and no crossing occurs for that side's contribution.
- `is_present` and `is_absent` MUST NOT be used with `ref_indicator` (they are unary). Validation MUST reject this combination.
- Configs using `ref_indicator` require `engine_version >= 1.8.0`.

**Cross-timeframe `previous` definition (NORMATIVE):** For `crosses_above` and `crosses_below`, `previous` refers to the value from the **previous completed bar of that indicator's own timeframe**, not the previous evaluation tick. If a 5m evaluation path references a 1h indicator, and the 1h bar has not changed since last evaluation, `previous` and `current` are the same value, and no crossing occurs. A crossing is detected only when the indicator's own timeframe bar updates.

**Extension Operators (design for, build later):**

| Operator | Meaning | Example |
|----------|---------|---------|
| `FOR N bars` | Duration condition | `bb_width.bandwidth < 0.02 FOR 20 bars` |
| `WITHIN X OF` | Proximity to level | `price WITHIN 2.0 ATR OF sr.resistance` |

**v1/v1.8 Scope Statement (NORMATIVE):** The core operators (v1.0) and v1.8.0 operators listed above are the operators available in engine_version 1.8.0. Extension operators (`FOR N bars`, `WITHIN X OF`) are designed for in the architecture but not implemented. Strategies requiring extension operators cannot be deployed until the relevant operator is implemented and the contract is amended. The `RELATIVE >` extension operator from v1.7.0 is superseded by cross-indicator condition references (above).

**Operator set versioning (NORMATIVE):** The set of available operators is scoped to the engine version. Strategy configs MUST declare `min_operator_version` (defaulting to "1.0" for v1 core operators). When extension operators are added in future versions, they receive a version tag. The engine MUST refuse to evaluate a config whose `min_operator_version` exceeds the engine's implemented operator set. This ensures old configs remain replayable and new configs are explicitly version-gated. Operator availability MUST be mapped to engine_version via a hard-coded compatibility table within the engine. The engine MUST reject any config using operators not explicitly enabled for its engine_version.

**Operator compatibility table (NORMATIVE):**

| Operator | Min engine_version |
|----------|--------------------|
| `>`, `<`, `>=`, `<=`, `==`, `crosses_above`, `crosses_below` | 1.0.0 |
| `is_present`, `is_absent` | 1.8.0 |
| Cross-indicator references (`ref_indicator`, `ref_output`) | 1.8.0 |

#### 4.1.1 Decision Price Definition (NORMATIVE)

In the signal DSL, `price` refers to the **decision price** for the current evaluation:

- For bar-based strategies (data_source: BAR): the close of the most recently completed bar at the path's evaluation cadence.
- For snapshot-based strategies (data_source: SNAPSHOT): the `signal_price` from the current snapshot, determined by the strategy's `signal_price_source` config (last, mark, or index — per LOW_LATENCY_SNAPSHOT_SUBSYSTEM §3.1).

`price` is always the price used for **signal evaluation**, NOT the execution price. Execution price (including slippage, spread crossing, and fill price) is determined by the OMS at order time and may differ from decision price.

All price comparisons in the DSL MUST operate on Decimal values normalised to 8 decimal places via `Decimal.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_EVEN)`, consistent with the Snapshot Subsystem's canonical serialisation (LOW_LATENCY_SNAPSHOT_SUBSYSTEM §8.2). Bar-derived prices AND price-denominated indicator outputs used in DSL comparisons MUST be normalised to the same 8-decimal precision and rounding mode before comparison. This prevents threshold boundary divergence from differing internal precision across implementations.

**Cross-contract consistency (NORMATIVE):** The `ROUND_HALF_EVEN` rounding mode used here MUST be identical to the rounding mode used in snapshot hash serialisation. If the Snapshot Subsystem does not explicitly mandate `ROUND_HALF_EVEN`, the framework implementation MUST use its own normalisation function for DSL comparisons (not reuse the snapshot serialiser) and this function MUST use `ROUND_HALF_EVEN`. Any future Snapshot Subsystem amendment that changes rounding mode requires a corresponding framework amendment.

**Non-price indicator outputs:** Indicator outputs that are not price-denominated (e.g. volatility ratios, slope values, dimensionless indices) retain their internal Decimal precision and are compared directly using Decimal arithmetic without 8-decimal normalisation. Normalisation applies only to outputs whose unit is the instrument's quote currency.

**None handling (NORMATIVE):** If a referenced indicator output is None at evaluation time (e.g. during warmup or due to data gap), the condition evaluates to `false`. No exception is raised. This is consistent with the conservative warmup approach: incomplete data cannot satisfy a trading condition. Similarly, if `price` itself is None (e.g. missing snapshot signal_price), all conditions referencing `price` evaluate to `false` and a deterministic diagnostic `price_missing=true` is recorded in the decision log.

**Decimal context (NORMATIVE):** All DSL comparisons MUST operate under a fixed Decimal context set at engine startup: precision ≥ 28 (Python default), rounding = ROUND_HALF_EVEN. The context MUST NOT be modified during evaluation. This ensures cross-platform determinism for all arithmetic operations in the signal evaluation path. The Decimal context (precision and rounding mode) MUST be recorded in the run manifest. Any change to Decimal context settings requires an engine_version increment.

### 4.2 Condition Composition

Conditions within a path or group are combined with implicit AND. To express OR, define multiple entry/exit paths — first matching fires.

### 4.3 Evaluation Cadence Rules

1. Entry and exit paths MAY have different cadences.
2. Gate rules evaluate at the fastest cadence among all active paths. At each gate evaluation timestamp, each referenced indicator uses its most recently completed bar for that indicator's timeframe. No partial-bar or in-progress values are ever used. If the indicator's most recent bar has not changed since the last evaluation, the same value is reused. This is expected and deterministic. **Performance optimisation (NORMATIVE):** Gate conditions referencing indicators whose timeframe bar has not updated since the previous evaluation MUST short-circuit and reuse the cached boolean result for that condition. This avoids redundant re-evaluation and prevents operator state anomalies for future stateful gate operators. **Gate state transition logging (NORMATIVE):** Gate state transitions (open→closed or closed→open) MUST be recorded in the decision log only when the underlying indicator(s) change value at their own timeframe close. Fast-cadence re-evaluations that observe no underlying indicator change MUST NOT produce new gate state transition events.
3. Risk overrides evaluate at every decision point regardless of cadence.
4. A strategy mixing bar-based and snapshot-based cadences MUST declare `data_source` per instance.
5. **Hybrid persistence validation (NORMATIVE):** If any indicator instance has `data_source: "BAR"` and the strategy uses `decision_timebase: "bar_close"`, then `persistence_mode` MUST be `"full"` (per LOW_LATENCY_SNAPSHOT_SUBSYSTEM §8.1.1). This applies even if other instances use SNAPSHOT data source. The most demanding persistence requirement wins. Additionally: if BAR-timeframe indicators (≥1m) are computed from bars that are themselves derived from the snapshot stream (rather than an external candle feed), `persistence_mode` MUST be `"full"` to ensure replay can reconstruct identical bars from the persisted snapshot series.
6. **Hybrid synchronisation (NORMATIVE):** For strategies mixing BAR and SNAPSHOT data sources, the evaluation cycle at timestamp T MUST NOT begin until both the snapshot stream and bar aggregation pipeline have completed all updates for T. The scheduler must wait for the slower pipeline before releasing the evaluation cycle. **Timeout (NORMATIVE):** If the slower pipeline has not completed within 5000ms of the faster pipeline's readiness, the scheduler MUST trigger HALT with diagnostic `pipeline_desync`. This is a fatal condition requiring operator intervention. The scheduler MUST NOT auto-retry or auto-recover — pipeline desynchronisation indicates a systemic issue that must be diagnosed. **Scope:** Pipeline desync HALT is symbol-scoped. Only the affected symbol's evaluation loop is halted. Other symbols in the same process continue independently. HALT means no further evaluation cycles for that symbol until operator restart.

### 4.4 DSL Examples

**MACD Multi-TF Confluence (from backtester):**
```yaml
entry_rules:
  - name: "long_macd_confluence"
    direction: LONG
    evaluation_cadence: "5m"
    condition_groups:
      - name: "macro_regime"
        conditions:
          - "filter_3d.slope_sign > 0"
          - "filter_1d.slope_sign > 0"
          - "filter_12h.slope_sign > 0"
      - name: "intraday_regime"
        conditions:
          - "filter_1h.slope_sign > 0"
          - "filter_30m.slope_sign > 0"
          - "filter_15m.slope_sign > 0"
    conditions:
      - "trigger_5m.macd_line.slope crosses_above 0"

exit_rules:
  - name: "daily_macd_turndown"
    applies_to: LONG
    evaluation_cadence: "1d"
    type: SIGNAL
    conditions:
      - "exit_daily.macd_line.slope < 0"
  - name: "hard_stop"
    applies_to: LONG
    type: STOP_LOSS
    parameters:
      mode: PERCENT
      percent: 0.05
      exchange_side: true
```

**DBMR Band Strategy:**
```yaml
gate_rules:
  - name: "chop_regime_gate"
    conditions:
      - "chop.choppiness > 0.5"
      - "atr_pct.atr_percent < 0.03"
      - "midpoint_slope.slope_sign == 0"
    on_close_policy: FORCE_FLAT  # Requires owner approval

entry_rules:
  - name: "long_at_lower_band"
    direction: LONG
    evaluation_cadence: "1m"
    conditions:
      - "price <= band.lower_band"
  - name: "short_at_upper_band"
    direction: SHORT
    evaluation_cadence: "1m"
    conditions:
      - "price >= band.upper_band"

exit_rules:
  - name: "flip_to_short"
    applies_to: LONG
    evaluation_cadence: "1m"
    type: SIGNAL
    conditions:
      - "price >= band.upper_band"
  - name: "flip_to_long"
    applies_to: SHORT
    evaluation_cadence: "1m"
    type: SIGNAL
    conditions:
      - "price <= band.lower_band"
```

### 4.5 Operator State Persistence (NORMATIVE)

The following operators are **stateful** — they require values from the previous evaluation to compute their current result:

- `crosses_above`: Requires previous value of the source
- `crosses_below`: Requires previous value of the source
- `FOR N bars` (extension): Requires consecutive-bar counter

**Coupling rule:** Operator state is coupled to indicator state. If indicator internal state is not persisted across restarts (as specified in LOW_LATENCY_SNAPSHOT_SUBSYSTEM §7.1), then operator state MUST also be reset on restart. Operator state is only valid when the indicator it references has completed warmup and produced at least one valid output in the current session.

**Post-restart behavior:** After restart and indicator warmup completion, stateful operators evaluate to `false` for their first evaluation tick (no previous value available). This is conservative — it prevents false crossover signals but may delay entry by one evaluation tick.

**Restart initialisation invariant (NORMATIVE):** On restart, all operator state MUST be explicitly initialised to "no previous value" BEFORE the first evaluation tick. Stale operator state from memory cache, prior checkpoint, or any other source MUST NOT be used if indicator state has been reset. The initialisation must occur after indicator warmup completes but before the first signal evaluation.

**Persistence exception:** If a future version implements full indicator state persistence, operator state MAY be persisted alongside it, provided both are restored atomically from the same checkpoint.

---

## 5. Regime Gate Semantics

### 5.1 Evaluation Pipeline (NORMATIVE)

The full evaluation pipeline, from outermost safety check to innermost strategy logic:

```
0. Indicator update and evaluation cycle coalescing (MANDATORY)
   → All path evaluations occurring at the same UTC timestamp are coalesced
     into a single evaluation cycle. A timestamp that is simultaneously a 5m
     boundary, 1h boundary, and 1D boundary produces ONE evaluation cycle,
     not three separate ones.
   → **Canonical timebase (NORMATIVE):** All internal timestamps in the
     evaluation engine are Unix nanoseconds UTC, consistent with Snapshot
     Subsystem §3.1 ts_utc. Config values specified in milliseconds (e.g.
     min_time_between_trades_ms) are converted to nanoseconds by
     multiplication (× 1_000_000), never by division. Cadence values in
     seconds are converted to nanoseconds by multiplication (× 1_000_000_000).
   → Within that cycle, ALL indicator instances whose timeframe bar has closed
     at or before this timestamp MUST be updated BEFORE any gate or signal
     evaluation. Gate and signal evaluation MUST operate on a single,
     consistent, post-update indicator state snapshot. No gate or signal
     condition may observe a partially-updated indicator set.
   → Evaluation engine MUST execute indicator update and signal evaluation
     in a single-threaded critical section per symbol. Multi-threaded
     evaluation of the same symbol's indicators and signals is prohibited.
   → All indicator state, operator state, scheduler state, rate/slippage
     counters, and evaluation locks are strictly per-symbol. Cross-symbol
     state sharing is prohibited. Each symbol's evaluation is fully
     independent (per LOW_LATENCY_SNAPSHOT_SUBSYSTEM §13.1).
   → Within that single cycle, precedence rules in §5.1.1 apply.
   → Specifically: within a coalesced cycle, ALL exit path conditions (across
     all cadences) MUST be evaluated before any entry path conditions,
     regardless of path ordering or cadence. This prevents same-cycle
     re-entry after exit.
   → If an exit (non-flip) fires for a position, entry paths for the SAME
     direction MUST NOT be evaluated in that cycle. Re-entry requires a
     subsequent evaluation timestamp. Opposite-direction entry is handled
     by flip detection (§5.1.2).

1. OMS incomplete_flip check (MANDATORY — per Snapshot Subsystem §5.3)
   → If incomplete_flip pending: HOLD_PENDING_FLIP. Block all entries, flips,
     and new position-increasing signals. However, reduce-only emergency
     actions ARE permitted: risk override EXIT_ALL (step 2) and gate
     FORCE_FLAT (step 5) may execute reduce-only exits during incomplete
     flip. This reconciles with Snapshot Subsystem §5.3 which allows
     reduce-only exits "if needed for safety" during incomplete flip.
   → If incomplete_flip persists beyond 30 seconds without state change:
     HALT with diagnostic `flip_stalled`. This prevents indefinite freeze
     if exchange never confirms the open leg of a flip.

2. Risk overrides (MANDATORY — per Snapshot Subsystem §5.2)
   a. Critical divergence (mark/index > threshold) → EXIT_ALL if exposed, HOLD if flat
   b. Spread anomaly (bid/ask > threshold) → HOLD if flat, CONSIDER_EXIT if exposed
   c. Deviation anomaly (lastPrice vs index > threshold) → EXIT_ALL if exposed, HOLD if flat
   d. Data unusable (any anomaly flag) → CONSIDER_EXIT if exposed, HOLD if flat
   e. Staleness confirmed (N consecutive stale ticks) → Apply stale_policy

3. Trade rate limiter check (MANDATORY — per §3.8)
   → Applies ONLY to entry and flip signals.
   → Exit signals (including FORCE_FLAT, EXIT_ALL, and signal-based exits)
     MUST bypass rate limiting unconditionally.

4. Slippage budget check (MANDATORY — per §3.9)
   → Applies ONLY to entry and flip signals.
   → Exit signals MUST bypass slippage budget unconditionally.
   → HALT_TRADING means "halt new entries," not "halt all order transmission."

5. Gate rules (strategy-defined regime conditions — per §2.2.3)
   → If gate closed: apply on_close_policy (FORCE_FLAT / HOLD_CURRENT / HANDOFF)
   → Gate-exit (FORCE_FLAT) bypasses steps 3 and 4 (it is an exit, not an entry)

6. Signal rules (entry/exit evaluation — per §2.2.1, §2.2.2)
   → Evaluate exit paths first, then entry paths (see §5.1.1)
   → First matching path fires within each category

7. Execution (OMS transmission — per BYBIT_OMS_CONTRACT)
   → Convert signal to TradePlan, transmit via OMS
```

Step 0 ensures atomic indicator state. Steps 1-4 are safety infrastructure. They are NOT configurable per strategy (though thresholds may be). Steps 5-6 are strategy-specific. Step 7 is the OMS boundary.

**Determinism invariant (NORMATIVE):** Given identical snapshot series, identical strategy config, identical OMS state, and identical Decimal context, the evaluation engine MUST produce identical decision sequences. No nondeterministic elements (system time, random numbers, unordered dict iteration, thread scheduling, environment-dependent defaults) are permitted in the evaluation path. This is the foundational invariant that enables replay and evidence chain verification.

**CONSIDER_EXIT mapping (NORMATIVE):** When a risk override (step 2) produces CONSIDER_EXIT (e.g. spread anomaly with open position), the strategy framework treats it as follows: exit path evaluation proceeds normally (not blocked, not forced). If any exit path fires, the exit executes. If no exit path fires, the position is held. A diagnostic flag `risk_override_consider_exit = true` is recorded in the decision log. CONSIDER_EXIT does NOT force an exit. It is distinct from EXIT_ALL (which forces immediate exit) and HOLD (which blocks all evaluation). **CONSIDER_EXIT does NOT qualify as an exit condition for flip detection.** Only exit paths (type SIGNAL, STOP_LOSS, TRAILING_STOP, TIME_LIMIT, GATE_EXIT) qualify as the exit leg of a flip.

**Version coupling (NORMATIVE):** The framework interpretation of CONSIDER_EXIT is locked to LOW_LATENCY_SNAPSHOT_SUBSYSTEM v1.3.0 semantics. If the Snapshot Subsystem amends the meaning of CONSIDER_EXIT in a future version, the Strategy Framework contract MUST be amended to match before the new snapshot version is deployed.

#### 5.1.1 Signal Precedence (NORMATIVE)

At any single evaluation timestamp, signals are resolved in this strict priority order:

1. **Risk override exits** (EXIT_ALL from step 2) — highest priority, immediate
2. **Gate-exit signals** (FORCE_FLAT from step 5) — second priority
3. **Flip detection** (if flip_enabled=true: check if exit condition AND opposite entry condition are both true — see §5.1.2) — third priority. Flip detection MUST be evaluated using the full post-update indicator state BEFORE resolving pure exit-only behavior, regardless of path list order. If flip conditions are met, emit FLIP. If flip conditions partially met (exit true but entry blocked by gate/rate/slippage), emit EXIT only. Exit-only resolution (priority 4) is applied only after flip detection has been attempted and failed.
4. **Exit path signals** (from step 6 exit evaluation, no matching entry) — fourth priority. If multiple exit paths evaluate to true simultaneously, the framework MUST log a `multi_exit_trigger` diagnostic event recording all satisfied exit paths and which path was selected (first in list order).
5. **Entry path signals** (from step 6 entry evaluation, no current position) — lowest priority

**Rule:** If a risk override exit (priority 1) or gate-exit (priority 2) fires, all lower priorities are skipped. Flip detection (priority 3) is evaluated before pure exit resolution (priority 4) so that flip-capable strategies do not accidentally decompose into exit-only when both sides are satisfied.

**Multi-path trigger logging:** If more than one entry path evaluates to true at the same timestamp, the framework MUST log a `multi_entry_trigger` diagnostic event recording all paths that were true and which path was selected (first in list order). This aids strategy debugging without changing behavior.

**Optional strict mode:** Strategies MAY set `strict_entry_paths: true` in execution params, which causes the framework to HALT with a diagnostic error if multiple entry paths evaluate to true simultaneously. Similarly, `strict_exit_paths: true` causes HALT if multiple exit paths evaluate to true simultaneously. Both default to `false` (log-only). These catch unintentional path overlap during RESEARCH.

#### 5.1.2 Flip Signal Generation (NORMATIVE)

A FLIP signal is generated when ALL of the following are true at the same evaluation timestamp:

1. An exit path for the current position direction fires (conditions satisfied)
2. An entry path for the opposite direction fires (conditions satisfied)
3. `flip_enabled = true` in execution params
4. All gates are open (no gate blocks entry)
5. Rate limiter and slippage budget permit entry (flip is entry-side for rate/slippage purposes)

When a FLIP is detected, the framework MUST emit a single FLIP intent to the OMS, not separate EXIT + ENTRY intents. The OMS handles the two-leg execution atomically per BYBIT_OMS_CONTRACT.

If any of conditions 3-5 fail, the framework emits EXIT only (no entry component). When flip decomposes to exit-only due to gate closure, rate limiting, or slippage budget, the framework MUST log a `flip_decomposed` diagnostic event recording the blocking reason (gate_closed, rate_limited, slippage_exceeded).

**Gate-flip interaction (NORMATIVE):** If a gate closes on the same timestamp that a flip would otherwise fire, the flip decomposes to EXIT only. The entry leg is suppressed. This is by design — System Law §19 makes the gate a hard precondition. If the gate says the regime is invalid, no new entry is permitted even as part of a flip. The DBMR strategy must account for this: if the chop gate closes while at a band boundary, the position exits but does not flip. Re-entry requires the gate to re-open.

**TIME_LIMIT in flip detection (NORMATIVE):** A TIME_LIMIT exit MAY qualify as the exit condition for flip detection, controlled by the config flag `time_limit_allows_flip: bool` (default `false`). When `false`, a TIME_LIMIT exit always produces EXIT only, never FLIP, even if opposite-direction entry conditions are true. When `true`, TIME_LIMIT participates in flip detection like any other exit type. Default is conservative (no flip on time expiry) to prevent unintended oscillation in mean-reversion strategies.

### 5.2 Gate-Exit Policy Execution

**FORCE_FLAT:** Immediately emit EXIT signal. Exit at next available price. FORCE_FLAT is classified as an exit signal and therefore bypasses rate limiting and slippage budget checks (§5.1 steps 3-4).

**HOLD_CURRENT:** Block new entries. Hold current position. Normal exit paths continue to evaluate — gate closure does NOT block exit path evaluation.

**HANDOFF (v1.8.0):** Block new entries. Suppress exit path evaluation strictly for exit rules where `exit_type == SIGNAL`. All other exit types (`STOP_LOSS`, `TRAILING_STOP`, `MTM_DRAWDOWN_EXIT`, `TIME_LIMIT`, `GATE_EXIT`) evaluate normally. Risk overrides (§5.1 step 2) evaluate normally.

**Suppression ordering:** SIGNAL exit suppression is applied at step 6 (signal evaluation) in the §5.1 pipeline. Steps 1–5 are unaffected. Within step 6, SIGNAL-type exit paths are skipped entirely — their conditions are not evaluated. Non-SIGNAL exit paths evaluate normally per list order.

**No persistent state:** Suppressed SIGNAL exits do not accumulate state. There is no "deferred signal" or "queued exit." Each evaluation cycle re-evaluates all conditions from scratch.

**Gate reopen:** When the gate reopens, SIGNAL exit evaluation resumes immediately on the next evaluation cycle. If SIGNAL exit conditions are true on the first evaluation after reopen, the exit fires on that bar. No delay, no cooldown.

**Version gating:** HANDOFF is available from engine_version 1.8.0. Configs specifying HANDOFF with engine_version < 1.8.0 MUST be rejected.

#### 5.2.1 Gate Policy Conflict Resolution (NORMATIVE)

If multiple gates close simultaneously with different on_close_policies, the most restrictive policy applies. Restriction order:

**FORCE_FLAT > HANDOFF > HOLD_CURRENT**

FORCE_FLAT always wins. If any gate demands FORCE_FLAT, position is closed regardless of other gates' policies.

**Validation diagnostic (NORMATIVE):** If a strategy config defines multiple gates with different on_close_policies, the framework MUST emit a validation warning identifying the conflicting policies and noting that the most restrictive will always apply. This helps strategy designers detect unintentional gate shadowing.

### 5.3 Gate-Exit Policy Approval

Gate-exit policies require system owner approval for SHADOW and LIVE. Implementers MAY set them freely in RESEARCH.

---

## 6. Multi-Timeframe Aggregation Contract

### 6.1 Supported Timeframes

| Timeframe | 1m bars | Alignment | Status |
|-----------|---------|-----------|--------|
| 1m | 1 | Raw | EXISTS |
| 5m | 5 | Divisible by 300s from epoch | REQUIRED |
| 15m | 15 | Divisible by 900s from epoch | REQUIRED |
| 30m | 30 | Divisible by 1800s from epoch | REQUIRED |
| 1h | 60 | Divisible by 3600s from epoch | REQUIRED |
| 4h | 240 | Divisible by 14400s from epoch | EXISTS |
| 12h | 720 | Divisible by 43200s from epoch | REQUIRED |
| 1d | 1440 | Divisible by 86400s from epoch | REQUIRED |
| 3d | 4320 | Divisible by 259200s from epoch | REQUIRED |

All alignments are anchored to **Unix epoch (1970-01-01 00:00:00 UTC)**. For 3d bars, this means boundaries do not align with calendar weeks (epoch is a Thursday). This is intentional — calendar alignment would require special-case logic that creates complexity without clear benefit. The anchor is deterministic and consistent across all implementations. Strategies requiring calendar-week or funding-cycle alignment MUST pre-aggregate externally and present the result as a custom indicator; the aggregation framework does not support non-epoch anchors.

Sub-minute (100ms, 1s, 5s) via Snapshot Subsystem.

### 6.2 Aggregation Rules (NORMATIVE)

Following the existing `aggregate_to_4h()` pattern:

1. **No-lookahead rule:** A higher-timeframe bar with close timestamp T is first visible at the evaluation of the 1m bar whose close timestamp equals T. The HTF bar becomes available at the same moment its final constituent 1m bar closes. It is NOT visible during any earlier 1m bar. Example: A 1D bar closing at 2026-02-12 00:00:00 UTC becomes visible when the 1m bar [2026-02-11 23:59:00, 2026-02-12 00:00:00) closes. It is not visible at 23:58, 23:57, etc.
2. **Complete bars only:** Only fully closed HTF bars are included. Partial bars are never exposed.
3. **OHLC construction:** Open = first 1m open, High = max of all 1m highs, Low = min of all 1m lows, Close = last 1m close.
4. **Volume:** Sum of all constituent 1m volumes.
5. **Gap propagation:** If any constituent 1m bar has a gap flag, the HTF bar is flagged as having a gap. Trading is blocked on gap-flagged bars. **SnapshotBar usable mapping (NORMATIVE):** For strategies using `bar_provider: "SNAPSHOT_BARS"`, if a SnapshotBar has `usable=false` (due to staleness, partial data, or gap_detected), the corresponding 1m bar is treated as having `gap_flag=true` for purposes of HTF aggregation gap propagation. This ensures that snapshot-layer data quality issues are surfaced in the bar aggregation layer. **Gap tolerance policy:** Default (conservative): any constituent gap flags the HTF bar. This is the mandatory policy for SHADOW and LIVE modes. For RESEARCH mode only, the system owner MAY configure a gap tolerance threshold per timeframe (e.g. allow up to 1% gap duration for 3d bars = 43 minutes). Gap tolerance for SHADOW and LIVE MUST be zero. Relaxation requires system owner approval and contract amendment.

**Gap tolerance in evidence chain (NORMATIVE):** The gap tolerance configuration MUST be included in the run manifest and affects the run manifest hash. Replay across runs with different gap tolerance settings is prohibited — the tolerance is part of the run's identity. A RESEARCH run with non-zero gap tolerance and a SHADOW run with zero tolerance are distinct runs even if strategy config is identical.
6. **Timestamp monotonicity:** All bar timestamps in a replay sequence MUST be strictly increasing. Duplicate timestamps are invalid and MUST cause replay abort with a diagnostic error. This applies to both 1m base bars and all aggregated HTF bars.
7. **Decision timestamp monotonicity (NORMATIVE):** Decision cycle timestamps emitted by the evaluation scheduler MUST be strictly increasing across the lifetime of a run. Any backward timestamp event (from clock drift correction, snapshot stream restart, or any other cause) MUST trigger HALT. This is distinct from bar timestamp monotonicity (rule 6) — it covers the scheduler's own output sequence. **Restart floor (NORMATIVE):** On restart, the decision scheduler MUST load the last persisted decision timestamp and use it as a floor. If sampler realignment or clock correction would produce a timestamp ≤ the floor, the scheduler MUST delay until the next strictly greater boundary. This prevents drift resets from violating monotonicity.

### 6.5 Mandatory Aggregation Tests (NORMATIVE)

The following test cases MUST be implemented and pass before the strategy framework engine is considered correct:

1. **Coalesced cycle test:** Create a timestamp that simultaneously closes a 5m, 1h, and 1d bar. Verify that exactly ONE evaluation cycle executes, all three timeframe indicators update before signal evaluation, exit paths evaluate before entry paths, and signal precedence rules apply correctly.
2. **HTF visibility test:** At a 1D close boundary, verify that the new 1D bar is visible in the same evaluation cycle as the final 1m bar, and is NOT visible one 1m bar earlier.
3. **Gap propagation test:** Insert a single 1m gap into a 3d bar's constituent data. Verify the 3d bar is flagged (zero-tolerance mode) and trading is blocked.
4. **Cadence boundary conformance test:** For each supported cadence (5m, 15m, 30m, 1h, 4h, 12h, 1d, 3d), verify that the scheduler detects boundaries at exact epoch-aligned nanosecond timestamps and that path evaluation fires only at correct boundaries when driven by 1-second ticks.
5. **Full determinism test:** Run the same snapshot series through the engine twice with identical config. Verify decision sequences are bit-identical.

### 6.3 Pre-computation

For backtesting: pre-compute all HTF aggregations for O(log n) lookup (following `PrecomputedHTF` pattern).

### 6.4 Effective Warmup (NORMATIVE)

#### 6.4.1 Per-Output Warmup (v1.8.0)

Each indicator in the Phase 4A/4B registry MUST implement a per-output warmup API:

```python
def get_warmup_bars_for_output(output_name: str, params: dict) -> int:
    """Return minimum warmup bars needed for this specific output with these params."""
```

**Examples:**
- MACD with `slow=26`: `get_warmup_bars_for_output("macd_line", {"slow": 26}) → 26`
- MACD with `slow=26`: `get_warmup_bars_for_output("slope_sign", {"slow": 26}) → 27` (slope is derivative of macd_line)
- Bollinger with `period=20`: `get_warmup_bars_for_output("upper", {"period": 20}) → 20`

**Instance warmup computation (v1.8.0):**
```
instance.warmup_bars = max(
    indicator.get_warmup_bars_for_output(output, instance.parameters)
    for output in instance.outputs_used
)
```

An instance's warmup is determined by the most demanding output it actually uses, not the most demanding output the indicator can produce.

**Backward compatibility:** If an indicator does not implement per-output warmup, the engine falls back to the single `warmup_bars` value for all outputs.

**Validation rule (NORMATIVE):** If a config explicitly sets `warmup_bars` for an instance, it MUST be `>=` the computed per-output maximum. If a config specifies `warmup_bars: 20` but the `outputs_used` require 27, validation MUST fail: `"Instance '{label}' warmup_bars (20) is less than required (27) for output 'slope_sign'."` If `warmup_bars` is not explicitly set, the engine computes it from per-output requirements.

#### 6.4.2 Instance and Strategy Warmup Conversion

For bar-based instances (data_source: BAR):
```
warmup_seconds = instance.warmup_bars × instance.timeframe_seconds
```

For snapshot-based instances (data_source: SNAPSHOT):
```
warmup_seconds = instance.warmup_bars × instance.cadence_seconds
```

The strategy's effective warmup is the maximum across ALL instances, converted to seconds:
```
effective_warmup_seconds = max(
  instance.warmup_seconds for instance in all_instances
)
```

No trading until ALL instances have completed warmup. This is conservative by design — if exit indicators aren't warmed up, the strategy cannot exit safely, so entering would be reckless.

**Warmup and open positions:** The warmup blocking rule prevents new entries and flips. It does NOT prevent exits. Risk overrides (§5.1 step 2), exchange-side stop losses, and warmup restart policy (§3.10) all operate independently of indicator warmup. An open position is never unmonitored during warmup — it is either flattened (EMERGENCY_FLATTEN) or protected by hard stops (HARD_STOPS policy).

**Warmup signal suppression (NORMATIVE):** During warmup, entry and flip signals are disabled unconditionally. Only exit signals and emergency stops are permitted. This applies regardless of whether individual indicator instances have completed their own warmup — the strategy-wide warmup gate must clear before any entry or flip is allowed. **Risk override precedence (NORMATIVE):** Risk overrides from the Snapshot Subsystem (§5.1 step 2) execute BEFORE warmup entry suppression. If staleness policy demands FORCE_FLAT during warmup, the exit executes. Warmup suppression does not override risk overrides — it only suppresses entries and flips.

### 6.6 Scheduler Cadence Boundary Detection (NORMATIVE)

In fixed_interval snapshot mode, the evaluation scheduler receives a decision snapshot at every `decision_interval_ms` tick. For each tick, the scheduler determines which path cadences have a bar closing at that timestamp:

- A cadence boundary is detected when `tick_timestamp_ns % (cadence_seconds × 1_000_000_000) == 0`, with cadence anchored to Unix epoch.
- Entry paths whose cadence boundary is detected are eligible for evaluation at this tick.
- Exit paths whose cadence boundary is detected are eligible for evaluation at this tick.
- If no path's cadence boundary is detected, only risk overrides and gate re-evaluation (short-circuited if unchanged) execute.

This ensures that a 1D exit path fires exactly at day boundaries even when the scheduler is driven by 1-second ticks.

---

## 7. Strategy Artifact Format

### 7.1 Serialisation

Single JSON file containing all three layers. Config hash = SHA-256 of canonical JSON (sorted keys, no whitespace, UTF-8).

**Schema strictness (NORMATIVE, v1.8.0):** The engine loader MUST reject any strategy config JSON that contains fields not defined in the Framework schema for the declared engine_version, **at any depth in the JSON hierarchy**. Schema validation is recursive — unknown fields inside nested objects (exit_rules, execution_params, gate_rules, condition objects, etc.) are rejected identically to unknown top-level fields. Unknown fields are not silently ignored — they are schema violations. This ensures that resolved artifacts produced by the composition compiler are pure Framework schema with no composition-layer metadata leaking through. The schema is closed. Adding new fields requires a Framework version bump and schema amendment.

**Config immutability (NORMATIVE):** Indicator parameters, execution params, signal rules, and all other strategy config fields are immutable for the duration of a run. No hot-reload, dynamic reconfiguration, or runtime mutation of any config field is permitted. Any mutation during execution is a contract violation and MUST trigger HALT. A new run with a new config hash is required for any configuration change.

**Decimal Serialisation for Config Hash (NORMATIVE):** All numeric fields in the strategy config JSON MUST be serialised as JSON strings (not JSON numbers) using the following canonical format:
- Decimal values: String with normalised representation, no trailing zeros, no scientific notation. Example: `"0.5"` not `"0.50"` or `0.5` (JSON number).
- Integer values: JSON number type. Example: `20` not `"20"`.

**Serialisation function separation (NORMATIVE):** Config decimal canonicalisation (no trailing zeros, string type) is distinct from snapshot decimal canonicalisation (fixed 8 decimal places, per LOW_LATENCY_SNAPSHOT_SUBSYSTEM §8.2). Implementations MUST use separate serialiser functions for config hashing vs snapshot hashing. Sharing a single serialiser risks accidental format contamination.

**Array order preservation (NORMATIVE):** JSON parsing MUST preserve array order exactly as defined in the source file. Arrays (entry_rules, exit_rules, gate_rules, conditions, condition_groups) MUST NOT be transformed to sets, sorted, or reordered during parsing, serialisation, or any intermediate processing. Array order is semantically binding per §2.2.4.

This prevents floating-point representation divergence across JSON serialisers.

**Engine version in config hash (NORMATIVE):** The canonical JSON used for config hash computation MUST include a top-level field `"engine_version": "<semantic_version>"` (e.g. `"1.3.0"`). This ensures that config hashes are scoped to a specific engine implementation. If the engine version changes (even for bug fixes that change evaluation behavior), config hashes change, preventing silent replay divergence.

**Version bump invariant (NORMATIVE):** Any change to signal evaluation semantics (operator definitions, precedence rules, normalisation, comparison behavior) MUST increment the engine version. Patch-level changes that affect evaluation output require at minimum a patch version bump. This is enforced by the replay determinism test: if replay under the new engine produces different decisions from the same config and data, the engine version MUST differ.

**Run manifest completeness (NORMATIVE):** The run manifest (evidence chain) MUST embed the snapshot_config hash alongside the strategy config hash. Two runs with identical strategy config but different snapshot thresholds (deviation, spread, divergence) are distinct runs and MUST produce distinct run manifest hashes. The run manifest MUST also include the snapshot subsystem version identifier, so that snapshot engine upgrades are traceable in the evidence chain.

### 7.2 Archetype Tags

| Tag | Description |
|-----|-------------|
| `trend_following` | Directional, persistent moves |
| `mean_reversion` | Counter-trend, oscillation harvest |
| `breakout` | Compression → explosion |
| `structure_fade` | Counter-trend at SR levels |
| `momentum_exhaustion` | Snap-back from overextension |
| `carry` | Funding/yield collection |
| `macro_shift` | Correlation regime changes |
| `vol_transition` | Volatility regime changes |
| `drawdown_recovery` | Post-crash recovery |
| `squeeze` | Liquidation cascades |

Tags are metadata for fleet management. They do not affect execution.

---

## 8. Promotion and Lifecycle

```
UNTESTED → TRIAGE_PASSED → BASELINE_PLUS_PASSED → SHADOW_VALIDATED → LIVE_APPROVED → RETIRED
```

Each transition produces a promotion artifact with config hash, test results, timestamp, and approver.

**Promotion integrity (NORMATIVE):** Promotion from RESEARCH to SHADOW requires re-validation under SHADOW-mode configuration (zero gap tolerance, production snapshot thresholds). A strategy that passes triage under relaxed RESEARCH settings MUST be re-run under SHADOW settings before promotion. The promotion artifact MUST reference the SHADOW-mode test results, not the RESEARCH-mode results.

**Open decision gate (NORMATIVE):** No strategy may be promoted beyond TRIAGE_PASSED while any Open Design Decision (§10) relevant to that strategy remains unresolved. Additionally, the engine MUST reject SHADOW and LIVE mode runs for any strategy that references an unresolved Open Design Decision — not just block promotion, but block execution in those modes.

---

## 9. Delegation Boundaries

### 9.1 Authorized for Implementers

1. Strategy framework engine (parser, DSL evaluator, scheduler)
2. Multi-timeframe aggregation (5m through 3d, following Phase 3 patterns)
3. Signal DSL core operators (with normative definitions per §4.1)
4. Execution primitives: market entry/exit, flip (via OMS), SL (exchange-side), trailing stop, time limit
5. Strategy artifact serialisation and config hashing
6. Backtest and triage integration

### 9.2 Requires System Owner Approval

1. Specific strategy definitions
2. Gate-exit policies for SHADOW/LIVE
3. Triage pass/fail thresholds
4. New execution primitives (limit orders, scale-in) — LIMIT entry type is a known extension requiring OMS contract amendment (partial fill semantics, cancel-replace, order lifecycle) and Strategy Framework contract amendment before implementation
5. New DSL operators (including promotion of extension operators to core)
6. Aggregation alignment rule changes
7. LIVE deployment

### 9.3 Handoff Framing

> "Implement the Strategy Framework engine per STRATEGY_FRAMEWORK_CONTRACT.
> Input: Strategy config JSON (Section 7 schema)
> Output: Signal evaluation at each decision point
> Constraints: Deterministic, integer arithmetic for decision-affecting computation, compatible with Phase 3 backtest runner and Phase 5 triage filter.
> Verification: System owner provides 2 reference strategies (MACD confluence, DBMR band) with expected behavior."

---

## 10. Open Design Decisions

1. **DBMR gate-exit policy**: FORCE_FLAT, HOLD_CURRENT, or HANDOFF? (Note: §5.1.2 gate-flip interaction means FORCE_FLAT will decompose flips to exit-only at gate boundaries)
2. **DBMR warmup restart policy**: EMERGENCY_FLATTEN or HARD_STOPS (and what hard stop %)?
3. **Scale-in OMS semantics**: Multi-entry for single trade thesis?
4. **Fleet archetype awareness**: Warn when multiple pods run same archetype?
5. **MACD confluence leverage**: What leverage for the trend-following strategy? (affects funding model)
6. **Snapshot subsystem thresholds**: Confirm defaults — deviation 5%, spread 1%, divergence 10%, or adjust per strategy?

---

## 11. Cross-Contract Action Items

The following items were identified during adversarial review but require amendments to the LOW_LATENCY_SNAPSHOT_SUBSYSTEM contract, not this document. They are recorded here for tracking:

| Item | Target Contract | Issue |
|------|----------------|-------|
| Explicit ROUND_HALF_EVEN in serialize_decimal() | Snapshot §8.2 | D1 (round 7) |
| Deprecate *_pct config fields, replace with *_bps (integer) | Snapshot §9 | D2 (round 7) |
| Integer-only slippage gating (bps_x100) | Snapshot §12.4 | D3 (round 7) |
| Pin risk override action thresholds as normative integer bps | Snapshot §5.2 | D7 (round 7) |
| Snapshot config serialisation rules (string Decimals, canonical hash) | Snapshot §8 | D12 (round 7) |
| Sampler restart floor clause (clamp start_ts above last emitted) | Snapshot §4.6.1 | D8 (round 7) |

These are recommended but NOT blocking for Strategy Framework freeze. The framework's own invariants (Decimal context, timestamp floor, rounding mode) provide sufficient defensive boundaries until snapshot amendments are applied.

---

## Amendment Log

### v1.0.0 → v1.1.0 (2026-02-12)

Amendments from adversarial review round 1 (ChatGPT red team, 18 issues identified, 17 accepted):

| Issue | Amendment | Section |
|-------|-----------|---------|
| 1 | Added signal precedence rules (exit before entry) | §5.1.1 (new) |
| 2 | Added flip signal generation rules | §5.1.2 (new) |
| 3 | Added gate policy conflict resolution (FORCE_FLAT wins) | §5.2.1 (new) |
| 4+5 | Rate limiter and slippage budget exempt all exits | §5.1 steps 3-4, §3.8, §3.9 |
| 6 | Added operator state persistence requirement | §4.5 (new) |
| 7 | Added v1 scope statement for operators | §4.1 |
| 8 | Replaced no-lookahead rule with exact timestamp semantics | §6.2 rule 1 |
| 9 | Added snapshot warmup calculation | §6.4 |
| 10 | Specified Decimal serialisation for config hash | §7.1 |
| 11 | Removed LIMIT from v1 entry_type enum | §2.3, §9.2 |
| 12+18 | Declared list ordering semantically binding | §2.2.4 (new) |
| 13 | Defined CONSIDER_EXIT mapping | §5.1 |
| 14 | Defined decision price semantics | §4.1.1 (new) |
| 15 | Added stale-HTF-in-gate clarification | §4.3 rule 2 |
| 16 | Specified Unix epoch anchor for all timeframes | §6.1 |
| 17 | REJECTED — conservative warmup is safety feature | No change |

### v1.1.0 → v1.2.0 (2026-02-12)

Amendments from adversarial review round 2 (ChatGPT red team, 15 issues identified, 11 accepted):

| Issue | Amendment | Section |
|-------|-----------|---------|
| R2-1 | Added Step 0: atomic indicator update before evaluation | §5.1 step 0 (new) |
| R2-2 | REJECTED — crosses_above flatline already handled by formal definition | No change |
| R2-3 | Added gate-flip interaction: gate wins, flip decomposes to exit-only | §5.1.2 |
| R2-4 | Operator state coupled to indicator state; reset together on restart | §4.5 (rewrite) |
| R2-5 | Gap tolerance configurable in RESEARCH, zero in SHADOW/LIVE | §6.2 rule 5 |
| R2-6 | REJECTED — exit type priority covered by §2.2.4 list ordering | No change |
| R2-7 | Price comparisons normalised to 8 decimal places | §4.1.1 |
| R2-8 | Multi-path trigger diagnostic logging | §5.1.1 |
| R2-9 | Warmup does not block exits or restart safety mechanisms | §6.4 |
| R2-10 | Funding model scoped to backtest only | §3.7 |
| R2-11 | Catastrophic slippage cooldown on entries after extreme exit slippage | §3.9 |
| R2-12 | Engine version included in config hash | §7.1 |
| R2-13 | REJECTED — gate conflict temporal ordering already covered by §4.3 | No change |
| R2-14 | Covered by R2-1 (atomic indicator snapshot) | §5.1 step 0 |
| R2-15 | Timestamp monotonicity invariant for replay | §6.2 rule 6 (new) |

### v1.2.0 → v1.3.0 (2026-02-12)

Amendments from adversarial review round 3 (ChatGPT red team, 15 issues identified, 10 accepted):

| Issue | Amendment | Section |
|-------|-----------|---------|
| R3-1 | Unified evaluation cycle: all cadences at same timestamp coalesced | §5.1 step 0 |
| R3-2 | "Previous" for crosses_* defined as indicator's own TF bar | §4.1 |
| R3-3 | REJECTED — post-warmup blind spot already documented as intentional | No change |
| R3-4 | Flip detection moved before pure-exit in precedence | §5.1.1 |
| R3-5 | REJECTED — rate limiter flip interaction already correct per §5.1.2 | No change |
| R3-6 | REJECTED — stale HTF in gate is deterministic and intended per §4.3 | No change |
| R3-7 | REJECTED — gap tolerance already addressed in R2-5 | No change |
| R3-8 | Indicator outputs normalised to 8 decimals for DSL comparison | §4.1.1 |
| R3-9 | Version bump invariant: any semantic change requires version increment | §7.1 |
| R3-10 | TIME_LIMIT qualifies as exit for flip detection | §5.1.2 |
| R3-11 | REJECTED — pipeline ordering already defines CONSIDER_EXIT vs gate | No change |
| R3-12 | Hybrid BAR/SNAPSHOT persistence validation rule | §4.3 rule 5 |
| R3-13 | Optional strict mode for overlapping entry path detection | §5.1.1 |
| R3-14 | HANDOFF explicitly reserved/disallowed in v1 | §2.2.3, §5.2 |
| R3-15 | 3d epoch drift: calendar alignment explicitly unsupported | §6.1 |

### v1.3.0 → v1.4.0 (2026-02-12)

Amendments from adversarial review round 4 (ChatGPT red team, 18 issues identified, 12 accepted):

| Issue | Amendment | Section |
|-------|-----------|---------|
| R4-1 | Exit-before-entry explicit within coalesced evaluation cycle | §5.1 step 0 |
| R4-2 | REJECTED — exit type priority covered by §2.2.4 list ordering | No change |
| R4-3 | Rounding mode specified: ROUND_HALF_EVEN for 8-decimal normalisation | §4.1.1 |
| R4-4 | Price vs non-price indicator normalisation distinguished | §4.1.1 |
| R4-5 | Gate condition short-circuit when HTF bar unchanged | §4.3 rule 2 |
| R4-6 | TIME_LIMIT flip participation controlled by config flag (default false) | §5.1.2, §2.3 |
| R4-7 | Warmup unconditionally blocks entries and flips | §6.4 |
| R4-8 | REJECTED — overlapping entry detection covered by R3-13 | No change |
| R4-9 | REJECTED — pipeline ordering already resolves risk vs gate precedence | No change |
| R4-10 | Hybrid BAR/SNAPSHOT synchronisation: wait for both pipelines | §4.3 rule 6 |
| R4-11 | VOL_TARGETED division-by-zero guard | §3.1 |
| R4-12 | REJECTED — covered by R3-9 version bump invariant | No change |
| R4-13 | Promotion requires re-validation under SHADOW settings | §8 |
| R4-14 | REJECTED — funding cycle alignment covered by R3-15 | No change |
| R4-15 | Run manifest embeds snapshot_config hash | §7.1 |
| R4-16 | TIME_LIMIT reference cadence field added | §2.3 |
| R4-17 | Scale-in v1 rejection guard | §3.6 |
| R4-18 | Decision timestamp monotonicity invariant | §6.2 rule 7 |

### v1.4.0 → v1.5.0 (2026-02-12)

Amendments from adversarial review round 5 (ChatGPT red team, 18 issues identified, 14 accepted):

| Issue | Amendment | Section |
|-------|-----------|---------|
| R5-1 | CONSIDER_EXIT excluded from flip detection | §5.1 |
| R5-2 | Exchange-side SL interaction and exit reason hierarchy | §3.2 |
| R5-3 | Multi-exit trigger diagnostic logging | §5.1.1 |
| R5-4 | REJECTED — gate indicator TF is strategy design, not contract | No change |
| R5-5 | Hybrid synchronisation timeout (5000ms → HALT) | §4.3 rule 6 |
| R5-6 | min_vol_threshold for VOL_TARGETED sizing | §3.1 |
| R5-7 | time_limit_reference_cadence validation | §2.3 |
| R5-8 | REJECTED — warmup flip suppression already in R4-7 | No change |
| R5-9 | Config vs snapshot serialiser separation | §7.1 |
| R5-10 | REJECTED — gate atomicity covered by step 0 | No change |
| R5-11 | Indicator instance limits (soft 64, hard 128) | §2.1 rule 8 |
| R5-12 | SHADOW mode uses exchange funding, not model | §3.7 |
| R5-13 | REJECTED — ROUND_HALF_EVEN is IEEE standard | No change |
| R5-14 | strict_exit_paths option added | §5.1.1 |
| R5-15 | Single-threaded evaluation per symbol | §5.1 step 0 |
| R5-16 | flip_decomposed diagnostic event | §5.1.2 |
| R5-17 | Snapshot subsystem version in run manifest | §7.1 |
| R5-18 | Mandatory coalesced cycle test case | §6.5 (new) |

### v1.5.0 → v1.6.0 (2026-02-12)

Amendments from adversarial review round 6 (ChatGPT red team — double pass: 18 main + 15 determinism = 33 issues, 25 accepted):

**Main pass:**
| Issue | Amendment | Section |
|-------|-----------|---------|
| R6-1 | No same-direction re-entry after exit in same cycle | §5.1 step 0 |
| R6-2 | REJECTED — exit type priority is logging-only, list order binding | No change |
| R6-3 | Exchange SL suppresses flip in same cycle | §3.2 |
| R6-4 | REJECTED — gate short-circuit already in R4-5, stateless in v1 | No change |
| R6-5 | Pipeline desync is fatal, requires operator intervention | §4.3 rule 6 |
| R6-6 | REJECTED — already covered by R5-6 min_vol_threshold | No change |
| R6-7 | TIME_LIMIT counter semantics: increments on reference cadence close | §2.3 |
| R6-8 | Contradictory gate policy validation warning | §5.2.1 |
| R6-9 | Operator state explicit initialisation on restart | §4.5 |
| R6-10 | Decision timestamp floor from last persisted (merged with D4) | §6.2 rule 7 |
| R6-11 | Path count limits (soft 32, hard 64) | §2.1 rule 9 |
| R6-12 | None indicator output → condition false | §4.1.1 |
| R6-13 | Condition group + standalone evaluation order | §2.2.1 |
| R6-14 | REJECTED — scheduler clock source covered by hybrid sync | No change |
| R6-15 | JSON array order preservation | §7.1 |
| R6-16 | Minimum exit requirement (exit path or stop_loss) | §2.2.2 |
| R6-17 | Per-symbol isolation (merged with D14) | §5.1 step 0 |
| R6-18 | Open decisions must be resolved before promotion | §8 |

**Determinism pass:**
| Issue | Amendment | Section |
|-------|-----------|---------|
| D1 | Cross-contract rounding mode consistency | §4.1.1 |
| D2 | Risk overrides before warmup suppression | §6.4 |
| D3 | CONSIDER_EXIT locked to snapshot v1.3.0 semantics | §5.1 |
| D4 | Merged with R6-10 (decision timestamp floor) | §6.2 rule 7 |
| D5 | Gap tolerance in evidence chain | §6.2 rule 5 |
| D6 | Flip stall timeout (30s → HALT) | §5.1 step 1 |
| D7 | REJECTED — TIME_LIMIT counts cadence bars, gap handling separate | No change |
| D8 | Snapshot-derived bars need full persistence | §4.3 rule 5 |
| D9 | Fixed Decimal context at engine startup | §4.1.1 |
| D10 | REJECTED — hybrid sync rule covers this | No change |
| D11 | REJECTED — snapshot version already in run manifest | No change |
| D12 | REJECTED — crosses on completed bars, gaps produce None → false | No change |
| D13 | Slippage arithmetic quantisation | §3.9 |
| D14 | Merged with R6-17 (per-symbol isolation) | §5.1 step 0 |
| D15 | Operator set versioning | §4.1 |

### v1.6.0 → v1.7.0 (2026-02-12)

Amendments from adversarial review round 7 (ChatGPT red team — double pass: 14 main + 14 determinism = 28 issues, 20 accepted, 2 rejected, 6 cross-contract):

**Main pass:**
| Issue | Amendment | Section |
|-------|-----------|---------|
| R7-1 | Flip detection before pure-exit in coalesced cycle (tightened) | §5.1.1 |
| R7-2 | Gate state transitions logged only on indicator change | §4.3 rule 2 |
| R7-3 | TIME_LIMIT counter does not increment on unusable bars | §2.3 |
| R7-4 | Pipeline desync HALT is symbol-scoped | §4.3 rule 6 |
| R7-5 | Exchange SL → FLAT → next cycle fresh entry, not flip | §3.2 |
| R7-6 | Decimal context recorded in run manifest | §4.1.1 |
| R7-7 | Config immutability during run | §7.1 |
| R7-8 | REJECTED — exit reason hierarchy already defined for logging | No change |
| R7-9 | Condition count limits per path (soft 32, hard 64) | §2.1 rule 10 |
| R7-10 | REJECTED — gate warning sufficient, error too aggressive | No change |
| R7-11 | Formal determinism invariant | §5.1 |
| R7-12 | SHADOW/LIVE blocked for unresolved open decisions | §8 |
| R7-13 | Leverage validation (> 0, ≤ exchange max) | §2.3 |
| R7-14 | Operator compatibility table hard-coded in engine | §4.1 |

**Determinism pass:**
| Issue | Amendment | Section |
|-------|-----------|---------|
| D1 | CROSS-CONTRACT: Snapshot rounding mode | §11 (noted) |
| D2 | CROSS-CONTRACT: Snapshot pct→bps fields | §11 (noted) |
| D3 | CROSS-CONTRACT: Integer-only slippage | §11 (noted) |
| D4 | Canonical timebase = Unix nanoseconds UTC | §5.1 step 0 |
| D5 | Scheduler cadence boundary detection rule | §6.6 (new) |
| D6 | None price → conditions false + diagnostic | §4.1.1 |
| D7 | CROSS-CONTRACT: Snapshot risk override thresholds | §11 (noted) |
| D8 | CROSS-CONTRACT: Sampler restart floor | §11 (noted) |
| D9 | bar_provider config field (EXCHANGE_CANDLES vs SNAPSHOT_BARS) | §2.1 |
| D10 | SnapshotBar.usable=false → gap flag for aggregation | §6.2 rule 5 |
| D11 | HOLD_PENDING_FLIP allows reduce-only emergency exits | §5.1 step 1 |
| D12 | CROSS-CONTRACT: Snapshot config serialisation | §11 (noted) |
| D13 | Covered by D6 (None price handling) | — |
| D14 | Mandatory scheduler determinism tests | §6.5 |

**Cumulative totals across 7 rounds:**
- Issues reviewed: 145
- Framework amendments accepted: 109
- Rejected: 30
- Cross-contract (noted, not blocking): 6

### v1.7.0 → v1.8.0 (2026-02-12)

Amendments required by Strategy Composition Contract v1.5.2. Six deltas applied:

| Delta | Amendment | Section |
|-------|-----------|---------|
| D1 | `is_present` and `is_absent` unary condition operators (v1.8.0) | §4.1 |
| D2 | `MTM_DRAWDOWN_EXIT` exit type with gross MTM P&L drawdown tracking | §2.2.2, §3.11 (new) |
| D3 | Per-output warmup via `get_warmup_bars_for_output()` API | §6.4 (restructured as §6.4.1, §6.4.2) |
| D4 | HANDOFF gate exit policy unlocked from reserved; full suppression semantics defined | §2.2.3, §5.2 |
| D5 | Cross-indicator condition references (`ref_indicator`, `ref_output`) with outputs_used enforcement | §4.1 |
| D6 | Schema strictness: recursive unknown-field rejection normatively stated | §7.1 |

Additional changes:
- Operator compatibility table added (§4.1)
- `RELATIVE >` extension operator superseded by cross-indicator references
- v1/v1.8 scope statement updated

**Cumulative totals across 7 rounds + composition alignment:**
- Issues reviewed: 145 (adversarial) + 6 (composition deltas)
- Framework amendments accepted: 115
- Rejected: 30
- Cross-contract (noted, not blocking): 6

---

## END OF CONTRACT v1.8.0

**Status**: DRAFT — Awaiting system owner approval
**Next**: System owner review → resolve 6 open decisions → approve → freeze → hand to Ralph
