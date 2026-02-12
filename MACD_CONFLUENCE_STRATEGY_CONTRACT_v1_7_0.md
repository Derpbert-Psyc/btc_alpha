# MACD CONFLUENCE STRATEGY CONTRACT

**Version**: 1.7.0
**Status**: DRAFT — Awaiting system owner freeze decision
**Date**: 2026-02-12
**Authority Class**: Strategy-level contract (instantiates STRATEGY_FRAMEWORK_CONTRACT v1.7.0)
**Depends on**: STRATEGY_FRAMEWORK_CONTRACT v1.7.0, PHASE4A_INDICATOR_CONTRACT, LOW_LATENCY_SNAPSHOT_SUBSYSTEM v1.3.0

---

## 1. Strategy Identity

**Archetype**: Multi-Timeframe Trend Confluence (Archetype #1 from framework §1.1)
**Trading thesis**: Enter when short-term momentum inflects in the direction of multi-timeframe trend alignment. Exit when the exit-trigger timeframe (default 1D, tunable) shows momentum reversal. The exit timeframe is deliberately slower than the entry timeframe (asymmetric: fast in, slow out).
**Instrument**: BTCUSDT perpetual (primary), BTCUSDT spot (variant)
**Direction**: Bidirectional (LONG and SHORT)

**Strategy ID**: `macd_confluence_v1`
**min_operator_version**: `"1.0"`
**engine_version**: Per framework §7.1

---

## 2. Indicator Instances

All instances use the TradingView-style MACD indicator (Phase 4A registered):
- EMA with SMA seed
- MACD line = EMA(fast) − EMA(slow)
- Slope sign = sign(M_t − M_{t−1})

### 2.1 Default MACD Parameters (tunable per instance)

```
macd_parameters:
  fast_period: 12
  slow_period: 26
  signal_period: 9       # Signal line (not used in v1 entry/exit, reserved)
```

These are the TradingView defaults. Research may determine different parameters are optimal per timeframe. Each instance MAY use different parameters.

**Parameter specification (NORMATIVE):** Every indicator instance MUST fully specify all parameters (fast, slow, signal). There is no implicit inheritance from defaults. If an instance should use the defaults, the default values must be explicitly written in the config. This prevents ambiguity across loaders and ensures the config hash captures the actual parameters used.

### 2.2 Instance Registry

Instances are tagged with a **role** that determines how they are referenced in entry/exit rules. Entry and exit conditions reference roles (not individual labels), and the engine evaluates all instances with that role.

**Role definitions:**

| Role | Purpose | Min Instances | Max Instances |
|------|---------|---------------|---------------|
| `macro` | Trend direction on slow timeframes | 1 | 5 |
| `intra` | Trend direction on fast timeframes | 1 | 5 |
| `entry_trigger` | Inflection detection for entry | 1 (exactly) | 1 |
| `exit_trigger` | Reversal detection for exit | 1 (exactly) | 1 |
| `exit_fallback` | Secondary exit when exit_trigger is None | 0 | 1 |

**exit_fallback rules:** Required when exit_trigger timeframe has gap invalidation > 24h (see §2.3). When present, must have timeframe < exit_trigger timeframe. Evaluated only when exit_trigger returns None at decision time. If both are None, only risk exits remain active.

**Fallback evaluation semantics (NORMATIVE):** The exit_fallback instance is a transparent substitute for exit_trigger. When exit_trigger returns None (due to gap invalidation or data unavailability), the engine evaluates the SAME exit conditions (§4.1/§4.2) but uses exit_fallback's outputs in place of exit_trigger's. No separate fallback conditions are needed. During role expansion (§2.2), the exit conditions reference the exit_trigger label. At evaluation time, if exit_trigger output is None and exit_fallback exists and is not None, the engine substitutes exit_fallback's output values into the resolved exit conditions. This substitution is logged as diagnostic `exit_fallback_substitution=true`.

**Default instance set** (8 instances):

| Label | Role | Timeframe | Outputs Used | data_source | bar_provider |
|-------|------|-----------|-------------|-------------|--------------|
| `macro_3d` | macro | 3d | slope_sign | BAR | EXCHANGE_CANDLES |
| `macro_1d` | macro | 1d | slope_sign | BAR | EXCHANGE_CANDLES |
| `macro_12h` | macro | 12h | slope_sign | BAR | EXCHANGE_CANDLES |
| `intra_1h` | intra | 1h | slope_sign | BAR | EXCHANGE_CANDLES |
| `intra_30m` | intra | 30m | slope_sign | BAR | EXCHANGE_CANDLES |
| `intra_15m` | intra | 15m | slope_sign | BAR | EXCHANGE_CANDLES |
| `entry_5m` | entry_trigger | 5m | macd_line, slope_sign | BAR | EXCHANGE_CANDLES |
| `exit_1d` | exit_trigger | 1d | slope_sign | BAR | EXCHANGE_CANDLES |

**Tunability:** Research may change the instance set freely as long as role constraints (min/max counts, timeframe ordering per §2.3) are satisfied. Changing the instance set (adding, removing, or changing timeframes) changes the config hash and produces a distinct strategy identity.

**Role-based evaluation (NORMATIVE):** Entry condition groups reference roles, not labels. The condition `role:macro.slope_sign > 0` means: for ALL instances with role=macro, slope_sign must be > 0. If any macro instance's slope_sign is not > 0, the group fails. Similarly for `role:intra`. The `entry_trigger` and `exit_trigger` roles reference exactly one instance each — their conditions use the specific label.

This design allows research to test `{macro: [1d, 12h], intra: [1h, 15m], entry: 5m, exit: 12h}` without changing any condition logic — only the instance registry changes.

**Role-reference expansion (NORMATIVE, cross-contract):** The `role_condition` and `{"role": ...}` JSON constructs are strategy-level abstractions that do NOT exist in the framework v1.7.0 condition evaluator. The config loader MUST expand ALL role references into concrete per-instance conditions before passing them to the framework. This includes:

- `role_condition` objects in condition groups (`quantifier: "ALL"` -> one condition per matching instance)
- `{"role": "<role_name>", ...}` objects in standalone `conditions` arrays (exactly-one roles like entry_trigger/exit_trigger -> rewritten to `{"indicator": "<label>", ...}`)

Expansion rules:

1. For `role_condition` with `quantifier: "ALL"`: enumerate all instances with the matching role, **sorted by (timeframe_seconds ascending, label ascending)**. Generate one framework condition per instance: `{"indicator": "<label>", "output": "<o>", "operator": "<op>", "value": "<val>"}`.
2. For standalone `{"role": "<n>", ...}` conditions: resolve role to the single matching instance label and rewrite to `{"indicator": "<label>", ...}`. If the role matches multiple instances, this is a validation error (only `role_condition` with quantifier supports multi-instance roles).
3. After expansion, NO `role_condition` or `"role"` keys may remain in the resolved config. The framework evaluator MUST receive only standard `{"indicator": ..., "output": ..., "operator": ..., "value": ...}` condition objects.
4. **Ordering (NORMATIVE):** Expanded condition arrays MUST be sorted by `(indicator label ascending)` before inclusion in the resolved config. This ensures deterministic hashing regardless of original array order.
5. The pre-expansion form is stored as a diagnostic for human readability.

**Canonical timeframe enum (NORMATIVE):** The following are the only valid timeframe strings in strategy configs. All timeframe references MUST use this exact spelling. Normalization to canonical form MUST occur at config parse time. Non-canonical strings (e.g. `"12H"`, `"12hr"`, `"720m"`) are validation errors.

```
VALID_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "3d"]
```


All timeframes listed in §2.2 are DEFAULT values. Research may determine different combinations are optimal. The following constraint set applies (single authoritative rule set):

1. **Strict ordering**: If macro has ≥2 instances, max(macro_timeframes) > min(macro_timeframes). If intra has ≥2 instances, max(intra_timeframes) > min(intra_timeframes). Cross-group: min(macro_timeframes) ≥ max(intra_timeframes) > entry_timeframe.
2. **Exit position**: exit_timeframe ≥ max(intra_timeframes)
3. **Entry ceiling**: entry_timeframe ≤ min(intra_timeframes)
4. **Macro floor**: All macro timeframes ≥ 12h
5. **Entry floor**: entry_timeframe ≥ 1m

These five rules are the ONLY timeframe constraints. No redundant secondary rules. Research tooling MUST validate all five before accepting a timeframe combination. Any combination satisfying all five is valid.

**Exit TF availability constraint (NORMATIVE):** If the `exit_trigger` timeframe has gap invalidation duration > 24 hours under zero gap tolerance (e.g. 3d = up to 3 days), the strategy MUST define a `fallback_exit` with a concrete secondary indicator instance:

```
fallback_exit:
  indicator_label: str    # Label of secondary exit instance (must exist in registry with role exit_fallback)
  timeframe: str          # Must be < exit_trigger timeframe, e.g. "1d" if exit is "3d"
```

The fallback is a separate indicator instance in the registry with role `exit_fallback`. When the primary `exit_trigger` returns None at decision time, the engine evaluates the fallback instead. If both are None, only risk exits (stop loss, MTM drawdown exit) remain active. If no fallback is defined and exit_trigger gap invalidation exceeds 24h: validation warning in RESEARCH, validation error in SHADOW/LIVE.

### 2.4 Minimum/Maximum Instance Counts

- Macro group: minimum 1, maximum 5 instances
- Intra group: minimum 1, maximum 5 instances
- Entry trigger: exactly 1 instance
- Exit trigger: exactly 1 instance

---

## 3. Entry Rules

### 3.1 LONG Entry Path

**evaluation_cadence**: Same as entry trigger timeframe (default: 5m)
**direction**: LONG
**entry_type**: MARKET

**condition_groups** (role-based evaluation):

```
group: "macro_alignment"
  - role:macro.slope_sign > 0    # ALL macro instances must have positive slope

group: "intra_alignment"
  - role:intra.slope_sign > 0    # ALL intra instances must have positive slope
```

**conditions** (standalone, AND'd with groups):

```
  - [entry_trigger].slope_sign crosses_above 0
```

The 5m MACD slope_sign was ≤ 0 on the previous bar and is > 0 on the current bar (turn-up).

**Formal definition** (per mathematical spec §5):

```
turn_up(k) = (M_k − M_{k−1}) > 0 AND (M_{k−1} − M_{k−2}) ≤ 0
```

Where M is the MACD line value at bar k of the entry timeframe. The turn-up is detected via `slope_sign crosses_above 0`, NOT via a crossing on macd_line itself.

**Strategy-level assertion (NORMATIVE):** Turn-up and turn-down are defined on the `slope_sign` output, not on `macd_line`. The `macd_line` output is not referenced in any entry or exit condition in v1.

### 3.2 SHORT Entry Path

**evaluation_cadence**: Same as entry trigger timeframe (default: 5m)
**direction**: SHORT
**entry_type**: MARKET

**condition_groups** (role-based evaluation):

```
group: "macro_alignment"
  - role:macro.slope_sign < 0    # ALL macro instances must have negative slope

group: "intra_alignment"
  - role:intra.slope_sign < 0    # ALL intra instances must have negative slope
```

**conditions** (standalone):

```
  - [entry_trigger].slope_sign crosses_below 0
```

Turn-down: `(M_k − M_{k−1}) < 0 AND (M_{k−1} − M_{k−2}) ≥ 0`

### 3.3 Entry Path List Order

```
entry_rules:
  1. LONG entry path
  2. SHORT entry path
```

These are mutually exclusive by construction (macro alignment cannot be simultaneously all-positive and all-negative). If for any reason both evaluate true at the same timestamp (impossible with current conditions, but defensively), LONG has priority per list order.

### 3.4 Slope Zero Handling (NORMATIVE)

All alignment conditions use strict inequality (> 0 for LONG, < 0 for SHORT). If any timeframe's slope_sign is exactly 0, neither LONG nor SHORT entry fires. This is an intentional design choice — zero slope means no directional conviction, and the strategy should not enter.

**Consequence:** In low-volatility environments where MACD flattens, the strategy becomes inactive. This is correct behavior — the thesis requires directional alignment, and zero slope does not satisfy it. Research MAY investigate a thresholded variant (slope_sign compared against ±epsilon instead of zero) as a v2 enhancement, but v1 uses strict zero comparison.

**Inactivity reporting (NORMATIVE):** Research backtests MUST report: total time the strategy is inactive (no position and no entry conditions met) as a percentage of total backtest period, and the number/magnitude of trends missed during inactive periods (measured as max favorable excursion during inactive windows). If inactivity exceeds 60% of the backtest period or missed-trend magnitude exceeds 20% of total captured trend magnitude, research MUST flag this for review.

---

## 4. Exit Rules

### 4.1 LONG Signal Exit

**evaluation_cadence**: Same as exit trigger timeframe (default: 1d). MUST equal `indicator_instances[role=exit_trigger].timeframe`.
**exit_type**: SIGNAL
**applies_to**: LONG positions only

**applies_to type (NORMATIVE):** The `applies_to` field is ALWAYS an array of direction strings (e.g. `["LONG"]`, `["SHORT"]`, `["LONG", "SHORT"]`). String values are not permitted — validation MUST reject non-array applies_to.

**conditions**:

```
  - [exit_trigger].slope_sign < 0
```

The exit_trigger instance's MACD slope turns negative → exit long.

### 4.2 SHORT Signal Exit

**evaluation_cadence**: Same as exit trigger timeframe (default: 1d). MUST equal `indicator_instances[role=exit_trigger].timeframe`.
**exit_type**: SIGNAL
**applies_to**: SHORT positions only

**conditions**:

```
  - [exit_trigger].slope_sign > 0
```

The exit_trigger instance's MACD slope turns positive → exit short.

### 4.3 Stop Loss

**exit_type**: STOP_LOSS
**exchange_side**: true (MANDATORY for LIVE)

Stop loss is configured per direction using the framework's StopLossConfig schema (§3.2), expressed as integer basis points.

```
stop_loss:
  mode: PERCENT
  value_long_bps: int       # e.g. 500 = 5%. Converted to Decimal fraction (bps/10000) for exchange API.
  value_short_bps: int      # May differ from LONG (asymmetric BTC tail risk)
  exchange_side: true       # MANDATORY for LIVE per framework §3.2
```

**Unit semantics (NORMATIVE):** `value_long_bps` and `value_short_bps` are integer basis points. The engine converts to Decimal fraction (`bps / 10000`) when placing exchange-side orders. Example: `500` → `Decimal("0.05")` → 5% stop. The field type `type: BPS` in the strategy schema maps to `mode: PERCENT` in the framework's StopLossConfig. This mapping is deterministic and documented here to prevent 100x interpretation errors.

**StopLossConfig lowering (NORMATIVE):** At config load time, the strategy's stop loss bps fields MUST be lowered to framework-native StopLossConfig fields in the resolved config. The lowering target schema is:

```
# Framework StopLossConfig target (version-pinned to framework v1.7.0)
stop_loss_resolved:
  mode: "PERCENT"
  percent_long: Decimal    # = value_long_bps / 10000. E.g. 500 → Decimal("0.05")
  percent_short: Decimal   # = value_short_bps / 10000
  exchange_side: bool      # Copied verbatim from authoring form
```

**Cross-contract note:** If the framework StopLossConfig defines different field names than `percent_long`/`percent_short`, this target schema MUST be updated to match. The strategy contract pins to the framework version in its header. The lowering target is authoritative for implementations of THIS strategy contract version. Framework field name changes require a strategy contract version bump.

The strategy-authoring bps fields do not appear in the resolved config. This lowering is part of the normative lowering pipeline (§11).

**Stop loss validation (NORMATIVE):** `value_long_bps` and `value_short_bps` MUST be non-null integers in [100, 2000] for SHADOW/LIVE promotion. For RESEARCH mode with `warmup_restart_policy = "HARD_STOPS"`, they MUST also be non-null integers. For RESEARCH mode with `warmup_restart_policy = "FLATTEN"`, they MAY be null (stop loss disabled for signal-quality research). The engine MUST reject configs with null stop loss values for any mode that permits position opening without valid stops, with error code `stop_loss_undefined`.

### 4.4 MTM Drawdown Exit (OPTIONAL)

**exit_type**: MTM_DRAWDOWN_EXIT (new primitive — NOT the framework's TRAILING_STOP)
**evaluation_cadence**: 1m (MANDATORY — must match backtester behavior)

**Important:** This is NOT the framework's ATR-based TrailingStopConfig (§3.3). The framework's TRAILING_STOP uses ATR distance with tighten conditions. This strategy uses a fundamentally different exit primitive: percentage drawdown from MTM peak. This requires the framework to register `MTM_DRAWDOWN_EXIT` as a new exit type in a version bump (noted as cross-contract action item).

Per mathematical spec §9. Parameters are per-direction, integer bps:

```
mtm_drawdown_exit:
  enabled: bool                     # false by default, enable after research
  evaluation_cadence: "1m"          # NORMATIVE: always 1m regardless of entry/exit cadence
  drawdown_bps_long: int            # e.g. 1000 = 10% from peak MTM. Integer bps.
  drawdown_bps_short: int           # May differ from LONG
  mtm_includes_funding: true
  mtm_includes_fees: true
```

MTM is computed as: `IM + unrealized_pnl − entry_fee − cumulative_funding`

Peak tracking: `MTM_peak = max(MTM_peak_prev, MTM_current)`

Trigger: `MTM_current ≤ MTM_peak × (1 − drawdown_bps / 10000)`

**MTM drawdown cadence (NORMATIVE):** MUST evaluate on every 1m bar close while a position is open. If triggered at 1m bar close i, exit is scheduled at the next 1m open (i+1) and this scheduled exit preempts other evaluation logic at that bar.

**Precedence (NORMATIVE):** MTM_DRAWDOWN_EXIT is a risk overlay positioned between RISK_OVERRIDE and GATE_EXIT:

`EXCHANGE_STOP > RISK_OVERRIDE > MTM_DRAWDOWN_EXIT > GATE_EXIT > TRAILING_STOP > TIME_LIMIT > SIGNAL`

It bypasses rate limiting and slippage budget. This differs from the framework's TRAILING_STOP which sits below GATE_EXIT. The distinction is intentional: MTM drawdown is a capital-protection mechanism, not a signal-following trailing stop.

**1m data requirement (NORMATIVE):** If enabled, the data layer MUST provide usable 1m bars throughout the position's lifetime. If 1m bars are missing or unusable for more than 5 consecutive bars, the engine MUST trigger FORCE_FLAT with diagnostic `mtm_drawdown_data_unavailable`.

**Cross-contract action item:** Framework v1.7.0 does not define MTM_DRAWDOWN_EXIT. A framework version bump is required to add this exit type with its precedence slot. Until then, implementations must treat this as a strategy-specific risk overlay operating outside the standard exit pipeline.

### 4.5 Funding Exit Overlay (OPTIONAL, perps only)

**exit_type**: SIGNAL (funding-triggered)
**evaluation_cadence**: Funding event timestamps only (every `funding_model.interval_hours`)
**applies_to**: Configurable per direction. Default: both LONG and SHORT.

Per mathematical spec §8:

```
funding_exit:
  enabled: bool                     # false by default
  threshold_bps_of_im: int          # e.g. 500 = 5% of initial margin. Integer bps.
  require_underwater: bool          # if true, only exit when unrealized_pnl ≤ cum_funding
  applies_to: ["LONG", "SHORT"]    # Configurable per direction
```

**Direction-aware funding semantics (NORMATIVE):** In live trading, funding can be negative (credits). The overlay trigger uses **adverse funding only** — defined as `funding_drag = max(0, cum_net_funding_cost)` where positive means the position is paying funding. Credits (negative cost) reduce funding_drag toward zero but do not make it negative. The trigger fires when `funding_drag ≥ (threshold_bps_of_im / 10000) × IM` AND (if require_underwater=true) `unrealized_pnl ≤ funding_drag`. This prevents exits on positions that are receiving favorable funding. For conservative backtest (credit_allowed=false), all funding is positive and funding_drag equals cumulative funding paid.

**Explicit opt-in for abs-mode:** If a research variant wants to trigger on total funding magnitude regardless of direction, set `funding_cost_mode: "ABSOLUTE"` (default: `"ADVERSE_ONLY"`). This must be a conscious choice with documentation that it can force exits on favorable-funding positions.

**Funding event timestamp mapping (NORMATIVE):** Funding events are not bar closes. For deterministic evaluation, funding exit overlay is evaluated at the close of the 1m bar whose interval contains the funding timestamp. Formally: if funding posts at timestamp T, evaluation occurs at the close of the 1m bar where `bar_open ≤ T < bar_close`. The funding event timestamp and the mapped 1m bar close timestamp are both recorded in the decision log diagnostic for reproducibility.

### 4.6 Time Limit (OPTIONAL)

```
time_limit:
  time_limit_bars: Optional[int]     # e.g. 30 (days if reference = 1d)
  time_limit_reference_cadence: "1d"
  time_limit_allows_flip: false
```

**Time limit semantics (NORMATIVE):** Time limit counts bars of `time_limit_reference_cadence` from position entry timestamp. When the count reaches `time_limit_bars`, a MARKET exit is triggered. Time limit is superseded by any higher-priority exit that fires first (stop loss, MTM drawdown). If the position closes for any reason before the time limit, the time limit counter is cancelled. `time_limit_allows_flip: false` means no flip is attempted at time limit expiry.

### 4.7 Exit Path List Order

```
exit_rules:
  1. Stop loss (exchange-side)
  2. MTM drawdown exit (1m risk overlay)
  3. Time limit
  4. Funding exit overlay
  5. LONG signal exit ([exit_trigger] turn-down)
  6. SHORT signal exit ([exit_trigger] turn-up)
```

Stop loss has highest execution priority. MTM drawdown exit is second (risk overlay, bypasses rate limiter). Signal exits are last — the slow exit cadence is intentional (fast in, slow out).

**Stop loss vs MTM drawdown interaction (NORMATIVE):** In backtest, exchange-side stop loss is simulated conservatively: at each 1m bar, LONG stop loss triggers if bar LOW ≤ stop price; SHORT stop loss triggers if bar HIGH ≥ stop price. This approximates intra-bar mark-price behavior (live exchange uses MarkPrice trigger per Bybit OMS §2.2) and avoids the close-only simulation bias where intra-bar stop violations are missed. MTM drawdown is evaluated at 1m bar close. If both trigger on the same bar, stop loss takes priority (evaluated first). In live, exchange-side stop loss is authoritative — if the exchange fills the stop loss, the MTM drawdown order (if any was pending) MUST be cancelled and the cancellation logged. The diagnostic log records which mechanism actually closed the position.

**Exit co-trigger attribution (NORMATIVE):** Per framework §5.1.1, when multiple exit paths evaluate true simultaneously, the framework logs a `multi_exit_trigger` diagnostic event recording all satisfied paths and which was selected. This strategy-level contract requires that research analytics tools consume this diagnostic to distinguish "model exit" (signal) from "risk cut" (stop loss, MTM drawdown) when both fire at the same timestamp.

**Analytics normalization rule (NORMATIVE):** In post-trade analysis, if a signal exit condition (long_signal_exit or short_signal_exit) was true at the same timestamp that a higher-priority exit fired (stop_loss, mtm_drawdown_exit), the trade MUST be labeled `signal_coincident_risk_exit`. These trades MUST be excluded from (or separately weighted in) MACD parameter optimization loss functions.

---

## 5. Gate Rules

### 5.1 No Dedicated Regime Gate in v1

The MACD confluence strategy uses the multi-timeframe slope alignment as its implicit regime filter. When all macro and intra timeframes agree on direction, the regime is trending. When they disagree, no entry fires.

**Alignment is an entry-only filter (NORMATIVE, system owner decision):** If macro or intra alignment breaks while a position is open, the position is NOT exited. Exits are governed solely by exit rules (signal exit, stop loss, MTM drawdown exit, time limit, funding overlay) and risk overrides from the snapshot subsystem. This is a deliberate design choice — the entry thesis is strict (require full alignment to enter), but the exit thesis is independent (exit on exit-TF reversal or risk limits, not on partial alignment degradation).

**Rationale:** Trend degradation cascades through timeframes. A 12H slope flip while 1D and 3D remain positive may be temporary. Exiting on partial alignment break would increase churn. The 1D signal exit and stop loss provide sufficient exit coverage.

**Research note:** A research-toggleable "alignment break exit" path may be investigated in v2 if data shows holding through partial alignment breaks produces excess drawdown.

**Aligned chop reporting (NORMATIVE):** Research backtests MUST report the following metrics separately for periods where all macro/intra slopes are aligned but realized volatility is below the 25th percentile of the backtest period: trade count, win rate, average PnL per trade, and total PnL contribution.

**Canonical volatility estimator for aligned chop classification:** Realized volatility is computed as the annualized standard deviation of 1h close-to-close log returns over a 14-day (336-bar) rolling window. Annualization factor = sqrt(8760). The 25th percentile is computed over all rolling windows in the backtest period. This exact estimator MUST be used for aligned chop classification to ensure comparable results across research runs.

If aligned-chop trades contribute negative total PnL exceeding 10% of strategy gross profit, research MUST flag this and recommend a volatility floor gate for the next contract version.

```
gate_rules: []    # Empty in v1
```

### 5.2 Implication

With no gate, there is no gate-exit policy to define. Exits are governed purely by exit rules and risk overrides from the snapshot subsystem.

---

## 6. Execution Parameters

### 6.1 Perpetual Variant

```
execution_params:
  direction: BIDIRECTIONAL
  leverage: Decimal          # Tunable 1x-10x, strategy-global (applies uniformly to all positions)
  entry_type: MARKET
  position_sizing:
    mode: FIXED
    risk_fraction_bps: int   # e.g. 10000 = 100% of allocated capital
  # NOTE: stop_loss is defined in exit_rules §4.3, NOT duplicated here.
  # NOTE: time_limit is defined in exit_rules §4.6, NOT duplicated here.
  # execution_params only carries non-exit config.
  flip_enabled: false        # See §6.3
```

**Single source of truth (NORMATIVE):** Stop loss configuration lives exclusively in exit_rules (§4.3). The `execution_params` block does NOT contain stop loss values. Exchange-side placement is controlled by the `exchange_side` flag in the exit_rules stop_loss entry. This prevents divergence between backtest and live stop loss levels.

### 6.2 Spot Variant

```
execution_params_spot:
  direction: BIDIRECTIONAL   # SHORT requires margin/borrow, may be LONG-only on some venues
  leverage: "1.0"            # Spot = no leverage
  entry_type: MARKET
  position_sizing:
    mode: FIXED
    risk_fraction_bps: 10000   # 10000 = 100% of allocated capital. Integer bps.
  # NOTE: stop_loss is defined in exit_rules §4.3, NOT duplicated here (same as perp).
  # NOTE: mtm_drawdown_exit is defined in exit_rules §4.4, NOT duplicated here.
  time_limit_bars: null
  time_limit_reference_cadence: "1d"
  time_limit_allows_flip: false
  flip_enabled: false
  funding_exit:
    enabled: false           # No funding in spot
```

**Spot-specific rules:**
- No funding model (funding_model.enabled = false)
- No liquidation model
- leverage MUST be "1.0"
- No funding exit overlay (funding_exit.enabled = false)
- Stop loss and MTM drawdown exit are configured in exit_rules (§4.3, §4.4), identical structure to perp
- All risk thresholds use integer bps per §10.4

**Note:** The perp canonical JSON (§11) is the authoritative schema. The spot variant differs only in the fields listed above. A separate spot canonical JSON is not provided — the spot config is the perp JSON with: `variant: "spot"`, `leverage: "1.0"`, `funding_model.enabled: false`, `funding_exit.enabled: false`, and `spot_capabilities` non-null.

**Venue capability handling (NORMATIVE):** Spot variant MUST declare `spot_capabilities` in config:
```
spot_capabilities:
  allow_short: bool    # false if venue has no margin/borrow
```
If `allow_short: false`, the engine MUST prune all SHORT entry paths and SHORT-specific exit paths at config load time. A deterministic reason code `short_pruned_no_venue_support` is recorded. The strategy then operates as LONG-only. This pruning is part of the config identity and affects the config hash.

**Reproducibility (NORMATIVE):** `spot_capabilities` MUST be explicitly set in the canonical JSON (not auto-detected). It MUST appear in the run manifest. Auto-detection of venue capabilities is permitted only if the auto-detection writes a signed resolved-config artifact that becomes the canonical input for hashing and replay. Two runs with different `allow_short` values produce different config hashes and are distinct strategy instances — comparisons across them must account for this.

### 6.3 Flip Decision

**flip_enabled: false** in v1.

Rationale: The asymmetric entry/exit cadences (5m entry, 1D exit) mean that a flip would require an entry condition on 5m AND an exit condition on 1D to fire at the same timestamp. Since 1D bar closes are also 5m bar closes, this is possible in principle. However:

- The macro alignment for LONG and SHORT are mutually exclusive (all positive vs all negative)
- A transition from all-positive to all-negative doesn't happen in one bar — it cascades through timeframes
- Flip adds complexity without clear benefit for trend-following

Flip can be reconsidered if research shows value.

### 6.4 VOL_TARGETED Sizing (Research Extension)

```
execution_params_vol:
  position_sizing:
    mode: VOL_TARGETED
    target_vol_annual: Decimal  # e.g. 0.20 = 20% annualised
    vol_indicator_label: "vol_metric"  # Requires new indicator instance
    max_leverage: Decimal       # e.g. 10.0
    min_leverage: Decimal       # e.g. 0.5
    min_vol_threshold: Decimal  # Per framework §3.1
```

This requires a volatility indicator instance to be added. Not in v1 — requires research to validate.

---

## 7. Warmup Calculation

Per framework §6.4:

```
warmup = max across all instances of:
  (indicator_warmup_bars × timeframe_seconds)
```

For default configuration:

| Instance | Warmup Bars | Timeframe | Warmup Duration |
|----------|------------|-----------|-----------------|
| macro_3d | 27 (slow EMA + 1 for slope) | 259200s (3d) | 27 × 3d = 81 days |
| macro_1d | 27 | 86400s (1d) | 27 days |
| macro_12h | 27 | 43200s (12h) | 13.5 days |
| intra_1h | 27 | 3600s (1h) | 27 hours |
| intra_30m | 27 | 1800s (30m) | 13.5 hours |
| intra_15m | 27 | 900s (15m) | 6.75 hours |
| entry_5m | 27 | 300s (5m) | ~2.25 hours |
| exit_1d | 27 | 86400s (1d) | 27 days |

**Effective warmup: 81 days** (dominated by 3d macro instance)

**Warmup derivation (NORMATIVE):** Warmup bars = `slow_period + 1` (not `slow_period`). The slow EMA requires `slow_period` bars for the SMA seed + at least 1 bar of EMA computation to produce the first valid MACD value. The slope_sign output requires TWO consecutive valid MACD values (`M_t` and `M_{t-1}`), so `slow_period + 1` bars are needed for the first valid slope_sign. This matches the Phase 3 backtester behavior.

This is significant. The 3d timeframe drives an 81-day warmup. If this is unacceptable:
- Remove 3d from macro group (reduces to 27 days)
- Or accept 81-day warmup as cost of long-horizon trend confirmation

**Warmup restart policy**: Per framework §6.4, during warmup entries and flips are blocked. Risk overrides and exchange-side stops remain active.

**Restart with open position (NORMATIVE):** If the engine restarts while this strategy has an open position and indicators have not completed warmup, the strategy MUST enforce one of:
- **Policy A (FLATTEN):** Close all positions immediately at market.
- **Policy B (HARD_STOPS):** Monitor position with exchange-side stop loss only (already configured per §4.3) until indicators complete warmup. No entries, exits, or signal evaluation until warmup completes.

The chosen policy is a config parameter: `warmup_restart_policy: "FLATTEN" | "HARD_STOPS"` (default: `"HARD_STOPS"` for SHADOW/LIVE, `"FLATTEN"` for RESEARCH). This binds the framework's warmup safety requirement and the snapshot subsystem's restart safety requirement at the strategy level. An implementer who only reads this contract will know what to do on restart.

**RESEARCH mode flexibility (NORMATIVE):** In RESEARCH mode, `warmup_restart_policy` defaults to `"FLATTEN"` to allow exploratory research where stop loss values are not yet determined. If a researcher explicitly sets `warmup_restart_policy = "HARD_STOPS"` in RESEARCH, stop loss values MUST be non-null. This prevents research workflows from being blocked by validation while maintaining safety when stops are configured.

---

## 8. Funding Model (Perps Only)

```
funding_model:
  enabled: true
  interval_hours: 8
  rate_source: str              # "HISTORICAL" | "FIXED_CONSERVATIVE"
  rate_per_interval_bps: int    # Required if rate_source = "FIXED_CONSERVATIVE". Integer bps.
  historical_data_path: str     # Required if rate_source = "HISTORICAL". Path to venue funding CSV.
  credit_allowed: false         # Conservative: always charged
```

Per framework §3.7: backtest only. SHADOW/LIVE use exchange-reported funding.

**Deterministic funding (NORMATIVE):** Backtest funding MUST use a deterministic source — either historical venue data (exact rates by timestamp) or a fixed conservative constant. The `rate_source` and associated data/value MUST be included in the config hash. Two backtests with identical signals but different funding sources produce different config hashes and are distinct runs. This prevents funding assumptions from silently biasing research conclusions.

**SHADOW/LIVE funding ledger (NORMATIVE):** In SHADOW and LIVE modes, the engine MUST persist a "funding ledger" artifact recording each funding event: `{timestamp, rate_bps, cost_or_credit, cumulative_drag}`. This artifact is required for research-vs-live reconciliation and explains divergence where backtest uses conservative assumptions but live experiences different funding. The ledger is a first-class input to any cross-mode performance comparison.

---

## 9. Expected Behavior at Boundary Conditions

### 9.1 Daily Close at 00:00 UTC

**3d epoch anchoring (IMPORTANT):** 3d bar boundaries are anchored to Unix epoch (Thursday 1970-01-01), NOT to calendar weeks. This means 3d bar closes occur at deterministic but non-calendar-intuitive times. Macro_3d slope transitions can happen mid-week. Research MUST report decision clustering by actual coalesced timestamps (including 3d boundaries), not only by 00:00 UTC behavior.

At 00:00 UTC, the following bars close simultaneously: 1d, 12h, 1h, 30m, 15m, 5m (but NOT necessarily 3d).

Per framework §5.1 step 0 (coalesced cycle):
1. All 8 indicator instances update (those with closing bars)
2. Risk overrides evaluate
3. Rate limiter / slippage budget check
4. No gates (gate_rules empty)
5. Exit paths evaluate (1D signal exit)
6. If no exit: entry paths evaluate (5m turn-up)

**Critical scenario**: 1D exit fires AND 5m entry fires at 00:00 UTC.
- Exit has priority (framework §5.1.1 priority 3-4)
- flip_enabled=false → no flip
- Position exits. Entry suppressed (no re-entry in same cycle per framework step 0)
- Re-entry possible at next 5m bar (00:05 UTC) if conditions still met

**Post-exit cooldown (OPTIONAL):** To prevent churn at daily boundaries where slopes oscillate, the strategy MAY configure a `post_exit_cooldown_bars` parameter (default: 0, disabled). If set, no new entry is permitted for N bars of the entry cadence after any exit. Example: `post_exit_cooldown_bars: 3` with 5m entry cadence = 15 minute cooldown after exit. This is research-tunable.

### 9.2 Macro Alignment Transition

When one macro timeframe flips slope (e.g. 12H turns negative while 3D and 1D still positive):
- Entry condition fails (not all macro slopes positive)
- No new LONG entries
- Existing LONG position continues — exits only on 1D turn-down or stop loss
- This is correct: the entry gate is strict, but exit is independent

### 9.3 Gap During 3D Bar

A single 1m gap within a 3D bar invalidates the entire bar (zero tolerance in SHADOW/LIVE).
- macro_3d indicator output becomes None (unusable)
- Per framework None handling: condition evaluates to false
- No entries possible until next valid 3D bar
- Existing positions unaffected (exits still evaluate on their own cadence)

**Macro unavailable state (NORMATIVE):** If the highest macro timeframe (default 3d) produces None due to gap invalidation, the strategy enters a "macro_unavailable" state. In this state: no new entries are possible (alignment conditions fail), all exit paths remain active, risk overrides remain active, and the strategy may be dormant for up to one full macro bar duration (up to 3 days for 3d). This is explicitly accepted as the cost of using long-horizon timeframes with zero gap tolerance. A diagnostic `macro_unavailable=true` is recorded in the decision log for the affected period.

**Degraded mode (OPTIONAL, research-toggleable):** If `allow_degraded_macro: true` is set in config, entries are permitted when the highest macro instance returns None but ALL remaining macro instances AND all intra instances align. In degraded mode, the strategy MUST apply stricter risk controls: stop loss is tightened by `degraded_stop_tightening_bps` (integer bps added to base stop) and/or position size is reduced by `degraded_sizing_pct` (integer percentage: 50 = 50% of normal size, 100 = full size). Default: `allow_degraded_macro: false`. This addresses the post-gap vulnerability where trends start while the highest macro TF is invalidated.

**Degraded mode validation (NORMATIVE):** If `allow_degraded_macro: true`:
- Stop loss MUST be non-null (any mode). Degraded entries without a stop loss are prohibited.
- For SHADOW/LIVE: `exchange_side` MUST be true.
- At least one of `degraded_stop_tightening_bps` or `degraded_sizing_pct` MUST be non-null (you must apply at least one risk tightening measure).
- If `degraded_sizing_pct` is set, it MUST be in [10, 100] (minimum 10% position size).

---

## 10. Determinism Requirements

All framework §5.1 determinism invariants apply. Strategy-specific additions:

1. **Turn-up/turn-down detection**: Uses exactly two consecutive MACD values. No lookahead. Previous bar is the indicator's own timeframe previous bar (framework §4.1 cross-TF rule).

2. **Slope sign computation**: `sign(M_t − M_{t−1})` uses Decimal arithmetic under the framework's fixed Decimal context (precision ≥ 28, ROUND_HALF_EVEN). The subtraction and comparison are exact for Decimal.

3. **Mutual exclusivity of LONG/SHORT entry**: Macro alignment for LONG requires all slopes > 0. For SHORT, all slopes < 0. These cannot simultaneously be true. If any slope is exactly 0, neither entry fires (> 0 is strict, < 0 is strict).

4. **Unit normalization (NORMATIVE):** All risk thresholds in the strategy config use **integer basis points (bps)** as the canonical unit. Specifically:
   - `stop_loss.value_long_bps`, `stop_loss.value_short_bps`: integer bps (e.g. 500 = 5%)
   - `mtm_drawdown_exit.drawdown_bps_long`, `mtm_drawdown_exit.drawdown_bps_short`: integer bps (e.g. 1000 = 10%)
   - `funding_exit.threshold_bps_of_im`: integer bps (e.g. 500 = 5% of IM)
   - `post_exit_cooldown_bars`: integer (not bps, just bar count)
   - `degraded_sizing_pct`: integer percentage (not bps — 50 = 50%)
   
   No Decimal fractions for risk thresholds. All comparisons involving these values use integer arithmetic. This eliminates cross-platform Decimal context divergence for gating decisions.

---

## 11. Config Schema (Canonical JSON)

**Completeness (NORMATIVE):** The JSON below is the FULL canonical schema for the default perp variant in **authoring form** (pre-expansion). Every field shown is required. Every referenced indicator label MUST exist in indicator_instances. Instance counts MUST satisfy §2.4. Unknown fields MUST be rejected by the loader (strict schema). This is not an excerpt.

**Config lowering pipeline (NORMATIVE):** Before the config is passed to the framework evaluator or hashed, the loader MUST execute these steps in order:

1. **Schema validation**: Reject unknown fields, verify required fields present, type-check values.
2. **Timeframe normalization**: Convert all timeframe strings to canonical enum form (§2.3). Reject non-canonical.
3. **Role expansion**: Expand all `role_condition` and `{"role": ...}` references to concrete `{"indicator": "<label>", ...}` conditions per §2.2. Sort expanded arrays by indicator label ascending.
4. **Derived field resolution**: Resolve `"DERIVED:exit_trigger.timeframe"` to actual timeframe string.
5. **StopLoss lowering**: Convert `value_long_bps`/`value_short_bps` to framework StopLossConfig fields (§4.3).
6. **Canonicalize ordering**: Sort all JSON objects by key. Sort arrays where order is non-semantic. **Semantic-order arrays** (order = priority, MUST NOT be sorted): `entry_rules`, `exit_rules`. **Non-semantic arrays** (MUST be sorted for deterministic hashing): `indicator_instances` (by label), `conditions` (by indicator label), `condition_groups` (by label), `applies_to` (alphabetical).
7. **Emit resolved config**: This is the only hashable artifact. No authoring-only constructs (`role_condition`, `role`, `DERIVED:`) may remain.
8. **Hash**: SHA-256 of canonical JSON (sorted keys, no whitespace, UTF-8).

The pre-lowering authoring form is preserved as a diagnostic artifact. The framework evaluator and all downstream consumers receive only the resolved config.

### 11.1 Authoring Form (pre-expansion)

The JSON below is the authoring form. It contains role_conditions, DERIVED: fields, and bps stop loss values. This is what a human writes. The lowering pipeline transforms it into the resolved form (§11.2).

```json
{
  "strategy_id": "macd_confluence_v1",
  "engine_version": "1.7.0",
  "min_operator_version": "1.0",
  "variant": "perp",
  "warmup_restart_policy": "HARD_STOPS",
  "post_exit_cooldown_bars": 0,
  "indicator_instances": [
    {
      "label": "macro_3d",
      "role": "macro",
      "indicator_id": "macd_tv",
      "timeframe": "3d",
      "parameters": {"fast": 12, "slow": 26, "signal": 9},
      "outputs_used": ["slope_sign"],
      "data_source": "BAR",
      "bar_provider": "EXCHANGE_CANDLES"
    },
    {
      "label": "macro_1d",
      "role": "macro",
      "indicator_id": "macd_tv",
      "timeframe": "1d",
      "parameters": {"fast": 12, "slow": 26, "signal": 9},
      "outputs_used": ["slope_sign"],
      "data_source": "BAR",
      "bar_provider": "EXCHANGE_CANDLES"
    },
    {
      "label": "macro_12h",
      "role": "macro",
      "indicator_id": "macd_tv",
      "timeframe": "12h",
      "parameters": {"fast": 12, "slow": 26, "signal": 9},
      "outputs_used": ["slope_sign"],
      "data_source": "BAR",
      "bar_provider": "EXCHANGE_CANDLES"
    },
    {
      "label": "intra_1h",
      "role": "intra",
      "indicator_id": "macd_tv",
      "timeframe": "1h",
      "parameters": {"fast": 12, "slow": 26, "signal": 9},
      "outputs_used": ["slope_sign"],
      "data_source": "BAR",
      "bar_provider": "EXCHANGE_CANDLES"
    },
    {
      "label": "intra_30m",
      "role": "intra",
      "indicator_id": "macd_tv",
      "timeframe": "30m",
      "parameters": {"fast": 12, "slow": 26, "signal": 9},
      "outputs_used": ["slope_sign"],
      "data_source": "BAR",
      "bar_provider": "EXCHANGE_CANDLES"
    },
    {
      "label": "intra_15m",
      "role": "intra",
      "indicator_id": "macd_tv",
      "timeframe": "15m",
      "parameters": {"fast": 12, "slow": 26, "signal": 9},
      "outputs_used": ["slope_sign"],
      "data_source": "BAR",
      "bar_provider": "EXCHANGE_CANDLES"
    },
    {
      "label": "entry_5m",
      "role": "entry_trigger",
      "indicator_id": "macd_tv",
      "timeframe": "5m",
      "parameters": {"fast": 12, "slow": 26, "signal": 9},
      "outputs_used": ["macd_line", "slope_sign"],
      "data_source": "BAR",
      "bar_provider": "EXCHANGE_CANDLES"
    },
    {
      "label": "exit_1d",
      "role": "exit_trigger",
      "indicator_id": "macd_tv",
      "timeframe": "1d",
      "parameters": {"fast": 12, "slow": 26, "signal": 9},
      "outputs_used": ["slope_sign"],
      "data_source": "BAR",
      "bar_provider": "EXCHANGE_CANDLES"
    }
  ],
  "entry_rules": [
    {
      "label": "long_entry",
      "direction": "LONG",
      "evaluation_cadence": "5m",
      "entry_type": "MARKET",
      "condition_groups": [
        {
          "label": "macro_alignment",
          "role_condition": {"role": "macro", "output": "slope_sign", "operator": ">", "value": "0", "quantifier": "ALL"}
        },
        {
          "label": "intra_alignment",
          "role_condition": {"role": "intra", "output": "slope_sign", "operator": ">", "value": "0", "quantifier": "ALL"}
        }
      ],
      "conditions": [
        {"role": "entry_trigger", "output": "slope_sign", "operator": "crosses_above", "value": "0"}
      ]
    },
    {
      "label": "short_entry",
      "direction": "SHORT",
      "evaluation_cadence": "5m",
      "entry_type": "MARKET",
      "condition_groups": [
        {
          "label": "macro_alignment",
          "role_condition": {"role": "macro", "output": "slope_sign", "operator": "<", "value": "0", "quantifier": "ALL"}
        },
        {
          "label": "intra_alignment",
          "role_condition": {"role": "intra", "output": "slope_sign", "operator": "<", "value": "0", "quantifier": "ALL"}
        }
      ],
      "conditions": [
        {"role": "entry_trigger", "output": "slope_sign", "operator": "crosses_below", "value": "0"}
      ]
    }
  ],
  "exit_rules": [
    {
      "label": "stop_loss",
      "exit_type": "STOP_LOSS",
      "mode": "PERCENT",
      "exchange_side": true,
      "value_long_bps": null,
      "value_short_bps": null
    },
    {
      "label": "mtm_drawdown_exit",
      "exit_type": "MTM_DRAWDOWN_EXIT",
      "enabled": false,
      "evaluation_cadence": "1m",
      "drawdown_bps_long": null,
      "drawdown_bps_short": null,
      "mtm_includes_funding": true,
      "mtm_includes_fees": true
    },
    {
      "label": "time_limit",
      "exit_type": "TIME_LIMIT",
      "enabled": false,
      "time_limit_bars": null,
      "time_limit_reference_cadence": "1d"
    },
    {
      "label": "funding_exit",
      "exit_type": "SIGNAL",
      "enabled": false,
      "evaluation_cadence": "FUNDING_EVENT",
      "applies_to": ["LONG", "SHORT"],
      "threshold_bps_of_im": null,
      "require_underwater": true,
      "funding_cost_mode": "ADVERSE_ONLY"
    },
    {
      "label": "long_signal_exit",
      "exit_type": "SIGNAL",
      "applies_to": ["LONG"],
      "evaluation_cadence": "DERIVED:exit_trigger.timeframe",
      "conditions": [
        {"role": "exit_trigger", "output": "slope_sign", "operator": "<", "value": "0"}
      ]
    },
    {
      "label": "short_signal_exit",
      "exit_type": "SIGNAL",
      "applies_to": ["SHORT"],
      "evaluation_cadence": "DERIVED:exit_trigger.timeframe",
      "conditions": [
        {"role": "exit_trigger", "output": "slope_sign", "operator": ">", "value": "0"}
      ]
    }
  ],
  "gate_rules": [],
  "execution_params": {
    "direction": "BIDIRECTIONAL",
    "leverage": "5.0",
    "position_sizing": {"mode": "FIXED", "risk_fraction_bps": 10000},
    "flip_enabled": false
  },
  "funding_model": {
    "enabled": true,
    "interval_hours": 8,
    "rate_source": "FIXED_CONSERVATIVE",
    "rate_per_interval_bps": 1,
    "historical_data_path": null,
    "credit_allowed": false
  },
  "spot_capabilities": null,
  "fallback_exit": null,
  "allow_degraded_macro": false,
  "degraded_stop_tightening_bps": null,
  "degraded_sizing_pct": null
}
```

**Validation rules (NORMATIVE):**
1. Every indicator label referenced in entry_rules or exit_rules MUST exist in indicator_instances.
2. Indicator instance count MUST satisfy §2.4 (min/max per group).
3. Unknown fields MUST cause a validation error (strict schema).
4. `stop_loss.value_long_bps` and `value_short_bps` MUST be non-null integers in [100, 2000] for SHADOW/LIVE promotion, and non-null if `warmup_restart_policy = "HARD_STOPS"` in any mode.
5. `spot_capabilities` MUST be non-null for variant="spot".
6. **Feature disable semantics (NORMATIVE):** The `enabled` field is the authoritative disable mechanism for optional features (mtm_drawdown_exit, time_limit, funding_exit). When `enabled: false`, all associated parameter fields MAY be null. When `enabled: true`, all required parameter fields MUST be non-null (validation error otherwise). The loader MUST NOT infer enabled/disabled from null parameters alone — only the `enabled` field controls this. Features without an `enabled` field (stop_loss, signal exits) are always active when their required parameters are non-null.
7. All timeframe strings MUST match the canonical enum (§2.3).
8. **Canonical JSON hashing (NORMATIVE):** Config hash is computed over a canonical JSON form with: sorted keys, no whitespace, UTF-8 encoding. All fields defined in the schema MUST be present (no omission). `null` is the explicit representation for absent/disabled values. Different serializers MUST NOT omit null fields — `"field": null` and field-omitted produce different hashes and are treated as different configs.
9. **Derived cadence resolution (NORMATIVE):** Fields with value `"DERIVED:exit_trigger.timeframe"` MUST be resolved to the actual timeframe of the exit_trigger instance at config load time, BEFORE hashing. The resolved value is what appears in the hashed canonical form. The pre-resolution `DERIVED:` syntax is a config authoring convenience only.

### 11.2 Resolved Form (post-lowering, hashable)

The JSON below is the RESOLVED form of the default perp config — the output of the 8-step lowering pipeline. This is what the framework evaluator receives and what step 8 hashes. All role references have been expanded to concrete indicator labels. All DERIVED: fields have been resolved. Stop loss bps have been lowered to framework fields. Arrays are sorted per step 6 rules (indicator_instances by label, conditions by indicator label; entry_rules and exit_rules preserve semantic order).

**Reference hash (SHA-256):** `1a1b5f0a92517ca39a9e766ab3211d59b598db0ec3865475ee46053dd88469c2`

Any conforming implementation that applies the lowering pipeline to the §11.1 authoring form MUST produce this exact hash. If hash does not match, the lowering implementation has a bug.

**Key differences from authoring form:**
- `role_condition` objects → expanded to per-instance `{"indicator": ...}` conditions (sorted by indicator label)
- `{"role": "entry_trigger", ...}` → `{"indicator": "entry_5m", ...}`
- `{"role": "exit_trigger", ...}` → `{"indicator": "exit_1d", ...}`
- `"DERIVED:exit_trigger.timeframe"` → `"1d"`
- `value_long_bps`/`value_short_bps` → `percent_long`/`percent_short` (null in default, Decimal when set)
- `indicator_instances` sorted by label
- `condition_groups` sorted by label
- `conditions` within groups sorted by indicator label
- `applies_to` standardized to arrays

```json
{
  "allow_degraded_macro": false,
  "degraded_sizing_pct": null,
  "degraded_stop_tightening_bps": null,
  "engine_version": "1.7.0",
  "entry_rules": [
    {
      "condition_groups": [
        {
          "conditions": [
            {"indicator": "intra_15m", "operator": ">", "output": "slope_sign", "value": "0"},
            {"indicator": "intra_1h", "operator": ">", "output": "slope_sign", "value": "0"},
            {"indicator": "intra_30m", "operator": ">", "output": "slope_sign", "value": "0"}
          ],
          "label": "intra_alignment"
        },
        {
          "conditions": [
            {"indicator": "macro_12h", "operator": ">", "output": "slope_sign", "value": "0"},
            {"indicator": "macro_1d", "operator": ">", "output": "slope_sign", "value": "0"},
            {"indicator": "macro_3d", "operator": ">", "output": "slope_sign", "value": "0"}
          ],
          "label": "macro_alignment"
        }
      ],
      "conditions": [
        {"indicator": "entry_5m", "operator": "crosses_above", "output": "slope_sign", "value": "0"}
      ],
      "direction": "LONG",
      "entry_type": "MARKET",
      "evaluation_cadence": "5m",
      "label": "long_entry"
    },
    {
      "condition_groups": [
        {
          "conditions": [
            {"indicator": "intra_15m", "operator": "<", "output": "slope_sign", "value": "0"},
            {"indicator": "intra_1h", "operator": "<", "output": "slope_sign", "value": "0"},
            {"indicator": "intra_30m", "operator": "<", "output": "slope_sign", "value": "0"}
          ],
          "label": "intra_alignment"
        },
        {
          "conditions": [
            {"indicator": "macro_12h", "operator": "<", "output": "slope_sign", "value": "0"},
            {"indicator": "macro_1d", "operator": "<", "output": "slope_sign", "value": "0"},
            {"indicator": "macro_3d", "operator": "<", "output": "slope_sign", "value": "0"}
          ],
          "label": "macro_alignment"
        }
      ],
      "conditions": [
        {"indicator": "entry_5m", "operator": "crosses_below", "output": "slope_sign", "value": "0"}
      ],
      "direction": "SHORT",
      "entry_type": "MARKET",
      "evaluation_cadence": "5m",
      "label": "short_entry"
    }
  ],
  "execution_params": {
    "direction": "BIDIRECTIONAL",
    "flip_enabled": false,
    "leverage": "5.0",
    "position_sizing": {"mode": "FIXED", "risk_fraction_bps": 10000}
  },
  "exit_rules": [
    {
      "exchange_side": true,
      "exit_type": "STOP_LOSS",
      "label": "stop_loss",
      "mode": "PERCENT",
      "percent_long": null,
      "percent_short": null
    },
    {
      "drawdown_bps_long": null,
      "drawdown_bps_short": null,
      "enabled": false,
      "evaluation_cadence": "1m",
      "exit_type": "MTM_DRAWDOWN_EXIT",
      "label": "mtm_drawdown_exit",
      "mtm_includes_fees": true,
      "mtm_includes_funding": true
    },
    {
      "enabled": false,
      "exit_type": "TIME_LIMIT",
      "label": "time_limit",
      "time_limit_bars": null,
      "time_limit_reference_cadence": "1d"
    },
    {
      "applies_to": ["LONG", "SHORT"],
      "enabled": false,
      "evaluation_cadence": "FUNDING_EVENT",
      "exit_type": "SIGNAL",
      "funding_cost_mode": "ADVERSE_ONLY",
      "label": "funding_exit",
      "require_underwater": true,
      "threshold_bps_of_im": null
    },
    {
      "applies_to": ["LONG"],
      "conditions": [
        {"indicator": "exit_1d", "operator": "<", "output": "slope_sign", "value": "0"}
      ],
      "evaluation_cadence": "1d",
      "exit_type": "SIGNAL",
      "label": "long_signal_exit"
    },
    {
      "applies_to": ["SHORT"],
      "conditions": [
        {"indicator": "exit_1d", "operator": ">", "output": "slope_sign", "value": "0"}
      ],
      "evaluation_cadence": "1d",
      "exit_type": "SIGNAL",
      "label": "short_signal_exit"
    }
  ],
  "fallback_exit": null,
  "funding_model": {
    "credit_allowed": false,
    "enabled": true,
    "historical_data_path": null,
    "interval_hours": 8,
    "rate_per_interval_bps": 1,
    "rate_source": "FIXED_CONSERVATIVE"
  },
  "gate_rules": [],
  "indicator_instances": [
    {"bar_provider": "EXCHANGE_CANDLES", "data_source": "BAR", "indicator_id": "macd_tv", "label": "entry_5m", "outputs_used": ["macd_line", "slope_sign"], "parameters": {"fast": 12, "signal": 9, "slow": 26}, "role": "entry_trigger", "timeframe": "5m"},
    {"bar_provider": "EXCHANGE_CANDLES", "data_source": "BAR", "indicator_id": "macd_tv", "label": "exit_1d", "outputs_used": ["slope_sign"], "parameters": {"fast": 12, "signal": 9, "slow": 26}, "role": "exit_trigger", "timeframe": "1d"},
    {"bar_provider": "EXCHANGE_CANDLES", "data_source": "BAR", "indicator_id": "macd_tv", "label": "intra_15m", "outputs_used": ["slope_sign"], "parameters": {"fast": 12, "signal": 9, "slow": 26}, "role": "intra", "timeframe": "15m"},
    {"bar_provider": "EXCHANGE_CANDLES", "data_source": "BAR", "indicator_id": "macd_tv", "label": "intra_1h", "outputs_used": ["slope_sign"], "parameters": {"fast": 12, "signal": 9, "slow": 26}, "role": "intra", "timeframe": "1h"},
    {"bar_provider": "EXCHANGE_CANDLES", "data_source": "BAR", "indicator_id": "macd_tv", "label": "intra_30m", "outputs_used": ["slope_sign"], "parameters": {"fast": 12, "signal": 9, "slow": 26}, "role": "intra", "timeframe": "30m"},
    {"bar_provider": "EXCHANGE_CANDLES", "data_source": "BAR", "indicator_id": "macd_tv", "label": "macro_12h", "outputs_used": ["slope_sign"], "parameters": {"fast": 12, "signal": 9, "slow": 26}, "role": "macro", "timeframe": "12h"},
    {"bar_provider": "EXCHANGE_CANDLES", "data_source": "BAR", "indicator_id": "macd_tv", "label": "macro_1d", "outputs_used": ["slope_sign"], "parameters": {"fast": 12, "signal": 9, "slow": 26}, "role": "macro", "timeframe": "1d"},
    {"bar_provider": "EXCHANGE_CANDLES", "data_source": "BAR", "indicator_id": "macd_tv", "label": "macro_3d", "outputs_used": ["slope_sign"], "parameters": {"fast": 12, "signal": 9, "slow": 26}, "role": "macro", "timeframe": "3d"}
  ],
  "min_operator_version": "1.0",
  "post_exit_cooldown_bars": 0,
  "spot_capabilities": null,
  "strategy_id": "macd_confluence_v1",
  "variant": "perp",
  "warmup_restart_policy": "HARD_STOPS"
}
```

---

## 12. Research Parameters (TBD)

The following parameters require research determination before promotion beyond RESEARCH mode:

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| MACD fast period | 12 | 8-20 | Per timeframe, fully specified |
| MACD slow period | 26 | 18-50 | Per timeframe, fully specified |
| Stop loss % (LONG) | TBD | 100-2000 bps | Required for SHADOW/LIVE |
| Stop loss % (SHORT) | TBD | 100-2000 bps | May differ from LONG |
| MTM drawdown exit % (LONG) | TBD | 500-2000 bps | Optional |
| MTM drawdown exit % (SHORT) | TBD | 500-2000 bps | Optional, may differ |
| Funding exit threshold | TBD | 200-1000 bps of IM | Optional, perps only, direction-aware |
| Time limit (bars) | null | 7-90 | Optional, in exit_reference_cadence bars |
| Leverage (perps) | 5x | 1x-10x | Strategy-global |
| Macro timeframes | 3D, 1D, 12H | Any ≥ 12H | Ordering constraints §2.3 |
| Intra timeframes | 1H, 30m, 15m | Constrained by §2.3 | |
| Entry timeframe | 5m | ≥ 1m, ≤ min(intra) | |
| Exit timeframe | 1D | ≥ max(intra) | Tunable, default 1D |
| Post-exit cooldown | 0 (disabled) | 0-12 bars | In entry cadence bars |

---

## 13. Variants Summary

| Variant | Direction | Leverage | Funding | Sizing |
|---------|-----------|----------|---------|--------|
| `macd_confluence_v1_perp` | Bidirectional | 1x-10x | Yes | FIXED |
| `macd_confluence_v1_spot` | Bidirectional* | 1x | No | FIXED |
| `macd_confluence_v1_vol` | Bidirectional | 1x-10x | Yes | VOL_TARGETED |

*SHORT on spot requires venue margin/borrow support. May be LONG-only on some venues.

VOL_TARGETED variant requires research validation before implementation.

---

## 14. Known Economic Risks and Mitigations

Identified during adversarial review. These are not bugs — they are structural properties of the strategy that create predictable vulnerability windows.

### 14.1 Daily Boundary Microstructure Risk

**Risk:** Liquidity thins near 00:00 UTC; spreads widen. The strategy's key evaluation (1D exit + potential re-entry) clusters at this boundary. MARKET orders get worse fills.

**Mitigation:** The framework's slippage budget (§3.9) and catastrophic slippage cooldown provide first-line defense. Additionally, the post-exit cooldown (§9.1) prevents immediate re-entry after boundary exits. Research should evaluate whether a "boundary blackout" window (no entries within X minutes of daily close) improves net performance.

### 14.2 Aligned Chop (Slopes Agree But Market Ranges)

**Risk:** Multi-TF slopes can all be positive in a slow upward drift/range, but 5m turn-ups fire on noise. Repeated entries get stopped out.

**Mitigation:** The stop loss provides capital protection. The 1D exit provides an eventual model-driven exit. Research should evaluate whether adding an ATR minimum threshold or realized volatility gate reduces chop losses without missing trend starts. This is a v2 gate investigation.

### 14.3 Warmup Inactivity Window

**Risk:** 81-day warmup means missing the start of trends after restart or new deployment. Not adversarial but operationally significant.

**Mitigation:** Removing 3d from macro group reduces warmup to 27 days. Alternatively, the system owner can accept the 81-day cost for the benefit of long-horizon trend confirmation. Research should quantify: does 3d macro add enough value to justify 81 days of inactivity?

### 14.4 MARKET Order Fill Quality on Breakouts

**Risk:** MARKET entries during fast moves fill at materially worse prices than signal price.

**Mitigation:** Framework slippage budget gates entry (§3.9). Research should evaluate whether a "marketable limit" with bounded deviation (max X bps from signal price) improves fill quality without missing entries.

---

## 15. Non-Goals (v1)

1. **No signal line crossover** — We use MACD slope (momentum direction), not signal line. Signal line is reserved for potential v2 enhancement.
2. **No histogram** — MACD histogram is the derivative of what we already use. No additional information.
3. **No divergence detection** — Price/MACD divergence requires complex multi-bar pattern recognition. Out of scope for v1.
4. **No adaptive parameters** — MACD periods are fixed per instance. Adaptive periods (vol-adjusted lookback) are a v2 research topic.
5. **No multi-instrument correlation** — Strategy operates on single instrument. Cross-instrument signals are out of scope.

---

## Amendment Log

**Note:** Amendment log entries reflect the state of the contract at each version. For current normative behavior, refer to the relevant section in the current version, not the amendment log. Field names, types, and semantics in earlier entries may have been superseded by later versions.

### v1.0.0 (2026-02-12)

Initial draft. Based on Phase 3 backtester formal mathematical specification, Phase 3 analysis showing asymmetric entry/exit, and system owner decisions.

### v1.0.0 → v1.1.0 (2026-02-12)

Amendments from adversarial review round 1 (ChatGPT red team, 14 issues + economic attack surface, all 14 accepted):

| Issue | Amendment | Section |
|-------|-----------|---------|
| 1 | Fixed entry expression: slope_sign crosses_above 0 (not macd_line) | §3.1, §3.2 |
| 2 | Thesis corrected: exit TF tunable (default 1D), not "slowest monitored" | §1 |
| 3 | Funding exit: added applies_to, evaluation_cadence, direction-aware semantics | §4.5 |
| 4 | Spot variant: added spot_capabilities with allow_short and pruning rules | §6.2 |
| 5 | Timeframe constraints collapsed to 5 consistent rules | §2.3 |
| 6 | Slope_sign=0 documented as intentional conservative design choice | §3.4 (new) |
| 7 | Alignment is entry-only filter (system owner decision, documented) | §5.1 |
| 8 | Stop loss validation: must be non-null Decimal for SHADOW/LIVE promotion | §4.3 |
| 9 | Funding overlay: direction-aware sign handling for live trading | §4.5 |
| 10 | Macro unavailable state: documented dormancy up to one macro bar | §9.3 |
| 11 | Exit co-trigger attribution: referenced framework multi_exit_trigger logging | §4.7 |
| 12 | Post-exit cooldown: optional, research-tunable, default disabled | §9.1 |
| 13 | Per-direction parameters: stop loss, trailing stop now per LONG/SHORT | §4.3, §4.4, §12 |
| 14 | Parameter specification: full per instance, no implicit inheritance | §2.1 |

Economic attack surface documented in new §14 (daily boundary risk, aligned chop, warmup window, fill quality).

### v1.1.0 → v1.2.0 (2026-02-12)

Amendments from adversarial review round 2 (ChatGPT red team, 12 issues, 11 accepted, 1 rejected):

| Issue | Amendment | Section |
|-------|-----------|---------|
| 1 | Trailing stop evaluation cadence = 1m (CRITICAL — matches backtester) | §4.4 |
| 2 | spot_capabilities in canonical JSON and run manifest, reproducibility rules | §6.2 |
| 3 | Exit TF invalidation fallback constraint | §2.3 |
| 4 | REJECTED — slope_sign=0 already documented as intentional in §3.4 | No change |
| 5 | post_exit_cooldown_bars in canonical JSON schema | §11 |
| 6 | Analytics normalization rule for signal-coincident risk exits | §4.7 |
| 7 | Funding overlay uses adverse-only cost, not abs() | §4.5 |
| 8 | Canonical timeframe enum with normalization at parse time | §2.3 |
| 9 | Strategy binds restart safety policy (FLATTEN or HARD_STOPS) | §7 |
| 10 | Aligned chop research reporting requirement | §5.1 |
| 11 | Full canonical JSON (all 8 instances, all fields, strict schema) | §11 |
| 12 | All risk thresholds unified to integer bps | §10 |

**Cumulative totals across 2 rounds:**
- Issues reviewed: 26
- Accepted: 25
- Rejected: 1

### v1.2.0 → v1.3.0 (2026-02-12)

Amendments from adversarial review round 3 (ChatGPT red team, 12 issues, all 12 accepted):

| Issue | Amendment | Section |
|-------|-----------|---------|
| 1 | Purged Decimal wording for risk thresholds — integer bps only | §4.3, §10.4 |
| 2 | Exit TF tunability via role-based binding (exit_trigger role) | §2.2, §4.1, §4.2, §11 |
| 3 | 3d epoch anchoring documented, research clustering requirement | §9.1 |
| 4 | Stop loss single source of truth in exit_rules, removed from execution_params | §6.1, §11 |
| 5 | Fallback exit as concrete secondary indicator instance (exit_fallback role) | §2.3 |
| 6 | Role-based group evaluation replaces hardcoded label references | §2.2, §3.1, §3.2, §11 |
| 7 | Trailing stop classified as risk exit, bypasses rate limiter | §4.4 |
| 8 | Funding event → 1m bar mapping defined | §4.5 |
| 9 | Canonical vol estimator specified for aligned chop reporting | §5.1 |
| 10 | Degraded macro mode (optional, research-toggleable) | §9.3 |
| 11 | HARD_STOPS requires non-null stop loss even in RESEARCH | §4.3, §7 |
| 12 | Canonical JSON hashing: null fields always present, no omission | §11 |

**Cumulative totals across 3 rounds:**
- Issues reviewed: 38
- Accepted: 37
- Rejected: 1

### v1.3.0 → v1.4.0 (2026-02-12)

Amendments from adversarial review round 4 (ChatGPT red team, 12 issues, 8 accepted, 4 rejected):

| Issue | Amendment | Section |
|-------|-----------|---------|
| 1 | Stop loss aligned with framework StopLossConfig (mode: PERCENT) | §4.3 |
| 2 | MTM drawdown exit renamed as new primitive (NOT framework TRAILING_STOP) | §4.4 |
| 3 | MTM drawdown precedence: between RISK_OVERRIDE and GATE_EXIT | §4.4 |
| 4 | REJECTED — thesis already updated in v1.1.0, exit TF tunable | — |
| 5 | Stop loss type renamed to BPS, explicit conversion semantics | §4.3 |
| 6 | RESEARCH default warmup_restart_policy = FLATTEN | §7 |
| 7 | 1m data requirement for MTM drawdown, FORCE_FLAT on data gap | §4.4 |
| 8 | REJECTED — gate deferred to research per §5.1 | — |
| 9 | REJECTED — 3d epoch anchoring already documented in v1.3.0 §9.1 | — |
| 10 | REJECTED — flip disabled is system owner decision | — |
| 11 | Deterministic funding series (HISTORICAL or FIXED_CONSERVATIVE) | §8 |
| 12 | Inactivity reporting requirement added | §3.4 |

**Cross-contract action items (new):**
- Framework v1.7.0 needs MTM_DRAWDOWN_EXIT exit type with precedence slot between RISK_OVERRIDE and GATE_EXIT

**Cumulative totals across 4 rounds:**
- Issues reviewed: 50
- Accepted: 45
- Rejected: 5

### v1.4.0 → v1.5.0 (2026-02-12)

Amendments from adversarial review round 5 (Claude self-review, 14 issues, 13 accepted, 1 rejected):

| Issue | Amendment | Section |
|-------|-----------|---------|
| 1 | Spot variant §6.2 rewritten — removed fossil fields, aligned with perp | §6.2 |
| 2 | role_condition expansion rule defined — expands to per-instance conditions at load | §2.2 |
| 3 | Signal exit evaluation_cadence derived from exit_trigger timeframe | §4.1, §4.2, §11 |
| 4 | Funding exit field standardized to threshold_bps_of_im everywhere | §4.5, §10.4, §11 |
| 5 | Warmup corrected: slow_period + 1 = 27 bars (81 days for 3d) | §7, §14.3 |
| 6 | Stop loss vs MTM drawdown race resolution defined | §4.7 |
| 7 | degraded_sizing_multiplier_bps renamed to degraded_sizing_pct (percentage) | §9.3, §10.4 |
| 8 | REJECTED — macd_line in outputs_used supports research debugging | — |
| 9 | Time limit §4.6 prose added (semantics, cancellation, priority) | §4.6 |
| 10 | fallback_exit_cadence replaced with fallback_exit object in JSON | §11 |
| 11 | §10.4 field names updated to match canonical JSON | §10.4 |
| 12 | Validation rule 6: trailing_stop → mtm_drawdown_exit | §11 |
| 13 | Constraint 4: "12H" → "12h" (canonical enum) | §2.3 |
| 14 | Amendment log disclaimer added | Amendment Log header |

**Cross-contract action items (cumulative):**
- Framework v1.7.0 needs MTM_DRAWDOWN_EXIT exit type with precedence slot
- Framework v1.7.0 needs role_condition expansion as a normative config loader step (or formal extension)

**Cumulative totals across 5 rounds:**
- Issues reviewed: 64
- Accepted: 58
- Rejected: 6

### v1.5.0 → v1.6.0 (2026-02-12)

Amendments from adversarial review round 6 (ChatGPT red team of v1.5.0, 13 issues, 11 accepted, 2 rejected):

| Issue | Amendment | Section |
|-------|-----------|---------|
| 1 | ALL role references expanded (not just role_condition) | §2.2 |
| 2 | REJECTED — slow_period + 1 = 27 is correct per SMA seed derivation | — |
| 3 | exit_fallback role added to role definitions table | §2.2 |
| 4 | Role expansion ordering: sorted by (timeframe_seconds, label) | §2.2 |
| 5 | StopLossConfig lowering step defined | §4.3 |
| 6 | Canonical timeframe enum example fixed (was self-contradictory) | §2.3 |
| 7 | Leverage wording corrected to strategy-global | §6.1 |
| 8 | Single canonical disable mechanism: `enabled` field is authoritative | §11 |
| 9 | SHADOW/LIVE must persist funding ledger artifact | §8 |
| 10 | REJECTED — MTM drawdown bypasses slippage by design; framework catastrophic cooldown is backstop | — |
| 11 | Degraded mode validation: stop loss required, minimum sizing | §9.3 |
| 12 | Ordering constraint handles singleton macro/intra | §2.3 |
| 13 | Normative lowering pipeline defined (8-step) | §11 |

**Cross-contract action items (cumulative):**
- Framework v1.7.0 needs MTM_DRAWDOWN_EXIT exit type with precedence slot
- Framework v1.7.0 needs role_condition expansion as normative config loader step (or formal extension)
- Framework StopLossConfig needs per-direction fields (percent_long, percent_short)

**Cumulative totals across 6 rounds:**
- Issues reviewed: 77
- Accepted: 69
- Rejected: 8

**Awaiting**: System owner freeze decision

### v1.6.0 → v1.7.0 (2026-02-12)

Amendments from adversarial review round 7 (Claude self-review of v1.6.0, 13 issues, 12 accepted, 1 rejected):

| Issue | Amendment | Section |
|-------|-----------|---------|
| 1 | StopLossConfig lowering target schema made explicit (version-pinned) | §4.3 |
| 2 | Resolved-form JSON with reference SHA-256 hash added | §11.2 (new) |
| 3 | time_limit removed from execution_params (single source of truth in exit_rules) | §6.1, §11 |
| 4 | Backtest stop loss simulation uses bar LOW/HIGH (not close) | §4.7 |
| 5 | applies_to standardized to array everywhere | §4.1, §11 |
| 6 | Canonical JSON funding rate_per_interval_bps set to 1 (was null, failed validation) | §11 |
| 7 | exit_fallback evaluation defined as transparent substitution | §2.2 |
| 8 | Semantic vs non-semantic array ordering explicitly listed | §11 pipeline step 6 |
| 9 | §12 research table: "trailing stop" → "MTM drawdown exit", leverage → "strategy-global" | §12 |
| 10 | §5.1: "trailing stop" → "MTM drawdown exit" | §5.1 |
| 11 | REJECTED — inactivity thresholds are system-owner guardrails, not research-tunable | — |
| 12 | EMERGENCY_FLATTEN → FLATTEN (consistent naming) | §7 |
| 13 | §2.3: "trailing stop" → "MTM drawdown exit" | §2.3 |

**Cross-contract action items (cumulative):**
- Framework v1.7.0 needs MTM_DRAWDOWN_EXIT exit type with precedence slot
- Framework v1.7.0 needs per-direction StopLossConfig fields (percent_long, percent_short)
- Framework v1.7.0 needs role_condition expansion as normative config loader step (or formal extension)

**Cumulative totals across 7 rounds:**
- Issues reviewed: 90
- Accepted: 81
- Rejected: 9

**Awaiting**: System owner freeze decision
