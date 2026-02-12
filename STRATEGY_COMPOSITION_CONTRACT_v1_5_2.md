# STRATEGY COMPOSITION CONTRACT

**Version**: 1.5.2
**Status**: FINAL — Ready for implementation (5 adversarial rounds + cross-contract alignment patch)
**Changelog**: v1.5.2 patches gate field naming to align with Strategy Framework v1.7.0 contract. See Amendment Log.
**Date**: 2026-02-12
**Authority Class**: Authoring-layer contract (compiles to STRATEGY_FRAMEWORK_CONTRACT v1.7.0+ artifacts)
**Depends on**: STRATEGY_FRAMEWORK_CONTRACT v1.7.0, PHASE4A_INDICATOR_CONTRACT
**Transitive dependency (via Framework):** LOW_LATENCY_SNAPSHOT_SUBSYSTEM v1.3.0 — applies only when the composition targets snapshot `bar_provider` or snapshot-specific timeframes. Non-snapshot compositions have no snapshot subsystem dependency.
**Does NOT replace**: Any runtime contract. The Strategy Framework is the sole runtime contract.

---

## 1. Purpose and Scope

### 1.1 What This Contract Defines

This contract defines the **strategy composition system** — an authoring and compilation layer that lets a researcher compose trading strategies from any combination of indicators, conditions, and execution parameters, and compile them into deterministic Strategy Framework artifacts.

The composition system is the "artist's palette." The Strategy Framework engine is the canvas and physics of paint. This contract defines the palette; it does not modify the canvas.

### 1.2 What This Contract Does NOT Define

- **Runtime evaluation semantics.** Those belong to the Strategy Framework contract.
- **Indicator computation.** That belongs to Phase 4A/4B contracts.
- **Data delivery.** That belongs to the Low Latency Snapshot Subsystem.
- **Order execution.** That belongs to the Bybit OMS contract.
- **A second execution engine.** There is one engine. This produces configs for it.

### 1.3 The Authoring/Runtime Boundary

This is the most important rule in the contract:

**The composition layer is compile-time only. The engine loads only resolved Strategy Framework configs. Roles, archetype tags, condition groups with human names, and all other authoring constructs are lowered away before the engine sees anything.**

If a feature requires runtime behavior that the Strategy Framework does not support, the correct action is to amend the Strategy Framework contract — not to add runtime semantics to this composition layer.

**Normative enforcement:** The resolved Strategy Framework artifact MUST NOT contain any of the following fields: `role`, `group`, `filter_group`, `composition_id`, `display_name`, `description`, `thesis`, `known_risks`, `fallback_bindings`, or any field not defined in the Strategy Framework schema. If any authoring-only field is present in a resolved artifact, the artifact is invalid and MUST be rejected by the engine loader. The engine MUST NOT accept, parse, or silently ignore unknown fields.

**Determinism invariant:** The composition system and all strategy configs it produces are fully deterministic. No stochastic, randomized, or entropy-dependent components are permitted in composition specs, lowering logic, or resolved artifacts. All strategy behavior is determined entirely by the OHLCV data series and the strategy config. This mirrors System Law determinism requirements.

### 1.4 Operating Model: Always-On Deterministic Spine

This system operates under a single model. There is no split between "research mode" and "deployment mode." Determinism, reproducibility, and auditability are always active, even though this is primarily a personal research system.

**Core principles:**

1. **Deterministic compilation is non-negotiable.** Given identical inputs (composition spec, target engine version, capability registry, framework schema), the compiler produces identical resolved artifacts and identical hashes. Always. This is not a testing requirement — it is the permanent operating posture.

2. **The resolved artifact is the only runtime truth.** The engine loads only the resolved artifact. The composition spec and lowering report never influence runtime behavior. Phase 5 reads metadata from the composition spec via explicit binding (`composition_spec_hash` in promotion artifacts), not by runtime coupling.

3. **All artifacts are content-addressed and immutable.** No hash prefixes in paths. No overwrites. Full-hash directory structure. Once an artifact is written, it is never modified.

4. **Promotion always binds to cryptographic identity.** Promotion evidence is anchored to exact artifacts via content hashes. This is not adversarial protection — it prevents the system owner from accidentally promoting a strategy that has silently changed.

5. **Auditability is structural, not optional.** It must be possible to reproduce any promoted result at any time without reconfiguring the system. The lowering report, promotion artifacts, and content hashes exist to make this trivially verifiable.

6. **Research iteration remains low-friction.** Soft constraints warn in RESEARCH and reject in SHADOW/LIVE (§5.2). The compilation pipeline is identical in all modes — only the validation threshold differs. This is graduated strictness, not a mode split. The `validation_context` parameter (`RESEARCH` | `SHADOW` | `LIVE`) controls soft constraint enforcement thresholds. It is an explicit input to compilation, not an implicit system mode. The lowering pipeline, canonicalization, and hashing are context-independent.

**What this is not:**

- Not adversarial protection against third-party compiler divergence
- Not designed for public ecosystem governance
- Not optimized for minimizing hash sensitivity to cosmetic changes
- Not attempting to cryptographically seal every subsystem in isolation

**What this is:** Structural self-discipline so that six months from now, any result can be reproduced exactly, capability semantics cannot change silently, and no strategy can be promoted without a complete evidence chain.

---

## 2. Composition Spec Schema (Authoring Form)

The Composition Spec is a JSON document that a researcher creates (via UI or text editor) to describe a strategy. It contains human-readable abstractions that are compiled into framework-native artifacts.

### 2.1 Top-Level Structure

```json
{
  "composition_id": "uuid or content-hash",
  "display_name": "Human-readable name",
  "description": "What this strategy does and why",
  "archetype_tags": ["trend_following", "multi_timeframe"],
  "version": "1.0.0",
  "target_engine_version": "1.7.0",
  "min_engine_version": "1.7.0",
  "target_instrument": "BTCUSDT",
  "target_variant": "perp",

  "indicator_instances": [...],
  "entry_rules": [...],
  "exit_rules": [...],
  "gate_rules": [...],
  "execution_params": {...},
  "metadata": {...}
}
```

### 2.2 Archetype Tags

Archetype tags are informational labels. They do NOT affect compilation or runtime behavior. They exist for:
- Research panel organisation and filtering
- Fleet exposure analysis (how many pods run trend-following vs mean-reversion)
- Preset discovery

**Canonical archetype tag vocabulary:**

| Tag | Description | Example |
|-----|-------------|---------|
| `trend_following` | Enter in trend direction, ride continuation | MACD Confluence |
| `mean_reversion` | Fade extremes, trade back to equilibrium | Band Oscillator |
| `breakout` | Enter on range expansion from compression | Volatility Breakout |
| `structure_fade` | Counter-trend at proven S/R levels | Structure Fade |
| `momentum_exhaustion` | Fade overextension for snap-back | Momentum Ignition |
| `carry` | Capture funding/yield, not direction | Funding Rate Arb |
| `regime_shift` | Trade correlation/vol regime transitions | Correlation Shift |
| `vol_transition` | Trade vol-of-vol instability | Vol-of-Vol |
| `recovery` | Enter after deep drawdown for mean-reversion | Drawdown Recovery |
| `squeeze` | Trade liquidation cascades | Liquidity Squeeze |

Tags are an open vocabulary — new tags can be added without contract changes. Multiple tags per composition are allowed.

**Tag format:** All archetype tags MUST be lowercase ASCII alphanumeric with underscores only, matching regex `^[a-z][a-z0-9_]*$`. Tags are case-sensitive. `"trend_following"` is valid; `"Trend_Following"`, `"trend-following"`, `"trendFollowing"` are invalid and MUST be rejected at schema validation.

### 2.3 Indicator Instances

Indicator instances in the Composition Spec are identical to the Strategy Framework schema with one addition: **role tags**.

```json
{
  "label": "macro_3d",
  "role": "filter",
  "indicator_id": "macd_tv",
  "timeframe": "3d",
  "parameters": {"fast": 12, "slow": 26, "signal": 9},
  "outputs_used": ["slope_sign"],
  "data_source": "BAR",
  "bar_provider": "EXCHANGE_CANDLES"
}
```

**Role tags are compile-time only.** The lowering pipeline strips them from the resolved config. The engine never sees roles.

**Available role tags:**

| Role | Purpose | Cardinality | Compile-time behavior |
|------|---------|-------------|----------------------|
| `filter` | Conditions that must be true for entry (AND) | 0-N | Expanded into condition_groups |
| `entry_signal` | Triggers entry when conditions met | 1+ per direction | Expanded into entry_rules conditions |
| `exit_signal` | Triggers signal exit | 1+ per direction | Expanded into exit_rules conditions |
| `gate` | Blocks entry when condition is true (veto) | 0-N | Expanded into gate_rules |
| `sizing_input` | Provides input for position sizing | 0-1 | Referenced in execution_params |
| `risk_overlay` | Provides additional exit/risk logic | 0-N | Expanded into exit_rules |
| `fallback` | Secondary instance when primary returns None | 0-N | See §2.8 |

Role tags are advisory. A researcher MAY assign roles however they wish. The only enforcement is at compile time: the lowering pipeline uses role tags to generate condition groups and validate structure. An instance with no role tag is valid but will not be auto-wired into any conditions — it must be explicitly referenced in entry/exit rules.

### 2.4 Entry Rules (Authoring Form)

Entry rules in the Composition Spec support human-readable condition groups:

```json
{
  "label": "long_entry",
  "direction": "LONG",
  "evaluation_cadence": "5m",
  "entry_type": "MARKET",
  "condition_groups": [
    {
      "label": "macro_alignment",
      "description": "All macro TF slopes must agree on direction",
      "role_condition": {
        "role": "filter",
        "filter_group": "macro",
        "output": "slope_sign",
        "operator": ">",
        "value": "0",
        "quantifier": "ALL"
      }
    },
    {
      "label": "intra_alignment",
      "description": "All intra TF slopes must agree",
      "role_condition": {
        "role": "filter",
        "filter_group": "intra",
        "output": "slope_sign",
        "operator": ">",
        "value": "0",
        "quantifier": "ALL"
      }
    }
  ],
  "conditions": [
    {
      "role": "entry_signal",
      "output": "slope_sign",
      "operator": "crosses_above",
      "value": "0"
    }
  ]
}
```

**filter_group (OPTIONAL):** Instances with the same role can be subdivided into named groups (e.g. "macro" vs "intra"). filter_group is matched against a `group` tag on indicator instances. If filter_group is omitted, the role_condition matches ALL instances with the specified role.

**Instance group tag:**
```json
{
  "label": "macro_3d",
  "role": "filter",
  "group": "macro",
  ...
}
```

### 2.5 Exit Rules (Authoring Form)

Exit rules follow the same pattern as the Strategy Framework, with role references for signal exits:

```json
[
  {
    "label": "stop_loss",
    "exit_type": "STOP_LOSS",
    "mode": "PERCENT",
    "exchange_side": true,
    "value_long_bps": 500,
    "value_short_bps": 600
  },
  {
    "label": "mtm_drawdown",
    "exit_type": "MTM_DRAWDOWN_EXIT",
    "enabled": true,
    "evaluation_cadence": "1m",
    "drawdown_bps_long": 1000,
    "drawdown_bps_short": 1200
  },
  {
    "label": "time_limit",
    "exit_type": "TIME_LIMIT",
    "enabled": true,
    "time_limit_bars": 30,
    "time_limit_reference_cadence": "1d"
  },
  {
    "label": "long_signal_exit",
    "exit_type": "SIGNAL",
    "applies_to": ["LONG"],
    "evaluation_cadence": "1d",
    "conditions": [
      {"role": "exit_signal", "output": "slope_sign", "operator": "<", "value": "0"}
    ]
  }
]
```

**Exit ordering is semantic.** The order in the authoring form defines execution priority. The lowering pipeline preserves this order.

### 2.6 Gate Rules (Authoring Form)

```json
[
  {
    "label": "chop_filter",
    "description": "Block entry during choppy markets",
    "conditions": [
      {"indicator": "chop_1d", "output": "choppiness", "operator": "<", "value": "60"}
    ],
    "exit_policy": "HOLD"
  }
]
```

**exit_policy** defines what happens when the gate closes while a position is open:
- `FORCE_FLAT`: Close position immediately at market
- `HOLD`: Hold position, re-evaluate at next gate evaluation
- `HANDOFF`: Defer to risk exits only (stop loss, MTM drawdown)

Gate exit_policy is compiled into the Strategy Framework's gate_rules with the appropriate `on_close_policy` field.

**Gate exit_policy → Framework on_close_policy mapping:**

| Composition exit_policy | Framework on_close_policy | Behavior |
|------------------------|--------------------------|----------|
| `FORCE_FLAT` | `FORCE_FLAT` | Close position immediately at market |
| `HOLD` | `HOLD_CURRENT` | Hold position, block new entries, exit via exit paths only |
| `HANDOFF` | `HANDOFF` | Defer to risk exits only (stop loss, MTM drawdown, trailing stop, time limit) |

The lowering pipeline maps composition shorthand to Framework canonical values. `HOLD` in the authoring form becomes `HOLD_CURRENT` in the resolved artifact to match Framework §2.2.3 GateExitPolicy enumeration. `gate_exit_policy` / `on_close_policy` is listed in §9 as a Framework v1.8.0 capability. HANDOFF requires Framework v1.8.0 (reserved in v1.7.0). FORCE_FLAT and HOLD_CURRENT exist in Framework v1.7.0 but the `on_close_policy` field in gate_rules is already defined there. Until the Framework implements HANDOFF, compositions using `exit_policy: HANDOFF` MUST be rejected at compile time. Compositions using only FORCE_FLAT or HOLD do not require the `gate_exit_policy` capability and may target v1.7.0.

**Precedence delegation:** The composition layer emits gate_rules, entry_rules, and exit_rules as separate arrays. It does NOT encode evaluation precedence, gate-flip interaction, or exit-vs-entry priority. All precedence semantics are governed by Strategy Framework §5.1 and §5.1.2. The lowering pipeline MUST NOT inject precedence metadata, priority fields, or evaluation-order hints into the resolved artifact.

### 2.7 Execution Params (Authoring Form)

```json
{
  "direction": "BIDIRECTIONAL",
  "leverage": "5.0",
  "position_sizing": {
    "mode": "FIXED",
    "risk_fraction_bps": 10000
  },
  "flip_enabled": false,
  "funding_model": {
    "enabled": true,
    "interval_hours": 8,
    "rate_source": "FIXED_CONSERVATIVE",
    "rate_per_interval_bps": 1,
    "credit_allowed": false
  }
}
```

### 2.8 Fallback Declarations

When a primary indicator instance may return None for extended periods (e.g. 3d timeframe), the composition can declare a fallback:

```json
{
  "fallback_bindings": [
    {
      "primary_label": "exit_3d",
      "fallback_label": "exit_1d",
      "reason": "3d bar invalidation can last up to 3 days"
    }
  ]
}
```

**Compile-time behavior:** The lowering pipeline emits explicit fallback conditions in the resolved config using the `is_present` condition primitive (see §3.4). This is NOT runtime magic — it produces concrete, visible framework conditions.

### 2.9 Metadata

```json
{
  "metadata": {
    "author": "system_owner",
    "created": "2026-02-12T10:00:00Z",
    "thesis": "Enter when short-term momentum inflects in direction of multi-TF trend alignment",
    "known_risks": ["Aligned chop produces false entries", "81-day warmup from 3d instance"],
    "triage_sensitive_params": ["slow_period", "fast_period", "stop_loss_bps"],
    "param_defaults": {"slow_period": 26, "fast_period": 12, "stop_loss_bps": 500},
    "param_bounds": {"slow_period": [18, 50], "fast_period": [8, 20], "stop_loss_bps": [100, 2000]}
  }
}
```

The `triage_sensitive_params`, `param_defaults`, and `param_bounds` fields are consumed by the Phase 5 Robustness Contract's Test 3 (parameter sensitivity). They MUST be present before triage can run.

**Phase 5 data source:** Phase 5 parameter sensitivity (Test 3) reads `triage_sensitive_params`, `param_defaults`, and `param_bounds` from the Composition Spec (`research/compositions/{composition_id}/composition.json`), NOT from the resolved artifact. The resolved artifact contains only framework-schema fields.

**Binding requirement:** Promotion artifacts MUST include `composition_spec_hash` — the SHA-256 of the canonical composition spec JSON at the time of compilation. This binds the promotion chain to the exact composition that produced the resolved artifact. If the composition spec is edited after compilation, the composition_spec_hash changes and a new compilation is required before promotion can proceed.

---

## 3. Condition Grammar

### 3.1 Condition Object Schema

Every condition in the Composition Spec is a JSON object:

```json
{
  "indicator": "<label>",        // OR "role": "<role_tag>"
  "output": "<output_name>",
  "operator": "<operator>",
  "value": "<literal_or_ref>"
}
```

When `role` is used instead of `indicator`, the lowering pipeline resolves it to a concrete indicator label.

### 3.2 Operator Set (v1)

These operators are available in v1. They map 1:1 to Strategy Framework operators:

| Operator | Syntax | Framework Equivalent | Semantics |
|----------|--------|---------------------|-----------|
| `>` | `output > value` | `GT` | Greater than (strict) |
| `<` | `output < value` | `LT` | Less than (strict) |
| `>=` | `output >= value` | `GTE` | Greater or equal |
| `<=` | `output <= value` | `LTE` | Less or equal |
| `==` | `output == value` | `EQ` | Exact equality |
| `crosses_above` | `output crosses_above value` | `CROSSES_ABOVE` | Was ≤ value, now > value |
| `crosses_below` | `output crosses_below value` | `CROSSES_BELOW` | Was ≥ value, now < value |

### 3.3 Value References

The `value` field can be:

- **Numeric literal:** `"0"`, `"0.5"`, `"100"` — compared directly
- **Indicator output reference:** `"<label>.<o>"` — cross-indicator comparison. Example: `"value": "bb_4h.upper"` means "compare this output against bb_4h's upper band output."

**Resolved format for cross-indicator references:** In the authoring form, cross-indicator references use dot notation: `"value": "label.output"`. The lowering pipeline MUST resolve these into the Strategy Framework's cross-indicator condition format. The resolved condition object replaces the `value` string with structured reference fields:

```json
{
  "indicator": "primary_label",
  "output": "primary_output",
  "operator": "GT",
  "ref_indicator": "bb_4h",
  "ref_output": "upper"
}
```

The exact field names (`ref_indicator`, `ref_output`) MUST match the Strategy Framework's cross-indicator condition schema for the target engine version. String-literal dot-notation references (e.g. `"value": "bb_4h.upper"`) MUST NOT appear in the resolved artifact. Step 1 validation MUST verify that all dot-notation value references resolve to valid indicator labels and declared outputs.

### 3.4 Future Operator Extensions (v2+)

The following operators are NOT in v1 but are designed for:

| Operator | Syntax | Archetype Need | Framework Amendment Required |
|----------|--------|---------------|----------------------------|
| `for_n_bars` | `output > value FOR 20 bars` | Volatility Breakout (#3), Correlation Shift (#7) | Yes — duration condition primitive |
| `is_present` | `is_present(label.output)` | Fallback substitution | Yes — presence-check primitive |
| `is_absent` | `is_absent(label.output)` | Fallback negation (primary unavailable) | Yes — absence-check primitive |
| `within` | `price WITHIN 0.5 ATR OF level` | Structure Fade (#4) | Yes — proximity primitive |
| `!=` / `NEQ` | `output != value` | Various | Yes — negation comparison |

Each of these requires a Strategy Framework contract amendment before they can be used. The composition system MUST NOT emulate these operators via lowering tricks. If the framework doesn't support an operator, the composition system rejects the config with an error identifying the missing framework capability.

**Critical constraint:** Adding a new operator to the composition system without a corresponding framework implementation is a contract violation. Operators are runtime semantics.

**Capability registry:** Each operator and exit type is mapped to a minimum required engine version. The lowering pipeline validates that all operators and exit types used in a composition are available in the `target_engine_version`. This registry is updated when the Strategy Framework is amended.

**Capability registry versioning:** The capability registry has a version identifier (`capability_registry_version`) and a content hash (`capability_registry_hash` — SHA-256 of the canonical JSON representation of the full registry). Both MUST be recorded in the lowering report for auditability. However, `capability_registry_hash` is EXCLUDED from the `lowering_report_semantic_hash` computation — the registry hash changes whenever any capability is added (even capabilities unrelated to the strategy), and including it would invalidate all existing promotions on every registry update. The strategy's actual capability dependencies are captured by `engine_version` in the resolved artifact and by the specific operators/exit types present in resolved conditions.

### 3.5 Expressibility Limitations (NORMATIVE)

The v1 condition grammar supports:
- **Conjunction (AND):** implicit within paths and groups
- **Disjunction (OR):** via multiple entry/exit paths (first match fires)
- **Negation:** available implicitly via inverted operators (e.g. `> 0` negated as `<= 0`). Explicit `NOT` as a boolean combinator is NOT in v1.

Complex boolean expressions (e.g. `(A AND B) OR (C AND NOT D)`) must be decomposed into multiple paths. This is a deliberate design constraint — it keeps the evaluation pipeline simple and deterministic at the cost of requiring more paths for complex logic.

**What this means for archetypes:** All 10 archetypes are expressible in the composition schema, but some may require 3-4 entry paths where a full boolean DSL would need 1. This is acceptable — path count is not a performance bottleneck, and explicit paths are more auditable than nested boolean trees.

**Extension path:** If research demonstrates that path decomposition is genuinely limiting (not merely verbose), a `NOT` combinator and/or nested boolean groups can be added in v2 via a Strategy Framework amendment.

---

## 4. Lowering Pipeline

### 4.1 Overview

The lowering pipeline is a deterministic compiler that transforms a Composition Spec into a resolved Strategy Framework artifact. Same input always produces identical output.

**Compilation inputs (all must be identical for identical output):**

| Input | Identity | Source |
|-------|----------|--------|
| Composition Spec | `composition_spec_hash` | Researcher-authored JSON |
| Target engine version | `target_engine_version` field | Composition spec |
| Capability registry | `capability_registry_version` + content | Bundled with compiler |
| Framework schema | `framework_schema_version` + content | Bundled with compiler |
| Lowering pipeline logic | `lowering_pipeline_version` | Compiler code |
| Validation context | `RESEARCH` / `SHADOW` / `LIVE` | Compilation parameter |

**Note on validation_context:** This parameter affects only soft constraint enforcement (§5.2). It does NOT affect the resolved artifact, the strategy_config_hash, or the lowering_report_semantic_hash. Two compilations of the same spec with different validation contexts produce identical resolved artifacts and identical hashes — the stricter context just fails compilation if soft constraints are violated.

**Compilation outputs:**

| Output | Identity | Deterministic? |
|--------|----------|---------------|
| Resolved Strategy Artifact | `strategy_config_hash` | Yes — byte-identical from identical inputs |
| Lowering Report | `lowering_report_semantic_hash` | Semantic hash: yes. Full hash: no (timestamp) |
| File paths | Derived from above hashes | Yes |

**Compiler refusal:** The compiler MUST refuse to compile if:
- The composition requires capabilities not present in the capability registry for the target engine version
- The framework schema does not match the target engine version
- Any canonicalization rule cannot be satisfied (e.g., unknown field type, unresolvable cross-reference)

**Implicit inputs locked to engine version:** The following are fully determined by `target_engine_version` and are not independent compilation inputs:
- Framework serialization rules (§7.1 of the Framework contract for the target version)
- Framework schema field set, optional/required classification, and default values
- Operator and exit type enumerations

These cannot change without a new engine version. A Framework amendment that changes serialization rules for an existing engine version is a breaking change requiring a version bump. The `framework_schema_version` and `framework_schema_hash` in the lowering report provide audit confirmation that the compiler used the correct schema.

**Schema and registry consistency guards:** At the start of compilation, the compiler MUST verify version-to-hash consistency for both the framework schema and capability registry:

- The compiler maintains (or loads) a mapping of `{schema_version → expected_schema_hash}` and `{registry_version → expected_registry_hash}`
- If the loaded schema's computed hash does not match the expected hash for its declared version, compilation MUST fail: `"Framework schema content has changed for version {version} without a version bump. Expected hash: {expected}. Got: {actual}."`
- The same check applies to the capability registry
- This catches the "I edited the schema/registry and forgot to bump the version" failure mode, which is the primary temporal drift adversary for a single-user system

```
Composition Spec (authoring form)
    ↓ [Lowering Pipeline]
Resolved Strategy Framework JSON (runtime form)
    + Lowering Report (traceability)
    + Strategy Config Hash (identity)
```

### 4.2 Pipeline Steps (Ordered)

The pipeline executes these steps in strict order:

**Step 1: Schema validation**
- Reject unknown fields
- Verify all required fields present
- Type-check values
- Verify all indicator labels referenced in conditions exist in indicator_instances
- Verify all role references match at least one instance
- Verify triage metadata is present and valid
- **Timeframe validation:** All timeframe strings MUST be one of the canonical values defined in Strategy Framework §6.1: `"1m"`, `"5m"`, `"15m"`, `"30m"`, `"1h"`, `"4h"`, `"12h"`, `"1d"`, `"3d"`. Sub-minute timeframes (`"100ms"`, `"1s"`, `"5s"`) valid for snapshot instances only. Non-canonical representations (e.g. `"60m"`, `"3600s"`) MUST be rejected, not silently normalized.
- **Snapshot timeframe grammar:** Sub-minute timeframe strings MUST match the regex `^[1-9][0-9]*(ms|s)$`. No fractional units, no leading zeros, case-sensitive (lowercase only). Examples: `"100ms"` valid, `"1s"` valid, `"5s"` valid. Invalid: `"0.1s"`, `"1S"`, `"01s"`, `"1000us"`. The canonical set of valid snapshot timeframes is defined by the Low Latency Snapshot Subsystem contract.
- **Operator validation:** Every operator used in conditions MUST exist in the capability registry for the `target_engine_version`. Unknown operators cause lowering failure.
- **Exit type validation:** Every `exit_type` in exit_rules MUST be a member of the ExitType enumeration defined in the Strategy Framework for the `target_engine_version`. Unknown exit types cause lowering failure.
- **Gate exit_policy validation:** If any gate_rule includes `exit_policy: HANDOFF`, the `gate_exit_policy_handoff` capability MUST exist in the capability registry for the `target_engine_version` (requires v1.8.0+). If not present, compilation MUST fail with: `"Capability 'gate_exit_policy_handoff' (HANDOFF) requires framework v1.8.0 or later."` Gate rules using only `FORCE_FLAT` or `HOLD` do not require v1.8.0 — the `on_close_policy` field and the `FORCE_FLAT`/`HOLD_CURRENT` values exist in Framework v1.7.0.

**Step 2: Indicator catalog validation**
- Every indicator_id MUST exist in the Phase 4A/4B catalog
- Every output referenced in conditions MUST be a declared output of that indicator
- Every parameter MUST be within the indicator's declared constraints
- Warmup per referenced output is computed using the indicator's `get_warmup_bars_for_output(output_name, params)` function. Warmup computation uses ONLY outputs listed in the instance's `outputs_used` field. Outputs that exist on the indicator but are not listed in `outputs_used` are ignored. Effective warmup for an instance = `max(get_warmup_bars_for_output(output, params) for output in outputs_used)`.
- **Effective warmup (cross-instance):** The strategy's effective warmup is determined by the instance with the maximum wall-clock warmup duration in seconds. Wall-clock duration = `warmup_bars × timeframe_seconds`. If two instances have equal wall-clock duration, tie-break by larger bar count. If still equal, tie-break by lexicographic instance label (ASCII ascending). The winning instance is recorded as `dominating_instance` in the lowering report.

**Step 3: Constraint validation**
- **Validation context:** Step 3 accepts a `validation_context` parameter (`RESEARCH`, `SHADOW`, or `LIVE`). Hard constraints (§5.1) are enforced identically in all contexts. Soft constraints (§5.2) warn in `RESEARCH` and reject in `SHADOW`/`LIVE`. The lowering pipeline, canonicalization, and hashing are identical regardless of context — only the pass/fail decision on soft constraints differs.
- Timeframe ordering (if archetype demands it — see §5)
- Entry signal cardinality (at least 1 per enabled direction)
- Mutual exclusivity check (structural): The compiler verifies that LONG and SHORT entry rules do not share identical condition sets. This is a shallow structural check, not a semantic proof. The compiler does NOT attempt to prove that LONG and SHORT conditions are logically mutually exclusive for all possible indicator values. Strategies where both directions could theoretically trigger simultaneously are valid; the Framework's signal precedence rules (§5.1) resolve conflicts at runtime.
- Exit coverage: every enabled direction MUST have at least one exit path
- Gate-exit policy is valid enum value
- Stop loss is non-null for SHADOW/LIVE

**Step 4: Role expansion**
- Every `role_condition` is expanded to concrete per-instance conditions
- Every `{"role": "..."}` in conditions is resolved to `{"indicator": "<label>"}`
- If a role references multiple instances and quantifier is "ALL", one condition per instance is generated
- Step 4 produces expanded conditions in an **unspecified intermediate order**. The ordering of expanded conditions is NOT authoritative at this step. Step 8 (canonicalization) is the sole authority for final ordering of all arrays in the resolved artifact. Implementers MUST NOT rely on Step 4 output order for hashing or serialization.

**Step 5: Fallback expansion**
- Each fallback_binding produces explicit conditional paths in exit rules:
  - Primary path: `is_present(primary.output) AND primary.output <op> value`
  - Fallback path: `is_absent(primary.output) AND is_present(fallback.output) AND fallback.output <op> value`
- Both `is_present` and `is_absent` are framework-level condition primitives requiring engine >= 1.8.0. If the target engine version does not support them, fallback_bindings MUST be rejected with an error identifying the missing capability.
- `NOT` as a general boolean combinator remains excluded from v1.
- **No deduplication:** Fallback expansion generates paths independently for each fallback_binding. If multiple bindings produce structurally identical conditions, those conditions are preserved as-is. The lowering pipeline MUST NOT merge, deduplicate, or optimize generated conditions.
- **Serialization of fallback conditions:** Each fallback_binding generates **two new exit paths** (primary path and fallback path) appended to the `exit_rules` array. They are complete, self-contained exit rule objects — not conditions injected into existing exit rules. Each path contains its conditions using implicit AND semantics. The primary path (`is_present` guard) and fallback path (`is_absent` guard) are mutually exclusive by construction. Generated paths are appended after all author-declared exit paths, in the order fallback_bindings appear in the composition spec. Their position in `exit_rules` is semantic (evaluation priority).

**Step 6: Unit lowering**
- Stop loss bps → framework StopLossConfig fields (per §4.3 of the MACD contract)
- All unit conversions from authoring-friendly formats to framework-native formats

**Step 7: Strip authoring-only fields**
- Remove: `role`, `group`, `filter_group`, `description`, `display_name`, `metadata`, `thesis`, `known_risks`, `fallback_bindings`, `composition_id`
- `archetype_tags` are NOT stripped — they are valid in the Strategy Framework schema (§7.2) and pass through for fleet management
- These fields do not exist in the Strategy Framework schema

**Step 8: Canonicalize ordering (NORMATIVE)**

The following rules produce deterministic output. Every array and object in the resolved artifact MUST be ordered by exactly these rules:

**Objects:** All JSON objects sorted by key (ASCII lexicographic ascending).

**Semantic-order arrays (preserve declaration order — reordering changes behavior or hash):**
- `entry_rules` — order = evaluation priority
- `exit_rules` — order = evaluation priority
- `gate_rules` — order = evaluation priority
- `condition_groups` within each entry path — structural grouping intent
- Conditions within each entry path, exit path, gate rule, or condition_group

**Non-semantic arrays (sort deterministically):**
- `indicator_instances` — sort by `label` (ASCII ascending)
- `applies_to` — sort alphabetically
- `archetype_tags` — sort alphabetically
- `outputs_used` — sort alphabetically

**Tie-breaking for role-expanded conditions:** When role expansion generates multiple conditions, sort by `(indicator_label ASC, output ASC, operator ASC, value ASC)`. This is a total ordering.

**Null fields:** All optional fields that are absent or null MUST be explicitly serialized as `null` in the resolved artifact. No field omission allowed. This ensures two semantically identical configs with different authoring styles produce identical resolved JSON.

**Schema-pinned injection:** Null injection and default injection operate against the complete field set defined in the Strategy Framework schema for the `target_engine_version`. The compiler MUST enumerate all optional fields defined by that schema version and inject `null` for any that are absent. The compiler MUST NOT inject fields that do not exist in the target schema version.

**Default injection:** All fields with documented default values MUST be explicitly populated with those defaults if omitted in the authoring spec. Default injection occurs in Step 6 (unit lowering), BEFORE canonicalization.

**Compile-time sorting boundary:** The canonical sorting rules in Step 8 apply ONLY during compilation (lowering pipeline execution). Once the resolved artifact is emitted and written to disk, its array orderings are frozen and authoritative. Runtime code (engine, loaders, serializers) MUST NOT reorder any arrays in the resolved artifact. The Framework's prohibition on array reordering (§2.2.4) applies to the resolved artifact from the moment it is written. Re-serialization for any purpose (display, copying, transmission) MUST preserve array order exactly.

**Condition group labels:** Group labels (e.g. `"macro_alignment"`) are included in the canonical resolved artifact and contribute to the strategy_config_hash. This is intentional — labels are part of the authored structure and their inclusion provides auditability. Renaming a condition group produces a new hash and a new resolved artifact.

**Step 9: Emit resolved config**
- The output is a single JSON file containing all three layers (indicator instances, signal rules, execution params) per Strategy Framework §7.1
- This is the ONLY file the engine loads
- No authoring-only constructs may remain
- Every field must be valid per the Strategy Framework schema

**Step 10: Hash**

Compute SHA-256 of canonical JSON using the serialization rules defined in Strategy Framework §7.1 for the `target_engine_version`. The composition contract does not independently define numeric encoding — it adopts the Framework's rules exactly.

**For reference (Framework v1.7.0 rules, non-normative here — Framework is authoritative):**
- All JSON object keys sorted lexicographically (ASCII ascending)
- No whitespace (compact serialization)
- UTF-8 encoding
- Decimal values as JSON strings, normalized, no trailing zeros, no scientific notation
- Integer values as JSON number type

If the Framework amends these rules in a future version, the composition system adopts the new rules automatically for compositions targeting that version.
- Top-level field `"engine_version"` MUST be present and included in hash.
- The `engine_version` field in the resolved artifact MUST be set to the value of `target_engine_version` from the composition spec. It is NOT the compiler binary version, NOT the framework contract version, and NOT the version of the engine that will eventually run the artifact. It is the declared target engine version that the composition was compiled against. The compiler MUST refuse to emit an artifact if it cannot validate the composition against the declared target engine version's capability registry.

The resulting SHA-256 hex string (lowercase) is the `strategy_config_hash`.

### 4.3 Lowering Determinism Requirement

**Same Composition Spec → identical resolved JSON → identical hash.** Always. Across machines, across time, across implementations.

To verify: lower the same spec in two separate processes. Compare byte-for-byte. If they differ, the lowering implementation has a bug.

---

## 5. Composition Constraints and Warnings

### 5.1 Hard Constraints (reject at compile time)

| Constraint | What it prevents |
|-----------|-----------------|
| Referenced indicator_id must exist in catalog | Referencing non-existent indicators |
| Referenced output must exist for that indicator | Referencing non-existent outputs |
| Referenced parameters within catalog bounds | Invalid indicator parameters |
| At least one exit path per enabled direction | Positions with no exit |
| Stop loss non-null for SHADOW/LIVE | Unprotected live positions |
| No duplicate instance labels | Ambiguous condition references |
| applies_to is always an array | Type inconsistency |
| exit_rules order defines priority | Ambiguous exit precedence |
| Operators must exist in target framework version | Composition using unavailable operators |

### 5.2 Soft Constraints (warn in RESEARCH, reject in SHADOW/LIVE)

| Constraint | What it catches |
|-----------|----------------|
| Exit signal cadence > entry signal cadence | Asymmetric fast-in/slow-out (intentional for trend-following, suspicious for mean-reversion) |
| Warmup > 90 days | Very long warmup — verify this is intentional. *(Rationale: 90 days consumes ~25% of a typical 1-2 year backtest window, reducing effective evaluation period. Research heuristic, adjustable without contract revision.)* |
| Filter references Class C indicator (system-state) | Potential circular dependency |
| Gate with no exit_policy | Undefined behavior when gate closes during position |
| Fallback declared but `is_present` not in target framework | Fallback will not function |

### 5.3 Archetype-Specific Validation (OPTIONAL)

If the composition declares archetype tags, additional validation can run:

| Archetype | Additional checks |
|-----------|------------------|
| `trend_following` | Should have filter instances at multiple timeframes |
| `mean_reversion` | Should have a gate (regime filter) |
| `breakout` | Should have a time-limited exit |
| `carry` | Should have funding_model enabled |

These are non-blocking recommendations, not hard constraints. They appear as warnings in the lowering report.

---

## 6. Saved Artifacts

### 6.1 Artifact 1: Composition Spec (editable)

**Purpose:** The human-editable strategy description. This is what the research panel loads and saves.

**Path:** `research/compositions/{composition_id}/composition.json`

**Identity:** `composition_id` — a UUID assigned at creation time. Immutable for the lifetime of the composition. Edits to the composition spec do NOT change the composition_id. The composition_id tracks "this is the same research idea being refined" while the strategy_config_hash tracks "this is the exact config that runs."

**Mutability:** The researcher can edit this file freely. Edits change the resolved config hash but do not change the composition_id.

### 6.2 Artifact 2: Resolved Strategy Artifact (immutable)

**Purpose:** The exact Strategy Framework config that the engine loads and runs. A single JSON file containing all three layers per Strategy Framework §7.1. Byte-for-byte deterministic (file contents, not filename).

**Path:** `research/strategies/{strategy_config_hash}/resolved.json`

**Identity:** `strategy_config_hash` — SHA-256 of canonical JSON. This is the SOLE runtime identity. Two compositions with different names but identical resolved configs produce the same hash and are the same strategy. Paths use the full hash (64 hex characters for SHA-256) — no prefix truncation. This eliminates collision risk entirely. For CLI display and human reference, tools MAY display truncated hashes (e.g. first 12 characters) but file paths MUST use the full hash.

**Immutability:** Once produced, a resolved artifact is NEVER modified. If the composition changes, a new resolved artifact with a new hash is produced. Old artifacts are retained for provenance.

**Relationship to Framework:** This file IS a Strategy Framework artifact. The lowering report (§6.3) is a separate companion file for traceability — it is NOT part of the runtime artifact, is NOT loaded by the engine, and is NOT included in the strategy config hash.

**Schema boundary rule:** The resolved artifact contains ONLY fields defined in the Strategy Framework schema for the target engine version. No composition-layer fields, no Phase 5 metadata, no authoring constructs. Any data required by Phase 5 or other downstream consumers that is not in the framework schema MUST be stored in a separate artifact (composition spec, lowering report, or a dedicated metadata artifact) and referenced by promotion artifacts via content hash.

### 6.3 Artifact 3: Lowering Report (traceability)

**Purpose:** Explains every transformation the lowering pipeline performed. Enables debugging and audit.

**Path:** `research/strategies/{strategy_config_hash}/lowering_report.json`

**Contents:**

```json
{
  "composition_id": "...",
  "composition_version": "1.0.0",
  "composition_spec_hash": "sha256:...",
  "strategy_config_hash": "sha256:...",
  "timestamp": "2026-02-12T10:15:00Z",
  "compiler_version": "1.0.0",
  "lowering_pipeline_version": "1.0.0",
  "capability_registry_version": "1.0.0",
  "capability_registry_hash": "sha256:...",
  "framework_schema_version": "1.7.0",
  "framework_schema_hash": "sha256:...",
  "transformations": [
    {"step": "role_expansion", "role": "filter/macro", "instances_matched": ["macro_3d", "macro_1d", "macro_12h"]},
    {"step": "role_expansion", "role": "filter/intra", "instances_matched": ["intra_1h", "intra_30m", "intra_15m"]},
    {"step": "role_expansion", "role": "entry_signal", "resolved_to": "entry_5m"},
    {"step": "role_expansion", "role": "exit_signal", "resolved_to": "exit_1d"},
    {"step": "unit_lowering", "field": "stop_loss.value_long_bps", "from": 500, "to": {"percent_long": "0.05"}},
    {"step": "field_stripped", "fields": ["role", "group", "description", "metadata"]}
  ],
  "warnings": [],
  "effective_warmup": {"bars": 27, "timeframe": "3d", "duration_days": 81, "dominating_instance": "macro_3d"}
}
```

**Lowering report field definitions:**

| Field | Source | Purpose |
|-------|--------|---------|
| `composition_id` | Composition spec | Links report to research idea |
| `composition_version` | Composition spec `version` field | Tracks authoring version |
| `composition_spec_hash` | SHA-256 of canonical composition spec | Binds to exact authoring input |
| `strategy_config_hash` | SHA-256 of resolved artifact | Binds to exact runtime artifact |
| `timestamp` | System clock at compilation | Non-deterministic; audit only |
| `compiler_version` | Compiler binary version | Tracks which compiler built this; audit only |
| `lowering_pipeline_version` | Pipeline logic version | Tracks lowering rules version |
| `capability_registry_version` | Registry version | Tracks available capabilities |
| `capability_registry_hash` | SHA-256 of canonical registry | Detects registry content drift; audit only |
| `framework_schema_version` | `target_engine_version` | Tracks which schema was used for null/default injection |
| `framework_schema_hash` | SHA-256 of canonical schema for target version | Detects schema content drift; audit only |
| `transformations` | Pipeline execution | Step-by-step compilation trace |
| `warnings` | Constraint validation | Soft constraint violations |
| `effective_warmup` | Step 2 computation | Warmup summary with dominating instance |

**Two hashes:**
- `lowering_report_semantic_hash`: SHA-256 of the lowering report with non-deterministic and drift-sensitive fields excluded (see exclusion set below). This hash is deterministic — same input always produces same semantic hash.
- `lowering_report_full_hash`: SHA-256 of the complete lowering report including all fields.

**Semantic hash exclusion set (NORMATIVE):** The `lowering_report_semantic_hash` is computed by excluding exactly the following top-level fields from the lowering report JSON before hashing:

- `timestamp` — non-deterministic (system clock)
- `environment` (if present) — non-deterministic (host, OS, Python version)
- `compiler_version` — audit trail only; compiler correctness is verified by Proof 3, not version tracking
- `capability_registry_hash` — changes on any registry update, even unrelated capabilities; version field is sufficient
- `framework_schema_hash` — changes on any schema update, even unrelated fields; `engine_version` in the resolved artifact is sufficient

All other fields (including `lowering_pipeline_version`, `capability_registry_version`, `framework_schema_version`, `composition_spec_hash`, `transformations`, `warnings`, `effective_warmup`, `composition_id`, `composition_version`, `strategy_config_hash`) are INCLUDED in the semantic hash.

**Rationale for exclusion choices:** The excluded fields are either non-deterministic (timestamp, environment) or over-coupled (compiler_version, registry hash, schema hash). Over-coupled means: changing the field invalidates promotions for strategies unaffected by the change. The included fields capture the actual compilation logic and its inputs. If `lowering_pipeline_version` changes, the semantic hash changes — this is correct because pipeline logic changes may affect output. If `capability_registry_version` changes, the semantic hash changes — this is correct because the version is a lightweight identifier, not a content hash of the entire registry.

**Intentional inclusion: `composition_version`.** A version bump by the researcher is a declaration that the composition has been revised, even if the revision does not change the resolved artifact. This produces a new `lowering_report_semantic_hash`, which is correct: the evidence trail should distinguish between compilations from different composition versions.

**Intentional inclusion: `lowering_pipeline_version`.** A change to the pipeline version signals that compilation logic may have changed. Even if the new version produces identical resolved artifacts, the evidence trail should record that a different pipeline was used. This is conservative — it may produce unnecessary promotion resets after pure refactors. The cost is low for a single-user system. If this becomes friction, the field can be moved to the exclusion set via contract revision.

This is an exhaustive exclusion list. Adding new non-deterministic fields to the lowering report REQUIRES a contract revision to update this exclusion set (see §11.1).

**Enforcement:** The compiler MUST compute `lowering_report_semantic_hash` by excluding exactly and only the fields listed above. No additional exclusions. No omission of listed exclusions. Any deviation is a contract violation and produces an invalid semantic hash. Changes to the exclusion set require a contract revision per §11.1.

**Compiler obligation:** The compiler MUST compute `lowering_report_semantic_hash` by excluding exactly and only the fields listed above. No additional fields may be excluded. No listed field may be included. Any deviation from this exclusion set is a contract violation requiring a contract revision, not an implementation decision.

Promotion artifacts MUST bind to `lowering_report_semantic_hash`, not the full hash. This ensures re-running the compiler on the same input produces the same promotion-relevant hash even if the timestamp differs.

### 6.4 Naming Rules

- **display_name:** Human-readable, mutable, no uniqueness constraint. E.g. "Bollinger Cloud Breakout v3"
- **composition_id:** UUID, immutable, assigned at creation. Tracks the research idea.
- **strategy_config_hash:** SHA-256 of resolved config. Tracks the exact runtime behavior.

A researcher can rename "Bollinger Cloud Breakout v3" to "My Lucky Strategy" without affecting the strategy_config_hash, the promotion chain, or any backtest results. Renaming updates only the index file.

**Mutable index file:** `research/index.json` maps display names to stable identifiers:

```json
{
  "compositions": {
    "<composition_id>": {
      "display_name": "MACD Confluence v1",
      "latest_compiled_hash": "sha256:abc123...",
      "created": "2026-02-12T10:00:00Z"
    }
  }
}
```

No artifact files are moved or renamed when a strategy is renamed.

**index.json constraints:**

1. **Runtime exclusion:** The engine, loaders, and Phase 5 runners MUST NOT read index.json. It is a UI/research-panel convenience file only.
2. **Atomic updates:** Updates to index.json MUST use atomic write semantics (write to temp file, fsync, rename). Partial writes are not permitted.
3. **Concurrency:** If multiple compilations run concurrently, last-writer-wins with monotonic `updated_at` timestamp per entry. No cross-entry consistency guarantee is required — each composition entry is independent.
4. **Rebuild:** index.json is fully reconstructible from the set of composition specs and resolved artifacts on disk. Loss of index.json is recoverable; loss of artifacts is not.

---

## 7. Promotion Workflow

Promotion acts on the **resolved artifact hash**, not the composition name.

### 7.1 Lifecycle States

```
DRAFT → COMPILED → TRIAGE_PASSED → BASELINE_PLUS_PASSED → SHADOW_VALIDATED → LIVE_APPROVED
```

- **DRAFT:** Composition exists but has not been compiled.
- **COMPILED:** Lowering pipeline has run, resolved artifact and report exist.
- **TRIAGE_PASSED:** Phase 5 Triage Filter passed on this resolved hash + dataset hash.
- **BASELINE_PLUS_PASSED:** Phase 5 Baseline-Plus passed.
- **SHADOW_VALIDATED:** Shadow trading confirms live-compatible behavior.
- **LIVE_APPROVED:** System owner approves for live trading.

**State derivation:** Lifecycle state is derived from the set of promotion artifacts present for a given `strategy_config_hash`. A strategy is in state X if and only if a valid promotion artifact exists for tier X with a matching `strategy_config_hash`. The highest tier with a PASS artifact determines the current state. If no promotion artifacts exist: COMPILED (if resolved artifact exists) or DRAFT. State is not stored as a mutable variable — it is computed on demand from the promotion artifact set. This ensures state cannot drift from the actual evidence.

**Two-dimensional state display:** When presenting lifecycle state for a composition, the UI or CLI MUST evaluate two dimensions:

1. **Evidence state:** The highest promotion tier with a PASS artifact for the composition's `latest_compiled_hash`. Derived from promotion artifacts as specified above.

2. **Binding state:** Whether the current composition spec on disk still corresponds to the promoted artifact. To check:
   - Compute `composition_spec_hash` of the current composition spec
   - Compare against the `composition_spec_hash` recorded in the promotion artifact for the `latest_compiled_hash`
   - If they match: the current spec is the promoted spec. Display state normally (e.g., `"TRIAGE_PASSED"`).
   - If they differ: the composition has been edited since promotion. Display state as stale (e.g., `"TRIAGE_PASSED (stale — composition edited since promotion)"`). Require recompilation and re-promotion before the edited spec can advance to promotion-gated operations (shadow deployment, live approval).

The binding check is a display-time and gating concern, not a compilation concern. The compiler does not check promotion state. The UI, CLI, or Phase 5 runner checks binding state before allowing promotion-gated operations.

Multi-dataset and multi-run semantics (e.g., whether TRIAGE_PASSED requires passing on all required datasets or any single dataset) are defined by the Phase 5 Robustness Contract, not this contract.

### 7.2 Promotion Artifact

Each promotion event produces an immutable artifact:

```json
{
  "strategy_config_hash": "sha256:...",
  "composition_id": "...",
  "composition_spec_hash": "sha256:...",
  "tier": "TRIAGE",
  "result": "PASS",
  "timestamp": "2026-02-12T10:30:00Z",
  "dataset_hash": "sha256:...",
  "runner_hash": "sha256:...",
  "lowering_report_semantic_hash": "sha256:..."
}
```

**Path:** `research/promotions/{strategy_config_hash}/{tier}_{dataset_hash[:12]}.json`

The path is deterministic: given a strategy config hash, tier, and dataset hash, the promotion artifact path is uniquely determined. The `dataset_hash[:12]` prefix is used for ergonomic file naming only — the full `dataset_hash` is stored inside the artifact and is authoritative.

**Validation when target path exists:** When the promotion writer (Phase 5 runner or CLI tool) attempts to write a promotion artifact and the target file already exists:

1. Parse the existing file as JSON. If parse fails (corrupted/truncated): HALT with error. Manual recovery required.
2. Compare `dataset_hash` fields. If they differ: HALT with prefix collision error identifying both full hashes.
3. Compare all fields except `timestamp`. If all match: existing artifact is retained, write is skipped (semantic idempotency per above).
4. If `dataset_hash` matches but other non-timestamp fields differ: HALT with determinism violation error.

**Lowering report binding:** Promotion artifacts bind to `lowering_report_semantic_hash` (not full hash), ensuring re-compilation without semantic changes does not invalidate promotion.

**Composition spec binding:** Promotion artifacts bind to `composition_spec_hash`, ensuring the exact authoring input that produced the resolved artifact is recorded. If the composition spec is edited, `composition_spec_hash` changes and a new compilation is required before promotion.

**`runner_hash` definition:** The content and computation of `runner_hash` is defined by the Phase 5 Robustness Contract (PHASE5_ROBUSTNESS_CONTRACT §9.4). The composition contract requires its presence in promotion artifacts but does not define its scope. Implementers MUST consult the Phase 5 contract for the authoritative definition of what is included in `runner_hash`.

**No overwrites:** Promotion artifacts are immutable. A promotion artifact for the same (strategy_config_hash, tier, dataset_hash) triple is written once and never modified.

**Promotion artifact serialization:** Promotion artifacts use the same canonical JSON serialization rules as the Strategy Framework (sorted keys, compact, UTF-8).

**Semantic idempotency:** Re-running a Phase 5 tier on the same (strategy_config_hash, dataset_hash, runner_hash) triple produces a promotion artifact with identical content except for `timestamp`. If a promotion artifact already exists at the target path, the system verifies that all fields except `timestamp` match the existing artifact. If they match, the existing artifact is retained (no overwrite). If they differ, the system halts with an error — this indicates a determinism violation in the Phase 5 runner.

### 7.3 Promotion Rules

1. Promotion is attached to `strategy_config_hash`, not `display_name` or `composition_id`.
2. If the composition is edited after promotion, the resolved hash changes and promotion resets to COMPILED.
3. No tier may be skipped. TRIAGE_PASSED is required before BASELINE_PLUS can run.
4. A promotion artifact is immutable once created.
5. Resolved artifact and lowering report MUST be produced atomically from a single compilation run. If the composition spec is edited after compilation, the resolved hash changes and all prior promotion artifacts for that hash remain valid (they reference the old hash). Promotion MUST NOT be initiated against a resolved hash unless the current composition spec, when recompiled, produces that same hash.

**Compilation commit protocol:**

1. **Immutability check:** If `research/strategies/{strategy_config_hash}/` already exists:
   a. Read the existing `resolved.json` and compute its SHA-256
   b. If the hash matches the newly compiled artifact: skip the write, log "artifact set already exists," and continue. The existing set is valid.
   c. If the hash differs: **HALT with a determinism violation error.** Two compilations that produce the same `strategy_config_hash` must produce identical file content. A mismatch indicates a canonicalization or hashing bug.
   d. The compiler MUST NOT overwrite, replace, or delete existing final-path artifacts.

2. **Atomic directory commit (new artifact):**
   a. Create temp directory: `research/strategies/.tmp/{strategy_config_hash}/`
   b. Write resolved artifact to `research/strategies/.tmp/{strategy_config_hash}/resolved.json`
   c. Write lowering report to `research/strategies/.tmp/{strategy_config_hash}/lowering_report.json`
   d. fsync both files and the temp directory
   e. Atomically rename temp directory to final path: `research/strategies/{strategy_config_hash}/`
   f. Update index.json (atomic per §6.4)

3. **Failure cleanup:** If any step fails before the directory rename (step 2e), remove the temp directory entirely. No partial artifact sets may exist at final paths.

4. **Platform note:** Directory rename is atomic on POSIX (`rename(2)`) when source and target are on the same filesystem. On non-POSIX systems where atomic directory rename is unavailable, fall back to: write both files to temp paths within the final directory, rename each file individually, then write a commit marker file (`research/strategies/{strategy_config_hash}/.committed`). Define "artifact set exists" as "commit marker exists." Tooling MUST NOT treat an artifact set as valid unless the commit marker (or atomically-renamed directory) is present.

5. **Promotion gating:** Promotion MUST NOT be initiated until the artifact set is verified as complete (directory exists with both files, or commit marker present) and resolved.json content hash matches the expected `strategy_config_hash`.

---

## 8. Preset Library

### 8.1 What Presets Are

Presets are Composition Specs for known, validated strategies. They serve as:
- Starting points for research (load and modify)
- Test vectors for the lowering pipeline (known input → known output)
- Reference implementations proving the composition system is general enough

### 8.2 Required Presets

| Preset | Archetype | Proof it provides |
|--------|-----------|------------------|
| MACD Confluence | trend_following | Lowered config must be semantically and behaviorally equivalent to MACD v1.7.0 reference |
| DBMR | breakout | Lowered config is valid Strategy Framework JSON |

**MACD Preset Equivalence — Migration Note:**

The MACD v1.7.0 §11.2 reference hash was computed before the composition system existed and uses a pre-composition artifact format. The composition system's canonical defaults, null serialization, and ordering rules produce a structurally different (but semantically identical) resolved JSON. Byte-identical hash equivalence is not possible.

**Proof requirement (revised):** The MACD preset, when lowered through the composition pipeline, produces a resolved Strategy Framework JSON that:
1. Is valid per Framework schema
2. Contains identical indicator instances, conditions, exit rules, and execution params as the v1.7.0 reference (semantic equivalence)
3. When loaded by the engine, produces identical trade decisions on the same dataset (behavioral equivalence)

The composition-produced hash becomes the new canonical reference hash. The v1.7.0 reference hash is deprecated. A one-time migration artifact documents the correspondence.

### 8.3 Cross-Archetype Mixing Test

To prove the palette is real, the following composition must compile without engine edits:

**Test composition:** "MACD macro filters + Bollinger entry trigger + ATR trailing exit"
- Filter instances: MACD at 1d, 12h (role: filter, group: macro)
- Entry signal: Bollinger percent_b at 1h (role: entry_signal)
- Exit signal: ATR-based trailing stop (role: risk_overlay)
- Gate: Choppiness < 60 at 4h (role: gate)

This must lower to a valid Strategy Framework JSON and produce a deterministic hash. It does NOT need to be profitable — it needs to be expressible.

---

## 9. What the Framework Must Support (v1.8.0 Amendment Inputs)

The composition contract identifies these required framework capabilities that do not exist in v1.7.0:

| Capability | Why needed | Which compositions need it |
|-----------|-----------|---------------------------|
| `MTM_DRAWDOWN_EXIT` exit type | Capital protection via peak-to-trough drawdown | Any composition using MTM drawdown |
| Per-output warmup: `get_warmup_bars_for_output(output, params)` | slope_sign needs slow_period+1, macd_line needs slow_period | Any composition referencing derived outputs |
| `is_present(label.output)` condition primitive | Explicit fallback when primary indicator returns None | Compositions with long-timeframe exit triggers |
| `is_absent(label.output)` condition primitive | Negation of presence check for fallback paths | Compositions with fallback bindings |
| `gate_exit_policy_handoff` (HANDOFF value for `on_close_policy`) | HANDOFF gate close behavior (defer to risk exits) | Mean-reversion, carry, any gated strategy needing HANDOFF |

These are inputs to the Strategy Framework v1.8.0 amendment. They are NOT implemented by the composition layer.

**Hard rule:** Until the framework implements a capability, compositions requiring that capability MUST be rejected at compile time with a clear error: `"Capability 'is_present' requires framework v1.8.0 or later. Target engine version is 1.7.0."`

---

## 10. Acceptance Criteria

The composition system is complete when these four proofs pass:

### Proof 1: MACD Preset Equivalence
The MACD Confluence preset, when lowered through the composition pipeline, produces a resolved Strategy Framework JSON that satisfies all three conditions:

**1a) Semantic equivalence:** The resolved artifact contains identical indicator instances (same indicator_ids, same parameters, same outputs_used, same timeframes), identical conditions (same operators, same values, same structure), identical exit rules, and identical execution params as the MACD v1.7.0 §11.2 reference. Comparison is field-by-field on semantic content, ignoring structural differences from canonicalization (field ordering, null injection, default population).

**1b) Behavioral equivalence:** When loaded by the engine and run on the reference dataset, the composition-produced artifact produces identical trade decisions (same entries, same exits, same timestamps) and identical metric hashes as the MACD v1.7.0 reference run.

**1c) Migration artifact:** A one-time migration artifact documents the correspondence between the legacy hash (MACD v1.7.0 §11.2) and the new canonical hash (composition-produced). The legacy hash is deprecated. The composition-produced hash becomes the canonical reference.

The phrase "exact same config hash" from v1.0.0 is explicitly rescinded. The legacy and composition hashes will differ due to canonicalization differences. Equivalence is semantic and behavioral, not byte-level.

### Proof 2: Cross-Archetype Mixing
The §8.3 test composition (MACD filters + Bollinger entry + ATR exit + Choppiness gate) compiles to valid Strategy Framework JSON without any engine modifications.

### Proof 3: Deterministic Lowering

**3a) Resolved JSON determinism:** The same Composition Spec, lowered in two separate processes (different machines, different timestamps), produces byte-identical resolved JSON and identical strategy_config_hash.

**3b) Semantic artifact determinism:** Given the same Composition Spec and the same composition_id, the lowering pipeline produces identical resolved JSON, identical lowering_report_semantic_hash, and identical file paths (since paths use composition_id and strategy_config_hash, not mutable names). Allowed to differ: lowering_report_full_hash (due to timestamp), index.json entries (UI metadata).

### Proof 4: Replay Determinism
A resolved Strategy Framework artifact, loaded by the engine and run on the same dataset twice, produces identical trade hashes and metric hashes. (This is a framework guarantee, not a composition guarantee — but the composition system must not break it.)

**Reproducibility requirement:** For any promoted strategy at any tier, it MUST be possible at any future time to:

1. Load `resolved.json` from `research/strategies/{strategy_config_hash}/`
2. Run against the dataset identified by the promotion artifact's `dataset_hash`
3. Produce identical trade log hashes and metric hashes as the original promotion run

This does not run automatically on every compile. But it must be possible and must pass when invoked. The promotion artifact's cryptographic bindings (`strategy_config_hash`, `dataset_hash`, `runner_hash`) provide the exact inputs needed for reproduction. If reproduction fails, the promotion is invalid.

---

## 11. Change Control

### 11.1 What Can Change Without a Contract Revision

- Adding new archetype tags
- Adding new presets to the preset library
- Adding soft constraint warnings
- Adding fields to the lowering report, PROVIDED the new field is either (a) non-deterministic AND added to the semantic hash exclusion list via contract revision, or (b) deterministic and therefore included in the semantic hash (which changes the semantic hash for all future compilations — existing promotions remain valid for their recorded hash). In practice: adding a new deterministic field does not require a contract revision but DOES change the semantic hash. Adding a new non-deterministic field REQUIRES a contract revision to update the exclusion set.
- Adding metadata fields to the Composition Spec

### 11.2 What Requires a Contract Revision

- Adding new operators to §3.2
- Adding new role types to §2.3
- Changing the lowering pipeline step order
- Changing the artifact naming scheme
- Changing promotion rules

### 11.3 What Requires a Strategy Framework Amendment (NOT a composition contract change)

- Adding new exit types
- Adding new condition primitives (is_present, for_n_bars, within)
- Adding new execution primitives (flip, scale_in, limit_entry)
- Changing evaluation semantics
- Changing warmup computation

The composition system is a consumer of framework capabilities, not a producer.

---

## Appendix A: Structural Demands Matrix

From the 10 strategy archetypes analysis, these are the structural capabilities required:

**Core primitives (needed by 3+ archetypes):**
- Compound AND/OR conditions
- Multi-timeframe indicator instances
- Diagnostic probes as signal triggers
- Gate-exit policy
- Duration conditions (v2)
- Trailing stops (adaptive)
- Slope/derivative computation
- Crosses conditions
- Vol-targeted sizing

**Extension points (needed by 1-2 archetypes, design for but build later):**
- Position flip (close + open atomic)
- Limit orders
- Scale-in / gradual entry
- Time-limited exits
- R:R pre-computation at signal time
- Cost/income breakeven calculation
- Relative comparison between indicator outputs

Each extension point requires a Strategy Framework amendment before it can be used in a composition.

---

## Amendment Log

### v1.0.0 (2026-02-12)

Initial draft.

### v1.0.0 → v1.1.0 (2026-02-12)

Amendments from adversarial review (ChatGPT, 2 passes, 24 issues, 23 accepted, 1 rejected):

| Issue | Amendment | Section |
|-------|-----------|---------|
| P1-1 | Complete canonicalization rules: semantic/non-semantic enumeration, null serialization, default injection, tie-breaking | §4.2 Step 8 (rewrite) |
| P1-2 | MACD hash equivalence → semantic + behavioral equivalence; migration artifact | §8.2 (rewrite) |
| P1-3 | Normative ban on authoring fields in resolved artifact; engine rejects unknowns | §1.3 |
| P1-4 | Expressibility limitations documented; path decomposition for complex logic | §3.5 (new) |
| P1-5 | Composition emits no precedence metadata | §2.6 |
| P1-6 | Capability registry; min_engine_version field; operator/exit validation | §2.1, §4.2 Step 1 |
| P1-7 | Warmup uses only outputs_used, not all indicator outputs | §4.2 Step 2 |
| P1-8 | Atomic compilation; promotion validates current hash matches | §7.3 |
| P1-9 | REJECTED — instance count caps are implementation concern, not contract | No change |
| P1-10 | Null fields serialized explicitly; defaults injected before canonicalization | §4.2 Step 8 (via P1-1) |
| P1-11 | Unknown exit types cause lowering failure | §4.2 Step 1 |
| P1-12 | Determinism invariant — no stochastic components | §1.3 |
| P2-1 | Resolved artifact = single Framework JSON; lowering report is companion | §6.2, §4.2 Step 9 |
| P2-2 | condition_groups moved to semantic-order arrays | §4.2 Step 8 (via P1-1) |
| P2-3 | Numeric serialization rules adopted from Framework verbatim | §4.2 Step 10 (rewrite) |
| P2-4 | `!=` / NEQ removed from v1 operator set; moved to extensions | §3.2, §3.4 |
| P2-5 | Two-hash lowering report; promotion binds to semantic hash | §6.3 |
| P2-6 | composition_id is UUID-only, not content-hash | §6.1 |
| P2-7 | Paths use composition_id and hash, not slug; mutable index for names | §6 (rewrite) |
| P2-8 | archetype_tags not stripped; pass through to resolved artifact | §4.2 Step 7 |
| P2-9 | Complete semantic/non-semantic enumeration | §4.2 Step 8 (via P1-1) |
| P2-10 | Timeframe strings constrained to Framework canonical forms | §4.2 Step 1 |
| P2-11 | Proof 3 split into 3a (resolved JSON) and 3b (artifact set) | §10 |
| P2-12 | "Byte-for-byte" applies to file contents, not filename | §6.2 |

**Awaiting**: Convergence assessment / Framework v1.8.0 amendment

**Status**: FINAL — Convergence reached after 4 adversarial rounds (58 issues, 57 accepted, 1 rejected)

### v1.4.0 → v1.5.0 (2026-02-12)

Amendments from adversarial review round 4 — convergence check (ChatGPT, 11 issues, all accepted, 0 rejected):

| Issue | Amendment | Section |
|-------|-----------|---------|
| R4-1 | `validation_context` as explicit compiler input; graduated strictness clarified | §1.4, §4.1 |
| R4-2 | Serialization rules explicitly declared as function of framework_schema_version | §4.1 |
| R4-3 | composition_version intentionally semantic; rationale documented | §6.3 |
| R4-4 | Gate exit_policy rejection explicitly in Step 1 validation | §4.2 Step 1 |
| R4-5 | Promotion serialization rules; semantic idempotency; mismatch halts | §7.2 |
| R4-6 | lowering_pipeline_version inclusion rationale documented | §6.3 |
| R4-7 | Fallback insertion rule: separate exit paths, ordering specified | §4.2 Step 5 |
| R4-8 | Atomic rename: platform-agnostic wording | §7.3 |
| R4-9 | runner_hash: cross-reference to Phase 5 contract | §7.2 |
| R4-10 | Warmup warning threshold rationale and configurable constant note | §5.2 |
| R4-11 | Semantic hash exclusion compiler obligation guard | §6.3 |

**Awaiting**: Implementation handoff to Ralph

### v1.5.1 → v1.5.2 (2026-02-12) — CROSS-CONTRACT ALIGNMENT PATCH

Gate field naming corrected to match Strategy Framework v1.7.0 contract (§2.2.3):

| Fix | v1.5.1 (incorrect) | v1.5.2 (corrected) | Rationale |
|-----|---------------------|---------------------|-----------|
| Resolved artifact field name | `on_close_action` | `on_close_policy` | Framework §2.2.3 defines `GateRule.on_close_policy` |
| "Hold" enum in resolved artifact | `HOLD` | `HOLD_CURRENT` | Framework §2.2.3 GateExitPolicy uses `HOLD_CURRENT` |
| §2.6 mapping table output column | `on_close_action` column | `on_close_policy` column | Alignment |
| §9 capability table | `gate_exit_policy` (all values) | `gate_exit_policy_handoff` (HANDOFF only) | FORCE_FLAT and HOLD_CURRENT already exist in Framework v1.7.0; only HANDOFF requires v1.8.0 |
| §4.2 Step 1 gate validation | Rejects all exit_policy if capability missing | Rejects only HANDOFF if v1.8.0 capability missing | FORCE_FLAT/HOLD can target v1.7.0 |

The authoring form retains `exit_policy: HOLD` as researcher-friendly shorthand. The lowering pipeline maps to Framework canonical values.

### v1.5.0 → v1.5.1 (2026-02-12) — FINAL PATCH

Amendments from adversarial review round 5 (ChatGPT, 6 issues, 5 accepted, 1 rejected):

| Issue | Amendment | Section |
|-------|-----------|---------|
| R5-1 | Atomic directory-swap commit protocol; commit marker fallback for non-POSIX | §7.3 |
| R5-2 | Immutability check on recompilation; REPLACE_EXISTING rescinded; determinism violation detection | §7.3 |
| R5-3 | Two-dimensional state display (evidence state + binding state); stale composition detection | §7.1 |
| R5-4 | Promotion writer actor corrected; edge case validation table for existing artifacts | §7.2 |
| R5-5 | Schema and registry hash-vs-version consistency guards at compilation start | §4.1 |
| R5-6 | REJECTED — Phase 5 evolution attribution over-couples composition to Phase 5 versioning | — |

### v1.4.0 → v1.5.0 (2026-02-12) — FINAL

Amendments from adversarial review round 4 / convergence check (ChatGPT, 11 issues, 10 accepted, 1 rejected):

| Issue | Amendment | Section |
|-------|-----------|---------|
| R4-1 | `validation_context` as explicit compilation parameter; graduated strictness clarified | §1.4, §4.1, §4.2 Step 3 |
| R4-2 | Serialization rules, schema, enumerations locked to engine_version; documented as implicit inputs | §4.1 |
| R4-3 | `composition_version` semantic inclusion rationale documented | §6.3 |
| R4-4 | Gate exit_policy explicit compile-time rejection in Step 1 | §4.2 Step 1 |
| R4-5 | Promotion artifact serialization rules; semantic idempotency defined | §7.2 |
| R4-6 | `lowering_pipeline_version` semantic inclusion rationale documented | §6.3 |
| R4-7 | Fallback conditions generate new exit paths; serialization and ordering specified | §4.2 Step 5 |
| R4-8 | Atomic rename cross-platform guidance | §7.3 |
| R4-9 | REJECTED — runner_hash scope defined by Phase 5 contract | — |
| R4-10 | Warmup warning threshold rationale documented | §5.2 |
| R4-11 | Semantic hash exclusion enforcement guard added | §6.3 |

### v1.3.0 → v1.4.0 (2026-02-12)

Reconciliation with "Always-On Deterministic Spine" proposal (ChatGPT/system owner collaboration). No adversarial round — structural integration of operating model.

| Amendment | Section |
|-----------|---------|
| Operating model: always-on determinism, no research/deployment split | §1.4 (new) |
| Compilation input/output model with explicit input table | §4.1 (expanded) |
| Compiler refusal conditions enumerated | §4.1 |
| `composition_spec_hash` added to lowering report | §6.3 |
| `compiler_version` added to lowering report (audit, excluded from semantic hash) | §6.3 |
| `framework_schema_version` added to lowering report | §6.3 |
| `framework_schema_hash` added to lowering report (audit, excluded from semantic hash) | §6.3 |
| Lowering report field definitions table | §6.3 |
| Semantic hash exclusion set expanded with rationale for each exclusion | §6.3 |
| Promotion artifact path supports multiple datasets per tier | §7.2 |
| Promotion artifacts declared immutable and idempotent | §7.2 |
| Promotion path moved under `research/promotions/` | §7.2 |
| Proof 4 strengthened: reproducibility requirement for any promoted strategy at any time | §10 |

### v1.2.0 → v1.3.0 (2026-02-12)

Amendments from adversarial review round 3 (ChatGPT, 12 issues, all accepted, 0 rejected):

| Issue | Amendment | Section |
|-------|-----------|---------|
| R3-1 | Null/default injection pinned to Framework schema for target engine version | §4.2 Step 8 |
| R3-2 | Cross-indicator references lowered to structured format; dot-notation banned from resolved artifact | §3.3 |
| R3-3 | Mutual exclusivity downscoped to structural check; no proof obligation | §4.2 Step 3 |
| R3-4 | Warmup dominance: max wall-clock seconds, tie-break bars then label | §4.2 Step 2 |
| R3-5 | Condition group labels explicitly semantic (hash-contributing) | §4.2 Step 8 |
| R3-6 | Snapshot timeframe regex: `^[1-9][0-9]*(ms\|s)$` | §4.2 Step 1 |
| R3-7 | `capability_registry_hash` excluded from semantic hash; kept for audit | §3.4, §6.3 |
| R3-8 | `composition_version` normatively required in lowering report | §6.3 |
| R3-9 | Serialization delegated to Framework §7.1 for target version | §4.2 Step 10 |
| R3-10 | No deduplication of fallback-generated conditions | §4.2 Step 5 |
| R3-11 | Gate exit_policy → Framework on_close_policy mapping table (corrected in v1.5.2 from on_close_action) | §2.6 |
| R3-12 | Archetype tags: `^[a-z][a-z0-9_]*$` regex enforced | §2.2 |

### v1.1.0 → v1.2.0 (2026-02-12)

Amendments from adversarial review round 2 (ChatGPT, 11 issues, all accepted, 0 rejected):

| Issue | Amendment | Section |
|-------|-----------|---------|
| R2-1 | Proof 1 rewritten: semantic + behavioral + migration; "exact hash" rescinded | §10 Proof 1 (rewrite) |
| R2-2 | Fallback uses `is_absent` not `NOT is_present`; `is_absent` added to extensions and §9 | §4.2 Step 5, §3.4, §9 |
| R2-3 | Promotion artifact field renamed to `lowering_report_semantic_hash` | §7.2 (JSON fix) |
| R2-4 | Full hash in all artifact paths; no prefix truncation | §6.2, §6.3, §7.2 |
| R2-5 | Phase 5 reads composition spec; promotion binds `composition_spec_hash` | §2.9, §7.2 |
| R2-6 | Schema boundary rule: resolved artifact = framework fields only | §6.2 |
| R2-7 | Step 4 intermediate order unspecified; Step 8 sole sort authority | §4.2 Step 4 |
| R2-8 | Lifecycle state derived from promotion artifacts, not stored mutably | §7.1 |
| R2-9 | Compile-time sorting boundary: runtime must not reorder | §4.2 (new rule) |
| R2-10 | Capability registry versioned and hashed; in lowering report | §3.4, §6.3 |
| R2-11 | Snapshot subsystem scoped as transitive dependency | Header |
| — | Semantic hash exclusion set enumerated normatively | §6.3 |
| — | Change control tightened for lowering report fields | §11.1 |
| — | engine_version = target_engine_version normatively | §4.2 Step 10 |
| — | index.json: runtime exclusion, atomic writes, concurrency, rebuild | §6.4 |
| — | Compilation commit protocol specified | §7.3 |
| — | Promotion artifact binds composition_spec_hash | §7.2 |
