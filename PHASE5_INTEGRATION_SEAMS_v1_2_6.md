# PHASE 5

## INTEGRATION SEAM CONTRACT

### Semantics-Frozen Core with Explicit Integration Seams

**Version:** 1.2.6  
**Status:** AUTHORITATIVE

**PSEUDOCODE DISCLAIMER**: All code blocks in this document are illustrative pseudocode only and are not normative. Implementation details belong in implementation guides.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | TBD | Initial draft |
| 1.1 | TBD | Added system owner definition, evidence chain, run mode transitions |
| 1.2 | TBD | Added wrapper prohibition, adapter constraints, CLI output integrity, testing requirements |
| 1.2.1 | TBD | Added shared infrastructure constraints, core version hash manifest, LIVE input handling |
| 1.2.2 | TBD | Added HALT completion semantics, configuration immutability, inspect command, failed attempt logging |
| 1.2.3 | TBD | Tightened pod identity uniqueness, defined inspect redaction and LIVE prohibition, tightened HALT and evidence-chain integrity checks, clarified corruption and compromise conditions |
| 1.2.4 | TBD | Added Research Validation Interface (Section 3.6), references PHASE5_ROBUSTNESS_CONTRACT.md |
| 1.2.5 | TBD | Added SeedHistory boundary discovery (Section 3.1.1), timestamp format requirements, canonical data hashing specification, no-assumed-dates principle |
| 1.2.6 | TBD | Added Bybit OMS reference (Section 3.7), updated canonical data hash to format-agnostic approach |

---

## 1. Purpose and Authority

This document defines the authoritative integration boundaries for the BTC Alpha system once the core has entered a Semantics-Frozen state.

Its purpose is to:

* Explicitly enumerate what is frozen and immutable.
* Explicitly define where integration is allowed.
* Prevent semantic drift during delegation and implementation.
* Enable safe, high-velocity development by downstream implementers.

This document has the same authority class as:

* SYSTEM_LAWS.md
* PHASE4A_INDICATOR_CONTRACT.md
* PHASE4B_CONTRACT_LOCKED.md

This document is itself a frozen artifact. Modifications require change control per Section 10.

Any change to frozen surfaces defined here requires explicit review and approval by the system owner. Silent deviation is a violation.

### 1.1 System Owner Definition

System Owner: For purposes of this contract, the system owner is the human principal who commissioned the system (the repository owner). Approval is granted only via signed commit to this repository with an explicit approval statement, or written confirmation in the primary project communication channel. Verbal or implicit approval is not valid. Delegation of approval authority must be explicit and documented in this repository.

### 1.2 Conflict Resolution

In case of conflict between this document and other authority documents (SYSTEM_LAWS.md, PHASE4A_INDICATOR_CONTRACT.md, PHASE4B_CONTRACT_LOCKED.md), the more restrictive interpretation applies. If ambiguity remains, the system owner must be consulted before proceeding. Implementers may not resolve conflicts by choosing the interpretation most convenient to their work.

---

## 2. Definition of the Frozen Core

The frozen core consists of all logic that defines meaning, behavior, and safety.

### 2.1 Frozen Files and Modules

The following are considered part of the frozen core and may not be modified without approval:

* `btc_alpha_phase4b_1_7_2.py` (Phase 4B v1.7.2 baseline)
* Indicator definitions and registrations (IDs 1"24)
* Diagnostic probe definitions and registrations (IDs 25"29)
* Core data types and enums:
  * `SemanticType`
  * `TypedValue`
  * `IndicatorOutput`
  * `SystemInputs`
* Core execution logic:
  * `IndicatorEngine`
  * `compute_all()` and its semantics
  * activation, warmup, eligibility, and dependency logic

### 2.2 Frozen Semantics

The following semantics are frozen:

* Indicator math and outputs
* Dependency resolution
* Warmup behavior
* State mutation rules
* Determinism guarantees
* Rounding and truncation behavior
* Run-mode failure semantics (RESEARCH, SHADOW, LIVE)

### 2.3 Bug Fix Constraints

Bug fixes are allowed only if:

1. The fix restores behavior documented in PHASE4A_INDICATOR_CONTRACT.md or PHASE4B_CONTRACT_LOCKED.md.
2. The fix does not change outputs for any inputs that previously produced valid (non-None) outputs.
3. The fix is reviewed and approved by the system owner.

A fix that changes outputs for previously valid inputs is a semantic change, not a bug fix, regardless of intent. Semantic changes require explicit approval and version increment.

### 2.4 Wrapper and Shim Prohibition

Creating wrapper functions, shim layers, or proxy objects that intercept calls to frozen core components and modify inputs or outputs is prohibited. The only permitted interaction with frozen components is direct invocation through documented seams. This prohibition includes but is not limited to:

* Monkey-patching frozen classes or methods
* Subclassing frozen classes to override behavior
* Decorators that modify frozen function behavior
* Middleware that transforms data between caller and core

---

## 3. Integration Seams (Explicit and Exclusive)

All integration with the frozen core must occur only through the seams defined below. Any integration path not explicitly defined here is forbidden.

### 3.1 Data Ingestion Seam

Adapters may supply data only in the following canonical forms:

* `candle_inputs`
* `system_inputs`
* `period_data`

Rules:

* Adapters may normalize, validate, and reorder data before submission.
* Adapters may not fabricate data to satisfy core expectations.
* Missing or malformed data must be surfaced, not patched.

The core assumes that any data passed across this seam is:

* schema-correct,
* type-correct,
* explicitly None where unavailable.

**Definition of Unavailable:**

Data is considered unavailable only if:

* The exchange did not provide it, or
* The exchange explicitly indicated it is missing, or
* The connection to the exchange failed.

Data that exists but is malformed must be surfaced as a validation error, not passed as None. Passing None for data that exists but is inconvenient to parse is a violation.

**Definition of Normalization:**

Normalization means converting between equivalent representations (e.g., basis points to decimal rate, milliseconds to seconds). Normalization must be semantically neutral. It must not change the meaning, magnitude, or sign of data. All normalization logic must be documented and deterministic. Adapters must log the raw value and normalized value at DEBUG level for audit purposes.

Adapters must not log secrets or credentials. Any field that could contain secrets must be redacted before logging.

**Normalization Documentation Requirement:**

Each adapter must maintain a normalization specification document listing:

* Every field that undergoes normalization
* The source format and target format
* The conversion formula
* Test cases demonstrating correctness

This specification must be reviewed before the adapter enters SHADOW mode.

### 3.1.1 Historical Dataset Boundary Discovery (SeedHistory + LiveExtension)

**CRITICAL PRINCIPLE**: All time bounds must be discovered from data content, never from filenames, comments, or assumed dates.

**SeedHistory Loading**:

```python
def load_seed_history(filepath: str) -> Tuple[pd.DataFrame, SeedMetadata]:
    """
    Load immutable historical dataset and discover its boundaries.
    
    Args:
        filepath: Path to historical Parquet file
    
    Returns:
        data: OHLCV DataFrame
        metadata: Discovered boundaries and content hash
    """
    data = pd.read_parquet(filepath)
    
    # Discover bounds from data itself (never assume dates from filename)
    seed_start_ts = data['timestamp'].min()
    seed_end_ts = data['timestamp'].max()
    
    # Compute canonical content hash
    content_hash = canonical_data_hash(data)  # See PHASE5_ROBUSTNESS_CONTRACT Section 8.1
    
    # Log discovered bounds (not assumed dates)
    logger.info(
        f"SeedHistory discovered: {seed_start_ts} to {seed_end_ts}, "
        f"rows={len(data)}, hash={content_hash[:16]}..."
    )
    
    return data, {
        'seed_start_ts': seed_start_ts,
        'seed_end_ts': seed_end_ts,
        'row_count': len(data),
        'content_hash': content_hash,
        'source_file': filepath  # Observational only, not used in logic
    }
```

**Live Extension Start Point**:

```python
def start_live_ingestion(seed_metadata: SeedMetadata, bar_period_seconds: int) -> pd.Timestamp:
    """
    Determine first live bar to fetch from exchange.
    
    Args:
        seed_metadata: Metadata from load_seed_history()
        bar_period_seconds: Bar period (e.g., 60 for 1-minute bars)
    
    Returns:
        Timestamp of first live bar to fetch (strictly after seed_end_ts)
    """
    live_start_ts = seed_metadata['seed_end_ts'] + pd.Timedelta(seconds=bar_period_seconds)
    
    logger.info(f"Live ingestion starting at: {live_start_ts}")
    return live_start_ts
```

**Boundary Integrity Checks**:

```python
def append_live_bar(
    seed_history: pd.DataFrame,
    live_bar: pd.Series,
    seed_end_ts: pd.Timestamp
) -> None:
    """
    Append live bar with boundary validation.
    
    Raises: BoundaryViolation if overlap or gap detected
    """
    bar_timestamp = live_bar['timestamp']
    
    # Check for overlap (duplicate timestamp)
    if bar_timestamp <= seed_end_ts:
        raise BoundaryViolation(
            f"LIVE_EXTENSION_OVERLAP: bar timestamp {bar_timestamp} "
            f"<= seed_end_ts {seed_end_ts}"
        )
    
    # Check for gap (missing bar)
    expected_timestamp = seed_end_ts + pd.Timedelta(seconds=60)  # For 1m bars
    if bar_timestamp > expected_timestamp:
        raise BoundaryViolation(
            f"LIVE_EXTENSION_GAP: expected bar at {expected_timestamp}, "
            f"first bar is {bar_timestamp}"
        )
    
    # Append bar (implementation-specific)
    # ...
```

**Filename Date Ranges (Observational Only)**:

- Filenames like `btcusdt_2018_to_2026.parquet` are for human readability
- Code MUST NOT parse filename dates
- Code MUST discover actual date range from data content via `min(timestamp)` and `max(timestamp)`
- If filename dates disagree with content dates: log WARNING but use content dates
- Example log: `"WARNING: Filename suggests 2018-2026 but data spans 2019-2025 (using data bounds)"`

**Timestamp Format Requirements**:

```python
# REQUIRED FORMAT: ISO 8601 with mandatory UTC indicator
valid_format = "YYYY-MM-DDTHH:MM:SS.fffZ"
example = "2026-02-08T12:34:56.789Z"

# Implementers MUST reject timestamps without explicit Z suffix or +00:00 offset
def validate_timestamp_format(ts_string: str) -> pd.Timestamp:
    """
    Validate and parse timestamp with mandatory UTC indicator.
    
    Raises: TimestampFormatError if invalid
    """
    if not (ts_string.endswith('Z') or '+00:00' in ts_string):
        raise TimestampFormatError(
            f"INVALID_TIMESTAMP: missing UTC indicator (Z suffix required), "
            f"got: {ts_string}"
        )
    
    # Parse with explicit UTC timezone
    ts = pd.Timestamp(ts_string, tz='UTC')
    
    # Verify millisecond precision
    if ts.microsecond % 1000 != 0:
        logger.warning(f"Timestamp {ts_string} has sub-millisecond precision, rounding")
        ts = ts.round('ms')
    
    return ts
```

**Duplicate Timestamp and Leap Second Handling**:

```python
def validate_no_duplicates(data: pd.DataFrame) -> None:
    """
    Check for duplicate timestamps.
    
    Raises: DuplicateTimestampError if found
    """
    duplicates = data[data.duplicated(subset=['timestamp'], keep=False)]
    
    if len(duplicates) > 0:
        first_dup = duplicates.iloc[0]
        raise DuplicateTimestampError(
            f"DUPLICATE_TIMESTAMP at {first_dup['timestamp']} "
            f"(indices: {duplicates.index.tolist()})"
        )

def normalize_leap_second(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Map leap seconds to nearest valid timestamp.
    
    Leap second (e.g., 23:59:60)  23:59:59.999
    """
    if ts.second == 60:
        logger.warning(f"Leap second detected at {ts}, mapping to 23:59:59.999")
        return ts.replace(second=59, microsecond=999000)
    
    return ts
```

---

### 3.2 Exchange Adapter Seam

Exchange adapters are responsible for:

* Translating exchange-specific data into canonical inputs.
* Handling exchange-specific anomalies (gaps, reorgs, rate limits).
* Enforcing monotonic timestamps and bar indices before core submission.

**Adapters MUST NOT:**

* Call internal core methods.
* Access indicator state directly.
* Catch and suppress core validation errors.
* Fabricate data or pass None for available-but-malformed data.
* Retry core calls with modified inputs after a validation failure.
* Cache or buffer core outputs and return stale data.
* Transform or filter core outputs before returning them to callers.

All adapter-to-core interaction occurs through the documented `compute_all()` interface.

**Adapter Validation Failure Handling:**

When the core returns a validation error or raises an exception:

1. The adapter must log the error at ERROR level with full context.
2. The adapter must propagate the error to its caller.
3. The adapter must not retry with "fixed" inputs.
4. The adapter must not substitute default or fallback values.

---

### 3.3 CLI Interface Seam (Frozen Surface)

The CLI is the only authorized interface to the frozen core in SHADOW and LIVE modes.

The CLI contract includes:

* Command names
* Required inputs
* Output schemas
* Determinism guarantees
* Safety prompts and confirmations

Rules:

* All commands must produce structured output (JSON or equivalent typed format).
* Identical inputs must produce identical outputs.
* No UI or dashboard may access the core directly.
* Any UI must invoke the CLI or an equivalent programmatic interface that enforces identical constraints.

**Safety Constraint:**

No CLI command or flag may bypass, skip, or weaken safety prompts, confirmation requirements, or scope restatements. Automation interfaces must enforce the same constraints as interactive interfaces. A `--yes`, `--force`, `--quiet`, `--no-confirm`, or equivalent flag that skips confirmation is explicitly forbidden.

**CLI Output Integrity:**

The CLI must not modify, summarize, filter, or transform core outputs. The CLI may wrap the raw core output in an envelope that adds non-semantic metadata (e.g., pod_id, timestamps, run_mode, health state) provided that:

1. The raw core output is included verbatim and unmodified.
2. The envelope schema is documented and deterministic.
3. The additional metadata does not affect core invocation or downstream decision logic.

### 3.4 CLI Command Skeleton (Required Before LIVE Mode)

The CLI contract is not complete until the following commands are defined with full input/output schemas:

| Command   | Purpose                                          |
|-----------|--------------------------------------------------|
| `status`  | Report pod health, run mode, and current state   |
| `run`     | Begin computation in specified mode              |
| `halt`    | Stop computation and enter HALTED state          |
| `resume`  | Resume from HALTED state (requires confirmation) |
| `replay`  | Deterministic replay from evidence chain         |
| `config`  | Display current configuration (read-only)        |
| `inspect` | Display internal state for debugging/audit (read-only, RESEARCH and SHADOW only) |

Until these commands are defined with complete schemas, the CLI seam is not frozen and the system is not ready for LIVE mode.

**CLI Command Addition Process:**

New CLI commands may be added by downstream implementers only if:

1. The command does not modify frozen core behavior.
2. The command does not bypass safety constraints.
3. The command schema is documented before implementation.
4. The command is reviewed by the system owner before use in LIVE mode.

**Debug and Test Mode Prohibition:**

No "debug", "test", "dev", or equivalent mode may exist that bypasses any constraint defined in this document. All modes must enforce all safety constraints. RESEARCH mode is for development but still enforces determinism, contract compliance, and halt semantics.

**Inspect Command Constraints:**

* `inspect` is forbidden in LIVE mode and must hard-fail if invoked in LIVE.
* `inspect` output must redact all secrets and credentials (API keys, tokens, private keys, signing material).
* `inspect` output must be deterministic for identical pod state and identical invocation parameters.

---

### 3.6 Research Validation Interface

The research layer must implement a **robustness validation gate** that filters strategy candidates before promotion to full backtesting.

**Purpose**: Early rejection of overfit/noise strategies during research iteration to minimize wasted compute on comprehensive validation.

**Interface Contract**:

```python
def run_triage_filter(
    strategy: Strategy,
    ohlcv_data: pd.DataFrame,
    config: TriageConfig
) -> TriageResult:
    """
    Runs ultra-fast plausibility filter on strategy candidate.
    
    Returns:
        TriageResult with:
        - decision: "PASS" | "FAIL"
        - reason: str (if FAIL)
        - artifacts_path: str (log directory)
        - runtime_seconds: float
    """
```

**Authority and Specification**:

- Full validation contract defined in: **`PHASE5_ROBUSTNESS_CONTRACT.md`** (authoritative)
- Test definitions, thresholds, and determinism requirements are frozen per that contract
- Infrastructure implementation is delegable, threshold tuning is not delegable

**Integration Requirements**:

- Triage filter must run **before** higher-tier validation (Baseline-Plus, Research-Grade)
- Triage results must be logged to evidence chain (if enabled in RESEARCH mode)
- Failed strategies must be archived with rejection reason
- Passed strategies must be queued for next validation tier

**Runtime Guarantees**:

- Target: 2-3 minutes wall-clock time
- Maximum: 5 minutes (hard timeout)
- Early exit on failure (average case: 90 seconds due to 70-75% rejection rate)

**Determinism Requirements**:

- All tests must be seedable and reproducible
- Logged artifacts must support exact replay and audit
- See `PHASE5_ROBUSTNESS_CONTRACT.md` Section 8 for full determinism contract

### 3.7 Exchange Adapter Seam (Bybit OMS)

**Scope**: BTCUSDT USDT Perpetuals (Linear), One-Way Mode

**Authority**: `BYBIT_OMS_CONTRACT_v1_0_0.md` (frozen, authoritative)

**Purpose**: Define order management and risk control against Bybit V5 API

**Integration Requirements**:

- All order placement goes through OMS (no direct API calls to exchange)
- OMS enforces mandatory exchange-side hard SL for all live positions
- OMS implements fail-safe close sequence per contract
- State machine transitions logged to evidence chain (if enabled)

**Frozen Decisions** (see BYBIT_OMS_CONTRACT for rationale):

- TP ladder: Client-side reduce-only limit orders (default)
- Partial TP/SL: Prohibited in v1.0
- Trailing stop: Exchange-native (default), client-side MarkPrice trailing (optional)
- Full-mode TP/SL: Market-only (Limit not supported)
- Hard SL: Mandatory for all live positions, exchange-side

**State Machines**:

- Order states: PENDING_ACK, ACTIVE, PARTIALLY_FILLED, FILLED, CANCELLED, REJECTED, EXPIRED
- Position states: FLAT, LONG, SHORT, PENDING_ENTRY, PENDING_CLOSE, FAILSAFE

**Degradation Handling**:

- WebSocket disconnect: Stop new orders, REST polling, 30s deadline
- On 30s deadline: Flatten all positions via fail-safe close, enter HALT
- Startup reconciliation: Attempt TradePlan reconstruction, place emergency SL if needed, await operator decision

See `BYBIT_OMS_CONTRACT_v1_0_0.md` for complete specification.

---

## 4. Pod Boundary Definition

A pod is the fundamental unit of isolation.

### 4.1 Pod Composition

One pod consists of:

* One `IndicatorEngine` instance
* One exchange sub-account
* One state space
* One configuration (immutable for the pod's lifetime; changes require pod restart)
* One run mode
* One evidence chain (required in SHADOW and LIVE; optional in RESEARCH)

### 4.2 Pod Isolation Rules

* Pods share no in-process state.
* Pods share no out-of-process state that influences computation.
* No cross-pod coordination exists inside the core.
* Any cross-pod logic must be external to the frozen core.

Out-of-process infrastructure is permitted (e.g., shared databases, shared object stores, shared log sinks) only if:

1. Pod data is strictly partitioned by immutable pod identifier, and
2. No pod can read any other pod's state in a way that influences its own computation or decisions, and
3. The infrastructure is not used to implement cross-pod coupling, signal propagation, or shared runtime configuration.

**External Cross-Pod Constraints:**

External systems that aggregate data from multiple pods for monitoring or reporting are permitted. External systems that feed data back into pods based on other pods' states are forbidden without explicit approval and documentation. This prohibition includes but is not limited to:

* Shared caches that influence pod behavior
* Cross-pod signal propagation
* Load balancing that transfers state
* Consensus mechanisms that couple pod decisions
* Shared configuration that can be modified at runtime
* Message queues where one pod's output becomes another pod's input

**Pod Identity:**

Each pod must have a unique, immutable identifier assigned at creation. This identifier must be included in all log entries, evidence chain records, and operator commands targeting the pod. Pod identifiers may not be reused.

Permitted pod identifier forms:

* A UUIDv4 string, or
* A ULID string, or
* A system-owner-approved deterministic format that guarantees uniqueness.

If a timestamp-based format is used, it must include a collision-resistant suffix (random nonce or monotonic counter) and must not rely on local system clock correctness.

**Configuration Immutability:**

Pod configuration is set at pod creation and cannot be modified while the pod is running. To change configuration:

1. HALT the pod.
2. Create a new pod with the new configuration.
3. The old pod's evidence chain remains intact for audit.

Hot-reloading configuration is forbidden.

### 4.3 Multi-Pod Operator Actions

Operator actions that target multiple pods must:

* Explicitly enumerate the target pods by identifier.
* Require deliberate confirmation (cannot be bypassed).
* Restate scope before execution.
* Log the action to each affected pod's evidence chain (if evidence chain is enabled for that mode).
* Execute against each pod independently (no atomic multi-pod operations).

**Multi-Pod Failure Handling:**

If a multi-pod operation fails on one pod:

1. The failure must be logged.
2. The operator must be notified which pods succeeded and which failed.
3. The operation must not automatically retry on failed pods.
4. Each pod's state must remain consistent regardless of other pods' outcomes.

---

## 5. Evidence Chain Definition

### 5.1 Purpose

A pod's evidence chain is a tamper-evident, append-only log sufficient to deterministically replay the pod's computation from any checkpoint.

### 5.2 Required Modes

* Evidence chain is mandatory in SHADOW and LIVE modes.
* Evidence chain is optional in RESEARCH mode. If enabled in RESEARCH, all requirements in this section apply.

### 5.3 Required Contents

The evidence chain must record:

* All inputs received (`candle_inputs`, `system_inputs`, `period_data`) with timestamps and bar indices
* All outputs produced (`IndicatorOutput` for each indicator, each bar)
* All computation attempts, including failed attempts with error details
* All state transitions (activation, deactivation, mode change, halt, resume)
* All operator actions targeting this pod
* All health state transitions (HEALTHY, DEGRADED, HALTED) with reasons
* All configuration in effect at each computation
* Cryptographic hash of the frozen core version in use

The "cryptographic hash of the frozen core version" must be computed from a deterministic manifest (file list + exact bytes) defined and versioned by the system owner.

**Failed Attempt Recording Requirements:**

Each failed attempt record must include:

* Deterministic digest of the exact inputs (including bar index and timestamps)
* Exception class name
* Exception message
* Component responsible (adapter, persistence, orchestration, core)
* Stack trace or equivalent diagnostic context

Failed attempt records must not omit context necessary to reproduce and diagnose the failure.

### 5.4 Integrity

Evidence chain integrity is a persistence-layer responsibility. Corruption or loss of the evidence chain is a HALT condition in SHADOW and LIVE modes. The evidence chain must be verifiable. It must be possible to detect tampering or incomplete records.

**Evidence Chain Preflight Check:**

Before each bar is processed in SHADOW and LIVE, orchestration must verify:

* Evidence chain is writable, and
* The evidence chain hash chain validates up to the latest committed record.

If verification fails, the pod must HALT before calling the core.

### 5.5 Evidence Chain Replay Requirement

It must be possible to replay any evidence chain segment and produce identical outputs. This requirement must be verified by automated testing before LIVE mode is permitted. Replay divergence is a HALT condition.

---

## 6. Run Mode Definitions and Halt Semantics

The term HALT has the following operational meaning:

* The engine enters a HALTED state.
* All computation stops immediately.
* Any in-flight operations (async writes, pending orders, network requests) must be cancelled or abandoned.
* Any attempted cancellations must be logged with outcome (cancelled, could not cancel, unknown).
* No new operations may be initiated after HALT is triggered.
* Orchestration must stop calling `compute_all()` after HALT is entered.
* The halt reason is logged with full context to the evidence chain (if enabled for that mode).
* Resumption requires explicit operator action via the CLI.
* No automatic resumption is permitted.

### 6.1 RESEARCH Mode

* HALT on any integrity, contract, or invariant violation.
* Persistence is optional.
* No recovery attempts. Fail fast.

### 6.2 SHADOW Mode

* HALT on integrity, contract, or persistence failure.
* Adapter outages log loudly and continue computation with available data.
* Adapter outages lasting more than 10 consecutive bars trigger HALT.

In SHADOW mode, the orchestration layer must not call the core with an incomplete required input set for the configured SHADOW workload. An incomplete required input set is an adapter outage event and must be handled per this section.

**Definition of Adapter Outage (SHADOW):**

An adapter outage is counted when the adapter fails to provide a complete required input set for the configured SHADOW workload for a bar close. The required input set must be explicitly defined by the orchestration layer configuration and must not vary at runtime without logging an explicit operator-approved configuration change.

### 6.3 LIVE Mode

HALT on:

* Persistence corruption
* State corruption
* Invariant violations implying semantic compromise
* Evidence chain corruption
* Recovery timeout expiration

All other failures:

* Force non-permissive regime
* Veto all orders
* Continue logging and recovery attempts
* Recovery bound: If recovery has not succeeded within 5 minutes or 10 bars (whichever comes first), the pod must HALT and require operator intervention.

In LIVE mode, the orchestration layer must not call the core with an incomplete required input set for the configured LIVE workload. An incomplete required input set must immediately:

1. Force non-permissive regime,
2. Veto all orders,
3. Transition required components to DEGRADED with reason,
4. Start the LIVE recovery bound timer.

**Definition of Bar (LIVE recovery bound):**

A bar is counted as each expected bar-close event for the configured bar interval. Time authority for bar-close scheduling is owned by orchestration.

**Definition of Recovery Success:**

Recovery is successful when:

1. All required components return to HEALTHY state, AND
2. The evidence chain is intact, AND
3. The next bar computes successfully with all required inputs present.

**Definition of State Corruption (LIVE HALT):**

State corruption includes any of the following:

* Persisted state cannot be loaded, validated, or checksum-verified.
* Persisted state bar index conflicts with incoming bar index monotonicity.
* Evidence chain indicates missing or out-of-order required records for the current bar.
* Core reports an invariant violation involving internal state consistency.

**Definition of Semantic Compromise (LIVE HALT):**

Invariant violations implying semantic compromise include any invariant violation raised by the core, and any mismatch between expected and observed determinism guarantees during replay verification.

### 6.4 Silent Degradation Prohibition

Silent degradation is forbidden in all modes. Any condition that reduces data quality, coverage, or reliability must be logged at ERROR level and reflected in health state. Specifically:

* Missing optional inputs must be logged at WARN level.
* Missing required inputs must be logged at ERROR level.
* Partial data (e.g., some indicators computed, others None due to missing dependencies) must be logged at WARN level.
* Adapter errors must be logged at ERROR level.

---

## 7. Run Mode Transitions

* A pod starts in RESEARCH mode by default.
* Transition from RESEARCH to SHADOW requires explicit operator command.
* Transition from SHADOW to LIVE requires explicit operator command with confirmation and scope restatement.
* Transition from LIVE to SHADOW or RESEARCH is always permitted without confirmation.
* Mid-session transitions are permitted only at bar boundaries.
* Run mode transitions must be logged in the evidence chain (if enabled for that mode).
* Run mode cannot be changed while the pod is in HALTED state (must resume first, then transition).

**Transition Preconditions:**

Before transitioning to SHADOW:

* Evidence chain must be configured and writable.
* Persistence layer must be configured and operational.

Before transitioning to LIVE:

* All CLI commands in Section 3.4 must be implemented and tested.
* Evidence chain replay must be verified.
* Adapter normalization specifications must be documented and reviewed.
* All required components must be in HEALTHY state.

---

## 8. Persistence and Time Authority

Persistence and time authority are explicitly external to the frozen core.

### 8.1 Persistence

* The core does not manage storage.
* Orchestration layers must define:
  * What state is persisted (at minimum: engine state, indicator states, current bar index, evidence chain state if enabled)
  * When persistence occurs (at minimum: every bar close, every state transition)
  * Recovery behavior on restart (must replay from last checkpoint using evidence chain if enabled)

Persistence failures are HALT conditions in SHADOW and LIVE modes.

**Persistence Implementation Constraints:**

* Persistence must be synchronous with bar close (no async writes that could be lost).
* Persistence must be atomic (partial writes must not corrupt state).
* Persistence must include a checksum or hash for integrity verification.
* Persistence format must be versioned to detect incompatible changes.

### 8.2 Time Authority and Timestamp Requirements

* The core does not own current time.
* All timestamps are supplied by callers.
* Market hours, clock drift, scheduling, and bar timing are orchestration concerns.
* The orchestration layer is responsible for ensuring `compute_all()` is called with correct, monotonic timestamps.

**Timestamp Constraints:**

* **Format**: ISO 8601 with mandatory UTC indicator: `YYYY-MM-DDTHH:MM:SS.fffZ` (see Section 3.1.1)
* **Precision**: Milliseconds required (`.fff` component)
* **Timezone**: UTC only (Z suffix mandatory, rejects timestamps without explicit UTC indicator)
* **Monotonicity**: Orchestration must reject bar submissions with timestamps  previous timestamp before calling core
* **Duplicate detection**: Raises error if duplicate timestamps detected (see Section 3.1.1)
* **Leap seconds**: Normalized to nearest valid timestamp (23:59:60  23:59:59.999)
* **Bounds discovery**: All time bounds discovered from `min(timestamp)` and `max(timestamp)`, never from filenames or assumptions

---

## 9. Health and Degradation Signaling

All non-core components must expose a health state:

* `HEALTHY` " operating normally
* `DEGRADED` " operating with reduced capability or data quality
* `HALTED` " not operating, requires intervention

### 9.1 Health State Rules

* Any transition to DEGRADED or HALTED must be logged at ERROR level with reason.
* A DEGRADED state must include a reason string describing what is degraded.
* The core may not be called while any required component for the active run mode is in HALTED state.
* While components are DEGRADED, degradation must be logged and must be visible via evidence chain entries (if enabled) and CLI status output. The core indicator outputs are not required to encode health state.

**Definition of Required Components:**

Required components must be declared by orchestration per run mode. At minimum:

* RESEARCH: data adapter (or historical feed), core process
* SHADOW: data adapter, persistence layer, evidence chain, core process
* LIVE: data adapter, persistence layer, evidence chain, order transmission adapter, core process

### 9.2 DEGRADED State Resolution Requirement

A DEGRADED state must not persist indefinitely. Each component must define a maximum DEGRADED duration. If degradation is not resolved within that duration, the component must either:

* Transition to HALTED, or
* Escalate to an operator via alerting

The maximum DEGRADED duration must be documented per component. Default maximum if not specified: 15 minutes.

In LIVE mode, any component-level maximum DEGRADED duration must be less than or equal to the LIVE recovery bound (5 minutes or 10 bars, whichever comes first). If a component declares a longer duration, the effective maximum in LIVE mode is the LIVE recovery bound.

### 9.3 Health Check Requirements

All components must implement a health check that:

* Can be invoked by the CLI `status` command.
* Returns current state (HEALTHY, DEGRADED, HALTED).
* Returns reason string if not HEALTHY.
* Returns timestamp of last state transition.
* Completes within 5 seconds (health checks must not block).

---

## 10. Change Control

Any modification to:

* Frozen files
* Frozen semantics
* Seam definitions
* This document

Requires:

1. Explicit written proposal describing the change and rationale
2. Explicit review by system owner
3. Explicit written approval by system owner
4. Version increment of affected artifacts
5. Update to relevant contract documents

Absent explicit written approval, changes are prohibited. "I thought it was okay" is not a defense.

### 10.1 Emergency Changes

In a genuine emergency (e.g., critical bug causing financial loss in LIVE mode), a downstream implementer may:

1. HALT the affected pod(s) immediately.
2. Document the emergency and proposed fix.
3. Await system owner approval before applying any fix.

Under no circumstances may a change be applied to the frozen core without approval, even in an emergency. The correct emergency response is always HALT, not unilateral modification.

---

## 11. Delegation Constraints

Downstream implementers (including Claude Code, Ralph, or any other agent) are authorized to:

* Build adapters that conform to the adapter seam contract
* Build CLI commands that conform to the CLI seam contract
* Build orchestration layers that conform to persistence and time authority constraints
* Build monitoring and observability that reads from evidence chains
* Build UI layers that invoke the CLI (never the core directly)
* Propose additions or clarifications to this document

Downstream implementers are not authorized to:

* Modify frozen files without explicit approval
* Reinterpret frozen semantics
* Add CLI flags that bypass safety constraints
* Create cross-pod coupling without explicit approval
* Extend DEGRADED durations beyond documented maximums
* Classify semantic changes as bug fixes
* Assume approval was granted without written evidence
* Create wrappers or shims that modify core behavior
* Implement automatic retry or recovery logic that bypasses HALT requirements
* Implement "convenience" features that weaken safety constraints
* Create debug/test modes that bypass any constraint in this document

When in doubt, the correct action is to stop and surface the question to the system owner, not to proceed with a reasonable interpretation.

---

## 12. Testing and Verification Requirements

Before LIVE mode is permitted:

### 12.1 Required Tests

* Determinism test: Same inputs produce same outputs across multiple runs.
* Replay test: Evidence chain replay produces identical outputs.
* Halt test: All HALT conditions correctly trigger HALT state.
* Recovery test: Recovery timeout correctly triggers HALT after bound expiration.
* Isolation test: Multi-pod operations do not leak state between pods.
* In-flight cancellation test: HALT correctly cancels or abandons pending operations.

### 12.2 Test Documentation

All tests must be documented with:

* Test purpose
* Test inputs
* Expected outputs
* Pass/fail criteria

Test results must be reviewed by system owner before LIVE mode approval.

---

## 13. Document Status

This document is authoritative.

It may be amended through the change control process defined in Section 10.

Downstream implementers may:

* Propose additions
* Request clarifications
* Surface ambiguities

They may not reinterpret, bypass, or selectively apply it.

Ambiguity in this document should be resolved by the system owner, not by implementer judgment.

---

## End of PHASE5_INTEGRATION_SEAMS.md
