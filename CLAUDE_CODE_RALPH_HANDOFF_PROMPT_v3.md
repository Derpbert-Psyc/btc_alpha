/ralph-loop:ralph-loop
# Prompt: Phase-5-Locked Handoff to Claude Code & Ralph

You are being handed a system whose **core semantics are frozen and approved**.

The repository you are receiving has reached the state defined as:

**"Semantics-Frozen Core with Explicit Integration Seams."**

This state is **intentional, final, and enforceable**.

Your task is **not** to reinterpret or improve the core.  
Your task is to **build against it**, surface issues, and harden the system *without violating the frozen boundary*.

---

## 1. What Is Frozen (Non-Negotiable)

The following are **authoritative and locked**:

* `btc_alpha_phase4b_1_7_2.py`
* Indicator semantics (IDs 1–24)
* Diagnostic probes (IDs 25–29)
* Determinism, warmup, dependency, activation, and state-mutation semantics
* Run-mode semantics (RESEARCH / SHADOW / LIVE)
* PHASE4A_INDICATOR_CONTRACT.md
* PHASE4B_CONTRACT_LOCKED.md
* SYSTEM_LAWS.md
* **PHASE5_INTEGRATION_SEAMS.md (APPROVED, AUTHORITATIVE)**
* **PHASE5_ROBUSTNESS_CONTRACT.md (APPROVED, AUTHORITATIVE)**
* **BYBIT_OMS_CONTRACT_v1_0_0.md (FROZEN, AUTHORITATIVE)**

You **may not**:

* Modify frozen files
* Wrap, shim, proxy, subclass, or monkey-patch frozen logic
* "Fix" semantics via adapters or orchestration
* Introduce convenience behavior that weakens safety or determinism
* Resolve ambiguity without escalation to the system owner
* Generate UI

UI DENY LIST (ABSOLUTE PROHIBITIONS)

You are explicitly forbidden from doing any of the following, directly or indirectly:
- Designing, implementing, prototyping, or discussing any user interface beyond plain CLI output
- Building or referencing any TUI, dashboard, panel, widget, menu system, or visual layout
- Creating wireframes, mockups, ASCII layouts, screenshots, or visual representations
- Introducing UI frameworks, libraries, or dependencies (terminal, web, desktop, or hybrid)
- Adding interactive navigation, keybindings, menus, forms, or cursor-driven behavior
- Emitting structured output intended for future UI consumption (JSON for UI, UI schemas, view models)
- Adding placeholders, stubs, adapters, hooks, or abstractions intended for later UI attachment
- Shaping internal APIs, data models, or module boundaries around hypothetical UI needs
- Optimizing output for aesthetics, readability, or presentation beyond minimal CLI correctness
- Implementing real-time visualization, charts, graphs, color-coding, or status dashboards
- Creating configuration flows, strategy builders, or visual composition tools
- Writing user manuals, UI documentation, onboarding flows, or usage guides beyond CLI flags
- Referencing future UI phases, suggesting UI directions, or reserving design space for UI
- Introducing any concept whose primary purpose is human interaction rather than execution or research
- Bypassing or weakening this restriction for convenience, clarity, or perceived usability gains

**PERMITTED EXCEPTION - Required Structured Output:**

Structured JSON output is explicitly **required and permitted** ONLY to satisfy:

1. CLI command schemas per PHASE5_INTEGRATION_SEAMS Section 3.3–3.4
2. Evidence chain envelope format per PHASE5_INTEGRATION_SEAMS Section 5
3. Triage and robustness artifact logging per PHASE5_ROBUSTNESS_CONTRACT Section 8

Any JSON schema, data model, or structured format beyond these three contracts is forbidden.
Any structuring motivated by "future UI needs," "dashboard consumption," or "visualization readiness" is forbidden.

If an action could reasonably be interpreted as UI-related, it is forbidden.
When in doubt, omit it.
This deny list overrides all implied permissions.
CLI-only execution and research tooling are the sole authorized scope.

### If something seems ambiguous, the correct action is to **stop and surface the question**.

---

## 2. What You Are Authorized to Build

You are explicitly authorized to implement:

1. **CLI (canonical interface)**
   * Exactly per Section 3.3–3.4 of PHASE5_INTEGRATION_SEAMS.md
   * Deterministic, structured output
   * No bypass flags
   * CLI is the *only* interface to the core in SHADOW and LIVE

2. **Orchestration Layer**
   * Run-mode enforcement
   * HALT semantics
   * Recovery bounds
   * Pod lifecycle management
   * Time authority
   * Health signaling

3. **Persistence + Evidence Chain**
   * Append-only, tamper-evident
   * Deterministic replay
   * Preflight integrity checks
   * Checkpoint + restore
   * Replay verification gates before LIVE

4. **Exchange Adapters**
   * Spot first, derivatives second
   * Strict normalization rules
   * Loud failure on malformed data
   * No fabrication, no silent degradation

5. **Multi-Pod Execution**
   * One pod per exchange sub-account
   * Strict isolation
   * Explicit operator scoping
   * No cross-pod coupling

6. **Research Validation Layer**
   * Implement Triage Filter test harness per PHASE5_ROBUSTNESS_CONTRACT.md
   * Infrastructure only: test execution, seed management, artifact logging, reproducibility tools
   * DO NOT tune thresholds (0.3 Sharpe, p<0.05, etc.) without system owner approval
   * DO NOT add/remove tests from the suite without system owner approval
   * DO NOT modify early-exit logic without system owner approval
   * Flag empirical data (false positive/negative rates) for system owner threshold review

7. **Bybit OMS Implementation**
   * Implement TradePlan execution per BYBIT_OMS_CONTRACT_v1_0_0.md
   * Frozen decisions: TP ladder (client-side reduce-only), Partial TP/SL (prohibited), trailing stop (exchange-native default with risk flag)
   * State machines: Order states, position states, transition rules per contract
   * WebSocket handling: disconnect/reconnect with 30s deadline, degraded state management
   * Fail-safe close: verified sequence with FAILSAFE state blocking
   * DO NOT modify frozen decisions without system owner approval
   * DO NOT change hard limits (500 orders, 10 conditionals, etc.)
   * DO NOT skip mandatory behaviors (exchange-side hard SL, OCO on exits, etc.)


---

## 3. Your Operating Posture (Critical)

You are not just implementers.  
You are **co-adversarial agents**.

While building, you must **actively attempt to break the system**:

* Try to violate determinism
* Try to induce silent degradation
* Try malformed adapter data
* Try persistence corruption
* Try recovery edge cases
* Try partial outages
* Try operator misuse
* Try multi-pod interference
* Try replay divergence
* Try to violate robustness filter determinism (same strategy, different results)
* Try to bypass triage gates (pass overfit strategies)
* Try parameter sensitivity edge cases (extreme degradation)
* Try to use assumed dataset dates (logic must discover bounds from data)
* Try timestamp without UTC indicator (must reject)
* Try duplicate timestamps (must halt with diagnostic)
* Try dataset prepend/append (split must remain reproducible)
* Try canonical data hash with different pandas versions (must match exactly)
* Try OMS race conditions (cancel-while-filling, order timeout, duplicate ack)
* Try WebSocket disconnect during critical operations (entry fill, exit placement)
* Try orphaned position on startup (cannot reconstruct TradePlan)
* Try exceeding order limits (500 active, 10 conditional)
* Try violating runtime budget (triage >5 minutes, individual tests >max)
* Try duplicate triage runs (same strategy+dataset hash)
* Try Tier 2 without Tier 1 PASS artifact (promotion enforcement)

If the system breaks **without HALT**, that is a defect.

If the system halts **incorrectly**, that is a defect.

If behavior is ambiguous, that is a defect.

Surface defects as:

* Concrete reproduction steps
* Exact boundary crossed
* Whether the issue is implementation, orchestration, or contract-level
* Proposed fix category:
  * **Implementation fix** (does not touch frozen artifacts) — you may proceed after documenting
  * **Contract clarification** (frozen artifacts unchanged but contract language ambiguous) — requires system owner review before proceeding
  * **Contract amendment** (frozen artifacts or semantics must change) — requires system owner approval; you must stop and wait

---

## 4. The Core Question You Must Answer

**Can this system now be iterated forward adversarially without semantic drift?**

If yes:

* Proceed implementing the remaining layers
* Continue adversarial testing as you go
* Treat PHASE5 as the guardrail

If no:

* Identify the *minimum blocking issue*
* Explain why it cannot be resolved without modifying frozen semantics
* Document the proposed contract amendment and **submit to system owner for approval**
* **Do not proceed with the blocked work until approval is granted**

You are explicitly invited to try to prove that we are *not* ready.  
Failure to break it is evidence of correctness.

---

## 5. What "Done" Means

This handoff is successful when:

* The system runs end-to-end in RESEARCH, SHADOW, and LIVE (with orders vetoed until approval)
* Evidence chains replay deterministically
* HALT semantics behave exactly as specified
* Adapters fail loudly and correctly
* Multi-pod execution is safe and isolated
* All remaining work is **plumbing, UI, or adapters only**

At that point, the system is no longer fragile.  
It becomes **mechanical**.

---

## 6. Final Constraint

If you ever find yourselves thinking:

> "This would be easier if we just changed X in the core…"

Stop.

That thought means the contract is working.

**Surface the friction** by:

1. Documenting the exact constraint causing friction
2. Documenting the work that is blocked
3. Proposing alternatives that do not require core modification
4. If no alternatives exist, submitting a contract amendment request to the system owner
5. Waiting for explicit approval before proceeding

Do not bypass. Do not reinterpret. Do not assume approval.

---

**You are cleared to proceed.**
