"""
Phase 5 — Pod Lifecycle + Orchestration

Defines RunMode, HealthState, PodState, PodConfig, ComponentHealth, and Pod.
The Pod owns a single AtomicBundleStore and IndicatorEngine instance.

References:
    PHASE5_INTEGRATION_SEAMS_v1_2_6.md:522-582 — Pod definition
    PHASE5_INTEGRATION_SEAMS_v1_2_6.md:660-742 — Run modes / HALT
    PHASE5_INTEGRATION_SEAMS_v1_2_6.md:819-862 — Health signaling
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from btc_alpha_phase4b_1_7_2 import (
    IndicatorEngine,
    IndicatorOutput,
    SystemInputs,
    TypedValue,
)
from phase5_bundle_store import (
    AtomicBundleStore,
    BarBundle,
    EventBundle,
    EvidenceManifest,
    ManifestImmutabilityError,
    RunMode,
)

_log = logging.getLogger(__name__)

# Re-export RunMode from bundle_store for convenience
__all__ = [
    "RunMode", "HealthState", "PodState", "PodConfig", "ComponentHealth", "Pod",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HealthState(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    HALTED = "HALTED"


class PodState(Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    HALTED = "HALTED"


# ---------------------------------------------------------------------------
# ComponentHealth
# ---------------------------------------------------------------------------

@dataclass
class ComponentHealth:
    """Health tracking for a single component."""
    name: str
    state: HealthState = HealthState.HEALTHY
    reason: str = ""
    last_transition_ts: str = ""
    degraded_since: Optional[str] = None
    max_degraded_seconds: int = 900  # 15 min default, capped to 5 min in LIVE

    def transition(self, new_state: HealthState, reason: str = "") -> None:
        now = _now_iso()
        self.state = new_state
        self.reason = reason
        self.last_transition_ts = now
        if new_state == HealthState.DEGRADED:
            self.degraded_since = now
        elif new_state == HealthState.HEALTHY:
            self.degraded_since = None
        _log.error(
            "Component %s -> %s: %s", self.name, new_state.value, reason
        ) if new_state != HealthState.HEALTHY else _log.info(
            "Component %s -> %s", self.name, new_state.value
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "reason": self.reason,
            "last_transition_ts": self.last_transition_ts,
            "degraded_since": self.degraded_since,
            "max_degraded_seconds": self.max_degraded_seconds,
        }


# ---------------------------------------------------------------------------
# PodConfig (frozen)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PodConfig:
    """Immutable pod configuration.  Canonical field names used everywhere."""
    pod_id: str                    # UUIDv4
    stream_id: str
    run_mode: str                  # "RESEARCH" | "SHADOW" | "LIVE"
    starting_capital_cents: int
    friction_preset: str
    bar_interval_seconds: int
    evidence_dir: str              # Base evidence directory path


def compute_config_hash(config: PodConfig) -> str:
    """SHA256 of canonical PodConfig JSON (sorted keys, compact separators)."""
    d = {
        "pod_id": config.pod_id,
        "stream_id": config.stream_id,
        "run_mode": config.run_mode,
        "starting_capital_cents": config.starting_capital_cents,
        "friction_preset": config.friction_preset,
        "bar_interval_seconds": config.bar_interval_seconds,
        "evidence_dir": config.evidence_dir,
    }
    js = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(js.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _compute_core_version_hash() -> str:
    """SHA256 of btc_alpha_phase4b_1_7_2.py file bytes."""
    p = Path(__file__).parent / "btc_alpha_phase4b_1_7_2.py"
    if not p.exists():
        return "CORE_NOT_FOUND"
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Pod
# ---------------------------------------------------------------------------

class Pod:
    """
    Pod lifecycle management.

    One Pod = one IndicatorEngine + one AtomicBundleStore + one PodConfig.
    The store is shared with OMS for event commits (same commit_seq chain).
    """

    def __init__(self, config: PodConfig, run_id: str) -> None:
        self._config = config
        self._run_id = run_id
        self._run_mode = RunMode(config.run_mode)
        self._state = PodState.CREATED
        self._halt_reason: Optional[str] = None
        self._bar_counter = 0
        self._last_bar_ts: Optional[int] = None

        # Core version hash
        self._core_version_hash = _compute_core_version_hash()

        # Config hash
        self._config_hash = compute_config_hash(config)

        # Store directory: <evidence_dir>/<pod_id>/<run_id>/
        store_dir = os.path.join(config.evidence_dir, config.pod_id, run_id)
        self._store = AtomicBundleStore(
            store_dir, config.pod_id, self._core_version_hash
        )

        # Create manifest
        manifest = EvidenceManifest(
            run_id=run_id,
            pod_id=config.pod_id,
            sub_account_id=None,
            core_version_hash=self._core_version_hash,
            config_hash=self._config_hash,
            schema_version=1,
            created_ts=_now_iso(),
        )
        self._store.create_manifest(manifest, self._run_mode)

        # Indicator engine
        self._engine = IndicatorEngine(stream_id=config.stream_id)
        self._engine.register_all()

        # Component health tracking
        self._components: Dict[str, ComponentHealth] = {
            "core": ComponentHealth("core"),
            "persistence": ComponentHealth("persistence"),
            "evidence_chain": ComponentHealth("evidence_chain"),
        }

        _log.info("Pod %s created (run_id=%s, mode=%s)", config.pod_id, run_id, config.run_mode)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def pod_id(self) -> str:
        return self._config.pod_id

    @property
    def run_mode(self) -> RunMode:
        return self._run_mode

    @property
    def state(self) -> PodState:
        return self._state

    @property
    def config(self) -> PodConfig:
        return self._config

    @property
    def bar_counter(self) -> int:
        return self._bar_counter

    def get_bundle_store(self) -> AtomicBundleStore:
        """Return the pod's store (for OMS to commit events through)."""
        return self._store

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def overall_health(self) -> HealthState:
        """Compute aggregate health from components."""
        states = [c.state for c in self._components.values()]
        if HealthState.HALTED in states:
            return HealthState.HALTED
        if HealthState.DEGRADED in states:
            return HealthState.DEGRADED
        return HealthState.HEALTHY

    # ------------------------------------------------------------------
    # Halt / Resume
    # ------------------------------------------------------------------

    def halt(self, reason: str) -> None:
        """Immediate HALT.  Stages+commits EventBundle.  No auto-resume."""
        self._state = PodState.HALTED
        self._halt_reason = reason
        for c in self._components.values():
            c.transition(HealthState.HALTED, reason)

        event = EventBundle(
            format_version=1,
            timestamp=int(time.time()),
            pod_id=self._config.pod_id,
            event_type="halt",
            payload={"reason": reason, "bar_counter": self._bar_counter},
        )
        try:
            key = self._store.stage_bundle(event)
            self._store.commit_bundle(key)
        except Exception as e:
            _log.error("Failed to commit halt event: %s", e)

        _log.error("Pod %s HALTED: %s", self._config.pod_id, reason)

    def resume(self, operator_pod_id: str) -> bool:
        """Resume from HALTED.  Requires pod_id match.  No --force."""
        if self._state != PodState.HALTED:
            _log.warning("Cannot resume: pod is not HALTED (state=%s)", self._state.value)
            return False

        if operator_pod_id != self._config.pod_id:
            _log.error(
                "Resume rejected: pod_id mismatch (expected=%s, got=%s)",
                self._config.pod_id, operator_pod_id,
            )
            return False

        self._state = PodState.RUNNING
        self._halt_reason = None
        for c in self._components.values():
            c.transition(HealthState.HEALTHY)

        event = EventBundle(
            format_version=1,
            timestamp=int(time.time()),
            pod_id=self._config.pod_id,
            event_type="resume",
            payload={"bar_counter": self._bar_counter},
        )
        try:
            key = self._store.stage_bundle(event)
            self._store.commit_bundle(key)
        except Exception as e:
            _log.error("Failed to commit resume event: %s", e)

        _log.info("Pod %s resumed", self._config.pod_id)
        return True

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def transition_mode(self, new_mode: str, confirmation: str = "") -> bool:
        """Transition run mode.  Preconditions per Section 7.

        - Cannot transition while HALTED.
        - RESEARCH → SHADOW → LIVE requires preconditions.
        - LIVE → SHADOW/RESEARCH always allowed.
        """
        if self._state == PodState.HALTED:
            _log.error("Cannot transition mode while HALTED")
            return False

        try:
            target = RunMode(new_mode)
        except ValueError:
            _log.error("Invalid run mode: %s", new_mode)
            return False

        if target == self._run_mode:
            return True  # No-op

        # Downgrade always allowed
        mode_order = {RunMode.RESEARCH: 0, RunMode.SHADOW: 1, RunMode.LIVE: 2}
        upgrading = mode_order.get(target, 0) > mode_order.get(self._run_mode, 0)

        if upgrading:
            # Preconditions for upgrade
            if target == RunMode.SHADOW:
                # Evidence chain must be writable
                if not self._store.preflight_check(RunMode.SHADOW):
                    _log.error("Cannot transition to SHADOW: evidence chain not writable")
                    return False
            elif target == RunMode.LIVE:
                # All components must be HEALTHY
                if self.overall_health() != HealthState.HEALTHY:
                    _log.error("Cannot transition to LIVE: components not all HEALTHY")
                    return False
                # Confirmation required
                if not confirmation:
                    _log.error("LIVE transition requires confirmation string")
                    return False

        old_mode = self._run_mode
        self._run_mode = target

        event = EventBundle(
            format_version=1,
            timestamp=int(time.time()),
            pod_id=self._config.pod_id,
            event_type="state_transition",
            payload={
                "from_mode": old_mode.value,
                "to_mode": target.value,
                "bar_counter": self._bar_counter,
            },
        )
        try:
            key = self._store.stage_bundle(event)
            self._store.commit_bundle(key)
        except Exception as e:
            _log.error("Failed to commit mode transition event: %s", e)

        _log.info("Pod %s: %s -> %s", self._config.pod_id, old_mode.value, target.value)
        return True

    # ------------------------------------------------------------------
    # Process bar
    # ------------------------------------------------------------------

    def process_bar(
        self,
        timestamp: int,
        candle_inputs: Dict[str, Optional[TypedValue]],
        system_inputs: Optional[SystemInputs] = None,
        period_data: Optional[Dict[str, Optional[TypedValue]]] = None,
    ) -> Optional[Dict[int, IndicatorOutput]]:
        """Process one bar.  Preflight → health → compute → commit.

        Returns indicator outputs or None on failure/HALT.
        """
        if self._state == PodState.HALTED:
            _log.warning("Cannot process bar: pod is HALTED")
            return None

        self._state = PodState.RUNNING

        # Preflight check
        preflight_ok = self._store.preflight_check(self._run_mode)
        if not preflight_ok:
            error_envelope = {
                "error": "PREFLIGHT_FAILED",
                "pod_id": self._config.pod_id,
                "run_mode": self._run_mode.value,
                "reason": "Evidence chain integrity check failed",
                "bar_index": self._bar_counter,
                "timestamp": _now_iso(),
                "action": "HALT",
            }
            _log.error("Preflight failed: %s", json.dumps(error_envelope))
            self.halt(f"PREFLIGHT_FAILED at bar {self._bar_counter}")
            return None

        # Health check
        if self.overall_health() == HealthState.HALTED:
            self.halt("Component HALTED during bar processing")
            return None

        # Compute indicators
        try:
            outputs = self._engine.compute_all(
                timestamp=timestamp,
                bar_index=self._bar_counter,
                candle_inputs=candle_inputs,
                system_inputs=system_inputs,
                period_data=period_data,
                stream_id=self._config.stream_id,
            )
        except Exception as e:
            _log.error("IndicatorEngine.compute_all failed: %s", e)
            self.halt(f"compute_all failed: {e}")
            return None

        # Serialize indicator outputs for bundle
        serialized_outputs: Dict[int, Dict[str, Any]] = {}
        for ind_id, out in outputs.items():
            values_ser = {}
            for k, v in out.values.items():
                if v is None:
                    values_ser[k] = None
                else:
                    values_ser[k] = {"v": v.value, "s": v.sem.value}
            serialized_outputs[ind_id] = values_ser

        # Compute engine state hash
        engine_state_hash = hashlib.sha256(
            json.dumps(
                serialized_outputs, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

        # Build candle_inputs serialization
        ci_ser: Dict[str, Any] = {}
        for k, v in candle_inputs.items():
            if v is None:
                ci_ser[k] = None
            else:
                ci_ser[k] = {"v": v.value, "s": v.sem.value}

        # Build system_inputs serialization
        si_ser: Dict[str, Any] = {}
        if system_inputs is not None:
            si_ser = system_inputs.to_dict()
            # Convert TypedValue objects in the dict
            for k, v in si_ser.items():
                if isinstance(v, TypedValue):
                    si_ser[k] = {"v": v.value, "s": v.sem.value}

        # Build period_data serialization
        pd_ser: Optional[Dict] = None
        if period_data is not None:
            pd_ser = {}
            for k, v in period_data.items():
                if v is None:
                    pd_ser[k] = None
                else:
                    pd_ser[k] = {"v": v.value, "s": v.sem.value}

        # Health state
        health_ser = {
            name: comp.state.value for name, comp in self._components.items()
        }

        # Stage + commit bar bundle
        bar_bundle = BarBundle(
            format_version=1,
            bar_index=self._bar_counter,
            timestamp=timestamp,
            pod_id=self._config.pod_id,
            core_version_hash=self._core_version_hash,
            candle_inputs=ci_ser,
            system_inputs=si_ser,
            period_data=pd_ser,
            indicator_outputs=serialized_outputs,
            engine_state_hash=engine_state_hash,
            health_state=health_ser,
        )

        try:
            key = self._store.stage_bundle(bar_bundle)
            self._store.commit_bundle(key)
        except Exception as e:
            _log.error("Failed to commit bar bundle: %s", e)
            if self._run_mode != RunMode.RESEARCH:
                self.halt(f"Bar bundle commit failed: {e}")
                return None

        self._bar_counter += 1
        self._last_bar_ts = timestamp
        return outputs

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """JSON-serializable status dict."""
        return {
            "pod_id": self._config.pod_id,
            "run_id": self._run_id,
            "run_mode": self._run_mode.value,
            "state": self._state.value,
            "health": self.overall_health().value,
            "halt_reason": self._halt_reason,
            "bar_counter": self._bar_counter,
            "last_bar_ts": self._last_bar_ts,
            "components": {
                name: comp.to_dict() for name, comp in self._components.items()
            },
            "watermark": self._store.watermark.to_dict(),
        }
