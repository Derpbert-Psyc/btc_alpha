"""
Phase 5 — Atomic Bundle Store + Evidence Chain

Implements Laws 10-13 (atomic commitment), Laws 33-37 (evidence & anchoring).
Commit-time allocation (Option B): commit_seq assigned at commit, not at staging.

References:
    SYSTEM_LAWS.md:40-53 (Laws 10-13)
    SYSTEM_LAWS.md:116-129 (Laws 33-37)
    PHASE5_INTEGRATION_SEAMS_v1_2_6.md:604-658 (Evidence chain)
    PHASE5_INTEGRATION_SEAMS_v1_2_6.md:779-799 (Persistence)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StoreInitError(Exception):
    """Raised when the bundle store cannot be initialised."""


class StoreAuthorityError(Exception):
    """Raised when a caller tries to set store-authoritative fields."""


class CommitError(Exception):
    """Raised when commit_bundle() fails."""


class ManifestImmutabilityError(Exception):
    """Raised on illegal manifest overwrite."""


class ChainIntegrityError(Exception):
    """Raised when the evidence chain is broken."""


# ---------------------------------------------------------------------------
# RunMode (lightweight — used only for branching on preflight / manifest)
# ---------------------------------------------------------------------------

class RunMode(Enum):
    RESEARCH = "RESEARCH"
    SHADOW = "SHADOW"
    LIVE = "LIVE"


# ---------------------------------------------------------------------------
# Bundle dataclasses (frozen, immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BarBundle:
    """Single artifact per bar computation.  Atomic commit (Law 10)."""
    format_version: int
    bar_index: int                        # Bar identity (not chain position)
    timestamp: int                        # Bar close timestamp (epoch seconds)
    pod_id: str
    core_version_hash: str

    candle_inputs: Dict[str, Any]         # {"v": int, "s": "PRICE"}
    system_inputs: Dict[str, Any]         # Serialized SystemInputs (scaled ints)
    period_data: Optional[Dict]

    indicator_outputs: Dict[int, Dict[str, Any]]  # indicator_id -> serialized output

    engine_state_hash: str
    health_state: Dict[str, str]

    # STORE-AUTHORITATIVE — caller must pass sentinels (-1, "", "")
    commit_seq: int = -1
    prev_packet_hash: str = ""
    packet_hash: str = ""


@dataclass(frozen=True)
class EventBundle:
    """Non-bar event.  Same stage/commit path as BarBundle."""
    format_version: int
    timestamp: int                        # Epoch seconds
    pod_id: str
    event_type: str                       # e.g. "state_transition", "halt", "resume"
    payload: Dict[str, Any]               # All values as scaled integers

    # STORE-AUTHORITATIVE — caller must pass sentinels (-1, "", "")
    commit_seq: int = -1
    prev_packet_hash: str = ""
    packet_hash: str = ""


@dataclass(frozen=True)
class StagingKey:
    """Immutable.  Produced by stage_bundle().  Required by commit_bundle() (Law 11)."""
    content_hash: str     # SHA256 of staged content (integrity verification)
    staged_path: str      # Path to staged/<uuid>.staged
    uuid: str             # UUID identifying this staged bundle


@dataclass(frozen=True)
class EvidenceManifest:
    """Stored once per run (Law 34).  Immutable after creation."""
    run_id: str
    pod_id: str
    sub_account_id: Optional[str]
    core_version_hash: str     # SHA256 of btc_alpha_phase4b_1_7_2.py bytes
    config_hash: str           # SHA256 of canonical PodConfig JSON
    schema_version: int
    created_ts: str            # ISO 8601 with Z suffix


# ---------------------------------------------------------------------------
# Canonical hashing — THE single routine for packet_hash computation
# ---------------------------------------------------------------------------

def compute_packet_hash(bundle_dict: dict) -> str:
    """
    THE canonical bundle hash.  Used by commit_bundle and verify_chain_integrity.

    Rules:
        1. Set bundle_dict["packet_hash"] = "" (force empty before hashing)
        2. canonical_json = json.dumps(bundle_dict, sort_keys=True,
                                       separators=(",", ":"), ensure_ascii=False)
        3. packet_hash = sha256(canonical_json.encode("utf-8")).hexdigest()
        4. Return packet_hash (always 64 hex chars, never empty)
    """
    bundle_dict["packet_hash"] = ""
    canonical_json = json.dumps(
        bundle_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bundle_to_dict(bundle: Union[BarBundle, EventBundle]) -> dict:
    """Convert a frozen bundle dataclass to a plain dict."""
    return asdict(bundle)


def _is_sentinel(bundle: Union[BarBundle, EventBundle]) -> None:
    """Validate that store-authoritative fields are sentinel values."""
    for field_name, sentinel in [
        ("commit_seq", -1),
        ("prev_packet_hash", ""),
        ("packet_hash", ""),
    ]:
        value = getattr(bundle, field_name)
        if value != sentinel:
            raise StoreAuthorityError(
                f"Caller must not set {field_name}; store is authoritative"
            )


def _fsync_directory(dir_path: str) -> None:
    """fsync a directory to ensure rename durability (POSIX only)."""
    if platform.system() == "Windows":
        return
    fd = os.open(dir_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(target_path: str, data: bytes, *, fsync_parent: bool = False) -> None:
    """
    Atomic File Write Protocol (OS-Agnostic).

    1. Write to temp file in same directory as target.
    2. fsync the temp file.
    3. os.replace(temp, target) — atomic on all platforms.
    4. Optionally fsync parent directory (for watermark.json / manifest.json).
    """
    parent = os.path.dirname(target_path)
    tmp_path = os.path.join(parent, f"_tmp_{uuid.uuid4().hex}")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, target_path)
    if fsync_parent:
        _fsync_directory(parent)


def _dict_to_bundle(d: dict) -> Union[BarBundle, EventBundle]:
    """Reconstruct the correct bundle type from a dict."""
    if "bar_index" in d:
        return BarBundle(**{
            k: v for k, v in d.items()
            if k in BarBundle.__dataclass_fields__
        })
    return EventBundle(**{
        k: v for k, v in d.items()
        if k in EventBundle.__dataclass_fields__
    })


# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------

@dataclass
class _Watermark:
    commit_seq: int          # -1 means empty store
    packet_hash: str         # "" if empty
    bar_index: int           # -1 if empty (highest bar_index seen, or -1)

    def to_dict(self) -> dict:
        return {
            "commit_seq": self.commit_seq,
            "packet_hash": self.packet_hash,
            "bar_index": self.bar_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> _Watermark:
        return cls(
            commit_seq=d["commit_seq"],
            packet_hash=d["packet_hash"],
            bar_index=d.get("bar_index", -1),
        )

    @classmethod
    def empty(cls) -> _Watermark:
        return cls(commit_seq=-1, packet_hash="", bar_index=-1)


# ---------------------------------------------------------------------------
# AtomicBundleStore
# ---------------------------------------------------------------------------

class AtomicBundleStore:
    """
    Evidence chain store with stage→commit model.

    Directory layout (per run):
        <store_dir>/
            manifest.json
            staged/           (uncommitted, cleaned on recovery)
            committed/        (immutable, Law 13)
            scratch/          (preflight only, excluded from recovery/integrity)
            watermark.json    (derived from committed/)
    """

    def __init__(self, store_dir: str, pod_id: str, core_version_hash: str) -> None:
        self._store_dir = store_dir
        self._pod_id = pod_id
        self._core_version_hash = core_version_hash
        self._chain_lock = threading.Lock()

        self._staged_dir = os.path.join(store_dir, "staged")
        self._committed_dir = os.path.join(store_dir, "committed")
        self._scratch_dir = os.path.join(store_dir, "scratch")
        self._watermark_path = os.path.join(store_dir, "watermark.json")
        self._manifest_path = os.path.join(store_dir, "manifest.json")

        # Create directories
        os.makedirs(self._staged_dir, exist_ok=True)
        os.makedirs(self._committed_dir, exist_ok=True)
        os.makedirs(self._scratch_dir, exist_ok=True)

        # Same-volume behavioural check
        self._verify_same_volume()

        # Recovery + watermark load
        self.crash_recovery()

    # ------------------------------------------------------------------
    # Same-volume check
    # ------------------------------------------------------------------

    def _verify_same_volume(self) -> None:
        """Verify staged/ and committed/ are on the same filesystem via behavioural test."""
        test_src = os.path.join(self._staged_dir, f"_vol_check_{uuid.uuid4().hex}")
        test_dst = os.path.join(self._committed_dir, f"_vol_check_{uuid.uuid4().hex}")
        try:
            with open(test_src, "w") as f:
                f.write("vol_check")
            os.replace(test_src, test_dst)
        except OSError as exc:
            # Clean up if partially written
            for p in (test_src, test_dst):
                try:
                    os.remove(p)
                except OSError:
                    pass
            raise StoreInitError(
                "staged/ and committed/ must be on the same filesystem/volume"
            ) from exc
        finally:
            for p in (test_src, test_dst):
                try:
                    os.remove(p)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def create_manifest(
        self,
        manifest: EvidenceManifest,
        run_mode: RunMode,
        *,
        allow_overwrite: bool = False,
    ) -> None:
        """Write manifest.json.  Immutability enforced per mode."""
        if os.path.exists(self._manifest_path):
            if run_mode in (RunMode.SHADOW, RunMode.LIVE):
                raise ManifestImmutabilityError(
                    f"manifest.json already exists in {run_mode.value} mode; pod must HALT"
                )
            if run_mode == RunMode.RESEARCH and not allow_overwrite:
                raise ManifestImmutabilityError(
                    "manifest.json already exists in RESEARCH mode (allow_overwrite=False)"
                )
            # RESEARCH + allow_overwrite — log warning, overwrite
            import logging
            logging.getLogger(__name__).warning(
                "Overwriting manifest.json in RESEARCH mode (allow_overwrite=True)"
            )

        data = json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":"))
        _atomic_write(self._manifest_path, data.encode("utf-8"), fsync_parent=True)

    # ------------------------------------------------------------------
    # Stage
    # ------------------------------------------------------------------

    def stage_bundle(self, bundle: Union[BarBundle, EventBundle]) -> StagingKey:
        """
        Stage a bundle for later commit.  Does NOT acquire _chain_lock.

        1. Validate sentinel values (store authority).
        2. Serialize to JSON.
        3. Compute content_hash (SHA256 of raw bytes).
        4. Write to staged/<uuid>.staged via atomic write.
        5. Return StagingKey.
        """
        _is_sentinel(bundle)

        bundle_uuid = uuid.uuid4().hex
        content_bytes = json.dumps(
            _bundle_to_dict(bundle),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

        content_hash = hashlib.sha256(content_bytes).hexdigest()
        staged_path = os.path.join(self._staged_dir, f"{bundle_uuid}.staged")
        _atomic_write(staged_path, content_bytes)

        return StagingKey(
            content_hash=content_hash,
            staged_path=staged_path,
            uuid=bundle_uuid,
        )

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit_bundle(self, key: StagingKey) -> None:
        """
        Commit a staged bundle.  Acquires _chain_lock for entire operation.

        Steps:
            1. Verify staged file exists.
            2. Re-read + re-hash to verify integrity.
            3. Allocate commit_seq from watermark.
            4. Set prev_packet_hash, commit_seq, compute packet_hash.
            5. Write committed/<commit_seq>.bundle (atomic).
            6. Update watermark.json (atomic + parent fsync).
            7. Delete staged file.
        """
        with self._chain_lock:
            staged_path = os.path.join(
                self._staged_dir, f"{key.uuid}.staged"
            )
            if not os.path.exists(staged_path):
                raise CommitError(
                    f"Staged file missing: {staged_path}"
                )

            # Re-read and verify content hash
            with open(staged_path, "rb") as f:
                raw = f.read()
            actual_hash = hashlib.sha256(raw).hexdigest()
            if actual_hash != key.content_hash:
                raise CommitError(
                    f"Content hash mismatch: expected {key.content_hash}, "
                    f"got {actual_hash}"
                )

            bundle_dict: dict = json.loads(raw.decode("utf-8"))

            # Allocate commit_seq
            new_seq = self._watermark.commit_seq + 1  # 0 if store was empty (-1 + 1)

            # Set chain fields
            if new_seq == 0:
                prev_hash = ""
            else:
                prev_hash = self._watermark.packet_hash

            bundle_dict["commit_seq"] = new_seq
            bundle_dict["prev_packet_hash"] = prev_hash

            # Compute packet_hash (the single canonical routine)
            pkt_hash = compute_packet_hash(bundle_dict)
            bundle_dict["packet_hash"] = pkt_hash

            # Verify target does not exist (Law 13 — never overwrite)
            committed_path = os.path.join(
                self._committed_dir, f"{new_seq}.bundle"
            )
            if os.path.exists(committed_path):
                raise CommitError(
                    f"committed/{new_seq}.bundle already exists — "
                    "cannot overwrite committed bundles"
                )

            # Write committed bundle
            committed_bytes = json.dumps(
                bundle_dict,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            _atomic_write(committed_path, committed_bytes)

            # Update watermark
            bar_index = bundle_dict.get("bar_index", self._watermark.bar_index)
            self._watermark = _Watermark(
                commit_seq=new_seq,
                packet_hash=pkt_hash,
                bar_index=max(bar_index, self._watermark.bar_index)
                if isinstance(bar_index, int) and bar_index >= 0
                else self._watermark.bar_index,
            )
            wm_bytes = json.dumps(
                self._watermark.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            _atomic_write(self._watermark_path, wm_bytes, fsync_parent=True)

            # Delete staged file
            try:
                os.remove(staged_path)
            except OSError:
                pass  # Best-effort; recovery will clean up

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    def crash_recovery(self) -> int:
        """
        Recovery: committed/ is source of truth (Law 12).

        1. Scan committed/ — sort by commit_seq.
        2. Walk hash chain — verify integrity.
        3. Derive watermark from last valid committed bundle.
        4. Reconcile / recreate watermark.json.
        5. Delete ALL staged files.
        6. Return watermark commit_seq (-1 if empty).
        """
        import logging
        log = logging.getLogger(__name__)

        # Scan committed bundles
        committed_bundles: Dict[int, dict] = {}
        if os.path.isdir(self._committed_dir):
            for fname in os.listdir(self._committed_dir):
                if fname.endswith(".bundle"):
                    seq_str = fname[:-7]  # strip ".bundle"
                    try:
                        seq = int(seq_str)
                    except ValueError:
                        continue
                    fpath = os.path.join(self._committed_dir, fname)
                    with open(fpath, "rb") as f:
                        data = json.loads(f.read().decode("utf-8"))
                    committed_bundles[seq] = data

        if not committed_bundles:
            # Empty store
            self._watermark = _Watermark.empty()
            self._write_watermark()
            self._clean_staged(log)
            return -1

        # Sort by commit_seq
        max_seq = max(committed_bundles.keys())

        # Walk chain — find last valid
        last_valid_seq = -1
        last_valid_hash = ""
        last_valid_bar = -1

        for seq in range(max_seq + 1):
            if seq not in committed_bundles:
                log.warning("Gap at commit_seq=%d during recovery", seq)
                break
            bd = committed_bundles[seq]
            # Verify prev_packet_hash linkage
            expected_prev = "" if seq == 0 else last_valid_hash
            if bd.get("prev_packet_hash", "") != expected_prev:
                log.warning(
                    "Hash chain break at commit_seq=%d: expected prev=%s, got=%s",
                    seq, expected_prev, bd.get("prev_packet_hash", ""),
                )
                break
            # Verify packet_hash
            stored_hash = bd.get("packet_hash", "")
            recomputed = compute_packet_hash(dict(bd))  # copy to avoid mutation
            bd["packet_hash"] = stored_hash  # restore after compute_packet_hash zeroes it
            if stored_hash != recomputed:
                log.warning(
                    "packet_hash mismatch at commit_seq=%d: stored=%s, recomputed=%s",
                    seq, stored_hash, recomputed,
                )
                break
            last_valid_seq = seq
            last_valid_hash = stored_hash
            bar_idx = bd.get("bar_index", -1)
            if isinstance(bar_idx, int) and bar_idx >= 0:
                last_valid_bar = max(last_valid_bar, bar_idx)

        # Derive watermark
        derived = _Watermark(
            commit_seq=last_valid_seq,
            packet_hash=last_valid_hash,
            bar_index=last_valid_bar,
        )

        # Reconcile with existing watermark.json
        if os.path.exists(self._watermark_path):
            try:
                with open(self._watermark_path, "r") as f:
                    existing = _Watermark.from_dict(json.load(f))
                if (existing.commit_seq != derived.commit_seq
                        or existing.packet_hash != derived.packet_hash):
                    log.warning(
                        "Watermark disagreement: file=%s, derived=%s — overwriting",
                        existing.to_dict(), derived.to_dict(),
                    )
            except (json.JSONDecodeError, KeyError, TypeError):
                log.warning("Corrupt watermark.json — rebuilding from committed/")

        self._watermark = derived
        self._write_watermark()
        self._clean_staged(log)
        return self._watermark.commit_seq

    def _write_watermark(self) -> None:
        wm_bytes = json.dumps(
            self._watermark.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _atomic_write(self._watermark_path, wm_bytes, fsync_parent=True)

    def _clean_staged(self, log: Any) -> None:
        """Delete all staged files (Law 12)."""
        if not os.path.isdir(self._staged_dir):
            return
        for fname in os.listdir(self._staged_dir):
            fpath = os.path.join(self._staged_dir, fname)
            if os.path.isfile(fpath):
                log.info("Recovery: deleting staged file %s", fname)
                try:
                    os.remove(fpath)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    def preflight_check(self, run_mode: RunMode) -> bool:
        """
        Pre-bar integrity check (Section 5.4).

        1. Writability check via scratch/.
        2. Verify watermark.json parseable.
        3. Load last committed bundle.
        4. Verify packet_hash matches watermark.

        Returns True if OK, False if failure.
        RESEARCH: logs WARNING, returns True on failure.
        SHADOW/LIVE: returns False on failure.
        """
        import logging
        log = logging.getLogger(__name__)

        try:
            # 1. Writability test in scratch/
            scratch_file = os.path.join(
                self._scratch_dir, f"_preflight_{int(time.time() * 1000)}"
            )
            fd = os.open(scratch_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.write(fd, b"preflight")
                os.fsync(fd)
            finally:
                os.close(fd)
            os.remove(scratch_file)

            # 2. Verify watermark.json
            if not os.path.exists(self._watermark_path):
                raise RuntimeError("watermark.json missing")
            with open(self._watermark_path, "r") as f:
                wm = json.load(f)
            _ = _Watermark.from_dict(wm)

            # 3-4. Verify last committed bundle (skip if empty store)
            if self._watermark.commit_seq >= 0:
                last_path = os.path.join(
                    self._committed_dir,
                    f"{self._watermark.commit_seq}.bundle",
                )
                if not os.path.exists(last_path):
                    raise RuntimeError(
                        f"committed/{self._watermark.commit_seq}.bundle missing"
                    )
                with open(last_path, "rb") as f:
                    bd = json.loads(f.read().decode("utf-8"))
                if bd.get("packet_hash", "") != self._watermark.packet_hash:
                    raise RuntimeError(
                        "Last committed bundle packet_hash does not match watermark"
                    )

            return True

        except Exception as exc:
            if run_mode == RunMode.RESEARCH:
                log.warning("Preflight failure in RESEARCH mode (continuing): %s", exc)
                return True
            log.error("Preflight failure in %s mode: %s", run_mode.value, exc)
            return False

    # ------------------------------------------------------------------
    # Chain integrity verification
    # ------------------------------------------------------------------

    def verify_chain_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Walk entire committed/ chain from commit_seq=0 to watermark.

        Checks:
            - Contiguity (every int 0..watermark.commit_seq has a .bundle file)
            - Genesis: commit_seq=0 has prev_packet_hash=""
            - Every subsequent prev_packet_hash matches predecessor's packet_hash
            - Every packet_hash is non-empty and matches recomputed value

        Returns (True, None) if valid, (False, error_message) if broken.
        """
        if self._watermark.commit_seq < 0:
            return (True, None)  # Empty store is valid

        prev_hash = ""
        for seq in range(self._watermark.commit_seq + 1):
            fpath = os.path.join(self._committed_dir, f"{seq}.bundle")
            if not os.path.exists(fpath):
                return (False, f"Missing committed/{seq}.bundle (gap in chain)")

            with open(fpath, "rb") as f:
                bd = json.loads(f.read().decode("utf-8"))

            # Verify commit_seq stored in bundle
            if bd.get("commit_seq") != seq:
                return (
                    False,
                    f"commit_seq mismatch in {seq}.bundle: "
                    f"stored={bd.get('commit_seq')}",
                )

            # Verify prev_packet_hash
            stored_prev = bd.get("prev_packet_hash", "")
            if seq == 0:
                if stored_prev != "":
                    return (
                        False,
                        f"Genesis bundle must have prev_packet_hash='', "
                        f"got '{stored_prev}'",
                    )
            else:
                if stored_prev != prev_hash:
                    return (
                        False,
                        f"prev_packet_hash mismatch at commit_seq={seq}: "
                        f"expected={prev_hash}, got={stored_prev}",
                    )

            # Verify packet_hash
            stored_hash = bd.get("packet_hash", "")
            if not stored_hash:
                return (False, f"Empty packet_hash at commit_seq={seq}")

            recomputed = compute_packet_hash(dict(bd))
            # Restore after compute_packet_hash zeroed it
            bd["packet_hash"] = stored_hash
            if stored_hash != recomputed:
                return (
                    False,
                    f"packet_hash mismatch at commit_seq={seq}: "
                    f"stored={stored_hash}, recomputed={recomputed}",
                )

            prev_hash = stored_hash

        return (True, None)

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self, from_seq: int, to_seq: int
    ) -> List[Union[BarBundle, EventBundle]]:
        """Load committed bundles in commit_seq order for deterministic replay."""
        results: List[Union[BarBundle, EventBundle]] = []
        for seq in range(from_seq, to_seq + 1):
            fpath = os.path.join(self._committed_dir, f"{seq}.bundle")
            if not os.path.exists(fpath):
                raise CommitError(f"Missing committed/{seq}.bundle during replay")
            with open(fpath, "rb") as f:
                bd = json.loads(f.read().decode("utf-8"))
            results.append(_dict_to_bundle(bd))
        return results

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def store_dir(self) -> str:
        return self._store_dir

    @property
    def watermark(self) -> _Watermark:
        return self._watermark

    @property
    def manifest_path(self) -> str:
        return self._manifest_path
