"""
Verification suite for phase5_bundle_store.py

Tests all 19 verification items from the Phase 5 plan Item 4.
Run: python verify_bundle_store.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from unittest.mock import patch

from phase5_bundle_store import (
    AtomicBundleStore,
    BarBundle,
    ChainIntegrityError,
    CommitError,
    EventBundle,
    EvidenceManifest,
    ManifestImmutabilityError,
    RunMode,
    StagingKey,
    StoreAuthorityError,
    StoreInitError,
    compute_packet_hash,
    _atomic_write,
)

logging.basicConfig(level=logging.WARNING)
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_store(prefix: str = "bs_test_") -> str:
    """Create a temp directory for a store."""
    return tempfile.mkdtemp(prefix=prefix)


def _make_bar(bar_index: int, ts: int, pod_id: str = "pod-1") -> BarBundle:
    return BarBundle(
        format_version=1,
        bar_index=bar_index,
        timestamp=ts,
        pod_id=pod_id,
        core_version_hash="abc123",
        candle_inputs={"close": {"v": 100_00, "s": "PRICE"}},
        system_inputs={"position_side": 0},
        period_data=None,
        indicator_outputs={1: {"ema": {"v": 99_00, "s": "PRICE"}}},
        engine_state_hash="state_hash_" + str(bar_index),
        health_state={"status": "HEALTHY"},
        # Sentinels — store-authoritative
        commit_seq=-1,
        prev_packet_hash="",
        packet_hash="",
    )


def _make_event(ts: int, etype: str = "halt", pod_id: str = "pod-1") -> EventBundle:
    return EventBundle(
        format_version=1,
        timestamp=ts,
        pod_id=pod_id,
        event_type=etype,
        payload={"reason_code": 1},
        commit_seq=-1,
        prev_packet_hash="",
        packet_hash="",
    )


def _make_manifest(run_id: str = "run-1", pod_id: str = "pod-1") -> EvidenceManifest:
    return EvidenceManifest(
        run_id=run_id,
        pod_id=pod_id,
        sub_account_id=None,
        core_version_hash="core_hash",
        config_hash="config_hash",
        schema_version=1,
        created_ts="2026-01-01T00:00:00Z",
    )


def _build_store(base: str | None = None) -> AtomicBundleStore:
    d = base or _tmp_store()
    return AtomicBundleStore(d, pod_id="pod-1", core_version_hash="abc123")


# ---------------------------------------------------------------------------
# Test 1: commit_seq monotonicity and contiguity
# ---------------------------------------------------------------------------

def test_commit_seq_monotonicity_and_contiguity():
    d = _tmp_store()
    try:
        store = _build_store(d)
        # 5 bars interleaved with 3 events → 8 total
        items = [
            _make_bar(0, 1000),
            _make_event(1001, "state_transition"),
            _make_bar(1, 1060),
            _make_event(1061, "halt"),
            _make_bar(2, 1120),
            _make_event(1121, "resume"),
            _make_bar(3, 1180),
            _make_bar(4, 1240),
        ]
        for item in items:
            key = store.stage_bundle(item)
            store.commit_bundle(key)

        # Verify contiguous commit_seq 0..7
        for seq in range(8):
            fpath = os.path.join(d, "committed", f"{seq}.bundle")
            assert os.path.exists(fpath), f"Missing {seq}.bundle"
            with open(fpath, "r") as f:
                bd = json.load(f)
            assert bd["commit_seq"] == seq, f"Wrong commit_seq at {seq}: {bd['commit_seq']}"

        ok, err = store.verify_chain_integrity()
        assert ok, f"Chain integrity failed: {err}"
        print("  PASS: commit_seq monotonicity and contiguity")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: Hash chain integrity
# ---------------------------------------------------------------------------

def test_hash_chain_integrity():
    d = _tmp_store()
    try:
        store = _build_store(d)
        for i in range(5):
            key = store.stage_bundle(_make_bar(i, 1000 + i * 60))
            store.commit_bundle(key)

        ok, err = store.verify_chain_integrity()
        assert ok, f"Pre-corruption check failed: {err}"

        # Corrupt bundle 2
        p = os.path.join(d, "committed", "2.bundle")
        with open(p, "r") as f:
            bd = json.load(f)
        bd["candle_inputs"]["close"]["v"] = 999_99  # tamper
        with open(p, "w") as f:
            json.dump(bd, f)

        ok, err = store.verify_chain_integrity()
        assert not ok, "Should detect corruption"
        assert "packet_hash mismatch" in err
        print("  PASS: Hash chain integrity + corruption detection")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: Genesis rules
# ---------------------------------------------------------------------------

def test_genesis_rules():
    d = _tmp_store()
    try:
        store = _build_store(d)
        key = store.stage_bundle(_make_bar(0, 1000))
        store.commit_bundle(key)

        p = os.path.join(d, "committed", "0.bundle")
        with open(p, "r") as f:
            bd = json.load(f)
        assert bd["commit_seq"] == 0
        assert bd["prev_packet_hash"] == ""
        assert bd["packet_hash"] != ""
        assert len(bd["packet_hash"]) == 64
        print("  PASS: Genesis rules (commit_seq=0, prev_packet_hash='', non-empty packet_hash)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: Store authority enforcement
# ---------------------------------------------------------------------------

def test_store_authority_enforcement():
    d = _tmp_store()
    try:
        store = _build_store(d)

        # Non-sentinel commit_seq
        bad = BarBundle(
            format_version=1, bar_index=0, timestamp=1000, pod_id="pod-1",
            core_version_hash="abc123",
            candle_inputs={}, system_inputs={}, period_data=None,
            indicator_outputs={}, engine_state_hash="x", health_state={},
            commit_seq=5, prev_packet_hash="", packet_hash="",
        )
        try:
            store.stage_bundle(bad)
            assert False, "Should have raised StoreAuthorityError"
        except StoreAuthorityError as e:
            assert "commit_seq" in str(e)

        # Non-sentinel prev_packet_hash
        bad2 = BarBundle(
            format_version=1, bar_index=0, timestamp=1000, pod_id="pod-1",
            core_version_hash="abc123",
            candle_inputs={}, system_inputs={}, period_data=None,
            indicator_outputs={}, engine_state_hash="x", health_state={},
            commit_seq=-1, prev_packet_hash="abc", packet_hash="",
        )
        try:
            store.stage_bundle(bad2)
            assert False, "Should have raised StoreAuthorityError"
        except StoreAuthorityError as e:
            assert "prev_packet_hash" in str(e)

        # Non-sentinel packet_hash
        bad3 = BarBundle(
            format_version=1, bar_index=0, timestamp=1000, pod_id="pod-1",
            core_version_hash="abc123",
            candle_inputs={}, system_inputs={}, period_data=None,
            indicator_outputs={}, engine_state_hash="x", health_state={},
            commit_seq=-1, prev_packet_hash="", packet_hash="deadbeef",
        )
        try:
            store.stage_bundle(bad3)
            assert False, "Should have raised StoreAuthorityError"
        except StoreAuthorityError as e:
            assert "packet_hash" in str(e)

        print("  PASS: Store authority enforcement (non-sentinel rejection)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 5: Caller sentinel acceptance
# ---------------------------------------------------------------------------

def test_caller_sentinel_acceptance():
    d = _tmp_store()
    try:
        store = _build_store(d)
        bar = _make_bar(0, 1000)
        assert bar.commit_seq == -1
        assert bar.prev_packet_hash == ""
        assert bar.packet_hash == ""

        key = store.stage_bundle(bar)
        store.commit_bundle(key)

        p = os.path.join(d, "committed", "0.bundle")
        with open(p, "r") as f:
            bd = json.load(f)
        assert bd["commit_seq"] == 0
        assert bd["packet_hash"] != ""
        assert len(bd["packet_hash"]) == 64
        # prev_packet_hash is "" for genesis, which is correct
        print("  PASS: Caller sentinel acceptance + store sets real values")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 6: Manifest immutability (SHADOW/LIVE)
# ---------------------------------------------------------------------------

def test_manifest_immutability_shadow_live():
    d = _tmp_store()
    try:
        store = _build_store(d)
        m = _make_manifest()
        store.create_manifest(m, RunMode.SHADOW)

        try:
            store.create_manifest(m, RunMode.SHADOW)
            assert False, "Should have raised ManifestImmutabilityError"
        except ManifestImmutabilityError:
            pass

        try:
            store.create_manifest(m, RunMode.LIVE)
            assert False, "Should have raised ManifestImmutabilityError"
        except ManifestImmutabilityError:
            pass

        print("  PASS: Manifest immutability (SHADOW/LIVE reject overwrite)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 7: Manifest immutability (RESEARCH default)
# ---------------------------------------------------------------------------

def test_manifest_immutability_research_default():
    d = _tmp_store()
    try:
        store = _build_store(d)
        store.create_manifest(_make_manifest(), RunMode.RESEARCH)

        try:
            store.create_manifest(_make_manifest(), RunMode.RESEARCH, allow_overwrite=False)
            assert False, "Should have raised ManifestImmutabilityError"
        except ManifestImmutabilityError:
            pass

        print("  PASS: Manifest immutability (RESEARCH default rejects)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 8: Manifest overwrite (RESEARCH debug)
# ---------------------------------------------------------------------------

def test_manifest_overwrite_research_debug():
    d = _tmp_store()
    try:
        store = _build_store(d)
        store.create_manifest(_make_manifest("run-1"), RunMode.RESEARCH)
        # Should succeed with allow_overwrite=True
        store.create_manifest(_make_manifest("run-2"), RunMode.RESEARCH, allow_overwrite=True)

        with open(os.path.join(d, "manifest.json"), "r") as f:
            data = json.load(f)
        assert data["run_id"] == "run-2"
        print("  PASS: Manifest overwrite (RESEARCH debug, allow_overwrite=True)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 9: Recovery — committed/ is source of truth
# ---------------------------------------------------------------------------

def test_recovery_committed_source_of_truth():
    d = _tmp_store()
    try:
        store = _build_store(d)
        for i in range(3):
            key = store.stage_bundle(_make_bar(i, 1000 + i * 60))
            store.commit_bundle(key)

        # Delete watermark
        wm_path = os.path.join(d, "watermark.json")
        os.remove(wm_path)

        # Recover
        store2 = _build_store(d)
        assert store2.watermark.commit_seq == 2
        assert store2.watermark.packet_hash != ""
        print("  PASS: Recovery rebuilds watermark from committed/")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 10: Recovery — watermark disagreement
# ---------------------------------------------------------------------------

def test_recovery_watermark_disagreement():
    d = _tmp_store()
    try:
        store = _build_store(d)
        for i in range(3):
            key = store.stage_bundle(_make_bar(i, 1000 + i * 60))
            store.commit_bundle(key)

        real_hash = store.watermark.packet_hash

        # Write wrong watermark
        wm_path = os.path.join(d, "watermark.json")
        bad_wm = {"commit_seq": 99, "packet_hash": "wrong_hash", "bar_index": 99}
        with open(wm_path, "w") as f:
            json.dump(bad_wm, f)

        # Recover
        store2 = _build_store(d)
        assert store2.watermark.commit_seq == 2
        assert store2.watermark.packet_hash == real_hash
        print("  PASS: Recovery corrects watermark disagreement")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 11: Recovery — staged cleanup
# ---------------------------------------------------------------------------

def test_recovery_staged_cleanup():
    d = _tmp_store()
    try:
        store = _build_store(d)
        # Stage but don't commit
        key1 = store.stage_bundle(_make_bar(0, 1000))
        key2 = store.stage_bundle(_make_event(1001))

        assert os.path.exists(key1.staged_path)
        assert os.path.exists(key2.staged_path)

        # Recover
        store2 = _build_store(d)
        staged_files = os.listdir(os.path.join(d, "staged"))
        assert len(staged_files) == 0, f"Staged files not cleaned: {staged_files}"
        print("  PASS: Recovery deletes all staged files")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 12: Recovery — crash after commit before watermark update
# ---------------------------------------------------------------------------

def test_recovery_crash_after_commit_before_watermark():
    d = _tmp_store()
    try:
        store = _build_store(d)
        for i in range(3):
            key = store.stage_bundle(_make_bar(i, 1000 + i * 60))
            store.commit_bundle(key)

        # Simulate: delete watermark after committing
        os.remove(os.path.join(d, "watermark.json"))

        # Recover
        store2 = _build_store(d)
        assert store2.watermark.commit_seq == 2
        ok, err = store2.verify_chain_integrity()
        assert ok, f"Chain broken after recovery: {err}"
        print("  PASS: Recovery after crash-before-watermark-update")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 13: Recovery — committed gap detection
# ---------------------------------------------------------------------------

def test_recovery_committed_gap_detection():
    d = _tmp_store()
    try:
        store = _build_store(d)
        for i in range(4):
            key = store.stage_bundle(_make_bar(i, 1000 + i * 60))
            store.commit_bundle(key)

        # Create a gap: delete bundle 2
        os.remove(os.path.join(d, "committed", "2.bundle"))

        ok, err = store.verify_chain_integrity()
        assert not ok, "Should detect gap"
        assert "gap" in err.lower() or "Missing" in err
        print("  PASS: verify_chain_integrity detects committed gap")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 14: Preflight per mode
# ---------------------------------------------------------------------------

def test_preflight_per_mode():
    d = _tmp_store()
    try:
        store = _build_store(d)
        key = store.stage_bundle(_make_bar(0, 1000))
        store.commit_bundle(key)

        # Healthy preflight
        assert store.preflight_check(RunMode.SHADOW) is True
        assert store.preflight_check(RunMode.LIVE) is True
        assert store.preflight_check(RunMode.RESEARCH) is True

        # Corrupt watermark
        wm_path = os.path.join(d, "watermark.json")
        with open(wm_path, "w") as f:
            f.write("{bad json")

        # SHADOW/LIVE should fail
        store._watermark.commit_seq  # still cached, but preflight re-reads file
        assert store.preflight_check(RunMode.SHADOW) is False
        assert store.preflight_check(RunMode.LIVE) is False
        # RESEARCH continues
        assert store.preflight_check(RunMode.RESEARCH) is True

        print("  PASS: Preflight per mode (SHADOW/LIVE fail, RESEARCH continues)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 15: Preflight scratch isolation
# ---------------------------------------------------------------------------

def test_preflight_scratch_isolation():
    d = _tmp_store()
    try:
        store = _build_store(d)

        # Preflight should not leave files in scratch/
        store.preflight_check(RunMode.SHADOW)
        scratch_files = os.listdir(os.path.join(d, "scratch"))
        assert len(scratch_files) == 0, f"Preflight left files: {scratch_files}"

        # Create a file in scratch/, verify recovery and integrity ignore it
        rogue = os.path.join(d, "scratch", "rogue_file.txt")
        with open(rogue, "w") as f:
            f.write("should be ignored")

        key = store.stage_bundle(_make_bar(0, 1000))
        store.commit_bundle(key)

        ok, err = store.verify_chain_integrity()
        assert ok, f"Integrity should ignore scratch/: {err}"

        store2 = _build_store(d)  # recovery runs
        assert os.path.exists(rogue), "Recovery should NOT touch scratch/"

        print("  PASS: Preflight scratch isolation (writes+deletes, ignored by recovery/integrity)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 16: No float/Decimal in bundles
# ---------------------------------------------------------------------------

def _check_no_float(obj, path=""):
    """Recursively check no float or Decimal in JSON-parsed structure."""
    if isinstance(obj, float):
        raise AssertionError(f"Float found at {path}: {obj}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_no_float(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_no_float(v, f"{path}[{i}]")


def test_no_float_in_bundles():
    d = _tmp_store()
    try:
        store = _build_store(d)
        for i in range(3):
            key = store.stage_bundle(_make_bar(i, 1000 + i * 60))
            store.commit_bundle(key)
        key = store.stage_bundle(_make_event(2000))
        store.commit_bundle(key)

        for seq in range(4):
            fpath = os.path.join(d, "committed", f"{seq}.bundle")
            with open(fpath, "r") as f:
                bd = json.load(f)
            _check_no_float(bd, f"bundle[{seq}]")

        print("  PASS: No float/Decimal in committed bundles")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 17: Replay determinism
# ---------------------------------------------------------------------------

def test_replay_determinism():
    d = _tmp_store()
    try:
        store = _build_store(d)
        for i in range(5):
            key = store.stage_bundle(_make_bar(i, 1000 + i * 60))
            store.commit_bundle(key)

        r1 = store.replay(0, 4)
        r2 = store.replay(0, 4)

        for i, (b1, b2) in enumerate(zip(r1, r2)):
            assert b1.commit_seq == b2.commit_seq == i
            assert b1.packet_hash == b2.packet_hash
            assert b1.prev_packet_hash == b2.prev_packet_hash

        print("  PASS: Replay determinism (identical content)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 18: Serialization (concurrent commits)
# ---------------------------------------------------------------------------

def test_concurrent_commits():
    d = _tmp_store()
    try:
        store = _build_store(d)
        errors: list = []
        keys: list = []

        # Pre-stage all bundles (lock-free)
        for i in range(10):
            keys.append(store.stage_bundle(_make_bar(i, 1000 + i * 60)))

        # Commit from two threads
        def commit_range(start, end):
            try:
                for i in range(start, end):
                    store.commit_bundle(keys[i])
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=commit_range, args=(0, 5))
        t2 = threading.Thread(target=commit_range, args=(5, 10))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Errors during concurrent commit: {errors}"

        # Verify contiguous commit_seq
        seqs = set()
        for seq in range(10):
            fpath = os.path.join(d, "committed", f"{seq}.bundle")
            assert os.path.exists(fpath), f"Missing {seq}.bundle"
            with open(fpath, "r") as f:
                bd = json.load(f)
            seqs.add(bd["commit_seq"])
        assert seqs == set(range(10))

        ok, err = store.verify_chain_integrity()
        assert ok, f"Chain integrity after concurrent commits: {err}"

        print("  PASS: Concurrent commits produce deterministic contiguous commit_seq")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 19: Same-volume check (mock os.replace failure)
# ---------------------------------------------------------------------------

def test_same_volume_check():
    d = _tmp_store()
    try:
        # Mock os.replace to fail
        original_replace = os.replace

        def failing_replace(src, dst):
            if "_vol_check_" in src:
                raise OSError("Cross-device link")
            return original_replace(src, dst)

        with patch("phase5_bundle_store.os.replace", side_effect=failing_replace):
            try:
                _build_store(_tmp_store())
                assert False, "Should have raised StoreInitError"
            except StoreInitError as e:
                assert "same filesystem" in str(e)

        print("  PASS: Same-volume check detects cross-device configuration")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("verify_bundle_store.py — 19 verification items")
    print("=" * 60)

    tests = [
        ("1. commit_seq monotonicity + contiguity", test_commit_seq_monotonicity_and_contiguity),
        ("2. Hash chain integrity", test_hash_chain_integrity),
        ("3. Genesis rules", test_genesis_rules),
        ("4. Store authority enforcement", test_store_authority_enforcement),
        ("5. Caller sentinel acceptance", test_caller_sentinel_acceptance),
        ("6. Manifest immutability (SHADOW/LIVE)", test_manifest_immutability_shadow_live),
        ("7. Manifest immutability (RESEARCH default)", test_manifest_immutability_research_default),
        ("8. Manifest overwrite (RESEARCH debug)", test_manifest_overwrite_research_debug),
        ("9. Recovery: committed/ source of truth", test_recovery_committed_source_of_truth),
        ("10. Recovery: watermark disagreement", test_recovery_watermark_disagreement),
        ("11. Recovery: staged cleanup", test_recovery_staged_cleanup),
        ("12. Recovery: crash after commit", test_recovery_crash_after_commit_before_watermark),
        ("13. Recovery: committed gap detection", test_recovery_committed_gap_detection),
        ("14. Preflight per mode", test_preflight_per_mode),
        ("15. Preflight scratch isolation", test_preflight_scratch_isolation),
        ("16. No float/Decimal in bundles", test_no_float_in_bundles),
        ("17. Replay determinism", test_replay_determinism),
        ("18. Concurrent commits (serialization)", test_concurrent_commits),
        ("19. Same-volume check", test_same_volume_check),
    ]

    passed = 0
    failed = 0
    for label, fn in tests:
        print(f"\n[{label}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
