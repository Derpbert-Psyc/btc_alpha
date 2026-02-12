"""Gate 2: Artifact immutability and lifecycle derivation."""

import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def clean_research(tmp_path, monkeypatch):
    """Provide a clean research directory."""
    research_dir = str(tmp_path / "research")
    os.makedirs(research_dir, exist_ok=True)
    import ui.services.composition_store as cs
    monkeypatch.setattr(cs, "RESEARCH_DIR", research_dir)
    monkeypatch.setattr(cs, "INDEX_PATH", os.path.join(research_dir, "index.json"))
    import ui.services.promotion_reader as pr
    monkeypatch.setattr(pr, "RESEARCH_DIR", research_dir)
    import ui.services.compiler_bridge as cb
    monkeypatch.setattr(cb, "RESEARCH_DIR", research_dir)
    monkeypatch.setattr(cb, "DEBUG_DIR", os.path.join(research_dir, ".debug"))
    return research_dir


def _load_macd_spec():
    preset_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "ui", "presets", "macd_confluence.json")
    with open(preset_path) as f:
        return json.load(f)


def test_compile_same_spec_twice_same_hash(clean_research):
    """Compile same spec twice in fresh calls → identical strategy_config_hash."""
    from ui.services.compiler_bridge import compile_spec

    spec = _load_macd_spec()
    r1 = compile_spec(spec)
    r2 = compile_spec(spec)
    assert r1["strategy_config_hash"] == r2["strategy_config_hash"]


def test_write_artifacts_idempotent(clean_research):
    """write_artifacts twice → second call is no-op (same content exists)."""
    from ui.services.compiler_bridge import compile_spec, save_artifacts

    spec = _load_macd_spec()
    result = compile_spec(spec)

    dir1 = save_artifacts(result)
    dir2 = save_artifacts(result)  # Should succeed — same content
    assert dir1 == dir2
    assert os.path.exists(os.path.join(dir1, "resolved.json"))
    assert os.path.exists(os.path.join(dir1, "lowering_report.json"))


def test_atomic_index_write(clean_research):
    """Atomic write: tmp without rename → loader ignores tmp, uses last good."""
    from ui.services.composition_store import load_index, save_index

    # Write good index
    save_index({"compositions": {"test": {"display_name": "Test"}}})

    # Create a stale .tmp file (simulating crash)
    index_dir = os.path.dirname(os.path.join(clean_research, "index.json"))
    tmp_path = os.path.join(index_dir, "index.json.tmp")
    with open(tmp_path, "w") as f:
        f.write("{corrupt data")

    # Load should return good index, ignoring .tmp
    index = load_index()
    assert "test" in index["compositions"]


def test_lifecycle_draft_no_hash(clean_research):
    """No compiled hash → DRAFT state."""
    from ui.services.promotion_reader import derive_lifecycle_state
    state, warning = derive_lifecycle_state("test-id", None)
    assert state == "DRAFT"
    assert warning is None


def test_lifecycle_compiled_with_artifact(clean_research):
    """Compiled hash with artifact → COMPILED state."""
    from ui.services.compiler_bridge import compile_spec, save_artifacts
    from ui.services.promotion_reader import derive_lifecycle_state

    spec = _load_macd_spec()
    result = compile_spec(spec)
    save_artifacts(result)

    hash_val = result["strategy_config_hash"]
    state, warning = derive_lifecycle_state("test-id", hash_val)
    assert state == "COMPILED"
    assert warning is None


def test_lifecycle_corrupted_missing_artifact(clean_research):
    """Compiled hash but no artifact files → CORRUPTED."""
    from ui.services.promotion_reader import derive_lifecycle_state
    state, warning = derive_lifecycle_state("test-id", "sha256:nonexistent")
    assert state == "CORRUPTED"
    assert warning is not None


def test_duplicate_as_variant(clean_research):
    """Duplicate as Variant creates new UUID with forked_from set."""
    from ui.services.composition_store import (
        create_composition, duplicate_composition, load_composition,
    )

    spec = _load_macd_spec()
    spec.pop("composition_id", None)
    source_id = create_composition(spec, display_name="Original")

    new_id = duplicate_composition(source_id, "Variant of Original")
    assert new_id != source_id

    variant = load_composition(new_id)
    assert variant is not None
    assert variant["display_name"] == "Variant of Original"
    assert variant["metadata"]["forked_from"] == source_id
    assert variant["composition_id"] == new_id


def test_archive_composition(clean_research):
    """Archive stores reason in metadata."""
    from ui.services.composition_store import (
        create_composition, load_composition, archive_composition,
    )

    spec = _load_macd_spec()
    spec.pop("composition_id", None)
    cid = create_composition(spec, display_name="To Archive")

    archive_composition(cid, "Superseded by v2")
    loaded = load_composition(cid)
    assert loaded["metadata"]["archive_reason"] == "Superseded by v2"
