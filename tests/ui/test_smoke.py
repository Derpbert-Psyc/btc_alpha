"""Gate 1: UI smoke tests — no data required."""

import json
import os
import shutil
import sys
import tempfile
import pytest

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def clean_research(tmp_path, monkeypatch):
    """Provide a clean research directory for tests."""
    research_dir = str(tmp_path / "research")
    os.makedirs(research_dir, exist_ok=True)
    # Patch the composition store to use tmp dir
    import ui.services.composition_store as cs
    monkeypatch.setattr(cs, "RESEARCH_DIR", research_dir)
    monkeypatch.setattr(cs, "INDEX_PATH", os.path.join(research_dir, "index.json"))
    # Patch promotion reader too
    import ui.services.promotion_reader as pr
    monkeypatch.setattr(pr, "RESEARCH_DIR", research_dir)
    # Patch compiler bridge
    import ui.services.compiler_bridge as cb
    monkeypatch.setattr(cb, "RESEARCH_DIR", research_dir)
    monkeypatch.setattr(cb, "DEBUG_DIR", os.path.join(research_dir, ".debug"))
    return research_dir


def test_import_ui_app():
    """UI module imports without side effects."""
    import ui.app  # noqa: F401


def test_import_services():
    """Service modules import cleanly."""
    import ui.services.composition_store  # noqa: F401
    import ui.services.compiler_bridge  # noqa: F401
    import ui.services.indicator_catalog  # noqa: F401
    import ui.services.promotion_reader  # noqa: F401


def test_empty_strategy_list(clean_research):
    """Strategy list works with empty research dir."""
    from ui.services.composition_store import list_compositions
    result = list_compositions()
    assert result == []


def test_create_composition(clean_research):
    """Creating a composition writes composition.json and index.json."""
    from ui.services.composition_store import (
        create_composition, load_composition, load_index,
    )
    spec = {
        "indicator_instances": [],
        "entry_rules": [],
        "exit_rules": [],
        "gate_rules": [],
        "archetype_tags": [],
        "execution_params": {},
        "metadata": {},
    }
    cid = create_composition(spec, display_name="Test Strategy")
    assert cid is not None

    # Verify files
    loaded = load_composition(cid)
    assert loaded is not None
    assert loaded["display_name"] == "Test Strategy"
    assert loaded["spec_version"] == "1.5.2"
    assert loaded["composition_id"] == cid

    index = load_index()
    assert cid in index["compositions"]
    assert index["compositions"][cid]["display_name"] == "Test Strategy"


def test_load_preset(clean_research):
    """Loading a preset creates a new composition with new UUID."""
    from ui.services.composition_store import create_composition, load_composition

    preset_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "ui", "presets", "macd_confluence_long.json")
    with open(preset_path) as f:
        spec = json.load(f)
    spec.pop("composition_id", None)
    cid = create_composition(spec, display_name=spec.get("display_name", "MACD"))

    loaded = load_composition(cid)
    assert loaded is not None
    assert loaded["display_name"] == "MACD Confluence Long"
    assert len(loaded["indicator_instances"]) == 8
    assert loaded["composition_id"] == cid  # New UUID


def test_compile_preset(clean_research):
    """Compile MACD Confluence preset — returns compilation result with hash."""
    from ui.services.composition_store import create_composition, load_composition
    from ui.services.compiler_bridge import compile_spec

    preset_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "ui", "presets", "macd_confluence_long.json")
    with open(preset_path) as f:
        spec = json.load(f)
    spec.pop("composition_id", None)
    cid = create_composition(spec, display_name="MACD Confluence")
    loaded = load_composition(cid)

    result = compile_spec(loaded)
    assert "strategy_config_hash" in result
    assert result["strategy_config_hash"].startswith("sha256:")
    assert "resolved_artifact" in result
    assert "lowering_report" in result
