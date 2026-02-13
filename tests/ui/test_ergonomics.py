"""Test research ergonomics: research_notes, forked_from, user_tags survive save/load cycle."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from ui.services.composition_store import (
    create_composition,
    save_composition,
    load_composition,
    duplicate_composition,
    archive_composition,
)


@pytest.fixture
def clean_research(tmp_path, monkeypatch):
    research_dir = str(tmp_path / "research")
    monkeypatch.setattr("ui.services.composition_store.RESEARCH_DIR", research_dir)
    monkeypatch.setattr("ui.services.composition_store.INDEX_PATH",
                        os.path.join(research_dir, "index.json"))
    os.makedirs(os.path.join(research_dir, "compositions"), exist_ok=True)
    return research_dir


class TestResearchNotes:

    def test_research_notes_survive_save_load(self, clean_research):
        """Add a research note → save → reload → note persists."""
        spec = {
            "display_name": "Test Notes",
            "indicator_instances": [],
            "entry_rules": [],
            "exit_rules": [],
            "gate_rules": [],
            "execution_params": {},
            "metadata": {
                "research_notes": [
                    {"timestamp": "2025-01-01T00:00:00Z", "text": "First observation"}
                ]
            }
        }
        cid = create_composition(spec, "Test Notes")

        # Reload
        loaded = load_composition(cid)
        assert loaded is not None
        notes = loaded["metadata"]["research_notes"]
        assert len(notes) == 1
        assert notes[0]["text"] == "First observation"

        # Add another note and re-save
        loaded["metadata"]["research_notes"].append({
            "timestamp": "2025-01-02T00:00:00Z",
            "text": "Second observation"
        })
        save_composition(cid, loaded)

        # Reload again
        reloaded = load_composition(cid)
        notes2 = reloaded["metadata"]["research_notes"]
        assert len(notes2) == 2
        assert notes2[1]["text"] == "Second observation"


class TestForkedFrom:

    def test_forked_from_populated_on_duplicate(self, clean_research):
        """Duplicate as variant → forked_from is set to source ID."""
        spec = {
            "display_name": "Original",
            "indicator_instances": [],
            "entry_rules": [],
            "exit_rules": [],
            "gate_rules": [],
            "execution_params": {},
            "metadata": {}
        }
        source_id = create_composition(spec, "Original")

        # Duplicate
        new_id = duplicate_composition(source_id, "Variant")

        # Load the variant
        variant = load_composition(new_id)
        assert variant is not None
        assert variant["metadata"]["forked_from"] == source_id
        assert variant["display_name"] == "Variant"

    def test_forked_from_not_set_on_new(self, clean_research):
        """New composition should not have forked_from."""
        spec = {
            "display_name": "Fresh",
            "indicator_instances": [],
            "entry_rules": [],
            "exit_rules": [],
            "gate_rules": [],
            "execution_params": {},
            "metadata": {}
        }
        cid = create_composition(spec, "Fresh")
        loaded = load_composition(cid)
        assert "forked_from" not in loaded.get("metadata", {})


class TestUserTags:

    def test_user_tags_survive_save_load(self, clean_research):
        """User tags persist through save/load cycle."""
        spec = {
            "display_name": "Tagged",
            "indicator_instances": [],
            "entry_rules": [],
            "exit_rules": [],
            "gate_rules": [],
            "execution_params": {},
            "metadata": {
                "user_tags": ["alpha", "beta", "gamma"]
            }
        }
        cid = create_composition(spec, "Tagged")
        loaded = load_composition(cid)
        assert loaded["metadata"]["user_tags"] == ["alpha", "beta", "gamma"]


class TestArchiveReason:

    def test_archive_reason_persists(self, clean_research):
        """Archive with reason → reason visible in spec."""
        spec = {
            "display_name": "To Archive",
            "indicator_instances": [],
            "entry_rules": [],
            "exit_rules": [],
            "gate_rules": [],
            "execution_params": {},
            "metadata": {}
        }
        cid = create_composition(spec, "To Archive")
        archive_composition(cid, "Replaced by v2")

        loaded = load_composition(cid)
        assert loaded["metadata"]["archive_reason"] == "Replaced by v2"


class TestSpecVersion:

    def test_spec_version_set_on_creation(self, clean_research):
        """spec_version should default to 1.5.2."""
        spec = {
            "display_name": "Version Test",
            "indicator_instances": [],
            "entry_rules": [],
            "exit_rules": [],
            "gate_rules": [],
            "execution_params": {},
            "metadata": {}
        }
        cid = create_composition(spec, "Version Test")
        loaded = load_composition(cid)
        assert loaded["spec_version"] == "1.5.2"
