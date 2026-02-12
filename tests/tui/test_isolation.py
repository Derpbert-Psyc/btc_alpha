"""Gate 8: TUI isolation — no ui.* imports, reads only status files, boots cleanly."""

import importlib
import inspect
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_tui_does_not_import_ui():
    """TUI modules must NOT import anything from ui.* package."""
    tui_modules = [
        "tui",
        "tui.services.status_reader",
        "tui.screens.fleet",
        "tui.screens.cockpit",
        "tui.screens.halt",
        "tui.app",
    ]

    for mod_name in tui_modules:
        mod = importlib.import_module(mod_name)
        source_file = inspect.getfile(mod)
        with open(source_file) as f:
            source = f.read()

        # Check for ui.* imports
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "from ui." not in stripped and "import ui." not in stripped, (
                f"{mod_name} ({source_file}) imports from ui.*: {stripped}"
            )


def test_tui_does_not_import_phase5_pod():
    """TUI must not import phase5_pod directly (reads status files instead)."""
    tui_modules = [
        "tui.services.status_reader",
        "tui.screens.fleet",
        "tui.screens.cockpit",
        "tui.screens.halt",
        "tui.app",
    ]

    for mod_name in tui_modules:
        mod = importlib.import_module(mod_name)
        source_file = inspect.getfile(mod)
        with open(source_file) as f:
            source = f.read()

        for line in source.splitlines():
            stripped = line.strip()
            # Skip comments and docstrings
            if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                continue
            # Check actual import statements only
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert "phase5_pod" not in stripped, (
                    f"{mod_name} imports phase5_pod: {stripped}"
                )
                assert "phase5_cli" not in stripped, (
                    f"{mod_name} imports phase5_cli: {stripped}"
                )


def test_status_reader_reads_status_files(tmp_path):
    """Status reader correctly parses pod status JSON."""
    from tui.services.status_reader import read_pod_status, scan_pod_statuses

    # Create a fake status file
    pod_dir = tmp_path / "pod_abc123"
    pod_dir.mkdir()
    status_data = {
        "pod_id": "abc123",
        "run_id": "run_001",
        "run_mode": "RESEARCH",
        "state": "RUNNING",
        "health": "HEALTHY",
        "halt_reason": None,
        "bar_counter": 42,
        "last_bar_ts": 1700000000,
        "components": {
            "core": {"state": "HEALTHY", "reason": ""},
            "persistence": {"state": "HEALTHY", "reason": ""},
        },
        "watermark": {"commit_seq": 5},
    }
    status_file = pod_dir / "status.json"
    status_file.write_text(json.dumps(status_data))

    # Read single
    status = read_pod_status(str(status_file))
    assert status.pod_id == "abc123"
    assert status.run_mode == "RESEARCH"
    assert status.state == "RUNNING"
    assert status.health == "HEALTHY"
    assert status.bar_counter == 42
    assert status.is_healthy
    assert status.is_running
    assert not status.is_halted

    # Scan
    statuses = scan_pod_statuses(str(tmp_path))
    assert len(statuses) == 1
    assert statuses[0].pod_id == "abc123"


def test_status_reader_empty_dir(tmp_path):
    """Scan returns empty list when no status files exist."""
    from tui.services.status_reader import scan_pod_statuses

    statuses = scan_pod_statuses(str(tmp_path))
    assert statuses == []

    # Non-existent dir also returns empty
    statuses = scan_pod_statuses(str(tmp_path / "nonexistent"))
    assert statuses == []


def test_halt_request_write(tmp_path):
    """Halt request writes correct JSON file."""
    from tui.services.status_reader import write_halt_request

    filepath = write_halt_request(str(tmp_path), "pod_xyz", "Manual halt test")
    assert os.path.exists(filepath)

    with open(filepath) as f:
        data = json.load(f)
    assert data["pod_id"] == "pod_xyz"
    assert data["reason"] == "Manual halt test"
    assert "requested_at" in data


def test_tui_boots_without_status_files():
    """OpsConsole can be instantiated without any pod status files."""
    from tui.app import OpsConsole

    # Should not raise
    app = OpsConsole(status_dir="/tmp/nonexistent_tui_test_dir")
    assert app.status_dir == "/tmp/nonexistent_tui_test_dir"
    assert app.TITLE == "BTC Alpha Ops Console"
