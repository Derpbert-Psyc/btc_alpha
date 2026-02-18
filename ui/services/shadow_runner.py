"""Shadow runner client -- spawns daemon subprocess, reads status files.

The UI never runs the WebSocket or trading logic directly.
It communicates with the shadow_daemon.py process via filesystem:
  - Start: writes config.json, then subprocess.Popen(shadow_daemon.py ...)
  - Stop: write command.json with {"command": "stop"} (atomic)
  - Status: read status.json
"""

import json
import os
import subprocess
import sys
import time
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STATUS_BASE = os.path.join(PROJECT_ROOT, "research", "shadow_status")
DAEMON_SCRIPT = os.path.join(PROJECT_ROOT, "shadow_daemon.py")


def start_shadow(instance_id: str, config: dict) -> bool:
    """Spawn shadow daemon as a background subprocess.

    Args:
        instance_id: unique identifier (e.g., "bybit-cx", "binance-big")
        config: full daemon configuration dict

    Returns True if daemon was started successfully.
    """
    status_dir = os.path.join(STATUS_BASE, instance_id)
    os.makedirs(status_dir, exist_ok=True)

    # Check if already running (status file fresh and not STOPPED)
    existing = read_shadow_status(instance_id)
    if existing and existing.get("status") in (
        "RUNNING", "CONNECTING", "RECONNECTING", "WARMING_UP", "STABILIZING"
    ):
        return True  # Already running

    # Write config.json (so daemon and systemd can read it)
    config_path = os.path.join(status_dir, "config.json")
    tmp_config = config_path + ".tmp"
    with open(tmp_config, "w") as f:
        json.dump(config, f, indent=2)
    os.replace(tmp_config, config_path)

    # Spawn daemon using same Python interpreter as the UI
    log_path = os.path.join(status_dir, "daemon.log")
    log_file = open(log_path, "a")
    proc = subprocess.Popen(
        [sys.executable, DAEMON_SCRIPT, "--instance-id", instance_id],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # Detach from parent (Linux/VPS deployment target)
    )
    log_file.close()  # Parent closes its copy; child keeps its own FD

    # Do NOT sleep -- the UI timer reads status.json when daemon writes it
    if proc.poll() is not None:
        return False  # Process exited immediately

    return True


def stop_shadow(instance_id: str):
    """Send stop command to the shadow daemon (atomic write)."""
    status_dir = os.path.join(STATUS_BASE, instance_id)
    command_path = os.path.join(status_dir, "command.json")
    tmp_path = command_path + ".tmp"
    os.makedirs(status_dir, exist_ok=True)

    with open(tmp_path, "w") as f:
        json.dump({"command": "stop", "requested_at": time.time()}, f)
    os.replace(tmp_path, command_path)


def read_shadow_status(instance_id: str) -> Optional[dict]:
    """Read current shadow daemon status from filesystem.

    Returns None if status file doesn't exist.
    Marks status as STALE if updated_at > 30s old and status was RUNNING.
    """
    status_path = os.path.join(STATUS_BASE, instance_id, "status.json")
    try:
        with open(status_path) as f:
            status = json.load(f)

        # Stale detection
        updated = status.get("updated_at", 0)
        if (time.time() - updated) > 30 and status.get("status") in (
            "RUNNING", "WARMING_UP", "CONNECTING", "RECONNECTING", "STABILIZING"
        ):
            status["status"] = "STALE"

        return status
    except (json.JSONDecodeError, IOError, FileNotFoundError):
        return None


def list_shadow_instances() -> list:
    """List all shadow daemon instance statuses."""
    instances = []
    if not os.path.isdir(STATUS_BASE):
        return instances
    for name in sorted(os.listdir(STATUS_BASE)):
        st = read_shadow_status(name)
        if st:
            instances.append(st)
    return instances
