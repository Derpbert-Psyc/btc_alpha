"""Status reader — reads pod status from filesystem.

ISOLATION: This module does NOT import phase5_pod, phase5_cli, or any
project-specific module. It only reads JSON files from the evidence directory.
"""

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PodStatus:
    """Pod status read from filesystem."""
    pod_id: str
    run_id: str = ""
    run_mode: str = "UNKNOWN"
    state: str = "UNKNOWN"
    health: str = "UNKNOWN"
    halt_reason: Optional[str] = None
    bar_counter: int = 0
    last_bar_ts: Optional[int] = None
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    watermark: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_healthy(self) -> bool:
        return self.health == "HEALTHY"

    @property
    def is_halted(self) -> bool:
        return self.state == "HALTED"

    @property
    def is_running(self) -> bool:
        return self.state == "RUNNING"


def read_pod_status(status_path: str) -> Optional[PodStatus]:
    """Read a single pod status from a JSON file.

    Expected format matches phase5_pod.Pod.status() output.
    """
    try:
        with open(status_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        return PodStatus(pod_id="unknown", error=str(e))

    return PodStatus(
        pod_id=data.get("pod_id", "unknown"),
        run_id=data.get("run_id", ""),
        run_mode=data.get("run_mode", "UNKNOWN"),
        state=data.get("state", "UNKNOWN"),
        health=data.get("health", "UNKNOWN"),
        halt_reason=data.get("halt_reason"),
        bar_counter=data.get("bar_counter", 0),
        last_bar_ts=data.get("last_bar_ts"),
        components=data.get("components", {}),
        watermark=data.get("watermark", {}),
        config=data.get("config", {}),
    )


def scan_pod_statuses(status_dir: str) -> List[PodStatus]:
    """Scan directory for pod status files.

    Looks for: {status_dir}/**/status.json
    Returns list of PodStatus, possibly empty.
    """
    if not os.path.isdir(status_dir):
        return []

    pattern = os.path.join(status_dir, "**", "status.json")
    files = glob.glob(pattern, recursive=True)

    statuses = []
    for path in sorted(files):
        status = read_pod_status(path)
        if status is not None:
            statuses.append(status)

    return statuses


def write_halt_request(status_dir: str, pod_id: str, reason: str) -> str:
    """Write a halt request file for a pod.

    The CLI/daemon watches for these files and executes the halt.
    Returns the path of the halt request file.
    """
    halt_dir = os.path.join(status_dir, pod_id, "halt_requests")
    os.makedirs(halt_dir, exist_ok=True)

    import time
    filename = f"halt_{int(time.time())}.json"
    filepath = os.path.join(halt_dir, filename)

    data = {
        "pod_id": pod_id,
        "reason": reason,
        "requested_at": time.time(),
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath
