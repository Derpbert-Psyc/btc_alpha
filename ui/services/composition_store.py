"""Composition store — CRUD for composition specs + index.json with atomic writes."""

import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "research")
INDEX_PATH = os.path.join(RESEARCH_DIR, "index.json")


def _ensure_dirs():
    os.makedirs(os.path.join(RESEARCH_DIR, "compositions"), exist_ok=True)
    os.makedirs(os.path.join(RESEARCH_DIR, "strategies"), exist_ok=True)
    os.makedirs(os.path.join(RESEARCH_DIR, "promotions"), exist_ok=True)
    os.makedirs(os.path.join(RESEARCH_DIR, "triage_results"), exist_ok=True)
    os.makedirs(os.path.join(RESEARCH_DIR, "sweep_results"), exist_ok=True)
    os.makedirs(os.path.join(RESEARCH_DIR, ".debug"), exist_ok=True)


def _atomic_write_json(path: str, data: Any) -> None:
    """Write JSON atomically: write tmp, fsync, rename."""
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_index() -> Dict[str, Any]:
    """Load index.json, returning empty structure if missing."""
    _ensure_dirs()
    if not os.path.exists(INDEX_PATH):
        return {"compositions": {}}
    try:
        with open(INDEX_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"compositions": {}}


def save_index(index: Dict[str, Any]) -> None:
    """Save index.json atomically."""
    _ensure_dirs()
    _atomic_write_json(INDEX_PATH, index)


def _composition_dir(composition_id: str) -> str:
    return os.path.join(RESEARCH_DIR, "compositions", composition_id)


def _composition_path(composition_id: str) -> str:
    return os.path.join(_composition_dir(composition_id), "composition.json")


def create_composition(spec: Dict[str, Any], display_name: str = "Untitled") -> str:
    """Create a new composition. Returns composition_id."""
    composition_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    spec["composition_id"] = composition_id
    spec.setdefault("display_name", display_name)
    spec.setdefault("spec_version", "1.5.2")
    spec.setdefault("target_engine_version", "1.8.0")
    spec.setdefault("target_instrument", "BTCUSDT")
    spec.setdefault("target_variant", "perp")
    spec.setdefault("metadata", {})
    spec["metadata"].setdefault("created", now)
    spec["metadata"]["updated_at"] = now

    _save_composition(composition_id, spec)
    _update_index_entry(composition_id, display_name, now)
    return composition_id


def save_composition(composition_id: str, spec: Dict[str, Any]) -> None:
    """Save (update) an existing composition."""
    now = datetime.now(timezone.utc).isoformat()
    spec["composition_id"] = composition_id
    spec.setdefault("metadata", {})
    spec["metadata"]["updated_at"] = now
    _save_composition(composition_id, spec)

    index = load_index()
    if composition_id in index["compositions"]:
        index["compositions"][composition_id]["updated_at"] = now
        index["compositions"][composition_id]["display_name"] = spec.get(
            "display_name", index["compositions"][composition_id].get("display_name", "Untitled"))
    save_index(index)


def load_composition(composition_id: str) -> Optional[Dict[str, Any]]:
    """Load a composition spec from disk."""
    path = _composition_path(composition_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def list_compositions() -> List[Dict[str, Any]]:
    """List all compositions from index.json."""
    index = load_index()
    result = []
    for cid, entry in index.get("compositions", {}).items():
        result.append({"composition_id": cid, **entry})
    return result


def delete_composition(composition_id: str) -> None:
    """Remove a composition from index (does not delete files)."""
    index = load_index()
    index["compositions"].pop(composition_id, None)
    save_index(index)


def update_compiled_hash(composition_id: str, strategy_config_hash: str) -> None:
    """Update the latest_compiled_hash in the index."""
    index = load_index()
    if composition_id in index["compositions"]:
        index["compositions"][composition_id]["latest_compiled_hash"] = strategy_config_hash
        index["compositions"][composition_id]["updated_at"] = (
            datetime.now(timezone.utc).isoformat())
    save_index(index)


def duplicate_composition(source_id: str, new_name: Optional[str] = None) -> str:
    """Duplicate a composition as a variant. Returns new composition_id."""
    spec = load_composition(source_id)
    if spec is None:
        raise ValueError(f"Composition {source_id} not found")

    new_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    spec["composition_id"] = new_id
    if new_name:
        spec["display_name"] = new_name
    else:
        spec["display_name"] = spec.get("display_name", "Untitled") + " (variant)"

    spec.setdefault("metadata", {})
    spec["metadata"]["forked_from"] = source_id
    spec["metadata"]["created"] = now
    spec["metadata"]["updated_at"] = now
    spec["metadata"].pop("archive_reason", None)

    _save_composition(new_id, spec)
    _update_index_entry(new_id, spec["display_name"], now)
    return new_id


def archive_composition(composition_id: str, reason: str) -> None:
    """Archive a composition with a reason."""
    spec = load_composition(composition_id)
    if spec is None:
        return
    spec.setdefault("metadata", {})
    spec["metadata"]["archive_reason"] = reason
    spec["metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_composition(composition_id, spec)


def _save_composition(composition_id: str, spec: Dict[str, Any]) -> None:
    """Write composition.json atomically."""
    path = _composition_path(composition_id)
    _atomic_write_json(path, spec)


def _update_index_entry(composition_id: str, display_name: str, now: str) -> None:
    """Add or update an index entry."""
    index = load_index()
    index["compositions"][composition_id] = {
        "display_name": display_name,
        "latest_compiled_hash": None,
        "created": now,
        "updated_at": now,
    }
    save_index(index)
