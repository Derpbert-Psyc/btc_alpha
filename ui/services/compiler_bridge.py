"""Compiler bridge — wraps compile_composition + write_artifacts."""

import json
import os
import traceback
from typing import Any, Dict, Optional, Tuple

from composition_compiler_v1_5_2 import (
    CompilationError,
    compile_composition,
    write_artifacts,
    write_promotion_artifact,
    CAPABILITY_REGISTRY,
)

RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "research")
DEBUG_DIR = os.path.join(RESEARCH_DIR, ".debug")


def compile_spec(spec: dict) -> Dict[str, Any]:
    """Compile a composition spec.

    Returns the compilation result dict on success.
    Raises CompilationError on failure.
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug_path = os.path.join(DEBUG_DIR, "last_compile_input.json")
    try:
        with open(debug_path, "w") as f:
            json.dump(spec, f, indent=2)
    except Exception:
        pass

    return compile_composition(spec)


def save_artifacts(compilation_result: dict) -> str:
    """Write resolved artifact and lowering report to research/strategies/.

    Returns the artifact directory path.
    """
    base_dir = os.path.join(RESEARCH_DIR, "strategies")
    return write_artifacts(compilation_result, base_dir=base_dir)


def save_promotion(
    strategy_config_hash: str,
    composition_id: str,
    composition_spec_hash: str,
    tier: str,
    result: str,
    dataset_hash: str,
    runner_hash: str,
    lowering_report_semantic_hash: str,
) -> str:
    """Write a promotion artifact. Returns file path."""
    base_dir = os.path.join(RESEARCH_DIR, "promotions")
    return write_promotion_artifact(
        strategy_config_hash=strategy_config_hash,
        composition_id=composition_id,
        composition_spec_hash=composition_spec_hash,
        tier=tier,
        result=result,
        dataset_hash=dataset_hash,
        runner_hash=runner_hash,
        lowering_report_semantic_hash=lowering_report_semantic_hash,
        base_dir=base_dir,
    )


def load_resolved_artifact(strategy_config_hash: str) -> Optional[dict]:
    """Load resolved artifact JSON for a compiled strategy.

    Compiler writes: research/strategies/{hash}/resolved.json
    """
    hash_val = strategy_config_hash
    if hash_val.startswith("sha256:"):
        hash_val = hash_val[7:]

    resolved_path = os.path.join(RESEARCH_DIR, "strategies", hash_val, "resolved.json")
    if not os.path.exists(resolved_path):
        return None

    with open(resolved_path, "r") as f:
        return json.load(f)


def get_capability_registry() -> Dict[str, Dict[str, str]]:
    """Return the capability registry for UI filtering."""
    return dict(CAPABILITY_REGISTRY)
