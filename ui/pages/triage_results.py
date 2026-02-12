"""Triage Results — run triage and display per-test results."""

import glob
import json
import math
import os
import traceback
from typing import Any, Dict, List, Optional

from nicegui import ui

from ui.services.backtest_runner import Bar, run_backtest
from ui.services.triage_bridge import run_triage_pipeline, save_triage_result
from ui.services.compiler_bridge import save_promotion
from phase5_triage_types import TriageConfig, StrategyMetadata, TriageResult
from btc_alpha_v3_final import Fixed, SemanticType

RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "research")
HISTORIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "historic_data")


def _find_dataset() -> Optional[str]:
    """Auto-detect dataset from historic_data/."""
    patterns = [
        os.path.join(HISTORIC_DIR, "*.parquet"),
    ]
    for pat in patterns:
        files = sorted(glob.glob(pat))
        if files:
            # Prefer 3mo
            for f in files:
                if "2025-10-01" in f:
                    return f
            return files[0]
    return None


def _load_bars_from_parquet(path: str) -> List[Bar]:
    """Load 1m bars from parquet file."""
    import pandas as pd
    df = pd.read_parquet(path)
    bars = []
    for i, row in enumerate(df.itertuples()):
        ts = int(row.timestamp) if hasattr(row, "timestamp") else int(row.ts)
        bars.append(Bar(
            ts=ts,
            o=float(row.open),
            h=float(row.high),
            l=float(row.low),
            c=float(row.close),
            v=float(row.volume) if hasattr(row, "volume") else 0.0,
            index=i,
        ))
    return bars


def triage_results_page(strategy_hash: str):
    """Triage runner and results display page."""
    with ui.column().classes("w-full max-w-6xl mx-auto p-4"):
        with ui.row().classes("items-center gap-4 mb-4"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props(
                "flat dense round")
            ui.label("Triage Results").classes("text-2xl font-bold")

        ui.label(f"Strategy: sha256:{strategy_hash[:24]}...").classes(
            "monospace text-sm text-gray-400")

        # Load resolved artifact
        resolved_path = os.path.join(RESEARCH_DIR, "strategies", strategy_hash, "resolved.json")
        if not os.path.exists(resolved_path):
            ui.label("Resolved artifact not found. Compile and save first.").classes(
                "text-red-400 py-4")
            return

        with open(resolved_path) as f:
            resolved_config = json.load(f)

        # Results container
        results_container = ui.column().classes("w-full")

        # Check for existing results
        _show_existing_results(strategy_hash, results_container)

        # Run triage button
        ui.separator()
        with ui.row().classes("gap-4 items-center"):
            ui.button("Run Triage", icon="science",
                      on_click=lambda: _run_triage(
                          strategy_hash, resolved_config, results_container)).props(
                "color=primary")
            ui.label("Runs backtest → triage pipeline on available dataset").classes(
                "text-sm text-gray-400")


def _show_existing_results(strategy_hash: str, container):
    """Display previously saved triage results."""
    results_dir = os.path.join(RESEARCH_DIR, "triage_results", strategy_hash)
    if not os.path.isdir(results_dir):
        return

    files = sorted(glob.glob(os.path.join(results_dir, "*_triage.json")), reverse=True)
    if not files:
        return

    with container:
        ui.label("Previous Results").classes("text-lg font-bold mt-4")
        for filepath in files[:5]:  # Show latest 5
            try:
                with open(filepath) as f:
                    data = json.load(f)
                _render_result(data, os.path.basename(filepath))
            except Exception:
                continue


async def _run_triage(strategy_hash: str, resolved_config: dict, container):
    """Run triage pipeline."""
    # Find dataset
    dataset_path = _find_dataset()
    if not dataset_path:
        ui.notify("No dataset found in historic_data/. Run scripts/fetch_data.sh", type="negative")
        return

    ui.notify("Running triage... this may take a moment", type="info")

    try:
        bars = _load_bars_from_parquet(dataset_path)

        # Build metadata
        metadata = _build_default_metadata(strategy_hash, resolved_config)
        triage_config = TriageConfig()

        # Run pipeline
        result = run_triage_pipeline(
            resolved_config=resolved_config,
            strategy_config_hash=f"sha256:{strategy_hash}",
            bars_1m=bars,
            metadata=metadata,
            triage_config=triage_config,
        )

        # Save result
        filepath = save_triage_result(
            f"sha256:{strategy_hash}", result, result.dataset_hash)

        # Display
        container.clear()
        with container:
            ui.label("Latest Result").classes("text-lg font-bold mt-4")
            _render_result_obj(result)

            if result.passed:
                ui.button("Write Promotion Artifact", icon="verified",
                          on_click=lambda: _write_promotion(strategy_hash, result)).props(
                    "color=positive")

        ui.notify(
            f"Triage {'PASSED' if result.passed else 'FAILED'}: {result.reason}",
            type="positive" if result.passed else "negative",
        )
    except Exception as e:
        ui.notify(f"Triage error: {e}", type="negative", timeout=10000)
        traceback.print_exc()


def _build_default_metadata(strategy_hash: str, config: dict) -> StrategyMetadata:
    """Build StrategyMetadata with sensible defaults."""
    # Extract 3 indicator params as triage-sensitive
    instances = config.get("indicator_instances", [])
    params = {}
    bounds = {}

    for inst in instances:
        for pname, pval in inst.get("parameters", {}).items():
            key = f"{inst['label']}.{pname}"
            if isinstance(pval, (int, float)):
                params[key] = int(pval)
                bounds[key] = (int(pval * 0.5), int(pval * 2))

    # Need exactly 3 sensitive params
    param_keys = list(params.keys())
    while len(param_keys) < 3:
        dummy = f"dummy_param_{len(param_keys)}"
        params[dummy] = 10
        bounds[dummy] = (5, 20)
        param_keys.append(dummy)

    sensitive = tuple(param_keys[:3])

    return StrategyMetadata(
        strategy_id=strategy_hash,
        strategy_version_hash=strategy_hash,
        param_defaults=params,
        param_bounds=bounds,
        triage_sensitive_params=sensitive,
    )


def _render_result(data: dict, filename: str = ""):
    """Render a triage result from saved JSON."""
    passed = data.get("passed", False)
    reason = data.get("reason", "")
    test_results = data.get("test_results", {})

    with ui.card().classes(f"w-full p-4 mb-2 {'compiler-success' if passed else 'compiler-error'}"):
        with ui.row().classes("items-center gap-4"):
            icon = "check_circle" if passed else "cancel"
            color = "green" if passed else "red"
            ui.icon(icon, color=color).classes("text-2xl")
            ui.label("PASS" if passed else "FAIL").classes(
                f"text-lg font-bold text-{color}-400")
            if filename:
                ui.label(filename).classes("monospace text-xs text-gray-400")

        ui.label(reason).classes("text-sm text-gray-300 mt-1")

        # Per-test breakdown
        if test_results:
            _render_test_breakdown(test_results)


def _render_result_obj(result: TriageResult):
    """Render a TriageResult object."""
    data = {
        "passed": result.passed,
        "reason": result.reason,
        "test_results": result.test_results,
        "train_sharpe": result.train_sharpe,
        "oos_sharpe": result.oos_sharpe,
    }
    _render_result(data)


def _render_test_breakdown(test_results: dict):
    """Render per-test results table."""
    columns = [
        {"name": "test", "label": "Test", "field": "test", "align": "left"},
        {"name": "status", "label": "Status", "field": "status", "align": "center"},
        {"name": "detail", "label": "Detail", "field": "detail", "align": "left"},
    ]

    rows = []
    for test_key in ["test_1", "test_1_5", "test_2", "test_3", "test_4"]:
        t = test_results.get(test_key, {})
        if not t:
            continue

        status = "PASS" if t.get("passed", True) else "FAIL"
        if test_key == "test_1_5":
            status = "INFO"

        detail_parts = []
        for k, v in t.items():
            if k not in ("passed", "reason"):
                detail_parts.append(f"{k}={v}")
        detail = ", ".join(detail_parts[:4])

        test_names = {
            "test_1": "1: OOS Holdout",
            "test_1_5": "1.5: Cost Sanity",
            "test_2": "2: MC Date-Shift",
            "test_3": "3: Param Sensitivity",
            "test_4": "4: Correlation",
        }

        rows.append({
            "test": test_names.get(test_key, test_key),
            "status": status,
            "detail": detail,
        })

    if rows:
        ui.table(columns=columns, rows=rows, row_key="test").classes(
            "w-full mt-2").props("flat bordered dense")


def _write_promotion(strategy_hash: str, result: TriageResult):
    """Write promotion artifact on triage PASS."""
    try:
        filepath = save_promotion(
            strategy_config_hash=f"sha256:{strategy_hash}",
            composition_id="",
            composition_spec_hash="",
            tier="TRIAGE",
            result="PASS",
            dataset_hash=result.dataset_hash,
            runner_hash="backtest_runner_v1",
            lowering_report_semantic_hash="",
        )
        ui.notify(f"Promotion artifact written: {filepath}", type="positive")
    except Exception as e:
        ui.notify(f"Promotion write error: {e}", type="negative")
