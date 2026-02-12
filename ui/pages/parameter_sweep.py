"""Parameter Sweep — vary one param, recompile + run, show results table."""

import copy
import json
import math
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nicegui import ui

from ui.services.composition_store import load_composition
from ui.services.compiler_bridge import compile_spec
from ui.services.backtest_runner import Bar, run_backtest
from phase5_triage_types import TriageConfig, StrategyMetadata
from phase5_triage import run_test_1
from btc_alpha_v3_final import Fixed, SemanticType

RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "research")
HISTORIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "historic_data")


def parameter_sweep_page(composition_id: str):
    """Parameter sweep runner and results display."""
    spec = load_composition(composition_id)
    if spec is None:
        with ui.column().classes("w-full max-w-6xl mx-auto p-4"):
            ui.label("Composition not found").classes("text-red-400")
        return

    with ui.column().classes("w-full max-w-6xl mx-auto p-4"):
        with ui.row().classes("items-center gap-4 mb-4"):
            ui.button(icon="arrow_back",
                      on_click=lambda: ui.navigate.to(f"/editor/{composition_id}")).props(
                "flat dense round")
            ui.label("Parameter Sweep").classes("text-2xl font-bold")

        ui.label(f"Composition: {spec.get('display_name', composition_id[:16])}").classes(
            "text-sm text-gray-400")

        # Get triage-sensitive params
        tsp = spec.get("metadata", {}).get("triage_sensitive_params", [])
        if not tsp:
            ui.label("No triage-sensitive parameters defined. Add them in the Metadata tab.").classes(
                "text-amber-400 py-4")
            return

        # Parameter selector
        param_options = {p.get("param", f"param_{i}"): p for i, p in enumerate(tsp)}
        param_select = ui.select(
            list(param_options.keys()),
            value=list(param_options.keys())[0] if param_options else "",
            label="Parameter to sweep",
        ).classes("w-64")

        # Steps
        n_steps = ui.number(value=10, min=3, max=50, label="Steps").classes("w-24")

        # Results container
        results_container = ui.column().classes("w-full mt-4")

        ui.button("Run Sweep", icon="tune",
                  on_click=lambda: _run_sweep(
                      spec, param_select.value, param_options, int(n_steps.value),
                      results_container)).props("color=primary")


async def _run_sweep(spec, param_name, param_options, n_steps, container):
    """Run parameter sweep."""
    param_info = param_options.get(param_name)
    if not param_info:
        ui.notify("Invalid parameter", type="negative")
        return

    p_min = param_info.get("min", 0)
    p_max = param_info.get("max", 100)
    p_default = param_info.get("default", (p_min + p_max) / 2)

    # Find dataset
    import glob
    parquet_files = sorted(glob.glob(os.path.join(HISTORIC_DIR, "*.parquet")))
    if not parquet_files:
        ui.notify("No dataset found", type="negative")
        return

    dataset_path = parquet_files[0]
    for f in parquet_files:
        if "2025-10-01" in f:
            dataset_path = f
            break

    ui.notify(f"Sweeping {param_name}: {p_min} → {p_max} in {n_steps} steps", type="info")

    try:
        import pandas as pd
        from ui.services.backtest_runner import Bar as BarCls

        df = pd.read_parquet(dataset_path)
        bars = []
        for i, row in enumerate(df.itertuples()):
            ts = int(row.timestamp) if hasattr(row, "timestamp") else int(row.ts)
            bars.append(BarCls(
                ts=ts, o=float(row.open), h=float(row.high),
                l=float(row.low), c=float(row.close),
                v=float(row.volume) if hasattr(row, "volume") else 0.0,
                index=i,
            ))

        # Generate parameter values
        if isinstance(p_min, int) and isinstance(p_max, int):
            step = max(1, (p_max - p_min) // (n_steps - 1))
            values = list(range(p_min, p_max + 1, step))[:n_steps]
        else:
            step = (p_max - p_min) / (n_steps - 1)
            values = [p_min + i * step for i in range(n_steps)]

        # Run sweep
        results = []
        for val in values:
            try:
                modified_spec = _apply_param_override(spec, param_name, val)
                compilation = compile_spec(modified_spec)
                resolved = compilation["resolved_artifact"]
                config_hash = compilation["strategy_config_hash"]

                trades, prices, n_bars = run_backtest(resolved, bars, config_hash)

                # Run only Test 1 (OOS Sharpe) for speed
                triage_config = TriageConfig()
                t1 = run_test_1(trades, n_bars, triage_config)

                results.append({
                    "param_value": val,
                    "oos_sharpe": t1.oos_sharpe,
                    "train_sharpe": t1.train_sharpe,
                    "passed": t1.passed,
                    "trades": len(trades),
                    "hash": config_hash[:16],
                    "is_default": abs(val - p_default) < 0.001,
                })
            except Exception as e:
                results.append({
                    "param_value": val,
                    "oos_sharpe": 0,
                    "train_sharpe": 0,
                    "passed": False,
                    "trades": 0,
                    "hash": "ERROR",
                    "is_default": abs(val - p_default) < 0.001,
                    "error": str(e),
                })

        # Display results
        container.clear()
        with container:
            _render_sweep_results(param_name, results)

        # Save results
        _save_sweep_results(spec, param_name, results)

        ui.notify("Sweep complete", type="positive")

    except Exception as e:
        ui.notify(f"Sweep error: {e}", type="negative")
        traceback.print_exc()


def _apply_param_override(spec: dict, param_path: str, value) -> dict:
    """Apply a parameter override to a composition spec.

    param_path format: "instance_label.param_name" or "exit_rules.N.field"
    """
    modified = copy.deepcopy(spec)
    parts = param_path.split(".")

    if len(parts) == 2:
        label, pname = parts
        # Check indicator instances
        for inst in modified.get("indicator_instances", []):
            if inst.get("label") == label:
                inst["parameters"][pname] = value
                return modified
        # Check exit rules by label
        for rule in modified.get("exit_rules", []):
            if rule.get("label") == label:
                rule[pname] = value
                return modified

    elif len(parts) == 3:
        section, idx_str, field = parts
        try:
            idx = int(idx_str)
            if section in modified and idx < len(modified[section]):
                modified[section][idx][field] = value
                return modified
        except ValueError:
            pass

    return modified


def _render_sweep_results(param_name: str, results: list):
    """Render sweep results table."""
    ui.label(f"Sweep: {param_name}").classes("text-lg font-bold mb-2")

    columns = [
        {"name": "param_value", "label": "Value", "field": "param_value", "align": "center"},
        {"name": "oos_sharpe", "label": "OOS Sharpe (×1e6)", "field": "oos_sharpe", "align": "center"},
        {"name": "train_sharpe", "label": "Train Sharpe (×1e6)", "field": "train_sharpe", "align": "center"},
        {"name": "trades", "label": "Trades", "field": "trades", "align": "center"},
        {"name": "passed", "label": "Pass?", "field": "pass_str", "align": "center"},
        {"name": "hash", "label": "Hash", "field": "hash", "align": "left"},
    ]

    rows = []
    for r in results:
        rows.append({
            **r,
            "pass_str": "PASS" if r["passed"] else "FAIL",
        })

    table = ui.table(columns=columns, rows=rows, row_key="param_value").classes(
        "w-full").props("flat bordered dense")

    # Highlight default row
    table.add_slot("body-cell-param_value", """
        <q-td :props="props">
            <span :class="props.row.is_default ? 'font-bold text-blue-400' : ''">
                {{ props.row.param_value }}
                <span v-if="props.row.is_default"> (default)</span>
            </span>
        </q-td>
    """)


def _save_sweep_results(spec: dict, param_name: str, results: list):
    """Save sweep results to research/sweep_results/."""
    comp_id = spec.get("composition_id", "unknown")
    dir_path = os.path.join(RESEARCH_DIR, "sweep_results", comp_id)
    os.makedirs(dir_path, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_param = param_name.replace(".", "_").replace("/", "_")
    filename = f"{safe_param}_{timestamp}.json"
    filepath = os.path.join(dir_path, filename)

    data = {
        "composition_id": comp_id,
        "param_name": param_name,
        "timestamp": timestamp,
        "results": results,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
