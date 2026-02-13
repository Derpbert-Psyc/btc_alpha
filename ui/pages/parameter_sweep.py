"""Parameter Sweep — vary one param, recompile + run, show results table."""

import asyncio
import traceback
from typing import Any, Dict, List, Optional

from nicegui import ui, run

from ui.services.composition_store import load_composition
from ui.services.research_services import (
    list_datasets,
    run_sweep_for_composition,
    SweepResult,
)


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

        # Dataset selector
        datasets = list_datasets()
        if not datasets:
            ui.label("No dataset found in historic_data/").classes("text-red-400 py-2")
            return

        dataset_options = {d["label"]: d["path"] for d in datasets}

        # Parameter selector
        param_options = {p.get("param", f"param_{i}"): p for i, p in enumerate(tsp)}
        param_select = ui.select(
            list(param_options.keys()),
            value=list(param_options.keys())[0] if param_options else "",
            label="Parameter to sweep",
        ).classes("w-64")

        with ui.row().classes("gap-4 items-end"):
            dataset_select = ui.select(
                list(dataset_options.keys()),
                value=list(dataset_options.keys())[0],
                label="Dataset",
            ).classes("w-96")
            n_steps = ui.number(value=5, min=3, max=50, label="Steps").classes("w-24")

        # Results container
        results_container = ui.column().classes("w-full mt-4")

        # Concurrency guard
        sweep_running = {"active": False}

        async def do_sweep():
            if sweep_running["active"]:
                ui.notify("Sweep already running.", type="warning")
                return

            sweep_running["active"] = True
            sweep_btn.disable()

            try:
                param_info = param_options.get(param_select.value)
                if not param_info:
                    ui.notify("Select a parameter", type="warning")
                    return

                dataset_path = dataset_options.get(dataset_select.value)
                if not dataset_path:
                    ui.notify("Select a dataset", type="warning")
                    return

                p_min = param_info.get("min", 0)
                p_max = param_info.get("max", 100)
                p_default = param_info.get("default", (p_min + p_max) / 2)
                steps = int(n_steps.value)

                ui.notify(f"Sweeping {param_select.value}: {p_min} -> {p_max} in {steps} steps",
                          type="info")

                sweep_result = await run.cpu_bound(
                    run_sweep_for_composition,
                    spec=spec,
                    param_name=param_select.value,
                    param_min=p_min,
                    param_max=p_max,
                    n_steps=steps,
                    dataset_path=dataset_path,
                    default_value=p_default,
                )

                results_container.clear()
                with results_container:
                    _render_sweep_results(param_select.value, sweep_result)

                ui.notify("Sweep complete", type="positive")

            except Exception as e:
                ui.notify(f"Sweep error: {e}", type="negative")
                traceback.print_exc()
            finally:
                sweep_running["active"] = False
                sweep_btn.enable()

        sweep_btn = ui.button("Start Sweep", icon="tune",
                              on_click=do_sweep).props("color=primary")


def _render_sweep_results(param_name: str, sweep: SweepResult):
    """Render sweep results table."""
    ui.label(f"Sweep: {param_name}").classes("text-lg font-bold mb-2")

    columns = [
        {"name": "param_value", "label": "Value", "field": "param_value", "align": "center"},
        {"name": "expectancy", "label": "Expectancy (bps)", "field": "expectancy_fmt", "align": "center"},
        {"name": "win_rate", "label": "Win Rate", "field": "win_rate_fmt", "align": "center"},
        {"name": "trades", "label": "Trades", "field": "trades", "align": "center"},
        {"name": "passed", "label": "Pass?", "field": "pass_str", "align": "center"},
        {"name": "hash", "label": "Hash", "field": "hash", "align": "left"},
    ]

    rows = []
    for r in sweep.results:
        exp = r.get("expectancy_bps", 0)
        wr = r.get("win_rate", 0)
        rows.append({
            **r,
            "expectancy_fmt": f"{exp:+.1f}" if isinstance(exp, (int, float)) else str(exp),
            "win_rate_fmt": f"{wr:.1%}" if isinstance(wr, (int, float)) else str(wr),
            "pass_str": "PASS" if r.get("passed") else "FAIL",
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

    if sweep.saved_path:
        ui.label(f"Results saved: {sweep.saved_path}").classes(
            "monospace text-xs text-gray-500 mt-2")
