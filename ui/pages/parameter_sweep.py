"""Parameter Sweep — vary one param, recompile + run, show results table."""

import asyncio
import math
import traceback
from typing import Any, Dict, List, Optional

from nicegui import ui, run

from ui.services.composition_store import load_composition
from ui.services.research_services import (
    list_datasets,
    run_sweep_for_composition,
    SweepResult,
)


def _generate_centered_values(
    current: float,
    n_steps: int,
    param_min: float,
    param_max: float,
    is_integer: bool = True,
) -> list:
    """Generate sweep values centered on the current parameter value.

    The current value is always the midpoint. Equal number of points
    above and below. Clamped to [param_min, param_max].
    """
    # Ensure odd number of steps
    if n_steps % 2 == 0:
        n_steps += 1

    half = (n_steps - 1) // 2

    # Calculate step size (~15% of current value, minimum 1 for int)
    if current == 0:
        step = 1 if is_integer else 0.1
    else:
        step = abs(current) * 0.15
        if is_integer:
            step = max(1, round(step))
        else:
            step = max(0.01, round(step, 4))

    # Generate raw centered values
    values = [current + (i - half) * step for i in range(n_steps)]

    # Clamp: shift window if it exceeds bounds
    lowest = values[0]
    if lowest < param_min:
        shift = param_min - lowest
        values = [v + shift for v in values]

    highest = values[-1]
    if highest > param_max:
        shift = highest - param_max
        values = [v - shift for v in values]

    # Final clamp (safety)
    values = [max(param_min, min(param_max, v)) for v in values]

    # Round integers
    if is_integer:
        values = [int(round(v)) for v in values]
    else:
        values = [round(v, 4) for v in values]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for v in values:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    return unique


def _extract_sweepable_params(spec: dict) -> List[dict]:
    """Auto-extract sweepable parameters from indicator_instances.

    Returns list of {param, default, min, max} dicts with dot-path param names
    (label.param_name) compatible with _apply_param_override.
    """
    params = []
    for inst in spec.get("indicator_instances", []):
        label = inst.get("label", "")
        if not label:
            continue
        for pname, pval in inst.get("parameters", {}).items():
            if not isinstance(pval, (int, float)):
                continue
            # Heuristic bounds: period-like params stay >= 2, others >= 0
            is_period = "period" in pname.lower()
            default = pval
            if is_period:
                lo = max(2, int(default * 0.5))
                hi = int(math.ceil(default * 2.5))
            elif default > 0:
                lo = round(default * 0.5, 4)
                hi = round(default * 2.0, 4)
            else:
                lo = 0
                hi = max(1, abs(default) * 2)
            params.append({
                "param": f"{label}.{pname}",
                "default": default,
                "min": lo,
                "max": hi,
            })
    return params


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

        # Merge manual triage_sensitive_params with auto-extracted indicator params
        manual_tsp = spec.get("metadata", {}).get("triage_sensitive_params", [])
        auto_tsp = _extract_sweepable_params(spec)

        # Manual params take priority — collect their param names for dedup
        manual_names = {p.get("param", "") for p in manual_tsp}
        merged = list(manual_tsp)
        for ap in auto_tsp:
            if ap["param"] not in manual_names:
                merged.append(ap)

        if not merged:
            ui.label("No sweepable parameters. Add indicator instances first.").classes(
                "text-amber-400 py-4")
            return

        if auto_tsp and not manual_tsp:
            ui.label("Parameters auto-extracted from indicator instances. "
                     "Define triage_sensitive_params in Metadata for custom ranges.").classes(
                "text-xs text-gray-500 mb-2")

        # Dataset selector
        datasets = list_datasets()
        if not datasets:
            ui.label("No dataset found in historic_data/").classes("text-red-400 py-2")
            return

        dataset_options = {d["label"]: d["path"] for d in datasets}

        # Parameter selector
        param_options = {p.get("param", f"param_{i}"): p for i, p in enumerate(merged)}
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

        # Sweep values preview
        preview_label = ui.label("").classes("text-xs text-gray-400 mt-1")

        def _update_preview(_=None):
            param_info = param_options.get(param_select.value)
            if not param_info:
                preview_label.text = ""
                return
            vals = _generate_centered_values(
                current=param_info.get("default", 0),
                n_steps=int(n_steps.value),
                param_min=param_info.get("min", 0),
                param_max=param_info.get("max", 100),
                is_integer=isinstance(param_info.get("default", 0), int),
            )
            current = param_info.get("default", 0)
            formatted = []
            for v in vals:
                if abs(v - current) < 0.001:
                    formatted.append(f"[{v}]")
                else:
                    formatted.append(str(v))
            preview_label.text = f"Sweep values: {', '.join(formatted)}"

        param_select.on("change", _update_preview)
        n_steps.on("change", _update_preview)
        _update_preview()

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

                # Generate centered sweep values
                sweep_values = _generate_centered_values(
                    current=p_default, n_steps=steps,
                    param_min=p_min, param_max=p_max,
                    is_integer=isinstance(p_default, int),
                )

                ui.notify(
                    f"Sweeping {param_select.value}: {len(sweep_values)} values centered on {p_default}",
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
                    param_values=sweep_values,
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
