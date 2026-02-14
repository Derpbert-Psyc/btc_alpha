"""Condition Builder — cascading contextual selection with hydration."""

from typing import Any, Callable, Dict, List, Optional

from nicegui import ui

from ui.services.output_descriptions import get_output_description


# ---------------------------------------------------------------------------
# Operator options (no is_present/is_absent — reserved for future)
# ---------------------------------------------------------------------------
OPERATOR_OPTIONS = {
    ">": "> (above)",
    "<": "< (below)",
    ">=": ">= (at or above)",
    "<=": "<= (at or below)",
    "==": "== (equals)",
    "crosses_above": "crosses_above \u2191 (was below, now above)",
    "crosses_below": "crosses_below \u2193 (was above, now below)",
}


# ---------------------------------------------------------------------------
# Pure helper functions (testable without UI)
# ---------------------------------------------------------------------------

def _format_key_params(ind_id: str, params: dict) -> str:
    """Format key parameters for display based on indicator type."""
    ind = ind_id.lower()
    if ind in ("ema", "rsi", "atr", "adx", "choppiness", "linreg", "hv",
               "roc", "donchian"):
        p = params.get("period", "")
        return f"period={p}" if p != "" else ""
    if ind in ("macd", "macd_tv"):
        fast = params.get("fast_period", params.get("fast", ""))
        slow = params.get("slow_period", params.get("slow", ""))
        sig = params.get("signal_period", params.get("signal", ""))
        if fast != "" and slow != "" and sig != "":
            return f"{fast}/{slow}/{sig}"
        return ""
    if ind == "bollinger":
        p = params.get("period", "")
        s = params.get("num_std", "")
        parts = []
        if p != "":
            parts.append(f"p={p}")
        if s != "":
            parts.append(f"std={s}")
        return ", ".join(parts)
    if ind == "pivot_structure":
        l = params.get("left", "")
        r = params.get("right", "")
        parts = []
        if l != "":
            parts.append(f"L={l}")
        if r != "":
            parts.append(f"R={r}")
        return ", ".join(parts)
    if ind == "floor_pivots":
        return ""
    if ind == "dynamic_sr":
        lb = params.get("lookback", "")
        return f"lookback={lb}" if lb != "" else ""
    if ind == "vol_targeting":
        tv = params.get("target_vol", "")
        return f"target={tv}" if tv != "" else ""
    if ind == "vrvp":
        nb = params.get("num_bins", "")
        return f"bins={nb}" if nb != "" else ""
    # All others: show all params
    if not params:
        return ""
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _instance_display_label(inst: dict) -> str:
    """Rich display label: 'label (indicator_type, timeframe, key_params)'."""
    label = inst.get("label", "")
    ind_id = inst.get("indicator_id", "")
    tf = inst.get("timeframe", "")
    params = inst.get("parameters", {})
    param_str = _format_key_params(ind_id, params)
    suffix = f"{ind_id}, {tf}"
    if param_str:
        suffix += f", {param_str}"
    return f"{label} ({suffix})"


def _instance_display_label_short(spec: dict, label: str) -> str:
    """Short label for condition summaries: 'label (indicator, timeframe)'."""
    for inst in spec.get("indicator_instances", []):
        if inst.get("label") == label:
            ind_id = inst.get("indicator_id", "")
            tf = inst.get("timeframe", "")
            return f"{label} ({ind_id}, {tf})"
    return label


def _output_options(spec: dict, instance_label: str) -> dict:
    """Build {output_name: 'output_name \u2014 description'} for the selected instance."""
    for inst in spec.get("indicator_instances", []):
        if inst.get("label") == instance_label:
            ind_id = inst.get("indicator_id", "")
            options = {}
            for out in inst.get("outputs_used", []):
                desc = get_output_description(ind_id, out)
                options[out] = f"{out} \u2014 {desc}"
            return options
    return {}


def _build_condition_summary(cond: dict, spec: dict) -> str:
    """Build a human-readable condition summary string."""
    inst_label = cond.get("indicator", "?")
    inst_display = _instance_display_label_short(spec, inst_label)
    output = cond.get("output", "?")
    op = cond.get("operator", "?")
    if cond.get("ref_indicator"):
        ref_label = cond["ref_indicator"]
        ref_display = _instance_display_label_short(spec, ref_label)
        ref_out = cond.get("ref_output", "?")
        rhs = f"{ref_display} {ref_out}"
    else:
        rhs = str(cond.get("value", "?"))
    return f"{inst_display} {output} {op} {rhs}"


def _hydrate_condition(cond: dict, spec: dict) -> dict:
    """Derive UI state from a condition dict.
    Returns dict with instance, output, operator, rhs_mode, value,
    ref_indicator, ref_output, instance_valid, output_valid, warnings.
    """
    warnings = []
    instance = cond.get("indicator", "") or None
    output = cond.get("output", "") or None
    operator = cond.get("operator", ">") or ">"

    # Validate instance
    instance_valid = False
    if instance:
        for inst in spec.get("indicator_instances", []):
            if inst.get("label") == instance:
                instance_valid = True
                break
        if not instance_valid:
            warnings.append(f"Instance '{instance}' not found")

    # Validate output
    output_valid = False
    if instance and output and instance_valid:
        for inst in spec.get("indicator_instances", []):
            if inst.get("label") == instance:
                if output in inst.get("outputs_used", []):
                    output_valid = True
                else:
                    warnings.append(f"Output '{output}' not available")
                break
    elif output and not instance:
        warnings.append("No instance selected")

    # Determine RHS mode
    ref_indicator = cond.get("ref_indicator", "") or None
    ref_output = cond.get("ref_output", "") or None
    if ref_indicator:
        rhs_mode = "Indicator"
    else:
        rhs_mode = "Value"

    value = cond.get("value", 0)
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError):
            value = 0

    return {
        "instance": instance,
        "output": output,
        "operator": operator,
        "rhs_mode": rhs_mode,
        "value": value,
        "ref_indicator": ref_indicator,
        "ref_output": ref_output,
        "instance_valid": instance_valid,
        "output_valid": output_valid,
        "warnings": warnings,
    }


def _unique_label(base: str, existing_labels: list) -> str:
    """Generate unique label with collision avoidance."""
    if base not in existing_labels:
        return base
    suffix = 2
    while f"{base}_{suffix}" in existing_labels:
        suffix += 1
    return f"{base}_{suffix}"


def _rename_indicator_references(spec: dict, old_label: str, new_label: str):
    """Rewrite all condition references from old_label to new_label."""
    for rule in spec.get("entry_rules", []):
        for cond in rule.get("conditions", []):
            if cond.get("indicator") == old_label:
                cond["indicator"] = new_label
            if cond.get("ref_indicator") == old_label:
                cond["ref_indicator"] = new_label
        for grp in rule.get("condition_groups", []):
            for cond in grp.get("conditions", []):
                if cond.get("indicator") == old_label:
                    cond["indicator"] = new_label
                if cond.get("ref_indicator") == old_label:
                    cond["ref_indicator"] = new_label
    for rule in spec.get("exit_rules", []):
        for cond in rule.get("conditions", []):
            if cond.get("indicator") == old_label:
                cond["indicator"] = new_label
            if cond.get("ref_indicator") == old_label:
                cond["ref_indicator"] = new_label
    for rule in spec.get("gate_rules", []):
        for cond in rule.get("conditions", []):
            if cond.get("indicator") == old_label:
                cond["indicator"] = new_label
            if cond.get("ref_indicator") == old_label:
                cond["ref_indicator"] = new_label
    # Rename instance itself
    for inst in spec.get("indicator_instances", []):
        if inst.get("label") == old_label:
            inst["label"] = new_label
            break


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------

def render_condition_builder(
    state,
    conditions: List[dict],
    key_prefix: str,
    on_change: Callable,
):
    """Render a condition builder for a list of conditions."""
    spec = state.working_spec

    with ui.column().classes("w-full"):
        for i, cond in enumerate(conditions):
            _render_condition_row(state, conditions, cond, i, spec, on_change)

        # Add condition button
        ui.button("Add Condition", icon="add",
                  on_click=lambda: _add_condition(conditions, state, on_change)).props(
            "flat dense")


def _render_condition_row(state, conditions, cond, index, spec, on_change):
    """Render a single condition row with cascade selectors."""
    hydrated = _hydrate_condition(cond, spec)
    has_warnings = bool(hydrated["warnings"])

    # Container for the whole condition row (for re-rendering)
    row_classes = "w-full p-2 mb-1 rounded"
    if has_warnings:
        row_classes += " border border-red-500"

    with ui.column().classes(row_classes):
        # Warning text
        if has_warnings:
            for w in hydrated["warnings"]:
                ui.label(w).classes("text-xs text-red-400")

        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            _hydrating = True

            # --- Build options ---
            instance_options = {
                inst.get("label", ""): _instance_display_label(inst)
                for inst in spec.get("indicator_instances", [])
            }
            out_opts = _output_options(spec, cond.get("indicator", ""))
            op_val = cond.get("operator", ">")
            if op_val not in OPERATOR_OPTIONS:
                op_val = ">"
            rhs_mode = hydrated["rhs_mode"]
            current_inst = cond.get("indicator", "")
            ref_instance_opts = {
                inst.get("label", ""): _instance_display_label(inst)
                for inst in spec.get("indicator_instances", [])
                if inst.get("label", "") != current_inst
            }
            ref_out_opts = _output_options(spec, cond.get("ref_indicator", ""))

            # --- Define handlers (use e.value — properly mapped by NiceGUI) ---
            def on_instance_change(e, idx=index):
                if _hydrating:
                    return
                c = conditions[idx]
                new_inst = e.value if e.value else ""
                # F6: if new instance equals ref_indicator, clear ref
                if new_inst and new_inst == c.get("ref_indicator", ""):
                    c.pop("ref_indicator", None)
                    c.pop("ref_output", None)
                c["indicator"] = new_inst
                # Reset cascade: output, operator, value, ref
                c["output"] = ""
                c["operator"] = ">"
                c["value"] = 0
                c.pop("ref_indicator", None)
                c.pop("ref_output", None)
                state.mark_changed()
                on_change()
                ui.navigate.to(f"/editor/{state.composition_id}")

            def on_output_change(e, idx=index):
                if _hydrating:
                    return
                c = conditions[idx]
                c["output"] = e.value if e.value else ""
                # Reset: operator, value, ref
                c["operator"] = ">"
                c["value"] = 0
                c.pop("ref_indicator", None)
                c.pop("ref_output", None)
                state.mark_changed()
                on_change()
                ui.navigate.to(f"/editor/{state.composition_id}")

            def on_operator_change(e, idx=index):
                if _hydrating:
                    return
                c = conditions[idx]
                c["operator"] = e.value if e.value else ">"
                state.mark_changed()
                on_change()

            def on_mode_toggle(e, idx=index):
                if _hydrating:
                    return
                c = conditions[idx]
                new_mode = e.value
                if new_mode == "Value":
                    c.pop("ref_indicator", None)
                    c.pop("ref_output", None)
                    c.setdefault("value", 0)
                else:
                    c.pop("value", None)
                    c["ref_indicator"] = ""
                    c["ref_output"] = ""
                state.mark_changed()
                on_change()
                ui.navigate.to(f"/editor/{state.composition_id}")

            def on_value_change(e, idx=index):
                if _hydrating:
                    return
                c = conditions[idx]
                c["value"] = e.args if e.args is not None else 0
                state.mark_changed()
                on_change()

            def on_ref_instance_change(e, idx=index):
                if _hydrating:
                    return
                c = conditions[idx]
                c["ref_indicator"] = e.value if e.value else ""
                c["ref_output"] = ""
                state.mark_changed()
                on_change()
                ui.navigate.to(f"/editor/{state.composition_id}")

            def on_ref_output_change(e, idx=index):
                if _hydrating:
                    return
                c = conditions[idx]
                c["ref_output"] = e.value if e.value else ""
                state.mark_changed()
                on_change()

            # --- Instance selector (on_change= gives properly mapped e.value) ---
            ui.select(
                options=instance_options,
                value=cond.get("indicator", "") or None,
                label="Instance",
                on_change=on_instance_change,
            ).classes("w-48").props("dense clearable")

            # --- Output selector ---
            ui.select(
                options=out_opts if out_opts else {"": "(select instance)"},
                value=cond.get("output", "") or None,
                label="Output",
                on_change=on_output_change,
            ).classes("w-48").props("dense clearable")

            # --- Operator selector ---
            ui.select(
                options=OPERATOR_OPTIONS,
                value=op_val,
                label="Op",
                on_change=on_operator_change,
            ).classes("w-48").props("dense")

            # --- RHS mode toggle ---
            ui.toggle(
                ["Value", "Indicator"],
                value=rhs_mode,
                on_change=on_mode_toggle,
            ).props("dense")

            # Value mode controls
            val_container = ui.row().classes("items-center gap-2")
            # Indicator mode controls
            ref_container = ui.row().classes("items-center gap-2")

            if rhs_mode == "Value":
                ref_container.style("display: none")
                with val_container:
                    val_input = ui.number(
                        value=hydrated["value"],
                        label="Value",
                    ).classes("w-24").props("dense")
                    val_input.on("change", on_value_change)
            else:
                val_container.style("display: none")

            if rhs_mode == "Indicator":
                with ref_container:
                    ui.select(
                        options=ref_instance_opts,
                        value=cond.get("ref_indicator", "") or None,
                        label="Ref Instance",
                        on_change=on_ref_instance_change,
                    ).classes("w-48").props("dense clearable")
                    ui.select(
                        options=ref_out_opts if ref_out_opts else {"": "(select instance)"},
                        value=cond.get("ref_output", "") or None,
                        label="Ref Output",
                        on_change=on_ref_output_change,
                    ).classes("w-40").props("dense clearable")

            # Delete button
            ui.button(icon="close",
                      on_click=lambda idx=index: _delete_condition(
                          conditions, idx, state, on_change)).props(
                "flat dense round size=sm color=negative")

            _hydrating = False

        # Summary line
        summary_text = _build_condition_summary(cond, spec)
        ui.label(summary_text).classes("text-xs text-gray-400 italic mt-1")


def _add_condition(conditions: list, state, on_change):
    conditions.append({
        "indicator": "",
        "output": "",
        "operator": ">",
        "value": 0,
    })
    state.mark_changed()
    on_change()
    ui.navigate.to(f"/editor/{state.composition_id}")


def _delete_condition(conditions: list, idx: int, state, on_change):
    if idx < len(conditions):
        conditions.pop(idx)
        state.mark_changed()
        on_change()
        ui.navigate.to(f"/editor/{state.composition_id}")
