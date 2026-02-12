"""Condition Builder — direct + cross-ref modes."""

from typing import Any, Callable, Dict, List

from nicegui import ui

from composition_compiler_v1_5_2 import CAPABILITY_REGISTRY, OPERATOR_TO_CAPABILITY
from strategy_framework_v1_8_0 import INDICATOR_OUTPUTS, INDICATOR_NAME_TO_ID, INDICATOR_ID_TO_NAME


def _get_available_operators(target_ev: str) -> List[str]:
    """Get operators available for the target engine version."""
    from strategy_framework_v1_8_0 import version_gte
    ops = []
    for op_str, cap_name in OPERATOR_TO_CAPABILITY.items():
        cap = CAPABILITY_REGISTRY.get(cap_name, {})
        min_ver = cap.get("min_engine_version", "1.0.0")
        if version_gte(target_ev, min_ver):
            ops.append(op_str)
    return ops


def _get_instance_labels(spec: dict) -> List[str]:
    """Get all indicator instance labels from spec."""
    return [inst.get("label", "") for inst in spec.get("indicator_instances", [])]


def _get_instance_outputs(spec: dict, label: str) -> List[str]:
    """Get output names for a given instance label."""
    for inst in spec.get("indicator_instances", []):
        if inst.get("label") == label:
            return list(inst.get("outputs_used", []))
    return []


def render_condition_builder(
    state,
    conditions: List[dict],
    key_prefix: str,
    on_change: Callable,
):
    """Render a condition builder for a list of conditions."""
    spec = state.working_spec
    target_ev = spec.get("target_engine_version", "1.8.0")
    available_ops = _get_available_operators(target_ev)
    labels = _get_instance_labels(spec)

    with ui.column().classes("w-full"):
        for i, cond in enumerate(conditions):
            with ui.row().classes("w-full items-center gap-2 mb-1"):
                # Instance selector
                ind_select = ui.select(
                    labels,
                    value=cond.get("indicator", ""),
                    label="Instance",
                ).classes("w-40").props("dense")

                # Output selector
                current_outputs = _get_instance_outputs(spec, cond.get("indicator", ""))
                out_select = ui.select(
                    current_outputs or ["(select instance)"],
                    value=cond.get("output", ""),
                    label="Output",
                ).classes("w-32").props("dense")

                # Update outputs when instance changes
                def update_outputs(e, os=out_select):
                    outs = _get_instance_outputs(spec, e.value)
                    os.options = outs or ["(none)"]
                ind_select.on("update:model-value", update_outputs)

                # Operator selector
                op_select = ui.select(
                    available_ops,
                    value=cond.get("operator", ">"),
                    label="Op",
                ).classes("w-32").props("dense")

                # Value — either numeric or cross-ref
                is_unary = cond.get("operator", "") in ("is_present", "is_absent")
                is_cross_ref = "ref_indicator" in cond

                if is_unary:
                    ui.label("(unary)").classes("text-gray-400 w-24")
                elif is_cross_ref:
                    ref_ind = ui.select(
                        labels,
                        value=cond.get("ref_indicator", ""),
                        label="Ref Instance",
                    ).classes("w-32").props("dense")
                    ref_out = ui.select(
                        _get_instance_outputs(spec, cond.get("ref_indicator", "")),
                        value=cond.get("ref_output", ""),
                        label="Ref Output",
                    ).classes("w-28").props("dense")
                else:
                    val_input = ui.number(
                        value=cond.get("value", 0),
                        label="Value",
                    ).classes("w-24").props("dense")

                # Cross-ref toggle
                xref_toggle = ui.checkbox("X-Ref", value=is_cross_ref).props("dense")

                # Delete
                ui.button(icon="close",
                          on_click=lambda i=i: _delete_condition(
                              conditions, i, state, on_change)).props(
                    "flat dense round size=sm color=negative")

                # Wire changes
                def save_cond(e=None, idx=i):
                    c = conditions[idx]
                    c["indicator"] = ind_select.value
                    c["output"] = out_select.value
                    c["operator"] = op_select.value
                    if op_select.value in ("is_present", "is_absent"):
                        c.pop("value", None)
                        c.pop("ref_indicator", None)
                        c.pop("ref_output", None)
                    elif xref_toggle.value:
                        c.pop("value", None)
                        c["ref_indicator"] = ref_ind.value if is_cross_ref else ""
                        c["ref_output"] = ref_out.value if is_cross_ref else ""
                    else:
                        c["value"] = val_input.value if not is_unary and not is_cross_ref else 0
                        c.pop("ref_indicator", None)
                        c.pop("ref_output", None)
                    state.mark_changed()
                    on_change()

                for widget in [ind_select, out_select, op_select]:
                    widget.on("update:model-value", save_cond)

        # Add condition button
        ui.button("Add Condition", icon="add",
                  on_click=lambda: _add_condition(conditions, state, on_change)).props(
            "flat dense")


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
