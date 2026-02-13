"""Gate Editor — conditions + policy + exit matrix."""

from typing import Callable
from nicegui import ui

from composition_compiler_v1_5_2 import CAPABILITY_REGISTRY, GATE_POLICY_TO_CAPABILITY
from strategy_framework_v1_8_0 import version_gte
from ui.components.condition_builder import render_condition_builder

POLICY_COLORS = {
    "FORCE_FLAT": "negative",
    "HOLD": "primary",
    "HANDOFF": "warning",
}

EXIT_MATRIX = {
    "FORCE_FLAT": "All positions closed immediately. New entries blocked.",
    "HOLD": "Existing positions held. New entries blocked. Exits still evaluated.",
    "HANDOFF": "SIGNAL exits suppressed. Risk exits (SL, trailing, MTM) still active. New entries blocked.",
}


def _available_policies(target_ev: str):
    result = []
    for pol, cap_name in GATE_POLICY_TO_CAPABILITY.items():
        cap = CAPABILITY_REGISTRY.get(cap_name, {})
        min_ver = cap.get("min_engine_version", "1.0.0")
        if version_gte(target_ev, min_ver):
            result.append(pol)
    return result


def render_gate_editor(state, on_change: Callable):
    """Render the gate rules editor."""
    spec = state.working_spec
    gates = spec.setdefault("gate_rules", [])
    target_ev = spec.get("target_engine_version", "1.8.0")
    policies = _available_policies(target_ev)

    with ui.column().classes("w-full"):
        # Multi-gate tooltip
        if len(gates) > 1:
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon("info", color="amber").classes("text-lg")
                ui.label(
                    "Most severe wins: FORCE_FLAT > HANDOFF > HOLD"
                ).classes("text-sm text-amber-400")

        ui.button("Add Gate", icon="add",
                  on_click=lambda: _add_gate(state, on_change)).props("color=primary")

        if not gates:
            ui.label("No gate rules — signals always active.").classes(
                "text-gray-400 py-4")
            return

        for i, gate in enumerate(gates):
            policy = gate.get("exit_policy", "HOLD")
            color = POLICY_COLORS.get(policy, "grey")

            with ui.card().classes(f"w-full mb-2 p-3 border-l-4").style(
                f"border-left-color: {'#ef4444' if policy == 'FORCE_FLAT' else '#f59e0b' if policy == 'HANDOFF' else '#3b82f6'}"):

                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-2"):
                        name_input = ui.input(
                            value=gate.get("label", ""),
                            label="Gate Name",
                        ).classes("w-48").props("dense")
                        name_input.on("change", lambda e, idx=i: _update_gate_field(
                            state, idx, "label", e.args, on_change))

                        pol_select = ui.select(
                            policies,
                            value=policy,
                            label="Exit Policy",
                        ).classes("w-40").props("dense")
                        pol_select.on("update:model-value", lambda e, idx=i: _update_gate_field(
                            state, idx, "exit_policy", e.args, on_change))

                        ui.badge(policy, color=color)

                    ui.button(icon="delete",
                              on_click=lambda idx=i: _delete_gate(state, idx, on_change)).props(
                        "flat dense round color=negative")

                # Inline exit matrix
                matrix_text = EXIT_MATRIX.get(policy, "")
                if matrix_text:
                    ui.label(matrix_text).classes("text-xs text-gray-400 mt-1 italic")

                # Conditions (gate open when ALL true)
                ui.label("Gate OPEN when ALL conditions true:").classes(
                    "text-sm text-gray-400 mt-2")
                render_condition_builder(
                    state, gate.setdefault("conditions", []),
                    f"gate_{i}_cond", on_change)


def _add_gate(state, on_change):
    gates = state.working_spec.setdefault("gate_rules", [])
    gates.append({
        "label": "New Gate",
        "exit_policy": "HOLD",
        "conditions": [],
    })
    state.mark_changed()
    on_change()
    ui.navigate.to(f"/editor/{state.composition_id}")


def _update_gate_field(state, idx, field, value, on_change):
    gates = state.working_spec.get("gate_rules", [])
    if idx < len(gates):
        gates[idx][field] = value
        state.mark_changed()
        on_change()


def _delete_gate(state, idx, on_change):
    gates = state.working_spec.get("gate_rules", [])
    if idx < len(gates):
        gates.pop(idx)
        state.mark_changed()
        on_change()
        ui.navigate.to(f"/editor/{state.composition_id}")
