"""Composition Editor — the painting surface with 6 tabbed panels."""

import copy
import json
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nicegui import ui

from ui.services.composition_store import (
    load_composition,
    save_composition,
    update_compiled_hash,
)
from ui.services.compiler_bridge import compile_spec, save_artifacts, get_capability_registry
from ui.services.indicator_catalog import (
    get_all_indicators,
    get_outputs_for_indicator,
    resolve_indicator_id,
)
from ui.components.indicator_picker import show_indicator_picker
from ui.components.condition_builder import render_condition_builder
from ui.components.exit_rule_editor import render_exit_rules
from ui.components.gate_editor import render_gate_editor
from ui.components.execution_form import render_execution_form
from ui.components.metadata_form import render_metadata_form
from ui.components.compiler_panel import render_compiler_panel

from composition_compiler_v1_5_2 import CompilationError, CAPABILITY_REGISTRY
from strategy_framework_v1_8_0 import INDICATOR_OUTPUTS, INDICATOR_NAME_TO_ID


class EditorState:
    """In-memory editor state — separate from on-disk spec."""

    def __init__(self, composition_id: str, spec: dict):
        self.composition_id = composition_id
        self.disk_spec = copy.deepcopy(spec)
        self.working_spec = copy.deepcopy(spec)
        self.last_compilation = None
        self.unsaved = False

    def mark_changed(self):
        self.unsaved = True

    def save_to_disk(self):
        save_composition(self.composition_id, self.working_spec)
        self.disk_spec = copy.deepcopy(self.working_spec)
        self.unsaved = False

    def revert(self):
        self.working_spec = copy.deepcopy(self.disk_spec)
        self.unsaved = False

    def compile(self):
        """Compile the current working spec (in-memory, not disk)."""
        result = compile_spec(self.working_spec)
        self.last_compilation = result
        return result


def composition_editor_page(composition_id: str):
    """Render the composition editor for a given composition."""
    spec = load_composition(composition_id)
    if spec is None:
        ui.label("Composition not found").classes("text-red-400 text-xl p-8")
        ui.button("Back to list", on_click=lambda: ui.navigate.to("/"))
        return

    state = EditorState(composition_id, spec)

    with ui.column().classes("w-full max-w-7xl mx-auto p-4"):
        # Header bar
        _render_header(state)

        # Unsaved indicator
        unsaved_label = ui.label("").classes("text-amber-400 text-sm")

        def update_unsaved():
            unsaved_label.text = "● Unsaved changes" if state.unsaved else ""

        # Tabbed panels
        with ui.tabs().classes("w-full") as tabs:
            tab_ind = ui.tab("Indicators")
            tab_entry = ui.tab("Entry Rules")
            tab_exit = ui.tab("Exit Rules")
            tab_gate = ui.tab("Gates")
            tab_exec = ui.tab("Execution")
            tab_meta = ui.tab("Metadata")

        with ui.tab_panels(tabs, value=tab_ind).classes("w-full"):
            with ui.tab_panel(tab_ind):
                _render_indicators_tab(state, update_unsaved)

            with ui.tab_panel(tab_entry):
                _render_entry_rules_tab(state, update_unsaved)

            with ui.tab_panel(tab_exit):
                render_exit_rules(state, update_unsaved)

            with ui.tab_panel(tab_gate):
                render_gate_editor(state, update_unsaved)

            with ui.tab_panel(tab_exec):
                render_execution_form(state, update_unsaved)

            with ui.tab_panel(tab_meta):
                render_metadata_form(state, update_unsaved)

        # Compiler feedback panel (bottom dock)
        ui.separator()
        compiler_container = ui.column().classes("w-full")
        render_compiler_panel(state, compiler_container)


def _render_header(state: EditorState):
    """Render the editor header with name, actions, tags."""
    spec = state.working_spec

    with ui.row().classes("w-full items-center justify-between mb-2"):
        with ui.row().classes("items-center gap-4"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props(
                "flat dense round")
            name_input = ui.input(
                value=spec.get("display_name", "Untitled"),
            ).classes("text-xl font-bold w-96").props("dense borderless")
            name_input.on("change", lambda e: _update_field(
                state, "display_name", e.value))

            # Engine version badge
            ev = spec.get("target_engine_version", "1.8.0")
            ui.badge(f"v{ev}").props(
                f"color={'purple' if ev == '1.8.0' else 'blue'}")

            # Spec version
            ui.badge(f"spec {spec.get('spec_version', '1.5.2')}").props(
                "color=grey outline")

        with ui.row().classes("gap-2"):
            ui.button("Compile", icon="build",
                      on_click=lambda: _do_compile(state)).props("color=primary")
            ui.button("Save Draft", icon="save",
                      on_click=lambda: _do_save(state)).props("color=positive")
            ui.button("Revert", icon="undo",
                      on_click=lambda: _do_revert(state)).props("color=warning outline")

    # Description
    desc_input = ui.input(
        value=spec.get("description", ""),
        label="Description",
    ).classes("w-full mb-2")
    desc_input.on("change", lambda e: _update_field(state, "description", e.value))

    # Archetype tags
    CANONICAL_ARCHETYPES = [
        "trend_following", "mean_reversion", "momentum", "breakout",
        "volatility_based", "multi_timeframe", "scalping", "swing",
        "macro", "statistical",
    ]
    with ui.row().classes("gap-2 mb-2 flex-wrap"):
        ui.label("Archetypes:").classes("text-sm text-gray-400")
        for tag in CANONICAL_ARCHETYPES:
            active = tag in spec.get("archetype_tags", [])
            chip = ui.chip(tag, selectable=True, selected=active).props("dense")
            chip.on("update:selected", lambda e, t=tag: _toggle_archetype(state, t, e.value))


def _update_field(state: EditorState, field: str, value):
    state.working_spec[field] = value
    state.mark_changed()


def _toggle_archetype(state: EditorState, tag: str, selected: bool):
    tags = state.working_spec.setdefault("archetype_tags", [])
    if selected and tag not in tags:
        tags.append(tag)
    elif not selected and tag in tags:
        tags.remove(tag)
    state.mark_changed()


def _do_compile(state: EditorState):
    """Compile current in-memory spec."""
    try:
        result = state.compile()
        hash_val = result["strategy_config_hash"]
        ui.notify(f"Compiled: {hash_val[:24]}...", type="positive")
    except CompilationError as e:
        ui.notify(f"Compilation error: {e}", type="negative", timeout=10000)
    except Exception as e:
        ui.notify(f"Unexpected error: {e}", type="negative", timeout=10000)


def _do_save(state: EditorState):
    """Save working spec to disk."""
    state.save_to_disk()
    if state.last_compilation:
        hash_val = state.last_compilation["strategy_config_hash"]
        update_compiled_hash(state.composition_id, hash_val)
        try:
            save_artifacts(state.last_compilation)
        except Exception as e:
            ui.notify(f"Artifact write warning: {e}", type="warning")
    ui.notify("Saved", type="positive")


def _do_revert(state: EditorState):
    """Revert to on-disk spec."""
    state.revert()
    ui.notify("Reverted to last save", type="info")
    ui.navigate.to(f"/editor/{state.composition_id}")


def _render_indicators_tab(state: EditorState, on_change):
    """Render the indicator instances tab."""
    spec = state.working_spec
    instances = spec.setdefault("indicator_instances", [])

    with ui.column().classes("w-full"):
        ui.button("Add Indicator", icon="add",
                  on_click=lambda: _add_indicator(state, on_change)).props(
            "color=primary")

        if not instances:
            ui.label("No indicators configured.").classes("text-gray-400 py-4")
            return

        columns = [
            {"name": "label", "label": "Label", "field": "label", "align": "left"},
            {"name": "indicator", "label": "Indicator", "field": "indicator_id", "align": "left"},
            {"name": "timeframe", "label": "TF", "field": "timeframe", "align": "center"},
            {"name": "outputs", "label": "Outputs", "field": "outputs_str", "align": "left"},
            {"name": "role", "label": "Role", "field": "role", "align": "center"},
            {"name": "group", "label": "Group", "field": "group", "align": "center"},
            {"name": "actions", "label": "", "field": "actions", "align": "center"},
        ]

        rows = []
        for i, inst in enumerate(instances):
            ind_id = inst.get("indicator_id", "")
            if isinstance(ind_id, str):
                display_name = ind_id
            else:
                from strategy_framework_v1_8_0 import INDICATOR_ID_TO_NAME
                display_name = INDICATOR_ID_TO_NAME.get(ind_id, str(ind_id))
            rows.append({
                "idx": i,
                "label": inst.get("label", ""),
                "indicator_id": display_name,
                "timeframe": inst.get("timeframe", ""),
                "outputs_str": ", ".join(inst.get("outputs_used", [])),
                "role": inst.get("role", ""),
                "group": inst.get("group", ""),
            })

        table = ui.table(columns=columns, rows=rows, row_key="idx").classes(
            "w-full").props("flat bordered dense")

        # Delete action slot
        table.add_slot("body-cell-actions", """
            <q-td :props="props">
                <q-btn flat dense round icon="delete" color="negative"
                       @click="$parent.$emit('delete', props.row.idx)" />
            </q-td>
        """)
        table.on("delete", lambda e: _delete_indicator(state, e.args, on_change))


async def _add_indicator(state: EditorState, on_change):
    """Open indicator picker and add the result."""
    result = await show_indicator_picker(
        state.working_spec.get("target_engine_version", "1.8.0"))
    if result:
        state.working_spec.setdefault("indicator_instances", []).append(result)
        state.mark_changed()
        on_change()
        ui.navigate.to(f"/editor/{state.composition_id}")


def _delete_indicator(state: EditorState, idx, on_change):
    """Delete an indicator if not referenced."""
    instances = state.working_spec.get("indicator_instances", [])
    if idx < 0 or idx >= len(instances):
        return

    label = instances[idx].get("label", "")
    # Check references
    if _is_indicator_referenced(state.working_spec, label):
        ui.notify(f"Cannot delete '{label}' — referenced in rules", type="negative")
        return

    instances.pop(idx)
    state.mark_changed()
    on_change()
    ui.navigate.to(f"/editor/{state.composition_id}")


def _is_indicator_referenced(spec: dict, label: str) -> bool:
    """Check if an indicator label is referenced in any rule."""
    for rule in spec.get("entry_rules", []):
        for cond in rule.get("conditions", []):
            if cond.get("indicator") == label:
                return True
        for grp in rule.get("condition_groups", []):
            for cond in grp.get("conditions", []):
                if cond.get("indicator") == label:
                    return True
    for rule in spec.get("exit_rules", []):
        for cond in rule.get("conditions", []):
            if cond.get("indicator") == label:
                return True
    for rule in spec.get("gate_rules", []):
        for cond in rule.get("conditions", []):
            if cond.get("indicator") == label:
                return True
    return False


def _render_entry_rules_tab(state: EditorState, on_change):
    """Render entry rules with LONG/SHORT sub-tabs."""
    spec = state.working_spec
    rules = spec.setdefault("entry_rules", [])

    with ui.column().classes("w-full"):
        with ui.tabs().classes("w-full") as sub_tabs:
            long_tab = ui.tab("LONG")
            short_tab = ui.tab("SHORT")

        with ui.tab_panels(sub_tabs, value=long_tab).classes("w-full"):
            with ui.tab_panel(long_tab):
                _render_entry_direction(state, "LONG", on_change)
            with ui.tab_panel(short_tab):
                _render_entry_direction(state, "SHORT", on_change)


def _render_entry_direction(state: EditorState, direction: str, on_change):
    """Render entry rules for one direction."""
    rules = state.working_spec.get("entry_rules", [])
    dir_rules = [(i, r) for i, r in enumerate(rules)
                 if r.get("direction") == direction]

    with ui.column().classes("w-full"):
        ui.button(f"Add {direction} Entry Rule", icon="add",
                  on_click=lambda: _add_entry_rule(state, direction, on_change)).props(
            "color=primary dense")

        if not dir_rules:
            ui.label(f"No {direction} entry rules.").classes("text-gray-400 py-2")
            return

        for idx, rule in dir_rules:
            with ui.card().classes("w-full mb-2 p-3"):
                with ui.row().classes("w-full items-center justify-between"):
                    name_input = ui.input(
                        value=rule.get("label", ""),
                        label="Rule name",
                    ).classes("w-64").props("dense")
                    name_input.on("change", lambda e, i=idx: _update_entry_field(
                        state, i, "label", e.value, on_change))

                    cadence = ui.select(
                        ["1m", "5m", "15m", "30m", "1h", "4h"],
                        value=rule.get("evaluation_cadence", "1m"),
                        label="Cadence",
                    ).classes("w-24").props("dense")
                    cadence.on("update:model-value", lambda e, i=idx: _update_entry_field(
                        state, i, "evaluation_cadence", e.value, on_change))

                    ui.button(icon="delete", on_click=lambda i=idx: _delete_entry_rule(
                        state, i, on_change)).props("flat dense round color=negative")

                # Conditions
                ui.label("Conditions:").classes("text-sm text-gray-400 mt-2")
                render_condition_builder(
                    state, rule.setdefault("conditions", []),
                    f"entry_{idx}_cond", on_change)

                # Condition groups
                groups = rule.setdefault("condition_groups", [])
                if groups:
                    for gi, grp in enumerate(groups):
                        with ui.card().classes("w-full ml-4 p-2 mt-1"):
                            ui.label(f"Group: {grp.get('label', f'Group {gi+1}')}").classes(
                                "text-sm font-bold")
                            render_condition_builder(
                                state, grp.setdefault("conditions", []),
                                f"entry_{idx}_grp_{gi}", on_change)

                ui.button("Add Condition Group", icon="add",
                          on_click=lambda i=idx: _add_condition_group(
                              state, i, on_change)).props("flat dense")


def _add_entry_rule(state: EditorState, direction: str, on_change):
    rules = state.working_spec.setdefault("entry_rules", [])
    rules.append({
        "label": f"{direction} Entry",
        "direction": direction,
        "evaluation_cadence": "15m",
        "conditions": [],
        "condition_groups": [],
    })
    state.mark_changed()
    on_change()
    ui.navigate.to(f"/editor/{state.composition_id}")


def _update_entry_field(state, idx, field, value, on_change):
    rules = state.working_spec.get("entry_rules", [])
    if idx < len(rules):
        rules[idx][field] = value
        state.mark_changed()
        on_change()


def _delete_entry_rule(state, idx, on_change):
    rules = state.working_spec.get("entry_rules", [])
    if idx < len(rules):
        rules.pop(idx)
        state.mark_changed()
        on_change()
        ui.navigate.to(f"/editor/{state.composition_id}")


def _add_condition_group(state, rule_idx, on_change):
    rules = state.working_spec.get("entry_rules", [])
    if rule_idx < len(rules):
        groups = rules[rule_idx].setdefault("condition_groups", [])
        groups.append({"label": f"Group {len(groups)+1}", "conditions": []})
        state.mark_changed()
        on_change()
        ui.navigate.to(f"/editor/{state.composition_id}")
