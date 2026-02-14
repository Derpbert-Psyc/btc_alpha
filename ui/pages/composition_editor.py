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
    save_last_compilation,
    load_last_compilation,
)
from ui.services.compiler_bridge import compile_spec, save_artifacts, get_capability_registry
from ui.services.indicator_catalog import (
    get_all_indicators,
    get_outputs_for_indicator,
    resolve_indicator_id,
)
from ui.components.indicator_picker import show_indicator_picker
from ui.components.condition_builder import render_condition_builder, _hydrate_condition
from ui.components.exit_rule_editor import render_exit_rules
from ui.components.gate_editor import render_gate_editor
from ui.components.execution_form import render_execution_form
from ui.components.metadata_form import render_metadata_form
from ui.components.compiler_panel import render_compiler_panel

from composition_compiler_v1_5_2 import CompilationError, CAPABILITY_REGISTRY
from strategy_framework_v1_8_0 import INDICATOR_OUTPUTS, INDICATOR_NAME_TO_ID

# Module-level state cache: persists EditorState across page navigations
# so that ui.navigate.to() for re-rendering doesn't lose in-memory edits.
_editor_states: Dict[str, 'EditorState'] = {}


def _resolve_role_conditions(spec: dict) -> None:
    """Resolve role-based conditions to indicator-based ones (for presets).
    Modifies spec in-place. Does NOT mark dirty — this is load-time fixup.
    """
    instances = spec.get("indicator_instances", [])
    if not instances:
        return

    for rule in spec.get("entry_rules", []):
        # Resolve individual conditions with 'role' but no 'indicator'
        for cond in rule.get("conditions", []):
            if "role" in cond and not cond.get("indicator"):
                role = cond["role"]
                matching = [inst for inst in instances if inst.get("role") == role]
                if len(matching) == 1:
                    cond["indicator"] = matching[0]["label"]
                # Convert string value to numeric
                if "value" in cond and isinstance(cond["value"], str):
                    try:
                        cond["value"] = float(cond["value"])
                    except (ValueError, TypeError):
                        pass

        # Expand role_conditions in condition_groups
        for grp in rule.get("condition_groups", []):
            if "role_condition" in grp and not grp.get("conditions"):
                rc = grp["role_condition"]
                role = rc.get("role", "")
                filter_group = rc.get("filter_group", "")
                output = rc.get("output", "")
                operator = rc.get("operator", ">")
                value = rc.get("value", 0)
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = 0

                matching = [
                    inst for inst in instances
                    if inst.get("role") == role
                    and (not filter_group or inst.get("group") == filter_group)
                ]
                grp["conditions"] = []
                for inst in matching:
                    grp["conditions"].append({
                        "indicator": inst["label"],
                        "output": output,
                        "operator": operator,
                        "value": value,
                    })

    # Also convert string values in exit and gate conditions
    for rule in spec.get("exit_rules", []):
        for cond in rule.get("conditions", []):
            if "value" in cond and isinstance(cond["value"], str):
                try:
                    cond["value"] = float(cond["value"])
                except (ValueError, TypeError):
                    pass
    for rule in spec.get("gate_rules", []):
        for cond in rule.get("conditions", []):
            if "value" in cond and isinstance(cond["value"], str):
                try:
                    cond["value"] = float(cond["value"])
                except (ValueError, TypeError):
                    pass


class EditorState:
    """In-memory editor state — separate from on-disk spec."""

    def __init__(self, composition_id: str, spec: dict):
        self.composition_id = composition_id
        self.disk_spec = copy.deepcopy(spec)
        self.working_spec = copy.deepcopy(spec)
        self.last_compilation = None
        self.unsaved = False
        self.active_tab = "Indicators"  # persists across re-renders
        self.active_entry_direction = "LONG"  # persists LONG/SHORT sub-tab

    @property
    def locked(self) -> bool:
        """Strategy is locked if promoted to SHADOW_VALIDATED or beyond."""
        if not hasattr(self, "_locked_cache"):
            self._locked_cache = None
        if self._locked_cache is None:
            from ui.services.promotion_reader import derive_lifecycle_state
            from ui.services.composition_store import load_index
            index = load_index()
            entry = index.get("compositions", {}).get(self.composition_id, {})
            compiled_hash = entry.get("latest_compiled_hash")
            if not compiled_hash:
                self._locked_cache = False
            else:
                lifecycle, _, _ = derive_lifecycle_state(self.composition_id, compiled_hash)
                self._locked_cache = lifecycle in ("SHADOW_VALIDATED", "LIVE_APPROVED")
        return self._locked_cache

    def mark_changed(self, affects_compilation: bool = True):
        self.unsaved = True
        if affects_compilation and self.last_compilation is not None:
            self.last_compilation = None
            from ui.services.composition_store import delete_last_compilation
            delete_last_compilation(self.composition_id)

    def save_to_disk(self):
        save_composition(self.composition_id, self.working_spec)
        self.disk_spec = copy.deepcopy(self.working_spec)
        self.unsaved = False

    def revert(self):
        self.working_spec = copy.deepcopy(self.disk_spec)
        self.unsaved = False
        # Reload saved compilation (revert may restore a compilable state)
        saved = load_last_compilation(self.composition_id)
        self.last_compilation = saved

    def compile(self):
        """Compile the current working spec (in-memory, not disk)."""
        result = compile_spec(self.working_spec)
        self.last_compilation = result
        return result


def composition_editor_page(composition_id: str):
    """Render the composition editor for a given composition."""
    # Reuse existing in-memory state if available (preserves unsaved edits
    # across ui.navigate.to re-renders). Create fresh state only on first load.
    if composition_id in _editor_states:
        state = _editor_states[composition_id]
    else:
        spec = load_composition(composition_id)
        if spec is None:
            ui.label("Composition not found").classes("text-red-400 text-xl p-8")
            ui.button("Back to list", on_click=lambda: ui.navigate.to("/"))
            return
        _resolve_role_conditions(spec)
        state = EditorState(composition_id, spec)
        # Load saved compilation from disk if available
        saved_compilation = load_last_compilation(composition_id)
        if saved_compilation is not None:
            state.last_compilation = saved_compilation
        _editor_states[composition_id] = state

    with ui.column().classes("w-full max-w-7xl mx-auto p-4"):
        # Header bar
        _render_header(state)

        # Unsaved indicator
        unsaved_label = ui.label("").classes("text-amber-400 text-sm")

        def update_unsaved():
            unsaved_label.text = "● Unsaved changes" if state.unsaved else ""

        # Tabbed panels
        with ui.tabs().classes("w-full") as tabs:
            tab_ind = ui.tab("Indicators").classes("tab-indicators")
            tab_entry = ui.tab("Entry Rules").classes("tab-entry")
            tab_exit = ui.tab("Exit Rules").classes("tab-exit")
            tab_gate = ui.tab("Gates").classes("tab-gates")
            tab_exec = ui.tab("Execution").classes("tab-execution")
            tab_meta = ui.tab("Metadata").classes("tab-metadata")

        tab_map = {
            "Indicators": tab_ind, "Entry Rules": tab_entry,
            "Exit Rules": tab_exit, "Gates": tab_gate,
            "Execution": tab_exec, "Metadata": tab_meta,
        }
        initial_tab = tab_map.get(state.active_tab, tab_ind)
        tabs.on("update:model-value", lambda e: setattr(state, 'active_tab', e.args))

        with ui.tab_panels(tabs, value=initial_tab).classes("w-full"):
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

        # Store container ref on state so _do_compile can re-render
        state._compiler_container = compiler_container


def _render_header(state: EditorState):
    """Render the editor header with name, actions, tags."""
    spec = state.working_spec

    with ui.row().classes("w-full items-center justify-between mb-2"):
        with ui.row().classes("items-center gap-4"):
            def _go_back():
                ui.navigate.to("/")
            ui.button(icon="arrow_back", on_click=_go_back).props(
                "flat dense round")
            name_input = ui.input(
                value=spec.get("display_name", "Untitled"),
            ).classes("text-xl font-bold w-96").props("dense borderless")
            name_input.on("change", lambda e: _update_field(
                state, "display_name", e.args))

            # Engine version badge
            ev = spec.get("target_engine_version", "1.8.0")
            ui.badge(f"v{ev}").props(
                f"color={'purple' if ev == '1.8.0' else 'blue'}")

            # Spec version
            ui.badge(f"spec {spec.get('spec_version', '1.5.2')}").props(
                "color=grey outline")

        if state.locked:
            with ui.row().classes("gap-2"):
                ui.button("Compile", icon="build").props("color=primary disable")
                ui.button("Save Draft", icon="save").props("color=positive disable")
                ui.button("Revert", icon="undo").props("color=warning outline disable")
        else:
            with ui.row().classes("gap-2"):
                ui.button("Compile", icon="build",
                          on_click=lambda: _do_compile(state)).props("color=primary")
                ui.button("Save Draft", icon="save",
                          on_click=lambda: _do_save(state)).props("color=positive")
                ui.button("Revert", icon="undo",
                          on_click=lambda: _do_revert(state)).props("color=warning outline")

    # Lock banner
    if state.locked:
        with ui.card().classes("w-full p-3 mb-2 accent-purple"):
            with ui.row().classes("items-center gap-4"):
                ui.icon("lock", color="purple").classes("text-xl")
                ui.label("This strategy is locked (promoted beyond triage). Duplicate to create an editable variant.").classes(
                    "text-purple-400")
                ui.button("Duplicate as Variant", icon="content_copy",
                          on_click=lambda: _duplicate_and_open(state)).props("outline dense")

    # Description
    desc_input = ui.input(
        value=spec.get("description", ""),
        label="Description",
    ).classes("w-full mb-2")
    if state.locked:
        desc_input.props("readonly")
    else:
        desc_input.on("change", lambda e: _update_field(state, "description", e.args))

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
            if not state.locked:
                chip.on("update:selected", lambda e, t=tag: _toggle_archetype(state, t, e.args))


def _duplicate_and_open(state: EditorState):
    """Duplicate a locked strategy and navigate to the new variant."""
    from ui.services.composition_store import duplicate_composition
    new_id = duplicate_composition(state.composition_id)
    ui.notify("Created unlocked variant", type="positive")
    ui.navigate.to(f"/editor/{new_id}")


def _update_field(state: EditorState, field: str, value):
    state.working_spec[field] = value
    cosmetic = field in ("display_name", "description")
    state.mark_changed(affects_compilation=not cosmetic)


def _toggle_archetype(state: EditorState, tag: str, selected: bool):
    tags = state.working_spec.setdefault("archetype_tags", [])
    if selected and tag not in tags:
        tags.append(tag)
    elif not selected and tag in tags:
        tags.remove(tag)
    state.mark_changed()


def _validate_conditions(spec: dict) -> List[str]:
    """Pre-check all conditions for validity. Returns list of issues."""
    issues = []
    all_conds = []
    for rule in spec.get("entry_rules", []):
        for cond in rule.get("conditions", []):
            all_conds.append(("entry", rule.get("label", "entry"), cond))
        for grp in rule.get("condition_groups", []):
            for cond in grp.get("conditions", []):
                all_conds.append(("entry group", grp.get("label", "group"), cond))
    for rule in spec.get("exit_rules", []):
        for cond in rule.get("conditions", []):
            all_conds.append(("exit", rule.get("label", "exit"), cond))
    for rule in spec.get("gate_rules", []):
        for cond in rule.get("conditions", []):
            all_conds.append(("gate", rule.get("label", "gate"), cond))

    for ctx, rule_label, cond in all_conds:
        # Skip empty/new conditions
        if not cond.get("indicator") and not cond.get("output"):
            continue
        h = _hydrate_condition(cond, spec)
        if not h["instance_valid"] and cond.get("indicator"):
            issues.append(f"{ctx} '{rule_label}': instance '{cond.get('indicator')}' not found")
        if not h["output_valid"] and cond.get("output") and h["instance_valid"]:
            issues.append(f"{ctx} '{rule_label}': output '{cond.get('output')}' not available")
    return issues


def _do_compile(state: EditorState):
    """Compile current in-memory spec and re-render compiler panel."""
    # F3: mandatory pre-check
    issues = _validate_conditions(state.working_spec)
    if issues:
        ui.notify(
            "Cannot compile: fix or remove invalid conditions (shown with red border)",
            type="negative", timeout=10000)
        return
    try:
        result = state.compile()
        hash_val = result["strategy_config_hash"]
        save_last_compilation(state.composition_id, result)
        ui.notify(f"Compiled: {hash_val[:24]}...", type="positive")
        # Re-render the compiler panel to show triage section
        if hasattr(state, "_compiler_container"):
            render_compiler_panel(state, state._compiler_container)
    except CompilationError as e:
        ui.notify(f"Compilation error: {e}", type="negative", timeout=10000)
    except Exception as e:
        ui.notify(f"Unexpected error: {e}", type="negative", timeout=10000)


def _do_save(state: EditorState):
    """Save working spec to disk."""
    state.save_to_disk()
    # Update cached state so disk_spec reflects the save
    _editor_states[state.composition_id] = state
    if state.last_compilation:
        hash_val = state.last_compilation["strategy_config_hash"]
        update_compiled_hash(state.composition_id, hash_val)
        save_last_compilation(state.composition_id, state.last_compilation)
        try:
            save_artifacts(state.last_compilation)
        except Exception as e:
            ui.notify(f"Artifact write warning: {e}", type="warning")
    ui.notify("Saved", type="positive")


def _do_revert(state: EditorState):
    """Revert to on-disk spec."""
    state.revert()
    # Update the cached state so the revert is reflected on re-render
    _editor_states[state.composition_id] = state
    ui.notify("Reverted to last save", type="info")
    ui.navigate.to(f"/editor/{state.composition_id}")


def _render_indicators_tab(state: EditorState, on_change):
    """Render the indicator instances tab."""
    spec = state.working_spec
    instances = spec.setdefault("indicator_instances", [])

    with ui.column().classes("w-full"):
        if not state.locked:
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
        if state.locked:
            table.add_slot("body-cell-actions", """
                <q-td :props="props">
                    <q-icon name="lock" color="grey" size="sm" />
                </q-td>
            """)
        else:
            table.add_slot("body-cell-actions", """
                <q-td :props="props">
                    <q-btn flat dense round icon="delete" color="negative"
                           @click="$parent.$emit('delete', props.row.idx)" />
                </q-td>
            """)
            table.on("delete", lambda e: _delete_indicator(state, e.args, on_change))


async def _add_indicator(state: EditorState, on_change):
    """Open indicator picker and add the result."""
    existing_labels = [
        inst.get("label", "") for inst in
        state.working_spec.get("indicator_instances", [])
    ]
    result = await show_indicator_picker(
        state.working_spec.get("target_engine_version", "1.8.0"),
        existing_labels=existing_labels)
    if result:
        state.working_spec.setdefault("indicator_instances", []).append(result)
        state.mark_changed()
        on_change()
        ui.navigate.to(f"/editor/{state.composition_id}")


async def _delete_indicator(state: EditorState, idx, on_change):
    """Delete an indicator, with cascade confirmation if referenced."""
    instances = state.working_spec.get("indicator_instances", [])
    if idx < 0 or idx >= len(instances):
        return

    label = instances[idx].get("label", "")
    refs = _find_indicator_references(state.working_spec, label)

    if refs:
        # Show cascade confirmation dialog
        with ui.dialog() as dialog, ui.card().classes("w-[600px]"):
            ui.label(f"Delete '{label}'?").classes("text-lg font-bold")
            ui.label("This indicator is referenced in:").classes("text-sm mt-2")
            for ref in refs:
                ui.label(f"  - {ref}").classes("text-sm text-amber-400")
            ui.label(
                "Deleting will remove the indicator AND all conditions that reference it. "
                "Rules with no remaining conditions will also be removed."
            ).classes("text-sm text-gray-400 mt-2")
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Delete All", on_click=lambda: dialog.submit(True)).props(
                    "color=negative")
        dialog.open()
        confirmed = await dialog
        if not confirmed:
            return
        _cascade_remove_references(state.working_spec, label)

    instances.pop(idx)
    state.mark_changed()
    on_change()
    ui.navigate.to(f"/editor/{state.composition_id}")


def _find_indicator_references(spec: dict, label: str) -> List[str]:
    """Find all rules/conditions that reference an indicator label (as indicator or ref_indicator)."""
    refs = []
    for rule in spec.get("entry_rules", []):
        rule_label = rule.get("label", "entry rule")
        for i, cond in enumerate(rule.get("conditions", [])):
            if cond.get("indicator") == label or cond.get("ref_indicator") == label:
                refs.append(f"Entry Rule '{rule_label}' condition {i+1}")
        for grp in rule.get("condition_groups", []):
            grp_label = grp.get("label", "group")
            for i, cond in enumerate(grp.get("conditions", [])):
                if cond.get("indicator") == label or cond.get("ref_indicator") == label:
                    refs.append(f"Entry Rule '{rule_label}' / {grp_label} condition {i+1}")
    for rule in spec.get("exit_rules", []):
        rule_label = rule.get("label", "exit rule")
        for i, cond in enumerate(rule.get("conditions", [])):
            if cond.get("indicator") == label or cond.get("ref_indicator") == label:
                refs.append(f"Exit Rule '{rule_label}' condition {i+1}")
    for rule in spec.get("gate_rules", []):
        rule_label = rule.get("label", "gate")
        for i, cond in enumerate(rule.get("conditions", [])):
            if cond.get("indicator") == label or cond.get("ref_indicator") == label:
                refs.append(f"Gate '{rule_label}' condition {i+1}")
    return refs


def _cascade_remove_references(spec: dict, label: str):
    """Remove all conditions referencing an indicator, and empty rules."""
    # Entry rules
    for rule in list(spec.get("entry_rules", [])):
        rule["conditions"] = [
            c for c in rule.get("conditions", [])
            if c.get("indicator") != label
        ]
        for grp in list(rule.get("condition_groups", [])):
            grp["conditions"] = [
                c for c in grp.get("conditions", [])
                if c.get("indicator") != label
            ]
        rule["condition_groups"] = [
            g for g in rule.get("condition_groups", [])
            if g.get("conditions")
        ]
    spec["entry_rules"] = [
        r for r in spec.get("entry_rules", [])
        if r.get("conditions") or r.get("condition_groups")
    ]

    # Exit rules
    for rule in list(spec.get("exit_rules", [])):
        rule["conditions"] = [
            c for c in rule.get("conditions", [])
            if c.get("indicator") != label
        ]
    spec["exit_rules"] = [
        r for r in spec.get("exit_rules", [])
        if r.get("conditions") or r.get("exit_type", "SIGNAL") != "SIGNAL"
    ]

    # Gate rules
    for rule in list(spec.get("gate_rules", [])):
        rule["conditions"] = [
            c for c in rule.get("conditions", [])
            if c.get("indicator") != label
        ]
    spec["gate_rules"] = [
        r for r in spec.get("gate_rules", [])
        if r.get("conditions")
    ]


def _render_entry_rules_tab(state: EditorState, on_change):
    """Render entry rules with LONG/SHORT sub-tabs."""
    spec = state.working_spec
    rules = spec.setdefault("entry_rules", [])

    with ui.column().classes("w-full"):
        with ui.tabs().classes("w-full") as sub_tabs:
            long_tab = ui.tab("LONG")
            short_tab = ui.tab("SHORT")

        sub_tab_map = {"LONG": long_tab, "SHORT": short_tab}
        initial_sub = sub_tab_map.get(state.active_entry_direction, long_tab)
        sub_tabs.on("update:model-value",
                    lambda e: setattr(state, 'active_entry_direction', e.args))

        with ui.tab_panels(sub_tabs, value=initial_sub).classes("w-full"):
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
        if not state.locked:
            ui.button(f"Add {direction} Entry Rule", icon="add",
                      on_click=lambda: _add_entry_rule(state, direction, on_change)).props(
                "color=primary dense")

        if not dir_rules:
            ui.label(f"No {direction} entry rules.").classes("text-gray-400 py-2")
            return

        for idx, rule in dir_rules:
            with ui.card().classes(f"w-full mb-2 p-3 direction-{direction.lower()}"):
                with ui.row().classes("w-full items-center justify-between"):
                    name_input = ui.input(
                        value=rule.get("label", ""),
                        label="Rule name",
                    ).classes("w-64").props("dense")
                    name_input.on("change", lambda e, i=idx: _update_entry_field(
                        state, i, "label", e.args, on_change))

                    raw_cadence = rule.get("evaluation_cadence", "1m")
                    cadence_opts = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "3d"]
                    if raw_cadence not in cadence_opts:
                        cadence_opts.append(raw_cadence)
                    ui.select(
                        cadence_opts,
                        value=raw_cadence,
                        label="Cadence",
                        on_change=lambda e, i=idx: _update_entry_field(
                            state, i, "evaluation_cadence", e.value, on_change),
                    ).classes("w-24").props("dense")

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
                        with ui.card().classes("w-full ml-4 p-2 mt-1 accent-blue"):
                            with ui.row().classes("w-full items-center justify-between"):
                                grp_name = ui.input(
                                    value=grp.get("label", f"Group {gi+1}"),
                                    label="Group name",
                                ).classes("w-48").props("dense")
                                grp_name.on("change", lambda e, ridx=idx, gidx=gi: _update_group_label(
                                    state, ridx, gidx, e.args, on_change))

                                with ui.row().classes("gap-1"):
                                    if gi > 0:
                                        ui.button(icon="arrow_upward",
                                                  on_click=lambda ridx=idx, gidx=gi: _move_group(
                                                      state, ridx, gidx, -1, on_change)).props(
                                            "flat dense round size=sm")
                                    if gi < len(groups) - 1:
                                        ui.button(icon="arrow_downward",
                                                  on_click=lambda ridx=idx, gidx=gi: _move_group(
                                                      state, ridx, gidx, 1, on_change)).props(
                                            "flat dense round size=sm")
                                    ui.button(icon="delete",
                                              on_click=lambda ridx=idx, gidx=gi: _delete_group(
                                                  state, ridx, gidx, on_change)).props(
                                        "flat dense round size=sm color=negative")

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


def _update_group_label(state, rule_idx, group_idx, value, on_change):
    rules = state.working_spec.get("entry_rules", [])
    if rule_idx < len(rules):
        groups = rules[rule_idx].get("condition_groups", [])
        if group_idx < len(groups):
            groups[group_idx]["label"] = value
            state.mark_changed()
            on_change()


def _delete_group(state, rule_idx, group_idx, on_change):
    rules = state.working_spec.get("entry_rules", [])
    if rule_idx < len(rules):
        groups = rules[rule_idx].get("condition_groups", [])
        if group_idx < len(groups):
            groups.pop(group_idx)
            state.mark_changed()
            on_change()
            ui.navigate.to(f"/editor/{state.composition_id}")


def _move_group(state, rule_idx, group_idx, direction, on_change):
    rules = state.working_spec.get("entry_rules", [])
    if rule_idx < len(rules):
        groups = rules[rule_idx].get("condition_groups", [])
        new_idx = group_idx + direction
        if 0 <= new_idx < len(groups):
            groups[group_idx], groups[new_idx] = groups[new_idx], groups[group_idx]
            state.mark_changed()
            on_change()
            ui.navigate.to(f"/editor/{state.composition_id}")
