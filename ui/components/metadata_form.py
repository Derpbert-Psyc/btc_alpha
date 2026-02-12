"""Metadata Form — thesis, risks, triage params, research notes, tags, lineage."""

from datetime import datetime, timezone
from typing import Callable
from nicegui import ui


def render_metadata_form(state, on_change: Callable):
    """Render metadata tab."""
    spec = state.working_spec
    meta = spec.setdefault("metadata", {})

    with ui.column().classes("w-full"):
        # Thesis
        thesis = ui.textarea(
            value=meta.get("thesis", ""),
            label="Thesis",
        ).classes("w-full").props("outlined")
        thesis.on("change", lambda e: _update_meta(state, "thesis", e.value, on_change))

        # Known risks
        ui.label("Known Risks").classes("text-sm font-bold mt-4")
        risks = meta.setdefault("known_risks", [])
        for i, risk in enumerate(risks):
            with ui.row().classes("w-full items-center gap-2"):
                ri = ui.input(value=risk).classes("flex-grow").props("dense")
                ri.on("change", lambda e, idx=i: _update_risk(state, idx, e.value, on_change))
                ui.button(icon="close",
                          on_click=lambda idx=i: _delete_risk(state, idx, on_change)).props(
                    "flat dense round size=sm color=negative")
        ui.button("Add Risk", icon="add",
                  on_click=lambda: _add_risk(state, on_change)).props("flat dense")

        # Triage sensitive params
        ui.separator()
        ui.label("Triage Sensitive Parameters").classes("text-sm font-bold mt-4")
        tsp = meta.setdefault("triage_sensitive_params", [])
        for i, p in enumerate(tsp):
            with ui.row().classes("w-full items-center gap-2"):
                pi = ui.input(value=p.get("param", ""), label="Param").classes("w-48").props("dense")
                pi.on("change", lambda e, idx=i: _update_tsp(state, idx, "param", e.value, on_change))
                di = ui.number(value=p.get("default", 0), label="Default").classes("w-24").props("dense")
                di.on("change", lambda e, idx=i: _update_tsp(state, idx, "default", e.value, on_change))
                mi = ui.number(value=p.get("min", 0), label="Min").classes("w-24").props("dense")
                mi.on("change", lambda e, idx=i: _update_tsp(state, idx, "min", e.value, on_change))
                mx = ui.number(value=p.get("max", 0), label="Max").classes("w-24").props("dense")
                mx.on("change", lambda e, idx=i: _update_tsp(state, idx, "max", e.value, on_change))
                ui.button(icon="close",
                          on_click=lambda idx=i: _delete_tsp(state, idx, on_change)).props(
                    "flat dense round size=sm color=negative")
        ui.button("Add Parameter", icon="add",
                  on_click=lambda: _add_tsp(state, on_change)).props("flat dense")

        # User tags
        ui.separator()
        ui.label("User Tags").classes("text-sm font-bold mt-4")
        user_tags = meta.setdefault("user_tags", [])
        tag_input = ui.input(label="Add tag (press Enter)").classes("w-64").props("dense")

        def add_tag(e):
            val = tag_input.value.strip()
            if val and val not in user_tags:
                user_tags.append(val)
                state.mark_changed()
                on_change()
                tag_input.value = ""
                ui.navigate.to(f"/editor/{state.composition_id}")
        tag_input.on("keydown.enter", add_tag)

        with ui.row().classes("gap-2 flex-wrap"):
            for t in user_tags:
                with ui.chip(t, removable=True) as chip:
                    chip.on("remove", lambda t=t: _remove_tag(state, t, on_change))

        # Forked from lineage
        forked_from = meta.get("forked_from")
        if forked_from:
            ui.separator()
            ui.label("Lineage").classes("text-sm font-bold mt-4")
            with ui.row().classes("items-center gap-2"):
                ui.label("Forked from:").classes("text-gray-400")
                ui.link(forked_from[:16] + "...", f"/editor/{forked_from}").classes(
                    "text-blue-400")

        # Archive reason (read-only)
        archive_reason = meta.get("archive_reason")
        if archive_reason:
            ui.separator()
            with ui.row().classes("items-center gap-2 mt-4"):
                ui.icon("archive", color="amber")
                ui.label(f"Archived: {archive_reason}").classes("text-amber-400")

        # Spec version (read-only)
        ui.separator()
        ui.label(f"Spec Version: {spec.get('spec_version', '1.5.2')}").classes(
            "text-sm text-gray-400 mt-4")

        # Research notes
        ui.separator()
        ui.label("Research Notes").classes("text-sm font-bold mt-4")
        notes = meta.setdefault("research_notes", [])

        note_input = ui.textarea(label="Add note").classes("w-full").props("outlined dense")
        ui.button("Add Note", icon="note_add",
                  on_click=lambda: _add_note(state, note_input, on_change)).props("dense")

        # Display in reverse chronological
        for note in reversed(notes):
            with ui.card().classes("w-full p-2 mb-1"):
                ui.label(note.get("timestamp", "")).classes("text-xs text-gray-400")
                ui.label(note.get("text", "")).classes("text-sm")


def _update_meta(state, field, value, on_change):
    state.working_spec.setdefault("metadata", {})[field] = value
    state.mark_changed()
    on_change()


def _update_risk(state, idx, value, on_change):
    risks = state.working_spec.get("metadata", {}).get("known_risks", [])
    if idx < len(risks):
        risks[idx] = value
        state.mark_changed()
        on_change()


def _add_risk(state, on_change):
    state.working_spec.setdefault("metadata", {}).setdefault("known_risks", []).append("")
    state.mark_changed()
    on_change()
    ui.navigate.to(f"/editor/{state.composition_id}")


def _delete_risk(state, idx, on_change):
    risks = state.working_spec.get("metadata", {}).get("known_risks", [])
    if idx < len(risks):
        risks.pop(idx)
        state.mark_changed()
        on_change()
        ui.navigate.to(f"/editor/{state.composition_id}")


def _update_tsp(state, idx, field, value, on_change):
    tsp = state.working_spec.get("metadata", {}).get("triage_sensitive_params", [])
    if idx < len(tsp):
        tsp[idx][field] = value
        state.mark_changed()
        on_change()


def _add_tsp(state, on_change):
    state.working_spec.setdefault("metadata", {}).setdefault(
        "triage_sensitive_params", []).append(
            {"param": "", "default": 0, "min": 0, "max": 0})
    state.mark_changed()
    on_change()
    ui.navigate.to(f"/editor/{state.composition_id}")


def _delete_tsp(state, idx, on_change):
    tsp = state.working_spec.get("metadata", {}).get("triage_sensitive_params", [])
    if idx < len(tsp):
        tsp.pop(idx)
        state.mark_changed()
        on_change()
        ui.navigate.to(f"/editor/{state.composition_id}")


def _remove_tag(state, tag, on_change):
    tags = state.working_spec.get("metadata", {}).get("user_tags", [])
    if tag in tags:
        tags.remove(tag)
        state.mark_changed()
        on_change()
        ui.navigate.to(f"/editor/{state.composition_id}")


def _add_note(state, note_input, on_change):
    text = note_input.value.strip()
    if not text:
        return
    notes = state.working_spec.setdefault("metadata", {}).setdefault("research_notes", [])
    notes.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
    })
    state.mark_changed()
    on_change()
    note_input.value = ""
    ui.navigate.to(f"/editor/{state.composition_id}")
