"""Strategy List — home screen with sort/filter, lifecycle, binding state."""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nicegui import ui

from ui.services.composition_store import (
    create_composition,
    delete_composition,
    duplicate_composition,
    list_compositions,
    load_composition,
    archive_composition,
    load_index,
)
from ui.services.promotion_reader import derive_lifecycle_state, derive_binding_state, get_best_triage_tier
from ui.pages.composition_editor import _editor_states

LIFECYCLE_ORDER = {
    "DRAFT": 0, "COMPILED": 1, "TRIAGE_PASSED": 2,
    "BASELINE_PLUS_PASSED": 3, "SHADOW_VALIDATED": 4,
    "LIVE_APPROVED": 5, "CORRUPTED": -1, "ARCHIVED": -2,
}

LIFECYCLE_COLORS = {
    "DRAFT": "grey",
    "COMPILED": "blue",
    "TRIAGE_PASSED": "green",
    "BASELINE_PLUS_PASSED": "teal",
    "SHADOW_VALIDATED": "purple",
    "LIVE_APPROVED": "positive",
    "CORRUPTED": "negative",
    "ARCHIVED": "grey",
}


def _build_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Build a display row from an index entry."""
    cid = entry["composition_id"]
    spec = load_composition(cid)

    lifecycle, dataset_count, warning = derive_lifecycle_state(
        cid, entry.get("latest_compiled_hash"))

    archetype_tags = []
    user_tags = []
    archive_reason = None
    if spec:
        archetype_tags = spec.get("archetype_tags", [])
        user_tags = spec.get("metadata", {}).get("user_tags", [])
        archive_reason = spec.get("metadata", {}).get("archive_reason")

    # Override lifecycle if archived
    if archive_reason:
        lifecycle = "ARCHIVED"

    # Format lifecycle with dataset count
    lifecycle_display = lifecycle
    if dataset_count > 0 and lifecycle not in ("DRAFT", "COMPILED", "CORRUPTED", "ARCHIVED"):
        ds_label = "dataset" if dataset_count == 1 else "datasets"
        lifecycle_display = f"{lifecycle} ({dataset_count} {ds_label})"

    # Get best triage tier (S/A/B or None)
    triage_tier = get_best_triage_tier(entry.get("latest_compiled_hash"))

    locked = lifecycle in ("SHADOW_VALIDATED", "LIVE_APPROVED")

    return {
        "composition_id": cid,
        "display_name": entry.get("display_name", "Untitled"),
        "archetype_tags": ", ".join(archetype_tags),
        "user_tags": ", ".join(user_tags),
        "lifecycle_state": lifecycle,
        "lifecycle_display": lifecycle_display,
        "lifecycle_warning": warning,
        "triage_tier": triage_tier or "",
        "locked": locked,
        "hash": (entry.get("latest_compiled_hash") or "\u2014")[:20],
        "updated_at": entry.get("updated_at", ""),
    }


def strategy_list_page():
    """Render the strategy list home screen."""
    with ui.column().classes("w-full max-w-7xl mx-auto p-4"):
        # Header
        with ui.row().classes("w-full items-center justify-between mb-4"):
            ui.label("BTC Alpha Research Panel").classes(
                "text-2xl font-bold")
            with ui.row().classes("gap-2"):
                ui.button("New Composition", on_click=_new_composition).props(
                    "color=primary")
                ui.button("Load Preset", on_click=lambda: ui.navigate.to("/presets")).props(
                    "color=secondary outline")

        # Filters
        filter_container = ui.row().classes("w-full gap-4 mb-4 items-end")
        with filter_container:
            sort_select = ui.select(
                ["updated_at", "display_name", "lifecycle_state"],
                value="updated_at",
                label="Sort by",
            ).classes("w-48")
            lifecycle_filter = ui.select(
                ["All", "DRAFT", "COMPILED", "TRIAGE_PASSED",
                 "BASELINE_PLUS_PASSED", "SHADOW_VALIDATED",
                 "LIVE_APPROVED", "CORRUPTED", "ARCHIVED"],
                value="All",
                label="Lifecycle",
            ).classes("w-48")
            tag_filter = ui.input(label="Filter by tag").classes("w-48")

        # Table container
        table_container = ui.column().classes("w-full")

        def refresh_table():
            table_container.clear()
            entries = list_compositions()
            rows = [_build_row(e) for e in entries]

            # Apply filters
            lf = lifecycle_filter.value
            if lf != "All":
                rows = [r for r in rows if r["lifecycle_state"] == lf]

            tf = tag_filter.value.strip().lower()
            if tf:
                rows = [r for r in rows
                        if tf in r["archetype_tags"].lower()
                        or tf in r["user_tags"].lower()]

            # Apply sort
            sv = sort_select.value
            if sv == "updated_at":
                rows.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
            elif sv == "display_name":
                rows.sort(key=lambda r: r.get("display_name", "").lower())
            elif sv == "lifecycle_state":
                rows.sort(key=lambda r: LIFECYCLE_ORDER.get(
                    r["lifecycle_state"], -1))

            with table_container:
                if not rows:
                    ui.label("No compositions found. Create one or load a preset.").classes(
                        "text-gray-400 py-8 text-center w-full")
                    return

                columns = [
                    {"name": "display_name", "label": "Name", "field": "display_name", "align": "left", "sortable": True},
                    {"name": "archetype_tags", "label": "Archetype", "field": "archetype_tags", "align": "left"},
                    {"name": "user_tags", "label": "Tags", "field": "user_tags", "align": "left"},
                    {"name": "lifecycle_state", "label": "Lifecycle", "field": "lifecycle_state", "align": "center"},
                    {"name": "triage_tier", "label": "Tier", "field": "triage_tier", "align": "center"},
                    {"name": "hash", "label": "Hash", "field": "hash", "align": "left"},
                    {"name": "updated_at", "label": "Updated", "field": "updated_at", "align": "left"},
                    {"name": "actions", "label": "", "field": "actions", "align": "center"},
                ]

                table = ui.table(
                    columns=columns,
                    rows=rows,
                    row_key="composition_id",
                ).classes("w-full").props("flat bordered dense")

                # Archetype tags as chips
                table.add_slot("body-cell-archetype_tags", """
                    <q-td :props="props">
                        <q-chip v-for="tag in (props.row.archetype_tags || '').split(', ').filter(t => t)"
                                :key="tag" :label="tag" dense size="sm" color="dark" text-color="white"
                                class="q-mr-xs" style="font-size: 10px;" />
                    </q-td>
                """)

                # User tags as chips
                table.add_slot("body-cell-user_tags", """
                    <q-td :props="props">
                        <q-chip v-for="tag in (props.row.user_tags || '').split(', ').filter(t => t)"
                                :key="tag" :label="tag" dense size="sm" outline
                                class="q-mr-xs" style="font-size: 10px;" />
                    </q-td>
                """)

                # Hash as monospace
                table.add_slot("body-cell-hash", """
                    <q-td :props="props">
                        <span class="monospace" style="font-size: 11px;">{{ props.row.hash }}</span>
                    </q-td>
                """)

                # Add slot for lifecycle badge
                table.add_slot("body-cell-lifecycle_state", """
                    <q-td :props="props">
                        <q-icon v-if="props.row.locked" name="lock" color="purple" size="xs" class="q-mr-xs" />
                        <q-badge :color="props.row.lifecycle_state === 'CORRUPTED' ? 'negative' :
                                         props.row.lifecycle_state === 'DRAFT' ? 'grey' :
                                         props.row.lifecycle_state === 'COMPILED' ? 'blue' :
                                         props.row.lifecycle_state === 'TRIAGE_PASSED' ? 'green' :
                                         props.row.lifecycle_state === 'ARCHIVED' ? 'grey' :
                                         'purple'"
                                 :label="props.row.lifecycle_display || props.row.lifecycle_state" />
                        <q-tooltip v-if="props.row.lifecycle_warning">
                            {{ props.row.lifecycle_warning }}
                        </q-tooltip>
                    </q-td>
                """)

                # Tier badge (S/A/B letter grade) with gradient styling
                table.add_slot("body-cell-triage_tier", """
                    <q-td :props="props">
                        <q-badge v-if="props.row.triage_tier"
                                 :color="props.row.triage_tier === 'S' ? 'purple' :
                                         props.row.triage_tier === 'A' ? 'green' :
                                         props.row.triage_tier === 'B' ? 'blue' : 'grey'"
                                 :label="props.row.triage_tier"
                                 :class="props.row.triage_tier === 'S' ? 'tier-badge-s' :
                                         props.row.triage_tier === 'A' ? 'tier-badge-a' :
                                         props.row.triage_tier === 'B' ? 'tier-badge-b' : ''" />
                    </q-td>
                """)

                # Actions column
                table.add_slot("body-cell-actions", """
                    <q-td :props="props">
                        <q-btn v-if="props.row.lifecycle_state === 'TRIAGE_PASSED'"
                               flat dense round icon="trending_up" size="sm" color="purple"
                               @click.stop="$parent.$emit('promote_shadow', props.row.composition_id)">
                            <q-tooltip>Promote to Shadow</q-tooltip>
                        </q-btn>
                        <q-btn v-if="props.row.lifecycle_state === 'SHADOW_VALIDATED'"
                               flat dense round icon="rocket_launch" size="sm" color="positive"
                               @click.stop="$parent.$emit('promote_live', props.row.composition_id)">
                            <q-tooltip>Promote to Live</q-tooltip>
                        </q-btn>
                        <q-btn flat dense round icon="content_copy" size="sm"
                               @click.stop="$parent.$emit('duplicate', props.row.composition_id)" />
                        <q-btn flat dense round icon="archive" size="sm" color="grey"
                               @click.stop="$parent.$emit('archive', props.row.composition_id)" />
                        <q-btn flat dense round icon="delete" size="sm" color="negative"
                               @click.stop="$parent.$emit('delete', props.row.composition_id)" />
                    </q-td>
                """)

                table.on("duplicate", lambda e: _do_duplicate(e.args, refresh_table))
                table.on("archive", lambda e: _do_archive(e.args, rows, refresh_table))
                table.on("delete", lambda e: _do_delete(e.args, rows, refresh_table))
                table.on("promote_shadow", lambda e: _do_promote(e.args, rows, "SHADOW_VALIDATED", refresh_table))
                table.on("promote_live", lambda e: _do_promote(e.args, rows, "LIVE_APPROVED", refresh_table))

                # Row click → editor
                table.on("row-click", lambda e: ui.navigate.to(
                    f"/editor/{e.args[1]['composition_id']}"))

                # Bottom action bar
                with ui.row().classes("w-full gap-2 mt-2"):
                    ui.button("Refresh", icon="refresh",
                              on_click=refresh_table).props("flat dense")

        # Wire filter/sort changes
        sort_select.on("update:model-value", lambda _: refresh_table())
        lifecycle_filter.on("update:model-value", lambda _: refresh_table())
        tag_filter.on("change", lambda _: refresh_table())

        refresh_table()


def _do_duplicate(cid, refresh_table):
    """Duplicate a composition and navigate to editor."""
    try:
        new_id = duplicate_composition(cid)
        ui.notify("Duplicated — opening variant", type="positive")
        ui.navigate.to(f"/editor/{new_id}")
    except Exception as e:
        ui.notify(f"Duplicate failed: {e}", type="negative")


async def _do_archive(cid, rows, refresh_table):
    """Archive a composition with reason dialog."""
    display_name = "composition"
    for r in rows:
        if r["composition_id"] == cid:
            display_name = r["display_name"]
            break

    with ui.dialog() as dialog, ui.card().classes("w-[500px]"):
        ui.label(f"Archive '{display_name}'?").classes("text-lg font-bold")
        ui.label("Enter a reason for archiving:").classes("text-sm mt-2")
        reason_input = ui.input(label="Reason").classes("w-full")
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            def do_archive():
                reason = reason_input.value.strip() or "No reason given"
                archive_composition(cid, reason)
                dialog.close()
                ui.notify(f"Archived: {display_name}", type="info")
                refresh_table()

            ui.button("Archive", on_click=do_archive).props("color=grey")

    dialog.open()


async def _do_delete(cid, rows, refresh_table):
    """Delete with 3-step confirmation."""
    display_name = "composition"
    for r in rows:
        if r["composition_id"] == cid:
            display_name = r["display_name"]
            break

    # Step 1
    with ui.dialog() as dialog1, ui.card().classes("w-[500px]"):
        ui.label(f"Delete '{display_name}'?").classes("text-lg font-bold")
        ui.label("This will permanently remove it.").classes("text-sm mt-2")
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dialog1.close).props("flat")
            ui.button("Continue", on_click=lambda: dialog1.submit(True)).props(
                "color=negative outline")
    dialog1.open()
    confirmed1 = await dialog1
    if not confirmed1:
        return

    # Step 2
    with ui.dialog() as dialog2, ui.card().classes("w-[500px]"):
        ui.label("This cannot be undone.").classes("text-lg font-bold text-red-400")
        ui.label("Type DELETE to confirm:").classes("text-sm mt-2")
        confirm_input = ui.input(label="Type DELETE").classes("w-full")
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dialog2.close).props("flat")
            delete_btn = ui.button("Delete Forever",
                                   on_click=lambda: dialog2.submit(True)).props(
                "color=negative disable")
            confirm_input.on("update:model-value",
                             lambda e: delete_btn.props(
                                 remove="disable") if e.args == "DELETE" else delete_btn.props("disable"))
    dialog2.open()
    confirmed2 = await dialog2
    if not confirmed2:
        return

    # Step 3: execute
    _editor_states.pop(cid, None)
    delete_composition(cid)
    ui.notify(f"Deleted: {display_name}", type="positive")
    refresh_table()


async def _do_promote(cid, rows, lifecycle_tier, refresh_table):
    """Promote a strategy to the next lifecycle stage."""
    display_name = "composition"
    for r in rows:
        if r["composition_id"] == cid:
            display_name = r["display_name"]
            break

    index = load_index()
    entry = index.get("compositions", {}).get(cid, {})
    compiled_hash = entry.get("latest_compiled_hash")
    if not compiled_hash:
        ui.notify("No compiled hash found — compile first", type="warning")
        return

    tier_labels = {
        "SHADOW_VALIDATED": ("Promote to Shadow", "shadow trading validation"),
        "LIVE_APPROVED": ("Promote to Live", "live trading with real capital"),
    }
    label, desc = tier_labels.get(lifecycle_tier, ("Promote", "next stage"))

    if lifecycle_tier == "LIVE_APPROVED":
        # Strong confirmation for live
        with ui.dialog() as dialog, ui.card().classes("w-[500px]"):
            ui.label(f"PROMOTE '{display_name}' TO LIVE?").classes("text-lg font-bold text-red-400")
            ui.label(f"This approves the strategy for {desc}.").classes("text-sm mt-2")
            ui.label("Type LIVE to confirm:").classes("text-sm text-gray-400 mt-2")
            confirm_input = ui.input(label="Type LIVE").classes("w-full")
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                go_btn = ui.button("Go Live", on_click=lambda: dialog.submit(True)).props(
                    "color=positive disable")
                confirm_input.on("update:model-value",
                    lambda e: go_btn.props(remove="disable") if e.args == "LIVE"
                    else go_btn.props("disable"))
        dialog.open()
        confirmed = await dialog
        if not confirmed:
            return
    else:
        with ui.dialog() as dialog, ui.card().classes("w-[500px]"):
            ui.label(f"{label}: '{display_name}'?").classes("text-lg font-bold")
            ui.label(f"This marks the strategy for {desc}.").classes("text-sm mt-2 text-gray-400")
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Promote", on_click=lambda: dialog.submit(True)).props("color=purple")
        dialog.open()
        confirmed = await dialog
        if not confirmed:
            return

    try:
        from ui.services.research_services import write_lifecycle_promotion
        from strategy_framework_v1_8_0 import compute_config_hash
        spec = load_composition(cid)
        spec_hash = compute_config_hash(spec) if spec else ""
        write_lifecycle_promotion(
            strategy_config_hash=compiled_hash,
            composition_spec_hash=spec_hash,
            dataset_prefix="manual",
            lifecycle_tier=lifecycle_tier,
        )
        ui.notify(f"Promoted to {lifecycle_tier}", type="positive")
        refresh_table()
    except Exception as e:
        ui.notify(f"Promotion error: {e}", type="negative")


async def _new_composition():
    """Create a new empty composition and navigate to editor."""
    spec = {
        "indicator_instances": [],
        "entry_rules": [],
        "exit_rules": [],
        "gate_rules": [],
        "execution_params": {
            "direction": "BOTH",
            "leverage": 1,
            "sizing_mode": "RISK_FRACTION",
            "risk_fraction_bps": 100,
            "flip_enabled": False,
        },
        "archetype_tags": [],
        "metadata": {
            "thesis": "",
            "known_risks": [],
            "triage_sensitive_params": [],
            "research_notes": [],
            "user_tags": [],
        },
    }
    cid = create_composition(spec, display_name="New Strategy")
    ui.navigate.to(f"/editor/{cid}")
