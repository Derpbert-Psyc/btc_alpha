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
from ui.services.promotion_reader import derive_lifecycle_state, derive_binding_state

LIFECYCLE_ORDER = {
    "DRAFT": 0, "COMPILED": 1, "TRIAGE_PASSED": 2,
    "BASELINE_PLUS_PASSED": 3, "SHADOW_VALIDATED": 4,
    "LIVE_APPROVED": 5, "CORRUPTED": -1,
}

LIFECYCLE_COLORS = {
    "DRAFT": "grey",
    "COMPILED": "blue",
    "TRIAGE_PASSED": "green",
    "BASELINE_PLUS_PASSED": "teal",
    "SHADOW_VALIDATED": "purple",
    "LIVE_APPROVED": "positive",
    "CORRUPTED": "negative",
}


def _build_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Build a display row from an index entry."""
    cid = entry["composition_id"]
    spec = load_composition(cid)

    lifecycle, warning = derive_lifecycle_state(
        cid, entry.get("latest_compiled_hash"))

    archetype_tags = []
    user_tags = []
    warmup = ""
    if spec:
        archetype_tags = spec.get("archetype_tags", [])
        user_tags = spec.get("metadata", {}).get("user_tags", [])
        # warmup derived from compiled hash if available
        hash_val = entry.get("latest_compiled_hash", "")

    return {
        "composition_id": cid,
        "display_name": entry.get("display_name", "Untitled"),
        "archetype_tags": ", ".join(archetype_tags),
        "user_tags": ", ".join(user_tags),
        "lifecycle_state": lifecycle,
        "lifecycle_warning": warning,
        "binding_state": "current",
        "warmup": warmup,
        "hash": (entry.get("latest_compiled_hash") or "—")[:20],
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
                 "LIVE_APPROVED", "CORRUPTED"],
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
                    {"name": "hash", "label": "Hash", "field": "hash", "align": "left"},
                    {"name": "updated_at", "label": "Updated", "field": "updated_at", "align": "left"},
                ]

                table = ui.table(
                    columns=columns,
                    rows=rows,
                    row_key="composition_id",
                ).classes("w-full").props("flat bordered dense")

                # Add slot for lifecycle badge
                table.add_slot("body-cell-lifecycle_state", """
                    <q-td :props="props">
                        <q-badge :color="props.row.lifecycle_state === 'CORRUPTED' ? 'negative' :
                                         props.row.lifecycle_state === 'DRAFT' ? 'grey' :
                                         props.row.lifecycle_state === 'COMPILED' ? 'blue' :
                                         props.row.lifecycle_state === 'TRIAGE_PASSED' ? 'green' :
                                         'purple'"
                                 :label="props.row.lifecycle_state" />
                        <q-tooltip v-if="props.row.lifecycle_warning">
                            {{ props.row.lifecycle_warning }}
                        </q-tooltip>
                    </q-td>
                """)

                # Row click → editor
                table.on("row-click", lambda e: ui.navigate.to(
                    f"/editor/{e.args[1]['composition_id']}"))

                # Action buttons per row
                for row in rows:
                    pass  # Actions handled via context menu below

                # Bottom action bar
                with ui.row().classes("w-full gap-2 mt-2"):
                    ui.button("Refresh", icon="refresh",
                              on_click=refresh_table).props("flat dense")

        # Wire filter/sort changes
        sort_select.on("update:model-value", lambda _: refresh_table())
        lifecycle_filter.on("update:model-value", lambda _: refresh_table())
        tag_filter.on("change", lambda _: refresh_table())

        refresh_table()


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
