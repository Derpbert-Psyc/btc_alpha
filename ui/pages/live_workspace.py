"""Live Workspace — strategies in LIVE_APPROVED lifecycle."""

from typing import Any, Dict, List

from nicegui import ui

from ui.components.workspace_nav import render_workspace_nav
from ui.services.composition_store import list_compositions, load_composition, load_index
from ui.services.promotion_reader import (
    derive_lifecycle_state,
    demote_lifecycle,
    get_best_triage_tier,
)
from ui.pages.shadow_workspace import _check_api_config


def _get_live_strategies() -> List[Dict[str, Any]]:
    """Get all strategies in LIVE_APPROVED lifecycle."""
    entries = list_compositions()
    index = load_index()
    compositions = index.get("compositions", {})
    results = []

    for entry in entries:
        cid = entry["composition_id"]
        idx_entry = compositions.get(cid, {})
        compiled_hash = idx_entry.get("latest_compiled_hash")
        lifecycle, dataset_count, warning = derive_lifecycle_state(cid, compiled_hash)

        if lifecycle != "LIVE_APPROVED":
            continue

        spec = load_composition(cid)
        triage_tier = get_best_triage_tier(compiled_hash)

        results.append({
            "composition_id": cid,
            "display_name": entry.get("display_name", "Untitled"),
            "compiled_hash": compiled_hash or "",
            "dataset_count": dataset_count,
            "triage_tier": triage_tier,
            "spec": spec,
        })

    return results


def live_workspace_page():
    """Render the Live operations workspace."""
    with ui.column().classes("w-full max-w-7xl mx-auto p-4"):
        render_workspace_nav("live")

        # Header
        with ui.row().classes("w-full items-center gap-4 mb-4"):
            ui.icon("bolt", color="green").classes("text-3xl")
            ui.label("LIVE OPERATIONS").classes("text-2xl font-bold")

        # Risk banner
        with ui.card().classes("w-full p-3 mb-4 accent-red"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("warning", color="red").classes("text-xl")
                ui.label(
                    "LIVE TRADING USES REAL FUNDS. "
                    "Ensure thorough shadow validation before enabling."
                ).classes("text-red-400 font-bold")

        strategies = _get_live_strategies()

        if not strategies:
            with ui.card().classes("w-full p-8 text-center"):
                ui.icon("bolt", color="grey").classes("text-4xl mb-2")
                ui.label("No strategies in live mode.").classes(
                    "text-lg text-gray-400")
                ui.label(
                    "Promote a strategy from Shadow Monitor to begin live trading."
                ).classes("text-sm text-gray-500 mt-1")
            return

        ui.label(f"Live Strategies: {len(strategies)}").classes(
            "text-sm text-gray-400 mb-2")

        for strat in strategies:
            _render_live_card(strat)


def _render_live_card(strat: dict):
    """Render a single live strategy card."""
    cid = strat["composition_id"]
    compiled_hash = strat["compiled_hash"]
    tier = strat["triage_tier"]

    with ui.card().classes("w-full mb-4 p-4 accent-green"):
        # Info row
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-3"):
                ui.label(strat["display_name"]).classes("text-lg font-bold")
                ui.badge("LIVE_APPROVED", color="positive")
                if tier:
                    tier_css = {"S": "tier-badge-s", "A": "tier-badge-a",
                                "B": "tier-badge-b"}.get(tier, "")
                    ui.badge(f"Tier {tier}", color="purple").classes(tier_css)
                if strat["dataset_count"] > 0:
                    ds = "dataset" if strat["dataset_count"] == 1 else "datasets"
                    ui.label(f"{strat['dataset_count']} {ds}").classes(
                        "text-sm text-gray-400")

            ui.label(compiled_hash[:24] + "...").classes(
                "monospace text-xs text-gray-400")

        # Live Execution Panel
        ui.separator().classes("my-3")
        _render_live_execution_panel()

        # Action buttons
        ui.separator().classes("my-3")
        with ui.row().classes("gap-3 flex-wrap"):
            if compiled_hash:
                hash_short = compiled_hash
                if hash_short.startswith("sha256:"):
                    hash_short = hash_short[7:]
                ui.button("View Triage Results", icon="science",
                          on_click=lambda h=hash_short: ui.navigate.to(
                              f"/triage/{h}")).props("outline dense")

            ui.button("Open in Editor", icon="edit",
                      on_click=lambda c=cid: ui.navigate.to(
                          f"/editor/{c}")).props("outline dense")

            ui.button("Demote to Shadow", icon="arrow_downward",
                      on_click=lambda h=compiled_hash: _demote_to_shadow(
                          h)).props("color=negative outline dense")


def _render_live_execution_panel():
    """Render the live execution panel based on API config state."""
    api = _check_api_config("live")

    if not api["configured"]:
        with ui.card().classes("w-full p-4 accent-red"):
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon("error", color="red").classes("text-xl")
                ui.label("NO PRODUCTION API CONFIGURED").classes(
                    "text-lg font-bold text-red-400")

            ui.label(
                "To run live trading, configure your Bybit PRODUCTION API "
                "credentials in config/api_keys.json:"
            ).classes("text-sm text-gray-400 mt-2")

            ui.code(
                '{\n'
                '  "bybit_live_key": "your-production-key",\n'
                '  "bybit_live_secret": "your-production-secret"\n'
                '}',
                language="json",
            ).classes("w-full mt-2")

            with ui.row().classes("items-center gap-2 mt-3"):
                ui.icon("warning", color="amber")
                ui.label(
                    "LIVE TRADING USES REAL FUNDS. "
                    "Ensure your strategy has been thoroughly validated "
                    "in shadow mode before going live."
                ).classes("text-sm text-amber-400")
    else:
        with ui.card().classes("w-full p-4 accent-green"):
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon("check_circle", color="green").classes("text-xl")
                ui.label("PRODUCTION API CONFIGURED").classes(
                    "text-lg font-bold text-green-400")

            ui.label(
                "Bybit production credentials detected. "
                "Live execution engine coming in Phase 2."
            ).classes("text-sm text-gray-400 mt-2")

            ui.button("Start Live", icon="play_arrow").props(
                "color=positive disable").classes("mt-2")
            ui.label("Live execution not yet implemented.").classes(
                "text-xs text-gray-500 mt-1")


async def _demote_to_shadow(compiled_hash: str):
    """Demote from LIVE_APPROVED back to SHADOW_VALIDATED."""
    with ui.dialog() as dlg, ui.card().classes("w-[500px]"):
        ui.label("Demote to Shadow?").classes("text-lg font-bold text-amber-400")
        ui.label(
            "This will remove the live approval status. "
            "The strategy will revert to SHADOW_VALIDATED."
        ).classes("text-sm mt-2 text-gray-400")
        ui.label(
            "All promotion history is preserved for audit."
        ).classes("text-sm text-gray-400")
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            ui.button("Demote", on_click=lambda: dlg.submit(True)).props(
                "color=negative outline")
    dlg.open()
    confirmed = await dlg
    if not confirmed:
        return

    try:
        count = demote_lifecycle(compiled_hash, "LIVE_APPROVED")
        if count > 0:
            ui.notify(f"Demoted to Shadow ({count} artifact(s) disabled)",
                      type="info")
        else:
            ui.notify("No live promotion artifacts found", type="warning")
        ui.navigate.to("/shadow")
    except Exception as e:
        ui.notify(f"Demotion error: {e}", type="negative")
