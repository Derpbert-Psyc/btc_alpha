"""Shadow Workspace — strategies in SHADOW_VALIDATED lifecycle."""

import json
import os
from typing import Any, Dict, List, Optional

from nicegui import ui

from ui.components.workspace_nav import render_workspace_nav
from ui.services.composition_store import list_compositions, load_composition, load_index
from ui.services.promotion_reader import (
    derive_lifecycle_state,
    demote_lifecycle,
    get_best_triage_tier,
)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _check_api_config(mode: str) -> dict:
    """Check for API credentials.

    mode: 'testnet' | 'live'
    Returns {"configured": bool, "key_name": str, "secret_name": str}
    """
    if mode == "testnet":
        key_name = "bybit_testnet_key"
        secret_name = "bybit_testnet_secret"
    else:
        key_name = "bybit_live_key"
        secret_name = "bybit_live_secret"

    config_path = os.path.join(PROJECT_ROOT, "config", "api_keys.json")
    if not os.path.exists(config_path):
        return {"configured": False, "key_name": key_name, "secret_name": secret_name}

    try:
        with open(config_path) as f:
            config = json.load(f)
        key_val = config.get(key_name, "").strip()
        secret_val = config.get(secret_name, "").strip()
        return {
            "configured": bool(key_val and secret_val),
            "key_name": key_name,
            "secret_name": secret_name,
        }
    except (json.JSONDecodeError, IOError):
        return {"configured": False, "key_name": key_name, "secret_name": secret_name}


def _get_shadow_strategies() -> List[Dict[str, Any]]:
    """Get all strategies in SHADOW_VALIDATED lifecycle."""
    entries = list_compositions()
    index = load_index()
    compositions = index.get("compositions", {})
    results = []

    for entry in entries:
        cid = entry["composition_id"]
        idx_entry = compositions.get(cid, {})
        compiled_hash = idx_entry.get("latest_compiled_hash")
        lifecycle, dataset_count, warning = derive_lifecycle_state(cid, compiled_hash)

        if lifecycle != "SHADOW_VALIDATED":
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


def shadow_workspace_page():
    """Render the Shadow monitoring workspace."""
    with ui.column().classes("w-full max-w-7xl mx-auto p-4"):
        render_workspace_nav("shadow")

        # Header
        with ui.row().classes("w-full items-center gap-4 mb-4"):
            ui.icon("visibility", color="purple").classes("text-3xl")
            ui.label("SHADOW MONITOR").classes("text-2xl font-bold")

        strategies = _get_shadow_strategies()

        if not strategies:
            with ui.card().classes("w-full p-8 text-center"):
                ui.icon("visibility_off", color="grey").classes("text-4xl mb-2")
                ui.label("No strategies in shadow mode.").classes(
                    "text-lg text-gray-400")
                ui.label(
                    "Promote a strategy from Research Lab to begin shadow validation."
                ).classes("text-sm text-gray-500 mt-1")
            return

        ui.label(f"Strategies in Shadow: {len(strategies)}").classes(
            "text-sm text-gray-400 mb-2")

        for strat in strategies:
            _render_shadow_card(strat)


def _render_shadow_card(strat: dict):
    """Render a single shadow strategy card."""
    cid = strat["composition_id"]
    compiled_hash = strat["compiled_hash"]
    tier = strat["triage_tier"]

    with ui.card().classes("w-full mb-4 p-4 accent-purple"):
        # Info row
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-3"):
                ui.label(strat["display_name"]).classes("text-lg font-bold")
                ui.badge("SHADOW_VALIDATED", color="purple")
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

        # Shadow Execution Panel
        ui.separator().classes("my-3")
        _render_shadow_execution_panel(strat)

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

            ui.button("Promote to Live", icon="rocket_launch",
                      on_click=lambda c=cid, h=compiled_hash: _promote_to_live(
                          c, h)).props("color=positive dense")

            ui.button("Demote to Research", icon="arrow_downward",
                      on_click=lambda h=compiled_hash: _demote_to_research(
                          h)).props("color=negative outline dense")


def _render_shadow_execution_panel(strat: dict):
    """Render shadow execution panel with start/stop and live status."""
    api = _check_api_config("testnet")
    cid = strat["composition_id"]

    if not api["configured"]:
        with ui.card().classes("w-full p-4 accent-amber"):
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon("warning", color="amber").classes("text-xl")
                ui.label("NO API CONFIGURED").classes(
                    "text-lg font-bold text-amber-400")

            ui.label(
                "To run shadow trading, configure your Bybit TESTNET API "
                "credentials in config/api_keys.json:"
            ).classes("text-sm text-gray-400 mt-2")

            ui.code(
                '{\n'
                '  "bybit_testnet_key": "your-testnet-key",\n'
                '  "bybit_testnet_secret": "your-testnet-secret"\n'
                '}',
                language="json",
            ).classes("w-full mt-2")

            ui.label(
                "Shadow mode uses the Bybit TESTNET API. "
                "No real funds are at risk."
            ).classes("text-sm text-gray-400 mt-2 italic")
        return

    from ui.services.shadow_runner import get_runner
    runner = get_runner(cid)

    with ui.card().classes("w-full p-4 accent-green"):
        with ui.row().classes("items-center gap-2 mb-2"):
            ui.icon("check_circle", color="green").classes("text-xl")
            ui.label("API CONFIGURED").classes("text-lg font-bold text-green-400")

        status_lbl = ui.label("").classes("text-sm font-bold text-green-400")
        with ui.row().classes("gap-6 mt-2 flex-wrap text-sm"):
            bars_lbl = ui.label("")
            uptime_lbl = ui.label("")
            trades_lbl = ui.label("")
            pnl_lbl = ui.label("")
        position_lbl = ui.label("").classes("text-sm font-bold text-blue-400 mt-1")

        def _refresh_status():
            r = get_runner(cid)
            if not r or r.status == "IDLE":
                status_lbl.text = "Status: IDLE"
                bars_lbl.text = ""
                uptime_lbl.text = ""
                trades_lbl.text = ""
                pnl_lbl.text = ""
                position_lbl.text = ""
                return
            st = r.get_status()
            status_lbl.text = f"Status: {st['status']}"
            bars_lbl.text = f"Bars: {st['bars_received']}"
            uptime_lbl.text = f"Uptime: {st['uptime_seconds']:.0f}s"
            t = st["tracker"]
            if t["total_trades"] > 0:
                trades_lbl.text = (
                    f"Trades: {t['total_trades']} | Win: {t['win_rate']:.1f}%"
                )
                pnl_lbl.text = (
                    f"PnL: {t['total_pnl_bps']:+.1f} bps | "
                    f"MaxDD: {t['max_drawdown_bps']:.1f} bps"
                )
            if t.get("open_position"):
                pos = t["open_position"]
                position_lbl.text = (
                    f"Open: {pos['side'].upper()} @ {pos['entry_price']:,.2f}"
                )
            else:
                position_lbl.text = ""

        ui.timer(2.0, _refresh_status)
        _refresh_status()

        if runner and runner.status == "RUNNING":
            ui.button(
                "Stop Shadow", icon="stop",
                on_click=lambda: _stop_shadow(cid),
            ).props("color=negative dense").classes("mt-2")
        else:
            status_text = ""
            if runner:
                status_text = f"Last status: {runner.status}"
                if runner.error:
                    status_text += f" — {runner.error}"

            ui.label(
                "Bybit testnet credentials detected. Ready for shadow execution."
            ).classes("text-sm text-gray-400 mt-2")

            if status_text:
                ui.label(status_text).classes("text-xs text-gray-500 mt-1")

            ui.button(
                "Start Shadow", icon="play_arrow",
                on_click=lambda s=strat: _start_shadow(s),
            ).props("color=positive dense").classes("mt-2")


async def _start_shadow(strat: dict):
    """Start shadow execution for a strategy."""
    from ui.services.shadow_runner import start_shadow
    from ui.services.compiler_bridge import load_resolved_artifact

    cid = strat["composition_id"]
    compiled_hash = strat["compiled_hash"]

    try:
        resolved = load_resolved_artifact(compiled_hash)
        if not resolved:
            ui.notify("No resolved artifact found. Recompile first.", type="negative")
            return

        await start_shadow(cid, resolved, strat.get("spec", {}))
        ui.notify("Shadow started — receiving live bars", type="positive")
        ui.navigate.to("/shadow")
    except Exception as e:
        ui.notify(f"Failed to start shadow: {e}", type="negative")


def _stop_shadow(composition_id: str):
    """Stop shadow execution."""
    from ui.services.shadow_runner import stop_shadow
    stop_shadow(composition_id)
    ui.notify("Shadow stopped", type="info")
    ui.navigate.to("/shadow")


async def _promote_to_live(composition_id: str, compiled_hash: str):
    """Promote a shadow strategy to live with typed confirmation."""
    with ui.dialog() as dlg, ui.card().classes("w-[500px]"):
        ui.label("PROMOTE TO LIVE?").classes("text-lg font-bold text-red-400")
        ui.label(
            "This approves the strategy for live trading with real capital."
        ).classes("text-sm mt-2")
        ui.label("Type LIVE to confirm:").classes("text-sm text-gray-400 mt-2")
        confirm_input = ui.input(label="Type LIVE").classes("w-full")
        notes_input = ui.input(label="Notes (optional)").classes("w-full mt-2")
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            go_btn = ui.button("Go Live", on_click=lambda: dlg.submit(True)).props(
                "color=positive disable")
            confirm_input.on(
                "update:model-value",
                lambda e: go_btn.props(remove="disable") if e.args == "LIVE"
                else go_btn.props("disable"))
    dlg.open()
    confirmed = await dlg
    if not confirmed:
        return

    try:
        from ui.services.research_services import write_lifecycle_promotion
        from strategy_framework_v1_8_0 import compute_config_hash

        spec = load_composition(composition_id)
        spec_hash = compute_config_hash(spec) if spec else ""
        write_lifecycle_promotion(
            strategy_config_hash=compiled_hash,
            composition_spec_hash=spec_hash,
            dataset_prefix="manual",
            lifecycle_tier="LIVE_APPROVED",
            notes=notes_input.value.strip(),
        )
        ui.notify("Promoted to LIVE_APPROVED", type="positive")
        ui.navigate.to("/live")
    except Exception as e:
        ui.notify(f"Promotion error: {e}", type="negative")


async def _demote_to_research(compiled_hash: str):
    """Demote from SHADOW_VALIDATED back to TRIAGE_PASSED."""
    with ui.dialog() as dlg, ui.card().classes("w-[500px]"):
        ui.label("Demote to Research?").classes("text-lg font-bold")
        ui.label(
            "This will remove the shadow validation status. "
            "The strategy will revert to TRIAGE_PASSED."
        ).classes("text-sm mt-2 text-gray-400")
        ui.label(
            "Triage results and promotion history are preserved."
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
        count = demote_lifecycle(compiled_hash, "SHADOW_VALIDATED")
        if count > 0:
            ui.notify(f"Demoted to Research ({count} artifact(s) disabled)",
                      type="info")
        else:
            ui.notify("No shadow promotion artifacts found", type="warning")
        ui.navigate.to("/shadow")
    except Exception as e:
        ui.notify(f"Demotion error: {e}", type="negative")
