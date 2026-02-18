"""Shadow Workspace -- daemon instance monitoring + lifecycle management."""

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
from ui.services.shadow_runner import (
    start_shadow,
    stop_shadow,
    read_shadow_status,
)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


# ---------------------------------------------------------------------------
# Predefined shadow daemon instance configs
# ---------------------------------------------------------------------------

SHADOW_INSTANCES = {
    "bybit-cx": {
        "instance_id": "bybit-cx",
        "exchange": "bybit",
        "symbol": "BTCUSDT",
        "category": "linear",
        "strategy": "macd_cx",
        "base_interval_seconds": 1,
        "timeframes": {
            "5s": 5, "15s": 15, "30s": 30, "1m": 60,
            "12m": 720, "24m": 1440, "72m": 4320,
        },
        "roles": {
            "macro": ["72m", "24m", "12m"],
            "intra": ["1m", "30s", "15s"],
            "entry": "5s",
            "exit": "24m",
        },
        "long_only": False,
        "macd_fast": 12,
        "macd_slow": 26,
        "round_trip_bps": 25.0,
        "paper_qty": 0.001,
        "stabilization_seconds": 900,
        "stop_loss_long_bps": 0,
        "stop_loss_short_bps": 0,
    },
    "bybit-big": {
        "instance_id": "bybit-big",
        "exchange": "bybit",
        "symbol": "BTCUSDT",
        "category": "linear",
        "strategy": "macd_big",
        "base_interval_seconds": 60,
        "timeframes": {
            "5m": 300, "15m": 900, "30m": 1800, "1h": 3600,
            "12h": 43200, "1d": 86400, "3d": 259200,
        },
        "roles": {
            "macro": ["3d", "1d", "12h"],
            "intra": ["1h", "30m", "15m"],
            "entry": "5m",
            "exit": "1d",
        },
        "long_only": False,
        "macd_fast": 12,
        "macd_slow": 26,
        "round_trip_bps": 25.0,
        "paper_qty": 0.001,
        "stabilization_seconds": 0,
        "stop_loss_long_bps": 0,
        "stop_loss_short_bps": 0,
    },
    "binance-big": {
        "instance_id": "binance-big",
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "category": "spot",
        "strategy": "macd_big",
        "base_interval_seconds": 60,
        "timeframes": {
            "5m": 300, "15m": 900, "30m": 1800, "1h": 3600,
            "12h": 43200, "1d": 86400, "3d": 259200,
        },
        "roles": {
            "macro": ["3d", "1d", "12h"],
            "intra": ["1h", "30m", "15m"],
            "entry": "5m",
            "exit": "1d",
        },
        "long_only": True,
        "macd_fast": 12,
        "macd_slow": 26,
        "round_trip_bps": 20.0,
        "paper_qty": 0.001,
        "stabilization_seconds": 0,
        "stop_loss_long_bps": 0,
        "stop_loss_short_bps": 0,
    },
}

# Badge colors for exchanges and strategies
_EXCHANGE_COLORS = {"bybit": "blue", "binance": "amber"}
_STRATEGY_LABELS = {"macd_cx": "MACD CX", "macd_big": "MACD Big"}


# ---------------------------------------------------------------------------
# API config check (imported by live_workspace.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shadow strategy listing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Status color helpers
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "RUNNING": "green",
    "WARMING_UP": "cyan",
    "CONNECTING": "amber",
    "STABILIZING": "purple",
    "STOPPED": "grey",
    "ERROR": "red",
    "STALE": "orange",
}


def _status_color(status: str) -> str:
    return _STATUS_COLORS.get(status, "grey")


# ---------------------------------------------------------------------------
# Daemon Instance Cards
# ---------------------------------------------------------------------------

def _render_daemon_instances_section():
    """Render shadow daemon instance cards."""
    with ui.row().classes("w-full items-center gap-2 mb-3"):
        ui.icon("dns", color="purple").classes("text-xl")
        ui.label("Shadow Daemon Instances").classes("text-lg font-bold")

    with ui.row().classes("w-full gap-4 flex-wrap"):
        for instance_id, config in SHADOW_INSTANCES.items():
            _render_instance_card(instance_id, config)


def _slope_sign_text(sign) -> str:
    """Convert slope sign to display text with color hint."""
    if sign == 1:
        return "+1"
    elif sign == -1:
        return "-1"
    elif sign == 0:
        return "0"
    return "--"


def _slope_sign_color(sign) -> str:
    """CSS color class for slope sign."""
    if sign == 1:
        return "text-green-400"
    elif sign == -1:
        return "text-red-400"
    elif sign == 0:
        return "text-yellow-400"
    return "text-gray-500"


def _render_instance_card(instance_id: str, config: dict):
    """Render a single daemon instance card with 3-section live-updating status."""
    exchange = config["exchange"]
    strategy = config.get("strategy", "")
    long_only = config.get("long_only", False)

    with ui.card().classes("w-[480px] p-4"):
        # Header row: instance ID + badges
        with ui.row().classes("w-full items-center gap-2 mb-2"):
            status_icon = ui.icon("circle", color="grey").classes("text-xs")
            ui.label(instance_id).classes("text-lg font-bold")

        with ui.row().classes("gap-2 mb-2"):
            ex_color = _EXCHANGE_COLORS.get(exchange, "grey")
            ui.badge(exchange.capitalize(), color=ex_color)
            strat_label = _STRATEGY_LABELS.get(strategy, strategy)
            ui.badge(strat_label, color="purple")
            if long_only:
                ui.badge("Long Only", color="teal")

        # Section 1: Status & Position
        sec1_container = ui.column().classes("w-full gap-0")

        # Section 2: Paper Trading Stats
        sec2_container = ui.column().classes("w-full gap-0 mt-2")

        # Section 3: Timeframe Indicators
        sec3_container = ui.column().classes("w-full gap-0 mt-2")

        # Button container
        btn_container = ui.row().classes("mt-2 gap-2")

        def _refresh():
            st = read_shadow_status(instance_id)

            # --- Section 1: Status & Position ---
            sec1_container.clear()
            with sec1_container:
                if not st:
                    status_icon.props("color=grey")
                    ui.label("NOT STARTED").classes("text-sm font-bold text-gray-400")
                    btn_container.clear()
                    with btn_container:
                        ui.button(
                            "Start Shadow", icon="play_arrow",
                            on_click=lambda iid=instance_id: _start_instance(iid),
                        ).props("color=positive dense")
                    sec2_container.clear()
                    sec3_container.clear()
                    return

                s = st.get("status", "UNKNOWN")
                color = _status_color(s)
                status_icon.props(f"color={color}")

                with ui.row().classes("w-full items-center gap-3"):
                    ui.label(s).classes(f"text-sm font-bold text-{color}-400")
                    ui.label(f"Uptime: {st.get('uptime_str', '--')}").classes(
                        "text-xs text-gray-400")
                    bars_text = f"Bars: {st.get('bars_received', 0):,}"
                    lbp = st.get("last_bar_price")
                    if lbp and s in ("RUNNING", "STABILIZING"):
                        bars_text += f" | ${lbp:,.2f}"
                    ui.label(bars_text).classes("text-xs text-gray-400")

                # Warmup progress
                if s == "WARMING_UP":
                    wp = st.get("warmup_progress", "")
                    if wp:
                        ui.label(wp).classes("text-xs text-cyan-400")

                # Stabilization countdown
                stab_rem = st.get("stabilization_remaining_s")
                if s == "STABILIZING" and stab_rem is not None:
                    ui.label(f"Stabilizing: {stab_rem}s remaining").classes(
                        "text-xs text-purple-400")

                # Gap flag
                if st.get("gap_flag"):
                    ui.label("GAP DETECTED - signals suppressed").classes(
                        "text-xs text-orange-400 font-bold")

                # Position
                t = st.get("tracker", {})
                pos = t.get("position", 0)
                entry = t.get("entry_fill")
                if pos == 1 and entry:
                    ui.label(f"LONG @ {entry:,.2f}").classes(
                        "text-sm font-bold text-green-400")
                elif pos == -1 and entry:
                    ui.label(f"SHORT @ {entry:,.2f}").classes(
                        "text-sm font-bold text-red-400")
                else:
                    ui.label("FLAT").classes("text-xs text-gray-500")

            # --- Section 2: Paper Trading Stats ---
            sec2_container.clear()
            with sec2_container:
                t = st.get("tracker", {})
                total = t.get("total_trades", 0)
                if total > 0:
                    ui.separator().classes("my-1")
                    ui.label("Paper Trading").classes(
                        "text-xs font-bold text-gray-400 mb-1")
                    # Stats grid: 2 columns
                    with ui.grid(columns=2).classes("w-full gap-x-4 gap-y-0"):
                        # Row 1
                        ui.label(f"Trades: {total}").classes("text-xs text-gray-300")
                        ui.label(
                            f"Win Rate: {t.get('win_rate', 0):.1f}%"
                        ).classes("text-xs text-gray-300")
                        # Row 2
                        pnl = t.get("total_pnl_bps", 0)
                        pnl_cls = "text-green-400" if pnl >= 0 else "text-red-400"
                        ui.label(f"PnL: {pnl:+.1f} bps").classes(
                            f"text-xs font-bold {pnl_cls}")
                        ui.label(
                            f"MaxDD: {t.get('max_drawdown_bps', 0):.1f} bps"
                        ).classes("text-xs text-gray-300")
                        # Row 3
                        avg = t.get("avg_pnl_per_trade_bps", 0)
                        avg_cls = "text-green-400" if avg >= 0 else "text-red-400"
                        ui.label(f"Avg/Trade: {avg:+.1f} bps").classes(
                            f"text-xs {avg_cls}")
                        ui.label(
                            f"Friction: {t.get('round_trip_bps', 0):.0f} bps RT"
                        ).classes("text-xs text-gray-500")
                        # Row 4
                        el = t.get("entries_long", 0)
                        es = t.get("entries_short", 0)
                        ui.label(f"Entries: {el}L / {es}S").classes(
                            "text-xs text-gray-300")
                        sl = t.get("stop_loss_count", 0)
                        exits_text = f"Exits: {t.get('exits_total', 0)}"
                        if sl > 0:
                            exits_text += f" (SL: {sl})"
                        ui.label(exits_text).classes("text-xs text-gray-300")

            # --- Section 3: Timeframe Indicators ---
            sec3_container.clear()
            with sec3_container:
                strat = st.get("strategy", {})
                tf_detail = strat.get("tf_detail", {}) if isinstance(strat, dict) else {}
                if tf_detail:
                    ui.separator().classes("my-1")
                    ui.label("Timeframe Indicators").classes(
                        "text-xs font-bold text-gray-400 mb-1")

                    # Determine role for each TF
                    roles = config.get("roles", {})
                    macro_set = set(roles.get("macro", []))
                    intra_set = set(roles.get("intra", []))
                    entry_tf = roles.get("entry", "")
                    exit_tf = roles.get("exit", "")

                    # Table header
                    with ui.row().classes("w-full gap-0"):
                        ui.label("TF").classes(
                            "text-xs font-bold text-gray-500 w-16")
                        ui.label("Role").classes(
                            "text-xs font-bold text-gray-500 w-16")
                        ui.label("Bars").classes(
                            "text-xs font-bold text-gray-500 w-20")
                        ui.label("Ready").classes(
                            "text-xs font-bold text-gray-500 w-14")
                        ui.label("Slope").classes(
                            "text-xs font-bold text-gray-500 w-12")

                    # Rows ordered by config timeframes
                    for tf_label in config.get("timeframes", {}).keys():
                        detail = tf_detail.get(tf_label, {})
                        bars_done = detail.get("bars_processed", 0)
                        required = detail.get("required", 27)
                        ready = detail.get("ready", False)
                        slope = detail.get("slope_sign")

                        # Determine role label
                        if tf_label in macro_set:
                            role = "macro"
                        elif tf_label in intra_set:
                            role = "intra"
                        elif tf_label == entry_tf:
                            role = "entry"
                        elif tf_label == exit_tf:
                            role = "exit"
                        else:
                            role = ""

                        with ui.row().classes("w-full gap-0 items-center"):
                            ui.label(tf_label).classes(
                                "text-xs text-gray-300 w-16 font-mono")
                            ui.label(role).classes(
                                "text-xs text-gray-500 w-16")
                            bar_text = f"{bars_done}/{required}"
                            bar_cls = "text-gray-300" if bars_done >= required else "text-yellow-400"
                            ui.label(bar_text).classes(
                                f"text-xs {bar_cls} w-20 font-mono")
                            ready_icon = "check_circle" if ready else "pending"
                            ready_color = "green" if ready else "grey"
                            ui.icon(ready_icon, color=ready_color).classes(
                                "text-xs w-14")
                            sign_text = _slope_sign_text(slope)
                            sign_cls = _slope_sign_color(slope)
                            ui.label(sign_text).classes(
                                f"text-xs font-bold {sign_cls} w-12 font-mono")

            # --- Buttons ---
            is_active = s in (
                "RUNNING", "WARMING_UP", "CONNECTING",
                "RECONNECTING", "STABILIZING", "STARTING",
            )
            btn_container.clear()
            with btn_container:
                if is_active:
                    ui.button(
                        "Stop Shadow", icon="stop",
                        on_click=lambda iid=instance_id: _stop_instance(iid),
                    ).props("color=negative dense")
                else:
                    ui.button(
                        "Start Shadow", icon="play_arrow",
                        on_click=lambda iid=instance_id: _start_instance(iid),
                    ).props("color=positive dense")

        ui.timer(2.0, _refresh)
        _refresh()


def _start_instance(instance_id: str):
    """Start a shadow daemon instance."""
    config = SHADOW_INSTANCES.get(instance_id)
    if not config:
        ui.notify(f"Unknown instance: {instance_id}", type="negative")
        return
    try:
        ok = start_shadow(instance_id, config)
        if ok:
            ui.notify(f"Starting {instance_id}...", type="positive")
        else:
            ui.notify(f"Failed to start {instance_id}", type="negative")
    except Exception as e:
        ui.notify(f"Error: {e}", type="negative")


def _stop_instance(instance_id: str):
    """Stop a shadow daemon instance."""
    try:
        stop_shadow(instance_id)
        ui.notify(f"Stop command sent to {instance_id}", type="info")
    except Exception as e:
        ui.notify(f"Error: {e}", type="negative")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def shadow_workspace_page():
    """Render the Shadow monitoring workspace."""
    with ui.column().classes("w-full max-w-7xl mx-auto p-4"):
        render_workspace_nav("shadow")

        # Header
        with ui.row().classes("w-full items-center gap-4 mb-4"):
            ui.icon("visibility", color="purple").classes("text-3xl")
            ui.label("SHADOW MONITOR").classes("text-2xl font-bold")

        # Section 1: Daemon Instances
        _render_daemon_instances_section()

        ui.separator().classes("my-6")

        # Section 2: Shadow-Validated Strategies (lifecycle)
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


# ---------------------------------------------------------------------------
# Strategy lifecycle cards (unchanged from v1 except execution panel removed)
# ---------------------------------------------------------------------------

def _render_shadow_card(strat: dict):
    """Render a single shadow strategy card (lifecycle only)."""
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


# ---------------------------------------------------------------------------
# Lifecycle promotion/demotion (unchanged from v1)
# ---------------------------------------------------------------------------

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
