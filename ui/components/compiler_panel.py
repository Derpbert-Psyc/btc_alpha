"""Compiler Feedback Panel — bottom dock showing compile results + triage inline."""

import asyncio
import json
import traceback
from typing import Optional

from nicegui import ui, run

from ui.services.research_services import (
    TriageRunResult,
    list_datasets,
    run_triage_for_composition,
    write_promotion_for_composition,
    load_latest_triage_result,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_STARTING_CAPITAL,
)
from strategy_framework_v1_8_0 import compute_config_hash


def render_compiler_panel(state, container):
    """Render the compiler feedback panel."""
    container.clear()
    result = state.last_compilation

    with container:
        if result is None:
            ui.label("No compilation yet. Press [Compile] to check your spec.").classes(
                "text-gray-400 py-4")
            return

        hash_val = result.get("strategy_config_hash", "")
        report = result.get("lowering_report", {})
        warnings = report.get("warnings", [])
        warmup = report.get("effective_warmup", {})
        resolved_ev = result.get("resolved_artifact", {}).get("engine_version", "")

        # Success banner
        with ui.card().classes("w-full p-4 compiler-success"):
            with ui.row().classes("items-center gap-4"):
                ui.icon("check_circle", color="green").classes("text-2xl")
                ui.label("Compilation Successful").classes("text-lg font-bold text-green-400")

            with ui.row().classes("gap-8 mt-2 flex-wrap"):
                with ui.column():
                    ui.label("Strategy Config Hash").classes("text-xs text-gray-400")
                    ui.label(hash_val).classes("monospace text-sm")
                with ui.column():
                    ui.label("Resolved Engine").classes("text-xs text-gray-400")
                    ev_text = resolved_ev
                    target_ev = state.working_spec.get("target_engine_version", "")
                    if resolved_ev != target_ev:
                        ev_text += f" (target was {target_ev})"
                    ui.label(ev_text).classes("text-sm")
                if warmup:
                    with ui.column():
                        ui.label("Effective Warmup").classes("text-xs text-gray-400")
                        warmup_text = f"{warmup.get('bars', '?')} bars"
                        warmup_days = warmup.get("duration_days", 0)
                        if warmup_days:
                            warmup_text += f" (~{warmup_days:.0f} days)"
                        dom = warmup.get("dominating_instance", "")
                        if dom:
                            warmup_text += f" ({dom})"
                        ui.label(warmup_text).classes("text-sm")

            # Warnings
            if warnings:
                ui.separator()
                with ui.card().classes("w-full p-2 mt-2 compiler-warning"):
                    ui.label(f"{len(warnings)} warning(s)").classes("text-sm text-amber-400 font-bold")
                    for w in warnings:
                        ui.label(f"  - {w}").classes("text-sm text-amber-300")

        # View buttons
        with ui.row().classes("gap-2 mt-2"):
            ui.button("View Lowering Report",
                      on_click=lambda: _show_report(report)).props("outline dense")
            ui.button("View Resolved Artifact",
                      on_click=lambda: _show_artifact(result)).props("outline dense")

        # --- Triage Section ---
        ui.separator().classes("my-4")
        _render_triage_section(state, result)

        # --- Sweep link ---
        ui.separator().classes("my-4")
        with ui.row().classes("gap-4 items-center"):
            ui.button("Parameter Sweep", icon="tune",
                      on_click=lambda: ui.navigate.to(
                          f"/sweep/{state.composition_id}")).props("color=secondary outline")


def _render_triage_section(state, compilation_result):
    """Render inline triage runner with dataset selector and results display."""
    hash_val = compilation_result.get("strategy_config_hash", "")
    resolved = compilation_result.get("resolved_artifact", {})

    try:
        from ui.services.compiler_bridge import save_artifacts
        save_artifacts(compilation_result)
    except Exception:
        pass

    with ui.column().classes("w-full"):
        ui.label("Triage").classes("text-lg font-bold")

        datasets = list_datasets()
        if not datasets:
            ui.label(
                "No dataset found in historic_data/. Run ./scripts/fetch_data.sh to download."
            ).classes("text-red-400 py-2")
            return

        dataset_options = {d["label"]: d["path"] for d in datasets}
        dataset_select = ui.select(
            list(dataset_options.keys()),
            value=list(dataset_options.keys())[0],
            label="Dataset",
        ).classes("w-96")

        results_container = ui.column().classes("w-full mt-2")

        # Show latest saved triage result if available
        latest_triage = load_latest_triage_result(hash_val)
        if latest_triage:
            with results_container:
                _display_saved_triage_result(latest_triage, state, hash_val)

        progress_label = ui.label("").classes("text-sm text-gray-400")
        progress_spinner = ui.spinner(size="sm").classes("hidden")
        triage_running = {"active": False}

        async def do_run_triage():
            if triage_running["active"]:
                ui.notify("Triage already running.", type="warning")
                return

            triage_running["active"] = True
            run_btn.disable()
            progress_spinner.classes(remove="hidden")

            try:
                dataset_path = dataset_options.get(dataset_select.value)
                if not dataset_path:
                    ui.notify("Select a dataset", type="warning")
                    return

                progress_label.text = "Running backtest + triage v2..."
                await asyncio.sleep(0.1)

                triage_run = await run.cpu_bound(
                    run_triage_for_composition,
                    resolved_config=resolved,
                    strategy_config_hash=hash_val,
                    dataset_path=dataset_path,
                    spec=state.working_spec,
                )

                progress_label.text = ""
                results_container.clear()

                with results_container:
                    _display_triage_v2_results(state, triage_run, hash_val)

                if triage_run.zero_trades:
                    ui.notify("Backtest produced 0 trades", type="warning")
                elif triage_run.triage_v2:
                    tier = triage_run.triage_v2.tier
                    ui.notify(
                        f"Triage: Tier {tier} — {triage_run.triage_v2.tier_action}",
                        type="positive" if tier in ("S", "A") else (
                            "warning" if tier in ("B", "C") else "negative"),
                    )

            except Exception as e:
                progress_label.text = ""
                results_container.clear()
                with results_container:
                    with ui.card().classes("w-full p-4 compiler-error"):
                        ui.icon("error", color="red").classes("text-2xl")
                        ui.label("Triage Error").classes("text-lg font-bold text-red-400")
                        ui.label(str(e)).classes("text-sm text-red-300 mt-1")
                        ui.code(traceback.format_exc()).classes("w-full mt-2 text-xs")
                ui.notify(f"Triage error: {e}", type="negative", timeout=10000)
            finally:
                triage_running["active"] = False
                run_btn.enable()
                progress_spinner.classes(add="hidden")

        with ui.row().classes("gap-4 items-center mt-2"):
            run_btn = ui.button("Run Triage", icon="science",
                                on_click=do_run_triage).props("color=primary")
            ui.label("Runs backtest + 7-test triage battery v2").classes(
                "text-xs text-gray-400")


# ---------------------------------------------------------------------------
# Triage v2 results display
# ---------------------------------------------------------------------------

TIER_COLORS = {"S": "purple", "A": "green", "B": "blue", "C": "amber", "F": "red"}
STATUS_COLORS = {"PASS": "green", "WARN": "amber", "FAIL": "red", "INSUFFICIENT DATA": "grey"}


def _display_saved_triage_result(saved_data: dict, state, strategy_hash: str):
    """Display a previously saved triage result from disk JSON."""
    tv2_dict = saved_data.get("triage_v2", {})
    if not tv2_dict:
        return

    tier = tv2_dict.get("tier", "F")
    tier_color = TIER_COLORS.get(tier, "grey")

    with ui.card().classes("w-full p-4"):
        with ui.row().classes("items-center gap-4"):
            _tier_css = {"S": "tier-badge-s", "A": "tier-badge-a", "B": "tier-badge-b"}.get(tier, "")
            ui.badge(f"TIER {tier}", color=tier_color).classes(f"text-xl px-3 py-1 {_tier_css}")
            ui.label(tv2_dict.get("tier_action", "")).classes("text-lg font-bold")
            ui.badge("saved result", color="grey").classes("text-xs")

        m = tv2_dict.get("metrics", {})
        with ui.row().classes("gap-6 mt-3 flex-wrap text-sm"):
            wr = m.get("win_rate", 0)
            ui.label(f"Win Rate: {wr:.1f}%").classes("font-bold")
            ui.label(f"Expectancy: {m.get('expectancy_bps', 0):+.1f} bps/trade")
            ui.label(f"PF: {m.get('profit_factor', '?')}")
            be = m.get("breakeven_cost_bps")
            if be is not None:
                ui.label(f"Breakeven: {be:.0f} bps")
            ui.label(f"Wilson LB: {m.get('wilson_lb', 0):.1f}%")
            ui.label(f"Max DD: {m.get('max_drawdown_pct', 0):.1f}%")

        # Test results summary
        test_results = tv2_dict.get("test_results", [])
        if test_results:
            with ui.row().classes("gap-2 mt-2 flex-wrap"):
                for tr in test_results:
                    status = tr.get("status", "?")
                    color = STATUS_COLORS.get(status, "grey")
                    ui.badge(f"{tr.get('name', '?')}: {status}", color=color).classes("text-xs")

        with ui.row().classes("gap-8 mt-2 flex-wrap text-sm text-gray-400"):
            ui.label(f"Dataset: {saved_data.get('dataset_filename', saved_data.get('dataset_prefix', ''))}")
            ui.label(f"Trades: {saved_data.get('trade_count', 0)}")
            ui.label(f"Bars: {saved_data.get('bar_count', 0):,}")
            ui.label(f"Saved: {saved_data.get('timestamp', '')}")

    # Lifecycle promotion buttons (Shadow / Live)
    _render_lifecycle_promotion_buttons(state, strategy_hash)


def _display_triage_v2_results(state, triage_run: TriageRunResult, strategy_hash: str):
    """Display triage v2 results inline."""
    if triage_run.zero_trades:
        with ui.card().classes("w-full p-4 compiler-warning"):
            ui.icon("warning", color="amber").classes("text-2xl")
            ui.label("Zero Trades").classes("text-lg font-bold text-amber-400")
            ui.label(triage_run.zero_trade_message).classes("text-sm text-amber-300 mt-1")
        return

    tv2 = triage_run.triage_v2
    if tv2 is None:
        return

    tier = tv2.tier
    tier_color = TIER_COLORS.get(tier, "grey")

    # Tier banner
    with ui.card().classes("w-full p-4"):
        with ui.row().classes("items-center gap-4"):
            _tier_css = {"S": "tier-badge-s", "A": "tier-badge-a", "B": "tier-badge-b"}.get(tier, "")
            ui.badge(f"TIER {tier}", color=tier_color).classes(f"text-2xl px-4 py-2 {_tier_css}")
            ui.label(tv2.tier_action).classes("text-lg font-bold")

        # Key metrics row
        with ui.row().classes("gap-6 mt-3 flex-wrap text-sm"):
            m = tv2.metrics
            ui.label(f"Win Rate: {m.get('win_rate', 0):.1f}%").classes("font-bold")
            ui.label(f"Expectancy: {m.get('expectancy_bps', 0):+.1f} bps/trade")
            ui.label(f"PF: {m.get('profit_factor', '?')}")
            be = m.get("breakeven_cost_bps")
            if be is not None:
                ui.label(f"Breakeven: {be:.0f} bps")
            else:
                ui.label("Breakeven: >200 bps")
            ui.label(f"Wilson LB: {m.get('wilson_lb', 0):.1f}%")
            ui.label(f"Max DD: {m.get('max_drawdown_pct', 0):.1f}%")
            ui.label(f"Max Consec Loss: {m.get('max_consecutive_losses', 0)}")

        # Flags and warnings
        if tv2.flags:
            ui.separator()
            for flag in tv2.flags:
                ui.label(f"  {flag}").classes("text-sm text-amber-400")

    # Per-test results table
    _render_test_results_table(tv2)

    # Cost ramp table
    if tv2.cost_ramp_table:
        _render_cost_ramp_table(tv2.cost_ramp_table)

    # Tier reasoning (collapsible)
    with ui.expansion("Tier Reasoning", icon="psychology").classes("w-full mt-2"):
        ui.code(tv2.tier_reasoning).classes("w-full text-sm")

    # Summary stats
    with ui.row().classes("gap-8 mt-2 flex-wrap text-sm text-gray-400"):
        ui.label(f"Dataset: {triage_run.dataset_filename}")
        ui.label(f"Trades: {triage_run.trade_count}")
        ui.label(f"Bars: {triage_run.bar_count:,}")
        ui.label(f"Runtime: {triage_run.runtime_seconds:.1f}s")

    if triage_run.saved_path:
        ui.label(f"Saved: {triage_run.saved_path}").classes(
            "monospace text-xs text-gray-500 mt-1")

    # Trade log
    if triage_run.trade_details:
        _render_trade_log(triage_run.trade_details)

    # Promotion button (S, A, B tiers)
    if tier in ("S", "A", "B"):
        _render_promotion_button(state, triage_run, strategy_hash)

    # Lifecycle promotion buttons (Shadow / Live)
    _render_lifecycle_promotion_buttons(state, strategy_hash)


def _render_test_results_table(tv2):
    """Render the 7-test results as a table with status badges."""
    columns = [
        {"name": "test", "label": "Test", "field": "test", "align": "left"},
        {"name": "status", "label": "Status", "field": "status", "align": "center"},
        {"name": "detail", "label": "Detail", "field": "detail", "align": "left"},
    ]

    rows = []
    for tr in tv2.test_results:
        rows.append({
            "test": tr.name,
            "status": tr.status,
            "detail": tr.detail[:120],
        })

    if rows:
        table = ui.table(columns=columns, rows=rows, row_key="test").classes(
            "w-full mt-2").props("flat bordered dense")

        table.add_slot("body-cell-status", """
            <q-td :props="props">
                <q-badge :color="props.row.status === 'PASS' ? 'green' :
                                 props.row.status === 'WARN' ? 'amber' :
                                 props.row.status === 'FAIL' ? 'red' : 'grey'"
                         :label="props.row.status" />
            </q-td>
        """)


def _render_cost_ramp_table(cost_ramp_data):
    """Render cost ramp table (4 rows)."""
    with ui.expansion("Cost Ramp Details", icon="trending_down").classes("w-full mt-2"):
        columns = [
            {"name": "cost", "label": "Cost (bps RT)", "field": "cost_bps", "align": "center"},
            {"name": "wr", "label": "Win Rate", "field": "win_rate_str", "align": "center"},
            {"name": "sharpe", "label": "Sharpe", "field": "sharpe_str", "align": "center"},
            {"name": "pnl", "label": "Total PnL (bps)", "field": "pnl_str", "align": "right"},
            {"name": "pf", "label": "Profit Factor", "field": "profit_factor", "align": "center"},
        ]

        rows = []
        for d in cost_ramp_data:
            rows.append({
                "cost_bps": d["cost_bps"],
                "win_rate_str": f"{d['win_rate']:.1f}%",
                "sharpe_str": f"{d['sharpe']:.4f}",
                "pnl_str": f"{d['total_pnl_bps']:+,.0f}",
                "profit_factor": str(d["profit_factor"]),
            })

        ui.table(columns=columns, rows=rows, row_key="cost_bps").classes(
            "w-full").props("flat bordered dense")


def _render_trade_log(trade_details: list):
    """Render collapsible trade log table."""
    with ui.expansion("Trade Log", icon="receipt_long").classes("w-full mt-2"):
        columns = [
            {"name": "num", "label": "#", "field": "num", "align": "center"},
            {"name": "direction", "label": "Direction", "field": "direction", "align": "center"},
            {"name": "entry_time", "label": "Entry Time", "field": "entry_time", "align": "left"},
            {"name": "exit_time", "label": "Exit Time", "field": "exit_time", "align": "left"},
            {"name": "entry_price", "label": "Entry Price", "field": "entry_price", "align": "right"},
            {"name": "exit_price", "label": "Exit Price", "field": "exit_price", "align": "right"},
            {"name": "pnl_usd", "label": "PnL ($)", "field": "pnl_usd", "align": "right"},
            {"name": "pnl_bps", "label": "PnL (bps)", "field": "pnl_bps", "align": "right"},
            {"name": "bars_held", "label": "Bars Held", "field": "bars_held", "align": "right"},
        ]

        show_all = {"value": False}
        display_limit = 50

        def _build_rows(show_all_flag):
            data = trade_details if show_all_flag else trade_details[:display_limit]
            return [{**td, "num": i + 1} for i, td in enumerate(data)]

        rows = _build_rows(False)
        table = ui.table(columns=columns, rows=rows, row_key="num").classes(
            "w-full").props("flat bordered dense")

        if len(trade_details) > display_limit:
            def toggle_show_all():
                show_all["value"] = not show_all["value"]
                table.rows = _build_rows(show_all["value"])
                table.update()
                show_btn.text = (
                    f"Show First {display_limit}" if show_all["value"]
                    else f"Show All ({len(trade_details)} trades)")
            show_btn = ui.button(
                f"Show All ({len(trade_details)} trades)",
                on_click=toggle_show_all).props("flat dense")


def _render_promotion_button(state, triage_run: TriageRunResult, strategy_hash: str):
    """Render [Write Promotion Artifact] button for promotable tiers."""
    tv2 = triage_run.triage_v2

    with ui.row().classes("gap-4 items-center mt-4"):
        def do_write_promotion():
            try:
                spec_hash = compute_config_hash(state.working_spec)

                triage_summary = tv2.to_dict()

                filepath = write_promotion_for_composition(
                    strategy_config_hash=strategy_hash,
                    composition_spec_hash=spec_hash,
                    dataset_prefix=triage_run.dataset_prefix,
                    runner_economics=triage_run.runner_economics,
                    triage_result_summary=triage_summary,
                    tier=tv2.tier,
                )

                from ui.services.composition_store import update_compiled_hash
                update_compiled_hash(state.composition_id, strategy_hash)

                ui.notify(f"Promotion artifact written: {filepath}", type="positive")
            except Exception as e:
                ui.notify(f"Promotion write error: {e}", type="negative", timeout=10000)

        ui.button("Write Promotion Artifact", icon="verified",
                  on_click=do_write_promotion).props("color=positive")
        ui.label(f"Tier {tv2.tier}: {tv2.tier_action}").classes("text-xs text-gray-400")


def _render_lifecycle_promotion_buttons(state, strategy_hash: str):
    """Render promote-to-Shadow and promote-to-Live buttons based on lifecycle state."""
    from ui.services.promotion_reader import derive_lifecycle_state
    from ui.services.composition_store import load_index

    index = load_index()
    entry = index.get("compositions", {}).get(state.composition_id, {})
    compiled_hash = entry.get("latest_compiled_hash") or strategy_hash
    if not compiled_hash:
        return

    lifecycle, _, _ = derive_lifecycle_state(state.composition_id, compiled_hash)

    if lifecycle == "TRIAGE_PASSED":
        with ui.row().classes("gap-4 items-center mt-3"):
            async def promote_to_shadow():
                with ui.dialog() as dlg, ui.card().classes("w-[500px]"):
                    ui.label("Promote to Shadow?").classes("text-lg font-bold")
                    ui.label(
                        "This marks the strategy for shadow trading validation. "
                        "The VPS operations console will pick it up from the artifact store."
                    ).classes("text-sm mt-2 text-gray-400")
                    notes_input = ui.input(label="Notes (optional)").classes("w-full")
                    with ui.row().classes("w-full justify-end gap-2 mt-4"):
                        ui.button("Cancel", on_click=dlg.close).props("flat")
                        ui.button("Promote", on_click=lambda: dlg.submit(True)).props(
                            "color=purple")
                dlg.open()
                confirmed = await dlg
                if not confirmed:
                    return
                try:
                    from ui.services.research_services import write_lifecycle_promotion
                    spec_hash = compute_config_hash(state.working_spec)
                    filepath = write_lifecycle_promotion(
                        strategy_config_hash=compiled_hash,
                        composition_spec_hash=spec_hash,
                        dataset_prefix="manual",
                        lifecycle_tier="SHADOW_VALIDATED",
                        notes=notes_input.value.strip(),
                    )
                    ui.notify(f"Promoted to SHADOW_VALIDATED", type="positive")
                except Exception as e:
                    ui.notify(f"Promotion error: {e}", type="negative")

            ui.button("Promote to Shadow", icon="visibility",
                      on_click=promote_to_shadow).props("color=purple outline")
            ui.label("Next step: shadow trading validation").classes("text-xs text-gray-400")

    elif lifecycle == "SHADOW_VALIDATED":
        with ui.row().classes("gap-4 items-center mt-3"):
            async def promote_to_live():
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
                        confirm_input.on("update:model-value",
                            lambda e: go_btn.props(remove="disable") if e.args == "LIVE"
                            else go_btn.props("disable"))
                dlg.open()
                confirmed = await dlg
                if not confirmed:
                    return
                try:
                    from ui.services.research_services import write_lifecycle_promotion
                    spec_hash = compute_config_hash(state.working_spec)
                    filepath = write_lifecycle_promotion(
                        strategy_config_hash=compiled_hash,
                        composition_spec_hash=spec_hash,
                        dataset_prefix="manual",
                        lifecycle_tier="LIVE_APPROVED",
                        notes=notes_input.value.strip(),
                    )
                    ui.notify(f"Promoted to LIVE_APPROVED", type="positive")
                except Exception as e:
                    ui.notify(f"Promotion error: {e}", type="negative")

            ui.button("Promote to Live", icon="rocket_launch",
                      on_click=promote_to_live).props("color=positive")
            ui.label("Requires typed confirmation").classes("text-xs text-gray-400")


async def _show_report(report):
    with ui.dialog() as dialog, ui.card().classes("w-[800px] max-h-[80vh] overflow-auto"):
        ui.label("Lowering Report").classes("text-lg font-bold")
        ui.code(json.dumps(report, indent=2), language="json").classes("w-full")
        ui.button("Close", on_click=dialog.close)
    dialog.open()


async def _show_artifact(result):
    artifact = result.get("resolved_artifact", {})
    with ui.dialog() as dialog, ui.card().classes("w-[800px] max-h-[80vh] overflow-auto"):
        ui.label("Resolved Artifact").classes("text-lg font-bold")
        ui.code(json.dumps(artifact, indent=2), language="json").classes("w-full")
        ui.button("Close", on_click=dialog.close)
    dialog.open()
