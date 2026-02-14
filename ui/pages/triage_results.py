"""Triage Results — standalone page for viewing triage v2 results for a strategy hash."""

import glob
import json
import os
from typing import Any, Dict, List, Optional

from nicegui import ui

RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "research")

TIER_COLORS = {"S": "purple", "A": "green", "B": "blue", "C": "amber", "F": "red"}
STATUS_COLORS = {"PASS": "green", "WARN": "amber", "FAIL": "red", "INSUFFICIENT_DATA": "grey"}


def triage_results_page(strategy_hash: str):
    """Triage results display page (read-only, no runner)."""
    with ui.column().classes("w-full max-w-6xl mx-auto p-4"):
        with ui.row().classes("items-center gap-4 mb-4"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props(
                "flat dense round")
            ui.label("Triage Results").classes("text-2xl font-bold")

        ui.label(f"Strategy: sha256:{strategy_hash[:24]}...").classes(
            "monospace text-sm text-gray-400")

        # Show existing results
        results_dir = os.path.join(RESEARCH_DIR, "triage_results", strategy_hash)
        if not os.path.isdir(results_dir):
            ui.label("No triage results found for this strategy.").classes(
                "text-gray-400 py-4")
            return

        files = sorted(
            [f for f in os.listdir(results_dir) if f.endswith(".json")],
            reverse=True
        )
        if not files:
            ui.label("No triage results found.").classes("text-gray-400 py-4")
            return

        for filename in files[:10]:
            try:
                with open(os.path.join(results_dir, filename)) as f:
                    data = json.load(f)
                _render_result(data, filename)
            except Exception:
                continue


def _render_result(data: dict, filename: str = ""):
    """Render a triage v2 result from saved JSON."""
    triage_v2 = data.get("triage_v2", {})
    tier = triage_v2.get("tier", "?")
    tier_action = triage_v2.get("tier_action", "")
    tier_color = TIER_COLORS.get(tier, "grey")

    with ui.card().classes("w-full p-4 mb-4"):
        # Header: tier badge + filename
        with ui.row().classes("items-center gap-4"):
            ui.badge(f"Tier {tier}", color=tier_color).classes("text-lg px-3 py-1")
            ui.label(tier_action).classes("text-sm text-gray-300")
            if filename:
                ui.label(filename).classes("monospace text-xs text-gray-500 ml-auto")

        # Summary stats
        with ui.row().classes("gap-4 mt-3 text-sm text-gray-400"):
            tc = data.get("trade_count")
            if tc is not None:
                ui.label(f"Trades: {tc}")
            bc = data.get("bar_count")
            if bc is not None:
                ui.label(f"Bars: {bc:,}")
            ds = data.get("dataset_prefix") or data.get("dataset_filename", "")
            if ds:
                ui.label(f"Dataset: {ds}")
            econ = data.get("runner_economics", {})
            if econ:
                fee = econ.get("fee_rate", "?")
                slip = econ.get("slippage_bps", "?")
                ui.label(f"Fee: {fee} | Slip: {slip}bps")

        # Key metrics row
        metrics = triage_v2.get("metrics", {})
        if metrics:
            _render_key_metrics(metrics)

        # Per-test results
        test_results = triage_v2.get("test_results", [])
        if test_results:
            _render_test_results(test_results)

        # Cost ramp table
        cost_ramp = triage_v2.get("cost_ramp_table", [])
        if cost_ramp:
            _render_cost_ramp(cost_ramp)

        # Tier reasoning
        flags = triage_v2.get("flags", [])
        warnings = triage_v2.get("warnings", [])
        if flags or warnings:
            with ui.expansion("Tier Reasoning", icon="info").classes("w-full mt-2"):
                if flags:
                    for flag in flags:
                        ui.label(f"  {flag}").classes("text-xs text-gray-400 monospace")
                if warnings:
                    for w in warnings:
                        ui.label(f"  {w}").classes("text-xs text-amber-400 monospace")


def _render_key_metrics(metrics: dict):
    """Render key metrics row."""
    metric_defs = [
        ("win_rate", "Win Rate",
         lambda v: f"{v:.1f}%" if isinstance(v, (int, float)) else str(v)),
        ("expectancy_bps", "Expectancy",
         lambda v: f"{v:+.1f} bps"),
        ("profit_factor", "PF",
         lambda v: v if isinstance(v, str) else f"{v:.2f}"),
        ("breakeven_cost_bps", "Breakeven",
         lambda v: f"{v:.0f} bps" if v else "N/A"),
        ("wilson_lb", "Wilson LB",
         lambda v: f"{v:.1f}%" if isinstance(v, (int, float)) else str(v)),
        ("max_drawdown_pct", "Max DD",
         lambda v: f"{v:.1f}%"),
        ("max_consecutive_losses", "Max Consec Loss",
         lambda v: str(int(v))),
    ]

    with ui.row().classes("gap-6 mt-3 flex-wrap"):
        for key, label, fmt in metric_defs:
            val = metrics.get(key)
            if val is not None:
                with ui.column().classes("items-center"):
                    ui.label(label).classes("text-xs text-gray-500")
                    ui.label(fmt(val)).classes("text-sm font-bold")


def _render_test_results(test_results: list):
    """Render per-test results table."""
    columns = [
        {"name": "test", "label": "Test", "field": "test", "align": "left"},
        {"name": "status", "label": "Status", "field": "status", "align": "center"},
        {"name": "detail", "label": "Detail", "field": "detail", "align": "left"},
    ]

    test_names = {
        "test_1": "1: Expectancy + PF",
        "test_2": "2: Profit Concentration",
        "test_3": "3: Temporal Consistency",
        "test_4": "4: Drawdown Survivability",
        "test_5": "5: OOS Holdout",
        "test_6": "6: Cost Ramp",
        "test_7": "7: Tier Classification",
    }

    rows = []
    for t in test_results:
        name = t.get("name", "")
        display_name = test_names.get(name, name)
        status = t.get("status", "?")
        detail = t.get("detail", "")
        rows.append({
            "test": display_name,
            "status": status,
            "detail": detail,
        })

    if rows:
        ui.label("Test Results").classes("text-sm font-bold mt-3 mb-1")
        ui.table(columns=columns, rows=rows, row_key="test").classes(
            "w-full").props("flat bordered dense")


def _render_cost_ramp(cost_ramp: list):
    """Render cost ramp table."""
    columns = [
        {"name": "cost", "label": "RT Cost (bps)", "field": "cost_bps", "align": "center"},
        {"name": "expectancy", "label": "Expectancy (bps)", "field": "expectancy_fmt", "align": "center"},
        {"name": "win_rate", "label": "Win Rate", "field": "win_rate_fmt", "align": "center"},
        {"name": "sharpe", "label": "Sharpe", "field": "sharpe_fmt", "align": "center"},
    ]

    rows = []
    for level in cost_ramp:
        rows.append({
            "cost_bps": level.get("cost_bps", "?"),
            "expectancy_fmt": f"{level.get('expectancy_bps', 0):+.1f}",
            "win_rate_fmt": f"{level.get('win_rate', 0):.1%}" if isinstance(level.get("win_rate"), (int, float)) else "?",
            "sharpe_fmt": f"{level.get('sharpe', 0):.3f}" if isinstance(level.get("sharpe"), (int, float)) else "?",
        })

    if rows:
        ui.label("Cost Ramp").classes("text-sm font-bold mt-3 mb-1")
        ui.table(columns=columns, rows=rows, row_key="cost_bps").classes(
            "w-full").props("flat bordered dense")
