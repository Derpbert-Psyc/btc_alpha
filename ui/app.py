"""BTC Alpha Research Panel — NiceGUI entry point."""

import sys
import os

# Ensure project root is on path for frozen module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nicegui import ui, app

from ui.pages.strategy_list import strategy_list_page
from ui.pages.composition_editor import composition_editor_page
from ui.pages.preset_library import preset_library_page
from ui.pages.triage_results import triage_results_page
from ui.pages.parameter_sweep import parameter_sweep_page

# Dark theme CSS
DARK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
:root {
    --bg-primary: #0a0f1a;
    --bg-panel: #0d1117;
    --bg-surface: #131a27;
    --border: #1e293b;
    --blue: #3b82f6;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
    --purple: #8b5cf6;
    --text: #e2e8f0;
    --text-dim: #94a3b8;
}
body {
    background-color: var(--bg-primary) !important;
    color: var(--text) !important;
    font-family: 'Inter', system-ui, sans-serif;
}
.nicegui-content {
    background-color: var(--bg-primary) !important;
}
/* Cards and panels */
.q-card {
    background-color: var(--bg-panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
.q-dialog .q-card { border-radius: 16px !important; }
/* Tables */
.q-table {
    background-color: var(--bg-panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden;
}
.q-table thead th {
    color: var(--text-dim) !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    border-bottom: 1px solid var(--border) !important;
}
.q-table tbody tr:nth-child(even) {
    background-color: var(--bg-surface) !important;
}
.q-table tbody tr:hover {
    background-color: rgba(59, 130, 246, 0.08) !important;
}
.q-table__bottom {
    color: var(--text-dim) !important;
}
/* Form inputs */
.q-field__control {
    background-color: var(--border) !important;
    color: var(--text) !important;
}
.q-field__label { color: var(--text-dim) !important; }
.q-field--focused .q-field__label { color: var(--blue) !important; }
.q-field__bottom { color: var(--text-dim) !important; }
/* Dropdown menus */
.q-menu { background-color: var(--bg-surface); border: 1px solid var(--border); border-radius: 10px !important; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
.q-item { color: var(--text); }
.q-item--active { background-color: rgba(59, 130, 246, 0.12) !important; }
.q-item:hover { background-color: var(--bg-surface) !important; }
/* Tabs */
.q-tabs { border-bottom: 1px solid var(--border) !important; }
.q-tab { color: var(--text-dim) !important; text-transform: none !important; }
.q-tab--active { color: var(--blue) !important; }
.q-tab-panel { padding: 16px 0 !important; background-color: transparent !important; }
/* Buttons */
.q-btn[class*="bg-primary"] { background: var(--blue) !important; }
.q-btn[class*="bg-positive"] { background: var(--green) !important; }
.q-btn[class*="bg-warning"] { background: var(--amber) !important; color: #0a0f1a !important; }
.q-btn[class*="bg-negative"] { background: var(--red) !important; }
.q-btn[class*="bg-secondary"] { background: var(--purple) !important; }
.q-btn--disabled { opacity: 0.35 !important; }
/* Toggle buttons */
.q-btn-toggle { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
/* Badges */
.q-badge { font-family: 'JetBrains Mono', monospace; font-size: 11px; }
/* Expansion panels */
.q-expansion-item { border: 1px solid var(--border); border-radius: 10px !important; overflow: hidden; margin-top: 8px; }
.q-expansion-item__container { background: var(--bg-panel) !important; }
/* Separators */
.q-separator { background: var(--border) !important; }
/* Monospace for data values */
.monospace { font-family: 'JetBrains Mono', 'Fira Code', monospace; }
/* Lifecycle badges */
.badge-draft { background: #334155; color: #94a3b8; }
.badge-compiled { background: #1e3a5f; color: #60a5fa; }
.badge-triage-passed { background: #064e3b; color: #6ee7b7; }
.badge-corrupted { background: #7f1d1d; color: #fca5a5; }
.badge-live { background: #14532d; color: #86efac; }
/* Compiler feedback panels */
.compiler-success { border-left: 4px solid var(--green); background: #064e3b22; }
.compiler-error { border-left: 4px solid var(--red); background: #7f1d1d22; }
.compiler-warning { border-left: 4px solid var(--amber); background: #78350f22; }
/* Unsaved indicator */
.unsaved-dot::after { content: '●'; color: var(--amber); margin-left: 4px; font-size: 10px; }
/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
/* C1: Rounded corners */
.q-field__control { border-radius: 8px !important; }
.q-btn { border-radius: 8px !important; }
.q-badge { border-radius: 6px !important; }
.q-tab { border-radius: 8px 8px 0 0 !important; }
.q-chip { border-radius: 8px !important; }
.q-tooltip { border-radius: 8px !important; }
/* C2: Direction colour system */
.direction-long { border-left: 3px solid var(--green) !important; background: linear-gradient(90deg, rgba(16,185,129,0.06), transparent) !important; }
.direction-short { border-left: 3px solid var(--red) !important; background: linear-gradient(90deg, rgba(239,68,68,0.06), transparent) !important; }
/* C3: Accent panels */
.accent-blue { border-left: 3px solid var(--blue) !important; background: linear-gradient(90deg, rgba(59,130,246,0.06), transparent) !important; }
.accent-green { border-left: 3px solid var(--green) !important; background: linear-gradient(90deg, rgba(16,185,129,0.06), transparent) !important; }
.accent-amber { border-left: 3px solid var(--amber) !important; background: linear-gradient(90deg, rgba(245,158,11,0.06), transparent) !important; }
.accent-red { border-left: 3px solid var(--red) !important; background: linear-gradient(90deg, rgba(239,68,68,0.06), transparent) !important; }
.accent-purple { border-left: 3px solid var(--purple) !important; background: linear-gradient(90deg, rgba(139,92,246,0.06), transparent) !important; }
.accent-cyan { border-left: 3px solid #06b6d4 !important; background: linear-gradient(90deg, rgba(6,182,212,0.06), transparent) !important; }
/* C4: Indicator group colours */
.group-trend { color: var(--blue) !important; }
.group-momentum { color: var(--green) !important; }
.group-volatility { color: var(--amber) !important; }
.group-volume { color: #06b6d4 !important; }
.group-sr { color: var(--purple) !important; }
.group-price { color: var(--red) !important; }
/* C5: Tab colour coding */
.tab-indicators.q-tab--active { color: var(--blue) !important; }
.tab-entry.q-tab--active { color: var(--green) !important; }
.tab-exit.q-tab--active { color: var(--amber) !important; }
.tab-gates.q-tab--active { color: var(--purple) !important; }
.tab-execution.q-tab--active { color: #06b6d4 !important; }
.tab-metadata.q-tab--active { color: #94a3b8 !important; }
/* C6: Strategy list polish */
.q-table tbody tr { cursor: pointer; transition: background-color 0.15s; }
/* C8: Triage tier badges */
.tier-badge-s { background: linear-gradient(135deg, #7c3aed, #a855f7) !important; color: white !important; border: 1px solid #a855f7; font-weight: 700; }
.tier-badge-a { background: linear-gradient(135deg, #059669, #10b981) !important; color: white !important; border: 1px solid #10b981; font-weight: 700; }
.tier-badge-b { background: linear-gradient(135deg, #2563eb, #3b82f6) !important; color: white !important; border: 1px solid #3b82f6; font-weight: 700; }
"""


@ui.page("/")
def index():
    ui.add_head_html(f"<style>{DARK_CSS}</style>")
    strategy_list_page()


@ui.page("/editor/{composition_id}")
def editor(composition_id: str):
    ui.add_head_html(f"<style>{DARK_CSS}</style>")
    composition_editor_page(composition_id)


@ui.page("/presets")
def presets():
    ui.add_head_html(f"<style>{DARK_CSS}</style>")
    preset_library_page()


@ui.page("/triage/{strategy_hash}")
def triage(strategy_hash: str):
    ui.add_head_html(f"<style>{DARK_CSS}</style>")
    triage_results_page(strategy_hash)


@ui.page("/sweep/{composition_id}")
def sweep(composition_id: str):
    ui.add_head_html(f"<style>{DARK_CSS}</style>")
    parameter_sweep_page(composition_id)


def main():
    ui.run(
        title="BTC Alpha Research Panel",
        host="0.0.0.0",
        port=8080,
        dark=True,
        reload=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
