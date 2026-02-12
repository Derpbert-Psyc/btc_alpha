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
:root {
    --bg-primary: #0a0f1a;
    --bg-panel: #0d1117;
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
.q-card, .q-table, .q-dialog {
    background-color: var(--bg-panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
.q-field__control {
    color: var(--text) !important;
}
.q-table__bottom, .q-table thead, .q-table th {
    color: var(--text-dim) !important;
}
.monospace { font-family: 'JetBrains Mono', 'Fira Code', monospace; }
.badge-draft { background: #334155; color: #94a3b8; }
.badge-compiled { background: #1e3a5f; color: #60a5fa; }
.badge-triage-passed { background: #064e3b; color: #6ee7b7; }
.badge-corrupted { background: #7f1d1d; color: #fca5a5; }
.badge-live { background: #14532d; color: #86efac; }
.compiler-success { border-left: 4px solid var(--green); background: #064e3b22; }
.compiler-error { border-left: 4px solid var(--red); background: #7f1d1d22; }
.compiler-warning { border-left: 4px solid var(--amber); background: #78350f22; }
.unsaved-dot::after { content: '●'; color: var(--amber); margin-left: 4px; font-size: 10px; }
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
