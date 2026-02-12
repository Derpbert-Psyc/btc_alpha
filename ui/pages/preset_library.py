"""Preset Library — load MACD Confluence or DBMR."""

import json
import os
from typing import Any, Dict

from nicegui import ui

from ui.services.composition_store import create_composition

PRESETS_DIR = os.path.join(os.path.dirname(__file__), "..", "presets")


def _load_preset_file(filename: str) -> Dict[str, Any]:
    path = os.path.join(PRESETS_DIR, filename)
    with open(path) as f:
        return json.load(f)


PRESET_INFO = [
    {
        "name": "MACD Confluence",
        "file": "macd_confluence.json",
        "description": "Multi-TF MACD alignment, momentum trigger, ATR sizing, Chop gating, SL+trail+time exits.",
        "archetype": "trend_following, multi_timeframe",
    },
    {
        "name": "DBMR — Double Bollinger Mean Reversion",
        "file": "dbmr.json",
        "description": "Bollinger mean reversion, structure-aware exits. Contrasting archetype.",
        "archetype": "mean_reversion, volatility_based",
    },
]


def preset_library_page():
    """Render preset library page."""
    with ui.column().classes("w-full max-w-4xl mx-auto p-4"):
        with ui.row().classes("items-center gap-4 mb-4"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props(
                "flat dense round")
            ui.label("Preset Library").classes("text-2xl font-bold")

        for preset in PRESET_INFO:
            with ui.card().classes("w-full mb-4 p-4"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.column():
                        ui.label(preset["name"]).classes("text-lg font-bold")
                        ui.label(preset["description"]).classes("text-sm text-gray-400")
                        ui.label(f"Archetypes: {preset['archetype']}").classes(
                            "text-xs text-gray-500")
                    ui.button("Load", icon="download",
                              on_click=lambda p=preset: _load_preset(p)).props(
                        "color=primary")


def _load_preset(preset_info: dict):
    """Load a preset — creates a new composition with new UUID."""
    spec = _load_preset_file(preset_info["file"])
    # Remove any existing composition_id — create_composition assigns a new one
    spec.pop("composition_id", None)
    cid = create_composition(spec, display_name=spec.get("display_name", preset_info["name"]))
    ui.notify(f"Loaded preset: {preset_info['name']}", type="positive")
    ui.navigate.to(f"/editor/{cid}")
