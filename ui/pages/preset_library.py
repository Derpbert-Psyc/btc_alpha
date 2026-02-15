"""Preset Library — browse and load strategy presets."""

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
        "name": "MACD Confluence Long",
        "file": "macd_confluence_long.json",
        "description": "Long-only multi-TF MACD alignment. Enter long on 5m momentum inflection with 6-TF consensus. Exit on 1D slope reversal.",
        "archetype": "trend_following, multi_timeframe",
    },
    {
        "name": "MACD Confluence Short",
        "file": "macd_confluence_short.json",
        "description": "Short-only multi-TF MACD alignment. Enter short on 5m momentum inflection with 6-TF consensus. Exit on 1D slope reversal.",
        "archetype": "trend_following, multi_timeframe",
    },
    {
        "name": "DBMR — Double Bollinger Mean Reversion",
        "file": "dbmr.json",
        "description": "Bollinger mean reversion, structure-aware exits. Contrasting archetype.",
        "archetype": "mean_reversion, volatility_based",
    },
    {
        "name": "Chop Harvester Clean",
        "file": "chop_harvester_clean.json",
        "description": "Mean-reversion for ranging markets. Donchian channel position entry, ATR 1.5x stop, 144-bar cooldown.",
        "archetype": "mean_reversion, range_trading",
    },
    {
        "name": "Chop Harvester Tight Stop",
        "file": "chop_harvester_tight_stop.json",
        "description": "Tighter stop variant (ATR 1.3x). Faster exit on adverse moves, same 144-bar cooldown.",
        "archetype": "mean_reversion, range_trading",
    },
    {
        "name": "Chop Harvester Conservative",
        "file": "chop_harvester_conservative.json",
        "description": "Conservative variant with doubled cooldown (288 bars). Fewer trades, avoids rapid re-entry whipsaws.",
        "archetype": "mean_reversion, range_trading",
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
