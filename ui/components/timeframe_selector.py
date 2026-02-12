"""Timeframe Selector — dual-mode: quick-select + custom."""

from nicegui import ui

QUICK_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "3d", "1w"]


def render_timeframe_selector(value: str = "15m", on_change=None):
    """Render a dual-mode timeframe selector. Returns the UI element."""
    return ui.select(
        QUICK_TIMEFRAMES,
        value=value,
        label="Timeframe",
        on_change=on_change,
    ).classes("w-32").props("dense")
