"""Lifecycle Badge — derived state badge component."""

from nicegui import ui

LIFECYCLE_COLORS = {
    "DRAFT": "grey",
    "COMPILED": "blue",
    "TRIAGE_PASSED": "green",
    "BASELINE_PLUS_PASSED": "teal",
    "SHADOW_VALIDATED": "purple",
    "LIVE_APPROVED": "positive",
    "CORRUPTED": "negative",
}


def render_lifecycle_badge(state: str, warning: str = None):
    """Render a lifecycle state badge."""
    color = LIFECYCLE_COLORS.get(state, "grey")
    badge = ui.badge(state, color=color)
    if warning:
        with badge:
            ui.tooltip(warning)
    return badge
