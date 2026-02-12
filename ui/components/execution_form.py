"""Execution Form — direction, leverage, sizing, flip, funding."""

from typing import Callable
from nicegui import ui


def render_execution_form(state, on_change: Callable):
    """Render execution parameters tab."""
    spec = state.working_spec
    ep = spec.setdefault("execution_params", {})

    with ui.column().classes("w-full"):
        with ui.row().classes("gap-4 flex-wrap"):
            direction = ui.select(
                ["BOTH", "LONG_ONLY", "SHORT_ONLY"],
                value=ep.get("direction", "BOTH"),
                label="Direction",
            ).classes("w-40").props("dense")
            direction.on("update:model-value", lambda e: _update(state, ep, "direction", e.value, on_change))

            leverage = ui.number(
                value=ep.get("leverage", 1),
                min=1, max=20,
                label="Leverage",
            ).classes("w-24").props("dense")
            leverage.on("change", lambda e: _update(state, ep, "leverage", int(e.value), on_change))

            sizing = ui.select(
                ["RISK_FRACTION", "FIXED_QTY", "FIXED_NOTIONAL"],
                value=ep.get("sizing_mode", "RISK_FRACTION"),
                label="Sizing Mode",
            ).classes("w-44").props("dense")
            sizing.on("update:model-value", lambda e: _update(state, ep, "sizing_mode", e.value, on_change))

            risk_frac = ui.number(
                value=ep.get("risk_fraction_bps", 100),
                label="Risk Fraction (bps)",
            ).classes("w-40").props("dense")
            risk_frac.on("change", lambda e: _update(state, ep, "risk_fraction_bps", int(e.value), on_change))

            flip = ui.checkbox(
                "Flip Enabled",
                value=ep.get("flip_enabled", False),
            ).props("dense")
            flip.on("update:model-value", lambda e: _update(state, ep, "flip_enabled", e.value, on_change))

        # Funding model
        ui.separator()
        ui.label("Funding Model").classes("text-sm font-bold text-gray-400 mt-2")
        fm = ep.setdefault("funding_model", {})
        with ui.row().classes("gap-4 flex-wrap"):
            fm_enabled = ui.checkbox(
                "Enabled",
                value=fm.get("enabled", False),
            ).props("dense")
            fm_enabled.on("update:model-value", lambda e: _update_fm(state, ep, "enabled", e.value, on_change))

            fm_interval = ui.number(
                value=fm.get("interval_hours", 8),
                label="Interval (hours)",
            ).classes("w-32").props("dense")
            fm_interval.on("change", lambda e: _update_fm(state, ep, "interval_hours", int(e.value), on_change))

            fm_source = ui.select(
                ["FIXED", "LIVE"],
                value=fm.get("rate_source", "FIXED"),
                label="Rate Source",
            ).classes("w-28").props("dense")
            fm_source.on("update:model-value", lambda e: _update_fm(state, ep, "rate_source", e.value, on_change))

            fm_rate = ui.number(
                value=fm.get("rate_per_interval_bps", 1),
                label="Rate (bps/interval)",
            ).classes("w-40").props("dense")
            fm_rate.on("change", lambda e: _update_fm(state, ep, "rate_per_interval_bps", int(e.value), on_change))

            fm_credit = ui.checkbox(
                "Credit Allowed",
                value=fm.get("credit_allowed", False),
            ).props("dense")
            fm_credit.on("update:model-value", lambda e: _update_fm(state, ep, "credit_allowed", e.value, on_change))


def _update(state, ep, field, value, on_change):
    ep[field] = value
    state.mark_changed()
    on_change()


def _update_fm(state, ep, field, value, on_change):
    fm = ep.setdefault("funding_model", {})
    fm[field] = value
    state.mark_changed()
    on_change()
