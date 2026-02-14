"""Exit Rule Editor — 6 types with drag reorder and priority badges."""

from typing import Callable
from nicegui import ui

from composition_compiler_v1_5_2 import CAPABILITY_REGISTRY, EXIT_TYPE_TO_CAPABILITY
from strategy_framework_v1_8_0 import version_gte
from ui.components.condition_builder import render_condition_builder

EXIT_TYPES = [
    "STOP_LOSS", "TRAILING_STOP", "TIME_LIMIT",
    "SIGNAL", "MTM_DRAWDOWN_EXIT", "TAKE_PROFIT",
]

PRIORITY_BADGES = ["❶", "❷", "❸", "❹", "❺", "❻"]


def _available_exit_types(target_ev: str):
    result = []
    for et in EXIT_TYPES:
        cap = EXIT_TYPE_TO_CAPABILITY.get(et)
        if cap is None:
            result.append(et)
            continue
        cap_info = CAPABILITY_REGISTRY.get(cap, {})
        min_ver = cap_info.get("min_engine_version", "1.0.0")
        if version_gte(target_ev, min_ver):
            result.append(et)
    return result


def render_exit_rules(state, on_change: Callable):
    """Render exit rules editor tab."""
    spec = state.working_spec
    rules = spec.setdefault("exit_rules", [])
    target_ev = spec.get("target_engine_version", "1.8.0")
    available_types = _available_exit_types(target_ev)

    with ui.column().classes("w-full"):
        if not state.locked:
            ui.button("Add Exit Rule", icon="add",
                      on_click=lambda: _add_exit_rule(state, on_change)).props("color=primary")

        if not rules:
            ui.label("No exit rules.").classes("text-gray-400 py-4")
            return

        for i, rule in enumerate(rules):
            et = rule.get("exit_type", "SIGNAL")
            badge = PRIORITY_BADGES[i] if i < len(PRIORITY_BADGES) else f"({i+1})"

            with ui.card().classes("w-full mb-2 p-3 accent-amber"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label(badge).classes("text-lg")

                        name_input = ui.input(
                            value=rule.get("label", ""),
                            label="Name",
                        ).classes("w-48").props("dense")
                        name_input.on("change", lambda e, idx=i: _update_exit_field(
                            state, idx, "label", e.args, on_change))

                        ui.select(
                            available_types,
                            value=et,
                            label="Type",
                            on_change=lambda e, idx=i: _update_exit_field(
                                state, idx, "exit_type", e.value, on_change),
                        ).classes("w-44").props("dense")

                        raw_cadence = rule.get("evaluation_cadence", "1m")
                        cadence_opts = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "3d"]
                        if raw_cadence not in cadence_opts:
                            cadence_opts.append(raw_cadence)
                        ui.select(
                            cadence_opts,
                            value=raw_cadence,
                            label="Cadence",
                            on_change=lambda e, idx=i: _update_exit_field(
                                state, idx, "evaluation_cadence", e.value, on_change),
                        ).classes("w-24").props("dense")

                    with ui.row().classes("gap-1"):
                        if i > 0:
                            ui.button(icon="arrow_upward",
                                      on_click=lambda idx=i: _move_exit(state, idx, -1, on_change)).props(
                                "flat dense round size=sm")
                        if i < len(rules) - 1:
                            ui.button(icon="arrow_downward",
                                      on_click=lambda idx=i: _move_exit(state, idx, 1, on_change)).props(
                                "flat dense round size=sm")
                        ui.button(icon="delete",
                                  on_click=lambda idx=i: _delete_exit(state, idx, on_change)).props(
                            "flat dense round size=sm color=negative")

                # applies_to direction checkboxes (H7: default injection does NOT mark dirty)
                if "applies_to" not in rule:
                    rule["applies_to"] = ["LONG", "SHORT"]
                applies_to = rule.get("applies_to", ["LONG", "SHORT"])
                if not applies_to:
                    rule["applies_to"] = ["LONG", "SHORT"]
                    applies_to = rule["applies_to"]

                with ui.row().classes("items-center gap-1 mt-1"):
                    ui.label("Applies to:").classes("text-xs text-gray-400")
                    long_cb = ui.checkbox("LONG", value="LONG" in applies_to).props("dense")
                    short_cb = ui.checkbox("SHORT", value="SHORT" in applies_to).props("dense")

                    def _update_applies_to(e=None, idx=i, lcb=long_cb, scb=short_cb, r=rule):
                        result = []
                        if lcb.value:
                            result.append("LONG")
                        if scb.value:
                            result.append("SHORT")
                        if not result:
                            lcb.value = True
                            scb.value = True
                            result = ["LONG", "SHORT"]
                            ui.notify("applies_to cannot be empty — reset to both", type="warning")
                        r["applies_to"] = result
                        state.mark_changed()
                        on_change()

                    long_cb.on("update:model-value", _update_applies_to)
                    short_cb.on("update:model-value", _update_applies_to)

                # Type-specific parameters
                if et == "STOP_LOSS":
                    _render_stop_loss_params(state, i, rule, on_change)
                elif et == "TRAILING_STOP":
                    _render_trailing_stop_params(state, i, rule, on_change)
                elif et == "TIME_LIMIT":
                    _render_time_limit_params(state, i, rule, on_change)
                elif et == "MTM_DRAWDOWN_EXIT":
                    _render_mtm_params(state, i, rule, on_change)
                elif et in ("SIGNAL", "TAKE_PROFIT"):
                    ui.label("Conditions:").classes("text-sm text-gray-400 mt-2")
                    render_condition_builder(
                        state, rule.setdefault("conditions", []),
                        f"exit_{i}_cond", on_change)


def _render_stop_loss_params(state, idx, rule, on_change):
    _MODE_ALIASES = {"PERCENT": "FIXED_PERCENT"}
    with ui.row().classes("gap-4 mt-2"):
        raw_mode = rule.get("mode", "FIXED_PERCENT")
        ui.select(["FIXED_PERCENT", "ATR_MULTIPLE"],
                  value=_MODE_ALIASES.get(raw_mode, raw_mode),
                  label="Mode",
                  on_change=lambda e: _update_exit_field(
                      state, idx, "mode", e.value, on_change),
                  ).classes("w-40").props("dense")

        vl = ui.number(value=rule.get("value_long_bps", 200),
                       label="Long (bps)").classes("w-28").props("dense")
        vl.on("change", lambda e: _update_exit_field(
            state, idx, "value_long_bps", int(e.args), on_change))

        vs = ui.number(value=rule.get("value_short_bps", 200),
                       label="Short (bps)").classes("w-28").props("dense")
        vs.on("change", lambda e: _update_exit_field(
            state, idx, "value_short_bps", int(e.args), on_change))


def _render_trailing_stop_params(state, idx, rule, on_change):
    _MODE_ALIASES = {"PERCENT": "FIXED_PERCENT"}
    with ui.row().classes("gap-4 mt-2"):
        raw_mode = rule.get("mode", "ATR_MULTIPLE")
        ui.select(["FIXED_PERCENT", "ATR_MULTIPLE"],
                  value=_MODE_ALIASES.get(raw_mode, raw_mode),
                  label="Mode",
                  on_change=lambda e: _update_exit_field(
                      state, idx, "mode", e.value, on_change),
                  ).classes("w-40").props("dense")

        vl = ui.number(value=rule.get("value_long_bps", 300),
                       label="Long (bps)").classes("w-28").props("dense")
        vl.on("change", lambda e: _update_exit_field(
            state, idx, "value_long_bps", int(e.args), on_change))

        vs = ui.number(value=rule.get("value_short_bps", 300),
                       label="Short (bps)").classes("w-28").props("dense")
        vs.on("change", lambda e: _update_exit_field(
            state, idx, "value_short_bps", int(e.args), on_change))


def _render_time_limit_params(state, idx, rule, on_change):
    with ui.row().classes("gap-4 mt-2"):
        bars = ui.number(value=rule.get("time_limit_bars", 96),
                         label="Max bars").classes("w-28").props("dense")
        bars.on("change", lambda e: _update_exit_field(
            state, idx, "time_limit_bars", int(e.args), on_change))

        raw_ref = rule.get("time_limit_reference_cadence", "15m")
        ref_opts = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "3d"]
        if raw_ref not in ref_opts:
            ref_opts.append(raw_ref)
        ui.select(ref_opts,
                  value=raw_ref,
                  label="Reference cadence",
                  on_change=lambda e: _update_exit_field(
                      state, idx, "time_limit_reference_cadence", e.value, on_change),
                  ).classes("w-32").props("dense")


def _render_mtm_params(state, idx, rule, on_change):
    with ui.row().classes("gap-4 mt-2"):
        dl = ui.number(value=rule.get("drawdown_bps_long", 500),
                       label="DD Long (bps)").classes("w-28").props("dense")
        dl.on("change", lambda e: _update_exit_field(
            state, idx, "drawdown_bps_long", int(e.args), on_change))

        ds = ui.number(value=rule.get("drawdown_bps_short", 500),
                       label="DD Short (bps)").classes("w-28").props("dense")
        ds.on("change", lambda e: _update_exit_field(
            state, idx, "drawdown_bps_short", int(e.args), on_change))


def _update_exit_field(state, idx, field, value, on_change):
    rules = state.working_spec.get("exit_rules", [])
    if idx < len(rules):
        rules[idx][field] = value
        state.mark_changed()
        on_change()


def _add_exit_rule(state, on_change):
    rules = state.working_spec.setdefault("exit_rules", [])
    rules.append({
        "label": "New Exit",
        "exit_type": "SIGNAL",
        "applies_to": ["LONG", "SHORT"],
        "evaluation_cadence": "1m",
        "conditions": [],
    })
    state.mark_changed()
    on_change()
    ui.navigate.to(f"/editor/{state.composition_id}")


def _delete_exit(state, idx, on_change):
    rules = state.working_spec.get("exit_rules", [])
    if idx < len(rules):
        rules.pop(idx)
        state.mark_changed()
        on_change()
        ui.navigate.to(f"/editor/{state.composition_id}")


def _move_exit(state, idx, direction, on_change):
    rules = state.working_spec.get("exit_rules", [])
    new_idx = idx + direction
    if 0 <= new_idx < len(rules):
        rules[idx], rules[new_idx] = rules[new_idx], rules[idx]
        state.mark_changed()
        on_change()
        ui.navigate.to(f"/editor/{state.composition_id}")
