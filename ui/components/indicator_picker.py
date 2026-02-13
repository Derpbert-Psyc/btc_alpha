"""Indicator Instance Picker — modal with 6 groups, configure instance."""

from typing import Any, Dict, List, Optional

from nicegui import ui

from ui.services.indicator_catalog import (
    get_all_indicators,
    get_outputs_for_indicator,
    load_indicator_groups,
    resolve_indicator_id,
)
from strategy_framework_v1_8_0 import INDICATOR_ID_TO_NAME

QUICK_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "3d"]
ROLES = ["trigger", "filter", "price", "sizing", "gate", "diagnostic"]


async def show_indicator_picker(target_engine_version: str = "1.8.0") -> Optional[Dict[str, Any]]:
    """Show indicator picker modal. Returns configured instance dict or None."""
    result = {"value": None}

    with ui.dialog() as dialog, ui.card().classes("w-[800px] max-w-[90vw]"):
        ui.label("Add Indicator Instance").classes("text-xl font-bold mb-4")

        groups = load_indicator_groups()
        all_indicators = get_all_indicators()

        # Selection state
        selected_id = {"value": None}
        config_container = ui.column().classes("w-full")

        # Group tabs
        with ui.tabs().classes("w-full") as group_tabs:
            tabs = {}
            for g in groups:
                tabs[g["name"]] = ui.tab(g["name"])

        with ui.tab_panels(group_tabs).classes("w-full"):
            for g in groups:
                with ui.tab_panel(tabs[g["name"]]):
                    with ui.row().classes("gap-2 flex-wrap"):
                        for iid in g["indicator_ids"]:
                            name = INDICATOR_ID_TO_NAME.get(iid, f"id_{iid}")
                            outputs = get_outputs_for_indicator(iid)
                            btn = ui.button(
                                f"{name} ({iid})",
                                on_click=lambda iid=iid: _select_indicator(
                                    iid, config_container, selected_id, result,
                                    dialog, target_engine_version),
                            ).props("flat dense")
                            with btn:
                                ui.tooltip(", ".join(outputs.keys()))

    dialog.open()
    r = await dialog
    return result["value"]


def _select_indicator(indicator_id, container, selected_id, result, dialog,
                      target_engine_version):
    """Configure the selected indicator."""
    selected_id["value"] = indicator_id
    name = INDICATOR_ID_TO_NAME.get(indicator_id, f"id_{indicator_id}")
    outputs = get_outputs_for_indicator(indicator_id)

    container.clear()
    with container:
        ui.separator()
        ui.label(f"Configure: {name}").classes("text-lg font-bold")

        # Label — auto-generates from {name}_{timeframe}
        label_manually_set = {"value": False}
        default_label = f"{name}_15m"
        label_input = ui.input(value=default_label, label="Instance Label").classes("w-full")

        def _on_label_manual_edit(e):
            label_manually_set["value"] = True
        label_input.on("change", _on_label_manual_edit)

        # Timeframe — dual mode
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("Timeframe:").classes("text-sm")
            tf_mode = ui.toggle(["Quick", "Custom"], value="Quick").props("dense")

        quick_tf = ui.select(QUICK_TIMEFRAMES, value="15m", label="Timeframe").classes("w-32")
        custom_row = ui.row().classes("gap-2 items-center").style("display: none")
        with custom_row:
            custom_val = ui.number(value=15, min=1, label="Value").classes("w-20")
            custom_unit = ui.select(["min", "hr", "day", "wk"], value="min").classes("w-20")

        def toggle_tf(e):
            if e.args == "Custom":
                quick_tf.style("display: none")
                custom_row.style("")
            else:
                quick_tf.style("")
                custom_row.style("display: none")
            _update_auto_label()
        tf_mode.on("update:model-value", toggle_tf)

        # Role
        role_select = ui.select(ROLES, value="trigger", label="Role").classes("w-32")

        # Group
        group_select = ui.select(
            [g["name"] for g in load_indicator_groups()],
            value="Trend",
            label="Group",
        ).classes("w-32")

        # Outputs used (multi-select)
        output_names = list(outputs.keys())
        outputs_select = ui.select(
            output_names,
            value=output_names,
            label="Outputs Used",
            multiple=True,
        ).classes("w-full")

        # Parameters
        ui.label("Parameters:").classes("text-sm text-gray-400 mt-2")
        param_inputs = {}
        default_params = _get_default_params(indicator_id)
        for pname, pval in default_params.items():
            inp = ui.number(value=pval, label=pname).classes("w-32")
            param_inputs[pname] = inp

        # Auto-label generation: {name}_{tf}, or price_{tf} for EMA period=1
        def _get_current_tf():
            if tf_mode.value == "Quick":
                return quick_tf.value
            return f"{int(custom_val.value)}{custom_unit.value}"

        def _update_auto_label(_=None):
            if label_manually_set["value"]:
                return
            tf = _get_current_tf()
            # Special case: EMA with period=1 → price_{tf}
            if indicator_id == 1 and "period" in param_inputs:
                if param_inputs["period"].value == 1:
                    label_input.value = f"price_{tf}"
                    return
            label_input.value = f"{name}_{tf}"

        # Wire auto-label to timeframe and parameter changes
        quick_tf.on("update:model-value", _update_auto_label)
        custom_val.on("change", _update_auto_label)
        custom_unit.on("update:model-value", _update_auto_label)
        if indicator_id == 1 and "period" in param_inputs:
            param_inputs["period"].on("change", _update_auto_label)

        # Confirm / Cancel
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            def confirm():
                tf = quick_tf.value if tf_mode.value == "Quick" else f"{int(custom_val.value)}{custom_unit.value}"
                params = {k: v.value for k, v in param_inputs.items()}
                instance = {
                    "label": label_input.value,
                    "indicator_id": name,
                    "timeframe": tf,
                    "parameters": params,
                    "outputs_used": outputs_select.value or output_names,
                    "role": role_select.value,
                    "group": group_select.value,
                }
                result["value"] = instance
                dialog.submit(instance)

            ui.button("Add", on_click=confirm).props("color=primary")


def _get_default_params(indicator_id: int) -> dict:
    """Get default parameters for an indicator."""
    defaults = {
        1: {"period": 20},            # EMA
        2: {"period": 14},            # RSI
        3: {"period": 14},            # ATR
        4: {"left": 5, "right": 5},   # Pivot Structure
        5: {"anchor": "session"},     # AVWAP
        7: {"fast_period": 12, "slow_period": 26, "signal_period": 9},  # MACD
        8: {"period": 14},            # ROC
        9: {"period": 14},            # ADX
        10: {"period": 14},           # Choppiness
        11: {"period": 20, "num_std": 2.0},  # Bollinger
        12: {"period": 14},           # LinReg
        13: {"period": 20},           # HV
        14: {"period": 20},           # Donchian
        15: {},                       # Floor Pivots
        16: {"lookback": 100},        # Dynamic SR
        17: {"target_vol": 0.15},     # Vol Targeting
        18: {"num_bins": 50},         # VRVP
    }
    return dict(defaults.get(indicator_id, {}))
