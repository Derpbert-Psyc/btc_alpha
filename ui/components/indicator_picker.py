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


# ---------------------------------------------------------------------------
# Auto-label generation — indicator-specific patterns
# ---------------------------------------------------------------------------

def _generate_auto_label(name: str, indicator_id: int, tf: str, params: dict) -> str:
    """Generate auto-label based on indicator-specific patterns."""
    n = name.lower()
    if n == "ema":
        period = params.get("period", 20)
        if isinstance(period, (int, float)):
            period = int(period)
        if period == 1:
            return f"price_{tf}"
        return f"ema_{tf}_p{period}"
    if n == "rsi":
        return f"rsi_{tf}_p{params.get('period', 14)}"
    if n == "atr":
        return f"atr_{tf}_p{params.get('period', 14)}"
    if n in ("macd", "macd_tv"):
        return f"macd_{tf}"
    if n == "bollinger":
        return f"boll_{tf}_p{params.get('period', 20)}"
    if n == "donchian":
        return f"dc_{tf}_p{params.get('period', 20)}"
    if n == "adx":
        return f"adx_{tf}_p{params.get('period', 14)}"
    if n == "choppiness":
        return f"chop_{tf}_p{params.get('period', 14)}"
    if n == "linreg":
        return f"linreg_{tf}_p{params.get('period', 14)}"
    if n == "hv":
        return f"hv_{tf}_p{params.get('period', 20)}"
    if n == "roc":
        return f"roc_{tf}_p{params.get('period', 14)}"
    if n == "pivot_structure":
        return f"pivot_{tf}"
    if n == "floor_pivots":
        return f"fpivot_{tf}"
    if n == "dynamic_sr":
        return f"dsr_{tf}"
    if n == "vol_targeting":
        return f"voltgt_{tf}"
    if n == "vrvp":
        return f"vrvp_{tf}"
    # All others: {name}_{tf}
    return f"{n}_{tf}"


def _unique_label(base: str, existing_labels: list) -> str:
    """Generate unique label with collision avoidance."""
    if base not in existing_labels:
        return base
    suffix = 2
    while f"{base}_{suffix}" in existing_labels:
        suffix += 1
    return f"{base}_{suffix}"


# ---------------------------------------------------------------------------
# Picker dialog
# ---------------------------------------------------------------------------

async def show_indicator_picker(
    target_engine_version: str = "1.8.0",
    existing_labels: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Show indicator picker modal. Returns configured instance dict or None."""
    result = {"value": None}
    if existing_labels is None:
        existing_labels = []

    with ui.dialog() as dialog, ui.card().classes("w-[800px] max-w-[90vw]"):
        ui.label("Add Indicator Instance").classes("text-xl font-bold mb-4")

        groups = load_indicator_groups()
        all_indicators = get_all_indicators()

        # Selection state
        selected_id = {"value": None}
        config_container = ui.column().classes("w-full")

        # Group tabs
        GROUP_COLORS = {
            "Trend": "group-trend", "Momentum": "group-momentum",
            "Volatility": "group-volatility", "Volume": "group-volume",
            "Support/Resistance": "group-sr", "Price": "group-price",
        }
        with ui.tabs().classes("w-full") as group_tabs:
            tabs = {}
            for g in groups:
                tab = ui.tab(g["name"])
                color_cls = GROUP_COLORS.get(g["name"])
                if color_cls:
                    tab.classes(color_cls)
                tabs[g["name"]] = tab

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
                                    dialog, target_engine_version, existing_labels),
                            ).props("flat dense")
                            with btn:
                                ui.tooltip(", ".join(outputs.keys()))

    dialog.open()
    r = await dialog
    return result["value"]


def _select_indicator(indicator_id, container, selected_id, result, dialog,
                      target_engine_version, existing_labels):
    """Configure the selected indicator."""
    selected_id["value"] = indicator_id
    name = INDICATOR_ID_TO_NAME.get(indicator_id, f"id_{indicator_id}")
    outputs = get_outputs_for_indicator(indicator_id)

    container.clear()
    with container:
        ui.separator()
        ui.label(f"Configure: {name}").classes("text-lg font-bold")

        # Label — auto-generates with indicator-specific pattern
        label_manually_set = {"value": False}
        default_params = _get_default_params(indicator_id)
        initial_label = _unique_label(
            _generate_auto_label(name, indicator_id, "15m", default_params),
            existing_labels,
        )
        label_input = ui.input(value=initial_label, label="Instance Label").classes("w-full")
        label_error = ui.label("").classes("text-xs text-red-400")

        def _on_label_manual_edit(e):
            label_manually_set["value"] = True
            _validate_label()
        label_input.on("change", _on_label_manual_edit)

        def _validate_label():
            val = label_input.value.strip()
            if val in existing_labels:
                label_error.text = "Label already exists"
                add_btn.props("disable")
            else:
                label_error.text = ""
                add_btn.props(remove="disable")

        # Timeframe — dual mode (handler defined before toggle, forward refs OK for async)
        def toggle_tf(e):
            is_custom = e.value == "Custom"
            quick_tf.set_visibility(not is_custom)
            custom_row.set_visibility(is_custom)
            _update_auto_label()

        with ui.row().classes("w-full items-center gap-2"):
            ui.label("Timeframe:").classes("text-sm")
            tf_mode = ui.toggle(["Quick", "Custom"], value="Quick",
                                on_change=toggle_tf).props("dense")

        quick_tf = ui.select(
            QUICK_TIMEFRAMES, value="15m", label="Timeframe",
            on_change=lambda e: _update_auto_label(),
        ).classes("w-32")
        custom_row = ui.row().classes("gap-2 items-center")
        custom_row.set_visibility(False)
        with custom_row:
            custom_val = ui.number(value=15, min=1, label="Value").classes("w-20")
            custom_unit = ui.select(
                ["m", "h", "d"], value="m",
                label="Unit",
                on_change=lambda e: _update_auto_label(),
            ).classes("w-20")

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
        for pname, pval in default_params.items():
            inp = ui.number(value=pval, label=pname).classes("w-32")
            param_inputs[pname] = inp

        # Auto-label generation
        def _get_current_tf():
            if tf_mode.value == "Quick":
                return quick_tf.value
            return f"{int(custom_val.value)}{custom_unit.value}"

        def _get_current_params():
            return {k: v.value for k, v in param_inputs.items()}

        def _update_auto_label(_=None):
            if label_manually_set["value"]:
                return
            tf = _get_current_tf()
            params = _get_current_params()
            base = _generate_auto_label(name, indicator_id, tf, params)
            label_input.value = _unique_label(base, existing_labels)
            label_error.text = ""

        # Wire auto-label to parameter changes (timeframe already wired via on_change=)
        custom_val.on("change", _update_auto_label)
        for pname, inp in param_inputs.items():
            inp.on("change", _update_auto_label)

        # Confirm / Cancel
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            def confirm():
                if label_input.value.strip() in existing_labels:
                    label_error.text = "Label already exists"
                    return
                tf = quick_tf.value if tf_mode.value == "Quick" else f"{int(custom_val.value)}{custom_unit.value}"
                params = {k: v.value for k, v in param_inputs.items()}
                instance = {
                    "label": label_input.value.strip(),
                    "indicator_id": name,
                    "timeframe": tf,
                    "parameters": params,
                    "outputs_used": outputs_select.value or output_names,
                    "role": role_select.value,
                    "group": group_select.value,
                }
                result["value"] = instance
                dialog.submit(instance)

            add_btn = ui.button("Add", on_click=confirm).props("color=primary")


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
