"""Compiler Feedback Panel — bottom dock showing compile results."""

from nicegui import ui


def render_compiler_panel(state, container):
    """Render the compiler feedback panel."""
    container.clear()
    result = state.last_compilation

    with container:
        if result is None:
            ui.label("No compilation yet. Press [Compile] to check your spec.").classes(
                "text-gray-400 py-4")
            return

        hash_val = result.get("strategy_config_hash", "")
        report = result.get("lowering_report", {})
        warnings = report.get("warnings", [])
        warmup = report.get("effective_warmup", {})
        resolved_ev = result.get("resolved_artifact", {}).get("engine_version", "")

        # Success banner
        with ui.card().classes("w-full p-4 compiler-success"):
            with ui.row().classes("items-center gap-4"):
                ui.icon("check_circle", color="green").classes("text-2xl")
                ui.label("Compilation Successful").classes("text-lg font-bold text-green-400")

            with ui.row().classes("gap-8 mt-2 flex-wrap"):
                with ui.column():
                    ui.label("Strategy Config Hash").classes("text-xs text-gray-400")
                    ui.label(hash_val).classes("monospace text-sm")
                with ui.column():
                    ui.label("Resolved Engine").classes("text-xs text-gray-400")
                    ev_text = resolved_ev
                    target_ev = state.working_spec.get("target_engine_version", "")
                    if resolved_ev != target_ev:
                        ev_text += f" (target was {target_ev})"
                    ui.label(ev_text).classes("text-sm")
                if warmup:
                    with ui.column():
                        ui.label("Effective Warmup").classes("text-xs text-gray-400")
                        warmup_text = f"{warmup.get('bars', '?')} bars"
                        dom = warmup.get("dominating_instance", "")
                        if dom:
                            warmup_text += f" ({dom})"
                        ui.label(warmup_text).classes("text-sm")

            # Warnings
            if warnings:
                ui.separator()
                with ui.card().classes("w-full p-2 mt-2 compiler-warning"):
                    ui.label(f"{len(warnings)} warning(s)").classes("text-sm text-amber-400 font-bold")
                    for w in warnings:
                        ui.label(f"  - {w}").classes("text-sm text-amber-300")

            # Action buttons
            with ui.row().classes("gap-2 mt-4"):
                ui.button("Run Triage", icon="science",
                          on_click=lambda: _navigate_triage(state)).props("color=primary")
                ui.button("View Lowering Report",
                          on_click=lambda: _show_report(report)).props("outline dense")
                ui.button("View Resolved Artifact",
                          on_click=lambda: _show_artifact(result)).props("outline dense")


def _navigate_triage(state):
    if state.last_compilation:
        hash_val = state.last_compilation["strategy_config_hash"]
        if hash_val.startswith("sha256:"):
            hash_val = hash_val[7:]
        ui.navigate.to(f"/triage/{hash_val}")


async def _show_report(report):
    import json
    with ui.dialog() as dialog, ui.card().classes("w-[800px] max-h-[80vh] overflow-auto"):
        ui.label("Lowering Report").classes("text-lg font-bold")
        ui.code(json.dumps(report, indent=2), language="json").classes("w-full")
        ui.button("Close", on_click=dialog.close)
    dialog.open()


async def _show_artifact(result):
    import json
    artifact = result.get("resolved_artifact", {})
    with ui.dialog() as dialog, ui.card().classes("w-[800px] max-h-[80vh] overflow-auto"):
        ui.label("Resolved Artifact").classes("text-lg font-bold")
        ui.code(json.dumps(artifact, indent=2), language="json").classes("w-full")
        ui.button("Close", on_click=dialog.close)
    dialog.open()
