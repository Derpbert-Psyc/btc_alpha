"""Workspace Navigation Bar — Research / Shadow / Live mode selector."""

from nicegui import ui

from ui.services.composition_store import list_compositions, load_index
from ui.services.promotion_reader import derive_lifecycle_state


def _get_lifecycle_counts() -> dict:
    """Count strategies by workspace: research, shadow, live."""
    counts = {"research": 0, "shadow": 0, "live": 0}
    entries = list_compositions()
    index = load_index()
    compositions = index.get("compositions", {})

    for entry in entries:
        cid = entry["composition_id"]
        idx_entry = compositions.get(cid, {})
        compiled_hash = idx_entry.get("latest_compiled_hash")
        lifecycle, _, _ = derive_lifecycle_state(cid, compiled_hash)

        if lifecycle in ("DRAFT", "COMPILED", "TRIAGE_PASSED",
                         "BASELINE_PLUS_PASSED", "CORRUPTED"):
            counts["research"] += 1
        elif lifecycle == "SHADOW_VALIDATED":
            counts["shadow"] += 1
        elif lifecycle == "LIVE_APPROVED":
            counts["live"] += 1
        # ARCHIVED strategies not counted in any workspace

    return counts


def _nav_btn(label: str, icon: str, href: str, active: bool,
             color: str, count: int):
    """Render a workspace navigation button."""
    if active:
        btn = ui.button(label, icon=icon).props(f"color={color} no-caps")
    else:
        btn = ui.button(label, icon=icon,
                        on_click=lambda: ui.navigate.to(href)).props(
            f"color={color} outline no-caps")
    with btn:
        if count > 0:
            ui.badge(str(count)).props(f"color={color} floating")


def render_workspace_nav(active: str):
    """Render the workspace navigation bar.

    active: 'research' | 'shadow' | 'live'
    """
    counts = _get_lifecycle_counts()

    with ui.row().classes("w-full items-center gap-4 mb-6 pb-4").style(
            "border-bottom: 1px solid var(--border)"):
        _nav_btn("Research Lab", "science", "/",
                 active == "research", "blue", counts["research"])
        _nav_btn("Shadow Monitor", "visibility", "/shadow",
                 active == "shadow", "purple", counts["shadow"])
        _nav_btn("Live Operations", "bolt", "/live",
                 active == "live", "positive", counts["live"])
