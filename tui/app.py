"""BTC Alpha Operations Console — Textual TUI entry point.

Launch: python3 -m tui.app [--status-dir PATH]

Reads pod status from filesystem only. Does NOT import phase5_pod or any
project-specific runtime modules.
"""

import argparse
import os
import sys

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer

from tui.screens.fleet import FleetScreen
from tui.screens.cockpit import CockpitScreen

# Default status directory
DEFAULT_STATUS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "research", "pod_status"
)


class OpsConsole(App):
    """BTC Alpha Operations Console."""

    TITLE = "BTC Alpha Ops Console"
    CSS_PATH = os.path.join(os.path.dirname(__file__), "styles", "theme.tcss")

    SCREENS = {
        "fleet": FleetScreen,
    }

    def __init__(self, status_dir: str = "") -> None:
        super().__init__()
        self.status_dir = status_dir or DEFAULT_STATUS_DIR

    def on_mount(self) -> None:
        fleet = FleetScreen(status_dir=self.status_dir)
        self.push_screen(fleet)

    def push_screen(self, screen, **kwargs):
        """Override to inject status_dir into screens."""
        if isinstance(screen, str):
            if screen == "cockpit":
                pod_id = kwargs.get("pod_id", "")
                screen = CockpitScreen(
                    pod_id=pod_id,
                    status_dir=self.status_dir,
                )
            else:
                return super().push_screen(screen)
        return super().push_screen(screen)


def main():
    parser = argparse.ArgumentParser(description="BTC Alpha Operations Console")
    parser.add_argument(
        "--status-dir",
        default=DEFAULT_STATUS_DIR,
        help="Directory containing pod status files",
    )
    args = parser.parse_args()

    app = OpsConsole(status_dir=args.status_dir)
    app.run()


if __name__ == "__main__":
    main()
