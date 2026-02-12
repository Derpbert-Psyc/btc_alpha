"""Halt screen — two-step Arm → Confirm flow."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Static, Label, Button, Input, Footer
from textual.reactive import reactive

from tui.services.status_reader import write_halt_request


class HaltScreen(Screen):
    """Two-step halt confirmation: Arm → Confirm."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    armed: reactive[bool] = reactive(False)

    def __init__(self, pod_id: str = "", status_dir: str = "") -> None:
        super().__init__()
        self.pod_id = pod_id
        self.status_dir = status_dir

    def compose(self) -> ComposeResult:
        with Vertical(id="halt-container"):
            yield Label("HALT POD", classes="halt-title")
            yield Label(
                f"Pod: {self.pod_id[:16]}...",
            )
            yield Static("")
            yield Label(
                "This will immediately halt the pod.\n"
                "All open positions remain as-is.\n"
                "Manual intervention required to resume.",
                classes="halt-warning",
            )
            yield Static("")
            yield Input(
                placeholder="Reason for halt...",
                id="halt-reason",
            )
            yield Static("")
            yield Button(
                "ARM HALT",
                id="arm-btn",
                variant="warning",
            )
            yield Button(
                "CONFIRM HALT",
                id="confirm-btn",
                variant="error",
                disabled=True,
            )
            yield Static("")
            yield Button(
                "Cancel",
                id="cancel-btn",
                variant="default",
            )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "arm-btn":
            self.armed = True
            event.button.disabled = True
            event.button.label = "ARMED"
            confirm = self.query_one("#confirm-btn", Button)
            confirm.disabled = False

        elif event.button.id == "confirm-btn" and self.armed:
            reason_input = self.query_one("#halt-reason", Input)
            reason = reason_input.value.strip() or "Manual halt via TUI"

            try:
                filepath = write_halt_request(
                    self.status_dir, self.pod_id, reason)
                self.notify(f"Halt request written: {filepath}", severity="warning")
            except Exception as e:
                self.notify(f"Error: {e}", severity="error")

            self.app.pop_screen()

        elif event.button.id == "cancel-btn":
            self.action_cancel()

    def action_cancel(self) -> None:
        self.app.pop_screen()
