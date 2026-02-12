"""Pod cockpit screen — 6 gauges + halt button."""

import os
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Grid
from textual.screen import Screen
from textual.widgets import Static, Label, Header, Footer, Button
from textual.reactive import reactive

from tui.services.status_reader import PodStatus, read_pod_status


class Gauge(Static):
    """Single metric gauge."""

    def __init__(self, label: str, value: str = "—", gauge_id: str = "") -> None:
        super().__init__(classes="gauge")
        self.gauge_label = label
        self.gauge_value = value
        self._gauge_id = gauge_id

    def compose(self) -> ComposeResult:
        yield Label(self.gauge_label, classes="gauge-label")
        yield Label(self.gauge_value, id=f"val-{self._gauge_id}", classes="gauge-value")

    def update_value(self, value: str) -> None:
        self.gauge_value = value
        try:
            val_label = self.query_one(f"#val-{self._gauge_id}", Label)
            val_label.update(value)
        except Exception:
            pass


class CockpitScreen(Screen):
    """Pod cockpit — 6 gauges for one pod."""

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("r", "refresh", "Refresh"),
        ("h", "halt_pod", "Halt"),
    ]

    def __init__(self, pod_id: str = "", status_dir: str = "") -> None:
        super().__init__()
        self.pod_id = pod_id
        self.status_dir = status_dir
        self._status: PodStatus | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Static(f"Pod: {self.pod_id[:16]}...", id="cockpit-title")
            with Grid(id="cockpit-grid"):
                yield Gauge("Run Mode", "—", "mode")
                yield Gauge("State", "—", "state")
                yield Gauge("Health", "—", "health")
                yield Gauge("Bar Count", "—", "bars")
                yield Gauge("Last Bar TS", "—", "ts")
                yield Gauge("Watermark Seq", "—", "wm")
            yield Horizontal(
                Button("Halt Pod", id="halt-btn", variant="error"),
                Button("Back", id="back-btn", variant="default"),
                classes="mt-2 gap-2 px-2",
            )
            yield Static("", id="component-detail")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()
        self.set_interval(3.0, self._refresh)

    def _refresh(self) -> None:
        status_path = os.path.join(self.status_dir, self.pod_id, "status.json")
        if not os.path.exists(status_path):
            # Try scanning subdirectories
            import glob
            pattern = os.path.join(self.status_dir, "**", "status.json")
            for f in glob.glob(pattern, recursive=True):
                s = read_pod_status(f)
                if s and s.pod_id == self.pod_id:
                    self._status = s
                    break
            else:
                self._status = PodStatus(pod_id=self.pod_id, error="Status file not found")
        else:
            self._status = read_pod_status(status_path)

        self._update_gauges()

    def _update_gauges(self) -> None:
        if not self._status:
            return

        s = self._status
        gauges = {
            "mode": s.run_mode,
            "state": s.state,
            "health": s.health,
            "bars": str(s.bar_counter),
            "ts": str(s.last_bar_ts or "—"),
            "wm": str(s.watermark.get("commit_seq", "—")),
        }

        for gauge_id, value in gauges.items():
            try:
                gauge = self.query_one(f"#val-{gauge_id}", Label)
                gauge.update(value)
            except Exception:
                pass

        # Component detail
        detail_lines = []
        for name, comp in s.components.items():
            state = comp.get("state", "UNKNOWN") if isinstance(comp, dict) else "UNKNOWN"
            reason = comp.get("reason", "") if isinstance(comp, dict) else ""
            marker = "[green]OK[/green]" if state == "HEALTHY" else f"[red]{state}[/red]"
            line = f"  {name:20s} {marker}"
            if reason:
                line += f"  ({reason})"
            detail_lines.append(line)

        detail_text = "Components:\n" + "\n".join(detail_lines) if detail_lines else ""
        if s.halt_reason:
            detail_text += f"\n\n[red]HALT REASON: {s.halt_reason}[/red]"

        try:
            detail = self.query_one("#component-detail", Static)
            detail.update(detail_text)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "halt-btn":
            self.action_halt_pod()
        elif event.button.id == "back-btn":
            self.action_go_back()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_refresh(self) -> None:
        self._refresh()

    def action_halt_pod(self) -> None:
        from tui.screens.halt import HaltScreen
        self.app.push_screen(HaltScreen(
            pod_id=self.pod_id,
            status_dir=self.status_dir,
        ))
