"""Fleet overview screen — lists all pods with live status."""

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Static, Label, Header, Footer, ListView, ListItem
from textual.reactive import reactive

from tui.services.status_reader import PodStatus, scan_pod_statuses


class PodCard(ListItem):
    """Single pod status card."""

    def __init__(self, status: PodStatus) -> None:
        super().__init__()
        self.pod_status = status

    def compose(self) -> ComposeResult:
        s = self.pod_status
        health_class = f"status-{s.health.lower()}" if s.health != "UNKNOWN" else "status-unknown"

        with Horizontal(classes="pod-card " + s.health.lower()):
            yield Label(f" [{s.run_mode:8s}] ", classes="pod-mode")
            yield Label(f"{s.pod_id[:12]}...", classes="pod-id")
            yield Label(f"  {s.state:8s}", classes=health_class)
            yield Label(f"  bars={s.bar_counter}", classes="pod-bars")
            if s.halt_reason:
                yield Label(f"  HALT: {s.halt_reason[:30]}", classes="status-halted")


class FleetScreen(Screen):
    """Fleet overview — all pods."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("q", "quit_app", "Quit"),
        ("escape", "quit_app", "Quit"),
    ]

    status_dir: reactive[str] = reactive("")

    def __init__(self, status_dir: str = "") -> None:
        super().__init__()
        self.status_dir = status_dir

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(id="fleet-container")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_pods()
        self.set_interval(5.0, self._refresh_pods)

    def _refresh_pods(self) -> None:
        container = self.query_one("#fleet-container", Vertical)
        container.remove_children()

        statuses = scan_pod_statuses(self.status_dir)

        if not statuses:
            container.mount(
                Static(
                    "[dim]No pods active[/dim]\n\n"
                    "Pods will appear here when started via CLI.\n"
                    "Status directory: " + (self.status_dir or "(not configured)"),
                    classes="empty-state",
                )
            )
            return

        # Summary line
        n_healthy = sum(1 for s in statuses if s.is_healthy)
        n_halted = sum(1 for s in statuses if s.is_halted)
        n_total = len(statuses)
        summary = (
            f"Fleet: {n_total} pod(s) | "
            f"[green]{n_healthy} healthy[/green] | "
            f"[red]{n_halted} halted[/red]"
        )
        container.mount(Static(summary))
        container.mount(Static(""))

        # Pod list
        pod_list = ListView()
        for status in statuses:
            pod_list.append(PodCard(status))
        container.mount(pod_list)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, PodCard):
            self.app.push_screen(
                "cockpit",
                pod_id=event.item.pod_status.pod_id,
            )

    def action_refresh(self) -> None:
        self._refresh_pods()

    def action_quit_app(self) -> None:
        self.app.exit()
