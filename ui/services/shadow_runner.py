"""Shadow execution engine — paper trading against live Bybit testnet data.

Connects to Bybit testnet WebSocket for live kline bars, feeds them into
the strategy's indicator engine, and tracks simulated PnL in-memory.

Phase 1: WebSocket + PaperTracker infrastructure. Signal evaluation is stubbed.
"""

import asyncio
import json
import logging
import time
from typing import Callable, Optional

import websockets

BYBIT_TESTNET_WS = "wss://stream-testnet.bybit.com/v5/public/linear"

logger = logging.getLogger(__name__)


class BybitKlineClient:
    """WebSocket client for Bybit testnet kline stream."""

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1"):
        self.symbol = symbol
        self.interval = interval
        self._ws = None
        self._running = False
        self._on_bar: Optional[Callable] = None
        self._on_connected: Optional[Callable] = None
        self._on_reconnect: Optional[Callable] = None

    async def connect(self, on_bar: Callable):
        """Connect and stream kline bars. on_bar receives dict with OHLCV."""
        self._on_bar = on_bar
        self._running = True

        while self._running:
            try:
                async with websockets.connect(BYBIT_TESTNET_WS) as ws:
                    self._ws = ws
                    sub = {
                        "op": "subscribe",
                        "args": [f"kline.{self.interval}.{self.symbol}"],
                    }
                    await ws.send(json.dumps(sub))

                    if self._on_connected:
                        self._on_connected()

                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        if data.get("topic", "").startswith("kline"):
                            for bar in data.get("data", []):
                                if bar.get("confirm"):  # Only completed bars
                                    await self._on_bar({
                                        "timestamp": int(bar["start"]) // 1000,
                                        "open": float(bar["open"]),
                                        "high": float(bar["high"]),
                                        "low": float(bar["low"]),
                                        "close": float(bar["close"]),
                                        "volume": float(bar["volume"]),
                                    })
            except Exception as e:
                if not self._running:
                    break
                logger.warning("WS disconnected: %s, reconnecting in 5s...", e)
                if self._on_reconnect:
                    self._on_reconnect()
                await asyncio.sleep(5)

    def stop(self):
        self._running = False


class PaperTracker:
    """Paper trading simulator — records signals as simulated trades."""

    def __init__(self):
        self.position: Optional[dict] = None  # {side, entry_price, entry_time}
        self.trades: list = []
        self.total_pnl_bps: float = 0
        self.peak_equity_bps: float = 0
        self.max_drawdown_bps: float = 0
        self.win_count: int = 0
        self.loss_count: int = 0

    def on_signal(self, signal: str, price: float, timestamp: int):
        """Process a strategy signal. signal: 'LONG', 'SHORT', 'FLAT'."""
        if signal == "FLAT" and self.position:
            self._close_position(price, timestamp)
        elif signal in ("LONG", "SHORT"):
            if self.position:
                self._close_position(price, timestamp)
            self.position = {
                "side": signal.lower(),
                "entry_price": price,
                "entry_time": timestamp,
            }

    def _close_position(self, exit_price: float, timestamp: int):
        if not self.position:
            return
        entry = self.position["entry_price"]
        if self.position["side"] == "long":
            pnl_bps = (exit_price - entry) / entry * 10000
        else:
            pnl_bps = (entry - exit_price) / entry * 10000

        self.trades.append({
            "side": self.position["side"],
            "entry_price": entry,
            "exit_price": exit_price,
            "entry_time": self.position["entry_time"],
            "exit_time": timestamp,
            "pnl_bps": round(pnl_bps, 2),
        })

        self.total_pnl_bps += pnl_bps
        if pnl_bps > 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        self.peak_equity_bps = max(self.peak_equity_bps, self.total_pnl_bps)
        dd = self.peak_equity_bps - self.total_pnl_bps
        self.max_drawdown_bps = max(self.max_drawdown_bps, dd)

        self.position = None

    def summary(self) -> dict:
        total = self.win_count + self.loss_count
        return {
            "total_trades": total,
            "win_rate": (self.win_count / total * 100) if total > 0 else 0,
            "total_pnl_bps": round(self.total_pnl_bps, 2),
            "max_drawdown_bps": round(self.max_drawdown_bps, 2),
            "open_position": self.position,
        }


class ShadowRunner:
    """Orchestrates shadow execution for a single strategy."""

    def __init__(self, composition_id: str, resolved_artifact: dict, spec: dict):
        self.composition_id = composition_id
        self.resolved = resolved_artifact
        self.spec = spec
        self.client = BybitKlineClient()
        self.tracker = PaperTracker()
        self.status = "IDLE"  # IDLE, CONNECTING, RUNNING, STOPPED, ERROR
        self.bars_received = 0
        self.started_at = None
        self.error = None
        self._task = None

    async def start(self):
        """Start shadow execution as background task."""
        self.status = "CONNECTING"
        self.started_at = time.time()
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        try:
            self.client._on_connected = lambda: setattr(self, "status", "RUNNING")
            self.client._on_reconnect = lambda: setattr(self, "status", "RECONNECTING")
            await self.client.connect(on_bar=self._on_bar)
        except Exception as e:
            self.status = "ERROR"
            self.error = str(e)

    async def _on_bar(self, bar: dict):
        """Process each completed bar from the WebSocket feed."""
        self.bars_received += 1

        signal = self._evaluate_signal(bar)

        if signal:
            self.tracker.on_signal(signal, bar["close"], bar["timestamp"])

    def _evaluate_signal(self, bar: dict) -> Optional[str]:
        """Evaluate strategy indicators and return signal.

        Returns 'LONG', 'SHORT', 'FLAT', or None (no signal).

        Phase 1 stub — returns None. Full signal evaluation in Phase 2.
        """
        return None

    def stop(self):
        self.client.stop()
        self.status = "STOPPED"
        if self._task:
            self._task.cancel()

    def get_status(self) -> dict:
        return {
            "status": self.status,
            "bars_received": self.bars_received,
            "uptime_seconds": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "tracker": self.tracker.summary(),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Module-level registry of active shadow runners
# ---------------------------------------------------------------------------
_active_runners: dict = {}  # composition_id -> ShadowRunner
_registry_lock = asyncio.Lock()


def get_runner(composition_id: str) -> Optional[ShadowRunner]:
    return _active_runners.get(composition_id)


async def start_shadow(
    composition_id: str, resolved_artifact: dict, spec: dict
) -> ShadowRunner:
    async with _registry_lock:
        if composition_id in _active_runners:
            _active_runners[composition_id].stop()
        runner = ShadowRunner(composition_id, resolved_artifact, spec)
        _active_runners[composition_id] = runner
        await runner.start()
        return runner


def stop_shadow(composition_id: str):
    runner = _active_runners.pop(composition_id, None)
    if runner:
        runner.stop()
