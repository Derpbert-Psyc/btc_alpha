"""Shadow execution daemon -- runs as a separate process.

Launch: python3 shadow_daemon.py --instance-id <id>
Stop:   write {"command": "stop"} to research/shadow_status/{id}/command.json

Reads:  research/shadow_status/{id}/config.json (instance configuration)
Writes: research/shadow_status/{id}/status.json (every 5s)
        research/pod_status/shadow_{id}/status.json (TUI-compatible, every 5s)

Note: All data feeds use mainnet public APIs (no auth required).
      No orders are placed. Paper trading only.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

log = logging.getLogger("shadow_daemon")


# ---------------------------------------------------------------------------
# Uptime formatting
# ---------------------------------------------------------------------------

def format_uptime(seconds: int) -> str:
    """Format seconds into adaptive human-readable units.

    Rules:
    - Units: months (30 days), weeks, days, hours, mins, secs
    - No years -- max unit is months
    - Omit zero-valued units (including intermediate zeros)
    - Show at most 5 non-zero units
    - Month is defined as exactly 30 days; no calendar-month arithmetic

    Examples:
        135       -> "2 mins 15 secs"
        3661      -> "1 hour 1 min 1 sec"
        2270530   -> "3 weeks 5 days 6 hours 42 mins 10 secs"
    """
    if seconds < 0:
        seconds = 0

    months, seconds = divmod(seconds, 30 * 24 * 3600)
    weeks, seconds = divmod(seconds, 7 * 24 * 3600)
    days, seconds = divmod(seconds, 24 * 3600)
    hours, seconds = divmod(seconds, 3600)
    mins, secs = divmod(seconds, 60)

    parts = []
    for val, singular, plural in [
        (months, "month", "months"),
        (weeks, "week", "weeks"),
        (days, "day", "days"),
        (hours, "hour", "hours"),
        (mins, "min", "mins"),
        (secs, "sec", "secs"),
    ]:
        if val > 0:
            label = singular if val == 1 else plural
            parts.append(f"{val} {label}")

    return " ".join(parts[:5]) or "0 secs"


# ---------------------------------------------------------------------------
# Bar and Aggregation
# ---------------------------------------------------------------------------

class Bar:
    """OHLCV bar with Unix timestamp (seconds).

    Timestamp semantics:
    - Base bars: ts = bar OPEN time (as provided by exchange REST/WS APIs).
      Bybit and Binance both provide kline open time (start of interval).
    - Aggregated TF bars: ts = last base bar's open time in the bucket.
      This means TF bar ts is the open time of the last contributing base bar,
      NOT the bucket close time.

    NOTE: The repo's phase5_bundle_store.BarBundle uses CLOSE timestamps.
    If the daemon is later integrated with the phase5 evidence chain,
    timestamps must be normalized: close_ts = open_ts + interval_seconds.
    For the shadow daemon as a standalone research tool, using open time
    consistently is correct and sufficient.
    """
    __slots__ = ("ts", "o", "h", "l", "c", "v")

    def __init__(self, ts: int, o: float, h: float, l: float, c: float, v: float = 0.0):
        self.ts = ts
        self.o = o
        self.h = h
        self.l = l
        self.c = c
        self.v = v


class BarAggregator:
    """Aggregates base-interval bars into multiple higher timeframes.

    Uses TIMESTAMP-ALIGNED buckets (not count-based) so that WS gaps
    and reconnects do not shift higher-TF bar boundaries.

    Each timeframe has a fixed bucket size in seconds.
    Bucket ID = floor(bar.ts / tf_seconds).
    A new bucket starts when bucket ID changes; the previous bucket
    is finalized and emitted.

    Constructor args:
        timeframes: dict mapping label -> seconds (e.g., {"5s": 5, "15s": 15, ...})
        base_interval_seconds: base bar interval for validation only
        on_tf_bar: callback(tf_label: str, bar: Bar) called when a TF bar completes
    """

    def __init__(self, timeframes: Dict[str, int], base_interval_seconds: int, on_tf_bar):
        self.timeframes = timeframes
        self.base_interval_seconds = base_interval_seconds
        self.on_tf_bar = on_tf_bar

        # Validate divisibility
        for tf, secs in timeframes.items():
            if secs % base_interval_seconds != 0:
                raise ValueError(
                    f"Timeframe '{tf}' ({secs}s) is not an exact multiple "
                    f"of base interval ({base_interval_seconds}s)"
                )

        # Per-TF state: current bucket ID and accumulated bars
        self._current_bucket: Dict[str, Optional[int]] = {tf: None for tf in timeframes}
        self._bucket_bars: Dict[str, List[Bar]] = {tf: [] for tf in timeframes}
        self.tf_bar_counts: Dict[str, int] = {tf: 0 for tf in timeframes}

    def push_bar(self, bar: Bar):
        """Push a base-interval bar. Emits completed TF bars on bucket boundaries.

        Timestamp-aligned: bucket_id = bar.ts // tf_seconds.
        Two emission triggers:
        1. Bucket change: when bucket_id advances, finalize the previous bucket.
        2. Bucket completion: when this bar is the LAST bar of the current
           bucket (detected by: (bar.ts + base_interval) falls into next bucket),
           finalize immediately without waiting for the next bucket's first bar.

        This ensures TF bars emit with zero latency on the bucket boundary,
        not delayed by one base bar.
        """
        for tf_label, tf_seconds in self.timeframes.items():
            bucket_id = bar.ts // tf_seconds

            if self._current_bucket[tf_label] is None:
                # First bar ever -- start accumulating
                self._current_bucket[tf_label] = bucket_id
                self._bucket_bars[tf_label] = [bar]
            elif bucket_id != self._current_bucket[tf_label]:
                # New bucket -- finalize previous bucket first
                self._emit_bucket(tf_label)
                # Start new bucket with current bar
                self._current_bucket[tf_label] = bucket_id
                self._bucket_bars[tf_label] = [bar]
            else:
                # Same bucket -- accumulate
                self._bucket_bars[tf_label].append(bar)

            # Check if this bar completes the current bucket:
            # Next bar's timestamp would fall into the next bucket
            next_bar_ts = bar.ts + self.base_interval_seconds
            if next_bar_ts // tf_seconds != bucket_id:
                # This is the last bar of the bucket -- emit immediately
                self._emit_bucket(tf_label)
                self._current_bucket[tf_label] = None  # Reset for next bucket

    def _emit_bucket(self, tf_label: str):
        """Finalize and emit a completed TF bar from accumulated bucket bars."""
        buf = self._bucket_bars[tf_label]
        if not buf:
            return
        tf_bar = Bar(
            ts=buf[-1].ts,
            o=buf[0].o,
            h=max(b.h for b in buf),
            l=min(b.l for b in buf),
            c=buf[-1].c,
            v=sum(b.v for b in buf),
        )
        self._bucket_bars[tf_label] = []
        self.tf_bar_counts[tf_label] += 1
        self.on_tf_bar(tf_label, tf_bar)


# ---------------------------------------------------------------------------
# MACD Slope Sign
# ---------------------------------------------------------------------------

class EMA_SMA_Seed:
    """EMA with SMA-seeded initialization (TradingView-compatible)."""

    def __init__(self, length: int):
        self.length = int(length)
        self.alpha = 2.0 / (self.length + 1.0)
        self._seed: List[float] = []
        self._ema: Optional[float] = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def update(self, x: float) -> Optional[float]:
        x = float(x)
        if not self._ready:
            self._seed.append(x)
            if len(self._seed) == self.length:
                self._ema = sum(self._seed) / float(self.length)
                self._ready = True
            return self._ema
        self._ema = (x - self._ema) * self.alpha + self._ema
        return self._ema


class MACD_SlopeSign:
    """MACD slope direction indicator. Output: +1, 0, or -1."""

    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = EMA_SMA_Seed(fast)
        self.slow = EMA_SMA_Seed(slow)
        self.prev_macd: Optional[float] = None
        self.last_sign: Optional[int] = None

    @property
    def ready(self) -> bool:
        return (
            self.fast.ready
            and self.slow.ready
            and self.prev_macd is not None
            and self.last_sign is not None
        )

    def update_close(self, close: float) -> Optional[int]:
        ef = self.fast.update(close)
        es = self.slow.update(close)
        if ef is None or es is None:
            return None

        macd = ef - es
        if self.prev_macd is None:
            self.prev_macd = macd
            return None

        delta = macd - self.prev_macd
        self.prev_macd = macd

        if delta > 0:
            s = 1
        elif delta < 0:
            s = -1
        else:
            s = 0
        self.last_sign = s
        return s


REQUIRED_BARS_FOR_READY = 27  # EMA(26) seed + 1 for first delta


# ---------------------------------------------------------------------------
# MACD Confluence Strategy
# ---------------------------------------------------------------------------

class MACDConfluenceStrategy:
    """Role-based MACD confluence strategy.

    Roles (from config):
      macro: list of TF labels -- ALL must agree on direction for entry
      intra: list of TF labels -- ALL must agree on direction for entry
      entry: single TF label -- slope_sign must cross above/below 0 to trigger
      exit: single TF label -- slope_sign reversal triggers exit

    Entry does NOT require the exit TF to agree.
    """

    def __init__(self, config: dict):
        self.timeframes = config["timeframes"]  # {label: seconds}
        roles = config["roles"]
        self.macro_tfs = roles["macro"]       # list of TF labels
        self.intra_tfs = roles["intra"]       # list of TF labels
        self.entry_tf = roles["entry"]        # single TF label
        self.exit_tf = roles["exit"]          # single TF label
        self.long_only = config.get("long_only", False)
        fast = config.get("macd_fast", 12)
        slow = config.get("macd_slow", 26)

        self.macd_by_tf: Dict[str, MACD_SlopeSign] = {
            tf: MACD_SlopeSign(fast, slow) for tf in self.timeframes
        }
        self.slope_sign_now: Dict[str, Optional[int]] = {tf: None for tf in self.timeframes}
        self.prev_entry_sign: Optional[int] = None

    def on_tf_bar(self, tf_label: str, bar: Bar):
        """Called by BarAggregator when a timeframe bar completes."""
        s = self.macd_by_tf[tf_label].update_close(bar.c)
        if s is not None:
            self.slope_sign_now[tf_label] = int(s)

    def all_ready(self) -> bool:
        return all(m.ready for m in self.macd_by_tf.values())

    def evaluate_entry(self) -> Optional[dict]:
        """Evaluate entry conditions. Called when entry TF bar completes.

        Updates prev_entry_sign (crossover tracking). Must NOT be called
        on exit TF completions -- that would corrupt cross detection.

        Returns dict with long_entry/short_entry booleans, or None if not ready.
        """
        if not self.all_ready():
            return None

        cur_entry = self.slope_sign_now.get(self.entry_tf)

        # Role-based group checks
        macro_all_bullish = all(self.slope_sign_now.get(tf) == 1 for tf in self.macro_tfs)
        macro_all_bearish = all(self.slope_sign_now.get(tf) == -1 for tf in self.macro_tfs)
        intra_all_bullish = all(self.slope_sign_now.get(tf) == 1 for tf in self.intra_tfs)
        intra_all_bearish = all(self.slope_sign_now.get(tf) == -1 for tf in self.intra_tfs)

        # Entry trigger: crossover on entry TF
        entry_crosses_above = (
            self.prev_entry_sign is not None
            and self.prev_entry_sign <= 0
            and cur_entry is not None
            and cur_entry > 0
        )
        entry_crosses_below = (
            self.prev_entry_sign is not None
            and self.prev_entry_sign >= 0
            and cur_entry is not None
            and cur_entry < 0
        )

        # Update crossover tracking (ONLY here, never in evaluate_exit)
        self.prev_entry_sign = cur_entry

        return {
            "long_entry": macro_all_bullish and intra_all_bullish and entry_crosses_above,
            "short_entry": macro_all_bearish and intra_all_bearish and entry_crosses_below,
        }

    def evaluate_exit(self) -> Optional[dict]:
        """Evaluate exit conditions. Called when exit TF bar completes.

        Side-effect free with respect to entry state. Does NOT touch
        prev_entry_sign or any entry-cross tracking.

        Returns dict with exit_long/exit_short booleans, or None if not ready.
        """
        if not self.all_ready():
            return None

        exit_sign = self.slope_sign_now.get(self.exit_tf)

        return {
            "exit_long": exit_sign == -1,   # exit TF slope turns bearish
            "exit_short": exit_sign == 1,    # exit TF slope turns bullish
        }

    def snapshot(self) -> dict:
        """Return current state for status.json."""
        # Build per-TF detail if bar counts are available
        tf_detail = {}
        bar_counts = getattr(self, "_tf_bar_counts", None)
        for tf in self.timeframes:
            macd = self.macd_by_tf[tf]
            bars_processed = bar_counts.get(tf, 0) if bar_counts else 0
            tf_detail[tf] = {
                "bars_processed": bars_processed,
                "required": REQUIRED_BARS_FOR_READY,
                "ready": macd.ready,
                "slope_sign": self.slope_sign_now.get(tf),
            }
        return {
            "slope_signs": dict(self.slope_sign_now),
            "all_ready": self.all_ready(),
            "macro_aligned": {tf: self.slope_sign_now.get(tf) for tf in self.macro_tfs},
            "intra_aligned": {tf: self.slope_sign_now.get(tf) for tf in self.intra_tfs},
            "entry_sign": self.slope_sign_now.get(self.entry_tf),
            "exit_sign": self.slope_sign_now.get(self.exit_tf),
            "tf_detail": tf_detail,
        }


# ---------------------------------------------------------------------------
# PaperTracker
# ---------------------------------------------------------------------------

class PaperTracker:
    """Paper trading simulator with configurable friction. Version 2.

    DIFFERENCES from ui/services/shadow_runner.py PaperTracker (v1):
    - v1 does direct reversals (LONG->SHORT in one call). v2 requires flat first.
    - v1 has no friction model. v2 applies half-side spread (round_trip_bps/2).
    - v1 tracks PnL in BPS only. v2 tracks absolute USD PnL via paper_qty.
    - v2 adds stop-loss (checked every base bar, highest priority exit).

    v2 is authoritative for shadow daemon. Results are NOT directly comparable
    to v1 PaperTracker results. This is intentional -- v2 is more realistic.

    Enters only when flat. Does NOT do direct reversals.
    """

    def __init__(self, round_trip_bps: float = 25.0, paper_qty: float = 0.001,
                 long_only: bool = False,
                 stop_loss_long_bps: int = 0, stop_loss_short_bps: int = 0):
        self.round_trip_bps = round_trip_bps
        self.half_side_mult = (round_trip_bps / 2.0) / 10000.0
        self.paper_qty = paper_qty
        self.long_only = long_only
        self.stop_loss_long_frac = stop_loss_long_bps / 10000.0
        self.stop_loss_short_frac = stop_loss_short_bps / 10000.0

        self.position = 0  # -1, 0, +1
        self.entry_fill: Optional[float] = None
        self._entry_mid: Optional[float] = None  # mid price at entry (for stop-loss)
        self.total_pnl_bps = 0.0
        self.max_drawdown_bps = 0.0
        self.peak_pnl_bps = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0
        self.stop_loss_count = 0
        self.entries_long = 0
        self.entries_short = 0
        self.exits_total = 0

    def force_flat(self, exit_price: float, reason: str):
        """Force-close any open position. Public API for gap exits and emergencies.

        Uses _close_position internally -- never manipulate position/entry
        fields directly outside of _close_position.
        """
        if self.position != 0:
            self._close_position(exit_price, reason)

    def process_signals(self, signals: dict, mid_price: float) -> Optional[str]:
        """Process strategy signals and execute paper trades.

        Returns action taken: "ENTRY_LONG", "ENTRY_SHORT", "EXIT_LONG", "EXIT_SHORT", or None.
        """
        action = None

        # Exits first (before entries)
        if self.position == 1 and signals.get("exit_long"):
            self._close_position(mid_price, "EXIT_LONG")
            action = "EXIT_LONG"

        elif self.position == -1 and signals.get("exit_short"):
            self._close_position(mid_price, "EXIT_SHORT")
            action = "EXIT_SHORT"

        # Entries (only when flat)
        if self.position == 0:
            if signals.get("long_entry"):
                self._open_position(1, mid_price)
                self.entries_long += 1
                action = "ENTRY_LONG"
            elif not self.long_only and signals.get("short_entry"):
                self._open_position(-1, mid_price)
                self.entries_short += 1
                action = "ENTRY_SHORT"

        return action

    def _open_position(self, side: int, mid_price: float):
        self.position = side
        self._entry_mid = mid_price  # for stop-loss calculation
        if side == 1:
            self.entry_fill = mid_price * (1.0 + self.half_side_mult)
        else:
            self.entry_fill = mid_price * (1.0 - self.half_side_mult)

    def check_stop_loss(self, bar_low: float, bar_high: float, bar_close: float) -> Optional[str]:
        """Check if stop-loss is hit. Called on every base bar (highest priority).

        Stop-loss is OPTIONAL. Set stop_loss_long_bps=0 and stop_loss_short_bps=0
        to disable (as used by Big strategy where stop-loss hurts performance).

        Uses bar LOW for LONG stops and bar HIGH for SHORT stops per
        MACD_CONFLUENCE_STRATEGY_CONTRACT S4.7 conservative simulation:
        "LONG stop loss triggers if bar LOW <= stop price;
         SHORT stop loss triggers if bar HIGH >= stop price."

        For CX 1s bars, low == high == close (carry-forward), so this is
        equivalent to close-only. For Big 1m bars (if stop-loss is ever
        enabled), this correctly catches intra-bar violations.

        Returns "STOP_LONG", "STOP_SHORT", or None.
        """
        if self.position == 0 or self._entry_mid is None:
            return None

        if self.position == 1 and self.stop_loss_long_frac > 0:
            stop_price = self._entry_mid * (1.0 - self.stop_loss_long_frac)
            if bar_low <= stop_price:
                self._close_position(bar_close, "STOP_LONG")
                self.stop_loss_count += 1
                return "STOP_LONG"
        elif self.position == -1 and self.stop_loss_short_frac > 0:
            stop_price = self._entry_mid * (1.0 + self.stop_loss_short_frac)
            if bar_high >= stop_price:
                self._close_position(bar_close, "STOP_SHORT")
                self.stop_loss_count += 1
                return "STOP_SHORT"
        return None

    def _close_position(self, mid_price: float, exit_type: str):
        if self.entry_fill is None:
            self.position = 0
            return

        if self.position == 1:
            exit_fill = mid_price * (1.0 - self.half_side_mult)
        else:
            exit_fill = mid_price * (1.0 + self.half_side_mult)

        pnl_usd = (exit_fill - self.entry_fill) * self.paper_qty * self.position
        pnl_bps = (pnl_usd / (self.entry_fill * self.paper_qty)) * 10000.0

        self.total_pnl_bps += pnl_bps
        self.total_trades += 1
        self.exits_total += 1

        if pnl_bps >= 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        if self.total_pnl_bps > self.peak_pnl_bps:
            self.peak_pnl_bps = self.total_pnl_bps
        dd = self.peak_pnl_bps - self.total_pnl_bps
        if dd > self.max_drawdown_bps:
            self.max_drawdown_bps = dd

        self.position = 0
        self.entry_fill = None
        self._entry_mid = None

    def snapshot(self) -> dict:
        avg_pnl = round(self.total_pnl_bps / self.total_trades, 2) if self.total_trades > 0 else 0.0
        return {
            "position": self.position,
            "entry_fill": self.entry_fill,
            "total_pnl_bps": round(self.total_pnl_bps, 2),
            "max_drawdown_bps": round(self.max_drawdown_bps, 2),
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "total_trades": self.total_trades,
            "stop_loss_count": self.stop_loss_count,
            "win_rate": round(self.win_count / self.total_trades * 100, 1) if self.total_trades > 0 else 0.0,
            "entries_long": self.entries_long,
            "entries_short": self.entries_short,
            "exits_total": self.exits_total,
            "round_trip_bps": self.round_trip_bps,
            "avg_pnl_per_trade_bps": avg_pnl,
        }


# ---------------------------------------------------------------------------
# Exchange Adapters
# ---------------------------------------------------------------------------

class BybitAdapter:
    """Bybit mainnet public API adapter.

    REST: https://api.bybit.com/v5/market/kline
    WS (kline): wss://stream.bybit.com/v5/public/linear (kline.{interval}.{symbol})
    WS (ticker): wss://stream.bybit.com/v5/public/linear (tickers.{symbol})

    For CX strategy (1s base bars): uses ticker stream + 1Hz bar clock.
    For Big strategy (1m base bars): uses kline.1.{symbol} stream.
    """

    BASE_REST = "https://api.bybit.com"
    BASE_WS = "wss://stream.bybit.com/v5/public/linear"

    async def fetch_warmup_bars(self, symbol: str, category: str,
                                 interval_minutes: int, count: int) -> List[Bar]:
        """Fetch historical kline bars via REST API.

        Bybit returns bars in REVERSE chronological order (newest first).
        We paginate backward by moving the end cursor.

        Returns: List[Bar] in chronological order (oldest first)
        """
        import aiohttp

        all_bars = []
        # Floor end_ms to last fully closed minute boundary.
        # Bybit REST returns the current in-progress candle with non-final
        # closePrice. We must exclude it by ending 1ms before the current
        # minute boundary.
        now_ms = int(time.time() * 1000)
        interval_ms = interval_minutes * 60 * 1000
        end_ms = (now_ms // interval_ms) * interval_ms - 1  # last closed candle
        prev_end_ms = None  # Guard against non-advancing cursor
        seen_ts = set()
        max_retries = 3
        warmup_deadline = time.monotonic() + 300  # 5 minute wall-clock deadline

        async with aiohttp.ClientSession() as session:
            while len(all_bars) < count:
                if time.monotonic() > warmup_deadline:
                    raise RuntimeError(
                        f"Bybit warm-up deadline exceeded: fetched {len(all_bars)}/{count} bars"
                    )

                params = {
                    "category": category,
                    "symbol": symbol,
                    "interval": str(interval_minutes),
                    "end": str(end_ms),
                    "limit": "1000",
                }
                url = f"{self.BASE_REST}/v5/market/kline"

                # Retry loop with exponential backoff
                for attempt in range(max_retries):
                    async with session.get(url, params=params) as resp:
                        if resp.status == 429 or resp.status == 403:
                            wait = 2 ** attempt
                            log.warning(f"Bybit rate limited ({resp.status}), backing off {wait}s")
                            await asyncio.sleep(wait)
                            continue
                        if resp.status != 200:
                            raise RuntimeError(
                                f"Bybit REST error: HTTP {resp.status}"
                            )
                        data = await resp.json()
                        break
                else:
                    raise RuntimeError("Bybit REST: max retries exceeded")

                # Validate response schema
                ret_code = data.get("retCode", -1)
                if ret_code != 0:
                    ret_msg = data.get("retMsg", "")
                    # Bybit can signal rate-limit via retCode (not just HTTP 429)
                    if "rate limit" in ret_msg.lower() or ret_code in (10006, 10018):
                        wait = 2 ** min(attempt, max_retries - 1)
                        log.warning(f"Bybit rate limited via retCode={ret_code}: "
                                    f"{ret_msg}. Backing off {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    raise RuntimeError(
                        f"Bybit REST API error: retCode={ret_code}, "
                        f"retMsg={ret_msg}"
                    )

                result_list = data.get("result", {}).get("list", [])
                if not result_list:
                    break

                # Bybit returns newest first -- reverse to chronological
                page_bars = []
                for item in reversed(result_list):
                    ts = int(item[0]) // 1000
                    if ts not in seen_ts:
                        seen_ts.add(ts)
                        page_bars.append(Bar(
                            ts=ts,
                            o=float(item[1]),
                            h=float(item[2]),
                            l=float(item[3]),
                            c=float(item[4]),
                            v=float(item[5]),
                        ))

                # Prepend (these are older bars)
                all_bars = page_bars + all_bars

                # Move end cursor backward: oldest bar in this page minus 1ms
                oldest_ts_ms = int(result_list[-1][0])  # last item = oldest (reverse sorted)
                new_end_ms = oldest_ts_ms - 1

                # Guard: cursor must advance backward
                if prev_end_ms is not None and new_end_ms >= prev_end_ms:
                    log.warning("Bybit pagination stall detected, breaking")
                    break
                prev_end_ms = end_ms
                end_ms = new_end_ms

                if len(result_list) < 1000:
                    break  # No more data available

                await asyncio.sleep(0.2)  # Rate limit courtesy

        # Return only the most recent 'count' bars
        result = all_bars[-count:]

        # Warm-up integrity validation (MANDATORY)
        # Must verify continuity before caller replays through MACD engines.
        if len(result) >= 2:
            for i in range(1, len(result)):
                if result[i].ts <= result[i - 1].ts:
                    raise RuntimeError(
                        f"Warm-up integrity: non-monotonic timestamps at index {i}: "
                        f"ts[{i - 1}]={result[i - 1].ts}, ts[{i}]={result[i].ts}"
                    )
                expected_gap = interval_minutes * 60
                actual_gap = result[i].ts - result[i - 1].ts
                if actual_gap != expected_gap:
                    log.warning(
                        f"Warm-up gap at index {i}: expected {expected_gap}s, "
                        f"got {actual_gap}s (ts={result[i - 1].ts} -> {result[i].ts}). "
                        f"Gap of {actual_gap - expected_gap}s."
                    )
                    # Gaps in warm-up data are tolerable: MACD will update with
                    # the available bars, and any minor discontinuity washes out
                    # over 116k+ bars. Log for diagnostics but do not abort.

        return result

    async def stream_kline_bars(self, symbol: str, interval: int, on_bar,
                                stop_event: asyncio.Event = None):
        """Stream confirmed kline bars via WebSocket.

        For Big strategy: interval=1 (1 minute).
        Calls on_bar(Bar) for each confirmed bar (confirm=true).
        Reconnects on disconnect. Exits cleanly when stop_event is set.
        """
        import websockets

        self._last_kline_ts = 0  # Dedup tracker

        while not (stop_event and stop_event.is_set()):
            try:
                topic = f"kline.{interval}.{symbol}"
                sub_msg = json.dumps({"op": "subscribe", "args": [topic]})

                async with websockets.connect(
                    self.BASE_WS, ping_interval=20, ping_timeout=10
                ) as ws:
                    await ws.send(sub_msg)
                    log.info(f"WS subscribed to {topic}")

                    async for raw in ws:
                        if stop_event and stop_event.is_set():
                            return
                        msg = json.loads(raw)
                        if "data" not in msg:
                            continue
                        # Sort by start time and dedup: Bybit snapshots can
                        # contain multiple candles or repeat confirmed candles.
                        candles = sorted(msg["data"], key=lambda c: int(c["start"]))
                        for candle in candles:
                            if candle.get("confirm", False):
                                ts = int(candle["start"]) // 1000
                                if ts <= self._last_kline_ts:
                                    continue  # Dedup: already processed this candle
                                self._last_kline_ts = ts
                                bar = Bar(
                                    ts=ts,
                                    o=float(candle["open"]),
                                    h=float(candle["high"]),
                                    l=float(candle["low"]),
                                    c=float(candle["close"]),
                                    v=float(candle["volume"]),
                                )
                                await on_bar(bar)

            except Exception as e:
                if stop_event and stop_event.is_set():
                    return
                log.warning(f"WS disconnect: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def stream_ticker_1hz(self, symbol: str, on_bar,
                                stop_event: asyncio.Event = None):
        """Stream 1-second bars via ticker + 1Hz bar clock.

        For CX strategy. Subscribes to tickers.{symbol}, builds flat
        1s carry-forward bars at each second boundary.
        Exits cleanly when stop_event is set.
        """
        import websockets

        last_price = None
        last_finalized_sec = None

        while not (stop_event and stop_event.is_set()):
            try:
                topic = f"tickers.{symbol}"
                sub_msg = json.dumps({"op": "subscribe", "args": [topic]})

                async with websockets.connect(
                    self.BASE_WS, ping_interval=20, ping_timeout=10
                ) as ws:
                    await ws.send(sub_msg)
                    log.info(f"WS subscribed to {topic} (1Hz bar clock)")

                    async def price_reader():
                        nonlocal last_price
                        self._last_ticker_msg_at = time.monotonic()
                        self._last_price_update_at = time.monotonic()
                        async for raw in ws:
                            if stop_event and stop_event.is_set():
                                return
                            msg = json.loads(raw)

                            # Subscription ACK (op response) -- skip
                            if "op" in msg:
                                if not msg.get("success", True):
                                    raise RuntimeError(
                                        f"Ticker subscription failed: {msg}"
                                    )
                                continue

                            data = msg.get("data")
                            if data is None:
                                continue  # ping/pong or unknown

                            # Update WS message liveness on ANY valid ticker message
                            self._last_ticker_msg_at = time.monotonic()

                            # Bybit V5 linear tickers: "data" is a dict
                            # for snapshot and delta messages.
                            # Defensively handle list format if encountered.
                            if isinstance(data, list) and data:
                                data = data[0]

                            if not isinstance(data, dict):
                                continue

                            lp = data.get("lastPrice")
                            if lp is not None:
                                last_price = float(lp)
                                self._last_price_update_at = time.monotonic()
                            # If lastPrice absent: price unchanged, carry forward

                    async def bar_clock():
                        nonlocal last_finalized_sec
                        # Configurable liveness thresholds
                        ws_timeout = 30    # seconds without ANY ticker message
                        price_timeout = 60  # seconds without a lastPrice update
                        while not (stop_event and stop_event.is_set()):
                            now_sec = int(time.time())
                            now_mono = time.monotonic()

                            # Dual liveness check:
                            # 1. WS message liveness -- is the WS connection alive?
                            if hasattr(self, '_last_ticker_msg_at'):
                                if now_mono - self._last_ticker_msg_at > ws_timeout:
                                    raise RuntimeError(
                                        f"No ticker messages for {ws_timeout}s "
                                        f"(WS feed dead)"
                                    )

                            # 2. Price liveness -- is lastPrice updating?
                            #    During stable prices Bybit sends deltas without
                            #    lastPrice, but the initial snapshot always has it.
                            #    If price hasn't updated in price_timeout seconds,
                            #    the feed may be stuck or the market is halted.
                            if hasattr(self, '_last_price_update_at'):
                                if now_mono - self._last_price_update_at > price_timeout:
                                    raise RuntimeError(
                                        f"No lastPrice update for {price_timeout}s "
                                        f"(price feed stale, possible frozen feed)"
                                    )

                            if last_price is None:
                                await asyncio.sleep(0.05)
                                continue

                            if last_finalized_sec is None:
                                last_finalized_sec = now_sec - 1
                                await asyncio.sleep(0.05)
                                continue

                            target = now_sec - 1
                            while last_finalized_sec < target:
                                last_finalized_sec += 1
                                bar = Bar(
                                    ts=last_finalized_sec,
                                    o=last_price,
                                    h=last_price,
                                    l=last_price,
                                    c=last_price,
                                    v=0.0,
                                )
                                await on_bar(bar)

                            await asyncio.sleep(0.05)

                    # Run both concurrently
                    await asyncio.gather(price_reader(), bar_clock())

            except Exception as e:
                if stop_event and stop_event.is_set():
                    return
                log.warning(f"Ticker WS disconnect: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)


class BinanceAdapter:
    """Binance mainnet public API adapter.

    REST: https://api.binance.com/api/v3/klines
    WS: wss://stream.binance.com:9443/ws/btcusdt@kline_1m
    """

    BASE_REST = "https://api.binance.com"

    async def fetch_warmup_bars(self, symbol: str, interval: str,
                                 count: int) -> List[Bar]:
        """Fetch historical kline bars from Binance REST API.

        Args:
            symbol: e.g. "BTCUSDT"
            interval: e.g. "1m", "1s"
            count: total number of bars to fetch

        Returns: List[Bar] in chronological order
        """
        import aiohttp

        bars = []
        interval_ms = self._interval_to_ms(interval)
        # Floor end_ms to last fully closed interval boundary
        now_ms = int(time.time() * 1000)
        end_ms = (now_ms // interval_ms) * interval_ms - 1  # last closed candle
        cursor_start = end_ms - (count * interval_ms)
        max_retries = 3
        warmup_deadline = time.monotonic() + 300  # 5 minute wall-clock deadline

        async with aiohttp.ClientSession() as session:
            while len(bars) < count:
                if time.monotonic() > warmup_deadline:
                    raise RuntimeError(
                        f"Binance warm-up deadline exceeded: fetched {len(bars)}/{count} bars"
                    )

                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": str(cursor_start),
                    "endTime": str(end_ms),
                    "limit": "1000",
                }
                url = f"{self.BASE_REST}/api/v3/klines"

                # Retry loop with exponential backoff
                for attempt in range(max_retries):
                    async with session.get(url, params=params) as resp:
                        if resp.status == 429 or resp.status == 418:
                            wait = 2 ** attempt
                            log.warning(f"Binance rate limited ({resp.status}), backing off {wait}s")
                            await asyncio.sleep(wait)
                            continue
                        if resp.status != 200:
                            raise RuntimeError(
                                f"Binance REST error: HTTP {resp.status}"
                            )
                        data = await resp.json()
                        break
                else:
                    raise RuntimeError("Binance REST: max retries exceeded")

                if not isinstance(data, list):
                    raise RuntimeError(
                        f"Binance REST: unexpected response type: {type(data)}"
                    )

                if not data:
                    break

                # Binance returns chronological order (oldest first)
                for item in data:
                    bars.append(Bar(
                        ts=int(item[0]) // 1000,
                        o=float(item[1]),
                        h=float(item[2]),
                        l=float(item[3]),
                        c=float(item[4]),
                        v=float(item[5]),
                    ))

                if data:
                    cursor_start = int(data[-1][0]) + interval_ms
                else:
                    break

                await asyncio.sleep(0.1)  # Rate limit courtesy

        result = bars[:count]

        # Warm-up integrity validation (same as Bybit adapter)
        if len(result) >= 2:
            interval_ms_val = self._interval_to_ms(interval)
            expected_gap = interval_ms_val // 1000
            for i in range(1, len(result)):
                if result[i].ts <= result[i - 1].ts:
                    raise RuntimeError(
                        f"Warm-up integrity: non-monotonic timestamps at index {i}"
                    )
                actual_gap = result[i].ts - result[i - 1].ts
                if actual_gap != expected_gap:
                    log.warning(
                        f"Warm-up gap at index {i}: expected {expected_gap}s, "
                        f"got {actual_gap}s"
                    )

        return result

    def _interval_to_ms(self, interval: str) -> int:
        """Convert Binance interval string to milliseconds."""
        units = {"s": 1000, "m": 60000, "h": 3600000, "d": 86400000}
        num = int(interval[:-1])
        unit = interval[-1]
        return num * units[unit]

    async def stream_kline_bars(self, symbol: str, interval: str, on_bar,
                                stop_event: asyncio.Event = None):
        """Stream confirmed kline bars via Binance WebSocket.

        Subscribes to {symbol_lower}@kline_{interval}.
        Calls on_bar(Bar) when kline is closed (x=true).
        Exits cleanly when stop_event is set.

        NOTE: websockets library handles ping/pong automatically by default.
        We explicitly set ping_interval=20 and ping_timeout=10 to match
        Binance's documented requirements. Binance disconnects idle
        connections after 24h -- the reconnect loop handles this.
        """
        import websockets

        symbol_lower = symbol.lower()
        ws_url = f"wss://stream.binance.com:9443/ws/{symbol_lower}@kline_{interval}"
        self._last_kline_ts = 0  # Dedup tracker

        while not (stop_event and stop_event.is_set()):
            try:
                async with websockets.connect(
                    ws_url, ping_interval=20, ping_timeout=10
                ) as ws:
                    log.info(f"Binance WS connected: {symbol_lower}@kline_{interval}")
                    async for raw in ws:
                        if stop_event and stop_event.is_set():
                            return
                        msg = json.loads(raw)
                        k = msg.get("k")
                        if k is None:
                            continue
                        if k.get("x", False):  # kline closed
                            ts = int(k["t"]) // 1000
                            # Dedup: skip if we've already processed this candle
                            if ts <= self._last_kline_ts:
                                continue
                            self._last_kline_ts = ts
                            bar = Bar(
                                ts=ts,
                                o=float(k["o"]),
                                h=float(k["h"]),
                                l=float(k["l"]),
                                c=float(k["c"]),
                                v=float(k["v"]),
                            )
                            await on_bar(bar)

            except Exception as e:
                if stop_event and stop_event.is_set():
                    return
                log.warning(f"Binance WS disconnect: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# ShadowDaemon Main Class
# ---------------------------------------------------------------------------

class ShadowDaemon:
    """Main daemon class. Orchestrates warm-up, live streaming, and status writing."""

    def __init__(self, instance_id: str, config: dict):
        self.instance_id = instance_id
        self.config = config
        self.status = "STARTING"
        self.error = None
        self.warmup_progress = None
        self.started_at = time.time()
        self.bars_received = 0
        self.last_bar_time = None
        self.last_bar_price = None
        self._last_bar_epoch_ts = None
        self._shutdown = False
        self._stabilization_end = None  # set in run() if stabilization_seconds > 0
        self._gap_flag = False           # True = gap detected, suppress trading
        self._gap_recovery_count = 0     # consecutive on-time bars since last gap
        self._expected_next_ts = None    # expected next base bar timestamp
        self._status_task = None
        self._command_task = None

        # Status directories
        self.status_dir = os.path.join(
            PROJECT_ROOT, "research", "shadow_status", instance_id
        )
        self.pod_status_dir = os.path.join(
            PROJECT_ROOT, "research", "pod_status", f"shadow_{instance_id}"
        )
        os.makedirs(self.status_dir, exist_ok=True)
        os.makedirs(self.pod_status_dir, exist_ok=True)

        # Components
        base_interval = config["base_interval_seconds"]
        self.strategy = MACDConfluenceStrategy(config)
        self.aggregator = BarAggregator(
            config["timeframes"], base_interval, self.strategy.on_tf_bar
        )
        # Wire bar counts so strategy snapshot can report per-TF progress
        self.strategy._tf_bar_counts = self.aggregator.tf_bar_counts
        self.tracker = PaperTracker(
            round_trip_bps=config.get("round_trip_bps", 25.0),
            paper_qty=config.get("paper_qty", 0.001),
            long_only=config.get("long_only", False),
            stop_loss_long_bps=config.get("stop_loss_long_bps", 0),
            stop_loss_short_bps=config.get("stop_loss_short_bps", 0),
        )

        # Shared stop event for deterministic shutdown
        self._stop_event = None  # set in run()

        # Exchange adapter
        exchange = config["exchange"]
        if exchange == "bybit":
            self.adapter = BybitAdapter()
        elif exchange == "binance":
            self.adapter = BinanceAdapter()
        else:
            raise ValueError(f"Unknown exchange: {exchange}")

    async def run(self):
        """Main daemon loop: warm-up -> live streaming -> signal evaluation."""
        log.info(f"Shadow daemon starting: {self.instance_id}")
        self._stop_event = asyncio.Event()

        try:
            # Phase 1: Warm-up from historical data
            await self._warmup()

            # Phase 2: Connect to live WebSocket
            self.status = "CONNECTING"
            self._write_status()

            # Start background tasks
            self._status_task = asyncio.create_task(self._status_loop())
            self._command_task = asyncio.create_task(self._command_loop())

            # Phase 3: Begin streaming
            # Stabilization: suppress entries for configurable window (CX only by default)
            stab_seconds = self.config.get("stabilization_seconds", 0)
            self._stabilization_end = None

            if stab_seconds > 0:
                self.status = "STABILIZING"
                self._stabilization_end = time.time() + stab_seconds
                log.info(f"Stabilization window: {stab_seconds}s "
                         f"(entries suppressed until {self._stabilization_end})")
            else:
                self.status = "RUNNING"

            self._write_status()

            live_task = asyncio.create_task(self._stream_live())

            # Wait for either live_task to end or shutdown to be requested
            while not self._shutdown:
                if live_task.done():
                    break
                await asyncio.sleep(0.5)

            # Signal adapters to stop cleanly, then cancel task
            self._stop_event.set()
            if not live_task.done():
                live_task.cancel()
                try:
                    await live_task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            self.status = "ERROR"
            self.error = str(e)
            log.exception(f"Daemon error: {e}")
            self._write_status()
        finally:
            if self.status != "ERROR":
                self.status = "STOPPED"
            self._write_status()

            # Structured cancellation: cancel background tasks
            for task in (self._status_task, self._command_task):
                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            log.info("Shadow daemon stopped")

    async def _warmup(self):
        """Fetch historical bars and replay through strategy."""
        self.status = "WARMING_UP"
        self._write_status()

        config = self.config
        exchange = config["exchange"]
        symbol = config["symbol"]
        base_interval = config["base_interval_seconds"]

        # Calculate how many base bars are needed
        max_tf_seconds = max(config["timeframes"].values())
        bars_per_max_tf = max_tf_seconds // base_interval
        warmup_base_bars = REQUIRED_BARS_FOR_READY * bars_per_max_tf

        log.info(f"Warm-up: need {warmup_base_bars} base bars "
                 f"({base_interval}s each, {max_tf_seconds}s max TF)")

        if exchange == "bybit":
            if base_interval == 1:
                # CX strategy: fetch 1m bars, subdivide into synthetic 1s
                warmup_minutes = (warmup_base_bars // 60) + 1
                log.info(f"CX warm-up: fetching {warmup_minutes} 1m bars from Bybit REST")
                one_min_bars = await self.adapter.fetch_warmup_bars(
                    symbol, config.get("category", "linear"), 1, warmup_minutes
                )
                log.info(f"CX warm-up: replaying {len(one_min_bars)} 1m bars as synthetic 1s")
                for i, bar_1m in enumerate(one_min_bars):
                    # Each 1m bar -> 60 synthetic 1s bars at the close price
                    # bar_1m.ts is the candle START time (seconds)
                    # Synthetic 1s bars span ts+0 through ts+59
                    for s in range(60):
                        synthetic_bar = Bar(
                            ts=bar_1m.ts + s,
                            o=bar_1m.c, h=bar_1m.c, l=bar_1m.c, c=bar_1m.c, v=0.0,
                        )
                        self._on_base_bar(synthetic_bar)
                    # Progress update every 100 minutes
                    if i % 100 == 0:
                        self.warmup_progress = f"Replaying {i}/{len(one_min_bars)} 1m bars..."
                        self._write_status()
            else:
                # Big strategy: fetch 1m bars directly
                log.info(f"Big warm-up: fetching {warmup_base_bars} 1m bars from Bybit REST")
                bars = await self.adapter.fetch_warmup_bars(
                    symbol, config.get("category", "linear"), 1, warmup_base_bars
                )
                log.info(f"Big warm-up: replaying {len(bars)} bars")
                for i, bar in enumerate(bars):
                    self._on_base_bar(bar)
                    if i % 10000 == 0:
                        self.warmup_progress = f"Replaying {i}/{len(bars)} bars..."
                        self._write_status()

        elif exchange == "binance":
            # Big strategy on Binance: fetch 1m bars
            log.info(f"Big warm-up: fetching {warmup_base_bars} 1m bars from Binance REST")
            bars = await self.adapter.fetch_warmup_bars(symbol, "1m", warmup_base_bars)
            log.info(f"Big warm-up: replaying {len(bars)} bars")
            for i, bar in enumerate(bars):
                self._on_base_bar(bar)
                if i % 10000 == 0:
                    self.warmup_progress = f"Replaying {i}/{len(bars)} bars..."
                    self._write_status()

        # Verify all MACD instances are ready
        if not self.strategy.all_ready():
            not_ready = [tf for tf, m in self.strategy.macd_by_tf.items() if not m.ready]
            self.status = "ERROR"
            self.error = f"Warm-up failed: MACDs not ready for TFs: {not_ready}"
            log.error(self.error)
            self._write_status()
            raise RuntimeError(self.error)

        log.info("Warm-up complete. All MACDs ready.")

    def _on_base_bar(self, bar: Bar):
        """Process a single base-interval bar (warm-up or live).

        Updates daemon state and pushes through aggregator -> strategy MACD update.
        Signal evaluation is NOT done here -- it happens in _on_live_bar (live only).
        """
        self.bars_received += 1
        self.last_bar_price = bar.c
        self._last_bar_epoch_ts = bar.ts
        self.last_bar_time = datetime.fromtimestamp(
            bar.ts, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Push through aggregator (which calls strategy.on_tf_bar for completed TFs)
        self.aggregator.push_bar(bar)

    async def _on_live_bar(self, bar: Bar):
        """Process a live bar (async wrapper for _on_base_bar + signal evaluation).

        Evaluation order (highest priority first):
        1. Gap detection: update gap_flag, force-flat if positioned + large gap
        2. Stop-loss: checked on EVERY base bar (active even during gap suppression)
        3. Exit: evaluated when exit TF bar completes (suppressed during gap_flag)
        4. Entry: evaluated when entry TF bar completes (suppressed during gap_flag/STABILIZING)

        Gap flag suppression matches reference backtester: "trading is blocked when
        any relevant timeframe gap flag is True." MACD state continues updating
        (it needs to track through gaps) but no signals are acted on.

        Entry evaluation ALWAYS calls evaluate_entry() to keep prev_entry_sign
        tracking correct, even when entries are suppressed.
        """
        entry_tf = self.config["roles"]["entry"]
        exit_tf = self.config["roles"]["exit"]
        base_interval = self.config["base_interval_seconds"]
        gap_recovery_bars = self.config.get("gap_recovery_bars", 5)

        # --- Gap detection for Big (not CX -- bar clock fills gaps) ---
        if base_interval > 1 and self._expected_next_ts is not None:
            if bar.ts != self._expected_next_ts:
                missed = (bar.ts - self._expected_next_ts) // base_interval
                if missed > 0:
                    log.warning(f"Gap: expected ts={self._expected_next_ts}, "
                                f"got ts={bar.ts}, missed ~{missed} bars")
                    self._gap_flag = True
                    self._gap_recovery_count = 0
                    # Force-flat if positioned and large gap
                    if self.tracker.position != 0 and missed >= 5:
                        log.warning("Force-flat: 5+ missing bars while positioned")
                        self.tracker.force_flat(bar.c, "FORCE_FLAT_GAP")
                else:
                    # On-time bar while gap_flag is set: count toward recovery
                    if self._gap_flag:
                        self._gap_recovery_count += 1
                        if self._gap_recovery_count >= gap_recovery_bars:
                            self._gap_flag = False
                            log.info(f"Gap recovered after {gap_recovery_bars} "
                                     f"consecutive on-time bars")
            else:
                # Exactly on time
                if self._gap_flag:
                    self._gap_recovery_count += 1
                    if self._gap_recovery_count >= gap_recovery_bars:
                        self._gap_flag = False
                        log.info(f"Gap recovered after {gap_recovery_bars} "
                                 f"consecutive on-time bars")

        # Update expected next timestamp
        if base_interval > 1:
            self._expected_next_ts = bar.ts + base_interval

        # --- Stop-loss check: every base bar, highest priority ---
        # Active even during gap suppression (safety).
        if self.tracker.position != 0:
            stop_action = self.tracker.check_stop_loss(bar.l, bar.h, bar.c)
            if stop_action:
                log.info(f"Stop loss: {stop_action} @ {bar.c:.2f}")
                self._on_base_bar(bar)
                return

        old_entry_count = self.aggregator.tf_bar_counts.get(entry_tf, 0)
        old_exit_count = self.aggregator.tf_bar_counts.get(exit_tf, 0)

        self._on_base_bar(bar)

        new_entry_count = self.aggregator.tf_bar_counts.get(entry_tf, 0)
        new_exit_count = self.aggregator.tf_bar_counts.get(exit_tf, 0)

        # --- Exit evaluation: fires on exit TF completion ---
        # Suppressed during gap_flag (except force-flat which already happened above)
        exit_delta = new_exit_count - old_exit_count
        for _ in range(exit_delta):
            if self._gap_flag:
                continue  # Signal exits suppressed during gap; stop-loss still active
            if self.tracker.position != 0 and self.last_bar_price is not None:
                exit_signals = self.strategy.evaluate_exit()
                if exit_signals:
                    action = None
                    if self.tracker.position == 1 and exit_signals["exit_long"]:
                        action = self.tracker.process_signals(
                            {"long_entry": False, "short_entry": False, **exit_signals},
                            self.last_bar_price,
                        )
                    elif self.tracker.position == -1 and exit_signals["exit_short"]:
                        action = self.tracker.process_signals(
                            {"long_entry": False, "short_entry": False, **exit_signals},
                            self.last_bar_price,
                        )
                    if action:
                        log.info(f"Exit signal: {action} @ {self.last_bar_price:.2f}")

        # --- Entry evaluation: fires on entry TF completion ---
        # ALWAYS call evaluate_entry to keep prev_entry_sign correct,
        # but suppress actual entries during gap_flag or STABILIZING.
        entry_delta = new_entry_count - old_entry_count
        for _ in range(entry_delta):
            entry_signals = self.strategy.evaluate_entry()

            # Suppress during gap
            if self._gap_flag:
                continue

            # Check stabilization window
            if self._stabilization_end is not None:
                if time.time() >= self._stabilization_end:
                    self._stabilization_end = None
                    self.status = "RUNNING"
                    self._write_status()
                    log.info("Stabilization complete. Entries enabled.")
                else:
                    continue  # Suppress entries during stabilization

            if self.tracker.position == 0 and self.last_bar_price is not None:
                if entry_signals:
                    action = self.tracker.process_signals(
                        {**entry_signals, "exit_long": False, "exit_short": False},
                        self.last_bar_price,
                    )
                    if action:
                        log.info(f"Entry signal: {action} @ {self.last_bar_price:.2f}")

    async def _stream_live(self):
        """Connect to live WebSocket and stream bars."""
        config = self.config
        exchange = config["exchange"]
        symbol = config["symbol"]
        base_interval = config["base_interval_seconds"]

        if exchange == "bybit":
            if base_interval == 1:
                # CX: ticker stream with 1Hz bar clock
                await self.adapter.stream_ticker_1hz(
                    symbol, self._on_live_bar, self._stop_event
                )
            else:
                # Big: kline.1 stream (1 minute bars)
                await self.adapter.stream_kline_bars(
                    symbol, 1, self._on_live_bar, self._stop_event
                )
        elif exchange == "binance":
            # Big: 1m kline stream
            await self.adapter.stream_kline_bars(
                symbol, "1m", self._on_live_bar, self._stop_event
            )

    async def _status_loop(self):
        """Write status.json every 5 seconds."""
        while not self._shutdown:
            self._write_status()
            await asyncio.sleep(5)

    async def _command_loop(self):
        """Poll for stop commands every 2 seconds."""
        command_path = os.path.join(self.status_dir, "command.json")
        while not self._shutdown:
            try:
                if os.path.exists(command_path):
                    with open(command_path) as f:
                        cmd = json.load(f)
                    if cmd.get("command") == "stop":
                        log.info("Stop command received")
                        self._shutdown = True
                        os.remove(command_path)
                        return
            except (json.JSONDecodeError, IOError):
                pass
            await asyncio.sleep(2)

    def _build_status_dict(self) -> dict:
        """Build the shadow-specific status dict."""
        now = time.time()
        uptime = int(now - self.started_at)

        stab_remaining = None
        if self._stabilization_end is not None:
            stab_remaining = max(0, int(self._stabilization_end - now))

        return {
            "instance_id": self.instance_id,
            "exchange": self.config.get("exchange"),
            "strategy": self.config.get("strategy"),
            "status": self.status,
            "error": self.error,
            "warmup_progress": self.warmup_progress,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "started_at_utc": datetime.fromtimestamp(
                self.started_at, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "uptime_seconds": uptime,
            "uptime_str": format_uptime(uptime),
            "updated_at": now,
            "bars_received": self.bars_received,
            "last_bar_time": self.last_bar_time,
            "last_bar_price": self.last_bar_price,
            "gap_flag": self._gap_flag,
            "stabilization_remaining_s": stab_remaining,
            "tracker": self.tracker.snapshot(),
            "strategy_state": self.strategy.snapshot(),
        }

    def _write_status(self):
        """Write status.json atomically (shadow-specific and TUI-compatible)."""
        # Shadow-specific status
        status_path = os.path.join(self.status_dir, "status.json")
        tmp_path = status_path + ".tmp"
        try:
            os.makedirs(self.status_dir, exist_ok=True)
            with open(tmp_path, "w") as f:
                json.dump(self._build_status_dict(), f, indent=2)
            os.replace(tmp_path, status_path)
        except Exception as e:
            log.warning(f"Failed to write shadow status: {e}")

        # TUI-compatible PodStatus
        pod_id = f"shadow_{self.instance_id}"
        pod_path = os.path.join(self.pod_status_dir, "status.json")
        pod_tmp = pod_path + ".tmp"
        try:
            os.makedirs(self.pod_status_dir, exist_ok=True)
            is_active = self.status in (
                "RUNNING", "CONNECTING", "RECONNECTING", "WARMING_UP", "STABILIZING"
            )
            is_error = self.status == "ERROR"
            pod_status = {
                "pod_id": pod_id,
                "run_id": self.instance_id,
                "run_mode": "SHADOW",
                "state": "RUNNING" if is_active else self.status,
                "health": "UNHEALTHY" if is_error else "HEALTHY",
                "halt_reason": self.error,
                "bar_counter": self.bars_received,
                "last_bar_ts": self._last_bar_epoch_ts,
                "components": {
                    "websocket": {
                        "state": "HEALTHY" if self.status == "RUNNING" else self.status
                    },
                    "paper_tracker": {"state": "HEALTHY"},
                },
                "watermark": {},
                "config": {
                    "instance_id": self.instance_id,
                    "exchange": self.config.get("exchange"),
                    "strategy": self.config.get("strategy"),
                },
            }
            with open(pod_tmp, "w") as f:
                json.dump(pod_status, f, indent=2)
            os.replace(pod_tmp, pod_path)
        except Exception as e:
            log.warning(f"Failed to write pod status: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Shadow execution daemon")
    parser.add_argument("--instance-id", required=True,
                        help="Instance ID (e.g., bybit-cx, bybit-big, binance-big)")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(
        PROJECT_ROOT, "research", "shadow_status",
        args.instance_id, "config.json"
    )
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    # Setup logging
    log_dir = os.path.join(
        PROJECT_ROOT, "research", "shadow_status", args.instance_id
    )
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "daemon.log")),
            logging.StreamHandler(),
        ],
    )

    daemon = ShadowDaemon(args.instance_id, config)
    asyncio.run(daemon.run())


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Manual Verification (do NOT execute in automated tests)
# ---------------------------------------------------------------------------
#
# # 1. Start a single instance
# mkdir -p research/shadow_status/bybit-big/
# cat > research/shadow_status/bybit-big/config.json << 'EOF'
# {"instance_id":"bybit-big","exchange":"bybit","symbol":"BTCUSDT","category":"linear","strategy":"macd_big","base_interval_seconds":60,"timeframes":{"5m":300,"15m":900,"30m":1800,"1h":3600,"12h":43200,"1d":86400,"3d":259200},"roles":{"macro":["3d","1d","12h"],"intra":["1h","30m","15m"],"entry":"5m","exit":"1d"},"long_only":false,"macd_fast":12,"macd_slow":26,"round_trip_bps":25.0,"paper_qty":0.001,"stabilization_seconds":0}
# EOF
# python3 shadow_daemon.py --instance-id bybit-big &
#
# # 2. Watch warm-up progress
# sleep 5 && cat research/shadow_status/bybit-big/status.json | python3 -m json.tool
#
# # 3. Wait for RUNNING, verify bars incrementing
# sleep 120 && cat research/shadow_status/bybit-big/status.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['status'], d['bars_received'])"
#
# # 4. Stop
# echo '{"command":"stop"}' > research/shadow_status/bybit-big/command.json
# sleep 5 && cat research/shadow_status/bybit-big/status.json | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])"
# # Should print: STOPPED
#
# # 5. TUI visibility
# python3 -m tui
# # Fleet screen should show shadow_bybit-big
