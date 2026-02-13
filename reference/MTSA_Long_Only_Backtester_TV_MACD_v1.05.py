"""
MTSA Long-Only Backtester TV MACD v1.05
1D exit + 8H funding (CONSERVATIVE_ALWAYS_PAY_FUNDING_V1)
Deterministic, event-ledger output (+ optional plots)

Core invariants
- Entry evaluation: 5m close
- Exit evaluation: 1D close (EXIT_1D_TURN_DOWN)
- Mark-to-market proxy: 1m CLOSE (mark_price_1m_close)
- Liquidation trigger proxy (punitive intrabar extreme):
    long: use 1m LOW
    short: use 1m HIGH (implementation generic; strategy long-only)
- Liquidation fill price includes extra slippage bps
- Funding worst-case:
    * only while position open
    * always charged (never credited)
    * cadence: every 8 hours on a fixed Unix-epoch grid (UTC)
    * notional dynamic: abs(qty) * price_proxy at funding timestamp
    * price_proxy: mark_price_1m_close (1m close proxy)
- Funding is auditable in event ledger:
    * event_type = FUNDING_CHARGE rows
    * funding_paid_event column holds the charge for funding rows and 0.0 otherwise
    * realized_pnl_event on funding rows is negative (=-funding_paid_event)
    * position quantity does not change on funding rows (correct)
    * equity_after changes (correct)

Funding-aware exit overlay (FUNDING_AWARE_EXIT_V1)
- Controlled only via cfg["funding_exit_overlay"] in the canonical config JSON.
- Evaluated only on FUNDING_CHARGE events, after funding is applied.
- Tracks cumulative funding per trade, resets on each new entry.
- If triggered, exits immediately using the existing exit execution path
  (fees, slippage, accounting identical to normal exits).
- Emits a normal EXIT_FILL event with:
    exit_reason = FUNDING_OVERLAY
    signal_id = FUNDING_OVERLAY_EXIT

Trailing MTM stop overlay (TRAILING_MTM_STOP_V1)
- Optional, enabled via cfg["trailing_mtm_stop_overlay"] and/or sweep override.
- Evaluated on every 1m close while a position is open.
- Tracks peak position MTM since entry:
    position_mtm = initial_margin + unrealized_pnl - cumulative_entry_fee - cumulative_funding_paid
  Notes:
    * This is position-scoped MTM, not account equity.
    * Funding and entry fee are explicitly included in the MTM definition.
- Trigger condition at 1m close:
    position_mtm_close <= peak_position_mtm * (1 - d)
- Execution:
    * If triggered at 1m close i, schedule an exit at next 1m open (bar i+1 open price).
    * Exit fill uses next 1m open as decision price, then applies standard exit slippage and fee.
    * Scheduled trailing stop exit executes before any other logic at that next bar.

Gap handling
- 1m CSV is reindexed to a full 1m grid
- missing bars become synthetic flat bars with gap_flag_1m True
- gap flags propagate to higher timeframes via resample max(gap_flag_1m)
- trading is blocked when any relevant timeframe gap flag is True
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# -----------------------------
# Constants and identifiers
# -----------------------------

TF_TO_PANDAS = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1H": "1h",
    "12H": "12h",
    "1D": "1D",
    "3D": "3D",
}

BASE_DT_MS = 60_000

EVENT_TYPE_ENTRY = "ENTRY_FILL"
EVENT_TYPE_EXIT = "EXIT_FILL"
EVENT_TYPE_LIQ = "LIQUIDATION_FILL"
EVENT_TYPE_EOD = "END_OF_DATA_FORCE_CLOSE"
EVENT_TYPE_FUND = "FUNDING_CHARGE"

EXIT_REASON_DAILY = "DAILY_EXIT_SIGNAL"
EXIT_REASON_LIQ = "LIQUIDATION"
EXIT_REASON_EOD = "END_OF_DATA_FORCE_CLOSE"
EXIT_REASON_FUND_OVERLAY = "FUNDING_OVERLAY"
EXIT_REASON_TRAIL_STOP = "TRAILING_MTM_STOP"

SIGNAL_ID_ENTRY = "ENTRY_5M_TURN_UP"
SIGNAL_ID_EXIT = "EXIT_1D_TURN_DOWN"
SIGNAL_ID_LIQ = "LIQUIDATION"
SIGNAL_ID_EOD = "END_OF_DATA_FORCE_CLOSE"
SIGNAL_ID_FUND = "FUNDING_CHARGE"
SIGNAL_ID_FUND_OVERLAY_EXIT = "FUNDING_OVERLAY_EXIT"
SIGNAL_ID_TRAIL_STOP_EXIT = "TRAILING_MTM_STOP_EXIT"


# -----------------------------
# Utilities
# -----------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _tf_ms(tf: str) -> int:
    if tf not in TF_TO_PANDAS:
        raise ValueError(f"Unsupported timeframe: {tf}")
    rule = TF_TO_PANDAS[tf]
    return int(pd.Timedelta(rule).value // 1_000_000)


# -----------------------------
# EMA / MACD (TradingView-like: EMA with SMA seed)
# -----------------------------

def ema_sma_seed(src: np.ndarray, length: int) -> np.ndarray:
    """
    TradingView-like EMA:
    - alpha = 2/(N+1)
    - seed EMA at index N-1 with SMA(src[0..N-1])
    - recurrence from t >= N
    Returns array same length as src with NaN until index N-1 seeded.
    """
    out = np.full(len(src), np.nan, dtype=np.float64)
    if length <= 0 or len(src) < length:
        return out

    alpha = 2.0 / (length + 1.0)
    seed = float(np.mean(src[:length], dtype=np.float64))
    out[length - 1] = seed

    prev = seed
    for t in range(length, len(src)):
        prev = (src[t] - prev) * alpha + prev
        out[t] = prev
    return out


def macd_line(close: np.ndarray, fast: int, slow: int) -> np.ndarray:
    ef = ema_sma_seed(close, fast)
    es = ema_sma_seed(close, slow)
    return ef - es


def slope_sign(series: np.ndarray) -> np.ndarray:
    """
    sign(delta):
    +1 if delta > 0, -1 if delta < 0, 0 if delta == 0
    """
    out = np.full(len(series), np.nan, dtype=np.float64)
    if len(series) < 2:
        return out
    delta = series[1:] - series[:-1]
    s = np.zeros(len(delta), dtype=np.float64)
    s[delta > 0] = 1.0
    s[delta < 0] = -1.0
    out[1:] = s
    return out


# -----------------------------
# Data loading and gap handling (1m)
# -----------------------------

def load_csv_1m_forward_fill_with_gap_flags(
    csv_path: str,
    columns_order: List[str],
    timestamp_unit: str = "ms",
    candle_timestamp_represents: str = "open_time",
) -> pd.DataFrame:
    """
    Reads raw 1m CSV and returns 1m dataframe keyed by open_time.
    - Builds full 1m grid from min..max open_time in 60,000 ms steps
    - Missing bars are synthetic flat bars (O=H=L=C=carried close)
    - gap_flag_1m True for synthetic bars and first bar after missing interval
    - Adds liquidation trigger proxy series:
        liq_trigger_price_long = low
        liq_trigger_price_short = high
    - Index is dt (UTC datetime) derived from open_time (ms)
    """
    if timestamp_unit != "ms":
        raise ValueError(f"Only timestamp_unit=ms supported, got: {timestamp_unit}")

    df = pd.read_csv(csv_path)

    def norm(s: str) -> str:
        return str(s).strip().lower().replace(" ", "_")

    required_norm = [norm(c) for c in columns_order]
    df_cols_norm = [norm(c) for c in df.columns]

    if set(required_norm).issubset(set(df_cols_norm)):
        norm_to_orig = {norm(orig): orig for orig in df.columns}
        df = df[[norm_to_orig[nc] for nc in required_norm]].copy()
        df.columns = columns_order
    else:
        df2 = pd.read_csv(csv_path, header=None)
        if df2.shape[1] == len(columns_order):
            df = df2.copy()
            df.columns = columns_order
        else:
            raise ValueError(
                "CSV columns do not match expected schema.\n"
                f"Expected {len(columns_order)} columns: {columns_order}\n"
                f"Got header columns: {list(df.columns)} with count {df.shape[1]}\n"
                f"Headerless count: {df2.shape[1]}"
            )

    if candle_timestamp_represents not in ("open_time", "close_time"):
        raise ValueError(f"Unsupported candle_timestamp_represents: {candle_timestamp_represents}")

    df["open_time"] = pd.to_numeric(df["open_time"], errors="raise").astype(np.int64)
    df = df.sort_values("open_time").reset_index(drop=True)

    if candle_timestamp_represents == "close_time":
        df["open_time"] = df["open_time"] - BASE_DT_MS

    df["close_time"] = df["open_time"] + BASE_DT_MS

    dt_raw = df["open_time"].diff()
    raw_gap_after = dt_raw.ne(BASE_DT_MS) & dt_raw.notna()
    gap_rows = df.loc[raw_gap_after, "open_time"].to_numpy(dtype=np.int64)

    min_ot = int(df["open_time"].iloc[0])
    max_ot = int(df["open_time"].iloc[-1])
    full_open_time = np.arange(min_ot, max_ot + BASE_DT_MS, BASE_DT_MS, dtype=np.int64)
    full_index = pd.Index(full_open_time, name="open_time")

    df_obs = df.set_index("open_time")
    observed_mask = full_index.isin(df_obs.index)
    is_synth = ~observed_mask

    df_full = df_obs.reindex(full_index)
    df_full = df_full.ffill()

    df_full["open_time"] = df_full.index.astype(np.int64)
    df_full["close_time"] = df_full["open_time"] + BASE_DT_MS

    synth_mask = is_synth
    for c in ("open", "high", "low"):
        if c not in df_full.columns:
            raise ValueError(f"Missing required column after canonicalization: {c}")
    if "close" not in df_full.columns:
        raise ValueError("Missing required column after canonicalization: close")

    df_full.loc[synth_mask, "open"] = df_full.loc[synth_mask, "close"]
    df_full.loc[synth_mask, "high"] = df_full.loc[synth_mask, "close"]
    df_full.loc[synth_mask, "low"] = df_full.loc[synth_mask, "close"]

    zero_cols = [
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    for c in zero_cols:
        if c in df_full.columns:
            df_full.loc[synth_mask, c] = 0

    gap_flag = np.zeros(len(df_full), dtype=bool)
    gap_flag[synth_mask] = True

    for i in range(1, len(observed_mask)):
        if observed_mask[i] and (not observed_mask[i - 1]):
            gap_flag[i] = True

    if gap_rows.size > 0:
        pos = full_index.get_indexer(gap_rows)
        pos = pos[pos >= 0]
        gap_flag[pos] = True

    df_full["gap_flag_1m"] = gap_flag
    df_full["is_synthetic_1m"] = synth_mask

    df_full["liq_trigger_price_long"] = pd.to_numeric(df_full["low"], errors="coerce").astype(np.float64)
    df_full["liq_trigger_price_short"] = pd.to_numeric(df_full["high"], errors="coerce").astype(np.float64)

    df_full["dt"] = pd.to_datetime(df_full["open_time"].to_numpy(dtype=np.int64), unit="ms", utc=True)
    df_full = df_full.set_index("dt", drop=False)
    df_full.index.name = "dt"

    for c in ("open", "high", "low", "close"):
        df_full[c] = pd.to_numeric(df_full[c], errors="coerce").astype(np.float64)

    return df_full


def resample_ohlcv_with_gap_propagation(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Resample 1m to timeframe tf keyed by bar open_time.
    gap_flag True if any constituent 1m gap_flag_1m True.
    """
    if tf not in TF_TO_PANDAS:
        raise ValueError(f"Unsupported timeframe: {tf}")

    rule = TF_TO_PANDAS[tf]
    tf_ms = _tf_ms(tf)

    o = df_1m["open"].resample(rule, label="left", closed="left", origin="start_day").first()
    h = df_1m["high"].resample(rule, label="left", closed="left", origin="start_day").max()
    l = df_1m["low"].resample(rule, label="left", closed="left", origin="start_day").min()
    c = df_1m["close"].resample(rule, label="left", closed="left", origin="start_day").last()
    v = df_1m["volume"].resample(rule, label="left", closed="left", origin="start_day").sum() if "volume" in df_1m.columns else None
    gap = df_1m["gap_flag_1m"].resample(rule, label="left", closed="left", origin="start_day").max().astype(bool)

    data = {"open": o, "high": h, "low": l, "close": c, "gap_flag": gap}
    if v is not None:
        data["volume"] = v

    out = pd.DataFrame(data).dropna(subset=["close"])
    out["open_time"] = (out.index.view("int64") // 1_000_000).astype(np.int64)
    out["close_time"] = out["open_time"] + tf_ms
    out.index.name = "dt"
    return out


# -----------------------------
# Indicator packs per timeframe
# -----------------------------

def compute_macd_pack(df_tf: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    close = df_tf["close"].to_numpy(dtype=np.float64)
    macd = macd_line(close, fast=fast, slow=slow)
    sgn = slope_sign(macd)
    out = df_tf.copy()
    out["macd"] = macd
    out["slope_sign"] = sgn
    return out


def _asof_right_df(series_df: pd.DataFrame, col: str) -> pd.DataFrame:
    right = series_df[[col]].reset_index()
    if "dt" not in right.columns:
        right = right.rename(columns={right.columns[0]: "dt"})
    return right[["dt", col]]


def asof_lookup(series_df: pd.DataFrame, ts_dtindex: pd.DatetimeIndex, col: str) -> np.ndarray:
    left = pd.DataFrame({"dt": ts_dtindex})
    right = _asof_right_df(series_df, col)
    merged = pd.merge_asof(left.sort_values("dt"), right.sort_values("dt"), on="dt", direction="backward")
    return merged[col].to_numpy(dtype=np.float64)


def asof_lookup_bool(series_df: pd.DataFrame, ts_dtindex: pd.DatetimeIndex, col: str) -> np.ndarray:
    left = pd.DataFrame({"dt": ts_dtindex})
    right = _asof_right_df(series_df, col)
    merged = pd.merge_asof(left.sort_values("dt"), right.sort_values("dt"), on="dt", direction="backward")
    return merged[col].fillna(True).astype(bool).to_numpy()


# -----------------------------
# Funding model
# -----------------------------

@dataclass
class FundingParams:
    enabled: bool
    interval_ms: int
    rate_per_interval: float
    epoch_ms: int = 0


def load_funding_params(funding_json_path: Optional[str]) -> FundingParams:
    if not funding_json_path:
        return FundingParams(enabled=False, interval_ms=0, rate_per_interval=0.0, epoch_ms=0)

    with open(funding_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    fm = raw.get("funding_model", {})
    cadence = fm.get("cadence", {})
    inputs = fm.get("inputs", {})
    scope = fm.get("scope", {})

    interval_hours = int(cadence.get("interval_hours", 8))
    if interval_hours <= 0:
        raise ValueError(f"funding cadence interval_hours must be > 0, got: {interval_hours}")

    rate = float(inputs.get("assumed_funding_rate_per_interval", 0.0))
    if rate < 0:
        raise ValueError(f"funding assumed_funding_rate_per_interval must be >= 0, got: {rate}")

    funding_credit_allowed = bool(scope.get("funding_credit_allowed", False))
    if funding_credit_allowed:
        raise ValueError("Expected funding_credit_allowed=false for always-pay worst-case model.")

    interval_ms = int(interval_hours * 3600 * 1000)
    return FundingParams(enabled=True, interval_ms=interval_ms, rate_per_interval=rate, epoch_ms=0)


def is_funding_timestamp(ts_ms: int, fp: FundingParams) -> bool:
    if not fp.enabled or fp.interval_ms <= 0:
        return False
    return ((ts_ms - fp.epoch_ms) % fp.interval_ms) == 0


# -----------------------------
# Funding-aware exit overlay
# -----------------------------

@dataclass
class FundingExitOverlayParams:
    enabled: bool
    id: str
    funding_fraction_of_initial_margin: float
    require_unrealized_not_covering_funding: bool

    @staticmethod
    def disabled() -> "FundingExitOverlayParams":
        return FundingExitOverlayParams(
            enabled=False,
            id="",
            funding_fraction_of_initial_margin=0.0,
            require_unrealized_not_covering_funding=False,
        )


def load_funding_exit_overlay_params(cfg: dict) -> FundingExitOverlayParams:
    raw = cfg.get("funding_exit_overlay", None)
    if not isinstance(raw, dict):
        return FundingExitOverlayParams.disabled()

    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return FundingExitOverlayParams.disabled()

    overlay_id = str(raw.get("id", "FUNDING_AWARE_EXIT_V1"))

    thresholds = raw.get("thresholds", {}) if isinstance(raw.get("thresholds", {}), dict) else {}
    frac = float(thresholds.get("funding_fraction_of_initial_margin", 0.0))
    if frac < 0:
        raise ValueError(f"funding_exit_overlay.thresholds.funding_fraction_of_initial_margin must be >= 0, got: {frac}")

    req_unreal = bool(thresholds.get("require_unrealized_not_covering_funding", True))

    scope = raw.get("evaluation_scope", {}) if isinstance(raw.get("evaluation_scope", {}), dict) else {}
    evt = str(scope.get("evaluate_only_on_event_type", EVENT_TYPE_FUND))
    only_when_open = bool(scope.get("only_when_position_open", True))

    if evt != EVENT_TYPE_FUND:
        raise ValueError("funding_exit_overlay must evaluate only on event_type FUNDING_CHARGE.")
    if not only_when_open:
        raise ValueError("funding_exit_overlay must require only_when_position_open=true.")

    return FundingExitOverlayParams(
        enabled=True,
        id=overlay_id,
        funding_fraction_of_initial_margin=frac,
        require_unrealized_not_covering_funding=req_unreal,
    )


# -----------------------------
# Trailing MTM stop overlay
# -----------------------------

@dataclass
class TrailingMtmStopParams:
    enabled: bool
    id: str
    d: float

    @staticmethod
    def disabled() -> "TrailingMtmStopParams":
        return TrailingMtmStopParams(enabled=False, id="", d=0.0)


def _validate_trail_d(d: float) -> float:
    d0 = float(d)
    if not (0.0 < d0 < 1.0):
        raise ValueError(f"trailing stop d must be in (0,1), got: {d0}")
    return d0


def load_trailing_mtm_stop_params(
    cfg: dict,
    *,
    override_d: Optional[float] = None,
) -> TrailingMtmStopParams:
    """
    Authority and override rules:
    - If override_d is not None: enabled=True and d=override_d.
    - Else use top-level cfg["trailing_mtm_stop_overlay"] as baseline:
        enabled = overlay.enabled AND overlay.parameters.d is not None
    - Also allow single-mode overlay block (cfg["backtest_run_mode"]["single"]["params"]["overlays"]["trailing_mtm_stop"])
      to enable and provide d when present.
    """
    if override_d is not None:
        return TrailingMtmStopParams(enabled=True, id="TRAILING_MTM_STOP_V1", d=_validate_trail_d(override_d))

    overlay = cfg.get("trailing_mtm_stop_overlay", {})
    overlay_enabled = bool(overlay.get("enabled", False))
    overlay_id = str(overlay.get("id", "TRAILING_MTM_STOP_V1"))
    overlay_params = overlay.get("parameters", {}) if isinstance(overlay.get("parameters", {}), dict) else {}
    overlay_d = overlay_params.get("d", None)

    single_d = None
    single_enabled = False
    try:
        single = cfg.get("backtest_run_mode", {}).get("single", {})
        single_params = single.get("params", {})
        ov = single_params.get("overlays", {}).get("trailing_mtm_stop", {})
        single_enabled = bool(ov.get("enabled", False))
        single_d = ov.get("d", None)
    except Exception:
        single_enabled = False
        single_d = None

    enabled = overlay_enabled or single_enabled
    d_candidate = single_d if (single_d is not None) else overlay_d

    if not enabled:
        return TrailingMtmStopParams.disabled()
    if d_candidate is None:
        return TrailingMtmStopParams.disabled()

    return TrailingMtmStopParams(enabled=True, id=overlay_id, d=_validate_trail_d(float(d_candidate)))


# -----------------------------
# Backtest core
# -----------------------------

@dataclass
class RunParams:
    initial_equity: float
    fraction_of_equity: float
    leverage_x: float
    fee_rate_per_fill: float
    slippage_bps_per_fill: float
    liquidation_extra_slippage_bps: float
    trailing_stop_d: Optional[float] = None


@dataclass
class PendingExit:
    active: bool = False
    exec_ts_ms: int = 0
    exec_i_1m: int = 0
    exit_reason: str = ""
    signal_id: str = ""


@dataclass
class PositionState:
    is_open: bool = False
    trade_id: int = 0
    entry_ts_ms: int = 0
    entry_fill_price: float = 0.0
    position_qty: float = 0.0
    notional_entry: float = 0.0
    initial_margin: float = 0.0
    direction: int = 1
    cumulative_funding_paid: float = 0.0
    cumulative_entry_fee_paid: float = 0.0
    trail_enabled: bool = False
    trail_d: float = 0.0
    trail_peak_mtm: float = float("nan")


def apply_fee_and_slippage_entry(price: float, notional: float, fee_rate: float, slip_bps: float, direction: int) -> Tuple[float, float]:
    if direction == 1:
        fill_price = price * (1.0 + slip_bps / 10_000.0)
    elif direction == -1:
        fill_price = price * (1.0 - slip_bps / 10_000.0)
    else:
        raise ValueError("direction must be +1 or -1")
    fee = fee_rate * notional
    return float(fill_price), float(fee)


def apply_fee_and_slippage_exit(price: float, notional: float, fee_rate: float, slip_bps: float, direction: int) -> Tuple[float, float]:
    if direction == 1:
        fill_price = price * (1.0 - slip_bps / 10_000.0)
    elif direction == -1:
        fill_price = price * (1.0 + slip_bps / 10_000.0)
    else:
        raise ValueError("direction must be +1 or -1")
    fee = fee_rate * notional
    return float(fill_price), float(fee)


def build_event_row(
    *,
    event_id: int,
    timestamp_ms: int,
    event_type: str,
    signal_id: str,
    trade_id: Optional[int],
    price_exec: float,
    mark_price: float,
    position_qty_after: float,
    entry_fill_price_after: float,
    notional_after: float,
    leverage_x: float,
    fees_paid_event: float,
    funding_paid_event: float,
    slippage_bps_event: float,
    realized_pnl_event: float,
    unrealized_pnl_after: float,
    equity_after: float,
    equity_with_mtm_after: float,
    drawdown_after: float,
    bars_held: int,
    exit_reason: Optional[str],
    gap_flag: bool,
) -> Dict[str, object]:
    return {
        "event_id": int(event_id),
        "timestamp": int(timestamp_ms),
        "event_type": str(event_type),
        "signal_id": str(signal_id),
        "trade_id": int(trade_id) if trade_id is not None else np.nan,
        "price_exec": float(price_exec),
        "mark_price": float(mark_price),
        "position_qty_after": float(position_qty_after),
        "entry_fill_price_after": float(entry_fill_price_after) if not math.isnan(entry_fill_price_after) else np.nan,
        "notional_after": float(notional_after),
        "leverage_x": float(leverage_x),
        "fees_paid_event": float(fees_paid_event),
        "funding_paid_event": float(funding_paid_event),
        "slippage_bps_event": float(slippage_bps_event),
        "realized_pnl_event": float(realized_pnl_event),
        "unrealized_pnl_after": float(unrealized_pnl_after),
        "equity_after": float(equity_after),
        "equity_with_mtm_after": float(equity_with_mtm_after),
        "drawdown_after": float(drawdown_after),
        "bars_held": int(bars_held),
        "exit_reason": exit_reason if exit_reason is not None else np.nan,
        "gap_flag": bool(gap_flag),
    }


def run_backtest_single(
    *,
    df_1m: pd.DataFrame,
    tf_packs: Dict[str, pd.DataFrame],
    cfg: dict,
    run_params: RunParams,
    funding_params: FundingParams,
    out_dir: str,
    write_diagnostics: bool,
    plots_enabled: bool,
    macd_panels: bool,
    run_name: str = "",
) -> Dict[str, object]:
    _ensure_dir(out_dir)

    entry_tf = cfg["timeframes"]["entry_evaluation_tf"]
    exit_tf = cfg["timeframes"]["exit_evaluation_tf"]
    macro_tfs = list(cfg["timeframes"]["long_regime_sets"]["macro"])
    intra_tfs = list(cfg["timeframes"]["long_regime_sets"]["intraday"])

    mmr_fraction = float(cfg["leverage_and_liquidation"]["position_accounting"]["maintenance_margin"]["mmr_fraction_of_notional"])

    overlay_funding = load_funding_exit_overlay_params(cfg)
    overlay_trail = load_trailing_mtm_stop_params(cfg, override_d=run_params.trailing_stop_d)

    ledger_cols = list(cfg["outputs"]["event_ledger_schema"]["required_columns"])
    if "funding_paid_event" not in ledger_cols:
        ledger_cols.append("funding_paid_event")

    open_time_ms_1m = df_1m["open_time"].to_numpy(dtype=np.int64)
    close_time_ms_1m = df_1m["close_time"].to_numpy(dtype=np.int64)

    open_price_1m = df_1m["open"].to_numpy(dtype=np.float64)
    mark_price_1m_close = df_1m["close"].to_numpy(dtype=np.float64)

    liq_trigger_long_1m = df_1m["liq_trigger_price_long"].to_numpy(dtype=np.float64)
    liq_trigger_short_1m = df_1m["liq_trigger_price_short"].to_numpy(dtype=np.float64)
    gap_flag_1m = df_1m["gap_flag_1m"].to_numpy(dtype=bool)

    df_entry = tf_packs[entry_tf]
    df_exit = tf_packs[exit_tf]

    entry_ct_ms = df_entry["close_time"].to_numpy(dtype=np.int64)
    exit_ct_ms = df_exit["close_time"].to_numpy(dtype=np.int64)

    map_1m_ct_to_i = {int(ct): i for i, ct in enumerate(close_time_ms_1m)}
    map_entry_ct_to_i = {int(ct): i for i, ct in enumerate(entry_ct_ms)}
    map_exit_ct_to_i = {int(ct): i for i, ct in enumerate(exit_ct_ms)}

    is_entry_eval = np.zeros(len(df_1m), dtype=bool)
    is_exit_eval = np.zeros(len(df_1m), dtype=bool)
    for ct in entry_ct_ms:
        j = map_1m_ct_to_i.get(int(ct))
        if j is not None:
            is_entry_eval[j] = True
    for ct in exit_ct_ms:
        j = map_1m_ct_to_i.get(int(ct))
        if j is not None:
            is_exit_eval[j] = True

    required_tfs_for_regime = macro_tfs + intra_tfs
    slopes_at_entry: Dict[str, np.ndarray] = {}
    gaps_at_entry: Dict[str, np.ndarray] = {}
    eval_entry_dt = df_entry.index

    for tf in required_tfs_for_regime:
        slopes_at_entry[tf] = asof_lookup(tf_packs[tf], eval_entry_dt, "slope_sign")
        gaps_at_entry[tf] = asof_lookup_bool(tf_packs[tf], eval_entry_dt, "gap_flag")

    macd_entry = df_entry["macd"].to_numpy(dtype=np.float64)
    gap_entry = df_entry["gap_flag"].to_numpy(dtype=bool)

    macd_exit = df_exit["macd"].to_numpy(dtype=np.float64)
    gap_exit = df_exit["gap_flag"].to_numpy(dtype=bool)

    diag_rows: List[Dict[str, object]] = []

    cash_equity = float(run_params.initial_equity)
    pos = PositionState(is_open=False)
    pending = PendingExit(active=False)

    event_rows: List[Dict[str, object]] = []

    dense_equity_with_mtm = np.full(len(df_1m), np.nan, dtype=np.float64)
    dense_drawdown = np.full(len(df_1m), np.nan, dtype=np.float64)
    peak_eq = cash_equity

    def current_notional(price: float) -> float:
        if not pos.is_open:
            return 0.0
        return abs(pos.position_qty) * float(price)

    def current_unreal_and_eq_with_mtm(i_1m: int) -> Tuple[float, float]:
        if not pos.is_open:
            return 0.0, cash_equity
        mp0 = float(mark_price_1m_close[i_1m])
        unreal = (mp0 - pos.entry_fill_price) * pos.position_qty
        eq_with_mtm0 = cash_equity + pos.initial_margin + unreal
        return float(unreal), float(eq_with_mtm0)

    def current_position_mtm(i_1m: int) -> float:
        """
        Position-scoped MTM including entry fee and funding paid so far.
        """
        if not pos.is_open:
            return 0.0
        mp0 = float(mark_price_1m_close[i_1m])
        unreal = (mp0 - pos.entry_fill_price) * pos.position_qty
        mtm = pos.initial_margin + unreal - pos.cumulative_entry_fee_paid - pos.cumulative_funding_paid
        return float(mtm)

    def bars_held_since_entry(ts_ms: int) -> int:
        if not pos.is_open:
            return 0
        return int((ts_ms - pos.entry_ts_ms) // BASE_DT_MS)

    def liquidation_trigger_price(i_1m: int) -> float:
        if pos.direction == 1:
            return float(liq_trigger_long_1m[i_1m])
        return float(liq_trigger_short_1m[i_1m])

    def liquidation_fill_price(trigger_price: float) -> float:
        extra = float(run_params.liquidation_extra_slippage_bps) / 10_000.0
        if pos.direction == 1:
            return float(trigger_price * (1.0 - extra))
        return float(trigger_price * (1.0 + extra))

    def recompute_peak_and_dd(eq_with_mtm_now: float) -> float:
        nonlocal peak_eq
        if eq_with_mtm_now > peak_eq:
            peak_eq = eq_with_mtm_now
        if peak_eq <= 0:
            return 0.0
        return (peak_eq - eq_with_mtm_now) / peak_eq

    def do_exit_event(
        *,
        ts_ms: int,
        i_1m: int,
        mark_price: float,
        decision_price: float,
        event_type: str,
        signal_id: str,
        exit_reason: str,
        slippage_bps_event: float,
        gap_flag: bool,
    ) -> None:
        nonlocal cash_equity, pos, event_id, pending

        if not pos.is_open:
            pending.active = False
            return

        exit_notional_pre_fill = abs(pos.position_qty) * float(decision_price)
        fill_price, fee = apply_fee_and_slippage_exit(
            price=float(decision_price),
            notional=float(exit_notional_pre_fill),
            fee_rate=run_params.fee_rate_per_fill,
            slip_bps=run_params.slippage_bps_per_fill,
            direction=pos.direction,
        )

        realized = (fill_price - pos.entry_fill_price) * pos.position_qty
        cash_equity = cash_equity + pos.initial_margin + realized - fee

        trade_id_for_event = pos.trade_id
        bars_held = bars_held_since_entry(ts_ms)

        pos.is_open = False
        pending.active = False

        dd_now = recompute_peak_and_dd(cash_equity)

        event_id += 1
        event_rows.append(
            build_event_row(
                event_id=event_id,
                timestamp_ms=ts_ms,
                event_type=event_type,
                signal_id=signal_id,
                trade_id=trade_id_for_event,
                price_exec=fill_price,
                mark_price=mark_price,
                position_qty_after=0.0,
                entry_fill_price_after=float("nan"),
                notional_after=0.0,
                leverage_x=run_params.leverage_x,
                fees_paid_event=fee,
                funding_paid_event=0.0,
                slippage_bps_event=float(slippage_bps_event),
                realized_pnl_event=realized,
                unrealized_pnl_after=0.0,
                equity_after=cash_equity,
                equity_with_mtm_after=cash_equity,
                drawdown_after=dd_now,
                bars_held=bars_held,
                exit_reason=exit_reason,
                gap_flag=gap_flag,
            )
        )

    def overlay_funding_triggers(unrealized_pnl_after: float) -> bool:
        if not overlay_funding.enabled:
            return False
        if not pos.is_open:
            return False
        threshold = overlay_funding.funding_fraction_of_initial_margin * float(pos.initial_margin)
        if pos.cumulative_funding_paid < threshold:
            return False
        if overlay_funding.require_unrealized_not_covering_funding:
            return float(unrealized_pnl_after) <= float(pos.cumulative_funding_paid)
        return True

    def maybe_schedule_trailing_stop(i_1m: int) -> None:
        """
        Evaluate at 1m close i_1m and schedule an exit at next 1m open (i_1m+1).
        Scheduling is a one-shot: once scheduled, it stays until executed or position closes.
        """
        nonlocal pending

        if not pos.is_open:
            return
        if pending.active:
            return
        if not pos.trail_enabled:
            return

        mtm = current_position_mtm(i_1m)
        if math.isnan(pos.trail_peak_mtm):
            pos.trail_peak_mtm = mtm
        if mtm > pos.trail_peak_mtm:
            pos.trail_peak_mtm = mtm

        stop_mtm = pos.trail_peak_mtm * (1.0 - pos.trail_d)
        if mtm <= stop_mtm:
            nxt = i_1m + 1
            if nxt >= len(df_1m):
                return
            pending.active = True
            pending.exec_i_1m = nxt
            pending.exec_ts_ms = int(open_time_ms_1m[nxt])
            pending.exit_reason = EXIT_REASON_TRAIL_STOP
            pending.signal_id = SIGNAL_ID_TRAIL_STOP_EXIT

    event_id = 0
    trade_id_counter = 0

    for i in range(len(df_1m)):
        ts_close_ms = int(close_time_ms_1m[i])
        mp_close = float(mark_price_1m_close[i])
        gap_1m_now = bool(gap_flag_1m[i])

        # Execute pending trailing stop at this bar's open (which is timestamp open_time_ms_1m[i]).
        # This must happen before any other logic at this bar.
        if pending.active and pending.exec_i_1m == i and pos.is_open:
            ts_open_ms = int(open_time_ms_1m[i])
            open_px = float(open_price_1m[i])
            gap_open = bool(gap_flag_1m[i])
            do_exit_event(
                ts_ms=ts_open_ms,
                i_1m=i,
                mark_price=open_px,
                decision_price=open_px,
                event_type=EVENT_TYPE_EXIT,
                signal_id=pending.signal_id,
                exit_reason=pending.exit_reason,
                slippage_bps_event=float(run_params.slippage_bps_per_fill),
                gap_flag=gap_open,
            )
            unreal_now, eq_with_mtm_now = current_unreal_and_eq_with_mtm(i)
            dense_equity_with_mtm[i] = eq_with_mtm_now
            dense_drawdown[i] = recompute_peak_and_dd(eq_with_mtm_now)
            continue

        # Funding at 8h timestamps on 1m close-time grid
        if pos.is_open and is_funding_timestamp(ts_close_ms, funding_params):
            notional_now = current_notional(mp_close)
            funding_cost = float(notional_now * funding_params.rate_per_interval)
            if funding_cost < 0:
                raise ValueError("Funding cost must be non-negative under always-pay model.")

            cash_equity -= funding_cost
            pos.cumulative_funding_paid += funding_cost

            unreal_now, eq_with_mtm_now = current_unreal_and_eq_with_mtm(i)
            dd_now = recompute_peak_and_dd(eq_with_mtm_now)

            event_id += 1
            event_rows.append(
                build_event_row(
                    event_id=event_id,
                    timestamp_ms=ts_close_ms,
                    event_type=EVENT_TYPE_FUND,
                    signal_id=SIGNAL_ID_FUND,
                    trade_id=pos.trade_id,
                    price_exec=mp_close,
                    mark_price=mp_close,
                    position_qty_after=pos.position_qty,
                    entry_fill_price_after=pos.entry_fill_price,
                    notional_after=current_notional(mp_close),
                    leverage_x=run_params.leverage_x,
                    fees_paid_event=0.0,
                    funding_paid_event=funding_cost,
                    slippage_bps_event=0.0,
                    realized_pnl_event=-funding_cost,
                    unrealized_pnl_after=unreal_now,
                    equity_after=cash_equity,
                    equity_with_mtm_after=eq_with_mtm_now,
                    drawdown_after=dd_now,
                    bars_held=bars_held_since_entry(ts_close_ms),
                    exit_reason=None,
                    gap_flag=gap_1m_now,
                )
            )

            if pos.is_open and overlay_funding_triggers(unreal_now):
                do_exit_event(
                    ts_ms=ts_close_ms,
                    i_1m=i,
                    mark_price=mp_close,
                    decision_price=mp_close,
                    event_type=EVENT_TYPE_EXIT,
                    signal_id=SIGNAL_ID_FUND_OVERLAY_EXIT,
                    exit_reason=EXIT_REASON_FUND_OVERLAY,
                    slippage_bps_event=float(run_params.slippage_bps_per_fill),
                    gap_flag=gap_1m_now,
                )
                unreal_now, eq_with_mtm_now = current_unreal_and_eq_with_mtm(i)
                dense_equity_with_mtm[i] = eq_with_mtm_now
                dense_drawdown[i] = recompute_peak_and_dd(eq_with_mtm_now)
                continue

        unreal_now, eq_with_mtm_now = current_unreal_and_eq_with_mtm(i)
        dd = recompute_peak_and_dd(eq_with_mtm_now)
        dense_equity_with_mtm[i] = eq_with_mtm_now
        dense_drawdown[i] = dd

        # Liquidation check (punitive intrabar extreme)
        if pos.is_open and (run_params.leverage_x > 1.0):
            liq_price = liquidation_trigger_price(i)

            trigger_notional = current_notional(liq_price)
            maintenance_margin = mmr_fraction * trigger_notional

            equity_in_position_at_liq = pos.initial_margin + (liq_price - pos.entry_fill_price) * pos.position_qty

            if equity_in_position_at_liq <= maintenance_margin:
                exit_price = liquidation_fill_price(liq_price)
                exit_notional = current_notional(exit_price)

                fee = run_params.fee_rate_per_fill * exit_notional if run_params.fee_rate_per_fill > 0 else 0.0
                realized = (exit_price - pos.entry_fill_price) * pos.position_qty

                cash_equity = cash_equity + pos.initial_margin + realized - fee

                trade_id_for_event = pos.trade_id
                bars_held = bars_held_since_entry(ts_close_ms)

                pos.is_open = False
                pending.active = False

                dd_now = recompute_peak_and_dd(cash_equity)

                event_id += 1
                event_rows.append(
                    build_event_row(
                        event_id=event_id,
                        timestamp_ms=ts_close_ms,
                        event_type=EVENT_TYPE_LIQ,
                        signal_id=SIGNAL_ID_LIQ,
                        trade_id=trade_id_for_event,
                        price_exec=exit_price,
                        mark_price=mp_close,
                        position_qty_after=0.0,
                        entry_fill_price_after=float("nan"),
                        notional_after=0.0,
                        leverage_x=run_params.leverage_x,
                        fees_paid_event=fee,
                        funding_paid_event=0.0,
                        slippage_bps_event=float(run_params.liquidation_extra_slippage_bps),
                        realized_pnl_event=realized,
                        unrealized_pnl_after=0.0,
                        equity_after=cash_equity,
                        equity_with_mtm_after=cash_equity,
                        drawdown_after=dd_now,
                        bars_held=bars_held,
                        exit_reason=EXIT_REASON_LIQ,
                        gap_flag=gap_1m_now,
                    )
                )
                continue

        # Trailing stop evaluation at 1m close (schedules exit at next 1m open)
        if pos.is_open and overlay_trail.enabled and pos.trail_enabled:
            maybe_schedule_trailing_stop(i)

        # Timeframe-based exit evaluation (1D close)
        if is_exit_eval[i] and pos.is_open:
            idx_d = map_exit_ct_to_i.get(ts_close_ms)
            if idx_d is not None:
                if bool(gap_exit[idx_d]) or gap_1m_now:
                    exit_allowed = False
                else:
                    if idx_d < 1 or np.isnan(macd_exit[idx_d]) or np.isnan(macd_exit[idx_d - 1]):
                        exit_allowed = False
                    else:
                        delta = macd_exit[idx_d] - macd_exit[idx_d - 1]
                        exit_allowed = bool(delta < 0)

                if exit_allowed:
                    price_close = float(df_exit.iloc[idx_d]["close"])
                    do_exit_event(
                        ts_ms=ts_close_ms,
                        i_1m=i,
                        mark_price=mp_close,
                        decision_price=price_close,
                        event_type=EVENT_TYPE_EXIT,
                        signal_id=SIGNAL_ID_EXIT,
                        exit_reason=EXIT_REASON_DAILY,
                        slippage_bps_event=float(run_params.slippage_bps_per_fill),
                        gap_flag=gap_1m_now,
                    )

        # Entry evaluation (5m close)
        if is_entry_eval[i] and (not pos.is_open):
            idx_5 = map_entry_ct_to_i.get(ts_close_ms)
            if idx_5 is not None:
                warm_ok = (
                    idx_5 >= 2
                    and (not np.isnan(macd_entry[idx_5]))
                    and (not np.isnan(macd_entry[idx_5 - 1]))
                    and (not np.isnan(macd_entry[idx_5 - 2]))
                )

                macro_ok = True
                intra_ok = True
                gap_any_tf = False

                for tf in macro_tfs:
                    s = slopes_at_entry[tf][idx_5]
                    g = bool(gaps_at_entry[tf][idx_5])
                    if g:
                        gap_any_tf = True
                    if np.isnan(s) or (s <= 0):
                        macro_ok = False

                for tf in intra_tfs:
                    s = slopes_at_entry[tf][idx_5]
                    g = bool(gaps_at_entry[tf][idx_5])
                    if g:
                        gap_any_tf = True
                    if np.isnan(s) or (s <= 0):
                        intra_ok = False

                if bool(gap_entry[idx_5]) or gap_any_tf or gap_1m_now:
                    regime_ok = False
                else:
                    regime_ok = macro_ok and intra_ok

                if warm_ok:
                    prev_delta = macd_entry[idx_5 - 1] - macd_entry[idx_5 - 2]
                    curr_delta = macd_entry[idx_5] - macd_entry[idx_5 - 1]
                    turn_up = (curr_delta > 0) and (prev_delta <= 0)
                else:
                    turn_up = False

                entry_allowed = bool(regime_ok and turn_up)

                if write_diagnostics:
                    diag_rows.append(
                        {
                            "timestamp_5m": int(ts_close_ms),
                            "regime_ok_macro": bool(macro_ok),
                            "regime_ok_intraday": bool(intra_ok),
                            "slope_3D": float(slopes_at_entry.get("3D", np.array([np.nan]))[idx_5]) if "3D" in slopes_at_entry else np.nan,
                            "slope_1D": float(slopes_at_entry.get("1D", np.array([np.nan]))[idx_5]) if "1D" in slopes_at_entry else np.nan,
                            "slope_12H": float(slopes_at_entry.get("12H", np.array([np.nan]))[idx_5]) if "12H" in slopes_at_entry else np.nan,
                            "slope_1H": float(slopes_at_entry.get("1H", np.array([np.nan]))[idx_5]) if "1H" in slopes_at_entry else np.nan,
                            "slope_30m": float(slopes_at_entry.get("30m", np.array([np.nan]))[idx_5]) if "30m" in slopes_at_entry else np.nan,
                            "slope_15m": float(slopes_at_entry.get("15m", np.array([np.nan]))[idx_5]) if "15m" in slopes_at_entry else np.nan,
                            "turn_up_5m": bool(turn_up),
                            "entry_allowed": bool(entry_allowed),
                            "exit_allowed": False,
                            "gap_flag_any_tf": bool(gap_any_tf or gap_entry[idx_5] or gap_1m_now),
                        }
                    )

                if entry_allowed:
                    price_5m_close = float(df_entry.iloc[idx_5]["close"])
                    leverage_x = float(run_params.leverage_x)

                    notional = cash_equity * float(run_params.fraction_of_equity) * leverage_x
                    if notional <= 0:
                        continue

                    direction = 1

                    fill_price, fee = apply_fee_and_slippage_entry(
                        price=price_5m_close,
                        notional=notional,
                        fee_rate=run_params.fee_rate_per_fill,
                        slip_bps=run_params.slippage_bps_per_fill,
                        direction=direction,
                    )

                    qty = direction * (notional / fill_price)
                    initial_margin = notional / leverage_x

                    cash_equity = cash_equity - initial_margin - fee

                    trade_id_counter += 1
                    pos.is_open = True
                    pos.trade_id = trade_id_counter
                    pos.entry_ts_ms = ts_close_ms
                    pos.entry_fill_price = fill_price
                    pos.position_qty = qty
                    pos.notional_entry = notional
                    pos.initial_margin = initial_margin
                    pos.direction = direction
                    pos.cumulative_funding_paid = 0.0
                    pos.cumulative_entry_fee_paid = float(fee)

                    pos.trail_enabled = bool(overlay_trail.enabled)
                    pos.trail_d = float(overlay_trail.d) if overlay_trail.enabled else 0.0
                    pos.trail_peak_mtm = float("nan")

                    pending.active = False

                    _, eq_with_mtm_after_entry = current_unreal_and_eq_with_mtm(i)
                    dd_now = recompute_peak_and_dd(eq_with_mtm_after_entry)

                    event_id += 1
                    event_rows.append(
                        build_event_row(
                            event_id=event_id,
                            timestamp_ms=ts_close_ms,
                            event_type=EVENT_TYPE_ENTRY,
                            signal_id=SIGNAL_ID_ENTRY,
                            trade_id=pos.trade_id,
                            price_exec=fill_price,
                            mark_price=mp_close,
                            position_qty_after=qty,
                            entry_fill_price_after=fill_price,
                            notional_after=notional,
                            leverage_x=run_params.leverage_x,
                            fees_paid_event=fee,
                            funding_paid_event=0.0,
                            slippage_bps_event=float(run_params.slippage_bps_per_fill),
                            realized_pnl_event=0.0,
                            unrealized_pnl_after=0.0,
                            equity_after=cash_equity,
                            equity_with_mtm_after=eq_with_mtm_after_entry,
                            drawdown_after=dd_now,
                            bars_held=0,
                            exit_reason=None,
                            gap_flag=gap_1m_now,
                        )
                    )

    # End of data force close
    last_i = len(df_1m) - 1
    last_ts_close_ms = int(close_time_ms_1m[last_i])
    last_mp_close = float(mark_price_1m_close[last_i])
    last_gap = bool(gap_flag_1m[last_i])

    if pos.is_open:
        do_exit_event(
            ts_ms=last_ts_close_ms,
            i_1m=last_i,
            mark_price=last_mp_close,
            decision_price=float(mark_price_1m_close[last_i]),
            event_type=EVENT_TYPE_EOD,
            signal_id=SIGNAL_ID_EOD,
            exit_reason=EXIT_REASON_EOD,
            slippage_bps_event=float(run_params.slippage_bps_per_fill),
            gap_flag=last_gap,
        )

    ledger_path = os.path.join(out_dir, cfg["outputs"]["event_ledger_filename"])
    if len(event_rows) == 0:
        ledger = pd.DataFrame(columns=ledger_cols)
    else:
        ledger = pd.DataFrame(event_rows).sort_values(["timestamp", "event_id"]).reset_index(drop=True)

        for c in ledger_cols:
            if c not in ledger.columns:
                if c in (
                    "fees_paid_event",
                    "funding_paid_event",
                    "slippage_bps_event",
                    "realized_pnl_event",
                    "unrealized_pnl_after",
                    "equity_after",
                    "equity_with_mtm_after",
                    "drawdown_after",
                    "price_exec",
                    "mark_price",
                    "notional_after",
                    "position_qty_after",
                    "leverage_x",
                ):
                    ledger[c] = 0.0
                else:
                    ledger[c] = np.nan

        ledger = ledger[ledger_cols]

    ledger.to_csv(ledger_path, index=False)
    if isinstance(run_name, str) and run_name.strip():
        ledger_path_tagged = os.path.join(out_dir, f"event_ledger_{run_name}.csv")
        ledger.to_csv(ledger_path_tagged, index=False)

    if write_diagnostics:
        diag_cols = cfg["outputs"]["diagnostics_schema"]["required_columns"]
        diag = pd.DataFrame(diag_rows)
        if not diag.empty:
            for c in diag_cols:
                if c not in diag.columns:
                    diag[c] = np.nan
            diag = diag[diag_cols]
            diag.to_csv(os.path.join(out_dir, "diagnostics.csv"), index=False)

    if plots_enabled and (plt is not None):
        plot_dir = os.path.join(out_dir, "plots")
        _ensure_dir(plot_dir)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(close_time_ms_1m, mark_price_1m_close, linewidth=0.8)
        ax.set_title("Price (1m close MTM) with Events")
        ax.set_xlabel("close_time (ms UTC)")
        ax.set_ylabel("price")

        if len(event_rows) > 0:
            for _, r in ledger.iterrows():
                ts = int(r["timestamp"])
                y = float(r["mark_price"])
                et = str(r["event_type"])
                if et == EVENT_TYPE_ENTRY:
                    ax.scatter([ts], [y], marker="^")
                elif et == EVENT_TYPE_EXIT:
                    ax.scatter([ts], [y], marker="v")
                elif et == EVENT_TYPE_LIQ:
                    ax.scatter([ts], [y], marker="x")
                elif et == EVENT_TYPE_EOD:
                    ax.scatter([ts], [y], marker="s")
                elif et == EVENT_TYPE_FUND:
                    ax.scatter([ts], [y], marker=".")

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "price_with_events.png"), dpi=150)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(close_time_ms_1m, dense_equity_with_mtm, linewidth=0.8)
        ax.set_title("Equity with MTM (1m close sampling)")
        ax.set_xlabel("close_time (ms UTC)")
        ax.set_ylabel("equity_with_mtm")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "equity_with_mtm.png"), dpi=150)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(close_time_ms_1m, dense_drawdown, linewidth=0.8)
        ax.set_title("Drawdown (fraction)")
        ax.set_xlabel("close_time (ms UTC)")
        ax.set_ylabel("drawdown")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "drawdown.png"), dpi=150)
        plt.close(fig)

        if macd_panels:
            panels_cfg = cfg["outputs"]["plots"].get("macd_panels_optional", {})
            tfs = panels_cfg.get("timeframes", []) if panels_cfg.get("enabled", False) else []
            for tf in tfs:
                if tf not in tf_packs:
                    continue
                dft = tf_packs[tf]
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(
                    dft["close_time"].to_numpy(dtype=np.int64),
                    dft["macd"].to_numpy(dtype=np.float64),
                    linewidth=0.8,
                )
                ax.set_title(f"MACD line ({tf})")
                ax.set_xlabel("close_time (ms UTC)")
                ax.set_ylabel("MACD")
                fig.tight_layout()
                fig.savefig(os.path.join(plot_dir, f"macd_{tf}.png"), dpi=150)
                plt.close(fig)

    final_eq = float(dense_equity_with_mtm[-1]) if len(dense_equity_with_mtm) else float("nan")
    max_dd = float(np.nanmax(dense_drawdown)) if len(dense_drawdown) else float("nan")

    return {
        "out_dir": out_dir,
        "event_ledger_path": ledger_path,
        "final_equity_with_mtm": final_eq,
        "max_drawdown": max_dd,
        "num_events": int(len(event_rows)),
    }


# -----------------------------
# Config and run-mode
# -----------------------------

def load_config(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["entry"]["signal_requirement"]["id"] = SIGNAL_ID_ENTRY
    cfg["exit"]["rule"]["id"] = SIGNAL_ID_EXIT
    if "leverage_and_liquidation" in cfg and "position_accounting" in cfg["leverage_and_liquidation"]:
        cfg["leverage_and_liquidation"]["position_accounting"]["liquidation_rule_long"]["id"] = SIGNAL_ID_LIQ

    defaults = cfg.get("fees_slippage", {}).get("defaults", {})
    if "liquidation_extra_slippage_bps" not in defaults:
        defaults["liquidation_extra_slippage_bps"] = 25.0
        cfg.setdefault("fees_slippage", {}).setdefault("defaults", defaults)

    _ = load_funding_exit_overlay_params(cfg)
    _ = load_trailing_mtm_stop_params(cfg)

    return cfg


def build_timeframes_packs(df_1m: pd.DataFrame, cfg: dict) -> Dict[str, pd.DataFrame]:
    fast = int(cfg["indicator"]["params"]["fast"])
    slow = int(cfg["indicator"]["params"]["slow"])

    tfs_needed = set()
    tfs_needed.add(cfg["timeframes"]["entry_evaluation_tf"])
    tfs_needed.add(cfg["timeframes"]["exit_evaluation_tf"])
    tfs_needed.update(cfg["timeframes"]["long_regime_sets"]["macro"])
    tfs_needed.update(cfg["timeframes"]["long_regime_sets"]["intraday"])

    packs: Dict[str, pd.DataFrame] = {}
    for tf in sorted(tfs_needed):
        df_tf = resample_ohlcv_with_gap_propagation(df_1m, tf)
        df_tf = compute_macd_pack(df_tf, fast=fast, slow=slow)
        packs[tf] = df_tf
    return packs


def write_resolved_config(cfg: dict, out_dir: str) -> None:
    if bool(cfg["outputs"].get("write_resolved_config_json", False)):
        _ensure_dir(out_dir)
        with open(os.path.join(out_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)


def make_run_params_from_cfg(
    cfg: dict,
    *,
    leverage_x: float,
    fee_rate: float,
    slippage_bps: float,
    fraction_of_equity: Optional[float] = None,
    liq_extra_bps: Optional[float] = None,
    trailing_stop_d: Optional[float] = None,
) -> RunParams:
    single = cfg["backtest_run_mode"]["single"]
    params = single["params"]
    initial_equity = float(params["initial_equity_quote"])

    if fraction_of_equity is None:
        frac = float(params["position_sizing"]["fraction_of_equity"])
    else:
        frac = float(fraction_of_equity)

    if liq_extra_bps is None:
        liq_extra_bps = float(cfg["fees_slippage"]["defaults"].get("liquidation_extra_slippage_bps", 25.0))

    return RunParams(
        initial_equity=initial_equity,
        fraction_of_equity=frac,
        leverage_x=float(leverage_x),
        fee_rate_per_fill=float(fee_rate),
        slippage_bps_per_fill=float(slippage_bps),
        liquidation_extra_slippage_bps=float(liq_extra_bps),
        trailing_stop_d=trailing_stop_d,
    )


def _format_d_token(d: Optional[float]) -> str:
    if d is None:
        return "trailDna"
    return f"trailD{float(d):g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to canonical JSON config")
    ap.add_argument("--csv", required=True, help="Path to 1m BTC CSV")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--funding_json", default="", help="Path to funding_model JSON (optional)")
    ap.add_argument("--liquidation_extra_slippage_bps", type=float, default=None, help="Override liquidation extra slippage bps")
    args = ap.parse_args()

    cfg = load_config(args.config)
    funding_params = load_funding_params(args.funding_json if args.funding_json else None)

    df_1m = load_csv_1m_forward_fill_with_gap_flags(
        csv_path=args.csv,
        columns_order=cfg["data"]["csv_columns_order"],
        timestamp_unit=cfg["data"]["timestamp_unit"],
        candle_timestamp_represents=cfg["data"]["candle_timestamp_represents"],
    )

    tf_packs = build_timeframes_packs(df_1m, cfg)

    base_out = cfg["outputs"]["directory"] if cfg["outputs"].get("directory") else args.out
    base_out = os.path.abspath(base_out)
    _ensure_dir(base_out)

    write_resolved_config(cfg, base_out)

    plots_enabled = bool(cfg["outputs"]["plots"].get("enabled", False))
    write_diagnostics = bool(cfg["outputs"].get("write_diagnostics_csv", False))
    macd_panels = bool(cfg["outputs"]["plots"].get("macd_panels_optional", {}).get("enabled", False))

    single_enabled = bool(cfg["backtest_run_mode"]["single"].get("enabled", False))
    sweep_enabled = bool(cfg["backtest_run_mode"]["sweep"].get("enabled", False))

    if single_enabled and (not sweep_enabled):
        defaults = cfg["fees_slippage"]["defaults"]
        fee = float(defaults["fee_rate_per_fill"])
        slip = float(defaults["slippage_bps_per_fill"])
        lev_cfg = cfg["backtest_run_mode"]["single"]["params"]["leverage"]
        leverage_x = float(lev_cfg.get("leverage_x", 1.0))
        frac = float(cfg["backtest_run_mode"]["single"]["params"]["position_sizing"]["fraction_of_equity"])

        # Single-mode overlay selection
        single_ov = cfg["backtest_run_mode"]["single"]["params"].get("overlays", {}).get("trailing_mtm_stop", {})
        d_single = single_ov.get("d", None) if bool(single_ov.get("enabled", False)) else None
        if d_single is None:
            top = cfg.get("trailing_mtm_stop_overlay", {})
            if bool(top.get("enabled", False)):
                d_single = top.get("parameters", {}).get("d", None)

        rp = make_run_params_from_cfg(
            cfg,
            leverage_x=leverage_x,
            fee_rate=fee,
            slippage_bps=slip,
            fraction_of_equity=frac,
            liq_extra_bps=args.liquidation_extra_slippage_bps,
            trailing_stop_d=None if d_single is None else float(d_single),
        )

        liq_used = (
            args.liquidation_extra_slippage_bps
            if args.liquidation_extra_slippage_bps is not None
            else cfg["fees_slippage"]["defaults"].get("liquidation_extra_slippage_bps", 25.0)
        )
        run_name = f"single_lev{leverage_x:g}_frac{rp.fraction_of_equity:g}_fee{fee:g}_slip{slip:g}_liq{liq_used:g}_{_format_d_token(rp.trailing_stop_d)}"

        res = run_backtest_single(
            df_1m=df_1m,
            tf_packs=tf_packs,
            cfg=cfg,
            run_params=rp,
            funding_params=funding_params,
            out_dir=base_out,
            write_diagnostics=write_diagnostics,
            plots_enabled=plots_enabled,
            macd_panels=macd_panels,
            run_name=run_name,
        )
        print(json.dumps(res, indent=2))
        return

    if sweep_enabled:
        grid = cfg["backtest_run_mode"]["sweep"]["grid"]

        fracs = [float(x) for x in grid.get("fraction_of_equity", [])]
        if len(fracs) == 0:
            raise ValueError("sweep.grid.fraction_of_equity is required.")

        levs = [float(x) for x in grid.get("leverage_x", [])]
        fees = [float(x) for x in grid.get("fee_rate_per_fill", [])]
        slips = [float(x) for x in grid.get("slippage_bps_per_fill", [])]

        if len(levs) == 0 or len(fees) == 0 or len(slips) == 0:
            raise ValueError("sweep.grid requires leverage_x, fee_rate_per_fill, slippage_bps_per_fill.")

        trail_ds_raw = grid.get("trailing_stop_d", [None])
        trail_ds: List[Optional[float]] = []
        for v in trail_ds_raw:
            if v is None:
                trail_ds.append(None)
            else:
                trail_ds.append(float(v))

        liq_extras = [float(cfg["fees_slippage"]["defaults"].get("liquidation_extra_slippage_bps", 25.0))]
        if "liquidation_extra_slippage_bps" in grid:
            liq_extras = [float(x) for x in grid["liquidation_extra_slippage_bps"]]

        max_runs = int(cfg["backtest_run_mode"]["sweep"]["constraints"]["max_total_runs"])
        combos = list(product(fracs, levs, fees, slips, liq_extras, trail_ds))
        if len(combos) > max_runs:
            combos = combos[:max_runs]

        results = []
        for frac, leverage_x, fee, slip, liq_extra, trail_d in combos:
            rp = make_run_params_from_cfg(
                cfg,
                leverage_x=leverage_x,
                fee_rate=fee,
                slippage_bps=slip,
                fraction_of_equity=frac,
                liq_extra_bps=liq_extra,
                trailing_stop_d=trail_d,
            )
            run_name = f"lev{leverage_x:g}_frac{rp.fraction_of_equity:g}_fee{fee:g}_slip{slip:g}_liq{liq_extra:g}_{_format_d_token(trail_d)}"
            out_dir = os.path.join(base_out, run_name)
            _ensure_dir(out_dir)
            write_resolved_config(cfg, out_dir)

            res = run_backtest_single(
                df_1m=df_1m,
                tf_packs=tf_packs,
                cfg=cfg,
                run_params=rp,
                funding_params=funding_params,
                out_dir=out_dir,
                write_diagnostics=write_diagnostics,
                plots_enabled=plots_enabled,
                macd_panels=macd_panels,
                run_name=run_name,
            )

            results.append(
                {
                    "run_name": run_name,
                    "fraction_of_equity": rp.fraction_of_equity,
                    "leverage_x": leverage_x,
                    "fee_rate_per_fill": fee,
                    "slippage_bps_per_fill": slip,
                    "liquidation_extra_slippage_bps": liq_extra,
                    "trailing_stop_d": trail_d,
                    "final_equity_with_mtm": res["final_equity_with_mtm"],
                    "max_drawdown": res["max_drawdown"],
                    "num_events": res["num_events"],
                    "event_ledger_path": res["event_ledger_path"],
                }
            )

        results_df = pd.DataFrame(results)
        agg_name = cfg["backtest_run_mode"]["sweep"]["output"].get("aggregate_results_csv", "sweep_results.csv")
        results_df.to_csv(os.path.join(base_out, agg_name), index=False)
        print(results_df.sort_values("final_equity_with_mtm", ascending=False).head(50).to_string(index=False))
        return

    raise ValueError("Invalid run mode: enable either single or sweep in config.")


if __name__ == "__main__":
    main()
