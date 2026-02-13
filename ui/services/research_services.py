"""Research Services — named functions that UI buttons call.

Every UI action (Run Triage, Write Promotion, Run Sweep) calls a named
function in this module. Integration tests import and call these same
functions, ensuring the test exercises the same code path as the UI.
"""

import copy
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from phase5_triage_types import TradeEvent

from ui.services.backtest_runner import Bar, run_backtest
from ui.services.compiler_bridge import compile_spec, save_artifacts
from ui.services.triage_v2 import run_triage_v2, TriageV2Result

RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "research")
HISTORIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "historic_data")

# Friction defaults
DEFAULT_FEE_RATE = 0.0006          # 6 bps per fill (Bybit taker)
DEFAULT_SLIPPAGE_BPS = 10          # 10 bps per fill
DEFAULT_STARTING_CAPITAL = 10000   # USD


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def list_datasets() -> List[Dict[str, Any]]:
    """List available .parquet datasets with parsed metadata."""
    if not os.path.isdir(HISTORIC_DIR):
        return []

    datasets = []
    pattern = re.compile(r'^.+_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.parquet$')

    for fn in sorted(os.listdir(HISTORIC_DIR)):
        if not fn.endswith('.parquet'):
            continue
        path = os.path.join(HISTORIC_DIR, fn)
        prefix = fn[:-len('.parquet')]
        m = pattern.match(fn)
        if m:
            start_date = m.group(1)
            end_date = m.group(2)
            # Compute approx months
            from datetime import date
            d1 = date.fromisoformat(start_date)
            d2 = date.fromisoformat(end_date)
            days = (d2 - d1).days
            months = round(days / 30)
            label = f"{months}mo ({start_date} to {end_date})"
        else:
            start_date = None
            end_date = None
            days = 0
            label = fn

        datasets.append({
            "path": path,
            "filename": fn,
            "prefix": prefix,
            "start_date": start_date,
            "end_date": end_date,
            "days": days,
            "label": label,
        })

    # Sort by date range length (shortest first)
    datasets.sort(key=lambda d: d["days"])
    return datasets


def load_bars_from_parquet(path: str) -> List[Bar]:
    """Load 1m bars from parquet file."""
    import pandas as pd
    df = pd.read_parquet(path)
    bars = []
    for i, row in enumerate(df.itertuples()):
        ts = int(row.timestamp) if hasattr(row, "timestamp") else int(row.ts)
        bars.append(Bar(
            ts=ts,
            o=float(row.open),
            h=float(row.high),
            l=float(row.low),
            c=float(row.close),
            v=float(row.volume) if hasattr(row, "volume") else 0.0,
            index=i,
        ))
    return bars



# ---------------------------------------------------------------------------
# Run Triage — the service function called by the UI [Run Triage] button
# ---------------------------------------------------------------------------

class TriageRunResult:
    """Result from run_triage_for_composition."""
    def __init__(self):
        self.triage_v2: Optional[TriageV2Result] = None
        self.trade_count: int = 0
        self.bar_count: int = 0
        self.runtime_seconds: float = 0.0
        self.dataset_filename: str = ""
        self.dataset_prefix: str = ""
        self.saved_path: str = ""
        self.runner_economics: Dict[str, Any] = {}
        self.zero_trades: bool = False
        self.zero_trade_message: str = ""
        self.error: Optional[str] = None
        self.trade_details: List[Dict[str, Any]] = []


def run_triage_for_composition(
    resolved_config: dict,
    strategy_config_hash: str,
    dataset_path: str,
    spec: Optional[dict] = None,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
) -> TriageRunResult:
    """Run full triage v2 pipeline: load data → backtest → triage → save.

    This is THE function the UI [Run Triage] button calls.
    Integration tests MUST call this same function.
    """
    result = TriageRunResult()
    start_time = time.time()

    hash_val = strategy_config_hash
    if hash_val.startswith("sha256:"):
        hash_val = hash_val[7:]

    dataset_fn = os.path.basename(dataset_path)
    result.dataset_filename = dataset_fn
    result.dataset_prefix = dataset_fn.replace('.parquet', '')
    result.runner_economics = {
        "fee_rate": fee_rate,
        "slippage_bps": slippage_bps,
        "starting_capital": starting_capital,
    }

    # Load data
    bars = load_bars_from_parquet(dataset_path)
    result.bar_count = len(bars)

    # Run backtest
    trades, close_prices_float, n_bars = run_backtest(
        resolved_config, bars, strategy_config_hash)
    result.trade_count = len(trades)

    # Build enriched trade details for UI display
    for t in trades:
        entry_ts = bars[t.entry_idx].ts if t.entry_idx < len(bars) else 0
        exit_ts = bars[t.exit_idx].ts if t.exit_idx < len(bars) else 0
        entry_p = t.entry_price.value / 100.0
        exit_p = t.exit_price.value / 100.0
        pnl_usd = (exit_p - entry_p) if t.side == "long" else (entry_p - exit_p)
        result.trade_details.append({
            "direction": t.side.upper(),
            "entry_time": datetime.utcfromtimestamp(entry_ts).strftime("%Y-%m-%d %H:%M") if entry_ts else "—",
            "exit_time": datetime.utcfromtimestamp(exit_ts).strftime("%Y-%m-%d %H:%M") if exit_ts else "—",
            "entry_price": f"{entry_p:,.2f}",
            "exit_price": f"{exit_p:,.2f}",
            "pnl_usd": f"{pnl_usd:+,.2f}",
            "pnl_bps": t.gross_return_bps,
            "bars_held": t.exit_idx - t.entry_idx,
        })

    # Handle zero trades
    if len(trades) == 0:
        result.zero_trades = True
        warmup_bars = 0
        for inst in resolved_config.get("indicator_instances", []):
            from strategy_framework_v1_8_0 import compute_instance_warmup, INDICATOR_NAME_TO_ID
            from ui.services.backtest_runner import parse_timeframe_seconds
            ind_id = inst.get("indicator_id", 0)
            if isinstance(ind_id, str):
                ind_id = INDICATOR_NAME_TO_ID.get(ind_id, 0)
            outputs = inst.get("outputs_used", [])
            wu = compute_instance_warmup(ind_id, outputs, inst.get("parameters", {}))
            tf_sec = parse_timeframe_seconds(inst.get("timeframe", "1m"))
            wu_1m = wu * max(1, tf_sec // 60)
            warmup_bars = max(warmup_bars, wu_1m)

        warmup_days = warmup_bars / 1440
        dataset_days = n_bars / 1440
        tradeable_days = max(0, dataset_days - warmup_days)
        result.zero_trade_message = (
            f"Backtest produced 0 trades on this dataset. "
            f"Warmup: {warmup_days:.0f} days ({warmup_bars} bars). "
            f"Dataset: {dataset_days:.0f} days ({n_bars} bars). "
            f"Tradeable window: {tradeable_days:.0f} days. "
        )
        if tradeable_days < 30:
            result.zero_trade_message += (
                "The tradeable window is very short. Try a longer dataset.")
        else:
            result.zero_trade_message += (
                "Entry conditions may never be satisfied on this data.")
        result.runtime_seconds = time.time() - start_time
        return result

    # Run triage v2
    round_trip_bps = int(fee_rate * 10000 * 2) + slippage_bps * 2
    triage_result = run_triage_v2(
        trades=trades,
        n_bars=n_bars,
        round_trip_cost_bps=round_trip_bps,
        starting_capital=starting_capital,
    )
    result.triage_v2 = triage_result

    # Save triage result
    dir_path = os.path.join(RESEARCH_DIR, "triage_results", hash_val)
    os.makedirs(dir_path, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{result.dataset_prefix}_{timestamp}.json"
    filepath = os.path.join(dir_path, filename)

    save_data = {
        "strategy_config_hash": strategy_config_hash,
        "dataset_prefix": result.dataset_prefix,
        "dataset_filename": result.dataset_filename,
        "runner_economics": result.runner_economics,
        "trade_count": result.trade_count,
        "bar_count": result.bar_count,
        "triage_v2": triage_result.to_dict(),
        "timestamp": timestamp,
    }

    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    result.saved_path = filepath

    result.runtime_seconds = time.time() - start_time
    return result


# ---------------------------------------------------------------------------
# Write Promotion — the service function called by UI [Write Promotion] button
# ---------------------------------------------------------------------------

def write_promotion_for_composition(
    strategy_config_hash: str,
    composition_spec_hash: str,
    dataset_prefix: str,
    runner_economics: Dict[str, Any],
    triage_result_summary: Dict[str, Any],
    tier: str = "TRIAGE_PASSED",
) -> str:
    """Write a promotion artifact. Returns file path.

    This is THE function the UI [Write Promotion] button calls.
    Service-level gate: tier must be S, A, or B.
    """
    triage_tier = triage_result_summary.get("tier", "F")
    if triage_tier in ("F", "C"):
        raise ValueError(
            f"Cannot write promotion artifact: tier {triage_tier} is not promotable. "
            f"Action: {triage_result_summary.get('tier_action', 'unknown')}"
        )

    # Validate required fields
    if not strategy_config_hash:
        raise ValueError("strategy_config_hash is required")
    if not dataset_prefix:
        raise ValueError("dataset_prefix is required")

    hash_val = strategy_config_hash
    if hash_val.startswith("sha256:"):
        hash_val = hash_val[7:]

    promotions_dir = os.path.join(RESEARCH_DIR, "promotions", hash_val)
    os.makedirs(promotions_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    ts_file = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{tier}_PASS_{dataset_prefix}_{ts_file}.json"
    filepath = os.path.join(promotions_dir, filename)

    artifact = {
        "tier": tier,
        "strategy_config_hash": strategy_config_hash,
        "composition_spec_hash": composition_spec_hash,
        "dataset_prefix": dataset_prefix,
        "runner_economics": runner_economics,
        "triage_result_summary": triage_result_summary,
        "timestamp": timestamp,
    }

    # Validate all required fields present
    required = ["tier", "strategy_config_hash", "composition_spec_hash",
                 "dataset_prefix", "runner_economics", "triage_result_summary",
                 "timestamp"]
    for field in required:
        if field not in artifact or artifact[field] is None:
            raise ValueError(f"Promotion artifact missing required field: {field}")

    with open(filepath, "w") as f:
        json.dump(artifact, f, indent=2, default=str)

    return filepath


# ---------------------------------------------------------------------------
# Run Sweep — the service function called by UI [Parameter Sweep] button
# ---------------------------------------------------------------------------

class SweepResult:
    """Result from a parameter sweep."""
    def __init__(self):
        self.param_name: str = ""
        self.results: List[Dict[str, Any]] = []
        self.saved_path: str = ""


def run_sweep_for_composition(
    spec: dict,
    param_name: str,
    param_min: float,
    param_max: float,
    n_steps: int,
    dataset_path: str,
    default_value: float = 0,
) -> SweepResult:
    """Run parameter sweep: vary one param, compile + backtest + Test 1.

    Does NOT write promotion artifacts or triage results.
    Does NOT modify the on-disk spec.
    Works entirely in memory except for compile-generated resolved artifacts.
    """
    sweep_result = SweepResult()
    sweep_result.param_name = param_name

    bars = load_bars_from_parquet(dataset_path)

    # Generate parameter values
    if isinstance(param_min, int) and isinstance(param_max, int):
        step = max(1, (param_max - param_min) // max(1, n_steps - 1))
        values = list(range(int(param_min), int(param_max) + 1, step))[:n_steps]
    else:
        step = (param_max - param_min) / max(1, n_steps - 1)
        values = [param_min + i * step for i in range(n_steps)]

    for val in values:
        try:
            modified_spec = _apply_param_override(spec, param_name, val)
            compilation = compile_spec(modified_spec)
            resolved = compilation["resolved_artifact"]
            config_hash = compilation["strategy_config_hash"]

            # Write resolved artifact (idempotent, deterministic)
            save_artifacts(compilation)

            trades, prices, n_bars_run = run_backtest(resolved, bars, config_hash)

            # Run only Test 1 (expectancy) for speed
            from ui.services.triage_v2 import run_test_1 as run_test_1_v2
            t1 = run_test_1_v2(trades, round_trip_cost_bps=32)

            sweep_result.results.append({
                "param_value": val,
                "expectancy_bps": t1.metrics.get("expectancy_bps", 0),
                "win_rate": t1.metrics.get("win_rate", 0),
                "passed": t1.status == "PASS",
                "trades": len(trades),
                "hash": config_hash[:16],
                "is_default": abs(val - default_value) < 0.001,
            })
        except Exception as e:
            sweep_result.results.append({
                "param_value": val,
                "expectancy_bps": 0,
                "win_rate": 0,
                "passed": False,
                "trades": 0,
                "hash": "ERROR",
                "is_default": abs(val - default_value) < 0.001,
                "error": str(e),
            })

    # Save sweep results
    comp_id = spec.get("composition_id", "unknown")
    dir_path = os.path.join(RESEARCH_DIR, "sweep_results", comp_id)
    os.makedirs(dir_path, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_param = param_name.replace(".", "_").replace("/", "_")
    filename = f"{safe_param}_{ts}.json"
    filepath = os.path.join(dir_path, filename)

    with open(filepath, "w") as f:
        json.dump({
            "composition_id": comp_id,
            "param_name": param_name,
            "timestamp": ts,
            "results": sweep_result.results,
        }, f, indent=2, default=str)

    sweep_result.saved_path = filepath
    return sweep_result


def _apply_param_override(spec: dict, param_path: str, value) -> dict:
    """Apply a parameter override to a deep copy of the spec. Original is never modified."""
    spec = copy.deepcopy(spec)
    parts = param_path.split(".")

    if len(parts) == 2:
        label, pname = parts
        for inst in spec.get("indicator_instances", []):
            if inst.get("label") == label:
                inst["parameters"][pname] = value
                return spec
        for rule in spec.get("exit_rules", []):
            if rule.get("label") == label:
                rule[pname] = value
                return spec

    elif len(parts) == 3:
        section, idx_str, field = parts
        try:
            idx = int(idx_str)
            if section in spec and idx < len(spec[section]):
                spec[section][idx][field] = value
                return spec
        except ValueError:
            pass

    return spec
