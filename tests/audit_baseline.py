#!/usr/bin/env python3
"""Baseline recording script — capture trade counts BEFORE indicator scale changes.

Run this ONCE before any code edits, save the output.
These numbers become regression assertions in test_indicator_audit.py.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from composition_compiler_v1_5_2 import compile_composition
from ui.services.backtest_runner import Bar, run_backtest

# ---------------------------------------------------------------------------
# Data loading (inline to avoid extra deps on research_services)
# ---------------------------------------------------------------------------

def load_bars(path: str):
    import pandas as pd
    df = pd.read_parquet(path)
    bars = []
    for i, row in enumerate(df.itertuples()):
        ts = int(row.timestamp) if hasattr(row, "timestamp") else int(row.ts)
        bars.append(Bar(
            ts=ts, o=float(row.open), h=float(row.high),
            l=float(row.low), c=float(row.close),
            v=float(row.volume) if hasattr(row, "volume") else 0.0,
            index=i,
        ))
    return bars

# ---------------------------------------------------------------------------

DATASET_13MO = os.path.join(
    os.path.dirname(__file__), "..",
    "historic_data", "btcusdt_binance_spot_1m_2025-01-01_to_2026-01-31.parquet",
)

MACD_COMP = os.path.join(
    os.path.dirname(__file__), "..",
    "research", "compositions", "e9163f2e-95d7-4668-8387-70ed258b9144", "composition.json",
)

CHOP_PRESET = os.path.join(
    os.path.dirname(__file__), "..",
    "ui", "presets", "chop_harvester_clean.json",
)


def run_one(label: str, spec_path: str, bars):
    with open(spec_path) as f:
        spec = json.load(f)
    result = compile_composition(spec)
    resolved = result["resolved_artifact"]
    config_hash = result["strategy_config_hash"]
    trades, _, bar_count = run_backtest(resolved, bars, strategy_hash=config_hash)
    gross_returns = [t.gross_return_bps for t in trades]
    avg_ret = sum(gross_returns) / len(gross_returns) if gross_returns else 0.0
    print(f"\n=== {label} ===")
    print(f"  Trade count : {len(trades)}")
    print(f"  Bar count   : {bar_count}")
    print(f"  Avg gross bps: {avg_ret:.2f}")
    return len(trades)


def main():
    print("Loading 13-month dataset...")
    bars = load_bars(DATASET_13MO)
    print(f"  Loaded {len(bars)} bars")

    macd_trades = run_one("MACD Confluence Bull & Bear", MACD_COMP, bars)
    chop_trades = run_one("Chop Harvester Clean", CHOP_PRESET, bars)

    print("\n--- Regression Constants ---")
    print(f"MACD_CONFLUENCE_TRADE_COUNT = {macd_trades}")
    print(f"CHOP_HARVESTER_TRADE_COUNT = {chop_trades}")


if __name__ == "__main__":
    main()
