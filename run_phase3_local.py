from pathlib import Path

from btc_alpha_phase3 import (
    load_data,
    validate_candles,
    GapPolicy,
    Phase3Config,
    FrictionPreset,
    BacktestRunner,
)

# run_phase3_local.py
#
# This runner is intentionally pinned to the 3-month dataset for fast, repeatable validation.
# It will NOT fall back to larger datasets.
#
# Expected dataset path (fetched via scripts/fetch_data.sh with default selector "3mo"):
#   historic_data/btcusdt_binance_spot_1m_2025-10-01_to_2025-12-31.parquet

DATA = (
    Path(__file__).resolve().parent
    / "historic_data"
    / "btcusdt_binance_spot_1m_2025-10-01_to_2025-12-31.parquet"
)


def run_once(preset: FrictionPreset):
    data_path = Path(DATA)
    if not data_path.exists():
        raise SystemExit(
            f"missing required 3-month dataset:\n"
            f"  {data_path}\n\n"
            f"fetch it with:\n"
            f"  ./scripts/fetch_data.sh\n"
        )

    raw = load_data(DATA)
    candles, report = validate_candles(raw, GapPolicy.HALT)
    assert report.is_valid, report.error_message

    candles = tuple(candles)
    start_ts = candles[0].ts
    end_ts = candles[-1].ts

    config = Phase3Config(
        friction_preset=preset,
        gap_policy=GapPolicy.HALT,
        regime_always_permissive=True,
    )
    runner = BacktestRunner(config)

    # Start after warmup to avoid intentional "no trade before warmup" effects
    start_ts = candles[0].ts + runner.max_lookback_1m * 60

    result = runner.run(candles, start_ts, end_ts)
    return result


def main():
    print("Phase 3 local run (3-month dataset only)")
    print(f"DATA: {DATA}")

    for preset in [FrictionPreset.CONSERVATIVE, FrictionPreset.PUNITIVE]:
        r1 = run_once(preset)
        r2 = run_once(preset)

        print()
        print(f"Preset: {preset.name}")
        print(f"Trades: {r1.metrics.trade_count}")
        print(f"Net PnL: {r1.metrics.net_pnl.value/100:.2f} USD")
        print(f"Max DD:  {r1.metrics.max_drawdown.value/100:.2f} USD")
        print("Hashes:")
        print(f"  trades:  {r1.trades_hash}")
        print(f"  equity:  {r1.equity_hash}")
        print(f"  metrics: {r1.metrics_hash}")
        print(f"  config:  {r1.config_hash}")

        ok = (
            r1.trades_hash == r2.trades_hash
            and r1.equity_hash == r2.equity_hash
            and r1.metrics_hash == r2.metrics_hash
            and r1.config_hash == r2.config_hash
        )
        print(f"Determinism rerun match: {ok}")


if __name__ == "__main__":
    main()
