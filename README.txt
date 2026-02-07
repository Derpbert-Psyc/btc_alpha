README

Repository purpose
This repository contains a phase-structured BTC research, validation, and backtesting system. It includes core numeric primitives, historical data ingestion and validation, a backtest execution pipeline, indicator prototypes, formal specification documents, and both real and synthetic datasets used for correctness and adversarial testing.

The system is intentionally layered. Phase 1 through Phase 3 are wired together via imports and form the executable pipeline. Phase 4B and Phase 5 materials are contract and integration focused and are not yet wired into execution.

Top-level directory structure

.gitignore
Git ignore rules.

Core Python modules (execution and logic)

btc_alpha_v3_final.py
Phase 1. Core numeric and semantic foundations.
Defines fixed-point arithmetic, semantic typing, scale rules, rounding behavior, and invariants.
This is the lowest-level dependency and is imported by Phase 2.

btc_alpha_phase2_v4.py
Phase 2. Strategy and domain kernel.
Builds on Phase 1 primitives.
Defines Candle types and other domain-level constructs.
Imports btc_alpha_v3_final.py.

btc_alpha_phase3.py
Phase 3. Historical data ingestion, validation, and backtest pipeline.
Loads CSV or Parquet OHLCV data.
Validates ordering, gaps, timestamp alignment, OHLC consistency, and bounds.
Provides the backtest runner used by the local entrypoint.
Imports btc_alpha_phase2_v4.py.

run_phase3_local.py
Local execution entrypoint for Phase 3.
Demonstrates and exercises the full Phase 1 → Phase 3 pipeline on a dataset.
Imports btc_alpha_phase3.py.

btc_alpha_phase4b_1_7_2.py
Phase 4B. Indicator prototype implementation.
Contains indicator logic under Phase 4B contracts.
Currently standalone and not imported by the Phase 1 → Phase 3 pipeline.

Text artifacts

PHASE3_BASELINE_RESULTS.txt
Stored output from a Phase 3 baseline run.
Used for comparison and regression reference.
Not imported by code.

Specification and contract documents (Markdown)

SYSTEM_LAWS.md
High-level system laws and non-negotiable architectural constraints.

PHASE2_INVARIANTS.md
Formal invariants and rules governing Phase 2 behavior.

PHASE4A_INDICATOR_CONTRACT.md
Contract defining expectations, inputs, outputs, and constraints for Phase 4A indicators.

Phase 4B Contract and Non-Goals.md
Scope definition for Phase 4B.
Clarifies what Phase 4B must and must not do.

PHASE4B_CONTRACT_LOCKED.md
Locked Phase 4B contract.
This document is authoritative and should not be modified without explicit versioning.

PHASE5_INTEGRATION_SEAMS_v1_2_3.md
Defines the integration seams and boundaries for Phase 5.
Describes how Phase 4 outputs will connect into later system stages.

Data directories

historic_data/
Real historical market data used for backtesting.
Not stored in this repository due to size.

The system supports running Phase 3 on:

synthetic_data/ (included in repo) for validation and adversarial tests

sample_real_data/ (optional, small real dataset included in repo) for quick realistic runs

full datasets (external) downloaded via scripts with sha256 verification

To obtain full datasets, use:
scripts/fetch_data.sh
This script downloads externally hosted parquet files and verifies sha256 before use.
Contents:

* btcusdt_binance_spot_1m_2018_to_2026.parquet
  Long-range 1-minute BTCUSDT spot data from Binance.

* btcusdt_binance_spot_1m_2025-01-01_to_*.parquet
  Recent subset used for faster iteration and targeted testing.

These files are consumed by Phase 3 ingestion and validation.

synthetic_data/
Synthetic Parquet datasets used to test validation logic and failure modes.
Included in repo. Used to test ingestion validation and failure semantics.
Contents:

* synthetic_ok_1m.parquet
  Fully valid synthetic dataset.

* synthetic_open_time_ms.parquet
  Dataset testing timestamp representation edge cases.

* bad_duplicate.parquet
  Contains duplicate candles.

* bad_gap.parquet
  Contains missing time intervals.

* bad_ohlc.parquet
  Contains invalid OHLC relationships.

* bad_out_of_order.parquet
  Contains candles out of chronological order.

* bad_price_bounds.parquet
  Contains prices violating expected bounds.

* bad_ts_alignment.parquet
  Contains timestamp alignment errors.

* synthetic_manifest.json
  Manifest describing the synthetic datasets and their intended failure modes.

These datasets are explicitly designed to ensure Phase 3 fails loudly and correctly under invalid conditions.

Wiring summary

Execution path:

run_phase3_local.py
→ imports btc_alpha_phase3.py
→ imports btc_alpha_phase2_v4.py
→ imports btc_alpha_v3_final.py

btc_alpha_phase4b_1_7_2.py is currently outside this execution chain.

This README is intended to be a factual map of what exists today, not a roadmap or aspirational design.
