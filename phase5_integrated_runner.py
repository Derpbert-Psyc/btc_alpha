"""
Phase 5 — Integrated Backtest Runner

Runs the Phase 3 backtest loop with Phase 4B indicator engine running alongside.
Hash equivalence gate: the Phase 3 results (trades, equity, metrics) must be
identical whether run standalone or integrated.

References:
    btc_alpha_phase3.py:1206-1494 — BacktestRunner.run() loop
    btc_alpha_phase4b_1_7_2.py:7189-7495 — IndicatorEngine.compute_all()
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from btc_alpha_v3_final import Fixed, SemanticType, SEMANTIC_SCALES, RoundingMode
from btc_alpha_phase2_v4 import (
    BaselineExecutionConfig,
    BaselineExecutionLogic,
    BaselineIndicatorConfig,
    BaselineIndicatorModule,
    BaselineSignalConfig,
    BaselineSignalModule,
    Candle,
    CommittedHistory,
    EvaluationMetrics,
    GateView,
    HistoricalView1m,
    HistoricalView4H,
    LedgerView,
    PositionView,
    TradeRecord,
    compute_metrics,
    create_gate_view,
    hash_metrics,
    hash_trades,
    SECONDS_PER_4H,
)
from btc_alpha_phase3 import (
    BacktestResult,
    GapError,
    GapPolicy,
    Phase3Config,
    PrecomputedHTF,
    SimulatedLedger,
    apply_fill,
    apply_slippage,
    compute_fee,
    hash_config,
    precompute_4h_index,
    SECONDS_PER_MINUTE,
)
from btc_alpha_phase4b_1_7_2 import (
    IndicatorEngine,
    IndicatorOutput as P4BIndicatorOutput,
    SystemInputs,
    TypedValue,
)
from phase5_type_bridge import (
    build_period_data,
    build_system_inputs,
    candle_to_candle_inputs,
    fixed_to_typed,
)


# ---------------------------------------------------------------------------
# Result type for integrated runs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntegratedResult:
    """Result from the integrated runner.

    phase3_result: identical to what BacktestRunner.run() would produce.
    phase4b_outputs: Dict[bar_counter -> Dict[indicator_id -> serialized output]].
    phase4b_hash: SHA256 of canonical JSON of all phase4b_outputs.
    """
    phase3_result: BacktestResult
    phase4b_outputs: Dict[int, Dict[int, Dict[str, Any]]]
    phase4b_hash: str


def _serialize_indicator_output(
    output: P4BIndicatorOutput,
) -> Dict[str, Any]:
    """Serialize a Phase 4B IndicatorOutput to a JSON-compatible dict."""
    values_ser = {}
    for k, v in output.values.items():
        if v is None:
            values_ser[k] = None
        else:
            values_ser[k] = {"v": v.value, "s": v.sem.value}
    return {
        "indicator_id": output.indicator_id,
        "timestamp": output.timestamp,
        "values": values_ser,
        "computed": output.computed,
        "eligible": output.eligible,
    }


def _hash_phase4b_outputs(
    outputs: Dict[int, Dict[int, Dict[str, Any]]],
) -> str:
    """Compute deterministic hash of all Phase 4B outputs."""
    # Sort by bar_counter (outer key), then by indicator_id (inner key)
    canonical = {}
    for bar_counter in sorted(outputs.keys()):
        inner = {}
        for ind_id in sorted(outputs[bar_counter].keys()):
            inner[str(ind_id)] = outputs[bar_counter][ind_id]
        canonical[str(bar_counter)] = inner
    js = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(js.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Integrated Backtest Runner
# ---------------------------------------------------------------------------

class IntegratedBacktestRunner:
    """
    Runs Phase 3 backtest loop with Phase 4B IndicatorEngine.

    Preserves ALL Phase 3 loop invariants:
        - Timestamp convention (close time)
        - Warmup via max_lookback
        - Gap handling (HALT policy)
        - Fill timing (t+1 open)
        - Friction order (slippage then fees)
        - 4H lookahead prevention
        - OHLC consistency
    """

    def __init__(
        self,
        config: Phase3Config,
        *,
        stream_id: str = "BTCUSDT-1m-integrated",
    ) -> None:
        self.config = config

        # Phase 2/3 modules — identical to BacktestRunner
        self.indicator_config = BaselineIndicatorConfig(
            sma_fast_period=config.sma_fast_period,
            sma_slow_period=config.sma_slow_period,
            ema_period=config.ema_period,
            donchian_period=config.donchian_period,
            atr_period=config.atr_period,
            htf_sma_period=config.htf_sma_period,
        )
        self.indicator_module = BaselineIndicatorModule(self.indicator_config)
        self.signal_module = BaselineSignalModule(BaselineSignalConfig())
        self.execution_config = BaselineExecutionConfig(
            risk_fraction=config.risk_fraction,
            min_qty=config.min_qty,
            max_qty=config.max_qty,
        )
        self.execution_logic = BaselineExecutionLogic(self.execution_config)

        self.max_lookback_1m = self.indicator_module.max_lookback_1m
        self.max_lookback_4h = self.indicator_module.max_lookback_4h

        # Phase 4B engine
        self._stream_id = stream_id
        self._engine = IndicatorEngine(stream_id=stream_id)
        self._engine.register_all()

    def run(
        self,
        candles_1m: Tuple[Candle, ...],
        start_ts: int,
        end_ts: int,
    ) -> IntegratedResult:
        """
        Run integrated backtest.

        Returns IntegratedResult with identical Phase 3 results and Phase 4B outputs.
        """
        # Initialize ledger — identical to Phase 3
        ledger = SimulatedLedger(
            cash=self.config.starting_capital,
            position_qty=Fixed.zero(SemanticType.QTY),
            avg_entry_price=None,
            total_fees_paid=Fixed.zero(SemanticType.USD),
            total_slippage_cost=Fixed.zero(SemanticType.USD),
            realized_pnl=Fixed.zero(SemanticType.USD),
        )

        trades: List[TradeRecord] = []
        equity_series: List[Tuple[int, Fixed]] = []
        leverage_series: List[Tuple[int, int]] = []

        # Track open trade — identical to Phase 3
        entry_ts: Optional[int] = None
        entry_price: Optional[Fixed] = None
        entry_qty: Optional[Fixed] = None
        entry_side: Optional[Literal["LONG", "SHORT"]] = None
        trade_fees: int = 0
        trade_slippage: int = 0

        # Phase 5 additional state
        bar_counter: int = 0
        entry_bar: Optional[int] = None
        prev_4h_close_count: int = 0
        phase4b_outputs: Dict[int, Dict[int, Dict[str, Any]]] = {}

        # =====================================================================
        # PRE-COMPUTATION PHASE — identical to Phase 3
        # =====================================================================

        candle_by_ts: Dict[int, Candle] = {c.ts: c for c in candles_1m}
        htf_index = precompute_4h_index(candles_1m)
        sorted_1m_candles = sorted(candles_1m, key=lambda c: c.ts)
        candle_index_by_ts: Dict[int, int] = {
            c.ts: i for i, c in enumerate(sorted_1m_candles)
        }

        # =====================================================================
        # SIMULATION LOOP — Phase 3 invariants preserved exactly
        # =====================================================================

        current_ts = start_ts
        total_minutes = (end_ts - start_ts) // SECONDS_PER_MINUTE + 1
        processed = 0

        while current_ts <= end_ts:
            processed += 1
            if processed % 10000 == 0:
                pct = (processed / total_minutes) * 100
                print(f"  Progress: {processed}/{total_minutes} ({pct:.1f}%)")

            # Gap check — identical to Phase 3
            if current_ts not in candle_by_ts:
                if self.config.gap_policy == GapPolicy.HALT:
                    raise GapError(
                        f"Missing candle at {current_ts} during backtest run. "
                        f"Gap policy is HALT - cannot continue."
                    )
                current_ts += SECONDS_PER_MINUTE
                continue

            current_candle = candle_by_ts[current_ts]

            # Build 1m historical view — identical to Phase 3
            if current_ts in candle_index_by_ts:
                current_idx = candle_index_by_ts[current_ts]
                start_idx = max(0, current_idx - self.max_lookback_1m - 100)
                visible_1m = tuple(sorted_1m_candles[start_idx:current_idx])
            else:
                visible_1m = ()

            view_1m = HistoricalView1m(candles=visible_1m, current_ts=current_ts)

            # Build 4H historical view — identical to Phase 3
            view_4h = htf_index.get_visible_4h(
                current_ts, max_lookback=self.max_lookback_4h + 10
            )

            history = CommittedHistory(view_1m=view_1m, view_4h=view_4h)

            # ---- Phase 2/3 indicators (for Phase 3 result) ----
            indicators = self.indicator_module.compute(history)

            # ---- Phase 4B indicators (additional output) ----
            candle_inputs = candle_to_candle_inputs(current_candle)

            # Build system inputs from current ledger state
            position_side = 0
            if ledger.position_qty.is_positive():
                position_side = 1
            elif ledger.position_qty.is_negative():
                position_side = -1

            system_inputs = build_system_inputs(
                equity=ledger.equity(current_candle.close),
                position_side=position_side,
                entry_index=entry_bar,
                benchmark_close=current_candle.close,
            )

            # Build period_data from most recent completed 4H candle
            prev_4h_candle = None
            if view_4h.candles:
                prev_4h_candle = view_4h.candles[-1]
            period_data = build_period_data(prev_4h_candle)

            p4b_results = self._engine.compute_all(
                timestamp=current_ts,
                bar_index=bar_counter,
                candle_inputs=candle_inputs,
                system_inputs=system_inputs,
                period_data=period_data,
                stream_id=self._stream_id,
            )

            # Serialize and store Phase 4B outputs
            phase4b_outputs[bar_counter] = {
                ind_id: _serialize_indicator_output(out)
                for ind_id, out in p4b_results.items()
            }

            # ---- Phase 3 decision pipeline — identical ----
            gates = create_gate_view(
                diagnostics_ok=True,
                diagnostics_veto=False,
                regime_permissive=self.config.regime_always_permissive,
                indicators=indicators,
            )

            decision_price = current_candle.close

            position = PositionView(
                position_qty=ledger.position_qty,
                avg_entry_price=ledger.avg_entry_price,
            )

            signal = self.signal_module.compute(indicators, position, decision_price)

            ledger_view = LedgerView(
                equity=ledger.equity(decision_price),
                position_qty=ledger.position_qty,
                avg_entry_price=ledger.avg_entry_price,
            )

            execution = self.execution_logic.compute(
                signal, gates, ledger_view, decision_price
            )

            # Execute orders at next minute open — identical to Phase 3
            if execution.orders:
                next_ts = current_ts + SECONDS_PER_MINUTE
                if next_ts in candle_by_ts:
                    next_candle = candle_by_ts[next_ts]
                    fill_reference_price = next_candle.open

                    for order in execution.orders:
                        fill_price = apply_slippage(
                            fill_reference_price,
                            order.side,
                            self.config.slippage_rate_bps,
                        )

                        from btc_alpha_phase3 import _integer_divide_with_rounding
                        slippage_cost_value = abs(
                            fill_price.value - fill_reference_price.value
                        )
                        slippage_notional = _integer_divide_with_rounding(
                            slippage_cost_value * order.qty.value,
                            10 ** SEMANTIC_SCALES[SemanticType.QTY],
                            RoundingMode.AWAY_FROM_ZERO,
                        )
                        slippage_cost = Fixed(
                            value=slippage_notional, sem=SemanticType.USD
                        )

                        fee = compute_fee(
                            fill_price, order.qty, self.config.fee_rate_bps
                        )

                        was_flat = ledger.position_qty.is_zero()
                        prev_side = (
                            "LONG"
                            if ledger.position_qty.is_positive()
                            else "SHORT"
                        )

                        ledger = apply_fill(
                            ledger, order.side, order.qty, fill_price,
                            fee, slippage_cost,
                        )

                        # Trade record management — identical to Phase 3
                        if was_flat and execution.action == "ENTER":
                            entry_ts = next_ts
                            entry_price = fill_price
                            entry_qty = order.qty
                            entry_side = "LONG" if order.side == "BUY" else "SHORT"
                            trade_fees = fee.value
                            trade_slippage = slippage_cost.value
                            entry_bar = bar_counter  # Phase 5 addition

                        elif execution.action == "EXIT" and entry_ts is not None:
                            trade_fees += fee.value
                            trade_slippage += slippage_cost.value

                            if entry_side == "LONG":
                                gross_pnl_per_unit = (
                                    fill_price.value - entry_price.value
                                )
                            else:
                                gross_pnl_per_unit = (
                                    entry_price.value - fill_price.value
                                )

                            gross_pnl_raw = entry_qty.value * gross_pnl_per_unit
                            gross_pnl_value = _integer_divide_with_rounding(
                                gross_pnl_raw,
                                10 ** SEMANTIC_SCALES[SemanticType.QTY],
                                RoundingMode.TRUNCATE,
                            )

                            net_pnl_value = (
                                gross_pnl_value - trade_fees - trade_slippage
                            )

                            trade = TradeRecord(
                                entry_ts=entry_ts,
                                exit_ts=next_ts,
                                side=entry_side,
                                entry_price=entry_price,
                                exit_price=fill_price,
                                qty=entry_qty,
                                gross_pnl=Fixed(
                                    value=gross_pnl_value, sem=SemanticType.USD
                                ),
                                fees=Fixed(value=trade_fees, sem=SemanticType.USD),
                                slippage=Fixed(
                                    value=trade_slippage, sem=SemanticType.USD
                                ),
                                net_pnl=Fixed(
                                    value=net_pnl_value, sem=SemanticType.USD
                                ),
                            )
                            trades.append(trade)

                            entry_ts = None
                            entry_price = None
                            entry_qty = None
                            entry_side = None
                            trade_fees = 0
                            trade_slippage = 0
                            entry_bar = None

            # Record equity and leverage — identical to Phase 3
            mark_price = current_candle.close
            equity_series.append((current_ts, ledger.equity(mark_price)))
            leverage_series.append((current_ts, ledger.leverage_bps(mark_price)))

            bar_counter += 1
            current_ts += SECONDS_PER_MINUTE

        # Compute metrics — identical to Phase 3
        metrics = compute_metrics(trades, equity_series, leverage_series)

        trades_hash, equity_hash, metrics_hash, config_hash_val = (
            BacktestResult.compute_hashes(
                trades, equity_series, metrics, self.config
            )
        )

        phase3_result = BacktestResult(
            trades=tuple(trades),
            equity_series=tuple(equity_series),
            leverage_series=tuple(leverage_series),
            metrics=metrics,
            config=self.config,
            trades_hash=trades_hash,
            equity_hash=equity_hash,
            metrics_hash=metrics_hash,
            config_hash=config_hash_val,
        )

        phase4b_hash = _hash_phase4b_outputs(phase4b_outputs)

        return IntegratedResult(
            phase3_result=phase3_result,
            phase4b_outputs=phase4b_outputs,
            phase4b_hash=phase4b_hash,
        )
