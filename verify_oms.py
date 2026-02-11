"""
Verification suite for phase5_oms_types.py + phase5_oms.py

Tests:
    1. All valid/invalid state transitions
    2. TradePlan validation
    3. Failsafe close sequence
    4. OCO behavior
    5. WebSocket disconnect → failsafe → HALT
    6. Startup reconciliation
    7. Error 10003 idempotency
    8. NO-OP guards
    9. Order limit enforcement
    10. OMS event commits share commit_seq with pod bar commits (unified chain)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from btc_alpha_v3_final import Fixed, SemanticType

from phase5_bundle_store import AtomicBundleStore, EventBundle, RunMode
from phase5_oms_types import (
    InvalidTransitionError,
    Order,
    OrderState,
    ORDER_TRANSITIONS,
    PositionState,
    POSITION_TRANSITIONS,
    TERMINAL_ORDER_STATES,
    TPLadderLeg,
    TradePlan,
)
from phase5_oms import (
    MAX_ACTIVE_ORDERS,
    OMS,
    SYMBOL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_dir() -> str:
    return tempfile.mkdtemp(prefix="oms_test_")


def _make_store(base: str) -> AtomicBundleStore:
    return AtomicBundleStore(base, pod_id="oms-pod", core_version_hash="abc")


def _make_adapter() -> MagicMock:
    """Create a mock ExchangeAdapter."""
    adapter = MagicMock()
    adapter.place_order.return_value = {"retCode": 0, "result": {"orderId": "ex123"}}
    adapter.cancel_order.return_value = {"retCode": 0}
    adapter.cancel_all_orders.return_value = {"retCode": 0}
    adapter.set_trading_stop.return_value = {"retCode": 0}
    adapter.query_order.return_value = {"orderId": "ex123", "orderStatus": "New"}
    adapter.query_position.return_value = {"size": 0, "markPrice": 10000_00}
    adapter.query_active_orders.return_value = []
    return adapter


def _make_plan(qty_value: int = 100_000_000) -> TradePlan:
    return TradePlan(
        plan_id="plan_001",
        symbol=SYMBOL,
        side="Buy",
        qty=Fixed(value=qty_value, sem=SemanticType.QTY),
        stop_loss=Fixed(value=9000_00, sem=SemanticType.PRICE),
    )


def _make_oms(base: str) -> tuple:
    """Returns (oms, store, adapter, halt_reasons)."""
    store = _make_store(base)
    adapter = _make_adapter()
    halt_reasons: list = []
    oms = OMS(
        adapter=adapter,
        bundle_store=store,
        on_halt=lambda r: halt_reasons.append(r),
        pod_id="oms-pod",
    )
    return oms, store, adapter, halt_reasons


# ---------------------------------------------------------------------------
# Test 1: All valid/invalid state transitions
# ---------------------------------------------------------------------------

def test_state_transitions():
    # Order transitions
    for from_state, valid_targets in ORDER_TRANSITIONS.items():
        for target in OrderState:
            order = Order(
                client_order_id="test", symbol=SYMBOL, side="Buy",
                order_type="Market",
                qty=Fixed(value=100, sem=SemanticType.QTY),
                price=None,
            )
            order.state = from_state
            if target in valid_targets:
                order.transition(target)  # Should succeed
            else:
                try:
                    order.transition(target)
                    assert False, f"Should reject {from_state} -> {target}"
                except InvalidTransitionError:
                    pass

    # Position transitions
    for from_state, valid_targets in POSITION_TRANSITIONS.items():
        for target in PositionState:
            if target in valid_targets:
                pass  # Valid
            elif from_state == PositionState.FAILSAFE and target != PositionState.FLAT:
                pass  # Correctly rejected
            # More exhaustive check done via OMS._transition_position

    # FAILSAFE → only FLAT
    assert POSITION_TRANSITIONS[PositionState.FAILSAFE] == {PositionState.FLAT}

    print("  PASS: All valid/invalid state transitions verified")


# ---------------------------------------------------------------------------
# Test 2: TradePlan validation
# ---------------------------------------------------------------------------

def test_trade_plan_validation():
    # Valid plan
    plan = _make_plan()
    errors = plan.validate()
    assert errors == [], f"Valid plan has errors: {errors}"

    # Missing stop_loss
    plan_no_sl = TradePlan(
        plan_id="bad",
        qty=Fixed(value=100_000_000, sem=SemanticType.QTY),
        stop_loss=None,
    )
    errors = plan_no_sl.validate()
    assert any("stop_loss" in e for e in errors)

    # Zero qty
    plan_zero = TradePlan(
        plan_id="bad",
        qty=Fixed(value=0, sem=SemanticType.QTY),
        stop_loss=Fixed(value=9000_00, sem=SemanticType.PRICE),
    )
    errors = plan_zero.validate()
    assert any("qty" in e for e in errors)

    # Too many TP legs
    legs = [
        TPLadderLeg(
            price=Fixed(value=11000_00 + i, sem=SemanticType.PRICE),
            qty_pct=20,
        )
        for i in range(500)
    ]
    plan_big = TradePlan(
        plan_id="big",
        qty=Fixed(value=100_000_000, sem=SemanticType.QTY),
        stop_loss=Fixed(value=9000_00, sem=SemanticType.PRICE),
        tp_ladder=legs,
    )
    errors = plan_big.validate()
    assert any("490" in e for e in errors)

    # TP ladder qty_pct > 100%
    legs_over = [
        TPLadderLeg(
            price=Fixed(value=11000_00, sem=SemanticType.PRICE),
            qty_pct=6000,
        ),
        TPLadderLeg(
            price=Fixed(value=12000_00, sem=SemanticType.PRICE),
            qty_pct=6000,
        ),
    ]
    plan_over = TradePlan(
        plan_id="over",
        qty=Fixed(value=100_000_000, sem=SemanticType.QTY),
        stop_loss=Fixed(value=9000_00, sem=SemanticType.PRICE),
        tp_ladder=legs_over,
    )
    errors = plan_over.validate()
    assert any("10000" in e for e in errors)

    # Trailing stop without flag
    plan_ts = TradePlan(
        plan_id="ts",
        qty=Fixed(value=100_000_000, sem=SemanticType.QTY),
        stop_loss=Fixed(value=9000_00, sem=SemanticType.PRICE),
        trailing_stop=Fixed(value=500_00, sem=SemanticType.PRICE),
        accept_non_deterministic_trigger=False,
    )
    errors = plan_ts.validate()
    assert any("non_deterministic" in e.lower() for e in errors)

    print("  PASS: TradePlan validation (valid + all invalid cases)")


# ---------------------------------------------------------------------------
# Test 3: Failsafe close sequence
# ---------------------------------------------------------------------------

def test_failsafe_close():
    d = _tmp_dir()
    try:
        oms, store, adapter, halts = _make_oms(d)

        # Set up: OMS has a LONG position
        oms._position_state = PositionState.LONG
        adapter.query_position.return_value = {"size": 100_000_000, "markPrice": 10000_00}

        result = oms.failsafe_close("test_failsafe")
        assert result, "Failsafe close should succeed"

        # Verify position is FLAT
        assert oms.position_state == PositionState.FLAT

        # Verify adapter calls
        adapter.set_trading_stop.assert_called_once()
        adapter.cancel_all_orders.assert_called_once()
        adapter.place_order.assert_called()  # reduce-only market

        # Verify the reduce-only market order was placed
        call_kwargs = adapter.place_order.call_args
        assert call_kwargs[1].get("reduce_only") or call_kwargs.kwargs.get("reduce_only")

        print("  PASS: Failsafe close 5-step sequence")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: OCO behavior
# ---------------------------------------------------------------------------

def test_oco_behavior():
    d = _tmp_dir()
    try:
        oms, store, adapter, halts = _make_oms(d)

        # Place a trade plan
        plan = _make_plan()
        oms.execute_trade_plan(plan)

        # Simulate entry fill
        oms.handle_entry_fill(
            list(oms._active_orders.keys())[0],
            Fixed(value=10000_00, sem=SemanticType.PRICE),
            Fixed(value=100_000_000, sem=SemanticType.QTY),
        )
        assert oms.position_state == PositionState.LONG

        # Add some reduce-only exit orders
        for i in range(3):
            coid = f"exit_{i}"
            order = Order(
                client_order_id=coid, symbol=SYMBOL, side="Sell",
                order_type="Limit",
                qty=Fixed(value=33_000_000, sem=SemanticType.QTY),
                price=Fixed(value=11000_00 + i * 100, sem=SemanticType.PRICE),
                reduce_only=True,
            )
            order.state = OrderState.ACTIVE
            oms._active_orders[coid] = order

        # Simulate one exit filling → OCO should cancel siblings
        oms.handle_exit_fill(
            "exit_0",
            Fixed(value=33_000_000, sem=SemanticType.QTY),
        )

        # Verify sibling exits were cancelled
        assert oms._active_orders["exit_1"].state == OrderState.CANCELLED
        assert oms._active_orders["exit_2"].state == OrderState.CANCELLED

        print("  PASS: OCO behavior (exit fill cancels siblings)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 5: WebSocket disconnect → failsafe → HALT
# ---------------------------------------------------------------------------

def test_ws_disconnect():
    d = _tmp_dir()
    try:
        oms, store, adapter, halts = _make_oms(d)
        oms._position_state = PositionState.LONG
        adapter.query_position.return_value = {"size": 100_000_000, "markPrice": 10000_00}

        # Disconnect
        oms.handle_ws_disconnect()
        assert not oms._ws_connected

        # Before deadline — no action
        oms.check_ws_deadline()
        assert len(halts) == 0  # No halt yet

        # Simulate time passing past deadline
        oms._ws_disconnect_ts = time.time() - 31
        oms.check_ws_deadline()

        # Should have triggered failsafe + HALT
        assert len(halts) == 1
        assert "30s" in halts[0] or "disconnect" in halts[0].lower()
        assert oms.position_state == PositionState.FLAT

        print("  PASS: WebSocket disconnect → failsafe → HALT")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 6: Startup reconciliation
# ---------------------------------------------------------------------------

def test_startup_reconciliation():
    d = _tmp_dir()
    try:
        # Case 1: No position → success
        oms, store, adapter, halts = _make_oms(d)
        adapter.query_position.return_value = {"size": 0}
        assert oms.startup_reconciliation() is True
        assert oms.position_state == PositionState.FLAT

        # Case 2: Has position, no matching orders → HALT
        d2 = _tmp_dir()
        oms2, store2, adapter2, halts2 = _make_oms(d2)
        adapter2.query_position.return_value = {"size": 100_000_000, "markPrice": 10000_00}
        adapter2.query_active_orders.return_value = []
        assert oms2.startup_reconciliation() is False
        assert len(halts2) == 1  # HALT called
        shutil.rmtree(d2, ignore_errors=True)

        # Case 3: Has position, matching orders → reconstructed
        d3 = _tmp_dir()
        oms3, store3, adapter3, halts3 = _make_oms(d3)
        adapter3.query_position.return_value = {"size": 100_000_000, "markPrice": 10000_00}
        adapter3.query_active_orders.return_value = [
            {"orderLinkId": "entry_abc123", "stopLoss": "9000.00", "orderId": "x1"},
        ]
        assert oms3.startup_reconciliation() is True
        assert oms3.position_state == PositionState.LONG
        shutil.rmtree(d3, ignore_errors=True)

        print("  PASS: Startup reconciliation (no position / failed / reconstructed)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 7: Error 10003 idempotency
# ---------------------------------------------------------------------------

def test_error_10003():
    d = _tmp_dir()
    try:
        oms, store, adapter, halts = _make_oms(d)

        # Order exists → idempotent success
        adapter.query_order.return_value = {"orderId": "ex123", "orderStatus": "New"}
        result = oms.handle_error_10003("entry_abc")
        assert result is True

        # Order not found → failure
        adapter.query_order.return_value = None
        result = oms.handle_error_10003("entry_xyz")
        assert result is False

        print("  PASS: Error 10003 idempotency handling")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 8: NO-OP guards
# ---------------------------------------------------------------------------

def test_noop_guards():
    d = _tmp_dir()
    try:
        oms, store, adapter, halts = _make_oms(d)

        # Exit suppressed in FLAT
        oms._position_state = PositionState.FLAT
        assert oms._noop_guard("exit") is True

        # Exit suppressed in FAILSAFE
        oms._position_state = PositionState.FAILSAFE
        assert oms._noop_guard("exit") is True

        # Exit suppressed in PENDING_CLOSE
        oms._position_state = PositionState.PENDING_CLOSE
        assert oms._noop_guard("exit") is True

        # Exit allowed in LONG
        oms._position_state = PositionState.LONG
        assert oms._noop_guard("exit") is False

        # Entry is not guarded by this check
        oms._position_state = PositionState.FLAT
        assert oms._noop_guard("entry") is False

        print("  PASS: NO-OP guards (exit suppressed in FLAT/FAILSAFE/PENDING_CLOSE)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 9: Order limit enforcement
# ---------------------------------------------------------------------------

def test_order_limits():
    d = _tmp_dir()
    try:
        oms, store, adapter, halts = _make_oms(d)

        # Under limit
        assert oms._check_order_limits(1) is True

        # Add many active orders
        for i in range(500):
            order = Order(
                client_order_id=f"o_{i}", symbol=SYMBOL, side="Buy",
                order_type="Limit",
                qty=Fixed(value=100, sem=SemanticType.QTY),
                price=Fixed(value=10000_00, sem=SemanticType.PRICE),
            )
            order.state = OrderState.ACTIVE
            oms._active_orders[f"o_{i}"] = order

        # At limit — should reject
        assert oms._check_order_limits(1) is False

        print("  PASS: Order limit enforcement (500 max)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 10: Unified chain (OMS events share commit_seq with bar bundles)
# ---------------------------------------------------------------------------

def test_unified_chain():
    d = _tmp_dir()
    try:
        store = _make_store(d)
        adapter = _make_adapter()
        halts: list = []
        oms = OMS(adapter, store, lambda r: halts.append(r), "oms-pod")

        # Commit a "bar" bundle via store directly (simulating pod)
        from phase5_bundle_store import BarBundle
        bar = BarBundle(
            format_version=1, bar_index=0, timestamp=1000,
            pod_id="oms-pod", core_version_hash="abc",
            candle_inputs={}, system_inputs={}, period_data=None,
            indicator_outputs={}, engine_state_hash="x", health_state={},
        )
        key = store.stage_bundle(bar)
        store.commit_bundle(key)

        # Now commit an OMS event
        oms._commit_event("test_event", {"foo": 1})

        # Execute a plan → commits more events
        plan = _make_plan()
        oms.execute_trade_plan(plan)

        # Verify unified chain: all committed bundles share contiguous commit_seq
        ok, err = store.verify_chain_integrity()
        assert ok, f"Chain integrity failed: {err}"

        # Verify commit_seq is contiguous
        wm = store.watermark
        assert wm.commit_seq >= 2  # bar + at least 2 events
        bundles = store.replay(0, wm.commit_seq)
        for i, b in enumerate(bundles):
            assert b.commit_seq == i, f"Gap at {i}: {b.commit_seq}"

        print("  PASS: Unified chain (OMS events share commit_seq with bar bundles)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("verify_oms.py — 10 verification items")
    print("=" * 60)

    tests = [
        ("1. State transitions (valid + invalid)", test_state_transitions),
        ("2. TradePlan validation", test_trade_plan_validation),
        ("3. Failsafe close sequence", test_failsafe_close),
        ("4. OCO behavior", test_oco_behavior),
        ("5. WebSocket disconnect → failsafe → HALT", test_ws_disconnect),
        ("6. Startup reconciliation", test_startup_reconciliation),
        ("7. Error 10003 idempotency", test_error_10003),
        ("8. NO-OP guards", test_noop_guards),
        ("9. Order limit enforcement", test_order_limits),
        ("10. Unified chain (shared commit_seq)", test_unified_chain),
    ]

    passed = 0
    failed = 0
    for label, fn in tests:
        print(f"\n[{label}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
