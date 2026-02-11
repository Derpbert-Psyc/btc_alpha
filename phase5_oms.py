"""
Phase 5 — Bybit OMS Implementation

Order Management System for BTCUSDT USDT Perpetual (Linear), One-Way Mode.
Commits EventBundles through the pod's shared AtomicBundleStore.

References:
    BYBIT_OMS_CONTRACT_v1_0_0.md — Complete document
    PHASE5_INTEGRATION_SEAMS_v1_2_6.md:484-518 — Bybit OMS seam
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from btc_alpha_v3_final import Fixed, SemanticType

from phase5_bundle_store import (
    AtomicBundleStore,
    EventBundle,
)
from phase5_oms_types import (
    ExchangeAdapter,
    ExecutionRecord,
    InvalidTransitionError,
    Order,
    OrderState,
    PositionState,
    POSITION_TRANSITIONS,
    TERMINAL_ORDER_STATES,
    TPLadderLeg,
    TradePlan,
)

_log = logging.getLogger(__name__)

# Constants
SYMBOL = "BTCUSDT"
MAX_ACTIVE_ORDERS = 500
MAX_CONDITIONAL_ORDERS = 10
TP_LADDER_MAX_LEGS = 490
ORDER_TIMEOUT_SECONDS = 10
WS_DISCONNECT_DEADLINE_SECONDS = 30
CLIENT_ORDER_ID_MAX_LEN = 36
EMERGENCY_SL_BPS = 500  # 5% from mark price


# ---------------------------------------------------------------------------
# OMS
# ---------------------------------------------------------------------------

class OMS:
    """
    Bybit Order Management System.

    Owns no bundle store — receives one from the Pod.
    All event commits go through the shared store (same commit_seq chain).
    """

    def __init__(
        self,
        adapter: ExchangeAdapter,
        bundle_store: AtomicBundleStore,
        on_halt: Callable[[str], None],
        pod_id: str = "",
    ) -> None:
        self._adapter = adapter
        self._store = bundle_store
        self._on_halt = on_halt
        self._pod_id = pod_id

        self._position_state = PositionState.FLAT
        self._active_orders: Dict[str, Order] = {}  # client_order_id -> Order
        self._execution_log: List[ExecutionRecord] = []
        self._current_plan: Optional[TradePlan] = None
        self._ws_connected = True
        self._ws_disconnect_ts: Optional[float] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position_state(self) -> PositionState:
        return self._position_state

    @property
    def active_order_count(self) -> int:
        return sum(
            1 for o in self._active_orders.values()
            if not o.is_terminal
        )

    # ------------------------------------------------------------------
    # Position state transition
    # ------------------------------------------------------------------

    def _transition_position(self, new_state: PositionState, reason: str = "") -> None:
        """Validate + perform position state transition + commit EventBundle."""
        valid = POSITION_TRANSITIONS.get(self._position_state, set())
        if new_state not in valid:
            raise InvalidTransitionError(
                f"Position: invalid transition "
                f"{self._position_state.value} -> {new_state.value}"
            )

        old = self._position_state
        self._position_state = new_state

        event = EventBundle(
            format_version=1,
            timestamp=int(time.time()),
            pod_id=self._pod_id,
            event_type="state_transition",
            payload={
                "from": old.value,
                "to": new_state.value,
                "reason": reason,
            },
        )
        try:
            key = self._store.stage_bundle(event)
            self._store.commit_bundle(key)
        except Exception as e:
            _log.error("Failed to commit position transition event: %s", e)

        _log.info("Position: %s -> %s (%s)", old.value, new_state.value, reason)

    # ------------------------------------------------------------------
    # Event commit helper
    # ------------------------------------------------------------------

    def _commit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Stage + commit an EventBundle."""
        event = EventBundle(
            format_version=1,
            timestamp=int(time.time()),
            pod_id=self._pod_id,
            event_type=event_type,
            payload=payload,
        )
        try:
            key = self._store.stage_bundle(event)
            self._store.commit_bundle(key)
        except Exception as e:
            _log.error("Failed to commit event (%s): %s", event_type, e)

    # ------------------------------------------------------------------
    # NO-OP guards
    # ------------------------------------------------------------------

    def _noop_guard(self, action: str) -> bool:
        """Check if action should be suppressed (NO-OP).

        Returns True if action is suppressed.
        """
        if action in ("exit", "close") and self._position_state in (
            PositionState.PENDING_CLOSE,
            PositionState.FLAT,
            PositionState.FAILSAFE,
        ):
            _log.warning(
                "NO-OP guard: %s suppressed in %s state",
                action, self._position_state.value,
            )
            return True
        return False

    def _check_order_limits(self, new_orders: int = 1) -> bool:
        """Check if placing new_orders would exceed limits."""
        current = self.active_order_count
        if current + new_orders > MAX_ACTIVE_ORDERS:
            _log.error(
                "Order limit exceeded: %d active + %d new > %d max",
                current, new_orders, MAX_ACTIVE_ORDERS,
            )
            return False
        return True

    def _generate_client_order_id(self, prefix: str = "oms") -> str:
        """Generate a unique client order ID (max 36 chars)."""
        uid = uuid.uuid4().hex[:24]
        coid = f"{prefix}_{uid}"
        return coid[:CLIENT_ORDER_ID_MAX_LEN]

    # ------------------------------------------------------------------
    # Execute trade plan
    # ------------------------------------------------------------------

    def execute_trade_plan(self, plan: TradePlan) -> bool:
        """
        Execute a trade plan.

        Preflight → PENDING_ENTRY → entry → hard SL → TP ladder → trailing stop → LONG/SHORT.
        Returns True if entry order placed successfully.
        """
        # Preflight validation
        errors = plan.validate()
        if errors:
            _log.error("TradePlan validation failed: %s", errors)
            self._commit_event("failed_attempt", {
                "plan_id": plan.plan_id,
                "errors": errors,
            })
            return False

        if self._position_state != PositionState.FLAT:
            _log.error(
                "Cannot execute plan: position is %s, not FLAT",
                self._position_state.value,
            )
            return False

        # Check order limits
        tp_leg_count = len(plan.tp_ladder)
        total_new = 1 + tp_leg_count  # entry + TP ladder legs
        if not self._check_order_limits(total_new):
            return False

        # Transition to PENDING_ENTRY
        self._transition_position(PositionState.PENDING_ENTRY, f"plan={plan.plan_id}")
        self._current_plan = plan

        # Place entry order with hard SL attached
        entry_coid = self._generate_client_order_id("entry")
        try:
            resp = self._adapter.place_order(
                symbol=plan.symbol,
                side=plan.side,
                order_type=plan.entry_type if plan.entry_type != "Conditional" else "Market",
                qty=plan.qty.value,
                price=plan.entry_price.value if plan.entry_price else None,
                client_order_id=entry_coid,
                stop_loss=plan.stop_loss.value if plan.stop_loss else None,
                take_profit=plan.take_profit.value if plan.take_profit else None,
                sl_trigger_by=plan.sl_trigger_by,
                tp_trigger_by=plan.tp_trigger_by,
            )
        except Exception as e:
            _log.error("Entry order placement failed: %s", e)
            self._transition_position(PositionState.FLAT, f"entry_failed: {e}")
            return False

        entry_order = Order(
            client_order_id=entry_coid,
            symbol=plan.symbol,
            side=plan.side,
            order_type=plan.entry_type if plan.entry_type != "Conditional" else "Market",
            qty=plan.qty,
            price=plan.entry_price,
        )
        self._active_orders[entry_coid] = entry_order

        self._commit_event("order_placed", {
            "plan_id": plan.plan_id,
            "client_order_id": entry_coid,
            "side": plan.side,
            "qty": plan.qty.value,
        })

        return True

    # ------------------------------------------------------------------
    # Handle entry fill
    # ------------------------------------------------------------------

    def handle_entry_fill(self, client_order_id: str, fill_price: Fixed, fill_qty: Fixed) -> None:
        """Handle an entry fill confirmation (WebSocket)."""
        order = self._active_orders.get(client_order_id)
        if order is None:
            _log.warning("Fill for unknown order: %s", client_order_id)
            return

        order.transition(OrderState.FILLED)
        order.avg_fill_price = fill_price
        order.filled_qty = fill_qty

        # Transition position
        if order.side == "Buy":
            self._transition_position(PositionState.LONG, f"entry_fill_{client_order_id}")
        else:
            self._transition_position(PositionState.SHORT, f"entry_fill_{client_order_id}")

        self._execution_log.append(ExecutionRecord(
            timestamp=int(time.time()),
            plan_id=self._current_plan.plan_id if self._current_plan else "",
            order_id=client_order_id,
            side=order.side,
            qty=fill_qty.value,
            price=fill_price.value,
            event="fill",
        ))

        # Place TP ladder legs if any
        if self._current_plan and self._current_plan.tp_ladder:
            self._place_tp_ladder(self._current_plan)

    def _place_tp_ladder(self, plan: TradePlan) -> None:
        """Place TP ladder legs as reduce-only limit orders."""
        close_side = "Sell" if plan.side == "Buy" else "Buy"
        for i, leg in enumerate(plan.tp_ladder):
            coid = self._generate_client_order_id(f"tp{i}")
            leg_qty_value = plan.qty.value * leg.qty_pct // 10000
            if leg_qty_value <= 0:
                continue

            try:
                self._adapter.place_order(
                    symbol=plan.symbol,
                    side=close_side,
                    order_type="Limit",
                    qty=leg_qty_value,
                    price=leg.price.value,
                    reduce_only=True,
                    client_order_id=coid,
                )
                leg_order = Order(
                    client_order_id=coid,
                    symbol=plan.symbol,
                    side=close_side,
                    order_type="Limit",
                    qty=Fixed(value=leg_qty_value, sem=SemanticType.QTY),
                    price=leg.price,
                    reduce_only=True,
                )
                self._active_orders[coid] = leg_order
            except Exception as e:
                _log.error("TP ladder leg %d failed: %s", i, e)

    # ------------------------------------------------------------------
    # Handle exit fill (OCO)
    # ------------------------------------------------------------------

    def handle_exit_fill(self, client_order_id: str, fill_qty: Fixed) -> None:
        """Handle an exit fill.  OCO: cancel all sibling exit orders."""
        order = self._active_orders.get(client_order_id)
        if order is None:
            _log.warning("Exit fill for unknown order: %s", client_order_id)
            return

        if self._noop_guard("exit"):
            return

        order.transition(OrderState.FILLED)
        order.filled_qty = fill_qty

        self._execution_log.append(ExecutionRecord(
            timestamp=int(time.time()),
            plan_id=self._current_plan.plan_id if self._current_plan else "",
            order_id=client_order_id,
            side=order.side,
            qty=fill_qty.value,
            price=order.price.value if order.price else 0,
            event="exit_fill",
        ))

        # OCO: cancel all sibling exit orders
        self._cancel_sibling_exits(client_order_id)

        # Transition to FLAT if position is fully closed
        if self._position_state in (PositionState.LONG, PositionState.SHORT):
            self._transition_position(
                PositionState.PENDING_CLOSE,
                f"exit_fill_{client_order_id}",
            )
        if self._position_state == PositionState.PENDING_CLOSE:
            self._transition_position(PositionState.FLAT, "exit_complete")

        self._current_plan = None

    def _cancel_sibling_exits(self, except_coid: str) -> None:
        """OCO: Cancel all non-terminal reduce-only orders except the filled one."""
        for coid, order in list(self._active_orders.items()):
            if coid == except_coid:
                continue
            if order.is_terminal:
                continue
            if order.reduce_only:
                try:
                    self._adapter.cancel_order(
                        symbol=order.symbol,
                        client_order_id=coid,
                    )
                    order.transition(OrderState.CANCELLED)
                    _log.info("OCO cancelled: %s", coid)
                except (InvalidTransitionError, Exception) as e:
                    _log.error("OCO cancel failed for %s: %s", coid, e)

        self._commit_event("oco_cancel", {
            "trigger_order": except_coid,
            "cancelled": [
                coid for coid, o in self._active_orders.items()
                if o.state == OrderState.CANCELLED and coid != except_coid
            ],
        })

    # ------------------------------------------------------------------
    # Failsafe close
    # ------------------------------------------------------------------

    def failsafe_close(self, reason: str) -> bool:
        """
        5-step failsafe close sequence.

        1. Enter FAILSAFE state
        2. Cancel position-linked TP/SL/TS
        3. Cancel all remaining orders
        4. Place reduce-only market order to flatten
        5. Verify flat (caller must check within 10s)

        Returns True if flatten order placed, False on error.
        """
        # Step 1: Enter FAILSAFE
        if self._position_state != PositionState.FAILSAFE:
            try:
                self._transition_position(PositionState.FAILSAFE, reason)
            except InvalidTransitionError:
                if self._position_state == PositionState.FLAT:
                    _log.info("Already FLAT, no failsafe needed")
                    return True
                _log.error(
                    "Cannot enter FAILSAFE from %s",
                    self._position_state.value,
                )
                return False

        self._commit_event("failsafe_start", {"reason": reason})

        # Step 2: Cancel position-linked TP/SL/TS
        try:
            self._adapter.set_trading_stop(
                symbol=SYMBOL,
                take_profit=0,
                stop_loss=0,
                trailing_stop=0,
            )
        except Exception as e:
            _log.error("Failed to clear trading stops: %s", e)

        # Step 3: Cancel all remaining orders
        try:
            self._adapter.cancel_all_orders(symbol=SYMBOL)
        except Exception as e:
            _log.error("Failed to cancel all orders: %s", e)

        # Mark all active orders as cancelled
        for order in self._active_orders.values():
            if not order.is_terminal:
                try:
                    order.transition(OrderState.CANCELLED)
                except InvalidTransitionError:
                    pass

        # Step 4: Place reduce-only market order
        # Determine close side from position state before FAILSAFE
        close_side = "Sell"  # Default; will be determined by position query
        try:
            pos_info = self._adapter.query_position(symbol=SYMBOL)
            pos_size = pos_info.get("size", 0)
            if pos_size > 0:
                close_side = "Sell"
                close_qty = abs(pos_size)
            elif pos_size < 0:
                close_side = "Buy"
                close_qty = abs(pos_size)
            else:
                # Already flat
                self._transition_position(PositionState.FLAT, "already_flat_in_failsafe")
                return True
        except Exception as e:
            _log.error("Failed to query position for failsafe: %s", e)
            return False

        coid = self._generate_client_order_id("failsafe")
        try:
            self._adapter.place_order(
                symbol=SYMBOL,
                side=close_side,
                order_type="Market",
                qty=close_qty,
                price=None,
                reduce_only=True,
                client_order_id=coid,
            )
        except Exception as e:
            _log.error("Failsafe market order failed: %s", e)
            return False

        self._commit_event("failsafe_close_order", {
            "client_order_id": coid,
            "side": close_side,
            "qty": close_qty,
        })

        # Step 5: Transition to FLAT (caller must verify within 10s)
        self._transition_position(PositionState.FLAT, "failsafe_close_complete")

        return True

    # ------------------------------------------------------------------
    # WebSocket disconnect handling
    # ------------------------------------------------------------------

    def handle_ws_disconnect(self) -> None:
        """Handle WebSocket disconnection.  Start 30s deadline timer."""
        self._ws_connected = False
        self._ws_disconnect_ts = time.time()
        _log.warning("WebSocket disconnected — starting 30s deadline")

        self._commit_event("health_change", {
            "component": "websocket",
            "state": "DISCONNECTED",
        })

    def handle_ws_reconnect(self) -> None:
        """Handle WebSocket reconnection before deadline."""
        self._ws_connected = True
        self._ws_disconnect_ts = None
        _log.info("WebSocket reconnected")

        self._commit_event("health_change", {
            "component": "websocket",
            "state": "CONNECTED",
        })

    def check_ws_deadline(self) -> None:
        """Check if 30s disconnect deadline has been exceeded."""
        if self._ws_connected or self._ws_disconnect_ts is None:
            return

        elapsed = time.time() - self._ws_disconnect_ts
        if elapsed >= WS_DISCONNECT_DEADLINE_SECONDS:
            _log.error(
                "WebSocket disconnect deadline exceeded (%.1fs) — "
                "executing failsafe + HALT",
                elapsed,
            )
            self.failsafe_close("ws_disconnect_deadline_exceeded")
            self._on_halt("WebSocket disconnected > 30s, position flattened")

    # ------------------------------------------------------------------
    # Startup reconciliation
    # ------------------------------------------------------------------

    def startup_reconciliation(self) -> bool:
        """
        Run on OMS startup.

        1. Query positions + orders via REST.
        2. Attempt to reconstruct TradePlan.
        3. On failure: ensure protective SL exists, then HALT.

        Returns True if reconciliation succeeded, False if HALT needed.
        """
        try:
            pos_info = self._adapter.query_position(symbol=SYMBOL)
            active_orders = self._adapter.query_active_orders(symbol=SYMBOL)
        except Exception as e:
            _log.error("Startup reconciliation REST query failed: %s", e)
            self._on_halt(f"Reconciliation query failed: {e}")
            return False

        pos_size = pos_info.get("size", 0)

        # No position — nothing to reconcile
        if pos_size == 0:
            self._position_state = PositionState.FLAT
            _log.info("Startup: no position, FLAT")
            return True

        # Has position — try to reconstruct TradePlan
        reconstructed = self._try_reconstruct_plan(pos_info, active_orders)
        if reconstructed:
            # Set position state
            if pos_size > 0:
                self._position_state = PositionState.LONG
            else:
                self._position_state = PositionState.SHORT
            _log.info("Startup: TradePlan reconstructed, resuming monitoring")
            return True

        # Reconstruction failed — ensure protective SL
        has_sl = any(
            o.get("stopLoss") and float(o.get("stopLoss", "0")) > 0
            for o in active_orders
        )

        if not has_sl:
            _log.warning("No protective SL found — placing emergency SL")
            try:
                mark_price = pos_info.get("markPrice", 0)
                if mark_price > 0:
                    # 5% from mark price
                    if pos_size > 0:
                        sl_price = int(mark_price * (10000 - EMERGENCY_SL_BPS) / 10000)
                    else:
                        sl_price = int(mark_price * (10000 + EMERGENCY_SL_BPS) / 10000)
                    self._adapter.set_trading_stop(
                        symbol=SYMBOL,
                        take_profit=None,
                        stop_loss=sl_price,
                        trailing_stop=None,
                    )
                    _log.info("Emergency SL placed at %d", sl_price)
            except Exception as e:
                _log.error("Failed to place emergency SL: %s", e)

        self._commit_event("reconciliation_failed", {
            "pos_size": pos_size,
            "active_orders": len(active_orders),
            "has_sl": has_sl,
        })

        self._on_halt("Startup reconciliation failed — operator intervention required")
        return False

    def _try_reconstruct_plan(
        self,
        pos_info: Dict[str, Any],
        active_orders: List[Dict[str, Any]],
    ) -> bool:
        """Attempt to reconstruct TradePlan from active orders."""
        # Look for orders with our clientOrderId prefix pattern
        our_orders = [
            o for o in active_orders
            if str(o.get("orderLinkId", "")).startswith(("entry_", "tp", "oms_"))
        ]
        if not our_orders:
            return False

        # Reconstruct plan (simplified)
        pos_size = pos_info.get("size", 0)
        side = "Buy" if pos_size > 0 else "Sell"
        plan = TradePlan(
            plan_id=f"reconstructed_{uuid.uuid4().hex[:8]}",
            symbol=SYMBOL,
            side=side,
            qty=Fixed(value=abs(pos_size), sem=SemanticType.QTY),
            stop_loss=None,  # Will be found from orders
        )

        # Find SL from active orders
        for o in our_orders:
            sl_val = o.get("stopLoss")
            if sl_val and float(sl_val) > 0:
                plan.stop_loss = Fixed(value=int(float(sl_val) * 100), sem=SemanticType.PRICE)
                break

        self._current_plan = plan
        _log.info("Reconstructed plan: %s", plan.plan_id)
        return True

    # ------------------------------------------------------------------
    # Error 10003 idempotency
    # ------------------------------------------------------------------

    def handle_error_10003(self, client_order_id: str) -> bool:
        """
        Handle Bybit error 10003 (duplicate request).

        1. Query order by clientOrderId.
        2. If exists: treat as success (idempotent).
        3. If not: the error is genuine.

        Returns True if order exists (idempotent success), False otherwise.
        """
        _log.warning("Error 10003 for %s — checking idempotency", client_order_id)

        try:
            order_info = self._adapter.query_order(
                symbol=SYMBOL,
                client_order_id=client_order_id,
            )
            if order_info is not None:
                _log.info(
                    "Error 10003: order %s exists on exchange (idempotent success)",
                    client_order_id,
                )
                self._commit_event("idempotent_success", {
                    "client_order_id": client_order_id,
                    "error_code": 10003,
                })
                return True
        except Exception as e:
            _log.error("Query after 10003 failed: %s", e)

        _log.error("Error 10003: order %s NOT found — genuine error", client_order_id)
        return False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """OMS status for CLI/monitoring."""
        return {
            "position_state": self._position_state.value,
            "active_orders": self.active_order_count,
            "ws_connected": self._ws_connected,
            "current_plan": self._current_plan.plan_id if self._current_plan else None,
            "execution_log_length": len(self._execution_log),
        }
