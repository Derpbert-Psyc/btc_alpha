"""
Phase 5 — Bybit OMS Types

Order/Position state machines, TradePlan, ExchangeAdapter Protocol.
All monetary values as Fixed.  No Decimal in state machines or persisted records.

References:
    BYBIT_OMS_CONTRACT_v1_0_0.md — Complete document
    PHASE5_INTEGRATION_SEAMS_v1_2_6.md:484-518 — Bybit OMS seam
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple

from btc_alpha_v3_final import Fixed, SemanticType


# ---------------------------------------------------------------------------
# Order State Machine
# ---------------------------------------------------------------------------

class OrderState(Enum):
    PENDING_ACK = "PENDING_ACK"
    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


# Valid transitions: from_state -> set of valid to_states
ORDER_TRANSITIONS: Dict[OrderState, set] = {
    OrderState.PENDING_ACK: {
        OrderState.ACTIVE,
        OrderState.REJECTED,
        OrderState.FILLED,
        OrderState.CANCELLED,
    },
    OrderState.ACTIVE: {
        OrderState.PARTIALLY_FILLED,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
    },
    OrderState.PARTIALLY_FILLED: {
        OrderState.PARTIALLY_FILLED,
        OrderState.FILLED,
        OrderState.CANCELLED,
    },
    # Terminal states — no transitions
    OrderState.FILLED: set(),
    OrderState.CANCELLED: set(),
    OrderState.REJECTED: set(),
    OrderState.EXPIRED: set(),
}

TERMINAL_ORDER_STATES = {
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.REJECTED,
    OrderState.EXPIRED,
}


# ---------------------------------------------------------------------------
# Position State Machine
# ---------------------------------------------------------------------------

class PositionState(Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"
    PENDING_ENTRY = "PENDING_ENTRY"
    PENDING_CLOSE = "PENDING_CLOSE"
    FAILSAFE = "FAILSAFE"


POSITION_TRANSITIONS: Dict[PositionState, set] = {
    PositionState.FLAT: {PositionState.PENDING_ENTRY},
    PositionState.PENDING_ENTRY: {
        PositionState.LONG,
        PositionState.SHORT,
        PositionState.FLAT,
    },
    PositionState.LONG: {
        PositionState.PENDING_CLOSE,
        PositionState.FAILSAFE,
    },
    PositionState.SHORT: {
        PositionState.PENDING_CLOSE,
        PositionState.FAILSAFE,
    },
    PositionState.PENDING_CLOSE: {
        PositionState.FLAT,
        PositionState.LONG,
        PositionState.SHORT,
        PositionState.FAILSAFE,
    },
    # FAILSAFE → FLAT only
    PositionState.FAILSAFE: {PositionState.FLAT},
}


# ---------------------------------------------------------------------------
# Order dataclass
# ---------------------------------------------------------------------------

class InvalidTransitionError(Exception):
    """Raised on invalid state machine transition."""


@dataclass
class Order:
    """Represents a single exchange order with state tracking."""
    client_order_id: str
    symbol: str
    side: Literal["Buy", "Sell"]
    order_type: Literal["Market", "Limit"]
    qty: Fixed                          # QTY
    price: Optional[Fixed]              # PRICE (None for Market orders)
    reduce_only: bool = False
    state: OrderState = OrderState.PENDING_ACK
    filled_qty: Fixed = field(default_factory=lambda: Fixed(value=0, sem=SemanticType.QTY))
    avg_fill_price: Optional[Fixed] = None  # PRICE
    created_ts: int = field(default_factory=lambda: int(time.time()))
    exchange_order_id: Optional[str] = None

    def transition(self, new_state: OrderState) -> None:
        """Validate and perform state transition."""
        valid = ORDER_TRANSITIONS.get(self.state, set())
        if new_state not in valid:
            raise InvalidTransitionError(
                f"Order {self.client_order_id}: "
                f"invalid transition {self.state.value} -> {new_state.value}"
            )
        self.state = new_state

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_ORDER_STATES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "qty": self.qty.value,
            "price": self.price.value if self.price else None,
            "reduce_only": self.reduce_only,
            "state": self.state.value,
            "filled_qty": self.filled_qty.value,
            "avg_fill_price": self.avg_fill_price.value if self.avg_fill_price else None,
            "created_ts": self.created_ts,
            "exchange_order_id": self.exchange_order_id,
        }


# ---------------------------------------------------------------------------
# TP Ladder Leg
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TPLadderLeg:
    """One leg of a take-profit ladder."""
    price: Fixed                        # PRICE
    qty_pct: int                        # Percentage of position (basis points * 100)
    order: Optional[Order] = None       # Associated order (set after placement)


# ---------------------------------------------------------------------------
# TradePlan
# ---------------------------------------------------------------------------

@dataclass
class TradePlan:
    """
    Express trading intent.  All monetary values as Fixed.

    Validation:
        - qty must be positive QTY
        - stop_loss must be a valid PRICE
        - TP ladder qty_pct sum must be <= 10000 (100%)
        - Max 490 TP ladder legs
    """
    plan_id: str
    symbol: str = "BTCUSDT"
    side: Literal["Buy", "Sell"] = "Buy"
    entry_type: Literal["Market", "Limit", "Conditional"] = "Market"
    entry_price: Optional[Fixed] = None         # PRICE
    qty: Fixed = field(default_factory=lambda: Fixed(value=0, sem=SemanticType.QTY))
    stop_loss: Optional[Fixed] = None           # PRICE (MANDATORY)
    sl_trigger_by: str = "MarkPrice"
    take_profit: Optional[Fixed] = None         # PRICE (Full mode TP)
    tp_trigger_by: str = "MarkPrice"
    tp_ladder: List[TPLadderLeg] = field(default_factory=list)
    trailing_stop: Optional[Fixed] = None       # PRICE distance
    active_price: Optional[Fixed] = None        # PRICE
    accept_non_deterministic_trigger: bool = False
    margin_mode: str = "Isolated"

    def validate(self) -> List[str]:
        """Validate plan.  Returns list of errors (empty = valid)."""
        errors: List[str] = []

        if self.qty.sem != SemanticType.QTY or self.qty.value <= 0:
            errors.append("qty must be positive QTY")

        if self.stop_loss is None:
            errors.append("stop_loss is MANDATORY (hard SL required)")
        elif self.stop_loss.sem != SemanticType.PRICE:
            errors.append("stop_loss must be PRICE type")

        if self.entry_price is not None and self.entry_price.sem != SemanticType.PRICE:
            errors.append("entry_price must be PRICE type")

        if self.take_profit is not None and self.take_profit.sem != SemanticType.PRICE:
            errors.append("take_profit must be PRICE type")

        # TP ladder validation
        if len(self.tp_ladder) > 490:
            errors.append(f"TP ladder has {len(self.tp_ladder)} legs (max 490)")

        total_pct = sum(leg.qty_pct for leg in self.tp_ladder)
        if total_pct > 10000:
            errors.append(
                f"TP ladder total qty_pct={total_pct} exceeds 10000 (100%)"
            )

        # Trailing stop validation
        if self.trailing_stop is not None:
            if not self.accept_non_deterministic_trigger:
                errors.append(
                    "trailing_stop requires accept_non_deterministic_trigger=True"
                )

        # Margin mode
        if self.margin_mode != "Isolated":
            errors.append(f"Only Isolated margin supported, got {self.margin_mode}")

        return errors


# ---------------------------------------------------------------------------
# Execution Record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionRecord:
    """Immutable record of an execution event."""
    timestamp: int
    plan_id: str
    order_id: str
    side: str
    qty: int                # QTY scaled int
    price: int              # PRICE scaled int
    event: str              # "fill", "cancel", "reject", etc.
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ExchangeAdapter Protocol
# ---------------------------------------------------------------------------

class ExchangeAdapter(Protocol):
    """Protocol for exchange interaction (Bybit-specific)."""

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: int,
        price: Optional[int],
        *,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
        stop_loss: Optional[int] = None,
        take_profit: Optional[int] = None,
        sl_trigger_by: str = "MarkPrice",
        tp_trigger_by: str = "MarkPrice",
    ) -> Dict[str, Any]:
        """Place an order.  Returns exchange response dict."""
        ...

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel a single order."""
        ...

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel all orders for a symbol."""
        ...

    def set_trading_stop(
        self,
        symbol: str,
        take_profit: Optional[int],
        stop_loss: Optional[int],
        trailing_stop: Optional[int],
    ) -> Dict[str, Any]:
        """Set/clear position-linked TP/SL/TS."""
        ...

    def query_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Query a single order by ID.  Returns None if not found."""
        ...

    def query_position(self, symbol: str) -> Dict[str, Any]:
        """Query current position for a symbol."""
        ...

    def query_active_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Query all active (non-terminal) orders."""
        ...
