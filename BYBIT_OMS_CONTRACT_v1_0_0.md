# BYBIT OMS CONTRACT v1.0.0

## BTCUSDT USDT Perpetuals Order Management System

**Status**: FROZEN - Ready for Delegation  
**Scope**: BTCUSDT USDT Perpetuals (Linear), One-Way Mode  
**Authority Class**: Same as PHASE5_INTEGRATION_SEAMS.md

**PSEUDOCODE DISCLAIMER**: All code blocks in this document are illustrative pseudocode only and are not normative. Implementation details belong in implementation guides.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | — | Initial frozen contract |

---

## 1. Core Principles

### 1.1 Bybit UI Parity
System MUST expose all capabilities available in Bybit UI programmatically.
This includes exchange-native trailing stops, TP/SL, and ladders.

### 1.2 Exchange-Side Protection is Foundational
Default to exchange-side protection where available.
Client-side mechanisms are additive supplements, never forced replacements.

### 1.3 TradePlan Expresses Intent, OMS Chooses Implementation
TradePlan declares desired behavior and constraints.
OMS selects exchange-native vs client-side implementation.

---

## 2. Frozen Decisions

### 2.1 TP Ladder
**Implementation**: Client-side reduce-only limit orders  
**Reason**: Exchange Partial TP/SL requires tpSize=slSize (incompatible with asymmetric ladders)  
**Limit**: Up to 490 legs (500 order limit - 10 buffer)  
**Non-Goal**: Partial TP/SL (tpslMode=Partial) prohibited in v1.0

### 2.2 Hard SL (Full Position)
**Implementation**: Exchange-native via /v5/position/trading-stop (tpslMode=Full)  
**Order Type**: Market only (Limit NOT supported in Full mode)  
**Trigger Source**: MarkPrice (default), IndexPrice, LastPrice (selectable via slTriggerBy)  
**Mandatory**: EVERY live position MUST have exchange-side hard SL active at all times

### 2.3 Hard TP (Full Position)
**Implementation**: Exchange-native via /v5/position/trading-stop (tpslMode=Full)  
**Order Type**: Market only (Limit NOT supported in Full mode)  
**Trigger Source**: MarkPrice (default), IndexPrice, LastPrice (selectable via tpTriggerBy)

### 2.4 Trailing Stop
**Default**: Exchange-native (Bybit UI parity)  
**Implementation**: POST /v5/position/trading-stop (trailingStop + activePrice)

**Risk Flag**: NON_DETERMINISTIC_VS_BACKTEST  
**Reason**: Trigger source not documented by Bybit (may use LastPrice, not MarkPrice)  
**Contract**: TradePlan MUST set `acceptNonDeterministicTrigger: true` to use exchange-native trailing

**Alternative**: Client-side MarkPrice trailing (deterministic, optional)  
**Requires**: Layer 1 static exchange-side SL as outage backstop  
**Use Case**: Research/backtest where determinism required

### 2.5 Partial TP/SL
**Status**: PROHIBITED in v1.0  
**Reason**: tpSize=slSize constraint, max 20 orders (error 110061), binding relationship pitfalls  
**Future**: May support for symmetric bracket strategies only

---

## 3. Hard Limits (Bybit-Imposed)

Encode and enforce these limits:

- **Max active orders per symbol**: 500
- **Max conditional orders per symbol**: 10
- **Partial TP/SL orders**: 20 max (error 110061) - not used in v1.0
- **clientOrderId max length**: 36 characters
- **Full-mode TP/SL**: Market order type only
- **reduceOnly orders**: Cannot attach TP/SL

---

## 4. Mandatory Behaviors

### 4.1 Exchange-Side Hard SL (Always Active)
- REQUIRED for all live positions
- Set at entry via /v5/order/create (takeProfit/stopLoss attached)
- OR immediately after entry via /v5/position/trading-stop
- Never removed until position is closed
- Client-side logic is additive only

### 4.2 OCO Behavior on Exits
- REQUIRED: Cancel sibling exits when any exit fills
- Implementation: Explicit cancel calls + WebSocket confirmation
- No opt-out: OCO is mandatory, not configurable

### 4.3 Fail-Safe Close Sequence
**Trigger Conditions**: State divergence, order timeout >10s, WebSocket down >30s

**Sequence** (deterministic, logged):
1. Enter FAILSAFE position state (blocks all new orders except close)
2. Cancel position-linked TP/SL/TS: POST /v5/position/trading-stop with takeProfit=0, stopLoss=0, trailingStop=0
3. Cancel all remaining orders: POST /v5/order/cancel-all (no orderFilter)
4. Place reduce-only market order: POST /v5/order/create (reduceOnly=true)
5. Verify flat position within 10 seconds via WebSocket
6. If not flat: HALT with error

### 4.4 WebSocket Disconnect Handling
**On Disconnect**:
- Stop placing new orders immediately
- Start REST polling (1-second interval)
- Attempt reconnect with exponential backoff
- Set 30-second deadline

**On 30s Deadline**:
- Flatten all positions via fail-safe close
- Enter HALT state
- Require operator intervention to resume

**On Reconnect**:
- Reconcile all positions
- Stop REST polling
- Resume normal operations

### 4.5 Startup Reconciliation
**On OMS Startup**:
1. Query positions + active orders + TP/SL via REST
2. Attempt to reconstruct TradePlan from orders (match by clientOrderId pattern)
3. If reconstruction succeeds: Resume monitoring
4. If reconstruction fails:
   - Check if position has protective SL
   - If no SL: Place emergency SL (5% from mark price)
   - Enter HALT state, await operator decision (flatten vs adopt vs monitor)

### 4.6 Idempotency and Duplicate Requests
**Error Code 10003**: "Invalid duplicate request"

**Handling**:
- Query order by clientOrderId via REST
- If exists: Treat as success (idempotent)
- If not exists: Retry with exponential backoff (max 3 attempts)
- If still fails: Log error and propagate

---

## 5. State Machines

### 5.1 Order States
- PENDING_ACK: Sent to exchange, awaiting confirmation
- ACTIVE: Confirmed active by WebSocket
- PARTIALLY_FILLED: Partial execution
- FILLED: Complete execution
- CANCELLED: Cancelled by user or system
- REJECTED: Rejected by exchange
- EXPIRED: Order expired

### 5.2 Position States
- FLAT: No position
- LONG: Long position active
- SHORT: Short position active
- PENDING_ENTRY: Entry order active, awaiting fill
- PENDING_CLOSE: Close order active, awaiting fill
- FAILSAFE: Emergency closure in progress (blocks all new orders except close)

### 5.3 State Transitions
All transitions logged to evidence chain (if enabled).

---

## 6. Risk Guards and Race Conditions

### 6.1 NO-OP Guards (Prevent Race Conditions)
**Before placing exit orders**:
- Check position state not in [PENDING_CLOSE, FLAT, FAILSAFE]
- Check exit qty ≤ current position qty
- Check clientOrderId not already in use

**On order timeout** (>10s without confirmation):
- Query order status via REST as fallback
- If still unknown: Enter FAILSAFE state

### 6.2 Asynchronous Confirmation
- REST API ack is "accepted only", not "confirmed"
- Wait for WebSocket confirmation before state transition
- Timeout: 10 seconds for order/position updates
- If timeout: Query via REST, then HALT if still unknown

---

## 7. Preflight Validation

**Required checks before TradePlan execution**:
- Order count limits (500 active, 10 conditional)
- Margin requirement calculation
- Price/qty precision (tick size, lot size)
- Min/max order size compliance
- Duplicate clientOrderId check
- TP ladder sanity (total qtyPct ≤ 100%)
- Full-mode order type validation (Market-only for TP/SL)

---

## 8. Execution Record Requirements

**Log for each TradePlan execution**:
- TradePlan ID, symbol, timestamp
- Implementation choices (exchange-native vs client-side for each component)
- All order IDs (exchange + client)
- State transitions with timestamps
- Fills with prices and quantities
- Any failures or degraded states

---

## 9. Non-Goals (v1.0)

**Out of Scope**:
- Partial TP/SL (tpslMode=Partial)
- Multi-symbol correlation trading
- Hedge mode (two-way positions)
- Options, spot, inverse perpetuals
- Cross-margin mode (isolated margin only)
- Margin trading, leverage adjustments

---

## 10. Open Questions (Verify Before Production)

**Requires Bybit Support Verification**:

1. **Trailing Stop Trigger Source**: What is default triggerBy for trailingStop parameter? (LastPrice/MarkPrice/IndexPrice?)
2. **Partial TP/SL Limit**: Is error 110061 "20 TP/SL" = 20 total or 20 per side?
3. **TP/SL Auto-Cancel**: Do position-linked TP/SL automatically cancel when position closes?

**Current Mitigations**:
1. Expose exchange-native trailing with explicit risk flag (acceptNonDeterministicTrigger)
2. Assume 20 total (conservative)
3. Assume auto-cancel (verify with testing)

---

## END OF CONTRACT v1.0.0

**Total Lines**: 280  
**Status**: FROZEN - Ready for Implementation  
**Next Step**: Implement test harness with race conditions, edge cases, and limit violations

