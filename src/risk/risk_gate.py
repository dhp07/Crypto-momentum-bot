"""
Risk gate — the mandatory checkpoint between strategy signals and order execution.

Every signal the strategy emits passes through here. The gate either approves it
as a sized order or rejects it with a reason. Nothing reaches the executor without
clearing this layer. This is the safety core of the bot.

Sizing model: 3% ALLOCATION. Each position is sized to a fraction
(position_allocation_pct) of current account equity — ~$30 on a $1,000 account.
The 1.5x ATR stop from the signal rides along as the loss-limiter but does NOT
drive the size.

The gate is pure: it takes the current risk STATE (equity, open positions, today's
realized P&L, consecutive losses) as input rather than tracking it itself. In the
live bot, that state comes from Redis; in tests, it's passed directly. This keeps
the gate deterministic and easy to verify.

On fees: the gate logs the round-trip fee cost for transparency but does NOT veto
trades on fee grounds. Whether a trade is "worth" its fees depends on expected
profit, which the gate cannot know — that's a backtest/strategy concern, not a
gate concern. The gate enforces only well-defined limits: balance, minimum order
size, and the circuit breakers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from src.strategy.base import Signal


logger = logging.getLogger(__name__)


class RejectReason(str, Enum):
    """Why a signal was rejected. Logged and surfaced for monitoring."""

    KILL_SWITCH = "daily_kill_switch_tripped"
    LOSS_PAUSE = "consecutive_loss_pause_active"
    ALREADY_HELD = "position_already_open_for_pair"
    MAX_POSITIONS = "max_concurrent_positions_reached"
    BELOW_MIN_SIZE = "below_exchange_minimum_size"
    INSUFFICIENT_BALANCE = "insufficient_free_balance"


@dataclass(frozen=True)
class RiskState:
    """
    Snapshot of everything the gate needs to make a decision.

    In the live bot this is assembled from Redis at decision time. The gate never
    mutates it — it only reads.
    """

    equity: Decimal                      # Current total account value (USD)
    free_balance: Decimal                # USD not tied up in open positions
    open_positions: dict[str, Decimal]   # pair -> notional currently held
    realized_pnl_today: Decimal          # Today's closed-trade P&L (negative = loss)
    consecutive_losses: int              # Count of losses in a row

    @property
    def num_open_positions(self) -> int:
        return len(self.open_positions)


@dataclass(frozen=True)
class ApprovedOrder:
    """A signal that cleared the gate, with a concrete size to execute."""

    signal: Signal
    notional_usd: Decimal     # Dollar size of the position to open
    quantity: Decimal         # notional / price — amount of the asset to buy
    binding_constraint: str   # Which limit set the size, for transparency
    est_round_trip_fee: Decimal  # Estimated buy+sell fee cost, logged not vetoed


@dataclass(frozen=True)
class RiskDecision:
    """The gate's verdict: either an approved order or a rejection."""

    approved: bool
    order: ApprovedOrder | None = None
    reject_reason: RejectReason | None = None
    detail: str = ""


@dataclass(frozen=True)
class RiskParams:
    """Risk configuration, normally loaded from the validated BotConfig."""

    position_allocation_pct: Decimal     # e.g. 0.03 -> 3% of equity per position
    max_concurrent_positions: int        # e.g. 3
    daily_loss_kill_switch_pct: Decimal  # e.g. 0.09 -> halt if down 9% on the day
    consecutive_loss_limit: int          # e.g. 5 -> pause after 5 losses in a row
    min_order_usd: Decimal               # exchange/practical minimum, e.g. 1.00
    taker_fee_pct: Decimal               # e.g. 0.012 for 1.2% per side

    @classmethod
    def from_config(cls, config) -> "RiskParams":
        """
        Build from the validated BotConfig (config/settings.py).

        Note: we reuse config.sizing.risk_per_trade as the ALLOCATION fraction.
        The field is named 'risk_per_trade' in the YAML, but in the allocation
        model it means 'fraction of equity per position'. 3% either way.
        """
        return cls(
            position_allocation_pct=Decimal(str(config.sizing.risk_per_trade)),
            max_concurrent_positions=config.sizing.max_concurrent_positions,
            daily_loss_kill_switch_pct=Decimal(str(config.risk.daily_loss_kill_switch)),
            consecutive_loss_limit=config.risk.consecutive_loss_limit,
            min_order_usd=Decimal(str(config.sizing.min_position_usd)),
            taker_fee_pct=Decimal(str(config.execution.taker_fee_bps)) / Decimal("10000"),
        )


class RiskGate:
    """The mandatory risk checkpoint. Construct with params; call evaluate per signal."""

    def __init__(self, params: RiskParams) -> None:
        self._p = params
        logger.info(
            f"RiskGate initialized: allocation={params.position_allocation_pct}, "
            f"max_positions={params.max_concurrent_positions}, "
            f"kill_switch={params.daily_loss_kill_switch_pct}, "
            f"loss_limit={params.consecutive_loss_limit}, "
            f"min_order=${params.min_order_usd}, taker_fee={params.taker_fee_pct}"
        )

    def evaluate(self, signal: Signal, state: RiskState) -> RiskDecision:
        """
        Run a signal through every check. Returns an approved order or a rejection.

        Checks run cheapest-veto-first: circuit breakers, then position checks,
        then sizing. The first failure wins; nothing is sized until all vetoes pass.
        """
        pair = signal.product_id

        # --- Circuit breaker 1: daily kill switch ---
        # If today's realized loss has reached the kill-switch threshold, halt all
        # new entries. (Live version uses day-start equity from Redis as the base;
        # here we use current equity, which is close enough for the check.)
        kill_threshold = -(self._p.daily_loss_kill_switch_pct * state.equity)
        if state.realized_pnl_today <= kill_threshold:
            detail = (
                f"daily P&L {state.realized_pnl_today} <= kill threshold {kill_threshold} "
                f"({self._p.daily_loss_kill_switch_pct:.0%} of {state.equity})"
            )
            logger.warning(f"REJECT {pair}: KILL_SWITCH — {detail}")
            return RiskDecision(False, reject_reason=RejectReason.KILL_SWITCH, detail=detail)

        # --- Circuit breaker 2: consecutive-loss pause ---
        if state.consecutive_losses >= self._p.consecutive_loss_limit:
            detail = (
                f"{state.consecutive_losses} consecutive losses "
                f">= limit {self._p.consecutive_loss_limit}"
            )
            logger.warning(f"REJECT {pair}: LOSS_PAUSE — {detail}")
            return RiskDecision(False, reject_reason=RejectReason.LOSS_PAUSE, detail=detail)

        # --- Position check 1: already holding this pair ---
        if pair in state.open_positions:
            detail = f"already holding {pair} (notional {state.open_positions[pair]})"
            logger.info(f"REJECT {pair}: ALREADY_HELD — {detail}")
            return RiskDecision(False, reject_reason=RejectReason.ALREADY_HELD, detail=detail)

        # --- Position check 2: max concurrent positions ---
        if state.num_open_positions >= self._p.max_concurrent_positions:
            detail = (
                f"{state.num_open_positions} open positions "
                f">= max {self._p.max_concurrent_positions}"
            )
            logger.info(f"REJECT {pair}: MAX_POSITIONS — {detail}")
            return RiskDecision(False, reject_reason=RejectReason.MAX_POSITIONS, detail=detail)

        # --- Sizing (3% allocation) ---
        target_notional = self._p.position_allocation_pct * state.equity

        # You can't deploy more than your free balance, even if 3% wants more.
        if target_notional <= state.free_balance:
            notional = target_notional
            binding = "allocation"  # the 3% rule set the size
        else:
            notional = state.free_balance
            binding = "free_balance"  # account ran out before the 3% rule did

        # Balance can't even cover the minimum order
        if state.free_balance < self._p.min_order_usd:
            detail = (
                f"free balance {state.free_balance} < min order {self._p.min_order_usd}"
            )
            logger.info(f"REJECT {pair}: INSUFFICIENT_BALANCE — {detail}")
            return RiskDecision(
                False, reject_reason=RejectReason.INSUFFICIENT_BALANCE, detail=detail
            )

        # Sized position is below the minimum tradeable size
        if notional < self._p.min_order_usd:
            detail = (
                f"sized notional {notional} < min order {self._p.min_order_usd} "
                f"(binding: {binding})"
            )
            logger.info(f"REJECT {pair}: BELOW_MIN_SIZE — {detail}")
            return RiskDecision(
                False, reject_reason=RejectReason.BELOW_MIN_SIZE, detail=detail
            )

        # --- Approved: compute quantity and fee estimate ---
        quantity = notional / signal.price
        # Round-trip fee = fee on the buy + fee on the eventual sell
        est_fee = notional * self._p.taker_fee_pct * Decimal("2")

        order = ApprovedOrder(
            signal=signal,
            notional_usd=notional,
            quantity=quantity,
            binding_constraint=binding,
            est_round_trip_fee=est_fee,
        )

        logger.info(
            f"APPROVE {pair}: notional=${notional:.2f} qty={quantity:.8f} "
            f"(binding: {binding}; 3% of ${state.equity:.2f} would be "
            f"${target_notional:.2f}, free balance ${state.free_balance:.2f}); "
            f"est round-trip fee ${est_fee:.4f}"
        )
        return RiskDecision(True, order=order)


# ============================================================
# Self-test
# ============================================================


if __name__ == "__main__":
    """
    Verify every veto fires and sizing is exact:
        python3 -m src.risk.risk_gate
    """
    import sys
    from datetime import datetime, timezone

    from src.strategy.base import Signal, SignalType

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    failed = False

    def expect(name: str, condition: bool) -> None:
        global failed
        if condition:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            failed = True

    # Known params: $1,000 account behavior
    params = RiskParams(
        position_allocation_pct=Decimal("0.03"),
        max_concurrent_positions=3,
        daily_loss_kill_switch_pct=Decimal("0.09"),
        consecutive_loss_limit=5,
        min_order_usd=Decimal("1.00"),
        taker_fee_pct=Decimal("0.012"),
    )
    gate = RiskGate(params)

    def make_signal(pair: str = "BTC-USD", price: float = 73000.0) -> Signal:
        return Signal(
            signal_type=SignalType.ENTER_LONG,
            product_id=pair,
            timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
            price=Decimal(str(price)),
            stop_price=Decimal(str(price - 120)),
            reason="test signal",
        )

    def clean_state(**overrides) -> RiskState:
        base = dict(
            equity=Decimal("1000"),
            free_balance=Decimal("1000"),
            open_positions={},
            realized_pnl_today=Decimal("0"),
            consecutive_losses=0,
        )
        base.update(overrides)
        return RiskState(**base)

    print("Testing RiskGate (params model a $1,000 account, 3% allocation)...")
    print()

    # --- Test 1: clean approve, exact sizing ---
    print("Test 1: Clean signal -> approved, sized to 3% = $30")
    d = gate.evaluate(make_signal(), clean_state())
    expect("approved", d.approved)
    if d.order:
        expect("notional == $30.00", d.order.notional_usd == Decimal("30.00"))
        expect("binding == allocation", d.order.binding_constraint == "allocation")
        expected_qty = Decimal("30.00") / Decimal("73000.0")
        expect("quantity == 30/73000", d.order.quantity == expected_qty)
    print()

    # --- Test 2: kill switch ---
    print("Test 2: Down 10% today (> 9% kill switch) -> rejected")
    d = gate.evaluate(make_signal(), clean_state(realized_pnl_today=Decimal("-100")))
    expect("rejected", not d.approved)
    expect("reason KILL_SWITCH", d.reject_reason == RejectReason.KILL_SWITCH)
    print()

    # --- Test 3: consecutive loss pause ---
    print("Test 3: 5 consecutive losses (>= limit) -> rejected")
    d = gate.evaluate(make_signal(), clean_state(consecutive_losses=5))
    expect("rejected", not d.approved)
    expect("reason LOSS_PAUSE", d.reject_reason == RejectReason.LOSS_PAUSE)
    print()

    # --- Test 4: already holding the pair ---
    print("Test 4: Already holding BTC-USD -> rejected")
    d = gate.evaluate(
        make_signal("BTC-USD"),
        clean_state(open_positions={"BTC-USD": Decimal("30")}),
    )
    expect("rejected", not d.approved)
    expect("reason ALREADY_HELD", d.reject_reason == RejectReason.ALREADY_HELD)
    print()

    # --- Test 5: max concurrent positions ---
    print("Test 5: 3 positions already open -> rejected")
    d = gate.evaluate(
        make_signal("LINK-USD"),
        clean_state(open_positions={
            "ETH-USD": Decimal("30"),
            "SOL-USD": Decimal("30"),
            "AVAX-USD": Decimal("30"),
        }),
    )
    expect("rejected", not d.approved)
    expect("reason MAX_POSITIONS", d.reject_reason == RejectReason.MAX_POSITIONS)
    print()

    # --- Test 6: insufficient balance (below min order) ---
    print("Test 6: Free balance $0.50 (< $1 min) -> rejected")
    d = gate.evaluate(
        make_signal(),
        clean_state(equity=Decimal("0.50"), free_balance=Decimal("0.50")),
    )
    expect("rejected", not d.approved)
    expect("reason INSUFFICIENT_BALANCE", d.reject_reason == RejectReason.INSUFFICIENT_BALANCE)
    print()

    # --- Test 7: balance binds before allocation ---
    print("Test 7: 3% wants $30 but only $10 free -> approved at $10, balance-bound")
    d = gate.evaluate(
        make_signal(),
        clean_state(equity=Decimal("1000"), free_balance=Decimal("10")),
    )
    expect("approved", d.approved)
    if d.order:
        expect("notional == $10 (capped by balance)", d.order.notional_usd == Decimal("10"))
        expect("binding == free_balance", d.order.binding_constraint == "free_balance")
    print()

    # --- Test 8: fee estimate is computed and logged (not vetoed) ---
    print("Test 8: Approved order carries a round-trip fee estimate")
    d = gate.evaluate(make_signal(), clean_state())
    if d.order:
        # $30 * 1.2% * 2 sides = $0.72
        expect("est round-trip fee == $0.72", d.order.est_round_trip_fee == Decimal("0.7200")
               or abs(d.order.est_round_trip_fee - Decimal("0.72")) < Decimal("0.0001"))
    print()

    if failed:
        print("✗ Some tests failed.")
        sys.exit(1)
    print("All tests passed. ✓")