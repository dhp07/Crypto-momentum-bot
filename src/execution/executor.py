"""
Execution layer — turns approved orders into fills and tracks the virtual account.

The PaperExecutor SIMULATES trading: it takes an ApprovedOrder from the risk gate,
applies realistic slippage and fees, and updates a virtual cash balance and set of
open positions. No real money, no real Coinbase orders.

It sits behind an Executor interface so that a future LiveExecutor (real Coinbase
orders) can be swapped in with identical method signatures — the trading loop and
position manager don't change. Paper and live share everything except the fill.

Money is handled in Decimal throughout to avoid float rounding on prices/balances.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from src.risk.risk_gate import ApprovedOrder


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fill:
    """A completed (simulated) buy or sell."""

    product_id: str
    side: str            # "BUY" or "SELL"
    price: Decimal       # actual fill price, after slippage
    quantity: Decimal
    fee: Decimal
    notional: Decimal    # price * quantity, before fee
    timestamp: datetime


@dataclass
class Position:
    """
    An open position. MUTABLE — the trailing stop updates current_stop and
    high_water_mark over the life of the trade.
    """

    product_id: str
    entry_price: Decimal      # actual fill price paid (after slippage)
    quantity: Decimal
    entry_time: datetime
    entry_cost: Decimal       # total cash deducted at open (notional + entry fee)
    initial_stop: Decimal     # stop at entry (entry - 1.5*ATR), from the signal
    trail_distance: Decimal   # entry_price - initial_stop; frozen for the trade's life
    current_stop: Decimal     # trails upward; starts at initial_stop
    high_water_mark: Decimal  # highest price seen since entry; starts at entry_price

    def market_value(self, price: Decimal) -> Decimal:
        """Current worth of the position at a given price."""
        return self.quantity * price


class Executor(ABC):
    """Interface every executor implements. Paper now; live later, same signatures."""

    @abstractmethod
    def open_position(self, order: ApprovedOrder, fill_time: datetime) -> Position | None:
        raise NotImplementedError

    @abstractmethod
    def close_position(
        self, position: Position, exit_price: Decimal, fill_time: datetime, reason: str
    ) -> Fill:
        raise NotImplementedError


class PaperExecutor(Executor):
    """
    Simulated executor with a virtual account.

    Applies slippage (fills are worse than the quoted price) and taker fees on both
    entry and exit, mirroring how real fills behave. Tracks cash and open positions.
    """

    def __init__(
        self,
        starting_cash: Decimal,
        slippage_bps: Decimal,
        taker_fee_bps: Decimal,
    ) -> None:
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self._slippage = slippage_bps / Decimal("10000")
        self._taker_fee = taker_fee_bps / Decimal("10000")
        self.positions: dict[str, Position] = {}
        self.realized_pnl: Decimal = Decimal("0")
        self.fills: list[Fill] = []
        logger.info(
            f"PaperExecutor: starting_cash=${starting_cash} "
            f"slippage={self._slippage} taker_fee={self._taker_fee}"
        )

    def equity(self, mark_prices: dict[str, Decimal]) -> Decimal:
        """Total account value: cash + marked-to-market open positions."""
        positions_value = sum(
            (pos.market_value(mark_prices.get(pair, pos.entry_price))
             for pair, pos in self.positions.items()),
            Decimal("0"),
        )
        return self.cash + positions_value

    def open_position(self, order: ApprovedOrder, fill_time: datetime) -> Position | None:
        signal = order.signal
        pair = signal.product_id

        # Buying: slippage pushes the fill price UP (we pay more than quoted)
        fill_price = signal.price * (Decimal("1") + self._slippage)

        # Spend the approved notional at the (worse) fill price -> slightly less quantity
        notional = order.notional_usd
        quantity = notional / fill_price
        fee = notional * self._taker_fee
        total_cost = notional + fee

        if total_cost > self.cash:
            logger.warning(
                f"open_position {pair}: total cost ${total_cost} exceeds cash ${self.cash}; skipping"
            )
            return None

        self.cash -= total_cost

        # Trail distance is fixed at entry: the entry-to-initial-stop gap (= 1.5*ATR)
        trail_distance = fill_price - signal.stop_price

        position = Position(
            product_id=pair,
            entry_price=fill_price,
            quantity=quantity,
            entry_time=fill_time,
            entry_cost=total_cost,
            initial_stop=signal.stop_price,
            trail_distance=trail_distance,
            current_stop=signal.stop_price,
            high_water_mark=fill_price,
        )
        self.positions[pair] = position
        self.fills.append(Fill(pair, "BUY", fill_price, quantity, fee, notional, fill_time))

        logger.info(
            f"OPEN {pair}: filled {quantity:.8f} @ ${fill_price:.4f} "
            f"(quoted ${signal.price:.4f}, slippage applied), fee ${fee:.4f}, "
            f"stop ${signal.stop_price:.4f}, trail_distance ${trail_distance:.4f}, "
            f"cash now ${self.cash:.2f}"
        )
        return position

    def close_position(
        self, position: Position, exit_price: Decimal, fill_time: datetime, reason: str
    ) -> Fill:
        pair = position.product_id

        # Selling: slippage pushes the fill price DOWN (we receive less than quoted)
        fill_price = exit_price * (Decimal("1") - self._slippage)
        proceeds = position.quantity * fill_price
        fee = proceeds * self._taker_fee
        net_proceeds = proceeds - fee

        self.cash += net_proceeds
        trade_pnl = net_proceeds - position.entry_cost
        self.realized_pnl += trade_pnl

        fill = Fill(pair, "SELL", fill_price, position.quantity, fee, proceeds, fill_time)
        self.fills.append(fill)
        del self.positions[pair]

        logger.info(
            f"CLOSE {pair} ({reason}): sold {position.quantity:.8f} @ ${fill_price:.4f} "
            f"(quoted ${exit_price:.4f}), fee ${fee:.4f}, "
            f"trade P&L ${trade_pnl:.4f}, cash now ${self.cash:.2f}, "
            f"total realized P&L ${self.realized_pnl:.4f}"
        )
        return fill


# ============================================================
# Self-test
# ============================================================


if __name__ == "__main__":
    """
    Verify fills, slippage, fees, and P&L math:
        python3 -m src.execution.executor
    """
    import sys
    from datetime import timezone

    from src.strategy.base import Signal, SignalType

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    failed = False

    def expect(name: str, got, expected, tol: str = "0.0001") -> None:
        global failed
        if abs(Decimal(str(got)) - Decimal(str(expected))) <= Decimal(tol):
            print(f"  ✓ {name}: {got}")
        else:
            print(f"  ✗ {name}: got {got}, expected {expected}")
            failed = True

    def make_order(price: float, notional: float, stop: float) -> ApprovedOrder:
        sig = Signal(
            signal_type=SignalType.ENTER_LONG,
            product_id="BTC-USD",
            timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
            price=Decimal(str(price)),
            stop_price=Decimal(str(stop)),
            reason="test",
        )
        return ApprovedOrder(
            signal=sig,
            notional_usd=Decimal(str(notional)),
            quantity=Decimal(str(notional)) / Decimal(str(price)),
            binding_constraint="allocation",
            est_round_trip_fee=Decimal("0"),
        )

    t = datetime(2026, 6, 1, tzinfo=timezone.utc)

    # --- Test 1: no slippage, no fee — clean P&L ---
    print("Test 1: Open $30 @ 100, close @ 110, no slippage/fee -> P&L +$3")
    ex = PaperExecutor(Decimal("1000"), slippage_bps=Decimal("0"), taker_fee_bps=Decimal("0"))
    pos = ex.open_position(make_order(100, 30, 94), t)
    expect("cash after open == 970", ex.cash, 970)
    expect("quantity == 0.3", pos.quantity, "0.3")
    ex.close_position(pos, Decimal("110"), t, "test")
    expect("cash after close == 1003", ex.cash, 1003)
    expect("realized P&L == 3", ex.realized_pnl, 3)
    print()

    # --- Test 2: with 1% fee ---
    print("Test 2: Same trade, 1% taker fee -> P&L +$2.37")
    ex = PaperExecutor(Decimal("1000"), slippage_bps=Decimal("0"), taker_fee_bps=Decimal("100"))
    pos = ex.open_position(make_order(100, 30, 94), t)
    # open: notional 30, fee 0.30, cash 1000 - 30.30 = 969.70
    expect("cash after open == 969.70", ex.cash, "969.70")
    ex.close_position(pos, Decimal("110"), t, "test")
    # close: proceeds 33, fee 0.33, net 32.67; P&L = 32.67 - 30.30 = 2.37
    expect("realized P&L == 2.37", ex.realized_pnl, "2.37")
    print()

    # --- Test 3: slippage direction ---
    print("Test 3: 1% slippage -> buy fills higher, sell fills lower than quoted")
    ex = PaperExecutor(Decimal("1000"), slippage_bps=Decimal("100"), taker_fee_bps=Decimal("0"))
    pos = ex.open_position(make_order(100, 30, 94), t)
    expect("buy fill == 101 (quoted 100 + 1%)", pos.entry_price, "101")
    fill = ex.close_position(pos, Decimal("110"), t, "test")
    expect("sell fill == 108.9 (quoted 110 - 1%)", fill.price, "108.9")
    print()

    # --- Test 4: insufficient cash is refused ---
    print("Test 4: Order larger than cash -> refused (None)")
    ex = PaperExecutor(Decimal("10"), slippage_bps=Decimal("0"), taker_fee_bps=Decimal("0"))
    pos = ex.open_position(make_order(100, 30, 94), t)  # wants $30, only $10
    expect("position is None", 1 if pos is None else 0, 1)
    expect("cash untouched == 10", ex.cash, 10)
    print()

    if failed:
        print("✗ Some tests failed.")
        sys.exit(1)
    print("All tests passed. ✓")