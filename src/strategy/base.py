"""
Strategy interface — the abstract base every strategy implements.

This is what makes the bot multi-strategy ready. The volume-breakout strategy is
the first implementation; later, additional strategies (for the adaptive layer)
implement this same interface, so the rest of the bot — risk gate, executor,
backtester — can treat any strategy identically.

A strategy's job is narrow and pure: given a feature frame (bars + indicators from
Phase 3) and whether a position is already open, decide whether to emit an entry
signal. It does NOT size positions, check account risk, or place orders. Those are
downstream concerns (Phase 5 risk gate, Phase 6 executor). Keeping the strategy
ignorant of them is what lets us test signal logic in isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

import polars as pl


class SignalType(str, Enum):
    """The kinds of signal a strategy can emit."""

    ENTER_LONG = "ENTER_LONG"
    # Future: ENTER_SHORT, EXIT — added when the strategy supports them.
    # Exits are currently handled by the trailing stop in the risk/execution
    # layer, not by the strategy, so the strategy only emits entries for now.


@dataclass(frozen=True)
class Signal:
    """
    A complete, self-contained trade instruction emitted by a strategy.

    Carries everything downstream needs to act: which pair, direction, the price
    the signal was generated at, and the stop level the strategy computed. The
    risk gate decides whether and how large to act; the executor places the order.
    """

    signal_type: SignalType
    product_id: str
    timestamp: datetime
    price: Decimal          # The bar close that triggered the signal
    stop_price: Decimal     # Computed by the strategy (entry - 1.5 * ATR for longs)
    reason: str             # Human-readable why, for logging and debugging

    @property
    def risk_per_unit(self) -> Decimal:
        """Distance from entry to stop, per unit. The risk gate uses this for sizing."""
        return abs(self.price - self.stop_price)


class Strategy(ABC):
    """
    Abstract base for all strategies.

    Subclasses implement evaluate(), which inspects the most recent bar of a
    feature frame and optionally returns a Signal. Everything else (looping over
    pairs, maintaining position state, applying risk) lives outside the strategy.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def evaluate(
        self,
        product_id: str,
        features: pl.DataFrame,
        position_open: bool,
    ) -> Signal | None:
        """
        Decide whether to emit a signal based on the latest bar.

        Args:
            product_id: The pair being evaluated, e.g. "BTC-USD"
            features: Feature frame from Phase 3 — bars with rolling_high,
                      avg_volume, atr columns. Evaluated on the LAST row.
            position_open: Whether a position is already open for this pair.
                           A strategy must not emit an entry if already in.

        Returns:
            A Signal if the entry conditions are met, else None.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Strategy name={self.name!r}>"