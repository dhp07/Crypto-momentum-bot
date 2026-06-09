"""
Trading engine — the orchestrator that wires all components into one loop.

This is where the six previously-isolated pieces finally run together:

    bar -> features -> strategy -> risk gate -> executor -> position manager

The engine is driven ONE BAR AT A TIME via on_bar(). It is source-agnostic: the
backtester (Phase 8) feeds it historical bars; the live loop (Phase 9) feeds it
websocket bars. Same engine, different bar source — which is exactly why it's
built as a pure, testable unit now.

Per pair, each bar does exactly ONE of:
  - If a position is open: manage it (trailing stop via position_manager).
  - If no position: evaluate the strategy for a new entry, size via the risk
    gate, and open via the executor.

A position opened on a given bar is NOT managed on that same bar — management
begins on the next bar. This prevents look-ahead (can't manage before opening).

The engine assembles the RiskState the gate needs (equity, free balance, open
positions, today's realized P&L, consecutive losses) from the executor's real
state, rolling the daily P&L baseline at date boundaries for the kill switch.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

import polars as pl

from config.settings import BotConfig
from src.execution.executor import Executor, Position
from src.execution.position_manager import process_bar
from src.risk.risk_gate import RiskGate, RiskParams, RiskState
from src.strategy.base import Strategy
from src.strategy.features import compute_features


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BarResult:
    """What happened when the engine processed one bar for one pair."""

    action: str                 # "opened" | "closed" | "held" | "none" | "rejected"
    pnl: Decimal | None = None   # set on "closed"
    reason: str = ""


class TradingEngine:
    """
    Orchestrates strategy, risk, and execution over a stream of bars.

    Construct with a config, a strategy, a risk gate, and an executor. Feed bars
    via on_bar(pair, bar). The engine maintains a rolling per-pair bar buffer,
    recomputes indicators each bar, and runs the full pipeline.
    """

    def __init__(
        self,
        config: BotConfig,
        strategy: Strategy,
        risk_gate: RiskGate,
        executor: Executor,
    ) -> None:
        self._config = config
        self._strategy = strategy
        self._gate = risk_gate
        self._executor = executor

        # Indicator windows, from config (never hardcoded)
        self._breakout_lookback = config.strategy.breakout_lookback_bars
        self._volume_lookback = config.strategy.volume_lookback_bars
        self._atr_lookback = config.exits.atr_lookback_bars
        self._ema_short_period = config.strategy.ema_short_period
        self._ema_long_period = config.strategy.ema_long_period

        # Keep enough history for the largest window, plus cushion. The long EMA
        # is now included: it needs many bars to stabilize, and is typically the
        # largest window in the set.
        self._buffer_len = max(
            self._breakout_lookback,
            self._volume_lookback,
            self._atr_lookback,
            self._ema_long_period,
        ) + 60
        self._buffers: dict[str, deque[dict]] = {}

        # Risk bookkeeping
        self._consecutive_losses = 0
        self._current_day: date | None = None
        self._realized_at_day_start: Decimal = Decimal("0")

        logger.info(
            f"TradingEngine ready: breakout_lookback={self._breakout_lookback}, "
            f"volume_lookback={self._volume_lookback}, atr_lookback={self._atr_lookback}, "
            f"ema_short={self._ema_short_period}, ema_long={self._ema_long_period}, "
            f"buffer_len={self._buffer_len}"
        )

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _buffer(self, pair: str) -> deque[dict]:
        if pair not in self._buffers:
            self._buffers[pair] = deque(maxlen=self._buffer_len)
        return self._buffers[pair]

    def _latest_close(self, pair: str) -> Decimal:
        buf = self._buffers.get(pair)
        if not buf:
            return Decimal("0")
        return Decimal(str(buf[-1]["close"]))

    def _mark_prices(self) -> dict[str, Decimal]:
        """Latest close for each pair we hold, for marking positions to market."""
        return {p: self._latest_close(p) for p in self._executor.positions}

    def _roll_day_if_needed(self, bar_ts: datetime | None) -> None:
        """
        At a new calendar day, reset the daily risk state:
          - the kill-switch P&L baseline, and
          - the consecutive-loss counter.

        Resetting consecutive losses daily is what prevents the loss-pause from
        latching permanently: a bad streak pauses new entries for the rest of
        that day, then releases at the next day boundary. Without this, once the
        pause engages no trade can open, so no win can occur to clear it, and the
        bot halts forever (which corrupted the first 30-day backtest).
        """
        bar_day = bar_ts.date() if bar_ts is not None else None
        if bar_day != self._current_day:
            # Only log a reset if we're actually rolling FROM a real prior day
            if self._current_day is not None and self._consecutive_losses > 0:
                logger.debug(
                    f"Day rolled {self._current_day} -> {bar_day}; "
                    f"consecutive_losses reset from {self._consecutive_losses} to 0"
                )
            self._current_day = bar_day
            self._realized_at_day_start = self._executor.realized_pnl
            self._consecutive_losses = 0
            logger.debug(f"Day rolled to {bar_day}; P&L baseline + loss counter reset")

    # --------------------------------------------------------
    # The loop body
    # --------------------------------------------------------

    def on_bar(self, pair: str, bar: dict) -> BarResult:
        """
        Process one completed bar for one pair.

        `bar` is a dict with keys: timestamp, open, high, low, close, volume.
        Returns a BarResult describing what the engine did.
        """
        buf = self._buffer(pair)
        buf.append(bar)

        bar_ts: datetime | None = bar.get("timestamp")
        self._roll_day_if_needed(bar_ts)

        # --- If we hold this pair: manage the position (trailing stop) ---
        if pair in self._executor.positions:
            position = self._executor.positions[pair]
            decision = process_bar(
                position,
                bar_high=Decimal(str(bar["high"])),
                bar_low=Decimal(str(bar["low"])),
            )
            if decision.should_exit:
                before = self._executor.realized_pnl
                self._executor.close_position(
                    position, decision.exit_price, bar_ts, decision.reason
                )
                trade_pnl = self._executor.realized_pnl - before
                if trade_pnl < 0:
                    self._consecutive_losses += 1
                else:
                    self._consecutive_losses = 0
                logger.info(
                    f"{pair}: position closed ({decision.reason}), trade P&L "
                    f"${trade_pnl:.4f}, consecutive_losses now {self._consecutive_losses}"
                )
                return BarResult("closed", pnl=trade_pnl, reason=decision.reason)
            return BarResult("held")

        # --- No position: look for an entry ---
        features = compute_features(
            pl.DataFrame(list(buf)),
            breakout_lookback=self._breakout_lookback,
            volume_lookback=self._volume_lookback,
            atr_lookback=self._atr_lookback,
            ema_short_period=self._ema_short_period,
            ema_long_period=self._ema_long_period,
        )
        signal = self._strategy.evaluate(pair, features, position_open=False)
        if signal is None:
            return BarResult("none")

        # Assemble the risk state from the executor's real state
        marks = self._mark_prices()
        equity = self._executor.equity(marks)
        open_positions = {
            p: pos.market_value(marks.get(p, pos.entry_price))
            for p, pos in self._executor.positions.items()
        }
        realized_today = self._executor.realized_pnl - self._realized_at_day_start
        state = RiskState(
            equity=equity,
            free_balance=self._executor.cash,
            open_positions=open_positions,
            realized_pnl_today=realized_today,
            consecutive_losses=self._consecutive_losses,
        )

        decision = self._gate.evaluate(signal, state)
        if not decision.approved:
            return BarResult("rejected", reason=str(decision.reject_reason))

        position = self._executor.open_position(decision.order, bar_ts)
        if position is None:
            return BarResult("rejected", reason="executor_refused")
        return BarResult("opened")

    # --------------------------------------------------------
    # Monitoring
    # --------------------------------------------------------

    def status(self) -> dict:
        """A snapshot of engine + account state, for logging and monitoring."""
        marks = self._mark_prices()
        return {
            "cash": self._executor.cash,
            "equity": self._executor.equity(marks),
            "open_positions": list(self._executor.positions.keys()),
            "num_open": len(self._executor.positions),
            "realized_pnl": self._executor.realized_pnl,
            "consecutive_losses": self._consecutive_losses,
            "num_fills": len(self._executor.fills),
        }


# ============================================================
# Self-test — full-pipeline integration
# ============================================================


if __name__ == "__main__":
    """
    Drive the whole chain with synthetic bars and verify a real trade happens
    end to end — entry, trailing-stop management, and exit:
        python3 -m src.engine
    """
    import sys
    from datetime import timedelta, timezone

    from config.settings import load_strategy_config
    from src.execution.executor import PaperExecutor
    from src.risk.risk_gate import RiskGate, RiskParams
    from src.strategy.breakout import VolumeBreakoutStrategy

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    failed = False

    def expect(name: str, condition: bool) -> None:
        global failed
        print(f"  {'✓' if condition else '✗'} {name}")
        if not condition:
            failed = True

    config = load_strategy_config()
    t0 = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)

    def bar(i, o, h, l, c, v):
        return {
            "timestamp": t0 + timedelta(minutes=i),
            "open": float(o), "high": float(h), "low": float(l),
            "close": float(c), "volume": float(v),
        }

    def fresh_engine():
        # Zero slippage/fee so P&L sign is clean to assert
        ex = PaperExecutor(Decimal("1000"), slippage_bps=Decimal("0"), taker_fee_bps=Decimal("0"))
        strat = VolumeBreakoutStrategy(config)
        gate = RiskGate(RiskParams.from_config(config))
        return TradingEngine(config, strat, gate, ex), ex

    # Warmup bars must establish an UPTREND so the EMA trend filter allows entry.
    # We need at least ema_long_period bars for the long EMA to stabilize, drifting
    # gently upward so ema_short ends up above ema_long. (Flat warmup would leave
    # the EMAs equal and the trend filter would correctly block the entry.)
    n_warmup = max(config.strategy.ema_long_period, 40) + 20
    warmup = [
        bar(i, 100 + i * 0.05, 100.5 + i * 0.05, 99.5 + i * 0.05, 100 + i * 0.05, 1000)
        for i in range(n_warmup)
    ]
    breakout_idx = n_warmup  # the bar index right after warmup

    # --- Scenario A: winning trade (entry, trail up, profitable exit) ---
    print("Scenario A: breakout entry -> trail up -> profitable exit")
    engine, ex = fresh_engine()
    for b in warmup:
        r = engine.on_bar("BTC-USD", b)
    expect("no entry during uptrend warmup (no breakout yet)", "BTC-USD" not in ex.positions)

    # Breakout bar: close well above the recent rolling high, on 5x volume, while
    # the EMA trend is up. Use a clear jump so it exceeds the drifted rolling high.
    base = 100 + (n_warmup - 1) * 0.05
    r = engine.on_bar("BTC-USD", bar(breakout_idx, base, base + 10, base, base + 10, 6000))
    expect("entry opened on breakout bar", r.action == "opened")
    expect("position exists", "BTC-USD" in ex.positions)

    # Rising bars with shallow pullbacks -> trail ratchets up, no exit
    p = base + 10
    r = engine.on_bar("BTC-USD", bar(breakout_idx + 1, p, p + 5, p + 4.5, p + 4, 1500))
    expect("held while rising (bar +1)", r.action == "held")
    p += 4
    r = engine.on_bar("BTC-USD", bar(breakout_idx + 2, p, p + 5, p + 4.5, p + 4, 1500))
    expect("held while rising (bar +2)", r.action == "held")
    p += 4
    r = engine.on_bar("BTC-USD", bar(breakout_idx + 3, p, p + 5, p + 4.5, p + 4, 1500))
    expect("held while rising (bar +3)", r.action == "held")

    # Reversal bar: low drops well below the trailed stop -> exit in profit
    p += 4
    r = engine.on_bar("BTC-USD", bar(breakout_idx + 4, p, p, base, base + 1, 1500))
    expect("exited on reversal", r.action == "closed")
    expect("trade was profitable (P&L > 0)", r.pnl is not None and r.pnl > 0)
    expect("position cleared", "BTC-USD" not in ex.positions)
    expect("consecutive_losses reset to 0", engine._consecutive_losses == 0)
    print(f"    (exit P&L was ${r.pnl:.4f})")
    print()

    # --- Scenario B: losing trade (entry, immediate stop-out) ---
    print("Scenario B: breakout entry -> immediate stop-out -> loss")
    engine, ex = fresh_engine()
    for b in warmup:
        engine.on_bar("ETH-USD", b)
    base = 100 + (n_warmup - 1) * 0.05
    r = engine.on_bar("ETH-USD", bar(breakout_idx, base, base + 10, base, base + 10, 6000))
    expect("entry opened", r.action == "opened")
    # Next bar collapses below the initial stop -> stop-out at a loss
    p = base + 10
    r = engine.on_bar("ETH-USD", bar(breakout_idx + 1, p, p, base - 5, base - 4, 1500))
    expect("exited on stop", r.action == "closed")
    expect("trade was a loss (P&L < 0)", r.pnl is not None and r.pnl < 0)
    expect("consecutive_losses incremented to 1", engine._consecutive_losses == 1)
    print(f"    (exit P&L was ${r.pnl:.4f})")
    print()

    # --- Scenario C: status snapshot is coherent ---
    print("Scenario C: status snapshot")
    st = engine.status()
    expect("status has equity", "equity" in st)
    expect("no open positions after stop-out", st["num_open"] == 0)
    expect("recorded fills (2 = buy+sell)", st["num_fills"] == 2)
    print(f"    status: {st}")
    print()

    if failed:
        print("✗ Some tests failed.")
        sys.exit(1)
    print("All tests passed. ✓")