"""
Live websocket subscriber for Coinbase Advanced Trade.

Opens a persistent connection, subscribes to the market_trades channel for all
configured pairs, and aggregates the live trade stream into 1-minute OHLCV bars.

Completed bars are handed off via a callback, in the same shape produced by the
historical fetcher (market_data.py), so downstream code is source-agnostic.

Key responsibilities:
- Connection lifecycle: open, subscribe, detect drops, reconnect with backoff
- Trade parsing: turn raw websocket messages into typed Trade objects
- Bar aggregation: group trades into 1-minute OHLCV, emit when each minute closes

This is the live counterpart to market_data.py's historical fetch.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from coinbase.websocket import WSClient

from config.settings import EnvSettings, env


logger = logging.getLogger(__name__)


WS_CHANNEL = "market_trades"
BAR_INTERVAL_SECONDS = 60


# ============================================================
# Value objects
# ============================================================


@dataclass(frozen=True)
class Trade:
    """A single executed trade from the live stream."""

    product_id: str
    price: Decimal
    size: Decimal
    side: str           # "BUY" or "SELL"
    timestamp: datetime


@dataclass(frozen=True)
class Bar:
    """A completed 1-minute OHLCV bar built from the live trade stream."""

    product_id: str
    timestamp: datetime   # Start of the minute (e.g. 10:05:00 for the 10:05 bar)
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    trade_count: int


# ============================================================
# Bar aggregator — turns ticks into bars
# ============================================================


class BarAggregator:
    """
    Aggregates a stream of trades into 1-minute OHLCV bars, per product.

    A bar for minute M stays "open" and accumulates trades until a trade
    arrives stamped at minute M+1 or later. At that point the bar for M is
    finalized and emitted via on_bar, and a fresh bar opens for the new minute.

    Not thread-safe on its own; the WebsocketHandler serializes calls.
    """

    def __init__(self, on_bar: Callable[[Bar], None]) -> None:
        self._on_bar = on_bar
        # product_id -> partial bar state
        self._open_bars: dict[str, dict] = {}

    @staticmethod
    def _floor_to_minute(ts: datetime) -> datetime:
        """Round a timestamp down to the start of its minute."""
        return ts.replace(second=0, microsecond=0)

    def add_trade(self, trade: Trade) -> None:
        """Incorporate a trade, emitting the previous bar if the minute rolled over."""
        minute = self._floor_to_minute(trade.timestamp)
        state = self._open_bars.get(trade.product_id)

        if state is None:
            # First trade we've seen for this product — open a new bar
            self._open_bars[trade.product_id] = self._new_bar_state(minute, trade)
            return

        if minute > state["minute"]:
            # The minute rolled over — finalize and emit the old bar, then start fresh
            self._emit(trade.product_id, state)
            self._open_bars[trade.product_id] = self._new_bar_state(minute, trade)
        elif minute == state["minute"]:
            # Same minute — update the open bar
            state["high"] = max(state["high"], trade.price)
            state["low"] = min(state["low"], trade.price)
            state["close"] = trade.price
            state["volume"] += trade.size
            state["trade_count"] += 1
        else:
            # Out-of-order trade older than the open bar — rare, just log and ignore
            logger.debug(
                f"Ignoring out-of-order trade for {trade.product_id}: "
                f"{trade.timestamp} < open bar {state['minute']}"
            )

    def _new_bar_state(self, minute: datetime, trade: Trade) -> dict:
        return {
            "minute": minute,
            "open": trade.price,
            "high": trade.price,
            "low": trade.price,
            "close": trade.price,
            "volume": trade.size,
            "trade_count": 1,
        }

    def _emit(self, product_id: str, state: dict) -> None:
        bar = Bar(
            product_id=product_id,
            timestamp=state["minute"],
            open=state["open"],
            high=state["high"],
            low=state["low"],
            close=state["close"],
            volume=state["volume"],
            trade_count=state["trade_count"],
        )
        try:
            self._on_bar(bar)
        except Exception as exc:
            # A bug in the callback should not kill the aggregator
            logger.error(f"on_bar callback raised for {product_id}: {exc}")

    def flush_all(self) -> None:
        """
        Force-emit all currently open bars.

        Useful on shutdown so the final partial bar isn't lost. Note the emitted
        bar may represent an incomplete minute.
        """
        for product_id, state in list(self._open_bars.items()):
            self._emit(product_id, state)
        self._open_bars.clear()


# ============================================================
# Websocket handler — connection + parsing
# ============================================================


class WebsocketHandler:
    """
    Manages the live websocket connection and feeds trades to a BarAggregator.

    Usage:
        handler = WebsocketHandler(pairs=["BTC-USD", ...], on_bar=my_callback)
        handler.start()   # blocks; runs until stop() or KeyboardInterrupt
    """

    RECONNECT_INITIAL_BACKOFF = 1.0
    RECONNECT_MAX_BACKOFF = 60.0

    def __init__(
        self,
        pairs: list[str],
        on_bar: Callable[[Bar], None],
        env_settings: EnvSettings | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("Must subscribe to at least one pair")

        self._pairs = pairs
        self._settings = env_settings or env
        self._aggregator = BarAggregator(on_bar=on_bar)

        self._ws: WSClient | None = None
        self._should_run = False
        self._reconnect_backoff = self.RECONNECT_INITIAL_BACKOFF

        # Stats for monitoring/debugging
        self._trades_seen = 0
        self._last_trade_at: float = 0.0
        self._lock = threading.Lock()

    # --------------------------------------------------------
    # Message handling
    # --------------------------------------------------------

    def _on_message(self, msg: str) -> None:
        """
        Called by the SDK for every websocket message (a JSON string).

        We care about market_trades messages of type "update"; everything else
        (snapshots, heartbeats, subscription confirmations) is ignored for bars.
        """
        try:
            data = json.loads(msg)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Could not parse websocket message: {msg[:200]}")
            return

        channel = data.get("channel")
        if channel == "subscriptions":
            logger.info("Subscription confirmed by Coinbase")
            return
        if channel != WS_CHANNEL:
            return  # heartbeats, etc.

        for event in data.get("events", []):
            # Skip "snapshot" — it contains historical trades that would pollute
            # the current minute's volume. We only build bars from live "update"s.
            if event.get("type") != "update":
                continue
            for raw_trade in event.get("trades", []):
                trade = self._parse_trade(raw_trade)
                if trade is not None:
                    with self._lock:
                        self._trades_seen += 1
                        self._last_trade_at = time.monotonic()
                    self._aggregator.add_trade(trade)

        # A successful message means the connection is healthy — reset backoff
        self._reconnect_backoff = self.RECONNECT_INITIAL_BACKOFF

    @staticmethod
    def _parse_trade(raw: dict) -> Trade | None:
        """Convert a raw trade dict from Coinbase into our Trade type."""
        try:
            # Coinbase trade time is ISO8601, e.g. "2026-05-30T05:59:00.123456Z"
            ts_str = raw["time"].replace("Z", "+00:00")
            return Trade(
                product_id=raw["product_id"],
                price=Decimal(str(raw["price"])),
                size=Decimal(str(raw["size"])),
                side=raw.get("side", "UNKNOWN"),
                timestamp=datetime.fromisoformat(ts_str).astimezone(timezone.utc),
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(f"Could not parse trade {raw}: {exc}")
            return None

    # --------------------------------------------------------
    # Connection lifecycle
    # --------------------------------------------------------

    def start(self) -> None:
        """
        Open the connection and run until stopped.

        Blocks the calling thread. Reconnects automatically on failure with
        exponential backoff. Call stop() from another thread or Ctrl+C to end.
        """
        self._should_run = True
        logger.info(f"Starting websocket handler for pairs: {self._pairs}")

        while self._should_run:
            try:
                self._connect_and_run()
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt — shutting down")
                break
            except Exception as exc:
                if not self._should_run:
                    break
                logger.error(
                    f"Websocket connection failed: {exc}. "
                    f"Reconnecting in {self._reconnect_backoff:.1f}s"
                )
                time.sleep(self._reconnect_backoff)
                self._reconnect_backoff = min(
                    self._reconnect_backoff * 2, self.RECONNECT_MAX_BACKOFF
                )

        # Emit any final partial bars on clean shutdown
        self._aggregator.flush_all()
        logger.info(f"Websocket handler stopped. Total trades seen: {self._trades_seen}")

    def _connect_and_run(self) -> None:
        """One connection attempt: open, subscribe, and pump messages until it drops."""
        self._ws = WSClient(
            api_key=self._settings.coinbase_api_key_name,
            api_secret=self._settings.coinbase_api_private_key,
            on_message=self._on_message,
        )

        self._ws.open()
        self._ws.subscribe(product_ids=self._pairs, channels=[WS_CHANNEL, "heartbeats"])
        logger.info("Websocket opened and subscribed")

        # The SDK pumps messages on a background thread. We block here, periodically
        # checking health, until the connection dies or we're told to stop.
        while self._should_run:
            self._ws.sleep_with_exception_check(sleep=5)
            self._check_staleness()

    def _check_staleness(self) -> None:
        """
        Warn if we haven't seen any trade in a while.

        On all 5 pairs combined, BTC alone should produce trades every few seconds.
        A long silence usually means the connection is half-dead even if it looks open.
        """
        with self._lock:
            last = self._last_trade_at
        if last == 0.0:
            return  # No trades yet — still warming up
        silence = time.monotonic() - last
        if silence > 120:
            logger.warning(
                f"No trades received in {silence:.0f}s — connection may be stale. "
                f"Forcing reconnect."
            )
            raise ConnectionError("Trade stream went silent")

    def stop(self) -> None:
        """Signal the handler to shut down. Safe to call from another thread."""
        logger.info("Stop requested")
        self._should_run = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception as exc:
                logger.debug(f"Error closing websocket (expected during shutdown): {exc}")

    @property
    def trades_seen(self) -> int:
        with self._lock:
            return self._trades_seen


# ============================================================
# Self-test
# ============================================================


if __name__ == "__main__":
    """
    Run this file directly to verify the websocket subscriber works:
        python3 -m src.data.websocket_handler

    It connects for ~90 seconds, prints completed bars as they form, then exits.
    BTC-USD should produce bars quickly; thin pairs may produce few or none.
    """
    import signal
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    TEST_PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD"]
    TEST_DURATION_SECONDS = 90

    bars_received: list[Bar] = []

    def handle_bar(bar: Bar) -> None:
        bars_received.append(bar)
        print(
            f"  BAR {bar.product_id:9} {bar.timestamp.strftime('%H:%M')} "
            f"O={float(bar.open):>12,.4f} H={float(bar.high):>12,.4f} "
            f"L={float(bar.low):>12,.4f} C={float(bar.close):>12,.4f} "
            f"V={float(bar.volume):>10,.4f} ({bar.trade_count} trades)"
        )

    print(f"Connecting to Coinbase websocket for {TEST_DURATION_SECONDS}s...")
    print(f"Pairs: {TEST_PAIRS}")
    print("Waiting for trades (bars emit when each minute closes)...")
    print()

    try:
        handler = WebsocketHandler(pairs=TEST_PAIRS, on_bar=handle_bar)
    except Exception as exc:
        print(f"✗ Failed to create handler: {exc}")
        sys.exit(1)

    # Run the handler in a background thread so we can time-box the test
    ws_thread = threading.Thread(target=handler.start, daemon=True)
    ws_thread.start()

    # Let it run for the test duration, then stop
    try:
        time.sleep(TEST_DURATION_SECONDS)
    except KeyboardInterrupt:
        print("\nInterrupted early.")

    print()
    print("Stopping handler (flushing open bars)...")
    handler.stop()
    ws_thread.join(timeout=10)

    print()
    print(f"Test summary:")
    print(f"  Total trades seen:  {handler.trades_seen}")
    print(f"  Total bars emitted: {len(bars_received)}")
    products_with_bars = {b.product_id for b in bars_received}
    print(f"  Pairs with bars:    {sorted(products_with_bars)}")
    print()

    if handler.trades_seen == 0:
        print("✗ No trades received. Possible causes:")
        print("  - Websocket auth failed (check .env credentials)")
        print("  - Network/firewall blocking websocket connections")
        print("  - Coinbase websocket endpoint changed")
        sys.exit(1)

    if len(bars_received) == 0:
        print("⚠ Trades received but no bars emitted.")
        print("  This can happen if the test ran less than a full minute boundary.")
        print("  Trades were flowing, so the connection works. Try a longer run.")
        sys.exit(0)

    print("All tests passed. ✓")
    print("(Bars only emit after a minute boundary, so the count depends on timing.)")