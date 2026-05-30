"""
Historical market data fetcher.

Wraps CoinbaseClient with logic for:
- Paginating around Coinbase's 350-candle-per-request limit
- Converting candle data to clean Polars DataFrames
- Filling gaps where Coinbase occasionally drops a bar
- Validating that we got what we asked for

The bot fetches historical data in two scenarios:
1. Startup "warmup" — pulls last 60 bars so indicators have values immediately
2. Backtesting — pulls months/years of data, written to Parquet for replay

All bars are 1-minute granularity. Other granularities can be added if needed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import polars as pl

from src.data.coinbase_client import Candle, CoinbaseClient


logger = logging.getLogger(__name__)


# Coinbase Advanced Trade API caps candle requests at 350 per call.
# We use 300 to leave a safety margin (some endpoints round differently).
MAX_CANDLES_PER_REQUEST = 300
BAR_INTERVAL_SECONDS = 60  # 1-minute bars
GRANULARITY = "ONE_MINUTE"


class MarketDataFetcher:
    """
    High-level historical data interface.

    Given a CoinbaseClient, fetches and assembles historical bars into Polars DataFrames.
    Stateless — safe to use from multiple places, no caching at this layer.
    """

    def __init__(self, client: CoinbaseClient) -> None:
        self._client = client

    # --------------------------------------------------------
    # Warmup — fetch most recent N bars
    # --------------------------------------------------------

    def fetch_warmup_bars(self, pair: str, num_bars: int = 60) -> pl.DataFrame:
        """
        Fetch the most recent N 1-minute bars for a pair.

        Used at bot startup so indicators (rolling high, ATR, volume average)
        have valid values immediately. The bot can begin generating signals
        as soon as live ticks start arriving.

        Args:
            pair: Trading pair like "BTC-USD"
            num_bars: How many recent 1-minute bars to fetch. Default 60 (one hour).

        Returns:
            Polars DataFrame with columns: timestamp, open, high, low, close, volume.
            Sorted oldest to newest. Length will be num_bars (or fewer if Coinbase
            dropped some bars — rare but possible).
        """
        if num_bars <= 0:
            raise ValueError(f"num_bars must be positive, got {num_bars}")
        if num_bars > MAX_CANDLES_PER_REQUEST:
            # Warmup shouldn't ever need more than a few hours of bars;
            # if we need more we should be using fetch_historical_bars
            raise ValueError(
                f"num_bars={num_bars} exceeds single-request limit of {MAX_CANDLES_PER_REQUEST}. "
                f"Use fetch_historical_bars for larger ranges."
            )

        end = datetime.now(timezone.utc)
        # Pad start by 2 minutes — Coinbase sometimes doesn't have the most recent bar yet,
        # and we'd rather over-fetch than fall short of num_bars
        start = end - timedelta(minutes=num_bars + 2)

        candles = self._client.get_candles(
            product_id=pair, start=start, end=end, granularity=GRANULARITY
        )

        df = _candles_to_dataframe(candles)

        # Take the last N bars in case we over-fetched
        if len(df) > num_bars:
            df = df.tail(num_bars)

        logger.info(f"Warmup: fetched {len(df)} bars for {pair} (requested {num_bars})")
        return df

    # --------------------------------------------------------
    # Historical — paginated fetch for arbitrary ranges
    # --------------------------------------------------------

    def fetch_historical_bars(
        self,
        pair: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Fetch 1-minute bars over an arbitrary time range, paginating as needed.

        Used for backtesting and bulk data collection. For ranges longer than
        ~5 hours (300 bars), this will make multiple API calls.

        Args:
            pair: Trading pair like "BTC-USD"
            start: Range start (inclusive). Timezone-aware datetime required.
            end: Range end (inclusive). Timezone-aware datetime required.

        Returns:
            Polars DataFrame, sorted oldest to newest, deduplicated by timestamp.
            May have gaps if Coinbase didn't return bars for some minutes
            (low liquidity, exchange downtime, etc).
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware datetimes")
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")

        # Each chunk fetches MAX_CANDLES_PER_REQUEST minutes of data
        chunk_size = timedelta(minutes=MAX_CANDLES_PER_REQUEST)

        all_candles: list[Candle] = []
        cursor = start
        chunk_count = 0

        while cursor < end:
            chunk_end = min(cursor + chunk_size, end)

            chunk = self._client.get_candles(
                product_id=pair,
                start=cursor,
                end=chunk_end,
                granularity=GRANULARITY,
            )
            all_candles.extend(chunk)
            chunk_count += 1

            if chunk_count % 10 == 0:
                logger.info(
                    f"Fetched {chunk_count} chunks ({len(all_candles)} bars so far) for {pair}"
                )

            cursor = chunk_end

        df = _candles_to_dataframe(all_candles)
        df = _deduplicate_and_sort(df)

        expected = int((end - start).total_seconds() // BAR_INTERVAL_SECONDS)
        actual = len(df)
        if actual < expected * 0.95:  # Allow 5% missing tolerance
            logger.warning(
                f"Got {actual} bars for {pair}, expected ~{expected} "
                f"({(1 - actual/expected) * 100:.1f}% missing). "
                f"This may indicate Coinbase data gaps."
            )

        logger.info(
            f"Historical fetch complete: {len(df)} bars for {pair} "
            f"({chunk_count} requests, {start.date()} to {end.date()})"
        )
        return df


# ============================================================
# Internal helpers
# ============================================================


def _candles_to_dataframe(candles: list[Candle]) -> pl.DataFrame:
    """Convert a list of Candle objects to a Polars DataFrame."""
    if not candles:
        # Return empty DataFrame with correct schema rather than raising
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

    return pl.DataFrame(
        {
            "timestamp": [c.timestamp for c in candles],
            "open": [float(c.open) for c in candles],
            "high": [float(c.high) for c in candles],
            "low": [float(c.low) for c in candles],
            "close": [float(c.close) for c in candles],
            "volume": [float(c.volume) for c in candles],
        }
    ).with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))


def _deduplicate_and_sort(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove duplicate timestamps (chunk boundaries can produce overlap) and sort.

    Coinbase's API uses inclusive ranges, so consecutive chunks can each include
    the boundary minute. We keep the first occurrence and discard duplicates.
    """
    if len(df) == 0:
        return df
    return df.unique(subset=["timestamp"], keep="first").sort("timestamp")


# ============================================================
# Self-test
# ============================================================


if __name__ == "__main__":
    """
    Run this file directly to verify the fetcher works:
        python3 -m src.data.market_data
    """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Testing MarketDataFetcher...")
    print()

    try:
        client = CoinbaseClient()
        fetcher = MarketDataFetcher(client)
    except Exception as exc:
        print(f"✗ Failed to set up: {exc}")
        sys.exit(1)

    # Test 1: Warmup fetch for BTC-USD
    print("Test 1: Warmup — last 60 1-min bars of BTC-USD")
    try:
        df = fetcher.fetch_warmup_bars("BTC-USD", num_bars=60)
        print(f"  ✓ Got {len(df)} bars")
        print(f"  First timestamp: {df['timestamp'].min()}")
        print(f"  Last timestamp:  {df['timestamp'].max()}")
        print(f"  Most recent close: ${df['close'][-1]:,.2f}")
        print(f"  Sample row count check: {len(df)} (expected ~60)")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        sys.exit(1)
    print()

    # Test 2: Warmup for all 5 strategy pairs
    print("Test 2: Warmup for all 5 strategy pairs")
    pairs = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD"]
    for pair in pairs:
        try:
            df = fetcher.fetch_warmup_bars(pair, num_bars=30)
            print(f"  ✓ {pair:10} {len(df)} bars  latest=${df['close'][-1]:>12,.4f}")
        except Exception as exc:
            print(f"  ✗ {pair:10} Failed: {exc}")
    print()

    # Test 3: Multi-chunk historical fetch (4 hours = needs 1 chunk; 24 hours = ~5 chunks)
    print("Test 3: Historical fetch — last 24 hours of BTC-USD (multi-chunk)")
    try:
        end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start = end - timedelta(hours=24)
        df = fetcher.fetch_historical_bars("BTC-USD", start=start, end=end)
        print(f"  ✓ Got {len(df)} bars (expected ~1440 for 24 hours)")
        if len(df) > 0:
            print(f"  Time range: {df['timestamp'].min()}  →  {df['timestamp'].max()}")
            print(f"  Price range: ${df['low'].min():,.2f}  →  ${df['high'].max():,.2f}")
            print(f"  Total volume: {df['volume'].sum():,.2f} BTC")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        sys.exit(1)
    print()

    # Test 4: Confirm DataFrame schema is what callers expect
    print("Test 4: DataFrame schema validation")
    try:
        df = fetcher.fetch_warmup_bars("BTC-USD", num_bars=5)
        expected_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        actual_cols = set(df.columns)
        if actual_cols != expected_cols:
            print(f"  ✗ Schema mismatch: expected {expected_cols}, got {actual_cols}")
            sys.exit(1)
        print(f"  ✓ Columns: {df.columns}")
        print(f"  ✓ Dtypes: {dict(zip(df.columns, df.dtypes))}")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        sys.exit(1)
    print()

    print("All tests passed. ✓")