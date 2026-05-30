"""
Bar storage layer — persists OHLCV bars to Parquet on disk.

Bars come from two sources (historical fetch and live websocket) and both land
here in one canonical format. Storage layout:

    data/<PAIR>/<YYYY-MM-DD>.parquet

One file per pair per day. At 1-minute granularity that's at most 1,440 rows
per file — small enough to rewrite the whole day-file on each append, which
keeps dedup trivially correct.

This is the persistence half of the data layer. Combined with market_data.py
(historical) and websocket_handler.py (live), the bot can build and keep its
own market dataset for backtesting and crash recovery.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from config.settings import env
from src.data.websocket_handler import Bar


logger = logging.getLogger(__name__)


# Canonical column schema for all stored bars. Historical bars have no
# trade_count, so it's nullable.
CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "trade_count": pl.Int64,
}


class BarStorage:
    """
    Reads and writes OHLCV bars to per-pair, per-day Parquet files.

    Construct once; reuse. Safe to call save_bars repeatedly as live bars arrive.
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir is not None else env.data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"BarStorage using data dir: {self._data_dir}")

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    def _pair_dir(self, pair: str) -> Path:
        return self._data_dir / pair

    def _file_path(self, pair: str, day: date) -> Path:
        return self._pair_dir(pair) / f"{day.isoformat()}.parquet"

    # --------------------------------------------------------
    # Writing
    # --------------------------------------------------------

    def save_bars(self, pair: str, bars: list[Bar] | pl.DataFrame) -> int:
        """
        Save bars for a pair, splitting across day-files as needed.

        Args:
            pair: Trading pair like "BTC-USD"
            bars: Either a list of Bar objects (from the websocket) or a Polars
                  DataFrame (from the historical fetcher). Both are normalized
                  to the canonical schema before writing.

        Returns:
            Number of bars written (after dedup against existing data).
        """
        df = self._normalize(bars)
        if len(df) == 0:
            return 0

        # Split into day-buckets and append each to its file
        df = df.with_columns(pl.col("timestamp").dt.date().alias("_day"))
        total_written = 0
        for (day_val,), day_df in df.group_by(["_day"], maintain_order=True):
            day_df = day_df.drop("_day")
            written = self._append_to_day(pair, day_val, day_df)
            total_written += written

        logger.info(f"Saved {total_written} bars for {pair}")
        return total_written

    def _append_to_day(self, pair: str, day: date, new_df: pl.DataFrame) -> int:
        """Merge new bars into a single day-file, deduplicating by timestamp."""
        path = self._file_path(pair, day)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pl.read_parquet(path)
            # New rows go after existing so that on a timestamp collision the
            # newer (more complete) bar wins with keep="last".
            combined = pl.concat([existing, new_df], how="vertical_relaxed")
        else:
            combined = new_df

        before = len(combined)
        combined = (
            combined.unique(subset=["timestamp"], keep="last").sort("timestamp")
        )
        after = len(combined)

        combined.write_parquet(path)

        # "Written" = net new rows added to this file
        prior_rows = before - len(new_df)
        net_new = after - prior_rows
        return max(net_new, 0)

    @staticmethod
    def _normalize(bars: list[Bar] | pl.DataFrame) -> pl.DataFrame:
        """Convert either input type into a DataFrame with the canonical schema."""
        if isinstance(bars, pl.DataFrame):
            df = bars
            # Historical DataFrames lack trade_count — add a null column
            if "trade_count" not in df.columns:
                df = df.with_columns(
                    pl.lit(None, dtype=pl.Int64).alias("trade_count")
                )
        else:
            if not bars:
                return pl.DataFrame(schema=CANONICAL_SCHEMA)
            df = pl.DataFrame(
                {
                    "timestamp": [b.timestamp for b in bars],
                    "open": [float(b.open) for b in bars],
                    "high": [float(b.high) for b in bars],
                    "low": [float(b.low) for b in bars],
                    "close": [float(b.close) for b in bars],
                    "volume": [float(b.volume) for b in bars],
                    "trade_count": [b.trade_count for b in bars],
                }
            )

        # Enforce column order and UTC timestamps
        df = df.select(list(CANONICAL_SCHEMA.keys()))
        df = df.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
        return df

    # --------------------------------------------------------
    # Reading
    # --------------------------------------------------------

    def load_bars(
        self, pair: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        """
        Load all stored bars for a pair within [start, end] (inclusive).

        Reads every day-file that overlaps the range, concatenates, filters to
        the exact window, deduplicates and sorts.

        Returns an empty (correctly-typed) DataFrame if nothing is stored.
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware")
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")

        frames: list[pl.DataFrame] = []
        day = start.date()
        last_day = end.date()
        while day <= last_day:
            path = self._file_path(pair, day)
            if path.exists():
                frames.append(pl.read_parquet(path))
            day += timedelta(days=1)

        if not frames:
            return pl.DataFrame(schema=CANONICAL_SCHEMA)

        df = pl.concat(frames, how="vertical_relaxed")
        df = (
            df.filter(
                (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
            )
            .unique(subset=["timestamp"], keep="last")
            .sort("timestamp")
        )
        return df

    def list_available_dates(self, pair: str) -> list[date]:
        """Return the sorted dates for which this pair has any stored data."""
        pair_dir = self._pair_dir(pair)
        if not pair_dir.exists():
            return []
        dates: list[date] = []
        for f in pair_dir.glob("*.parquet"):
            try:
                dates.append(date.fromisoformat(f.stem))
            except ValueError:
                logger.warning(f"Skipping unexpected file in {pair_dir}: {f.name}")
        return sorted(dates)


# ============================================================
# Self-test
# ============================================================


if __name__ == "__main__":
    """
    Run this file directly to verify storage works:
        python3 -m src.data.storage

    Uses a temporary directory so it never touches real bot data, and cleans
    up after itself.
    """
    import shutil
    import sys
    import tempfile
    from decimal import Decimal

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Testing BarStorage...")
    print()

    tmp_dir = Path(tempfile.mkdtemp(prefix="barstorage_test_"))
    storage = BarStorage(data_dir=tmp_dir)

    def make_bar(pair: str, minute: int, price: float, day_str: str = "2026-05-30") -> Bar:
        ts = datetime.fromisoformat(f"{day_str}T10:{minute:02d}:00+00:00")
        return Bar(
            product_id=pair,
            timestamp=ts,
            open=Decimal(str(price)),
            high=Decimal(str(price + 5)),
            low=Decimal(str(price - 5)),
            close=Decimal(str(price + 2)),
            volume=Decimal("1.5"),
            trade_count=10,
        )

    failed = False

    # Test 1: Save Bar objects, reload, verify
    print("Test 1: Save and reload Bar objects")
    try:
        bars = [make_bar("BTC-USD", m, 73000 + m) for m in range(5)]
        written = storage.save_bars("BTC-USD", bars)
        start = datetime(2026, 5, 30, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 5, 30, 23, 59, tzinfo=timezone.utc)
        loaded = storage.load_bars("BTC-USD", start, end)
        assert written == 5, f"expected 5 written, got {written}"
        assert len(loaded) == 5, f"expected 5 loaded, got {len(loaded)}"
        assert loaded["trade_count"][0] == 10
        print(f"  ✓ Wrote 5, loaded {len(loaded)} bars with trade_count intact")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        failed = True
    print()

    # Test 2: Dedup — saving overlapping minutes does not create duplicates
    print("Test 2: Deduplication on overlapping save")
    try:
        # Re-save minute 4 with a different price; should overwrite, not duplicate
        overlap = [make_bar("BTC-USD", 4, 99999)]
        storage.save_bars("BTC-USD", overlap)
        loaded = storage.load_bars("BTC-USD", start, end)
        assert len(loaded) == 5, f"expected 5 after dedup, got {len(loaded)}"
        # The newer write (price 99999 -> open 99999) should win
        minute4 = loaded.filter(pl.col("timestamp").dt.minute() == 4)
        assert minute4["open"][0] == 99999.0, f"expected newer value, got {minute4['open'][0]}"
        print(f"  ✓ Overlap deduplicated; newer bar won (still {len(loaded)} rows)")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        failed = True
    print()

    # Test 3: Cross-day split into separate files
    print("Test 3: Bars spanning two days go to two files")
    try:
        day1 = [make_bar("ETH-USD", m, 2000 + m, "2026-05-30") for m in range(3)]
        day2 = [make_bar("ETH-USD", m, 2100 + m, "2026-05-31") for m in range(3)]
        storage.save_bars("ETH-USD", day1 + day2)
        dates = storage.list_available_dates("ETH-USD")
        assert dates == [date(2026, 5, 30), date(2026, 5, 31)], f"got {dates}"
        f1 = storage._file_path("ETH-USD", date(2026, 5, 30))
        f2 = storage._file_path("ETH-USD", date(2026, 5, 31))
        assert f1.exists() and f2.exists()
        print(f"  ✓ Two day-files created: {[d.isoformat() for d in dates]}")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        failed = True
    print()

    # Test 4: Range query crossing the day boundary
    print("Test 4: Load range spanning both days")
    try:
        s = datetime(2026, 5, 30, 0, 0, tzinfo=timezone.utc)
        e = datetime(2026, 5, 31, 23, 59, tzinfo=timezone.utc)
        loaded = storage.load_bars("ETH-USD", s, e)
        assert len(loaded) == 6, f"expected 6 across two days, got {len(loaded)}"
        assert loaded["timestamp"].is_sorted()
        print(f"  ✓ Loaded {len(loaded)} bars across the day boundary, sorted")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        failed = True
    print()

    # Test 5: DataFrame input path (mimics historical fetcher output, no trade_count)
    print("Test 5: Save a DataFrame without trade_count (historical path)")
    try:
        hist = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 5, 30, 11, m, tzinfo=timezone.utc) for m in range(4)
                ],
                "open": [50.0, 51.0, 52.0, 53.0],
                "high": [51.0, 52.0, 53.0, 54.0],
                "low": [49.0, 50.0, 51.0, 52.0],
                "close": [50.5, 51.5, 52.5, 53.5],
                "volume": [100.0, 200.0, 150.0, 175.0],
            }
        )
        written = storage.save_bars("SOL-USD", hist)
        s = datetime(2026, 5, 30, 0, 0, tzinfo=timezone.utc)
        e = datetime(2026, 5, 30, 23, 59, tzinfo=timezone.utc)
        loaded = storage.load_bars("SOL-USD", s, e)
        assert written == 4, f"expected 4 written, got {written}"
        assert loaded["trade_count"].is_null().all(), "trade_count should be null for historical"
        print(f"  ✓ Historical DataFrame stored; trade_count null as expected")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        failed = True
    print()

    # Test 6: Loading a pair with no data returns empty, correctly typed
    print("Test 6: Empty load for unknown pair")
    try:
        empty = storage.load_bars(
            "DOGE-USD",
            datetime(2026, 5, 30, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 30, 23, 59, tzinfo=timezone.utc),
        )
        assert len(empty) == 0
        assert empty.columns == list(CANONICAL_SCHEMA.keys())
        print(f"  ✓ Empty result with correct schema: {empty.columns}")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        failed = True
    print()

    # Clean up the temp directory
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Cleaned up temp dir: {tmp_dir}")
    print()

    if failed:
        print("✗ Some tests failed.")
        sys.exit(1)
    print("All tests passed. ✓")