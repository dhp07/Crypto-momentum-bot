"""
Collect historical bars for backtesting.

Fetches 1-minute OHLCV bars for every configured pair over the last N days and
stores them to Parquet via BarStorage. Run this ONCE; the backtester then reads
from storage repeatedly without re-fetching (important when we later sweep
parameters and don't want to re-download data every run).

Reuses the Phase 2 data layer: CoinbaseClient -> MarketDataFetcher -> BarStorage.

Usage (on the server, where .env credentials live):
    python3 -m backtest.collect_data --days 30
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone

from config.settings import load_config
from src.data.coinbase_client import CoinbaseClient
from src.data.market_data import MarketDataFetcher
from src.data.storage import BarStorage


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect historical bars for backtesting")
    parser.add_argument("--days", type=int, default=30, help="How many days of history to fetch")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    config, _env = load_config()
    pairs = config.universe.pairs

    client = CoinbaseClient()
    fetcher = MarketDataFetcher(client)
    storage = BarStorage()

    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=args.days)

    print(f"Collecting {args.days} days of 1-min bars for {len(pairs)} pairs.")
    print(f"Range: {start.isoformat()}  ->  {end.isoformat()}")
    print("This makes many paginated API calls and will take several minutes.")
    print("Progress prints per pair; that's normal — it is not stuck.")
    print()

    total_stored = 0
    for i, pair in enumerate(pairs, start=1):
        print(f"[{i}/{len(pairs)}] Fetching {pair} ...")
        df = fetcher.fetch_historical_bars(pair, start=start, end=end)
        stored = storage.save_bars(pair, df)
        total_stored += stored
        print(f"    {pair}: fetched {len(df)} bars, stored {stored}")
        dates = storage.list_available_dates(pair)
        if dates:
            print(f"    {pair}: storage now covers {dates[0]} .. {dates[-1]} ({len(dates)} day-files)")
        print()

    print(f"Done. Stored {total_stored} bars total across {len(pairs)} pairs.")
    print("Now run:  python3 -m backtest.backtest")


if __name__ == "__main__":
    main()