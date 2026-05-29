"""
Coinbase Advanced Trade API client.

Wraps the official coinbase-advanced-py SDK with our own interface so that:
- The rest of the bot doesn't need to know SDK-specific details
- We can swap to a different exchange later by writing a new client with the same interface
- All API calls go through one place with consistent logging, retry, and error handling

This is the lowest-level data module — it handles raw API calls only.
Higher-level modules (market_data.py, websocket_handler.py) build on top of this.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from coinbase.rest import RESTClient

from config.settings import EnvSettings, env


logger = logging.getLogger(__name__)


# ============================================================
# Data classes — our internal representations
# ============================================================
# These are our own types, not the SDK's. Decoupling means we can swap
# exchanges without changing every file that handles account/candle data.


@dataclass(frozen=True)
class AccountBalance:
    """Balance for a single currency on the exchange."""

    currency: str
    available: Decimal       # Free balance, usable for new trades
    hold: Decimal            # Locked in open orders
    total: Decimal           # available + hold

    @property
    def is_zero(self) -> bool:
        return self.total == 0


@dataclass(frozen=True)
class Candle:
    """OHLCV bar — one period of price action."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


# ============================================================
# The client
# ============================================================


class CoinbaseClient:
    """
    REST client for Coinbase Advanced Trade.

    Handles auth, retries, and translation between Coinbase's API and our internal types.
    Construct once at bot startup; reuse for the lifetime of the bot.
    """

    # Conservative rate limit. Coinbase allows ~120 req/min on most endpoints;
    # we stay well under to leave headroom for websocket auth and bursts.
    MIN_REQUEST_INTERVAL_SECONDS: float = 0.6

    # Retry config
    MAX_RETRIES: int = 3
    INITIAL_BACKOFF_SECONDS: float = 1.0

    def __init__(self, env_settings: EnvSettings | None = None) -> None:
        """
        Initialize the client.

        Args:
            env_settings: Environment settings object. If None, uses the global singleton.
        """
        settings = env_settings or env

        # The SDK reads two strings: the key NAME (an organization/key path)
        # and the PEM-encoded EC private key. Both come from .env via settings.
        self._sdk = RESTClient(
            api_key=settings.coinbase_api_key_name,
            api_secret=settings.coinbase_api_private_key,
        )

        self._last_request_at: float = 0.0

        logger.info("CoinbaseClient initialized")

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _rate_limit(self) -> None:
        """Block briefly if needed to respect the minimum interval between calls."""
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(self.MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        self._last_request_at = time.monotonic()

    def _call_with_retry(self, fn_name: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Call an SDK function with rate limiting and exponential backoff on failure.

        We retry on transient errors (network issues, 5xx server errors).
        We do NOT retry on 4xx client errors (bad request, auth failure) — those
        won't fix themselves and retrying just wastes rate limit budget.
        """
        last_exception: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            self._rate_limit()
            try:
                result = fn(*args, **kwargs)
                logger.debug(f"{fn_name} succeeded on attempt {attempt + 1}")
                return result
            except Exception as exc:
                last_exception = exc
                # Heuristic: error messages containing these substrings are usually transient
                error_str = str(exc).lower()
                is_transient = any(
                    marker in error_str
                    for marker in ("timeout", "connection", "5xx", "500", "502", "503", "504")
                )

                if not is_transient:
                    logger.error(f"{fn_name} failed with non-retryable error: {exc}")
                    raise

                if attempt < self.MAX_RETRIES - 1:
                    backoff = self.INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(
                        f"{fn_name} failed (attempt {attempt + 1}/{self.MAX_RETRIES}): {exc}. "
                        f"Retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)

        # Should not reach here, but just in case
        assert last_exception is not None
        raise last_exception

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def health_check(self) -> bool:
        """
        Verify we can authenticate and reach the API.

        Returns True if successful. Used at bot startup to fail fast if
        credentials are wrong or Coinbase is down.
        """
        try:
            # Listing accounts is a low-cost authenticated call
            self._call_with_retry("health_check", self._sdk.get_accounts, limit=1)
            logger.info("Coinbase API health check passed")
            return True
        except Exception as exc:
            logger.error(f"Coinbase API health check failed: {exc}")
            return False

    def get_accounts(self) -> list[AccountBalance]:
        """Fetch all account balances. Returns a list of AccountBalance, one per currency."""
        response = self._call_with_retry("get_accounts", self._sdk.get_accounts)

        balances: list[AccountBalance] = []
        for account in response.accounts:
            available = Decimal(str(account.available_balance["value"]))
            hold = Decimal(str(account.hold["value"]))
            balances.append(
                AccountBalance(
                    currency=account.currency,
                    available=available,
                    hold=hold,
                    total=available + hold,
                )
            )

        logger.debug(f"Retrieved {len(balances)} account balances")
        return balances

    def get_account(self, currency: str) -> AccountBalance | None:
        """
        Fetch the balance for a specific currency.

        Returns None if no account exists for that currency (e.g. you've never held SOL).
        """
        currency = currency.upper()
        for balance in self.get_accounts():
            if balance.currency == currency:
                return balance
        return None

    def get_product(self, product_id: str) -> dict[str, Any]:
        """
        Fetch metadata for a trading pair.

        Returns the raw dict from Coinbase (includes min/max sizes, current price, status, etc).
        We pass it through as-is since callers may need different fields.
        """
        response = self._call_with_retry(
            "get_product",
            self._sdk.get_product,
            product_id=product_id,
        )
        # The SDK returns a typed object; convert to dict for simpler downstream use
        return response.to_dict() if hasattr(response, "to_dict") else dict(response)

    def get_candles(
        self,
        product_id: str,
        start: datetime,
        end: datetime,
        granularity: str = "ONE_MINUTE",
    ) -> list[Candle]:
        """
        Fetch historical OHLCV candles for a product.

        Args:
            product_id: e.g. "BTC-USD"
            start: start of the time range (inclusive)
            end: end of the time range (inclusive)
            granularity: ONE_MINUTE | FIVE_MINUTE | FIFTEEN_MINUTE | THIRTY_MINUTE |
                         ONE_HOUR | TWO_HOUR | SIX_HOUR | ONE_DAY

        Returns:
            List of Candle objects, sorted oldest to newest.

        Note: Coinbase caps a single request at 350 candles. For longer ranges,
        higher-level code (in market_data.py) will paginate.
        """
        # Coinbase expects UNIX timestamps as strings
        start_ts = str(int(start.replace(tzinfo=timezone.utc).timestamp()))
        end_ts = str(int(end.replace(tzinfo=timezone.utc).timestamp()))

        response = self._call_with_retry(
            "get_candles",
            self._sdk.get_candles,
            product_id=product_id,
            start=start_ts,
            end=end_ts,
            granularity=granularity,
        )

        candles: list[Candle] = []
        for raw in response.candles:
            candles.append(
                Candle(
                    timestamp=datetime.fromtimestamp(int(raw.start), tz=timezone.utc),
                    open=Decimal(str(raw.open)),
                    high=Decimal(str(raw.high)),
                    low=Decimal(str(raw.low)),
                    close=Decimal(str(raw.close)),
                    volume=Decimal(str(raw.volume)),
                )
            )

        # Coinbase returns newest-first; we want oldest-first for natural iteration
        candles.sort(key=lambda c: c.timestamp)
        logger.debug(
            f"Retrieved {len(candles)} {granularity} candles for {product_id} "
            f"from {start.isoformat()} to {end.isoformat()}"
        )
        return candles


# ============================================================
# Self-test
# ============================================================


if __name__ == "__main__":
    """
    Run this file directly to verify the client works:
        python3 src/data/coinbase_client.py
    """
    import sys

    # Set up basic logging so we can see what's happening
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Testing Coinbase API client...")
    print()

    try:
        client = CoinbaseClient()
    except Exception as exc:
        print(f"✗ Failed to create client: {exc}")
        print("  Check that .env has COINBASE_API_KEY_NAME and COINBASE_API_PRIVATE_KEY set")
        sys.exit(1)

    # Test 1: Health check
    print("Test 1: Health check (auth + connectivity)")
    if not client.health_check():
        print("✗ Health check failed. See logs above for the error.")
        sys.exit(1)
    print("  ✓ Authentication and connectivity OK")
    print()

    # Test 2: List accounts
    print("Test 2: List account balances")
    try:
        accounts = client.get_accounts()
        non_zero = [a for a in accounts if not a.is_zero]
        print(f"  ✓ Retrieved {len(accounts)} accounts ({len(non_zero)} with non-zero balance)")
        if non_zero:
            print("  Balances:")
            for a in non_zero[:10]:  # Show at most 10
                print(f"    {a.currency:8} available={a.available} hold={a.hold}")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        sys.exit(1)
    print()

    # Test 3: Get product info for BTC-USD
    print("Test 3: Get BTC-USD product info")
    try:
        product = client.get_product("BTC-USD")
        # Field names vary slightly by SDK version; show whatever's there
        price = product.get("price") or product.get("mid_market_price") or "unknown"
        print(f"  ✓ BTC-USD current price: {price}")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        sys.exit(1)
    print()

    # Test 4: Fetch recent candles
    print("Test 4: Fetch last hour of 1-minute candles for BTC-USD")
    try:
        from datetime import timedelta
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=1)
        candles = client.get_candles("BTC-USD", start=start, end=end, granularity="ONE_MINUTE")
        print(f"  ✓ Retrieved {len(candles)} candles")
        if candles:
            latest = candles[-1]
            print(f"  Most recent: {latest.timestamp.isoformat()} "
                  f"open={latest.open} high={latest.high} "
                  f"low={latest.low} close={latest.close} volume={latest.volume}")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        sys.exit(1)
    print()

    print("All tests passed. ✓")