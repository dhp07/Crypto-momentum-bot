"""
Volume-confirmed breakout strategy.

The bot's first strategy and the one all the parameters were designed around.

Entry rule (long only, for now):
    A long signal fires on the latest bar when BOTH:
      1. close > rolling_high   (price closed above the prior-N-bar high)
      2. volume >= multiplier * avg_volume   (volume confirmed the breakout)

Using CLOSE (not high) for the breakout test is deliberate: it requires the price
to actually finish above the level, filtering out intrabar wicks that spike through
and fall back. Volume confirmation filters breakouts that lack participation, which
tend to fail.

The stop is computed at signal time as: entry - stop_atr_multiplier * ATR.
The strategy emits the stop with the signal; it does not manage the position after.

All parameters come from strategy_params.yaml via the validated config — nothing
is hardcoded, so the backtester can sweep them and the adaptive layer can vary
them per pair later.
"""

from __future__ import annotations

import logging
from decimal import Decimal

import polars as pl

from config.settings import BotConfig
from src.strategy.base import Signal, SignalType, Strategy


logger = logging.getLogger(__name__)


class VolumeBreakoutStrategy(Strategy):
    """Volume-confirmed breakout. Long entries only; exits handled downstream."""

    def __init__(self, config: BotConfig) -> None:
        super().__init__(name=config.strategy.name)
        # Pull the parameters once at construction
        self._volume_multiplier = config.strategy.volume_multiplier_threshold
        self._stop_atr_multiplier = config.exits.stop_atr_multiplier
        logger.info(
            f"VolumeBreakoutStrategy initialized: "
            f"volume_multiplier={self._volume_multiplier}, "
            f"stop_atr_multiplier={self._stop_atr_multiplier}"
        )

    def evaluate(
        self,
        product_id: str,
        features: pl.DataFrame,
        position_open: bool,
    ) -> Signal | None:
        # Already in a position for this pair — never stack a second entry
        if position_open:
            return None

        # Need at least one bar to evaluate
        if len(features) == 0:
            return None

        # We evaluate the most recent (last) bar
        bar = features.tail(1)

        # Pull the values we need
        close = bar["close"][0]
        volume = bar["volume"][0]
        rolling_high = bar["rolling_high"][0]
        avg_volume = bar["avg_volume"][0]
        atr = bar["atr"][0]

        # Minimum-data guard: if any indicator is null (not enough history yet,
        # e.g. a thin pair like AVAX with gaps), do not trade. This is the guard
        # we flagged back in Phase 2 when AVAX returned fewer bars than requested.
        if rolling_high is None or avg_volume is None or atr is None:
            logger.debug(
                f"{product_id}: indicators not ready "
                f"(rolling_high={rolling_high}, avg_volume={avg_volume}, atr={atr}); skipping"
            )
            return None

        # Guard against a zero/garbage average volume (a fully flat window)
        if avg_volume <= 0:
            logger.debug(f"{product_id}: avg_volume <= 0; skipping")
            return None

        # --- The two breakout conditions ---
        broke_out = close > rolling_high
        volume_confirmed = volume >= self._volume_multiplier * avg_volume

        if not (broke_out and volume_confirmed):
            return None

        # Both conditions met — build the signal with its stop
        close_dec = Decimal(str(close))
        atr_dec = Decimal(str(atr))
        stop_dec = close_dec - (Decimal(str(self._stop_atr_multiplier)) * atr_dec)

        volume_ratio = volume / avg_volume
        reason = (
            f"close {close:.4f} > rolling_high {rolling_high:.4f} "
            f"and volume {volume:.2f} = {volume_ratio:.2f}x avg "
            f"(>= {self._volume_multiplier}x); stop @ {float(stop_dec):.4f} "
            f"({self._stop_atr_multiplier}x ATR {atr:.4f})"
        )

        logger.info(f"SIGNAL {product_id}: {reason}")

        return Signal(
            signal_type=SignalType.ENTER_LONG,
            product_id=product_id,
            timestamp=bar["timestamp"][0],
            price=close_dec,
            stop_price=stop_dec,
            reason=reason,
        )


# ============================================================
# Self-test — synthetic feature frames with known outcomes
# ============================================================


if __name__ == "__main__":
    """
    Run directly to verify the strategy fires exactly when it should:
        python3 -m src.strategy.volume_breakout
    """
    import sys
    from datetime import datetime, timezone

    from config.settings import load_strategy_config
    from src.strategy.features import compute_features

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    failed = False

    def expect(name: str, condition: bool) -> None:
        global failed
        if condition:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            failed = True

    # Load the real config so we test against your actual locked parameters
    config = load_strategy_config()
    strategy = VolumeBreakoutStrategy(config)
    print(f"Loaded strategy '{strategy.name}' with your real parameters")
    print(f"  volume_multiplier = {config.strategy.volume_multiplier_threshold}")
    print(f"  stop_atr_multiplier = {config.exits.stop_atr_multiplier}")
    print()

    lookback = config.strategy.breakout_lookback_bars
    vol_lookback = config.strategy.volume_lookback_bars
    atr_lookback = config.exits.atr_lookback_bars
    n = max(lookback, vol_lookback, atr_lookback) + 5

    def build_frame(last_close: float, last_volume: float,
                    base_price: float = 100.0, base_vol: float = 1000.0) -> pl.DataFrame:
        """Build a flat baseline frame, then override the final bar's close & volume."""
        closes = [base_price] * (n - 1) + [last_close]
        highs = [base_price + 0.5] * (n - 1) + [max(last_close, base_price + 0.5)]
        lows = [base_price - 0.5] * (n - 1) + [min(base_price - 0.5, last_close)]
        opens = [base_price] * n
        vols = [base_vol] * (n - 1) + [last_volume]
        ts = [datetime(2026, 6, 1, 0, m, tzinfo=timezone.utc) for m in range(n)]
        df = pl.DataFrame({
            "timestamp": ts, "open": opens, "high": highs,
            "low": lows, "close": closes, "volume": vols,
        })
        return compute_features(df, lookback, vol_lookback, atr_lookback)

    mult = config.strategy.volume_multiplier_threshold

    # Test 1: breakout + volume confirmed -> SIGNAL
    print("Test 1: Breakout WITH volume confirmation -> fires")
    feats = build_frame(last_close=200.0, last_volume=1000.0 * mult * 1.5)
    sig = strategy.evaluate("BTC-USD", feats, position_open=False)
    expect("signal emitted", sig is not None)
    if sig:
        expect("type is ENTER_LONG", sig.signal_type == SignalType.ENTER_LONG)
        expect("stop below entry", sig.stop_price < sig.price)
        expect("risk_per_unit positive", sig.risk_per_unit > 0)
    print()

    # Test 2: breakout but volume NOT confirmed -> no signal
    print("Test 2: Breakout WITHOUT volume confirmation -> silent")
    feats = build_frame(last_close=200.0, last_volume=1000.0 * 0.5)  # low volume
    sig = strategy.evaluate("BTC-USD", feats, position_open=False)
    expect("no signal (volume too low)", sig is None)
    print()

    # Test 3: volume spike but NO breakout -> no signal
    print("Test 3: Volume spike WITHOUT breakout -> silent")
    feats = build_frame(last_close=100.0, last_volume=1000.0 * mult * 2)  # no price break
    sig = strategy.evaluate("BTC-USD", feats, position_open=False)
    expect("no signal (no breakout)", sig is None)
    print()

    # Test 4: conditions met BUT position already open -> no signal
    print("Test 4: Conditions met BUT already in position -> silent")
    feats = build_frame(last_close=200.0, last_volume=1000.0 * mult * 1.5)
    sig = strategy.evaluate("BTC-USD", feats, position_open=True)
    expect("no signal (already long)", sig is None)
    print()

    # Test 5: null indicators (insufficient data) -> no signal
    print("Test 5: Insufficient data (null indicators) -> silent")
    short = pl.DataFrame({
        "timestamp": [datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)],
        "open": [100.0], "high": [101.0], "low": [99.0],
        "close": [200.0], "volume": [99999.0],
    })
    short = compute_features(short, lookback, vol_lookback, atr_lookback)
    sig = strategy.evaluate("BTC-USD", short, position_open=False)
    expect("no signal (indicators null)", sig is None)
    print()

    # Test 6: stop distance equals 1.5x ATR (within rounding)
    print("Test 6: Stop is exactly stop_atr_multiplier x ATR below entry")
    feats = build_frame(last_close=200.0, last_volume=1000.0 * mult * 1.5)
    sig = strategy.evaluate("BTC-USD", feats, position_open=False)
    if sig:
        atr_val = feats.tail(1)["atr"][0]
        expected_stop = float(sig.price) - config.exits.stop_atr_multiplier * atr_val
        actual_stop = float(sig.stop_price)
        expect(
            f"stop {actual_stop:.4f} == entry - {config.exits.stop_atr_multiplier}xATR {expected_stop:.4f}",
            abs(actual_stop - expected_stop) < 1e-6,
        )
    else:
        expect("signal existed to check stop", False)
    print()

    if failed:
        print("✗ Some tests failed.")
        sys.exit(1)
    print("All tests passed. ✓")