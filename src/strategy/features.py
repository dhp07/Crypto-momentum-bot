"""
Indicator computation — pure functions over OHLCV bars.

These are deliberately separated from strategy logic (breakout.py) so they can be
tested in isolation and reused by other strategies. Nothing here knows anything
about signals, positions, or trading — just math over a DataFrame.

All functions take and return Polars DataFrames. Window sizes are passed in by
the caller (which reads them from strategy_params.yaml), never hardcoded here.

Indicators computed:
  - rolling_high:  highest HIGH of the prior N bars (EXCLUDING the current bar)
  - avg_volume:    mean VOLUME over the prior N bars (excluding the current bar)
  - atr:           Wilder's Average True Range over N bars
  - ema_short/long: exponential moving averages of close (trend filter)
  - rsi:           Wilder's Relative Strength Index (momentum / overbought-oversold)
  - macd/macd_signal/macd_hist: MACD line, signal line, histogram (momentum)

The "excluding the current bar" choice on rolling_high and avg_volume is critical:
it prevents look-ahead bias. The current bar must be compared against history that
does NOT include itself, or it could never break out of its own high.
"""

from __future__ import annotations

import polars as pl


def add_rolling_high(df: pl.DataFrame, lookback: int, col_name: str = "rolling_high") -> pl.DataFrame:
    """
    Add a column with the highest HIGH of the prior `lookback` bars, excluding the current bar.

    For bar i, rolling_high = max(high[i-lookback .. i-1]).
    The first `lookback` bars have null rolling_high (not enough history yet).
    """
    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")
    return df.with_columns(
        pl.col("high").shift(1).rolling_max(window_size=lookback).alias(col_name)
    )


def add_avg_volume(df: pl.DataFrame, lookback: int, col_name: str = "avg_volume") -> pl.DataFrame:
    """
    Add a column with the mean VOLUME of the prior `lookback` bars, excluding the current bar.

    For bar i, avg_volume = mean(volume[i-lookback .. i-1]).
    The first `lookback` bars have null avg_volume.
    """
    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")
    return df.with_columns(
        pl.col("volume").shift(1).rolling_mean(window_size=lookback).alias(col_name)
    )


def add_atr(df: pl.DataFrame, lookback: int, col_name: str = "atr") -> pl.DataFrame:
    """
    Add Wilder's Average True Range over `lookback` bars.

    True Range for bar i is the max of:
        high[i] - low[i]
        abs(high[i] - close[i-1])
        abs(low[i]  - close[i-1])

    Wilder's ATR smooths TR with an exponential-style recursion:
        ATR[i] = (ATR[i-1] * (n-1) + TR[i]) / n
    seeded by a simple mean of the first `n` true ranges.

    The first `lookback` bars have null ATR (the seed needs that many TRs).
    """
    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")

    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    df = df.with_columns(tr.alias("_tr"))

    tr_values = df["_tr"].to_list()
    atr_values: list[float | None] = [None] * len(tr_values)

    if len(tr_values) > lookback:
        seed_window = tr_values[1 : lookback + 1]
        if all(v is not None for v in seed_window):
            atr = sum(seed_window) / lookback
            atr_values[lookback] = atr
            for i in range(lookback + 1, len(tr_values)):
                tr_i = tr_values[i]
                if tr_i is None:
                    atr_values[i] = atr
                    continue
                atr = (atr * (lookback - 1) + tr_i) / lookback
                atr_values[i] = atr

    df = df.with_columns(pl.Series(name=col_name, values=atr_values, dtype=pl.Float64))
    df = df.drop("_tr")
    return df


def add_ema(df: pl.DataFrame, period: int, col_name: str) -> pl.DataFrame:
    """
    Add an Exponential Moving Average of the close price.

    EMA weights recent bars more heavily than older ones (unlike a simple
    average), so it reacts faster to trend changes. Polars has this built in
    via ewm_mean with span=period, which uses the standard EMA smoothing
    (alpha = 2 / (period + 1)).

    Used here as a TREND FILTER: when a short-period EMA is above a long-period
    EMA, the recent trend is up.
    """
    return df.with_columns(
        pl.col("close").ewm_mean(span=period, adjust=False).alias(col_name)
    )


def add_rsi(df: pl.DataFrame, lookback: int = 14, col_name: str = "rsi") -> pl.DataFrame:
    """
    Add Wilder's Relative Strength Index over `lookback` bars.

    RSI measures how stretched a price move is, on a 0-100 scale:
        RSI = 100 - 100 / (1 + RS),  where RS = avg_gain / avg_loss

    "Gain" is the up-move from the prior close (0 if the bar fell); "loss" is the
    down-move magnitude (0 if the bar rose). Wilder smooths both with the same
    recursion as ATR (seed = simple mean of the first `lookback` values, then
    avg[i] = (avg[i-1]*(n-1) + value[i]) / n).

    Reading: RSI > 70 is conventionally "overbought" (a move may be exhausted),
    RSI < 30 is "oversold". For a breakout strategy, a breakout into an already
    high RSI (>70) is often the END of a move rather than the start — so RSI can
    serve as an entry FILTER ("don't buy if already overbought").

    The first `lookback` bars are null (the seed needs that many deltas). When
    avg_loss is 0 (only gains in the window), RSI is defined as 100.

    Implemented with an explicit Python recursion for transparency and to match
    Wilder's seeding exactly (small data; clarity over cleverness), mirroring
    the ATR approach already in this module.
    """
    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")

    closes = df["close"].to_list()
    n = len(closes)
    rsi_values: list[float | None] = [None] * n

    # Per-bar gains and losses from the prior close. delta[0] is undefined (no prior).
    gains: list[float] = [0.0] * n
    losses: list[float] = [0.0] * n
    for i in range(1, n):
        if closes[i] is None or closes[i - 1] is None:
            gains[i] = 0.0
            losses[i] = 0.0
            continue
        delta = closes[i] - closes[i - 1]
        gains[i] = delta if delta > 0 else 0.0
        losses[i] = -delta if delta < 0 else 0.0

    # Seed at index `lookback` using deltas from indices 1..lookback (that's
    # `lookback` deltas), then Wilder-smooth forward.
    if n > lookback:
        avg_gain = sum(gains[1 : lookback + 1]) / lookback
        avg_loss = sum(losses[1 : lookback + 1]) / lookback

        def rsi_from(ag: float, al: float) -> float:
            if al == 0:
                return 100.0  # no losses -> maximally strong
            rs = ag / al
            return 100.0 - 100.0 / (1.0 + rs)

        rsi_values[lookback] = rsi_from(avg_gain, avg_loss)
        for i in range(lookback + 1, n):
            avg_gain = (avg_gain * (lookback - 1) + gains[i]) / lookback
            avg_loss = (avg_loss * (lookback - 1) + losses[i]) / lookback
            rsi_values[i] = rsi_from(avg_gain, avg_loss)

    return df.with_columns(pl.Series(name=col_name, values=rsi_values, dtype=pl.Float64))


def add_macd(
    df: pl.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    prefix: str = "macd",
) -> pl.DataFrame:
    """
    Add MACD: the Moving Average Convergence Divergence momentum indicator.

    Three derived columns (default names macd / macd_signal / macd_hist):
        macd        = EMA(close, fast) - EMA(close, slow)
        macd_signal = EMA(macd, signal)
        macd_hist   = macd - macd_signal

    Reading: when the MACD line crosses ABOVE its signal line (histogram turns
    positive), short-term momentum is strengthening relative to the longer trend
    — a bullish cue. Crossing below is bearish. As an entry filter, a strategy
    might require macd > macd_signal (momentum confirming) before going long.

    NOTE: MACD is built from EMAs of close, so it is correlated with the
    ema_short/ema_long trend filter already in this module. Stacking both may add
    little independent information — worth keeping in mind when interpreting a
    sweep that toggles them together.

    Uses Polars ewm_mean (adjust=False), the standard EMA smoothing. The MACD line
    itself has no nulls (EMA is defined from bar 0), so signal/hist are also dense;
    early values are just less meaningful until the slow EMA stabilizes.
    """
    if not (fast_period > 0 and slow_period > 0 and signal_period > 0):
        raise ValueError("MACD periods must all be positive")
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be < slow_period ({slow_period})"
        )

    df = df.with_columns(
        pl.col("close").ewm_mean(span=fast_period, adjust=False).alias("_ema_fast"),
        pl.col("close").ewm_mean(span=slow_period, adjust=False).alias("_ema_slow"),
    )
    df = df.with_columns(
        (pl.col("_ema_fast") - pl.col("_ema_slow")).alias(prefix)
    )
    df = df.with_columns(
        pl.col(prefix).ewm_mean(span=signal_period, adjust=False).alias(f"{prefix}_signal")
    )
    df = df.with_columns(
        (pl.col(prefix) - pl.col(f"{prefix}_signal")).alias(f"{prefix}_hist")
    )
    df = df.drop(["_ema_fast", "_ema_slow"])
    return df


def compute_features(
    df: pl.DataFrame,
    breakout_lookback: int,
    volume_lookback: int,
    atr_lookback: int,
    ema_short_period: int = 20,
    ema_long_period: int = 50,
    rsi_lookback: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pl.DataFrame:
    """Add all indicator columns the strategy (and sweep) may need."""
    df = add_rolling_high(df, breakout_lookback)
    df = add_avg_volume(df, volume_lookback)
    df = add_atr(df, atr_lookback)
    df = add_ema(df, ema_short_period, "ema_short")
    df = add_ema(df, ema_long_period, "ema_long")
    df = add_rsi(df, rsi_lookback)
    df = add_macd(df, macd_fast, macd_slow, macd_signal)
    return df


# ============================================================
# Self-test — hand-verifiable synthetic data
# ============================================================


if __name__ == "__main__":
    """
    Run directly to verify indicator math against hand-computed values:
        python3 -m src.strategy.features
    """
    import sys

    failed = False

    def check(name: str, got, expected, tol: float = 1e-6) -> None:
        global failed
        if got is None and expected is None:
            print(f"  ✓ {name}: null (as expected)")
            return
        if got is None or expected is None:
            print(f"  ✗ {name}: got {got}, expected {expected}")
            failed = True
            return
        if abs(got - expected) <= tol:
            print(f"  ✓ {name}: {got:.6f}")
        else:
            print(f"  ✗ {name}: got {got:.6f}, expected {expected:.6f}")
            failed = True

    print("Testing indicators with hand-verifiable data...")
    print()

    # --- Rolling high test ---
    print("Test 1: Rolling high (lookback=2)")
    df1 = pl.DataFrame({
        "high":   [10.0, 12.0, 11.0, 15.0, 13.0],
        "low":    [9.0, 11.0, 10.0, 14.0, 12.0],
        "close":  [9.5, 11.5, 10.5, 14.5, 12.5],
        "volume": [100.0, 100.0, 100.0, 100.0, 100.0],
    })
    r1 = add_rolling_high(df1, lookback=2)["rolling_high"].to_list()
    check("bar 0 (null)", r1[0], None)
    check("bar 1 (null)", r1[1], None)
    check("bar 2 = max(10,12)", r1[2], 12.0)
    check("bar 3 = max(12,11)", r1[3], 12.0)
    check("bar 4 = max(11,15)", r1[4], 15.0)
    print()

    # --- Average volume test ---
    print("Test 2: Average volume (lookback=2)")
    df2 = pl.DataFrame({
        "high":   [1.0, 1.0, 1.0, 1.0],
        "low":    [1.0, 1.0, 1.0, 1.0],
        "close":  [1.0, 1.0, 1.0, 1.0],
        "volume": [100.0, 200.0, 300.0, 400.0],
    })
    r2 = add_avg_volume(df2, lookback=2)["avg_volume"].to_list()
    check("bar 0 (null)", r2[0], None)
    check("bar 1 (null)", r2[1], None)
    check("bar 2 = mean(100,200)", r2[2], 150.0)
    check("bar 3 = mean(200,300)", r2[3], 250.0)
    print()

    # --- ATR test (Wilder's, lookback=3) ---
    print("Test 3: Wilder's ATR (lookback=3)")
    df3 = pl.DataFrame({
        "high":   [10.0, 11.0, 12.0, 13.0, 20.0],
        "low":    [8.0, 9.0, 10.0, 11.0, 11.0],
        "close":  [9.0, 10.0, 11.0, 12.0, 19.0],
        "volume": [100.0, 100.0, 100.0, 100.0, 100.0],
    })
    r3 = add_atr(df3, lookback=3)["atr"].to_list()
    check("bar 0 (null)", r3[0], None)
    check("bar 2 (null, pre-seed)", r3[2], None)
    check("bar 3 = seed mean(2,2,2)", r3[3], 2.0)
    check("bar 4 = (2*2+9)/3", r3[4], 13.0 / 3.0)
    print()

    # --- RSI test (Wilder's, lookback=3) ---
    # closes: 10, 11, 12, 13, 12, 11  -> deltas from index 1: +1,+1,+1,-1,-1
    #  gains:  [_, 1, 1, 1, 0, 0]   losses: [_, 0, 0, 0, 1, 1]
    # Seed at index 3 (lookback=3) using deltas 1..3 (the three +1s):
    #   avg_gain = (1+1+1)/3 = 1.0,  avg_loss = (0+0+0)/3 = 0.0
    #   avg_loss == 0 -> RSI = 100.0 at bar 3
    # bar 4: gain=0, loss=1
    #   avg_gain = (1.0*2 + 0)/3 = 0.6667;  avg_loss = (0.0*2 + 1)/3 = 0.3333
    #   RS = 0.6667/0.3333 = 2.0 -> RSI = 100 - 100/3 = 66.6667
    # bar 5: gain=0, loss=1
    #   avg_gain = (0.6667*2 + 0)/3 = 0.4444;  avg_loss = (0.3333*2 + 1)/3 = 0.5556
    #   RS = 0.4444/0.5556 = 0.8 -> RSI = 100 - 100/1.8 = 44.4444
    print("Test 4: Wilder's RSI (lookback=3)")
    df4 = pl.DataFrame({
        "high":   [10.0, 11.0, 12.0, 13.0, 12.0, 11.0],
        "low":    [10.0, 11.0, 12.0, 13.0, 12.0, 11.0],
        "close":  [10.0, 11.0, 12.0, 13.0, 12.0, 11.0],
        "volume": [100.0] * 6,
    })
    r4 = add_rsi(df4, lookback=3)["rsi"].to_list()
    check("bar 2 (null, pre-seed)", r4[2], None)
    check("bar 3 (all gains -> 100)", r4[3], 100.0)
    check("bar 4 = 66.6667", r4[4], 200.0 / 3.0)
    check("bar 5 = 44.4444", r4[5], 100.0 - 100.0 / 1.8)
    print()

    # --- MACD test ---
    # Verify the structural identity rather than hand-rolling EMAs: the histogram
    # must equal macd - macd_signal exactly, and macd must equal fast EMA - slow
    # EMA. We check on a ramp where everything is dense (no nulls).
    print("Test 5: MACD structural identities")
    closes = [100.0 + i for i in range(60)]
    df5 = pl.DataFrame({
        "high": closes, "low": closes, "close": closes,
        "volume": [100.0] * 60,
    })
    m = add_macd(df5, fast_period=12, slow_period=26, signal_period=9)
    macd_v = m["macd"].to_list()
    sig_v = m["macd_signal"].to_list()
    hist_v = m["macd_hist"].to_list()
    # independent recompute of fast/slow EMA to confirm macd = fast - slow
    fast = df5["close"].ewm_mean(span=12, adjust=False).to_list()
    slow = df5["close"].ewm_mean(span=26, adjust=False).to_list()
    ok_macd = all(abs(macd_v[i] - (fast[i] - slow[i])) < 1e-9 for i in range(60))
    ok_hist = all(abs(hist_v[i] - (macd_v[i] - sig_v[i])) < 1e-9 for i in range(60))
    # On a steady uptrend, fast EMA > slow EMA, so macd should be positive late
    ok_sign = macd_v[-1] > 0
    check("macd == fastEMA - slowEMA (all bars)", 1.0 if ok_macd else 0.0, 1.0)
    check("hist == macd - signal (all bars)", 1.0 if ok_hist else 0.0, 1.0)
    check("macd positive on steady uptrend", 1.0 if ok_sign else 0.0, 1.0)
    print()

    # --- Integration test: compute_features adds every column ---
    print("Test 6: compute_features adds all columns")
    df6 = pl.DataFrame({
        "timestamp": [None] * 60,
        "open":   [100.0 + i for i in range(60)],
        "high":   [101.0 + i for i in range(60)],
        "low":    [99.0 + i for i in range(60)],
        "close":  [100.5 + i for i in range(60)],
        "volume": [1000.0 + i * 10 for i in range(60)],
    })
    out = compute_features(df6, breakout_lookback=30, volume_lookback=30, atr_lookback=14)
    expected_cols = {
        "rolling_high", "avg_volume", "atr", "ema_short", "ema_long",
        "rsi", "macd", "macd_signal", "macd_hist",
    }
    if expected_cols.issubset(set(out.columns)):
        print("  ✓ All indicator columns present")
        last = out.tail(1)
        vals = {c: last[c][0] for c in expected_cols}
        if all(v is not None for v in vals.values()):
            print(f"  ✓ Last bar fully populated (rsi={vals['rsi']:.2f}, macd={vals['macd']:.4f})")
        else:
            nulls = [c for c, v in vals.items() if v is None]
            print(f"  ✗ Last bar has nulls in: {nulls}")
            failed = True
    else:
        print(f"  ✗ Missing columns. Got: {sorted(out.columns)}")
        failed = True
    print()

    if failed:
        print("✗ Some tests failed.")
        sys.exit(1)
    print("All tests passed. ✓")