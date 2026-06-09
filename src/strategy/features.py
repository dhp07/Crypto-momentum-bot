"""
Indicator computation — pure functions over OHLCV bars.

These are deliberately separated from strategy logic (volume_breakout.py, later)
so they can be tested in isolation and reused by other strategies. Nothing here
knows anything about signals, positions, or trading — just math over a DataFrame.

All functions take and return Polars DataFrames. Window sizes are passed in by
the caller (which reads them from strategy_params.yaml), never hardcoded here.

Indicators computed:
  - rolling_high:  highest HIGH of the prior N bars (EXCLUDING the current bar)
  - avg_volume:    mean VOLUME over the prior N bars (excluding the current bar)
  - atr:           Wilder's Average True Range over N bars

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
        # shift(1) drops the current bar out of the window, then rolling_max over `lookback`
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

    # True range components
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    df = df.with_columns(tr.alias("_tr"))

    # Wilder's smoothing. Polars' ewm with alpha = 1/n IS Wilder's smoothing,
    # but Wilder seeds with a simple average of the first n TRs rather than the
    # first TR value. We reproduce that exactly with a manual recursion in Python
    # for correctness and transparency (the data is small; clarity beats cleverness).
    tr_values = df["_tr"].to_list()
    atr_values: list[float | None] = [None] * len(tr_values)

    # Need `lookback` true ranges to seed. TR[0] is null (no prev_close), so the
    # first usable TR is at index 1. Seed ATR at index `lookback` using TR[1..lookback].
    if len(tr_values) > lookback:
        seed_window = tr_values[1 : lookback + 1]
        if all(v is not None for v in seed_window):
            atr = sum(seed_window) / lookback
            atr_values[lookback] = atr
            for i in range(lookback + 1, len(tr_values)):
                tr_i = tr_values[i]
                if tr_i is None:
                    atr_values[i] = atr  # carry forward if a TR is missing
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
    EMA, the recent trend is up. The strategy only takes long entries in that
    condition — i.e. it won't buy breakouts during a downtrend.

    Args:
        df: must contain a 'close' column
        period: EMA span in bars
        col_name: name for the output column (e.g. 'ema_short')
    """
    return df.with_columns(
        pl.col("close").ewm_mean(span=period, adjust=False).alias(col_name)
    )

def compute_features(
    df: pl.DataFrame,
    breakout_lookback: int,
    volume_lookback: int,
    atr_lookback: int,
    ema_short_period: int = 20,
    ema_long_period: int = 50,
) -> pl.DataFrame:
    """Add all indicator columns the strategy needs."""
    df = add_rolling_high(df, breakout_lookback)
    df = add_avg_volume(df, volume_lookback)
    df = add_atr(df, atr_lookback)
    df = add_ema(df, ema_short_period, "ema_short")
    df = add_ema(df, ema_long_period, "ema_long")
    return df


# ============================================================
# Self-test — hand-verifiable synthetic data
# ============================================================


if __name__ == "__main__":
    """
    Run directly to verify indicator math against hand-computed values:
        python3 -m src.strategy.indicators
    """
    import sys

    failed = False

    def check(name: str, got, expected, tol: float = 1e-9) -> None:
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
    # highs: 10, 12, 11, 15, 13  with lookback=2
    # bar 0: null (no history)
    # bar 1: max(high[0])      ... wait, lookback=2 needs 2 prior bars
    # bar 2: max(10,12) = 12
    # bar 3: max(12,11) = 12
    # bar 4: max(11,15) = 15
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
    # volumes: 100, 200, 300, 400 with lookback=2
    # bar 2: mean(100,200) = 150
    # bar 3: mean(200,300) = 250
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
    # Construct simple bars where TR is easy to compute.
    # high, low, close:
    #  bar0: H=10 L=8  C=9     TR0 = null (no prev close)
    #  bar1: H=11 L=9  C=10    TR1 = max(11-9, |11-9|, |9-9|)   = max(2,2,0)   = 2
    #  bar2: H=12 L=10 C=11    TR2 = max(12-10,|12-10|,|10-10|) = max(2,2,0)   = 2
    #  bar3: H=13 L=11 C=12    TR3 = max(13-11,|13-11|,|11-11|) = max(2,2,0)   = 2
    #  bar4: H=20 L=11 C=19    TR4 = max(20-11,|20-12|,|11-12|) = max(9,8,1)   = 9
    # Seed ATR at bar3 = mean(TR1,TR2,TR3) = mean(2,2,2) = 2
    # ATR4 = (ATR3*(3-1) + TR4)/3 = (2*2 + 9)/3 = 13/3 = 4.333333
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

    # --- Integration test: compute_features adds all three columns ---
    print("Test 4: compute_features adds all columns")
    df4 = pl.DataFrame({
        "timestamp": [None] * 40,
        "open":   [100.0 + i for i in range(40)],
        "high":   [101.0 + i for i in range(40)],
        "low":    [99.0 + i for i in range(40)],
        "close":  [100.5 + i for i in range(40)],
        "volume": [1000.0 + i * 10 for i in range(40)],
    })
    out = compute_features(df4, breakout_lookback=30, volume_lookback=30, atr_lookback=14)
    expected_cols = {"rolling_high", "avg_volume", "atr"}
    if expected_cols.issubset(set(out.columns)):
        print(f"  ✓ All three indicator columns present")
        # Last bar should have non-null values for all three (40 bars > all windows)
        last = out.tail(1)
        rh = last["rolling_high"][0]
        av = last["avg_volume"][0]
        at = last["atr"][0]
        if rh is not None and av is not None and at is not None:
            print(f"  ✓ Last bar fully populated: rolling_high={rh:.2f} avg_volume={av:.2f} atr={at:.4f}")
        else:
            print(f"  ✗ Last bar has nulls: rolling_high={rh} avg_volume={av} atr={at}")
            failed = True
    else:
        print(f"  ✗ Missing columns. Got: {out.columns}")
        failed = True
    print()

    if failed:
        print("✗ Some tests failed.")
        sys.exit(1)
    print("All tests passed. ✓")