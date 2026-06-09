"""
Backtester — replays stored historical bars through the live TradingEngine and
reports honest performance metrics.

This is the SAME engine the live bot uses (src/engine.py). We just feed it
historical bars instead of websocket bars — which is the whole reason the engine
was built source-agnostic. So a passing backtest exercises the exact code path
that will trade live.

Bars are fed in TRUE GLOBAL chronological order across all pairs (not pair by
pair), because there is one shared account and the risk gate's concurrent-position
and shared-equity logic only behaves correctly in real time order.

TIMEFRAME / RESAMPLING:
Data is collected and stored at 1-minute resolution (the raw source of truth).
The backtester resamples those 1-minute bars up to the interval configured in
strategy_params.yaml (bar_interval_seconds) before feeding the engine. So setting
bar_interval_seconds: 900 backtests on 15-minute bars built from the stored
1-minute data — no re-collection needed, and every timeframe is tested on the
exact same underlying price history (a clean comparison).

Resampling aggregation per N-minute bucket:
    open   = first open in the bucket
    high   = max high
    low    = min low
    close  = last close
    volume = sum of volume

IMPORTANT framing: a single backtest over one window is an IN-SAMPLE first look.
It tells us whether the strategy has a pulse on real data after fees — NOT whether
the parameters are optimal. Rigorous out-of-sample evaluation (holding out data
the optimizer never sees) comes with parameter sweeps (backtest/sweep.py).

Usage (on the server, after collect_data has stored bars):
    python3 -m backtest.backtest --days 30 --cash 1000
    python3 -m backtest.backtest --selftest      # verify metrics + resample math
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import polars as pl

from config.settings import BotConfig, load_config
from src.data.storage import BarStorage
from src.engine import TradingEngine
from src.execution.executor import PaperExecutor
from src.risk.risk_gate import RiskGate, RiskParams
from src.strategy.breakout import VolumeBreakoutStrategy


logger = logging.getLogger(__name__)


# ============================================================
# Resampling — aggregate 1-minute bars to the configured interval
# ============================================================


def resample_bars(df: pl.DataFrame, interval_seconds: int) -> pl.DataFrame:
    """
    Aggregate 1-minute OHLCV bars up to `interval_seconds`-sized bars.

    The stored data is 1-minute resolution. To backtest on, say, 15-minute bars
    (interval_seconds=900), we group consecutive 1-minute bars into 15-minute
    buckets aligned to the clock and aggregate each bucket into one bar.

    If interval_seconds <= 60, the data is already at (or finer than) the target
    and is returned unchanged — so the 1-minute case is a no-op.

    Assumes df has columns: timestamp, open, high, low, close, volume, sorted by
    timestamp. Returns the same schema at the coarser interval.
    """
    if interval_seconds <= 60:
        return df
    if len(df) == 0:
        return df

    every = f"{interval_seconds}s"

    resampled = (
        df.sort("timestamp")
        .group_by_dynamic("timestamp", every=every, closed="left")
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        )
        .sort("timestamp")
    )
    return resampled


# ============================================================
# Metrics — pure functions, independently testable
# ============================================================


def compute_metrics(
    equity_curve: list[Decimal],
    trade_pnls: list[Decimal],
    total_fees: Decimal,
    starting_cash: Decimal,
) -> dict:
    """
    Turn an equity curve and a list of per-trade P&Ls into honest metrics.

    Pure and deterministic — the --selftest path verifies this against hand
    inputs, separate from any real-data run.
    """
    final_equity = equity_curve[-1] if equity_curve else starting_cash
    total_return_pct = (final_equity - starting_cash) / starting_cash * Decimal("100")

    num_trades = len(trade_pnls)
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    num_wins, num_losses = len(wins), len(losses)
    win_rate = (Decimal(num_wins) / Decimal(num_trades) * Decimal("100")) if num_trades else Decimal("0")

    gross_win = sum(wins, Decimal("0"))
    gross_loss = abs(sum(losses, Decimal("0")))
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        profit_factor = Decimal("inf")
    else:
        profit_factor = Decimal("0")

    avg_win = (gross_win / Decimal(num_wins)) if num_wins else Decimal("0")
    avg_loss = (gross_loss / Decimal(num_losses)) if num_losses else Decimal("0")

    # Max drawdown: largest peak-to-trough decline in the equity curve
    peak: Decimal | None = None
    max_dd = Decimal("0")
    for eq in equity_curve:
        peak = eq if peak is None else max(peak, eq)
        if peak > 0:
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

    return {
        "starting_cash": starting_cash,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "num_trades": num_trades,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "win_rate_pct": win_rate,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown_pct": max_dd * Decimal("100"),
        "total_fees": total_fees,
    }


def print_report(metrics: dict, benchmark_pct: Decimal | None, interval_seconds: int) -> None:
    m = metrics
    pf = m["profit_factor"]
    pf_str = "inf" if pf == Decimal("inf") else f"{float(pf):.2f}"
    interval_label = (
        f"{interval_seconds // 60}-min" if interval_seconds >= 60 else f"{interval_seconds}s"
    )

    print()
    print("=" * 56)
    print(f"  BACKTEST RESULTS  ({interval_label} bars — in-sample, not optimized)")
    print("=" * 56)
    print(f"  Starting capital      ${float(m['starting_cash']):>12,.2f}")
    print(f"  Final equity          ${float(m['final_equity']):>12,.2f}")
    print(f"  Total return          {float(m['total_return_pct']):>12.2f}%")
    if benchmark_pct is not None:
        print(f"  Buy-and-hold (BTC)    {float(benchmark_pct):>12.2f}%   <- did the strategy beat doing nothing?")
    print("-" * 56)
    print(f"  Trades                {m['num_trades']:>12}")
    print(f"  Win rate              {float(m['win_rate_pct']):>12.2f}%   ({m['num_wins']}W / {m['num_losses']}L)")
    print(f"  Avg win               ${float(m['avg_win']):>12.4f}")
    print(f"  Avg loss              ${float(m['avg_loss']):>12.4f}")
    print(f"  Profit factor         {pf_str:>13}   (gross win / gross loss; >1 = profitable)")
    print("-" * 56)
    print(f"  Max drawdown          {float(m['max_drawdown_pct']):>12.2f}%   <- worst peak-to-trough decline")
    print(f"  Total fees paid       ${float(m['total_fees']):>12.4f}")
    print("=" * 56)
    print()
    print("  Reminder: this is ONE in-sample window. A good number here is")
    print("  necessary but NOT sufficient — out-of-sample testing comes next.")
    print()


# ============================================================
# Core replay — shared by the CLI run and the sweep
# ============================================================


def _replay(
    config: BotConfig,
    start: datetime,
    end: datetime,
    starting_cash: Decimal,
) -> tuple[list[Decimal], list[Decimal], Decimal, Decimal | None] | None:
    """
    Load + resample stored bars over [start, end], replay through the engine,
    and return the raw materials for metrics:
        (equity_curve, trade_pnls, total_fees, benchmark_pct)
    Returns None if there is no data in the window.

    This is the shared core. run_backtest() wraps it with printing; the sweep's
    run_backtest_metrics() wraps it returning a metrics dict.
    """
    pairs = config.universe.pairs
    storage = BarStorage()
    interval_seconds = config.strategy.bar_interval_seconds

    frames = []
    for pair in pairs:
        df = storage.load_bars(pair, start, end)
        if len(df) == 0:
            continue
        df = resample_bars(df, interval_seconds)
        if len(df) == 0:
            continue
        frames.append(df.with_columns(pl.lit(pair).alias("pair")))

    if not frames:
        return None

    merged = pl.concat(frames, how="vertical_relaxed").sort("timestamp")

    executor = PaperExecutor(
        starting_cash=starting_cash,
        slippage_bps=Decimal(str(config.execution.assumed_slippage_bps)),
        taker_fee_bps=Decimal(str(config.execution.taker_fee_bps)),
    )
    strategy = VolumeBreakoutStrategy(config)
    gate = RiskGate(RiskParams.from_config(config))
    engine = TradingEngine(config, strategy, gate, executor)

    # Quiet the per-bar INFO logging from components during the replay
    logging.getLogger("src").setLevel(logging.WARNING)

    equity_curve: list[Decimal] = []
    trade_pnls: list[Decimal] = []
    total = len(merged)
    for i, row in enumerate(merged.iter_rows(named=True)):
        bar = {
            "timestamp": row["timestamp"],
            "open": row["open"], "high": row["high"], "low": row["low"],
            "close": row["close"], "volume": row["volume"],
        }
        result = engine.on_bar(row["pair"], bar)
        if result.action == "closed" and result.pnl is not None:
            trade_pnls.append(result.pnl)
        equity_curve.append(engine.status()["equity"])

        if (i + 1) % 20000 == 0:
            print(f"  ... {i + 1:,}/{total:,} bars processed, "
                  f"equity ${float(equity_curve[-1]):,.2f}, trades {len(trade_pnls)}")

    total_fees = sum((f.fee for f in executor.fills), Decimal("0"))

    # Buy-and-hold BTC benchmark over the same window (resampled too)
    benchmark_pct: Decimal | None = None
    btc = storage.load_bars("BTC-USD", start, end)
    if len(btc) > 0:
        btc = resample_bars(btc, interval_seconds)
    if len(btc) >= 2:
        first_close = Decimal(str(btc["close"][0]))
        last_close = Decimal(str(btc["close"][-1]))
        benchmark_pct = (last_close - first_close) / first_close * Decimal("100")

    return equity_curve, trade_pnls, total_fees, benchmark_pct


# ============================================================
# The CLI backtest run (prints a report)
# ============================================================


def run_backtest(days: int, starting_cash: Decimal) -> None:
    config, _env = load_config()
    interval_seconds = config.strategy.bar_interval_seconds

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    interval_label = f"{interval_seconds // 60}-min" if interval_seconds >= 60 else f"{interval_seconds}s"
    print(f"Backtesting on {interval_label} bars (bar_interval_seconds={interval_seconds}).")
    if interval_seconds > 60:
        print("Stored 1-minute bars will be resampled up to this interval before replay.")
    print()

    result = _replay(config, start, end, starting_cash)
    if result is None:
        print("No data found in storage. Run:  python3 -m backtest.collect_data --days 30")
        return

    equity_curve, trade_pnls, total_fees, benchmark_pct = result
    metrics = compute_metrics(equity_curve, trade_pnls, total_fees, starting_cash)
    print_report(metrics, benchmark_pct, interval_seconds)


# ============================================================
# Metrics for the sweep (returns a plain-float dict, no printing)
# ============================================================


def run_backtest_metrics(
    config: BotConfig,
    start: datetime,
    end: datetime,
    starting_cash: Decimal,
) -> dict | None:
    """
    Run one backtest with a GIVEN config over [start, end] and RETURN metrics.

    Used by the sweep (backtest/sweep.py) to evaluate a parameter combination
    in-process, without printing or touching disk. Returns a plain-float metrics
    dict, or None if there is no data in the window.

    Reuses the exact same _replay core as the CLI run_backtest, so a sweep result
    for a given config matches what run_backtest would print for that config.
    """
    result = _replay(config, start, end, starting_cash)
    if result is None:
        return None

    equity_curve, trade_pnls, total_fees, _benchmark = result
    m = compute_metrics(equity_curve, trade_pnls, total_fees, starting_cash)

    pf = m["profit_factor"]
    return {
        "total_return_pct": float(m["total_return_pct"]),
        "num_trades": m["num_trades"],
        "win_rate_pct": float(m["win_rate_pct"]),
        "profit_factor": float("inf") if pf == Decimal("inf") else float(pf),
        "total_fees": float(m["total_fees"]),
    }


# ============================================================
# Self-test (deterministic, no stored data needed)
# ============================================================


def _selftest() -> int:
    failed = False

    def expect(name, cond):
        nonlocal failed
        print(f"  {'✓' if cond else '✗'} {name}")
        if not cond:
            failed = True

    print("Metrics self-test (hand-computed inputs)...")
    equity = [Decimal("1000"), Decimal("1100"), Decimal("1050"), Decimal("1200")]
    trades = [Decimal("50"), Decimal("-20"), Decimal("30"), Decimal("-10")]
    m = compute_metrics(equity, trades, Decimal("5"), Decimal("1000"))

    expect("total_return == 20%", m["total_return_pct"] == Decimal("20"))
    expect("num_trades == 4", m["num_trades"] == 4)
    expect("win_rate == 50%", m["win_rate_pct"] == Decimal("50"))
    expect("gross_win == 80", m["gross_win"] == Decimal("80"))
    expect("gross_loss == 30", m["gross_loss"] == Decimal("30"))
    expect("profit_factor == 80/30", abs(m["profit_factor"] - (Decimal("80") / Decimal("30"))) < Decimal("0.0001"))
    expect("avg_win == 40", m["avg_win"] == Decimal("40"))
    expect("avg_loss == 15", m["avg_loss"] == Decimal("15"))
    expect("max_drawdown ~ 4.5454%", abs(m["max_drawdown_pct"] - Decimal("4.545454545454545454545454545")) < Decimal("0.001"))

    m2 = compute_metrics([Decimal("1000")], [], Decimal("0"), Decimal("1000"))
    expect("no trades -> 0% return", m2["total_return_pct"] == Decimal("0"))
    expect("no trades -> profit_factor 0", m2["profit_factor"] == Decimal("0"))

    print()
    print("Resample self-test (1-min -> 15-min aggregation)...")
    t0 = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(30):
        rows.append({
            "timestamp": t0 + timedelta(minutes=i),
            "open": 100.0 + i,
            "high": 100.0 + i + 0.5,
            "low": 100.0 + i - 0.5,
            "close": 100.0 + i + 0.25,
            "volume": 100.0,
        })
    df = pl.DataFrame(rows)
    rs = resample_bars(df, 900)
    expect("two 15-min buckets from 30 1-min bars", len(rs) == 2)
    b0 = rs.row(0, named=True)
    expect("bucket0 open == first open (100.0)", abs(b0["open"] - 100.0) < 1e-9)
    expect("bucket0 high == max high (114.5)", abs(b0["high"] - 114.5) < 1e-9)
    expect("bucket0 low == min low (99.5)", abs(b0["low"] - 99.5) < 1e-9)
    expect("bucket0 close == last close (114.25)", abs(b0["close"] - 114.25) < 1e-9)
    expect("bucket0 volume == sum (1500.0)", abs(b0["volume"] - 1500.0) < 1e-9)

    rs_noop = resample_bars(df, 60)
    expect("interval 60 is a no-op (unchanged length)", len(rs_noop) == len(df))

    print()
    print("✗ Some tests failed." if failed else "All tests passed. ✓")
    return 1 if failed else 0


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Backtest the strategy on stored data")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--cash", type=float, default=1000.0)
    parser.add_argument("--selftest", action="store_true", help="Verify metrics + resample math only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.selftest:
        sys.exit(_selftest())

    run_backtest(days=args.days, starting_cash=Decimal(str(args.cash)))