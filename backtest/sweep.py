"""
Parameter sweep — runs the strategy across a grid of parameter combinations,
on TWO separate time windows, and writes a full-metrics table.

Why two windows: a combination that looks good on one 30-day window may just be
fitting that window's noise. Running every combination on a second, non-overlapping
window is the cross-check — a result that holds up on BOTH is worth a second look;
one that's great on window A and falls apart on window B is an overfit mirage, and
the table puts them side by side so you can see which is which.

There is NO single "winner" ranking by design. With 108 combinations, ranking by
one number would crown whichever combo best fit the noise. The table shows every
metric for both windows; you judge which combos are robust.

The sweep overrides config IN MEMORY per combination (model_copy on the validated
BotConfig), so nothing is written to disk and the real strategy_params.yaml is
never touched. Each combo gets a clean, independent config copy.

Usage (on the server, after collecting ~60 days of data):
    python3 -m backtest.sweep
    python3 -m backtest.sweep --out results/sweep.txt
    nohup python3 -m backtest.sweep --out results/sweep.txt > sweep.log 2>&1 &   # survives logout

Grid (108 combinations):
    timeframe:      15-min / 30-min / 60-min
    volume mult:    2x / 3x / 4x
    EMA pair:       20/50, 10/30, 50/100
    RSI filter:     off / on  (on = block entry if RSI > 70)
    MACD filter:    off / on  (on = require MACD line > signal line)
"""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from config.settings import BotConfig, load_config
from backtest.backtest import run_backtest_metrics


logger = logging.getLogger(__name__)


# ============================================================
# The grid
# ============================================================

TIMEFRAMES = [900, 1800, 3600]          # seconds: 15-min, 30-min, 60-min
VOLUME_MULTS = [2.0, 3.0, 4.0]
EMA_PAIRS = [(20, 50), (10, 30), (50, 100)]
RSI_FILTER = [False, True]
MACD_FILTER = [False, True]


@dataclass(frozen=True)
class Combo:
    bar_interval_seconds: int
    volume_multiplier_threshold: float
    ema_short_period: int
    ema_long_period: int
    rsi_filter_enabled: bool
    macd_filter_enabled: bool

    def label(self) -> str:
        tf = f"{self.bar_interval_seconds // 60}m"
        rsi = "RSI" if self.rsi_filter_enabled else "---"
        macd = "MACD" if self.macd_filter_enabled else "----"
        return (f"{tf:>4} v{self.volume_multiplier_threshold:>3.0f} "
                f"e{self.ema_short_period}/{self.ema_long_period:<3} {rsi} {macd}")


def generate_grid() -> list[Combo]:
    return [
        Combo(tf, vol, es, el, rsi, macd)
        for tf, vol, (es, el), rsi, macd in itertools.product(
            TIMEFRAMES, VOLUME_MULTS, EMA_PAIRS, RSI_FILTER, MACD_FILTER
        )
    ]


def config_for(base: BotConfig, combo: Combo) -> BotConfig:
    """Return a fresh BotConfig with this combo's strategy params applied."""
    new_strategy = base.strategy.model_copy(update={
        "bar_interval_seconds": combo.bar_interval_seconds,
        "volume_multiplier_threshold": combo.volume_multiplier_threshold,
        "ema_short_period": combo.ema_short_period,
        "ema_long_period": combo.ema_long_period,
        "rsi_filter_enabled": combo.rsi_filter_enabled,
        "macd_filter_enabled": combo.macd_filter_enabled,
    })
    return base.model_copy(update={"strategy": new_strategy})


# ============================================================
# Two-window definitions
# ============================================================

@dataclass(frozen=True)
class Window:
    name: str
    start: datetime
    end: datetime


def default_windows(total_days: int = 60) -> tuple[Window, Window]:
    """
    Split the available history into two non-overlapping 30-day windows.

    Window B is the most recent 30 days; window A is the 30 days before that.
    Requires ~`total_days` of collected data (default 60).
    """
    now = datetime.now(timezone.utc)
    half = total_days // 2
    b_start = now - timedelta(days=half)
    a_start = now - timedelta(days=total_days)
    return (
        Window("A", a_start, b_start),
        Window("B", b_start, now),
    )


# ============================================================
# Run
# ============================================================

def run_sweep(out_path: str | None, total_days: int, starting_cash: Decimal) -> None:
    base_config, _env = load_config()
    grid = generate_grid()
    win_a, win_b = default_windows(total_days)

    print(f"Sweep: {len(grid)} combinations x 2 windows = {len(grid) * 2} backtests.")
    print(f"  Window A: {win_a.start.date()} .. {win_a.end.date()}")
    print(f"  Window B: {win_b.start.date()} .. {win_b.end.date()}")
    print("  (A result that holds up on BOTH windows is worth attention;")
    print("   one strong only on a single window is likely overfit.)")
    print()

    rows: list[dict] = []
    for i, combo in enumerate(grid, start=1):
        cfg = config_for(base_config, combo)
        m_a = run_backtest_metrics(cfg, win_a.start, win_a.end, starting_cash)
        m_b = run_backtest_metrics(cfg, win_b.start, win_b.end, starting_cash)
        rows.append({"combo": combo, "A": m_a, "B": m_b})
        if i % 10 == 0 or i == len(grid):
            print(f"  ... {i}/{len(grid)} combinations done")

    table = format_table(rows)
    print()
    print(table)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(table + "\n")
        print(f"\nWritten to {out_path}")


def _fmt_metric(m: dict, key: str) -> str:
    """Safe pull from a metrics dict that may have failed (None)."""
    if m is None:
        return "  ERR"
    return m.get(key)


def format_table(rows: list[dict]) -> str:
    header = (
        f"{'combination':<26} | "
        f"{'A_ret':>7} {'A_trd':>5} {'A_win':>6} {'A_pf':>5} {'A_fee':>7} | "
        f"{'B_ret':>7} {'B_trd':>5} {'B_win':>6} {'B_pf':>5} {'B_fee':>7}"
    )
    lines = [
        "Full sweep results — no single ranking; compare A vs B for robustness.",
        "A combo strong on A but weak on B is overfit to A's noise.",
        "",
        header,
        "-" * len(header),
    ]
    for r in rows:
        c = r["combo"]
        a, b = r["A"], r["B"]

        def cells(m: dict) -> str:
            if m is None:
                return f"{'ERR':>7} {'':>5} {'':>6} {'':>5} {'':>7}"
            pf = m["profit_factor"]
            pf_s = "inf" if pf == float("inf") else f"{pf:>5.2f}"
            return (
                f"{m['total_return_pct']:>6.2f}% {m['num_trades']:>5} "
                f"{m['win_rate_pct']:>5.1f}% {pf_s:>5} ${m['total_fees']:>6.2f}"
            )

        lines.append(f"{c.label():<26} | {cells(a)} | {cells(b)}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter sweep across two windows")
    parser.add_argument("--out", type=str, default=None, help="Write results table to this file")
    parser.add_argument("--days", type=int, default=60, help="Total days of data to split into two windows")
    parser.add_argument("--cash", type=float, default=1000.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    run_sweep(args.out, args.days, Decimal(str(args.cash)))