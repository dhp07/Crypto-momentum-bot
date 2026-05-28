# Crypto Momentum Bot

A volume-confirmed breakout trading bot for crypto markets on Coinbase Advanced Trade.

## What this bot does

This bot trades a single strategy: **volume-confirmed breakouts on a 30-minute lookback window**. When the price of a tracked pair exceeds its 30-minute high *and* the current bar's volume is at least 2x the recent average, the bot opens a long position. Positions exit via a trailing stop based on Average True Range (ATR).

## Configuration summary

- **Pairs traded:** BTC-USD, ETH-USD, SOL-USD, AVAX-USD, LINK-USD
- **Breakout lookback:** 30 minutes
- **Volume threshold:** 2.0x the 30-minute average
- **Stop loss:** 1.5x ATR(14) on 1-minute bars
- **Exit:** Trailing stop only (no time-based exit)
- **Risk per trade:** 3.0% of capital
- **Max concurrent positions:** 3
- **Daily loss kill switch:** -9% of capital
- **Consecutive loss pause:** 5 losing trades in a row

## Project structure

```
crypto-momentum-bot/
├── config/          # Configuration files (params, secrets templates)
├── src/             # Bot source code
│   ├── data/        # Market data feeds and storage
│   ├── strategy/    # Trading strategy logic
│   ├── risk/        # Risk management and safety limits
│   ├── execution/   # Order placement (paper and live)
│   ├── portfolio/   # Position tracking and P&L
│   └── monitoring/  # Logging, alerting, metrics
├── backtest/        # Historical backtesting engine
├── tests/           # Unit and integration tests
├── scripts/         # Standalone utility scripts
├── data/            # Tick data, bars, signals, trades (gitignored)
└── logs/            # Runtime logs (gitignored)
```

## Build phases

The bot is built in phases. Do not skip phases or move ahead before the current one is stable.

1. **Foundation** — Server setup, Coinbase API authentication
2. **Data layer** — Live websocket feed, bar construction, indicator computation
3. **Storage** — Persistent data storage in Parquet
4. **Strategy** — Volume-confirmed breakout signal generation
5. **Risk** — Risk gate, position sizing, kill switch
6. **Execution** — Paper executor (simulated fills against live data)
7. **Monitoring** — Logging, alerting, metrics
8. **Backtest** — Historical strategy validation
9. **Paper trade** — Minimum 30 days against live market with no real capital
10. **Live trading** — Start with $500-1,000, scale gradually

## Running the bot

Setup instructions are in `scripts/setup_environment.sh`.

Once set up:

```bash
# Backtest
python scripts/run_backtest.py

# Paper trade (live data, simulated fills)
python scripts/run_paper.py

# Live trade (real money — has confirmation prompt)
python scripts/run_live.py
```

## Safety notes

- The bot has a daily kill switch at -9% of capital. When triggered, the bot stops and requires manual restart.
- The bot pauses after 5 consecutive losing trades. Manual restart required.
- API keys are stored in environment variables, never committed to the repo.
- Trade and transfer permissions are off by default. They must be explicitly enabled in configuration.

## Disclaimer

This is a learning project. Algorithmic trading involves substantial risk of loss. Past performance in backtesting is not indicative of future results. Do not deploy capital you cannot afford to lose.
