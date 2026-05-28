"""
Settings module — loads and validates all bot configuration.

This is the single source of truth for configuration. Everywhere else in the
bot imports from here rather than reading config files or env vars directly.

Configuration comes from two places:
1. config/strategy_params.yaml — strategy and risk parameters (committed to git)
2. .env file or environment variables — secrets and per-deployment settings (NOT committed)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


# Load .env file if it exists. In production on the server, env vars come from
# systemd or the shell environment instead.
load_dotenv()


# ============================================================
# Pydantic models — these validate config at load time and give
# helpful errors if something is missing or wrong
# ============================================================


class UniverseConfig(BaseModel):
    """Which markets the bot trades."""

    pairs: list[str]
    quote_currency: str

    @field_validator("pairs")
    @classmethod
    def validate_pairs(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Universe pairs cannot be empty")
        for pair in v:
            if "-" not in pair:
                raise ValueError(
                    f"Pair {pair!r} must be in format BASE-QUOTE (e.g. BTC-USD)"
                )
        return v


class StrategyConfig(BaseModel):
    """Volume-confirmed breakout parameters."""

    name: str
    version: float
    breakout_lookback_bars: int = Field(gt=0)
    volume_multiplier_threshold: float = Field(gt=0)
    volume_lookback_bars: int = Field(gt=0)
    bar_interval_seconds: int = Field(gt=0)


class ExitsConfig(BaseModel):
    """How positions exit."""

    stop_atr_multiplier: float = Field(gt=0)
    atr_lookback_bars: int = Field(gt=0)
    trailing_stop_enabled: bool
    trailing_stop_atr_multiplier: float = Field(gt=0)
    time_exit_enabled: bool
    time_exit_minutes: int | None


class SizingConfig(BaseModel):
    """Position sizing rules."""

    risk_per_trade: float = Field(gt=0, le=0.1)  # Cap at 10% — anything more is reckless
    max_concurrent_positions: int = Field(gt=0, le=20)
    min_position_usd: float = Field(gt=0)


class RiskConfig(BaseModel):
    """Circuit breakers."""

    daily_loss_kill_switch: float = Field(gt=0, le=0.5)
    consecutive_loss_limit: int = Field(gt=0)
    auto_reset_enabled: bool


class ExecutionConfig(BaseModel):
    """How orders are placed."""

    entry_order_type: Literal["market", "limit"]
    exit_order_type: Literal["market", "limit"]
    limit_order_offset_bps: float = Field(ge=0)
    limit_order_timeout_seconds: int = Field(gt=0)
    assumed_slippage_bps: float = Field(ge=0)
    taker_fee_bps: float = Field(ge=0)
    maker_fee_bps: float = Field(ge=0)


class LoggingConfig(BaseModel):
    """What gets logged."""

    log_all_signals: bool
    log_market_data: bool
    log_indicators: bool
    log_orders: bool
    log_fills: bool


class BotConfig(BaseModel):
    """Complete bot configuration. Top-level container."""

    universe: UniverseConfig
    strategy: StrategyConfig
    exits: ExitsConfig
    sizing: SizingConfig
    risk: RiskConfig
    execution: ExecutionConfig
    logging: LoggingConfig


# ============================================================
# Loading functions
# ============================================================


def get_project_root() -> Path:
    """Returns the project root directory regardless of where the bot is run from."""
    return Path(__file__).parent.parent


def load_strategy_config(
    config_path: Path | str | None = None,
) -> BotConfig:
    """
    Load and validate the strategy parameters YAML file.

    Args:
        config_path: Optional explicit path. If None, uses the default location.

    Returns:
        Validated BotConfig object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config values are invalid.
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "strategy_params.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Strategy config file not found at {config_path}. "
            f"Make sure you've copied config/strategy_params.yaml to this location."
        )

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # The "mode" section is informational only — actual mode comes from env
    raw_config.pop("mode", None)

    return BotConfig(**raw_config)


# ============================================================
# Environment-based configuration
# ============================================================


class EnvSettings:
    """
    Configuration that comes from environment variables.

    This includes secrets, paths, and per-deployment settings that
    shouldn't be in the YAML file.
    """

    @property
    def bot_mode(self) -> Literal["backtest", "paper", "live"]:
        mode = os.environ.get("BOT_MODE", "paper").lower()
        if mode not in ("backtest", "paper", "live"):
            raise ValueError(
                f"BOT_MODE must be one of: backtest, paper, live. Got: {mode!r}"
            )
        return mode  # type: ignore[return-value]

    @property
    def account_capital_usd(self) -> float:
        return float(os.environ.get("ACCOUNT_CAPITAL_USD", "10000"))

    @property
    def coinbase_api_key_name(self) -> str:
        value = os.environ.get("COINBASE_API_KEY_NAME")
        if not value:
            raise ValueError(
                "COINBASE_API_KEY_NAME not set. "
                "Copy .env.example to .env and fill in your API credentials."
            )
        return value

    @property
    def coinbase_api_private_key(self) -> str:
        value = os.environ.get("COINBASE_API_PRIVATE_KEY")
        if not value:
            raise ValueError(
                "COINBASE_API_PRIVATE_KEY not set. "
                "Copy .env.example to .env and fill in your API credentials."
            )
        # Handle escaped newlines from .env files
        return value.replace("\\n", "\n")

    @property
    def data_dir(self) -> Path:
        default = get_project_root() / "data"
        return Path(os.environ.get("DATA_DIR", str(default)))

    @property
    def logs_dir(self) -> Path:
        default = get_project_root() / "logs"
        return Path(os.environ.get("LOGS_DIR", str(default)))

    @property
    def redis_host(self) -> str:
        return os.environ.get("REDIS_HOST", "localhost")

    @property
    def redis_port(self) -> int:
        return int(os.environ.get("REDIS_PORT", "6379"))

    @property
    def redis_db(self) -> int:
        return int(os.environ.get("REDIS_DB", "0"))

    @property
    def telegram_bot_token(self) -> str | None:
        value = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        return value if value else None

    @property
    def telegram_chat_id(self) -> str | None:
        value = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        return value if value else None

    @property
    def log_level(self) -> str:
        return os.environ.get("LOG_LEVEL", "INFO").upper()


# Module-level singleton
env = EnvSettings()


# ============================================================
# Convenience loader
# ============================================================


def load_config(strategy_config_path: Path | str | None = None) -> tuple[BotConfig, EnvSettings]:
    """
    One-call helper to load both the YAML config and env settings.

    Most bot modules will use this.

    Returns:
        Tuple of (strategy_config, env_settings).
    """
    config = load_strategy_config(strategy_config_path)
    return config, env


if __name__ == "__main__":
    # Self-test: run this file directly to verify config loads
    try:
        config, env_settings = load_config()
        print("✓ Configuration loaded successfully")
        print(f"  Strategy: {config.strategy.name} v{config.strategy.version}")
        print(f"  Pairs: {', '.join(config.universe.pairs)}")
        print(f"  Risk per trade: {config.sizing.risk_per_trade * 100:.1f}%")
        print(f"  Max concurrent positions: {config.sizing.max_concurrent_positions}")
        print(f"  Daily kill switch: -{config.risk.daily_loss_kill_switch * 100:.1f}%")
        print(f"  Mode (from env): {env_settings.bot_mode}")
    except Exception as e:
        print(f"✗ Configuration failed to load: {e}")
        raise
