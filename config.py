"""
Central configuration module.
All config values are loaded from .env — never import os.environ outside this file.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    # Polymarket
    POLYMARKET_PRIVATE_KEY: str = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    POLYMARKET_FUNDER_ADDRESS: str = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "")
    POLYMARKET_API_KEY: str = os.environ.get("POLYMARKET_API_KEY", "")
    POLYMARKET_API_SECRET: str = os.environ.get("POLYMARKET_API_SECRET", "")
    POLYMARKET_API_PASSPHRASE: str = os.environ.get("POLYMARKET_API_PASSPHRASE", "")

    # Anthropic
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.environ.get("TELEGRAM_CHAT_ID", "")

    # Bot Config
    PAPER_TRADING: bool = os.environ.get("PAPER_TRADING", "true").lower() == "true"
    MAX_POSITION_SIZE_USDC: float = float(os.environ.get("MAX_POSITION_SIZE_USDC", "25"))
    SCAN_INTERVAL_MINUTES: int = int(os.environ.get("SCAN_INTERVAL_MINUTES", "60"))
    MIN_MARKET_VOLUME_USD: float = float(os.environ.get("MIN_MARKET_VOLUME_USD", "1000"))
    MAX_SPREAD: float = float(os.environ.get("MAX_SPREAD", "0.08"))
    MIN_EVIDENCE_QUALITY: float = float(os.environ.get("MIN_EVIDENCE_QUALITY", "0.55"))
    MIN_EDGE_THRESHOLD: float = float(os.environ.get("MIN_EDGE_THRESHOLD", "0.07"))
    MAX_PORTFOLIO_EXPOSURE: float = float(os.environ.get("MAX_PORTFOLIO_EXPOSURE", "0.20"))
    MAX_DRAWDOWN_GATE: float = float(os.environ.get("MAX_DRAWDOWN_GATE", "0.15"))
    KELLY_FRACTION: float = float(os.environ.get("KELLY_FRACTION", "0.25"))
    MIN_BACKTEST_SHARPE: float = float(os.environ.get("MIN_BACKTEST_SHARPE", "0.8"))

    PREFERRED_CATEGORIES = {"CRYPTO", "MACRO", "TECHNOLOGY", "SCIENCE", "POLITICS"}
    SKIP_CATEGORIES = {"WEATHER", "SPORTS"}

    @classmethod
    def validate(cls) -> None:
        """Check all required keys are present. Raise ValueError if any are missing."""
        required = {
            "POLYMARKET_PRIVATE_KEY": cls.POLYMARKET_PRIVATE_KEY,
            "ANTHROPIC_API_KEY": cls.ANTHROPIC_API_KEY,
            "TELEGRAM_BOT_TOKEN": cls.TELEGRAM_BOT_TOKEN,
            "TELEGRAM_CHAT_ID": cls.TELEGRAM_CHAT_ID,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill in the values."
            )

        if cls.PAPER_TRADING:
            logger.info("Running in PAPER TRADING mode — no real money will be spent.")
        else:
            logger.warning("Running in LIVE TRADING mode — real money is at risk!")

        if cls.MAX_POSITION_SIZE_USDC > 500:
            raise ValueError(
                f"MAX_POSITION_SIZE_USDC={cls.MAX_POSITION_SIZE_USDC} exceeds the $500 "
                "hard safety cap. If you genuinely need larger positions, raise the cap "
                "in config.py after deliberate review."
            )
        if cls.MAX_POSITION_SIZE_USDC > 25:
            logger.warning(
                "MAX_POSITION_SIZE_USDC is %.2f USDC — keep it at $25 or less "
                "until backtest proves profitability.",
                cls.MAX_POSITION_SIZE_USDC,
            )
