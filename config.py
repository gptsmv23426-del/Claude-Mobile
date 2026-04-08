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
    POLYMARKET_CHAIN_ID: int = int(os.environ.get("POLYMARKET_CHAIN_ID", "137"))  # 137 = Polygon

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
    API_CALL_DELAY_SECONDS: float = float(os.environ.get("API_CALL_DELAY_SECONDS", "2.0"))
    MAX_MARKETS_PER_CYCLE: int = int(os.environ.get("MAX_MARKETS_PER_CYCLE", "5"))

    PREFERRED_CATEGORIES = {"CRYPTO", "MACRO", "TECHNOLOGY", "SCIENCE", "POLITICS"}
    SKIP_CATEGORIES = {"WEATHER", "SPORTS"}

    # API base URLs — single source of truth, imported by market_scanner and backtester
    GAMMA_API_BASE: str = "https://gamma-api.polymarket.com"
    CLOB_API_BASE: str = "https://clob.polymarket.com"

    # Portfolio file paths — single source of truth, imported by executor and risk_manager
    PAPER_PORTFOLIO_FILE: str = "logs/paper_portfolio.json"
    TRADES_LOG_FILE: str = "logs/trades.jsonl"

    # -------------------------------------------------------------------------
    # PLANNED — Phase 2A: Cross-Reference Signals (zero new deps)
    # Add to .env when Phase 2A is implemented:
    #   ENABLE_CROSS_REFERENCE=true
    # -------------------------------------------------------------------------
    ENABLE_CROSS_REFERENCE: bool = os.environ.get("ENABLE_CROSS_REFERENCE", "false").lower() == "true"

    # -------------------------------------------------------------------------
    # PLANNED — Phase 2B: FRED Macro Context (requires: pip install fredapi)
    # Add to .env when Phase 2B is implemented:
    #   FRED_API_KEY=your_key_here   (free at fred.stlouisfed.org)
    #   ENABLE_MACRO_CONTEXT=true
    # -------------------------------------------------------------------------
    FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")
    ENABLE_MACRO_CONTEXT: bool = os.environ.get("ENABLE_MACRO_CONTEXT", "false").lower() == "true"

    # -------------------------------------------------------------------------
    # PLANNED — Phase 2C: Local Polymarket historical data for backtester
    # Clone https://github.com/SII-WANGZJ/Polymarket_data then set:
    #   POLYMARKET_DATA_PATH=/path/to/Polymarket_data
    # Backtester will use local files instead of CLOB API (removes 60-market cap)
    # -------------------------------------------------------------------------
    POLYMARKET_DATA_PATH: str = os.environ.get("POLYMARKET_DATA_PATH", "")

    # -------------------------------------------------------------------------
    # PLANNED — Phase 3: Ensemble Forecasting
    # Runs 3 independent Haiku forecasts per market and averages them.
    # Costs ~3x tokens but reduces single-call variance significantly.
    #   ENABLE_ENSEMBLE_FORECAST=false   (keep off until paper trading proves value)
    # -------------------------------------------------------------------------
    ENABLE_ENSEMBLE_FORECAST: bool = os.environ.get("ENABLE_ENSEMBLE_FORECAST", "false").lower() == "true"

    # -------------------------------------------------------------------------
    # PLANNED — Phase 4: Trade Evaluation & Learning Loop
    #   WEEKLY_EVAL_DAY=monday           (day to run Sonnet strategy review)
    #   LEARNED_THRESHOLDS_PATH=logs/learned_thresholds.json
    # The evaluator writes threshold recommendations here after each Sonnet review.
    # Config will read this file on next startup and apply overrides.
    # -------------------------------------------------------------------------
    WEEKLY_EVAL_DAY: str = os.environ.get("WEEKLY_EVAL_DAY", "monday")
    LEARNED_THRESHOLDS_PATH: str = os.environ.get("LEARNED_THRESHOLDS_PATH", "logs/learned_thresholds.json")

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


        # Range validation for critical trading parameters
        if not (0 < cls.KELLY_FRACTION <= 1.0):
            raise ValueError(f"KELLY_FRACTION must be in (0, 1.0], got {cls.KELLY_FRACTION}")
        if not (0 < cls.MAX_DRAWDOWN_GATE < 1.0):
            raise ValueError(f"MAX_DRAWDOWN_GATE must be in (0, 1.0), got {cls.MAX_DRAWDOWN_GATE}")
        if cls.MAX_PORTFOLIO_EXPOSURE <= 0 or cls.MAX_PORTFOLIO_EXPOSURE > 1.0:
            raise ValueError(f"MAX_PORTFOLIO_EXPOSURE must be in (0, 1.0], got {cls.MAX_PORTFOLIO_EXPOSURE}")
        if cls.MIN_EDGE_THRESHOLD <= 0:
            raise ValueError(f"MIN_EDGE_THRESHOLD must be > 0, got {cls.MIN_EDGE_THRESHOLD}")

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
