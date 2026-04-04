"""
Risk manager — runs 10 checks before approving any trade.
All checks must pass or the trade is blocked.
"""

import json
import logging
import os
from typing import Optional

from pydantic import BaseModel

from config import Config
from forecaster import ForecastResult

logger = logging.getLogger(__name__)

PAPER_PORTFOLIO_FILE = "logs/paper_portfolio.json"


class RiskDecision(BaseModel):
    approved: bool
    position_size_usdc: float
    blocked_reason: Optional[str] = None
    forecast: Optional[ForecastResult] = None


def _load_portfolio() -> dict:
    if os.path.exists(PAPER_PORTFOLIO_FILE):
        try:
            with open(PAPER_PORTFOLIO_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "balance": 1000.0,
        "peak_balance": 1000.0,
        "open_positions": {},
        "trade_history": [],
    }


def _kelly_position_size(edge: float, entry_price: float, portfolio_balance: float) -> float:
    """
    Fractional Kelly sizing for a binary prediction market.

    Standard Kelly: f* = edge / (1 - entry_price)
    where edge = forecast_probability - market_price (already computed by forecaster).

    Using the entry_price (what you pay per share) in the denominator is correct.
    Using the opposite side's price was wrong: it diverges from entry_price whenever
    the spread is non-zero, systematically over-sizing YES bets and under-sizing NO bets.
    """
    denominator = 1.0 - entry_price
    if denominator <= 0:
        return 0.0
    kelly = edge / denominator
    size = portfolio_balance * kelly * Config.KELLY_FRACTION
    return min(size, Config.MAX_POSITION_SIZE_USDC)


def evaluate_trade(forecast: ForecastResult) -> RiskDecision:
    """
    Run all 10 risk checks. Return RiskDecision indicating approval or block reason.
    """
    portfolio = _load_portfolio()
    balance = portfolio.get("balance", 1000.0)
    peak_balance = portfolio.get("peak_balance", balance)
    open_positions: dict = portfolio.get("open_positions", {})

    def block(reason: str) -> RiskDecision:
        logger.info("TRADE BLOCKED — %s | Market: %s", reason, forecast.question[:50])
        return RiskDecision(
            approved=False,
            position_size_usdc=0.0,
            blocked_reason=reason,
            forecast=forecast,
        )

    # Check 1: Confidence must be MEDIUM or HIGH
    if forecast.confidence == "LOW":
        return block("Confidence is LOW")
    logger.debug("Check 1 PASS: confidence=%s", forecast.confidence)

    # Check 2: Edge must be >= MIN_EDGE_THRESHOLD
    if forecast.edge < Config.MIN_EDGE_THRESHOLD:
        return block(f"Edge {forecast.edge:.4f} < threshold {Config.MIN_EDGE_THRESHOLD}")
    logger.debug("Check 2 PASS: edge=%.4f", forecast.edge)

    # Check 3: Evidence quality must be >= MIN_EVIDENCE_QUALITY
    if forecast.evidence_quality < Config.MIN_EVIDENCE_QUALITY:
        return block(f"Evidence quality {forecast.evidence_quality:.2f} < {Config.MIN_EVIDENCE_QUALITY}")
    logger.debug("Check 3 PASS: evidence_quality=%.2f", forecast.evidence_quality)

    # Check 4: Position size must be <= MAX_POSITION_SIZE_USDC
    entry_price = forecast.yes_price if forecast.side == "YES" else forecast.no_price
    raw_size = _kelly_position_size(forecast.edge, entry_price, balance)
    if raw_size <= 0:
        return block("Kelly sizing produced zero position size")
    logger.debug("Check 4 PASS: kelly_size=%.2f USDC", raw_size)

    # Check 5: Single market exposure <= MAX_PORTFOLIO_EXPOSURE * balance
    max_exposure = balance * Config.MAX_PORTFOLIO_EXPOSURE
    if raw_size > max_exposure:
        raw_size = max_exposure
    logger.debug("Check 5 PASS: capped_size=%.2f USDC (max_exposure=%.2f)", raw_size, max_exposure)

    # Check 6: Drawdown gate — if down > MAX_DRAWDOWN_GATE from peak, pause all trading
    if peak_balance > 0:
        drawdown = (peak_balance - balance) / peak_balance
        if drawdown > Config.MAX_DRAWDOWN_GATE:
            return block(
                f"Portfolio drawdown {drawdown:.1%} exceeds gate {Config.MAX_DRAWDOWN_GATE:.1%} — trading paused"
            )
    logger.debug("Check 6 PASS: drawdown within limits")

    # Check 7: Days to expiry must be >= 2
    if forecast.days_to_expiry < 2:
        return block(f"Days to expiry {forecast.days_to_expiry:.1f} < 2")
    logger.debug("Check 7 PASS: days_to_expiry=%.1f", forecast.days_to_expiry)

    # Check 8: Market volume must be > MIN_MARKET_VOLUME_USD
    if forecast.volume_usd < Config.MIN_MARKET_VOLUME_USD:
        return block(f"Volume ${forecast.volume_usd:,.0f} < ${Config.MIN_MARKET_VOLUME_USD:,.0f}")
    logger.debug("Check 8 PASS: volume=$%.0f", forecast.volume_usd)

    # Check 9: Spread must be < MAX_SPREAD
    if forecast.spread > Config.MAX_SPREAD:
        return block(f"Spread {forecast.spread:.4f} > max {Config.MAX_SPREAD}")
    logger.debug("Check 9 PASS: spread=%.4f", forecast.spread)

    # Check 10: No open position already in this market
    if forecast.market_id in open_positions:
        return block(f"Already have open position in market {forecast.market_id}")
    logger.debug("Check 10 PASS: no duplicate position")

    logger.info(
        "ALL CHECKS PASSED — Approving trade: %s | Side: %s | Size: $%.2f",
        forecast.question[:50],
        forecast.side,
        raw_size,
    )
    return RiskDecision(
        approved=True,
        position_size_usdc=round(raw_size, 2),
        blocked_reason=None,
        forecast=forecast,
    )


def check_drawdown_gate() -> tuple[bool, float]:
    """
    Returns (is_paused, drawdown_pct).
    Call this before starting a trading session.
    """
    portfolio = _load_portfolio()
    balance = portfolio.get("balance", 1000.0)
    peak = portfolio.get("peak_balance", balance)
    if peak <= 0:
        return False, 0.0
    drawdown = (peak - balance) / peak
    return drawdown > Config.MAX_DRAWDOWN_GATE, drawdown
