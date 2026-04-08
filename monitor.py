"""
Monitor — Telegram alerts via direct HTTP (requests.post to Bot API).

Uses requests.post() directly instead of python-telegram-bot library.
Reason: python-telegram-bot>=20.0 is fully async — calling Bot methods
synchronously returns a coroutine object and never sends. Direct HTTP
is simpler, sync-safe, and has no library version concerns.
"""

import html
import logging
from datetime import datetime
from typing import Optional

import requests

from config import Config

logger = logging.getLogger(__name__)

_TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"


def _send(text: str) -> None:
    """Send a message to the configured Telegram chat. Swallow errors — alerts must never crash the bot."""
    if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured — skipping alert.")
        return
    try:
        url = _TELEGRAM_URL.format(token=Config.TELEGRAM_BOT_TOKEN)
        resp = requests.post(
            url,
            json={"chat_id": Config.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            logger.error("Telegram send failed: %s %s", resp.status_code, resp.text[:200])
        else:
            logger.debug("Telegram alert sent: %s", text[:80])
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)


def alert_trade_entry(
    question: str,
    side: str,
    amount: float,
    edge: float,
    confidence: str,
    critic_concern: Optional[str] = None,
    critic_summary: Optional[str] = None,
) -> None:
    body = (
        f"<b>TRADE ENTRY</b>\n"
        f"Market: {html.escape(question)}\n"
        f"Side: {html.escape(side)}\n"
        f"Size: ${amount:.2f}\n"
        f"Edge: {edge:.4f}\n"
        f"Confidence: {html.escape(confidence)}"
    )
    if critic_concern and critic_concern != "LOW":
        body += f"\nCritic: {html.escape(critic_concern)}"
        if critic_summary:
            body += f" — {html.escape(critic_summary[:120])}"
    _send(body)


def alert_trade_exit(question: str, pnl: float) -> None:
    result = "WIN" if pnl > 0 else "LOSS"
    _send(
        f"<b>TRADE EXIT</b>\n"
        f"Market: {html.escape(question)}\n"
        f"P&amp;L: ${pnl:+.2f}\n"
        f"Result: {result}"
    )


def alert_trade_resolved(
    question: str,
    side: str,
    pnl: float,
    predicted_prob: float,
    actual_outcome: bool,
    brier_contribution: float,
) -> None:
    """Richer resolution alert replacing alert_trade_exit for confirmed outcomes."""
    result = "WIN" if pnl > 0 else "LOSS"
    outcome_str = "YES" if actual_outcome else "NO"
    actual_val = "1.0" if actual_outcome else "0.0"
    _send(
        f"<b>TRADE RESOLVED</b>\n"
        f"Market: {html.escape(question)}\n"
        f"Side: {html.escape(side)} | Outcome: {outcome_str}\n"
        f"Result: {result} ${pnl:+.2f}\n"
        f"Predicted: {predicted_prob:.1%} | Actual: {actual_val}\n"
        f"Brier: {brier_contribution:.3f}"
    )


def alert_calibration_report(
    brier_score: float,
    win_rate: float,
    n_trades: int,
    worst_category: str,
    best_category: str,
    top_lesson: str,
    threshold_changes: dict,
) -> None:
    """Weekly calibration report to Telegram. Called from main._run_weekly_evaluation()."""
    changes_str = ", ".join(
        f"{k}: {v}" for k, v in threshold_changes.items() if v
    ) or "none"
    _send(
        f"<b>WEEKLY CALIBRATION REPORT</b>\n"
        f"Trades evaluated: {n_trades}\n"
        f"Brier Score: {brier_score:.3f} (target &lt;0.20)\n"
        f"Win Rate: {win_rate:.1%}\n"
        f"Best category: {html.escape(best_category)}\n"
        f"Worst category: {html.escape(worst_category)}\n"
        f"Sonnet lesson: {html.escape(top_lesson[:200])}\n"
        f"Threshold changes: {html.escape(changes_str)}"
    )


def alert_daily_summary(n_trades: int, wins: int, pnl: float, balance: float) -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")
    _send(
        f"<b>DAILY SUMMARY</b>\n"
        f"Date: {date_str}\n"
        f"Trades: {n_trades}\n"
        f"Wins: {wins}\n"
        f"P&amp;L: ${pnl:+.2f}\n"
        f"Balance: ${balance:.2f}"
    )


def alert_trade_blocked(question: str, reason: str) -> None:
    _send(
        f"<b>TRADE BLOCKED</b>\n"
        f"Market: {html.escape(question)}\n"
        f"Reason: {html.escape(reason)}"
    )


def alert_error(error_type: str, message: str) -> None:
    _send(
        f"<b>ERROR</b>\n"
        f"{html.escape(error_type)}: {html.escape(message)}"
    )


def alert_drawdown_gate(pct: float) -> None:
    _send(
        f"<b>TRADING PAUSED</b>\n"
        f"Drawdown exceeded {pct:.1%}\n"
        f"Manual review required"
    )


def alert_startup(paper_trading: bool, balance: float) -> None:
    mode = "PAPER" if paper_trading else "LIVE"
    _send(
        f"<b>BOT STARTED</b>\n"
        f"Mode: {mode} TRADING\n"
        f"Balance: ${balance:.2f}\n"
        f"Scan interval: {Config.SCAN_INTERVAL_MINUTES} min"
    )


def alert_heartbeat_missed(minutes_ago: int) -> None:
    """Sent when the bot's main loop is alive but no trading cycle has completed recently."""
    _send(
        f"<b>HEARTBEAT MISSED</b>\n"
        f"Last completed cycle was {minutes_ago} min ago.\n"
        f"Bot process is alive but cycle may be stalled.\n"
        f"Check terminal logs."
    )


def send_test_message() -> None:
    """Send a test message to verify Telegram integration is working."""
    _send("<b>TEST MESSAGE</b>\nPolymarket bot Telegram integration is working.")
    logger.info("Test message sent.")
