"""
Monitor — Telegram alerts using python-telegram-bot (sync version).

PLANNED UPGRADES (do not implement until approved):

Phase 4 — Weekly Calibration Report Alert
  Add alert_calibration_report() function:
    def alert_calibration_report(
        brier_score: float,
        win_rate: float,
        n_trades: int,
        worst_category: str,
        best_category: str,
        top_lesson: str,
        threshold_changes: dict,
    ) -> None:
    Message format:
      <b>WEEKLY CALIBRATION REPORT</b>
      Trades evaluated: {n_trades}
      Brier Score: {brier_score:.3f} (target <0.20)
      Win Rate: {win_rate:.1%}
      Best category: {best_category}
      Worst category: {worst_category}
      Sonnet lesson: {top_lesson}
      Threshold changes: {threshold_changes}

  Called from main.py on Config.WEEKLY_EVAL_DAY at the same 14:00 UTC schedule slot.

Phase 4 — Outcome Resolution Alert
  Add alert_trade_resolved() to replace alert_trade_exit() with richer data:
    def alert_trade_resolved(
        question: str,
        side: str,
        pnl: float,
        predicted_prob: float,
        actual_outcome: bool,
        brier_contribution: float,
    ) -> None:
    Message format:
      <b>TRADE RESOLVED</b>
      Market: {question}
      Side: {side} | Outcome: {"YES" if actual_outcome else "NO"}
      Result: {"WIN" if pnl > 0 else "LOSS"} ${pnl:+.2f}
      Predicted: {predicted_prob:.1%} | Actual: {"1.0" if actual_outcome else "0.0"}
      Brier: {brier_contribution:.3f}
"""

import logging
from datetime import datetime

import telegram

from config import Config

logger = logging.getLogger(__name__)

_bot: telegram.Bot | None = None


def _get_bot() -> telegram.Bot:
    global _bot
    if _bot is None:
        _bot = telegram.Bot(token=Config.TELEGRAM_BOT_TOKEN)
    return _bot


def _send(text: str) -> None:
    """Send a message to the configured Telegram chat. Swallow errors — alerts must never crash the bot."""
    try:
        bot = _get_bot()
        bot.send_message(
            chat_id=Config.TELEGRAM_CHAT_ID,
            text=text,
            parse_mode="HTML",
        )
        logger.debug("Telegram alert sent: %s", text[:80])
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)


def alert_trade_entry(question: str, side: str, amount: float, edge: float, confidence: str) -> None:
    _send(
        f"<b>TRADE ENTRY</b>\n"
        f"Market: {question}\n"
        f"Side: {side}\n"
        f"Size: ${amount:.2f}\n"
        f"Edge: {edge:.4f}\n"
        f"Confidence: {confidence}"
    )


def alert_trade_exit(question: str, pnl: float) -> None:
    result = "WIN" if pnl > 0 else "LOSS"
    _send(
        f"<b>TRADE EXIT</b>\n"
        f"Market: {question}\n"
        f"P&L: ${pnl:+.2f}\n"
        f"Result: {result}"
    )


def alert_daily_summary(n_trades: int, wins: int, pnl: float, balance: float) -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")
    _send(
        f"<b>DAILY SUMMARY</b>\n"
        f"Date: {date_str}\n"
        f"Trades: {n_trades}\n"
        f"Wins: {wins}\n"
        f"P&L: ${pnl:+.2f}\n"
        f"Balance: ${balance:.2f}"
    )


def alert_trade_blocked(question: str, reason: str) -> None:
    _send(
        f"<b>TRADE BLOCKED</b>\n"
        f"Market: {question}\n"
        f"Reason: {reason}"
    )


def alert_error(error_type: str, message: str) -> None:
    _send(
        f"<b>ERROR</b>\n"
        f"{error_type}: {message}"
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


def send_test_message() -> None:
    """Send a test message to verify Telegram integration is working."""
    _send("<b>TEST MESSAGE</b>\nPolymarket bot Telegram integration is working.")
    logger.info("Test message sent.")
