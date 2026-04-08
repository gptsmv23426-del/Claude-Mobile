"""
Telegram command handler — polls for incoming messages and responds.

Supports:
  /status   — balance, open positions count, last trade time, bot uptime
  /portfolio — detailed open positions
  /pause    — pause trading (skip cycles until /resume)
  /resume   — resume trading

Designed to be called from main.py's existing 30s sleep loop.
No threads, no async — sync HTTP polling via getUpdates.
"""

import html
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from config import Config
from executor import get_portfolio_summary, IS_PAPER_TRADING

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.telegram.org/bot{token}"
_last_update_id = 0
_paused = False


def is_paused() -> bool:
    """Check if trading is paused via Telegram command."""
    return _paused


def poll_and_handle() -> None:
    """
    Poll Telegram for new messages, handle any commands.
    Call this every ~30s from the main loop. Non-blocking (timeout=2s).
    Swallows all errors — must never crash the bot.
    """
    global _last_update_id

    if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
        return

    try:
        url = _BASE_URL.format(token=Config.TELEGRAM_BOT_TOKEN) + "/getUpdates"
        resp = requests.get(
            url,
            params={"offset": _last_update_id + 1, "timeout": 2, "limit": 10},
            timeout=5,
        )
        if not resp.ok:
            return

        data = resp.json()
        if not data.get("ok"):
            return

        for update in data.get("result", []):
            _last_update_id = update["update_id"]
            msg = update.get("message", {})
            chat_id = str(msg.get("chat", {}).get("id", ""))
            text = msg.get("text", "").strip()

            # Only respond to configured chat
            if chat_id != str(Config.TELEGRAM_CHAT_ID):
                continue

            if text.startswith("/status"):
                _handle_status(chat_id)
            elif text.startswith("/portfolio"):
                _handle_portfolio(chat_id)
            elif text.startswith("/pause"):
                _handle_pause(chat_id)
            elif text.startswith("/resume"):
                _handle_resume(chat_id)

    except Exception as exc:
        logger.debug("Telegram poll failed (non-fatal): %s", exc)


def _reply(chat_id: str, text: str) -> None:
    """Send a reply to a specific chat."""
    try:
        url = _BASE_URL.format(token=Config.TELEGRAM_BOT_TOKEN) + "/sendMessage"
        requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as exc:
        logger.debug("Telegram reply failed: %s", exc)


def _handle_status(chat_id: str) -> None:
    """Respond to /status with bot overview."""
    summary = get_portfolio_summary()
    mode = "PAPER" if IS_PAPER_TRADING else "LIVE"

    # Last trade time from trades.jsonl
    last_trade = "never"
    try:
        with open("logs/trades.jsonl") as f:
            lines = f.readlines()
        if lines:
            last = json.loads(lines[-1])
            last_trade = last.get("timestamp", "unknown")[:19]
    except Exception:
        pass

    paused_str = " (PAUSED)" if _paused else ""

    _reply(chat_id, (
        f"<b>BOT STATUS{paused_str}</b>\n"
        f"Mode: {mode}\n"
        f"Balance: ${summary['balance']:.2f}\n"
        f"Peak: ${summary['peak_balance']:.2f}\n"
        f"Open positions: {summary['open_positions']}\n"
        f"Total trades: {summary['total_trades']}\n"
        f"Win/Loss: {summary['wins']}/{summary['losses']}\n"
        f"Total P&amp;L: ${summary['total_pnl']:+.2f}\n"
        f"Last trade: {html.escape(last_trade)}"
    ))


def _handle_portfolio(chat_id: str) -> None:
    """Respond to /portfolio with open position details."""
    try:
        with open(Config.PAPER_PORTFOLIO_FILE) as f:
            portfolio = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _reply(chat_id, "No portfolio data found.")
        return

    positions = portfolio.get("open_positions", {})
    if not positions:
        _reply(chat_id, (
            f"<b>PORTFOLIO</b>\n"
            f"No open positions.\n"
            f"Balance: ${portfolio.get('balance', 0):.2f}"
        ))
        return

    lines = [f"<b>PORTFOLIO — {len(positions)} open</b>\n"]
    for pos in positions.values():
        entry = pos.get("entry_time", "")[:10]
        lines.append(
            f"• <b>{html.escape(pos.get('question', '?')[:50])}</b>\n"
            f"  Side: {pos.get('side', '?')} | Size: ${pos.get('size_usdc', 0):.2f}\n"
            f"  Edge: {pos.get('edge', 0):.3f} | Conf: {pos.get('confidence', '?')}\n"
            f"  Entry: {entry}"
        )
    lines.append(f"\nBalance: ${portfolio.get('balance', 0):.2f}")
    _reply(chat_id, "\n".join(lines))


def _handle_pause(chat_id: str) -> None:
    """Pause trading cycles."""
    global _paused
    _paused = True
    logger.info("Trading PAUSED via Telegram command.")
    _reply(chat_id, "<b>TRADING PAUSED</b>\nBot will skip trading cycles until /resume.\nMonitoring continues.")


def _handle_resume(chat_id: str) -> None:
    """Resume trading cycles."""
    global _paused
    _paused = False
    logger.info("Trading RESUMED via Telegram command.")
    _reply(chat_id, "<b>TRADING RESUMED</b>\nBot will execute trading cycles normally.")
