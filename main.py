"""
Polymarket Autonomous Trading Bot
Entry point and main loop.
"""

import logging
import os
import time
import traceback
from datetime import datetime

import schedule

from config import Config
from market_scanner import scan_markets
from researcher import research_markets
from forecaster import forecast_markets
from risk_manager import evaluate_trade, check_drawdown_gate
from executor import execute_trade, monitor_open_positions, get_portfolio_summary, IS_PAPER_TRADING
from monitor import (
    alert_startup,
    alert_trade_entry,
    alert_trade_blocked,
    alert_trade_exit,
    alert_daily_summary,
    alert_error,
    alert_drawdown_gate,
)
from backtester import run_backtest, backtest_already_run
from calibration_tracker import log_forecast, check_and_update_resolutions


def _configure_logging() -> None:
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/bot_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )


def _send_daily_summary() -> None:
    summary = get_portfolio_summary()
    alert_daily_summary(
        n_trades=summary["total_trades"],
        wins=summary["wins"],
        pnl=summary["total_pnl"],
        balance=summary["balance"],
    )


def _run_weekly_evaluation() -> None:
    """
    Run the weekly Sonnet strategy review and send calibration report to Telegram.
    Scheduled to run on Config.WEEKLY_EVAL_DAY at 14:00 UTC.

    PLANNED (Phase 4) — the evaluator functions called here raise NotImplementedError
    until Phase 4 is implemented. This function is intentionally a no-op until then.
    """
    logger = logging.getLogger("main.weekly_eval")
    logger.info("Weekly evaluation triggered.")
    try:
        # PLANNED (Phase 4): uncomment when evaluator is implemented
        # from evaluator import run_weekly_review
        # from monitor import alert_calibration_report
        # learned = run_weekly_review()
        # if learned:
        #     alert_calibration_report(
        #         brier_score=...,   # from CalibrationReport
        #         win_rate=...,
        #         n_trades=...,
        #         worst_category=learned ... ,
        #         best_category=...,
        #         top_lesson=learned.sonnet_rationale,
        #         threshold_changes={
        #             "category_min_edge": learned.category_min_edge,
        #             "category_skip": learned.category_skip,
        #             "ensemble_recommended": learned.ensemble_recommended,
        #         },
        #     )
        logger.info("Weekly evaluation is planned but not yet implemented (Phase 4).")
    except Exception as exc:
        logger.error("Weekly evaluation failed (non-fatal): %s", exc)


def _run_trading_cycle() -> None:
    logger = logging.getLogger("main.cycle")

    # Check drawdown gate before starting cycle
    paused, drawdown = check_drawdown_gate()
    if paused:
        logger.warning("Trading paused — drawdown gate active (%.1f%%). Skipping cycle.", drawdown * 100)
        alert_drawdown_gate(drawdown)
        return

    # Poll for resolutions from previous cycles before scanning new ones.
    # This keeps the calibration log up to date without a separate process.
    try:
        n_resolved = check_and_update_resolutions()
        if n_resolved:
            logger.info("Calibration: %d market(s) resolved since last cycle.", n_resolved)
    except Exception as exc:
        logger.warning("Calibration resolution check failed (non-fatal): %s", exc)

    logger.info("=== Starting trading cycle ===")

    # Step 1: Scan markets
    opportunities = scan_markets()
    if not opportunities:
        logger.info("No qualifying markets found this cycle.")
        return

    # Step 2: Research
    research_results = research_markets(opportunities)
    if not research_results:
        logger.info("No markets passed research quality threshold.")
        return

    # Step 3: Forecast
    forecasts = forecast_markets(research_results)
    if not forecasts:
        logger.info("No forecasts produced.")
        return

    # Step 4: Risk check and execute each forecast
    for forecast in forecasts:
        decision = evaluate_trade(forecast)

        if not decision.approved:
            alert_trade_blocked(forecast.question, decision.blocked_reason or "unknown")
            continue

        # Execute the trade
        trade_record = execute_trade(decision)
        if trade_record:
            try:
                log_forecast(forecast)
            except Exception as exc:
                logger.warning("Failed to log forecast to calibration tracker: %s", exc)
            alert_trade_entry(
                question=forecast.question,
                side=forecast.side,
                amount=decision.position_size_usdc,
                edge=forecast.edge,
                confidence=forecast.confidence,
            )

    # Step 5: Monitor open positions for exits
    closed_positions = monitor_open_positions()
    for pos in closed_positions:
        alert_trade_exit(question=pos["question"], pnl=pos["pnl"])

    logger.info("=== Trading cycle complete ===")


def main() -> None:
    _configure_logging()
    logger = logging.getLogger("main")

    logger.info("Polymarket Autonomous Trading Bot starting up...")

    # Step 1: Validate config — fail fast if keys missing
    try:
        Config.validate()
    except ValueError as exc:
        logger.critical("Config validation failed: %s", exc)
        raise SystemExit(1)

    # Step 2: Run backtest on first launch
    if not backtest_already_run():
        logger.info("First launch — running backtest...")
        try:
            run_backtest()
        except Exception as exc:
            logger.warning("Backtest failed (non-fatal): %s", exc)
    else:
        logger.info("Backtest already completed — skipping.")

    # Step 3: Send startup alert
    summary = get_portfolio_summary()
    alert_startup(paper_trading=IS_PAPER_TRADING, balance=summary["balance"])

    # Step 4: Schedule daily summary.
    # DAILY_SUMMARY_TIME_UTC is read from .env (default "14:00" ≈ 8 AM CT).
    # The `schedule` library uses the server's local clock, so deploy in UTC
    # or set DAILY_SUMMARY_TIME_UTC to match your server timezone offset.
    summary_time = os.environ.get("DAILY_SUMMARY_TIME_UTC", "14:00")
    schedule.every().day.at(summary_time).do(_send_daily_summary)

    # Step 4b: Schedule weekly Sonnet strategy review (Phase 4 — no-op until implemented)
    # Runs on Config.WEEKLY_EVAL_DAY at 14:00 UTC, same window as daily summary.
    _weekly_schedule = getattr(schedule.every(), Config.WEEKLY_EVAL_DAY, schedule.every().monday)
    _weekly_schedule.at("14:00").do(_run_weekly_evaluation)

    logger.info(
        "Bot running | Mode: %s | Scan interval: %d min",
        "PAPER" if IS_PAPER_TRADING else "LIVE",
        Config.SCAN_INTERVAL_MINUTES,
    )

    # Step 5: Main loop
    _run_trading_cycle()  # Run once immediately on startup

    next_cycle_time = time.time() + Config.SCAN_INTERVAL_MINUTES * 60

    while True:
        try:
            # Tick the scheduler every 30 seconds so scheduled jobs fire on time
            # regardless of how long the trading cycle took.
            schedule.run_pending()
            time.sleep(30)

            if time.time() >= next_cycle_time:
                _run_trading_cycle()
                next_cycle_time = time.time() + Config.SCAN_INTERVAL_MINUTES * 60

        except KeyboardInterrupt:
            logger.info("Shutdown requested via keyboard interrupt.")
            break
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Unhandled exception in main loop:\n%s", tb)
            try:
                alert_error(type(exc).__name__, str(exc)[:200])
            except Exception:
                pass  # Don't let Telegram failure cascade
            logger.info("Sleeping 5 minutes before retry...")
            time.sleep(300)
            next_cycle_time = time.time() + Config.SCAN_INTERVAL_MINUTES * 60


if __name__ == "__main__":
    main()
