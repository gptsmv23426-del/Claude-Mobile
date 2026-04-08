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
from critic import challenge_forecast
from monitor import (
    alert_startup,
    alert_trade_entry,
    alert_trade_blocked,
    alert_trade_exit,
    alert_trade_resolved,
    alert_daily_summary,
    alert_error,
    alert_drawdown_gate,
    alert_heartbeat_missed,
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
    Requires >= 10 resolved trades in evaluation_log.jsonl to produce a report.
    """
    logger = logging.getLogger("main.weekly_eval")
    logger.info("Weekly evaluation triggered.")
    try:
        from evaluator import run_weekly_review, get_calibration_summary
        from monitor import alert_calibration_report

        report = get_calibration_summary()
        learned = run_weekly_review()

        if learned and report:
            alert_calibration_report(
                brier_score=report.brier_score,
                win_rate=report.win_rate,
                n_trades=report.n_trades,
                worst_category=report.worst_category,
                best_category=report.best_category,
                top_lesson=learned.sonnet_rationale,
                threshold_changes={
                    "category_min_edge": learned.category_min_edge,
                    "category_skip": learned.category_skip,
                    "ensemble_recommended": learned.ensemble_recommended,
                },
            )
            logger.info(
                "Weekly evaluation complete. Brier=%.3f Win=%.1f%% Trades=%d",
                report.brier_score, report.win_rate * 100, report.n_trades,
            )
        else:
            logger.info("Weekly evaluation skipped — insufficient resolved trade data.")
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

    # Cap markets per cycle to stay within Anthropic rate limits
    if len(opportunities) > Config.MAX_MARKETS_PER_CYCLE:
        logger.info(
            "Capping cycle to %d markets (found %d) to avoid rate limiting.",
            Config.MAX_MARKETS_PER_CYCLE, len(opportunities),
        )
        opportunities = opportunities[:Config.MAX_MARKETS_PER_CYCLE]

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

    # Step 4: Critic + risk check + execute
    for forecast in forecasts:

        # Step 3.5: Devil's advocate critique
        critique = None
        try:
            critique = challenge_forecast(forecast)
        except Exception as exc:
            logger.warning("Critic unavailable for '%s': %s", forecast.question[:50], exc)

        if critique is not None and critique.concern_level == "HIGH":
            alert_trade_blocked(
                forecast.question,
                f"Critic veto: {critique.rationale}",
            )
            continue

        decision = evaluate_trade(forecast, critique=critique)

        if not decision.approved:
            alert_trade_blocked(forecast.question, decision.blocked_reason or "unknown")
            continue

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
                critic_concern=getattr(critique, "concern_level", None),
                critic_summary=(
                    critique.counter_arguments[0] if critique and critique.counter_arguments else None
                ),
            )

    # Step 5: Monitor open positions for exits
    closed_positions = monitor_open_positions()
    for pos in closed_positions:
        if pos.get("resolved_yes") is not None and pos.get("predicted_prob") is not None:
            # Confirmed outcome — use richer resolved alert with Brier score
            predicted_prob = pos["predicted_prob"]
            actual_outcome = pos["resolved_yes"]
            brier = (predicted_prob - (1.0 if actual_outcome else 0.0)) ** 2
            alert_trade_resolved(
                question=pos["question"],
                side=pos.get("side", ""),
                pnl=pos["pnl"],
                predicted_prob=predicted_prob,
                actual_outcome=actual_outcome,
                brier_contribution=round(brier, 4),
            )
        else:
            # Stop-loss or unresolved exit — use basic exit alert
            alert_trade_exit(question=pos["question"], pnl=pos["pnl"])

    logger.info("=== Trading cycle complete ===")


def main() -> None:
    _configure_logging()
    logger = logging.getLogger("main")

    logger.info("Polymarket Autonomous Trading Bot starting up...")

    try:
        Config.validate()
    except ValueError as exc:
        logger.critical("Config validation failed: %s", exc)
        raise SystemExit(1)

    if not backtest_already_run():
        logger.info("First launch — running backtest...")
        try:
            run_backtest()
        except Exception as exc:
            logger.warning("Backtest failed (non-fatal): %s", exc)
    else:
        logger.info("Backtest already completed — skipping.")

    summary = get_portfolio_summary()
    alert_startup(paper_trading=IS_PAPER_TRADING, balance=summary["balance"])

    summary_time = os.environ.get("DAILY_SUMMARY_TIME_UTC", "14:00")
    schedule.every().day.at(summary_time).do(_send_daily_summary)

    _weekly_schedule = getattr(schedule.every(), Config.WEEKLY_EVAL_DAY, schedule.every().monday)
    _weekly_schedule.at("14:00").do(_run_weekly_evaluation)

    logger.info(
        "Bot running | Mode: %s | Scan interval: %d min",
        "PAPER" if IS_PAPER_TRADING else "LIVE",
        Config.SCAN_INTERVAL_MINUTES,
    )

    _heartbeat_file = "logs/last_heartbeat.txt"
    _heartbeat_alerted = False

    def _write_heartbeat() -> None:
        try:
            with open(_heartbeat_file, "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass

    def _check_heartbeat() -> None:
        nonlocal _heartbeat_alerted
        try:
            with open(_heartbeat_file) as f:
                last_ts = float(f.read().strip())
            elapsed_min = (time.time() - last_ts) / 60
            threshold = Config.SCAN_INTERVAL_MINUTES * 2
            if elapsed_min > threshold:
                if not _heartbeat_alerted:
                    alert_heartbeat_missed(int(elapsed_min))
                    _heartbeat_alerted = True
            else:
                _heartbeat_alerted = False
        except Exception:
            pass

    _run_trading_cycle()  # Run once immediately on startup
    _write_heartbeat()

    next_cycle_time = time.time() + Config.SCAN_INTERVAL_MINUTES * 60

    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
            _check_heartbeat()

            if time.time() >= next_cycle_time:
                _run_trading_cycle()
                _write_heartbeat()
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
                pass
            logger.info("Sleeping 5 minutes before retry...")
            time.sleep(300)
            next_cycle_time = time.time() + Config.SCAN_INTERVAL_MINUTES * 60


if __name__ == "__main__":
    main()
