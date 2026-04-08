"""
Polymarket Autonomous Trading Bot
Entry point and main loop.
"""

import json
import logging
import os
import time
import traceback
from datetime import datetime, timedelta, timezone

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
    alert_drought,
)
from backtester import run_backtest, backtest_already_run
from calibration_tracker import log_forecast, check_and_update_resolutions
from rate_limiter import TokenRateLimiter

EVALUATED_CACHE_FILE = "logs/evaluated_markets.json"
EVALUATED_COOLDOWN_CYCLES = 2  # skip a market for this many cycles after evaluating it


def _load_evaluated_cache() -> dict:
    try:
        with open(EVALUATED_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_evaluated_cache(cache: dict) -> None:
    with open(EVALUATED_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


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
    limiter = TokenRateLimiter(tokens_per_minute=Config.TPM_LIMIT)

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

    # Filter out markets already held in open positions (avoid wasting research tokens)
    try:
        with open(Config.PAPER_PORTFOLIO_FILE) as f:
            portfolio = json.load(f)
        open_market_ids = set(portfolio.get("open_positions", {}).keys())
    except (FileNotFoundError, json.JSONDecodeError):
        open_market_ids = set()

    opportunities = [o for o in opportunities if o.market_id not in open_market_ids]
    if open_market_ids:
        logger.info(f"Filtered {len(open_market_ids)} already-held markets before research.")

    # Skip markets evaluated recently (within 2 cycle-lengths) to force rotation
    evaluated_cache = _load_evaluated_cache()
    cooldown_minutes = Config.SCAN_INTERVAL_MINUTES * EVALUATED_COOLDOWN_CYCLES
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    def _on_cooldown(market_id: str) -> bool:
        if market_id not in evaluated_cache:
            return False
        last = datetime.fromisoformat(evaluated_cache[market_id]).replace(tzinfo=None)
        return (now - last).total_seconds() < cooldown_minutes * 60

    before = len(opportunities)
    opportunities = [o for o in opportunities if not _on_cooldown(o.market_id)]
    skipped = before - len(opportunities)
    if skipped:
        logger.info(f"Skipped {skipped} markets on evaluation cooldown.")

    # Enforce category diversity and cap to stay within Anthropic rate limits
    category_counts: dict = {}
    diverse_opportunities = []
    for opp in opportunities:
        cat = opp.category
        if category_counts.get(cat, 0) < Config.MAX_MARKETS_PER_CATEGORY:
            diverse_opportunities.append(opp)
            category_counts[cat] = category_counts.get(cat, 0) + 1
        if len(diverse_opportunities) >= Config.MAX_MARKETS_PER_CYCLE:
            break
    if len(diverse_opportunities) < len(opportunities):
        logger.info(
            "Capping cycle to %d diverse markets (found %d, max %d per category) to avoid rate limiting.",
            len(diverse_opportunities), len(opportunities), Config.MAX_MARKETS_PER_CATEGORY,
        )
    opportunities = diverse_opportunities

    # Update evaluated cache for all markets about to be researched
    now_iso = datetime.now(timezone.utc).isoformat()
    cache = _load_evaluated_cache()
    for opp in opportunities:
        cache[opp.market_id] = now_iso
    # Prune entries older than 7 days to prevent unbounded growth
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    cache = {k: v for k, v in cache.items() if v > cutoff}
    _save_evaluated_cache(cache)

    # Step 1.5: Pre-screen to avoid wasting research tokens on low-potential markets
    if Config.PRE_SCREEN_ENABLED and len(opportunities) > 2:
        from pre_screener import pre_screen_markets
        screen_results, screen_response = pre_screen_markets(opportunities)
        if screen_response:
            limiter.record_from_response(screen_response)
        tradeable_ids = {r.market_id for r in screen_results if r.tradeable}
        before_screen = len(opportunities)
        opportunities = [o for o in opportunities if o.market_id in tradeable_ids]
        if len(opportunities) < before_screen:
            logger.info(
                "Pre-screen filtered %d/%d markets (saved ~%dk research tokens).",
                before_screen - len(opportunities), before_screen,
                (before_screen - len(opportunities)) * 8,
            )
        if not opportunities:
            logger.info("No markets passed pre-screen.")
            return

    # Step 2: Research
    research_results = research_markets(opportunities, limiter=limiter)
    if not research_results:
        logger.info("No markets passed research quality threshold.")
        return

    # Phase gap: let rate limiter decide how long to wait
    limiter.wait_if_needed(next_call_estimate=3_000)

    # Step 3: Forecast
    forecasts = forecast_markets(research_results, limiter=limiter)
    if not forecasts:
        logger.info("No forecasts produced.")
        return

    # Phase gap: let rate limiter decide how long to wait
    limiter.wait_if_needed(next_call_estimate=2_000)

    # Step 4: Critic + risk check + execute
    for i, forecast in enumerate(forecasts):
        if i > 0:
            limiter.wait_if_needed(next_call_estimate=2_000)

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

    # Drought tracking — alert if no trades for 48 hours
    _drought_hours = 48
    _drought_alerted = False
    _cycle_count = 0

    def _get_last_trade_time() -> float:
        """Get timestamp of most recent trade from trades.jsonl."""
        try:
            with open("logs/trades.jsonl") as f:
                lines = f.readlines()
            if lines:
                last = json.loads(lines[-1])
                from datetime import datetime as _dt
                return _dt.fromisoformat(last["timestamp"]).timestamp()
        except Exception:
            pass
        return time.time()  # no trades file = assume now

    def _check_drought() -> None:
        nonlocal _drought_alerted
        hours = (time.time() - _get_last_trade_time()) / 3600
        if hours >= _drought_hours and not _drought_alerted:
            alert_drought(hours, _cycle_count)
            _drought_alerted = True
            logger.warning("Trade drought: %d hours, %d cycles without a trade.", int(hours), _cycle_count)
        elif hours < _drought_hours:
            _drought_alerted = False  # reset after a trade

    _run_trading_cycle()  # Run once immediately on startup
    _write_heartbeat()
    _cycle_count += 1

    next_cycle_time = time.time() + Config.SCAN_INTERVAL_MINUTES * 60

    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
            _check_heartbeat()
            _check_drought()

            if time.time() >= next_cycle_time:
                _run_trading_cycle()
                _write_heartbeat()
                _cycle_count += 1
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
