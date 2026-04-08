"""
Evaluator — trade evaluation, calibration tracking, and Sonnet learning loop.

PURPOSE:
  Answers: "Are our probability forecasts actually correct, and are we learning?"

  1. Record every resolved trade with predicted probability and actual outcome
  2. Compute calibration metrics (Brier score, win rate by category, confidence accuracy)
  3. Run a weekly Claude Sonnet strategy review that outputs specific threshold changes

LEARNING LOOP SEQUENCE:
  executor.py  -> resolves trade -> calls record_resolved_trade()
  evaluator.py -> writes to logs/evaluation_log.jsonl
  main.py      -> weekly on Config.WEEKLY_EVAL_DAY -> calls run_weekly_review()
  evaluator.py -> reads log -> builds CalibrationReport -> calls Sonnet
               -> writes to Config.LEARNED_THRESHOLDS_PATH
  config.py    -> reads learned_thresholds.json on next startup

Relationship with calibration_tracker.py:
  calibration_tracker: lightweight continuous tracker (forecasts + resolution polling)
  evaluator: richer periodic analysis with Brier scores and Sonnet strategy review
  They are complementary and write to separate log files.
"""

import json
import logging
import os
import statistics
from datetime import datetime, timezone
from typing import Optional

import anthropic
import numpy as np
from pydantic import BaseModel

from config import Config

logger = logging.getLogger(__name__)

EVALUATION_LOG_FILE = "logs/evaluation_log.jsonl"


# =============================================================================
# DATA MODELS
# =============================================================================

class ResolvedTrade(BaseModel):
    """
    One fully resolved trade — written to evaluation_log.jsonl.
    Both predicted_probability and actual_outcome must be known before writing.
    """
    timestamp_resolved: str
    market_id: str
    question: str
    category: str
    side: str
    size_usdc: float
    entry_price: float
    predicted_probability: float   # bot's YES probability forecast (0.0-1.0)
    actual_outcome: bool           # True = YES resolved, False = NO resolved
    pnl: float
    edge_at_entry: float
    confidence: str
    evidence_quality: float
    days_held: float
    cross_ref_metaculus: Optional[float] = None
    cross_ref_manifold: Optional[float] = None


class CalibrationReport(BaseModel):
    """Aggregated metrics across all resolved trades. Fed to Sonnet for weekly review."""
    n_trades: int
    brier_score: float
    win_rate: float
    total_pnl: float
    avg_edge_predicted: float
    avg_edge_realized: float
    win_rate_by_category: dict
    brier_by_category: dict
    confidence_win_rate: dict
    evidence_quality_correlation: float
    worst_category: str
    best_category: str
    report_generated_at: str


class LearnedThresholds(BaseModel):
    """Written by run_weekly_review(). Read by config.py on next startup."""
    generated_at: str
    sonnet_rationale: str
    global_min_edge: Optional[float] = None
    global_min_confidence: Optional[str] = None
    category_min_edge: dict = {}
    category_skip: list = []
    ensemble_recommended: bool = False


# =============================================================================
# WRITE / READ EVALUATION LOG
# =============================================================================

def record_resolved_trade(trade: ResolvedTrade) -> None:
    """Append one resolved trade to evaluation_log.jsonl."""
    os.makedirs("logs", exist_ok=True)
    with open(EVALUATION_LOG_FILE, "a") as f:
        f.write(trade.model_dump_json() + "\n")
    logger.debug("Evaluation log: recorded %s (outcome=%s)", trade.market_id, trade.actual_outcome)


def load_evaluation_log() -> list[ResolvedTrade]:
    """Read all resolved trades from evaluation_log.jsonl. Returns empty list if file missing."""
    trades = []
    if not os.path.exists(EVALUATION_LOG_FILE):
        return trades
    with open(EVALUATION_LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(ResolvedTrade.model_validate_json(line))
                except Exception as exc:
                    logger.warning("Skipping malformed evaluation log line: %s", exc)
    return trades


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def _brier_score(predicted: float, actual: bool) -> float:
    """Single-trade Brier contribution."""
    return (predicted - (1.0 if actual else 0.0)) ** 2


def compute_calibration_report(trades: list[ResolvedTrade]) -> Optional[CalibrationReport]:
    """
    Compute calibration metrics. Returns None if fewer than 10 trades.

    Brier score interpretation:
      < 0.10: excellent   0.10-0.20: good   0.20-0.25: poor   > 0.25: problematic
    """
    if len(trades) < 10:
        logger.info("Too few trades (%d) for meaningful calibration report.", len(trades))
        return None

    brier_scores = [_brier_score(t.predicted_probability, t.actual_outcome) for t in trades]
    wins = [t for t in trades if t.pnl > 0]

    categories = list({t.category for t in trades})
    win_rate_by_cat: dict = {}
    brier_by_cat: dict = {}
    for cat in categories:
        cat_trades = [t for t in trades if t.category == cat]
        if cat_trades:
            cat_wins = [t for t in cat_trades if t.pnl > 0]
            win_rate_by_cat[cat] = round(len(cat_wins) / len(cat_trades), 4)
            brier_by_cat[cat] = round(
                statistics.mean([_brier_score(t.predicted_probability, t.actual_outcome) for t in cat_trades]), 4
            )

    confidence_groups: dict = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for t in trades:
        confidence_groups.get(t.confidence, confidence_groups["LOW"]).append(t.pnl > 0)
    confidence_win_rate = {
        k: round(statistics.mean(v), 4) if v else 0.0 for k, v in confidence_groups.items()
    }

    worst_cat = min(win_rate_by_cat, key=win_rate_by_cat.get, default="UNKNOWN") if win_rate_by_cat else "UNKNOWN"
    best_cat = max(win_rate_by_cat, key=win_rate_by_cat.get, default="UNKNOWN") if win_rate_by_cat else "UNKNOWN"

    avg_edge_realized = statistics.mean([t.pnl / t.size_usdc for t in trades if t.size_usdc > 0])
    avg_edge_predicted = statistics.mean([t.edge_at_entry for t in trades])

    # Evidence quality correlation: does higher quality actually predict wins?
    qualities = [t.evidence_quality for t in trades]
    wins_binary = [1.0 if t.pnl > 0 else 0.0 for t in trades]
    try:
        if len(set(qualities)) > 1:
            corr_matrix = np.corrcoef(qualities, wins_binary)
            evidence_quality_corr = round(float(corr_matrix[0, 1]), 4)
        else:
            evidence_quality_corr = 0.0
    except Exception:
        evidence_quality_corr = 0.0

    return CalibrationReport(
        n_trades=len(trades),
        brier_score=round(statistics.mean(brier_scores), 4),
        win_rate=round(len(wins) / len(trades), 4),
        total_pnl=round(sum(t.pnl for t in trades), 2),
        avg_edge_predicted=round(avg_edge_predicted, 4),
        avg_edge_realized=round(avg_edge_realized, 4),
        win_rate_by_category=win_rate_by_cat,
        brier_by_category=brier_by_cat,
        confidence_win_rate=confidence_win_rate,
        evidence_quality_correlation=evidence_quality_corr,
        worst_category=worst_cat,
        best_category=best_cat,
        report_generated_at=datetime.now(timezone.utc).isoformat(),
    )


def get_calibration_summary() -> Optional[CalibrationReport]:
    """
    Convenience function for researcher/forecaster to inject calibration context
    into prompts. Returns None if no data or too few trades (< 10).
    """
    trades = load_evaluation_log()
    return compute_calibration_report(trades)


# =============================================================================
# SONNET STRATEGY REVIEW (THE LEARNING LOOP)
# =============================================================================

_SONNET_REVIEW_SYSTEM = """You are a trading strategy analyst reviewing a prediction market bot's performance.
You will receive a calibration report showing how well the bot's probability forecasts match actual outcomes.
Your job is to identify specific, actionable threshold changes that would improve future performance.

Rules:
- Only recommend changes supported by the data. Do not speculate.
- Require at least 15 trades per category before recommending category-level changes.
- If the sample size is too small, say so and recommend no change.
- Output ONLY a JSON object. No preamble. No markdown. Raw JSON only.
- JSON schema:
  {
    "sonnet_rationale": "1-2 sentence explanation of the most important finding",
    "global_min_edge": null or float (e.g., 0.09 to raise threshold),
    "global_min_confidence": null or "MEDIUM" or "HIGH",
    "category_min_edge": {} or {"POLITICS": 0.12},
    "category_skip": [] or ["WEATHER"],
    "ensemble_recommended": false or true
  }
"""


def run_weekly_review() -> Optional[LearnedThresholds]:
    """
    Run the weekly Claude Sonnet strategy review.
    Reads evaluation_log.jsonl -> builds CalibrationReport -> calls Sonnet ->
    writes learned_thresholds.json -> returns LearnedThresholds.
    Returns None if insufficient trade data (< 10 trades).
    """
    trades = load_evaluation_log()
    report = compute_calibration_report(trades)
    if report is None:
        logger.info("Weekly review skipped — insufficient trade data (%d trades).", len(trades))
        return None

    client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    user_prompt = f"Calibration report:\n{report.model_dump_json(indent=2)}"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=_SONNET_REVIEW_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in response.content if hasattr(b, "text")).strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    data = json.loads(text)

    learned = LearnedThresholds(
        generated_at=datetime.now(timezone.utc).isoformat(),
        sonnet_rationale=data.get("sonnet_rationale", ""),
        global_min_edge=data.get("global_min_edge"),
        global_min_confidence=data.get("global_min_confidence"),
        category_min_edge=data.get("category_min_edge", {}),
        category_skip=data.get("category_skip", []),
        ensemble_recommended=data.get("ensemble_recommended", False),
    )

    os.makedirs("logs", exist_ok=True)
    with open(Config.LEARNED_THRESHOLDS_PATH, "w") as f:
        f.write(learned.model_dump_json(indent=2))
    logger.info("Learned thresholds written to %s", Config.LEARNED_THRESHOLDS_PATH)
    return learned


def load_learned_thresholds() -> Optional[LearnedThresholds]:
    """
    Load the most recent Sonnet-generated threshold recommendations.
    Returns None if file does not exist (bot uses Config defaults).
    """
    path = Config.LEARNED_THRESHOLDS_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return LearnedThresholds.model_validate_json(f.read())
    except Exception as exc:
        logger.warning("Could not load learned thresholds from %s: %s", path, exc)
        return None


# =============================================================================
# BACKTEST SEEDING
# =============================================================================

def seed_from_backtest(simulated_trades: list[dict]) -> int:
    """
    Convert backtester simulated trades into ResolvedTrade objects and seed
    evaluation_log.jsonl with historical calibration data.
    Called by backtester.run_backtest() at the end of a backtest run.
    Returns number of trades successfully written.
    """
    count = 0
    for t in simulated_trades:
        if t.get("resolved_yes") is None:
            continue
        try:
            resolved = ResolvedTrade(
                timestamp_resolved=datetime.now(timezone.utc).isoformat(),
                market_id=t.get("condition_id", ""),
                question=t.get("question", ""),
                category=t.get("category", "UNKNOWN"),
                side=t.get("side", "YES"),
                size_usdc=t.get("size_usdc", 0.0),
                entry_price=t.get("entry_price", 0.0),
                predicted_probability=t.get("predicted_prob", 0.5),
                actual_outcome=bool(t.get("resolved_yes")),
                pnl=t.get("pnl", 0.0),
                edge_at_entry=t.get("edge", 0.0),
                confidence=t.get("confidence", "MEDIUM"),
                evidence_quality=t.get("evidence_quality", 0.5),
                days_held=t.get("days_held", 0.0),
            )
            record_resolved_trade(resolved)
            count += 1
        except Exception as exc:
            logger.warning("Could not seed backtest trade: %s", exc)
    logger.info("Seeded %d backtest trades into evaluation log.", count)
    return count
