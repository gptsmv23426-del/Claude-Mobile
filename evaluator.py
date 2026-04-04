"""
Evaluator — trade evaluation, calibration tracking, and Sonnet learning loop.

STATUS: PLANNED — skeleton only. Functions raise NotImplementedError until implemented.

PURPOSE:
  This module answers the most important question about the bot:
  "Are our probability forecasts actually correct, and are we learning from mistakes?"

  It does three things:
    1. Record every resolved trade with its predicted probability and actual outcome
    2. Compute calibration metrics (Brier score, win rate by category, confidence accuracy)
    3. Run a weekly Claude Sonnet strategy review that outputs specific threshold changes

WHY BRIER SCORE (not just win rate):
  Win rate alone is misleading in prediction markets. If we say 90% confidence and win,
  that's not impressive. If we say 90% and lose, that's catastrophic. Brier score captures
  both — it penalizes overconfidence on losses and rewards accurate probability estimates.

  Brier Score = mean((predicted_prob - actual_outcome)^2) across all trades
  - 0.0  = perfect calibration
  - 0.25 = equivalent to always guessing 0.5 (coin flip)
  - 1.0  = perfectly wrong every time
  Target: Brier < 0.20 before scaling position sizes.

WHY SONNET FOR STRATEGY REVIEW (not a rule engine):
  Threshold adjustments are not mechanical. "POLITICS markets underperform" could mean:
    a) Raise MIN_EDGE_THRESHOLD for POLITICS
    b) Skip POLITICS entirely
    c) The sample size is too small to conclude anything
  Claude Sonnet can reason about which interpretation is correct given the data.
  This respects the CLAUDE.md hard rule: Claude API is the reasoning engine.

LEARNING LOOP SEQUENCE (Phase 4):
  executor.py  → resolves trade → calls record_resolved_trade()
  evaluator.py → writes to logs/evaluation_log.jsonl
  main.py      → weekly on Config.WEEKLY_EVAL_DAY → calls run_weekly_review()
  evaluator.py → reads evaluation_log.jsonl → builds calibration_report
              → calls Claude Sonnet with calibration_report
              → Sonnet outputs threshold recommendations as JSON
              → writes to Config.LEARNED_THRESHOLDS_PATH
  config.py    → reads learned_thresholds.json on next startup
  forecaster.py → applies per-category edge thresholds from learned file
  researcher.py → applies calibration shrinkage from Brier score
"""

import json
import logging
import os
import statistics
from datetime import datetime, timezone
from typing import Optional

import anthropic
from pydantic import BaseModel

from config import Config

logger = logging.getLogger(__name__)

EVALUATION_LOG_FILE = "logs/evaluation_log.jsonl"


# =============================================================================
# DATA MODELS
# =============================================================================

class ResolvedTrade(BaseModel):
    """
    One fully resolved trade record — written to evaluation_log.jsonl.
    Both predicted_probability and actual_outcome must be known before writing.
    """
    timestamp_resolved: str          # ISO UTC when outcome was confirmed
    market_id: str
    question: str
    category: str
    side: str                        # "YES" or "NO"
    size_usdc: float
    entry_price: float
    predicted_probability: float     # bot's forecast (0.0–1.0 for YES)
    actual_outcome: bool             # True = YES resolved, False = NO resolved
    pnl: float                       # real binary P&L in USDC
    edge_at_entry: float             # predicted_prob minus market price at entry
    confidence: str                  # "LOW" | "MEDIUM" | "HIGH"
    evidence_quality: float
    days_held: float
    cross_ref_metaculus: Optional[float] = None   # Phase 2A: Metaculus prob at entry
    cross_ref_manifold: Optional[float] = None    # Phase 2A: Manifold prob at entry


class CalibrationReport(BaseModel):
    """
    Aggregated metrics across all resolved trades.
    Fed to Claude Sonnet for the weekly strategy review.
    """
    n_trades: int
    brier_score: float               # mean((predicted - actual)^2)
    win_rate: float                  # fraction of trades with pnl > 0
    total_pnl: float
    avg_edge_predicted: float        # mean edge we thought we had
    avg_edge_realized: float         # mean actual pnl / size_usdc
    win_rate_by_category: dict       # {"CRYPTO": 0.65, "POLITICS": 0.40, ...}
    brier_by_category: dict          # {"CRYPTO": 0.18, "POLITICS": 0.27, ...}
    confidence_win_rate: dict        # {"HIGH": 0.72, "MEDIUM": 0.58, "LOW": 0.0}
    evidence_quality_correlation: float  # correlation between quality score and win
    worst_category: str
    best_category: str
    report_generated_at: str


class LearnedThresholds(BaseModel):
    """
    Written by run_weekly_review() after Sonnet's analysis.
    Read by config.py on next startup.
    Sonnet populates only the fields it has high confidence changing.
    """
    generated_at: str
    sonnet_rationale: str            # Sonnet's explanation for the changes
    global_min_edge: Optional[float] = None       # override Config.MIN_EDGE_THRESHOLD
    global_min_confidence: Optional[str] = None   # override to "HIGH" only if poorly calibrated
    category_min_edge: dict = {}     # {"POLITICS": 0.12, "CRYPTO": 0.07}
    category_skip: list = []         # categories to add to SKIP_CATEGORIES temporarily
    ensemble_recommended: bool = False  # suggest enabling ENABLE_ENSEMBLE_FORECAST


# =============================================================================
# WRITE / READ EVALUATION LOG
# =============================================================================

def record_resolved_trade(trade: ResolvedTrade) -> None:
    """
    Append one resolved trade to evaluation_log.jsonl.
    Called by executor.py after _close_position() confirms the actual outcome.

    PLANNED — not yet called from executor.py. Implement Phase 4.
    """
    # TODO (Phase 4): Implement this.
    # os.makedirs("logs", exist_ok=True)
    # with open(EVALUATION_LOG_FILE, "a") as f:
    #     f.write(trade.model_dump_json() + "\n")
    raise NotImplementedError("record_resolved_trade: Phase 4 not yet implemented")


def load_evaluation_log() -> list[ResolvedTrade]:
    """
    Read all resolved trades from evaluation_log.jsonl.
    Returns empty list if file does not exist.

    PLANNED — implement in Phase 4.
    """
    # TODO (Phase 4): Implement this.
    # trades = []
    # if not os.path.exists(EVALUATION_LOG_FILE):
    #     return trades
    # with open(EVALUATION_LOG_FILE) as f:
    #     for line in f:
    #         line = line.strip()
    #         if line:
    #             try:
    #                 trades.append(ResolvedTrade.model_validate_json(line))
    #             except Exception as exc:
    #                 logger.warning("Skipping malformed evaluation log line: %s", exc)
    # return trades
    raise NotImplementedError("load_evaluation_log: Phase 4 not yet implemented")


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def _brier_score(predicted: float, actual: bool) -> float:
    """Single-trade Brier contribution."""
    return (predicted - (1.0 if actual else 0.0)) ** 2


def compute_calibration_report(trades: list[ResolvedTrade]) -> Optional[CalibrationReport]:
    """
    Compute all calibration metrics from a list of resolved trades.
    Returns None if fewer than 10 trades (not enough data to be meaningful).

    PLANNED — implement in Phase 4.

    Key metrics computed:
      - brier_score: mean((predicted_prob - actual_outcome)^2)
          Interpretation:
            < 0.10 = excellent (better than prediction markets)
            0.10–0.20 = good (competitive with good forecasters)
            0.20–0.25 = poor (barely better than coin flip)
            > 0.25 = problematic (consider pausing live trading)

      - win_rate_by_category: separate win rates for each category
          If any category has win_rate < 0.40 over 20+ trades → candidate for skip

      - confidence_win_rate: HIGH should > MEDIUM should > LOW
          If HIGH win rate < MEDIUM win rate → confidence scoring is miscalibrated

      - edge_realization_rate: avg_edge_realized / avg_edge_predicted
          If < 0.5 → we are systematically overestimating our edge
          If > 1.0 → edge estimates are conservative (good problem to have)
    """
    # TODO (Phase 4): Implement this.
    # if len(trades) < 10:
    #     logger.info("Too few trades (%d) for meaningful calibration report.", len(trades))
    #     return None
    #
    # brier_scores = [_brier_score(t.predicted_probability, t.actual_outcome) for t in trades]
    # wins = [t for t in trades if t.pnl > 0]
    #
    # categories = list({t.category for t in trades})
    # win_rate_by_cat = {}
    # brier_by_cat = {}
    # for cat in categories:
    #     cat_trades = [t for t in trades if t.category == cat]
    #     if cat_trades:
    #         cat_wins = [t for t in cat_trades if t.pnl > 0]
    #         win_rate_by_cat[cat] = len(cat_wins) / len(cat_trades)
    #         brier_by_cat[cat] = statistics.mean(
    #             [_brier_score(t.predicted_probability, t.actual_outcome) for t in cat_trades]
    #         )
    #
    # confidence_groups = {"HIGH": [], "MEDIUM": [], "LOW": []}
    # for t in trades:
    #     confidence_groups.get(t.confidence, []).append(t.pnl > 0)
    # confidence_win_rate = {
    #     k: statistics.mean(v) if v else 0.0 for k, v in confidence_groups.items()
    # }
    #
    # worst_cat = min(win_rate_by_cat, key=win_rate_by_cat.get, default="UNKNOWN")
    # best_cat = max(win_rate_by_cat, key=win_rate_by_cat.get, default="UNKNOWN")
    #
    # avg_edge_realized = statistics.mean([t.pnl / t.size_usdc for t in trades if t.size_usdc > 0])
    # avg_edge_predicted = statistics.mean([t.edge_at_entry for t in trades])
    #
    # return CalibrationReport(
    #     n_trades=len(trades),
    #     brier_score=round(statistics.mean(brier_scores), 4),
    #     win_rate=round(len(wins) / len(trades), 4),
    #     total_pnl=round(sum(t.pnl for t in trades), 2),
    #     avg_edge_predicted=round(avg_edge_predicted, 4),
    #     avg_edge_realized=round(avg_edge_realized, 4),
    #     win_rate_by_category=win_rate_by_cat,
    #     brier_by_category=brier_by_cat,
    #     confidence_win_rate=confidence_win_rate,
    #     evidence_quality_correlation=0.0,  # TODO: compute pearsonr(quality, win)
    #     worst_category=worst_cat,
    #     best_category=best_cat,
    #     report_generated_at=datetime.now(timezone.utc).isoformat(),
    # )
    raise NotImplementedError("compute_calibration_report: Phase 4 not yet implemented")


def get_calibration_summary() -> Optional[CalibrationReport]:
    """
    Convenience function called by researcher.py and forecaster.py to inject
    recent calibration context into prompts. Returns None if no data or too few trades.

    PLANNED — implement in Phase 4.
    """
    # TODO (Phase 4): Implement this.
    # trades = load_evaluation_log()
    # return compute_calibration_report(trades)
    return None  # Safe no-op until Phase 4 is implemented


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
    Reads evaluation_log.jsonl → builds calibration report → calls Sonnet →
    writes learned_thresholds.json → returns LearnedThresholds.

    Called from main.py on Config.WEEKLY_EVAL_DAY.
    PLANNED — implement in Phase 4.

    Implementation steps:
      1. Load all resolved trades from evaluation_log.jsonl
      2. Compute calibration report (need >= 10 trades)
      3. Format report as readable JSON for Sonnet
      4. Call claude-sonnet-4-6 with _SONNET_REVIEW_SYSTEM + calibration report
      5. Parse Sonnet's JSON response into LearnedThresholds
      6. Write to Config.LEARNED_THRESHOLDS_PATH
      7. Return LearnedThresholds for Telegram alert
    """
    # TODO (Phase 4): Implement this.
    # trades = load_evaluation_log()
    # report = compute_calibration_report(trades)
    # if report is None:
    #     logger.info("Weekly review skipped — insufficient trade data.")
    #     return None
    #
    # client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    # user_prompt = f"Calibration report:\n{report.model_dump_json(indent=2)}"
    #
    # response = client.messages.create(
    #     model="claude-sonnet-4-6",
    #     max_tokens=512,
    #     system=_SONNET_REVIEW_SYSTEM,
    #     messages=[{"role": "user", "content": user_prompt}],
    # )
    # text = "".join(b.text for b in response.content if hasattr(b, "text")).strip()
    # data = json.loads(text)
    #
    # learned = LearnedThresholds(
    #     generated_at=datetime.now(timezone.utc).isoformat(),
    #     sonnet_rationale=data.get("sonnet_rationale", ""),
    #     global_min_edge=data.get("global_min_edge"),
    #     global_min_confidence=data.get("global_min_confidence"),
    #     category_min_edge=data.get("category_min_edge", {}),
    #     category_skip=data.get("category_skip", []),
    #     ensemble_recommended=data.get("ensemble_recommended", False),
    # )
    #
    # os.makedirs("logs", exist_ok=True)
    # with open(Config.LEARNED_THRESHOLDS_PATH, "w") as f:
    #     f.write(learned.model_dump_json(indent=2))
    # logger.info("Learned thresholds written to %s", Config.LEARNED_THRESHOLDS_PATH)
    # return learned
    raise NotImplementedError("run_weekly_review: Phase 4 not yet implemented")


def load_learned_thresholds() -> Optional[LearnedThresholds]:
    """
    Load the most recent Sonnet-generated threshold recommendations.
    Returns None if file does not exist (bot uses Config defaults).
    Called by forecaster.py and config.py on startup.

    PLANNED — safe no-op until Phase 4.
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
# BACKTEST SEEDING (Phase 4 integration with backtester.py)
# =============================================================================

def seed_from_backtest(simulated_trades: list[dict]) -> int:
    """
    Convert backtester simulated trade records into ResolvedTrade objects and
    write them to evaluation_log.jsonl. This seeds the evaluator with historical
    calibration data before any live trades occur.

    Called by backtester.run_backtest() at the end of a backtest run.
    PLANNED — implement in Phase 4.

    Args:
        simulated_trades: list of dicts from backtester _simulate_trades(),
                          each with keys: condition_id, question, category,
                          side, size_usdc, entry_price, predicted_prob,
                          resolved_yes, pnl, edge, confidence, evidence_quality

    Returns:
        Number of trades successfully written.
    """
    # TODO (Phase 4): Implement this.
    # count = 0
    # for t in simulated_trades:
    #     if t.get("resolved_yes") is None:
    #         continue  # skip unresolved
    #     try:
    #         resolved = ResolvedTrade(
    #             timestamp_resolved=datetime.now(timezone.utc).isoformat(),
    #             market_id=t.get("condition_id", ""),
    #             question=t.get("question", ""),
    #             category=t.get("category", "UNKNOWN"),
    #             side=t.get("side", "YES"),
    #             size_usdc=t.get("size_usdc", 0.0),
    #             entry_price=t.get("entry_price", 0.0),
    #             predicted_probability=t.get("predicted_prob", 0.5),
    #             actual_outcome=bool(t.get("resolved_yes")),
    #             pnl=t.get("pnl", 0.0),
    #             edge_at_entry=t.get("edge", 0.0),
    #             confidence=t.get("confidence", "MEDIUM"),
    #             evidence_quality=t.get("evidence_quality", 0.5),
    #             days_held=t.get("days_held", 0.0),
    #         )
    #         record_resolved_trade(resolved)
    #         count += 1
    #     except Exception as exc:
    #         logger.warning("Could not seed backtest trade: %s", exc)
    # logger.info("Seeded %d backtest trades into evaluation log.", count)
    # return count
    raise NotImplementedError("seed_from_backtest: Phase 4 not yet implemented")
