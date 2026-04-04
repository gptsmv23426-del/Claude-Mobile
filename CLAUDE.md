# Polymarket Bot — Claude Context

## What this is
Fully autonomous Polymarket prediction market trading bot.
Paper trading by default. Real money when PAPER_TRADING=false.

## Hard Rules
- Never hardcode API keys. .env only, always.
- All amounts in USDC, not cents. Log WARNING if cents detected.
- Log every trade decision with rationale BEFORE executing.
- MAX_POSITION_SIZE_USDC is a ceiling, not a suggestion.
- Never let an exception kill the main loop. Catch, log, alert, continue.
- Use claude-haiku-4-5 for all routine tasks (scanning, parsing, alerts).
- Use claude-sonnet-4-6 only for deep research and strategy backtesting.

## What we are NOT doing
- No async. Sync only. Predictability over speed.
- No ML models or neural nets. Claude API is the reasoning engine.
- No framework magic in the trading loop. Keep main.py readable.
- No external servers storing private keys.

## Build Order (Completed)
1. Scaffold + CLAUDE.md + config
2. Market scanner
3. Researcher + forecaster
4. Risk manager + backtester
5. Executor (paper mode first)
6. Telegram monitor
7. Main loop
8. Docker setup
9. README

## Phase 1 (DONE) — Backtester Credibility
- Real resolution via Gamma API outcomePrices field
- Real entry prices via CLOB prices-history endpoint
- VectorBT Portfolio.from_orders() for metrics
- Pandas fallback if vectorbt not installed

## Phase 2 — Signal Enrichment (PLANNED, NOT YET IMPLEMENTED)
Goal: Give the researcher richer context before Claude makes a forecast.
All sources below are free. No new installs required for 2A.

### Phase 2A — Cross-Reference Forecaster (zero new deps)
File to create: signals/cross_reference.py
- Query Metaculus API (https://www.metaculus.com/api2/questions/) for matching questions
  - Search by keyword from market.question
  - Pull community_prediction.full.q2 (median forecast)
  - Free, no auth required
- Query Manifold Markets API (https://api.manifold.markets/v0/markets) for matching binary markets
  - Search by keyword, filter contractType=BINARY
  - Pull probability field
  - Free, no auth required
- Return a CrossReferenceResult with {metaculus_prob, manifold_prob, agreement_score}
Integration point: researcher.py passes CrossReferenceResult into the forecaster prompt
Effect on forecaster: if all 3 sources agree → confidence can be HIGH. If diverge >0.15 → shrink toward 0.5.

### Phase 2B — FRED Macro Context (one new dep: pip install fredapi)
File to create: signals/macro_context.py
- Pull 5 FRED series on startup and cache for 24h:
  - FEDFUNDS — Fed Funds Rate
  - CPIAUCSL — CPI (inflation)
  - UNRATE   — Unemployment Rate
  - SP500    — S&P 500 level
  - DGS10    — 10-Year Treasury Yield
- Requires FRED_API_KEY in .env (free at https://fred.stlouisfed.org/docs/api/fred/)
Integration point: researcher.py injects current macro snapshot into system prompt for MACRO category markets only

### Phase 2C — Backtester Data Depth (git clone, no pip)
- Clone https://github.com/SII-WANGZJ/Polymarket_data locally
- Add a POLYMARKET_DATA_PATH config var pointing to the clone
- backtester.py: if POLYMARKET_DATA_PATH is set, load from local files instead of CLOB API
- Removes the 60-market CLOB rate limit cap entirely
- Enables multi-thousand-market backtests without API calls

## Phase 3 — Ensemble Forecasting (PLANNED)
Goal: Reduce single-call variance in the forecaster.
File: forecaster.py — modify forecast_market()
- Run 3 independent Haiku calls per market, each with a different system prompt frame:
  1. Base rate anchor: "Start from the historical base rate for this event type, then update."
  2. Bullish frame: "Steelman the case for YES. What probability does the evidence support?"
  3. Bearish frame: "Steelman the case for NO. What probability does the evidence support?"
- Average the 3 probabilities → final forecast
- If std dev across 3 > 0.12 → downgrade confidence to LOW (too uncertain to trust average)
- Cost: ~3x Haiku tokens per market. At $0.25/M tokens, negligible.

## Phase 4 — Trade Evaluation & Learning Loop (PLANNED)
Goal: Know if the bot is actually working and improve thresholds without ML.
File to create: evaluator.py

### What the evaluator measures
- Brier Score: (predicted_prob - actual_outcome)^2 per trade. Mean across all trades = calibration score.
  - 0.0 = perfect calibration. 0.25 = coin flip. Lower is always better.
- Win rate by category: CRYPTO, MACRO, POLITICS, etc. separately
- Edge realization: predicted_edge vs (actual_pnl / size_usdc). Are we capturing the edge we think we have?
- Confidence accuracy: HIGH confidence trades should win more than MEDIUM. Track separately.
- Evidence quality correlation: does quality score actually predict wins?

### The learning loop (Claude as reasoning engine, no ML)
Step 1 (evaluator.py): After each market resolves, fetch actual outcome from Gamma API outcomePrices
         and write {predicted_prob, actual_outcome, category, confidence, evidence_quality, edge, pnl}
         to logs/evaluation_log.jsonl

Step 2 (evaluator.py — weekly): Aggregate evaluation_log.jsonl into a calibration_report dict:
         {brier_score, win_rate, win_rate_by_category, confidence_accuracy, edge_realization_rate}

Step 3 (evaluator.py — weekly): Feed calibration_report to Claude Sonnet with this prompt:
         "You are a trading strategy analyst. Here is the bot's performance over the last N trades.
          Identify which categories to avoid, whether edge threshold should be raised or lowered,
          whether confidence filter is calibrated, and output specific .env value recommendations."

Step 4 (evaluator.py): Write Sonnet's recommended threshold changes to logs/learned_thresholds.json
         Config reads learned_thresholds.json on startup and overrides defaults where present.

Step 5 (monitor.py): Send weekly calibration report to Telegram with Brier score, win rate, top lesson.

### The critical prerequisite for evaluation
executor.py _close_position() currently uses exit_price = entry_price (neutral assumption).
This must be fixed BEFORE evaluation is meaningful:
- After time exit (4h before expiry), call Gamma API to get outcomePrices for the market
- Compute real binary P&L: if YES and resolved YES → profit = size * (1/entry_price - 1)
                           if YES and resolved NO  → loss  = -size
                           if NO  and resolved NO  → profit = size * (1/entry_price - 1)
                           if NO  and resolved YES → loss  = -size
- Store actual_outcome (True/False) alongside predicted_probability in every closed trade record

## .env Variables to Add (PLANNED)
FRED_API_KEY=                    # Free at fred.stlouisfed.org — needed for Phase 2B
POLYMARKET_DATA_PATH=            # Local path to cloned Polymarket_data repo — Phase 2C
ENABLE_CROSS_REFERENCE=true      # Toggle Metaculus/Manifold lookups — Phase 2A
ENABLE_MACRO_CONTEXT=true        # Toggle FRED macro injection — Phase 2B
ENABLE_ENSEMBLE_FORECAST=false   # Toggle 3x Haiku ensemble — Phase 3 (costs 3x tokens)
WEEKLY_EVAL_DAY=monday           # Day to run Sonnet strategy review — Phase 4
LEARNED_THRESHOLDS_PATH=logs/learned_thresholds.json  # Phase 4 output

## New Files to Create (PLANNED)
signals/
  __init__.py
  cross_reference.py    # Phase 2A — Metaculus + Manifold lookup
  macro_context.py      # Phase 2B — FRED macro indicators
evaluator.py            # Phase 4 — Brier score, calibration report, Sonnet learning loop
