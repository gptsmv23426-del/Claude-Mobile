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

## Build Order
1. Scaffold + CLAUDE.md + config
2. Market scanner
3. Researcher + forecaster
4. Risk manager + backtester
5. Executor (paper mode first)
6. Telegram monitor
7. Main loop
8. Docker setup
9. README
