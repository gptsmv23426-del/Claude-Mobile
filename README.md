# Polymarket Autonomous Trading Bot

## What this is
A fully autonomous AI trading bot for Polymarket prediction markets, powered by Claude AI for research and probability forecasting. Runs in paper trading mode by default — no real money until you flip one config flag.

## Prerequisites
- A Polymarket account with an L2 wallet (Polygon) and USDC funded via the Polymarket Safe
- A Telegram bot token and chat ID (create via [@BotFather](https://t.me/BotFather))
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- Docker and Docker Compose installed

## Setup
1. Clone the repository: `git clone <repo-url> && cd polymarket-bot`
2. Copy the env template: `cp .env.example .env`
3. Fill in all values in `.env` (private key, API keys, Telegram credentials)
4. Start the bot: `docker-compose up -d`
5. Watch Telegram — you'll receive a startup alert and trade notifications as they happen

## Going live
Change `PAPER_TRADING=false` in `.env` and restart with `docker-compose down && docker-compose up -d`.

## Safety rules
- Link a prepaid or virtual card to your Anthropic account with a monthly spending cap
- Never share your `.env` file or commit it to version control
- Start with `MAX_POSITION_SIZE_USDC=10` or lower until the backtest proves consistent profitability
- Keep `MAX_DRAWDOWN_GATE=0.15` — the bot will auto-pause if you lose more than 15% from peak
- Review `logs/trades.jsonl` regularly to understand every decision the bot is making
