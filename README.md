# Crypto Trading Signal Bot

A complete signal-only trading bot for **futures/macro signals** that generates LONG/SHORT/EXIT signals with Entry/SL/TP1/TP2 levels and sends them via Telegram.

## Features

- **Multi-Timeframe Analysis**: Uses trend (4h), entry (15m), and S/R (1h) timeframes
- **Technical Scoring**: Score-based signal generation with configurable thresholds
- **Timing Analysis**: Gann angles, Square of 9, 52-cycle, lunar phases, FOMC calendar
- **Risk Management**: Automated Entry/SL/TP calculation with configurable R:R ratios
- **Telegram Notifications**: Formatted signals with all analysis details
- **Flexible Market Source**: Binance/CCXT, direct MT5, or MT5 bridge (for broker-aligned pricing)

## Project Structure

```
crypto_signal_bot/
├── main.py              # Entry point
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
├── README.md            # This file
├── bot/
│   ├── __init__.py
│   ├── exchange.py      # CCXT + MT5 bridge adapters
│   ├── indicators.py    # SMA, EMA, RSI, ATR
│   ├── zones.py         # S/R zone detection
│   ├── patterns.py      # Fractals & candlestick patterns
│   ├── gann.py          # Gann angles & Square of 9
│   ├── time_cycles.py   # 52-cycle & lunar timing
│   ├── calendar_events.py # FOMC schedule
│   ├── signals.py       # Scoring engine
│   ├── risk.py          # Entry/SL/TP calculations
│   ├── notifier.py      # Telegram notifications
│   ├── storage.py       # Cooldown tracking
│   └── utils.py         # Utilities
└── tests/               # Pytest tests
```

## Direct MT5 Feed

If you want the bot to read prices and candles directly from MetaTrader 5 using the official Python package:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. In `config.yaml`:
- `exchange_name: mt5`
- fill `mt5.login`
- fill `mt5.password`
- fill `mt5.server`
- optionally set `mt5.path` to your `terminal64.exe`
- set `mt5.symbol_suffix` / `mt5.symbol_map` to match your Market Watch symbols

3. Put the symbols you want the bot to scan in `symbols:`.
Examples:
- `XAU/USD`
- `EURUSD`
- `BTC/USDT:USDT`

Notes:
- MT5 terminal must be installed on Windows and logged in to the target account.
- `strict_mode: true` stops the bot if MT5 is unavailable.
- The bot keeps using the same internal signal logic; only the market data source changes.

## MT5 Real32 Bridge (Optional)

If you want signals/outcomes to match your MT5 account feed:

1. Run bridge server on the same machine that has MT5 terminal logged in:
```bash
python mt5_real32_bridge_server.py
```

2. Set bridge token in `.env` (optional but recommended):
```bash
MT5_BRIDGE_TOKEN=your_shared_token
```

3. In `config.yaml`:
- `exchange_name: mt5_bridge`
- set `mt5_bridge.base_url` to bridge URL (docker on Mac can use `http://host.docker.internal:18080`)
- fill `mt5_bridge.symbol_map` according to your Market Watch symbol names on Real32

Note:
- `strict_mode: true` -> bot stops if bridge is down.
- `strict_mode: false` -> bot falls back to `fallback_exchange`.

## Quick Start

### 1. Install Dependencies

Using Docker (recommended):
```bash
cd /Users/adel/trading
docker-compose build
docker-compose run --rm trading bash
```

Or using pip:
```bash
cd /Users/adel/trading/crypto_signal_bot
pip install -r requirements.txt
```

### 2. Configure Telegram (Optional)

Set environment variables:
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

Or create a `.env` file in the project directory.

Telegram runtime control commands (when bot is running):
```text
/pause   -> Pause scanning loop
/resume  -> Resume scanning
/status  -> Show runtime status
/help    -> Show commands
/stop    -> Stop bot process
```

### 3. Run the Bot

Single scan (test mode):
```bash
python main.py --once
```

Continuous scanning:
```bash
python main.py
```

With verbose logging:
```bash
python main.py -v
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Symbols to scan
symbols:
  - "BTC/USDT:USDT"
  - "ETH/USDT:USDT"
  - "SOL/USDT:USDT"

# Scoring threshold
scoring:
  base_threshold: 7
  add_timing_to_score: true
  max_timing_points_used: 4

# Risk parameters
risk:
  buffer_pct: 0.15   # SL buffer
  rr_tp1: 1.0        # TP1 at 1:1 R:R
  rr_tp2: 2.0        # TP2 at 1:2 R:R
```

## Signal Scoring

### Technical Score (max ~11 points)
| Factor | Points |
|--------|--------|
| Trend match | +2 |
| Price in S/R zone | +2 |
| Candle pattern | +2 |
| Volume spike | +2 |
| Wave 3 setup | +2 |
| RSI divergence against | -2 |

### Timing Score (max 4 points, capped)
| Factor | Points |
|--------|--------|
| Gann angle confluence | 0-2 |
| Square of 9 proximity | 0-2 |
| 52-cycle window | 0-1 |
| Lunar event window | 0-1 |
| FOMC high-vol window | -1-0 (filter) |

A signal is generated when `total_score >= threshold` (default: 7).

## Example Telegram Message

```
🟢 LONG Signal: BTC/USDT:USDT

📊 Score: 9.5/7 (Tech: 8, Timing: 1.5)
📈 Trend: UP
💰 Price: 97,245.50

━━━━ Levels ━━━━
▫️ Entry: 97,250.00
🛑 Stop Loss: 95,800.00 (1.49%)
🎯 TP1: 98,700.00 (1:1)
🎯 TP2: 100,150.00 (1:2)

━━━━ Technical Analysis ━━━━
  ✓ Trend: up
  ✓ At support zone
  ✓ Bullish Engulfing
  ✓ Volume spike

━━━━ Timing Analysis ━━━━
  Gann: above_angles (score: 1.0)
  Sq9: 97,344.00 (0.10%, score: 0.5)

⏱ Timeframes: 4h/15m/1h
🕐 2026-02-04 19:15 UTC
```

## Running Tests

```bash
cd /Users/adel/trading/crypto_signal_bot
pytest tests/ -v
```

## Important Notes

- **No auto-trading**: This bot generates signals only
- **No API keys needed**: Uses public Binance endpoints
- **Public data only**: OHLCV and ticker data via ccxt
- **FOMC data**: Fetched from federalreserve.gov and cached for 7 days

## Troubleshooting

### Rate Limiting
The bot includes automatic rate limiting and exponential backoff. If you see rate limit errors, increase `scan_interval_seconds`.

### Missing Lunar Data
Install the `ephem` package for lunar phase analysis:
```bash
pip install ephem
```

### Telegram Not Working
1. Check environment variables are set
2. Verify bot token with BotFather
3. Ensure chat_id is correct (use @userinfobot)

## License

MIT License
