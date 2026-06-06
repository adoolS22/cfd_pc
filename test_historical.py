import sqlite3
import pandas as pd
from bot.utils import load_config
from bot.yahoo_data import fetch_yahoo_ohlcv
from bot.signals import analyze_symbol

config = load_config('config.yaml')
config.openai.enabled = False

conn = sqlite3.connect('signals.db')
cur = conn.cursor()
# Get last 3 signals sent before today
cur.execute("SELECT symbol, side, timestamp FROM signals ORDER BY timestamp DESC LIMIT 3")
old_signals = cur.fetchall()

print("=== Backtesting Old Trades with New SMC Logic ===")
for row in old_signals:
    symbol_full, original_side, timestamp_str = row
    symbol = symbol_full.split('_')[0].replace('/', '-')
    
    # We need to fetch data ending exactly at timestamp_str
    print(f"\n[+] Testing old trade: {symbol} | Original Side: {original_side} | Time: {timestamp_str}")
    
    try:
        # Convert timestamp to pandas datetime and fetch data up to that point
        end_time = pd.to_datetime(timestamp_str)
        # Yahoo finance might be tricky with exact historical intraday limits, but we try
        df_trend = fetch_yahoo_ohlcv(symbol, timeframe=config.trend_tf, limit=100)
        df_entry = fetch_yahoo_ohlcv(symbol, timeframe=config.entry_tf, limit=100)
        df_sr = fetch_yahoo_ohlcv(symbol, timeframe=config.sr_tf, limit=100)
        
        # Filter data up to the timestamp
        if df_trend is not None and not df_trend.empty:
            df_trend = df_trend[df_trend.index <= end_time]
        if df_entry is not None and not df_entry.empty:
            df_entry = df_entry[df_entry.index <= end_time]
        if df_sr is not None and not df_sr.empty:
            df_sr = df_sr[df_sr.index <= end_time]
            
        if df_entry is None or len(df_entry) < 20:
            print("  -> Not enough historical data available for this timestamp.")
            continue
            
        ticker = {"last": float(df_entry["close"].iloc[-1])}
        
        # Run SMC Gate
        result = analyze_symbol(
            symbol=symbol_full.replace('_futures', '').replace('-', '/'),
            df_trend=df_trend,
            df_entry=df_entry,
            df_sr=df_sr,
            ticker=ticker,
            config=config,
        )
        
        print(f"  -> New Logic Decision: {result.side} (Valid: {result.is_valid})")
        print(f"  -> Score: {result.total_score}")
        print("  -> Reasons:")
        for r in result.reasons:
            print(f"     * {str(r).replace('✓', '[OK]').encode('ascii', errors='ignore').decode('ascii')}")
            
    except Exception as e:
        print(f"  -> Error backtesting: {e}")

