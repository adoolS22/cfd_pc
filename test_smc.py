import yaml
from bot.utils import load_config
from bot.yahoo_data import fetch_yahoo_ohlcv
from bot.signals import analyze_symbol

config = load_config('config.yaml')
config.openai.enabled = False

def run():
    print("Fetching BTC data...")
    df_trend = fetch_yahoo_ohlcv("BTC-USD", timeframe=config.trend_tf, limit=100)
    df_entry = fetch_yahoo_ohlcv("BTC-USD", timeframe=config.entry_tf, limit=100)
    df_sr = fetch_yahoo_ohlcv("BTC-USD", timeframe=config.sr_tf, limit=100)
    
    ticker = {"last": float(df_entry["close"].iloc[-1])}
    
    print("\n[ Analyzing Current BTC Market with SMC ]")
    result = analyze_symbol(
        symbol="BTC/USDT",
        df_trend=df_trend,
        df_entry=df_entry,
        df_sr=df_sr,
        ticker=ticker,
        config=config,
    )
    
    print(f"Price: {ticker['last']}")
    print(f"Signal: {result.side} (Valid: {result.is_valid})")
    print(f"Score: {result.total_score}")
    print("\nReasons:")
    for r in result.reasons:
        print(f" - {r}")
        
    if result.risk_levels:
        print("\nRisk Parameters:")
        print(f"Entry: {result.risk_levels.entry}")
        print(f"Stop Loss: {result.risk_levels.stop_loss} ({abs(result.risk_levels.entry - result.risk_levels.stop_loss)/result.risk_levels.entry*100:.2f}%)")
        print(f"TP1: {result.risk_levels.take_profit_1}")
        print(f"TP2: {result.risk_levels.take_profit_2}")
        print(f"TP Near: {result.risk_levels.take_profit_near}")

if __name__ == "__main__":
    run()
