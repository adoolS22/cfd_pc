import sys
import io
# Force UTF-8 encoding on standard streams to prevent Windows console encoding crashes
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import yaml
from bot.utils import load_config
from bot.yahoo_data import fetch_yahoo_ohlcv
from bot.signals import analyze_symbol
from bot.expert_advisor import get_expert_trade_opinion
from bot.risk import RiskLevels

config = load_config('config.yaml')

# Force OpenAI adapter to use local Ollama
config.openai.enabled = True
config.openai.api_key = "ollama"
config.openai.model = "qwen2.5:7b"
config.openai.base_url = "http://localhost:11434/v1"
config.time_analysis.expert_advisor = True

def run():
    print("=" * 80)
    print("  🧪 اختبار رأي الخبير الفني المباشر (qwen2.5:7b)  ")
    print("  يتم الآن جلب بيانات البيتكوين الفعلية وتحليلها وفقاً لمنهجيتك الجديدة")
    print("=" * 80)
    
    print("\nFetching BTC-USD data...")
    df_trend = fetch_yahoo_ohlcv("BTC-USD", timeframe=config.trend_tf, limit=100)
    df_entry = fetch_yahoo_ohlcv("BTC-USD", timeframe=config.entry_tf, limit=100)
    df_sr = fetch_yahoo_ohlcv("BTC-USD", timeframe=config.sr_tf, limit=100)
    
    ticker = {"last": float(df_entry["close"].iloc[-1])}
    
    print("Analyzing current market...")
    result = analyze_symbol(
        symbol="BTC/USDT",
        df_trend=df_trend,
        df_entry=df_entry,
        df_sr=df_sr,
        ticker=ticker,
        config=config,
    )
    
    print(f"\n[+] السعر الحالي: {ticker['last']}")
    print(f"[+] إشارة التحليل الفني: {result.side} (Valid: {result.is_valid})")
    
    # Now check the expert opinion from Qwen!
    if result.expert_opinion:
        op = result.expert_opinion
        print("\n" + "=" * 60)
        print(" 🧠 قرار ورأي الخبير الذكي (منهجية SMC الجديدة):")
        print("=" * 60)
        print(f"القرار (Decision)  : {op.get('decision')}")
        print(f"الثقة (Confidence)  : {op.get('confidence')}%")
        print(f"التبرير الفني (Rationale):\n{op.get('rationale')}")
        
        if op.get('stop_loss'):
            print(f"وقف الخسارة (SL)  : {op.get('stop_loss')}")
        if op.get('take_profit_1'):
            print(f"الهدف الأول (TP1)  : {op.get('take_profit_1')}")
        if op.get('take_profit_2'):
            print(f"الهدف الثاني (TP2)  : {op.get('take_profit_2')}")
        print("=" * 60)
    else:
        print("\n[-] لم يتم الحصول على رأي الخبير. يرجى التحقق من تشغيل سيرفر Ollama.")

if __name__ == "__main__":
    run()
