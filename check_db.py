import sqlite3
import pandas as pd

def check_db():
    conn = sqlite3.connect('signals.db')
    
    print("--- Outcomes ---")
    try:
        df = pd.read_sql("SELECT outcome, COUNT(*) as count FROM signal_outcomes GROUP BY outcome", conn)
        print(df)
    except Exception as e:
        print("Error reading outcomes:", e)

    print("\n--- Recent Closed Trades ---")
    try:
        df2 = pd.read_sql("SELECT symbol, side, outcome, pnl_pct, closed_at FROM signal_outcomes WHERE outcome != 'OPEN' ORDER BY closed_at DESC LIMIT 10", conn)
        print(df2)
    except Exception as e:
        print("Error reading recent trades:", e)
        
    print("\n--- Rejections ---")
    try:
        df3 = pd.read_sql("SELECT rejection_source, COUNT(*) as count FROM rejected_signals GROUP BY rejection_source", conn)
        print(df3)
    except Exception as e:
        print("Error reading rejections:", e)
        
    conn.close()

if __name__ == "__main__":
    check_db()
