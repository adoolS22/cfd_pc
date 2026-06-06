import sqlite3

conn = sqlite3.connect('signals.db')
cur = conn.cursor()

print("--- OLD LOGIC (Real Live Trades) ---")
try:
    cur.execute("SELECT outcome, COUNT(*), AVG(pnl_pct) FROM signal_outcomes GROUP BY outcome")
    rows = cur.fetchall()
    total_trades = 0
    wins = 0
    losses = 0
    breakeven = 0
    total_pnl = 0.0
    for row in rows:
        outcome, count, avg_pnl = row
        print(f"Outcome: {outcome}, Count: {count}, Avg PnL: {avg_pnl}")
        total_trades += count
        if outcome in ['TP1', 'TP2', 'TP_NEAR', 'WIN']:
            wins += count
        elif outcome in ['SL', 'LOSS']:
            losses += count
        else:
            breakeven += count
        
        if avg_pnl:
            total_pnl += (count * avg_pnl)
            
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    print(f"\nTotal Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total PnL %: {total_pnl:.2f}%")
except Exception as e:
    print("Error:", e)

print("\n--- NEW LOGIC (Shadow Tracking / Rejected Signals) ---")
try:
    cur.execute("SELECT outcome, COUNT(*) FROM rejected_signals GROUP BY outcome")
    rows = cur.fetchall()
    total_rejected = 0
    saved_from_loss = 0
    missed_win = 0
    expired = 0
    for row in rows:
        outcome, count = row
        print(f"Outcome: {outcome}, Count: {count}")
        total_rejected += count
        if outcome == 'SHADOW_SL':
            saved_from_loss += count
        elif outcome in ['SHADOW_TP1', 'SHADOW_TP2', 'SHADOW_TP_NEAR']:
            missed_win += count
        else:
            expired += count
            
    print(f"\nTotal Rejected: {total_rejected}")
    if total_rejected > 0:
        print(f"Saved from Loss (Good Rejections): {saved_from_loss} ({(saved_from_loss/total_rejected)*100:.1f}%)")
        print(f"Missed Wins (Bad Rejections): {missed_win} ({(missed_win/total_rejected)*100:.1f}%)")
        print(f"Expired/Sideways (Good Rejections): {expired} ({(expired/total_rejected)*100:.1f}%)")
except Exception as e:
    print("Error:", e)

