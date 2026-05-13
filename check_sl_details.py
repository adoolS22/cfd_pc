import sqlite3

conn = sqlite3.connect('y:/trading/trading/crypto_signal_bot/signals.db')
conn.row_factory = sqlite3.Row
c = conn.cursor()

print("--- RECENT SL_HIT DETAILS ---")
rows = c.execute('''
    SELECT symbol, side, entry, stop_loss, take_profit_near, take_profit_2, pnl_pct, closed_at 
    FROM signal_outcomes 
    WHERE outcome="SL_HIT" AND date(closed_at) >= date('now', '-1 day')
    ORDER BY closed_at DESC LIMIT 10
''').fetchall()

for r in rows:
    pct_sl = abs(r['entry'] - r['stop_loss']) / r['entry'] * 100
    pct_tp = abs(r['entry'] - (r['take_profit_2'] or r['take_profit_near'])) / r['entry'] * 100
    print(f"{r['symbol']} {r['side']} | Entry: {r['entry']} | SL: {r['stop_loss']} (Risk: {pct_sl:.2f}%) | TP: {r['take_profit_2']} (Reward: {pct_tp:.2f}%) | PNL: {r['pnl_pct']:.2f}% | Closed: {r['closed_at']}")

