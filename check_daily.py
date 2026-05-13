import sqlite3
from datetime import datetime, timedelta

conn = sqlite3.connect('y:/trading/trading/crypto_signal_bot/signals.db')
conn.row_factory = sqlite3.Row
c = conn.cursor()

print("--- LAST 24-48 HOURS PERFORMANCE ---")
# Get stats from the last 2 days
rows = c.execute('''
    SELECT date(closed_at) as day, outcome, COUNT(*) as cnt, SUM(pnl_pct) as sum_pnl 
    FROM signal_outcomes 
    WHERE closed_at >= date('now', '-2 day')
    GROUP BY day, outcome
    ORDER BY day DESC, outcome
''').fetchall()

current_day = None
for r in rows:
    if r['day'] != current_day:
        print(f"\\n[{r['day']}]")
        current_day = r['day']
    print(f"  {r['outcome']:12} | Count: {r['cnt']:<3} | PnL: {r['sum_pnl'] or 0:.2f}%")
