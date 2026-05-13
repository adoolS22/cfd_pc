import sqlite3
import json

conn = sqlite3.connect('y:/trading/trading/crypto_signal_bot/signals.db')
conn.row_factory = sqlite3.Row
c = conn.cursor()

# Get recent trades with reasons
rows = c.execute('''
    SELECT o.symbol, o.side, o.outcome, o.pnl_pct, o.closed_at, s.reasons
    FROM signal_outcomes o
    JOIN signals s ON o.signal_id = s.id
    WHERE date(o.closed_at) >= date('now', '-2 day')
    ORDER BY o.closed_at DESC
    LIMIT 10
''').fetchall()

print("--- RECENT TRADES REASONS ---")
for r in rows:
    print(f"\\n[{r['closed_at']}] {r['symbol']} {r['side']} | Outcome: {r['outcome']} ({r['pnl_pct']}%)")
    try:
        # Reasons might be JSON string
        reasons = json.loads(r['reasons']) if r['reasons'] else "No reasons logged"
        if isinstance(reasons, list):
            for res in reasons:
                print(f"  - {res}")
        else:
            print(f"  - {reasons}")
    except Exception as e:
        print(f"  - {r['reasons']}")
