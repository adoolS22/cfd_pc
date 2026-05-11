import sqlite3

conn = sqlite3.connect('signals.db')
cur = conn.cursor()

# Last 5 closed trades
cur.execute("""
    SELECT symbol, side, outcome, ROUND(pnl_pct,3), closed_at
    FROM signal_outcomes
    WHERE outcome != 'OPEN'
    ORDER BY closed_at DESC
    LIMIT 5
""")
print('=== LAST 5 CLOSED TRADES ===')
for r in cur.fetchall():
    print(' ', r)

# Learning stats per symbol/side
cur.execute("""
    SELECT symbol, side, COUNT(*) as cnt,
    ROUND(AVG(CASE WHEN outcome IN ('TP1_HIT','TP2_HIT','TP_NEAR_HIT','EXITED','TRAIL_HIT') THEN 1.0 ELSE 0.0 END)*100,1) as winrate,
    ROUND(AVG(pnl_pct),3) as avg_pnl
    FROM signal_outcomes
    WHERE outcome != 'OPEN'
    GROUP BY symbol, side
    ORDER BY cnt DESC
    LIMIT 12
""")
print('\n=== LEARNING STATS (symbol, side, trades, winrate%, avg_pnl%) ===')
for r in cur.fetchall():
    print(' ', r)

# Overall summary
cur.execute("""
    SELECT COUNT(*) as total,
    ROUND(AVG(CASE WHEN outcome IN ('TP1_HIT','TP2_HIT','TP_NEAR_HIT','EXITED','TRAIL_HIT') THEN 1.0 ELSE 0.0 END)*100,1) as winrate,
    ROUND(AVG(pnl_pct),3) as avg_pnl,
    ROUND(SUM(pnl_pct),2) as total_pnl
    FROM signal_outcomes
    WHERE outcome != 'OPEN'
""")
print('\n=== OVERALL SUMMARY (total, winrate%, avg_pnl%, total_pnl%) ===')
print(' ', cur.fetchone())

# LLM reviews
cur.execute("SELECT COUNT(*) FROM llm_trade_reviews")
print(f'\n=== LLM REVIEWS: {cur.fetchone()[0]} reviews ===')

conn.close()
