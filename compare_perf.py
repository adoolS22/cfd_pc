import sqlite3
import pandas as pd

conn = sqlite3.connect('signals.db')

print("=== ابريل 2026 - أداء كل رمز ===")
df = pd.read_sql_query("""
SELECT symbol, COUNT(*) as count,
    SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
    ROUND(100.0*SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END)/COUNT(*),1) as win_pct,
    ROUND(SUM(pnl_pct),3) as total_pnl
FROM signal_outcomes
WHERE strftime('%Y-%m', closed_at) = '2026-04'
AND pnl_pct IS NOT NULL
GROUP BY symbol ORDER BY count DESC LIMIT 15
""", conn)
print(df)

print("\n=== قبل وبعد تعديلات 23 ابريل ===")
df2 = pd.read_sql_query("""
SELECT 
    CASE WHEN closed_at < '2026-04-23' THEN 'Before_Edit' ELSE 'After_Edit' END as period,
    COUNT(*) as total,
    SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
    ROUND(100.0*SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END)/COUNT(*),1) as win_pct,
    ROUND(AVG(pnl_pct),4) as avg_pnl,
    ROUND(SUM(pnl_pct),3) as total_pnl
FROM signal_outcomes
WHERE strftime('%Y-%m', closed_at) = '2026-04'
AND pnl_pct IS NOT NULL
GROUP BY period
""", conn)
print(df2)

conn.close()
