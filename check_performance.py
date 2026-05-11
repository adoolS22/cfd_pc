import sqlite3
import pandas as pd

conn = sqlite3.connect('signals.db')

print("=== نتائج كل الصفقات بقاعدة البيانات ===")
df = pd.read_sql_query("""
SELECT outcome, COUNT(*) as count, ROUND(AVG(pnl_pct),4) as avg_pnl
FROM signal_outcomes
WHERE pnl_pct IS NOT NULL
GROUP BY outcome
ORDER BY count DESC
""", conn)
print(df)

print("\n=== ملخص الأداء الكلي ===")
df2 = pd.read_sql_query("""
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN pnl_pct < 0 THEN 1 ELSE 0 END) as losses,
    ROUND(AVG(pnl_pct),4) as avg_pnl_pct,
    ROUND(SUM(pnl_pct),4) as total_pnl_pct
FROM signal_outcomes
WHERE outcome IN ('TP1', 'TP2', 'SL', 'WIN', 'LOSS', 'TP')
AND pnl_pct IS NOT NULL
""", conn)
print(df2)

print("\n=== الأداء الشهري (آخر 6 أشهر) ===")
df3 = pd.read_sql_query("""
SELECT strftime('%Y-%m', closed_at) as month,
    COUNT(*) as total,
    SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
    ROUND(AVG(pnl_pct),4) as avg_pnl,
    ROUND(SUM(pnl_pct),4) as total_pnl
FROM signal_outcomes
WHERE pnl_pct IS NOT NULL AND closed_at IS NOT NULL
GROUP BY month ORDER BY month DESC LIMIT 6
""", conn)
print(df3)

print("\n=== أحسن 5 رموز أداءً ===")
df4 = pd.read_sql_query("""
SELECT symbol,
    COUNT(*) as total,
    SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
    ROUND(100.0*SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END)/COUNT(*),1) as win_rate_pct,
    ROUND(SUM(pnl_pct),4) as total_pnl
FROM signal_outcomes
WHERE pnl_pct IS NOT NULL
GROUP BY symbol HAVING total > 5
ORDER BY win_rate_pct DESC LIMIT 5
""", conn)
print(df4)

print("\n=== أسوأ 5 رموز أداءً ===")
df5 = pd.read_sql_query("""
SELECT symbol,
    COUNT(*) as total,
    SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
    ROUND(100.0*SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END)/COUNT(*),1) as win_rate_pct,
    ROUND(SUM(pnl_pct),4) as total_pnl
FROM signal_outcomes
WHERE pnl_pct IS NOT NULL
GROUP BY symbol HAVING total > 5
ORDER BY win_rate_pct ASC LIMIT 5
""", conn)
print(df5)

conn.close()
