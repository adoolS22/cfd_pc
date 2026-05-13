import sqlite3

def check_db():
    conn = sqlite3.connect('y:/trading/trading/crypto_signal_bot/signals.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    print("--- ALL TIME PERFORMANCE ---")
    rows = c.execute('SELECT outcome, COUNT(*) as cnt, AVG(pnl_pct) as avg_pnl, SUM(pnl_pct) as sum_pnl FROM signal_outcomes GROUP BY outcome').fetchall()
    total_pnl = 0
    total_trades = 0
    for r in rows:
        print(dict(r))
        if r['outcome'] != 'OPEN':
            total_pnl += r['sum_pnl'] or 0
            total_trades += r['cnt']
            
    print(f"\\nTotal Signals: {total_trades} | Total PnL: {total_pnl:.2f}%")
    
    print("\\n--- RECENT 10 SIGNALS ---")
    recent = c.execute('SELECT id, symbol, side, outcome, pnl_pct FROM signal_outcomes ORDER BY id DESC LIMIT 10').fetchall()
    for r in recent:
        print(dict(r))

if __name__ == '__main__':
    check_db()
