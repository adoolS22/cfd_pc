import sqlite3

def check_open():
    conn = sqlite3.connect('signals.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    rows = c.execute("""
        SELECT so.id, so.symbol, so.side, so.entry, s.timestamp AS signal_timestamp, so.outcome 
        FROM signal_outcomes so
        LEFT JOIN signals s ON s.id = so.signal_id
        WHERE so.outcome = 'OPEN' 
        ORDER BY so.id DESC
    """).fetchall()
    print(f"Total OPEN signals: {len(rows)}")
    for r in rows:
        print(dict(r))
    conn.close()

if __name__ == '__main__':
    check_open()
