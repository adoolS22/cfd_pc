import sqlite3
conn = sqlite3.connect('signals.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables in signals.db:", cur.fetchall())

try:
    cur.execute("SELECT symbol, side, created_at, outcome FROM signals ORDER BY created_at DESC LIMIT 5")
    print("\nRecent Signals:")
    for row in cur.fetchall():
        print(row)
except Exception as e:
    print("Error querying signals:", e)

try:
    cur.execute("SELECT symbol, side, created_at, outcome FROM rejected_signals ORDER BY created_at DESC LIMIT 5")
    print("\nRecent Rejected:")
    for row in cur.fetchall():
        print(row)
except Exception as e:
    print("Error querying rejected:", e)
