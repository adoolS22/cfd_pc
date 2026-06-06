import sqlite3

conn = sqlite3.connect('signals.db')
cur = conn.cursor()

tables = ['signals', 'signal_outcomes', 'llm_trade_reviews', 'pending_entries', 'rejected_signals']
for table in tables:
    cur.execute(f"DELETE FROM {table}")
    print(f"Cleared {table}")

conn.commit()
conn.close()
print("All historical trades wiped. Blank slate ready.")
