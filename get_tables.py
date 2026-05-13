import sqlite3
conn = sqlite3.connect("y:/trading/trading/crypto_signal_bot/signals.db")
print(conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())