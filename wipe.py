import sqlite3; db=sqlite3.connect('signals.db'); db.execute('DELETE FROM llm_trade_reviews'); db.commit(); print('LEARNING MEMORY WIPED.'); db.close()
