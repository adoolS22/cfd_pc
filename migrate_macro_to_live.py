import sqlite3

def migrate_macro_trades():
    src = sqlite3.connect('signals.db.bak')
    dst = sqlite3.connect('signals.db')
    
    src.row_factory = sqlite3.Row
    trades = src.execute("""
        SELECT * FROM signal_outcomes 
        WHERE symbol LIKE '%XAU%' OR symbol LIKE '%XAG%' OR symbol LIKE '%USOIL%' 
        OR symbol LIKE '%SNP500%' OR symbol LIKE '%SPX500%' OR symbol LIKE '%EURUSD%'
        OR symbol LIKE '%US30%' OR symbol LIKE '%USTEC%'
    """).fetchall()
    
    if not trades:
        print("No macro trades found.")
        return
        
    print(f"Found {len(trades)} macro trades in backup database.")
    
    # Get columns from current database
    dst.row_factory = sqlite3.Row
    dst_columns = [col['name'] for col in dst.execute("PRAGMA table_info(signal_outcomes)").fetchall()]
    
    count = 0
    for t in trades:
        # Check if already exists
        exists = dst.execute("SELECT 1 FROM signal_outcomes WHERE signal_id = ?", (t['signal_id'],)).fetchone()
        if exists:
            continue
            
        # Build insert
        row_dict = dict(t)
        cols_to_insert = [c for c in row_dict.keys() if c in dst_columns]
        placeholders = ', '.join(['?'] * len(cols_to_insert))
        cols_str = ', '.join(cols_to_insert)
        vals = [row_dict[c] for c in cols_to_insert]
        
        dst.execute(f"INSERT INTO signal_outcomes ({cols_str}) VALUES ({placeholders})", vals)
        count += 1
        
    dst.commit()
    print(f"Migrated {count} macro trades to the live database!")
    
    src.close()
    dst.close()

if __name__ == '__main__':
    migrate_macro_trades()
