import sqlite3
import json

def run_stats():
    conn = sqlite3.connect('signals.db')
    cur = conn.cursor()

    cur.execute("SELECT outcome, COUNT(*), AVG(pnl_pct) FROM signal_outcomes GROUP BY outcome")
    old_rows = cur.fetchall()
    
    avg_loss_pct = 0.0
    avg_win_pct = 0.0
    total_losses = 0
    total_wins = 0
    total_loss_pnl = 0.0
    total_win_pnl = 0.0
    
    for row in old_rows:
        outcome, count, avg_pnl = row
        if not avg_pnl: continue
        if outcome in ['SL_HIT', 'LOSS']:
            total_losses += count
            total_loss_pnl += (count * avg_pnl)
        elif outcome in ['TP1_HIT', 'TP2_HIT', 'TP_NEAR_HIT', 'WIN']:
            total_wins += count
            total_win_pnl += (count * avg_pnl)
            
    avg_loss_pct = total_loss_pnl / total_losses if total_losses else -1.0
    avg_win_pct = total_win_pnl / total_wins if total_wins else 1.0

    cur.execute("SELECT outcome, COUNT(*) FROM rejected_signals GROUP BY outcome")
    shadow_rows = cur.fetchall()
    
    avoided_losses = 0
    missed_wins = 0
    
    for row in shadow_rows:
        outcome, count = row
        if outcome == 'SHADOW_SL':
            avoided_losses += count
        elif outcome in ['SHADOW_TP1', 'SHADOW_TP2', 'SHADOW_TP_NEAR']:
            missed_wins += count
            
    pnl_saved_from_losses = avoided_losses * abs(avg_loss_pct)
    pnl_missed_from_wins = missed_wins * avg_win_pct
    net_pnl_impact = pnl_saved_from_losses - pnl_missed_from_wins

    res = {
        "avg_loss": avg_loss_pct,
        "avg_win": avg_win_pct,
        "avoided_losses": avoided_losses,
        "pnl_saved": pnl_saved_from_losses,
        "missed_wins": missed_wins,
        "pnl_missed": pnl_missed_from_wins,
        "net_impact": net_pnl_impact
    }
    
    print(json.dumps(res))

if __name__ == '__main__':
    run_stats()
