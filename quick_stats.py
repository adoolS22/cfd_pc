import sqlite3, datetime
conn = sqlite3.connect('signals.db')
c = conn.cursor()

WIN_SET = ('TP1_HIT','TP2_HIT','TP_NEAR_HIT','BE_HIT','TRAIL_HIT')
LOSS_SET = ('SL_HIT','EXITED','EXPIRED')

def show_period(label, cutoff=None):
    where = 'AND s.timestamp > ?' if cutoff else ''
    params = (cutoff,) if cutoff else ()
    rows = c.execute(
        'SELECT o.outcome, COUNT(*), COALESCE(AVG(o.pnl_pct),0), COALESCE(SUM(o.pnl_pct),0) '
        'FROM signal_outcomes o JOIN signals s ON o.signal_id=s.id '
        'WHERE o.outcome NOT IN (''OPEN'') ' + where +
        ' GROUP BY o.outcome', params
    ).fetchall()
    total=wins=pnl=0
    print()
    print('=== ' + label + ' ===')
    for r in rows:
        n, avg, s = r[1], r[2], r[3]
        total += n
        if r[0] in WIN_SET: wins += n
        pnl += s
        tag = 'WIN ' if r[0] in WIN_SET else 'LOSS'
        print(f'  {tag}  {r[0]:<15} {n:>5} trades  avg={avg:+.3f}%  sum={s:+.2f}%')
    if total > 0:
        print()
        print(f'  TOTAL: {total} trades | WR: {wins/total*100:.1f}% | Net PnL: {pnl:+.2f}% | avg/trade: {pnl/total:+.3f}%')
    sym_rows = c.execute(
        'SELECT o.symbol, COUNT(*), SUM(CASE WHEN o.outcome IN (''TP1_HIT'',''TP2_HIT'',''TP_NEAR_HIT'',''BE_HIT'',''TRAIL_HIT'') THEN 1 ELSE 0 END), COALESCE(SUM(o.pnl_pct),0) '
        'FROM signal_outcomes o JOIN signals s ON o.signal_id=s.id '
        'WHERE o.outcome NOT IN (''OPEN'') ' + where +
        ' GROUP BY o.symbol ORDER BY COALESCE(SUM(o.pnl_pct),0) DESC', params
    ).fetchall()
    print()
    for s in sym_rows:
        wr = s[2]/s[1]*100 if s[1] else 0
        m = '+' if s[3] > 0 else '-'
        print(f'  {m} {s[0]:<26} {s[1]:>4} trades  WR={wr:.0f}%  PnL={s[3]:+.2f}%')

d30 = (datetime.datetime.now()-datetime.timedelta(days=30)).isoformat()
d7  = (datetime.datetime.now()-datetime.timedelta(days=7)).isoformat()
show_period('ALL TIME')
show_period('LAST 30 DAYS', d30)
show_period('LAST 7 DAYS', d7)
conn.close()
