#!/usr/bin/env python3
"""
Performance Analysis Dashboard
================================
Queries signals.db and prints a comprehensive performance report.
Run with: python analyze_performance.py

Covers:
  1. Overall performance overview
  2. Pre vs Post improvement comparison (before/after Apr 8 2026)
  3. Win rate by hour of day (validates session filter)
  4. Per-symbol breakdown
  5. Score bucket analysis (validates scoring engine)
  6. R:R analysis (validates TP target changes)
  7. Day-of-week patterns
"""

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple

DB_PATH = str(Path(__file__).resolve().parent / "signals.db")
IMPROVEMENT_DATE = "2026-04-08T19:00:00+00:00"  # When we applied the big improvements

WIN_OUTCOMES  = {"TP_NEAR_HIT", "TP1_HIT", "TP2_HIT", "TRAIL_HIT", "BE_HIT"}
LOSS_OUTCOMES = {"SL_HIT"}
COST_PCT      = 0.08  # estimated round-trip cost %
LOT_SIZE      = 0.1   # requested simulation size
SPREAD_PROFILE_NAME = "exness_islamic"

# Approximate contract units per 1.0 lot.
# IMPORTANT: adjust these to match your broker/platform spec.
CONTRACT_UNITS_PER_1_LOT = {
    "EURUSD": 100000.0,
    "XAUUSD": 100.0,
    "XAGUSD": 5000.0,
    "OILUSD": 1000.0,
    "SNP500": 1.0,
    "US500": 1.0,
    "US30": 1.0,
    "USTEC": 1.0,
}
# For crypto futures we treat 1.0 lot as 1.0 coin by default.
CRYPTO_UNITS_PER_1_LOT = 1.0

# Approximate one-way spread (ask-bid) in price units for Exness Islamic.
# IMPORTANT: tune to your live account average spread.
SPREAD_PRICE_BY_SYMBOL = {
    "EURUSD": 0.00012,   # ~1.2 pip
    "XAUUSD": 0.30,      # $0.30
    "XAGUSD": 0.035,     # $0.035
    "OILUSD": 0.04,      # $0.04
    "SNP500": 0.80,      # index points
    "US500": 0.80,
    "US30": 4.00,
    "USTEC": 1.50,
    "BTC": 8.00,
    "ETH": 0.80,
    "SOL": 0.04,
    "ARKM": 0.00030,
    "PYTH": 0.00005,
    "ORDI": 0.02000,
    "VAI": 0.00002,
}
# Fallback spread as percentage of entry when symbol is not configured above.
DEFAULT_SPREAD_PCT_CRYPTO = 0.03
DEFAULT_SPREAD_PCT_MACRO = 0.01

SEP = "─" * 65


def q(conn, sql, params=()):
    return conn.execute(sql, params).fetchall()


def pct(n, d):
    return f"{100*n/d:.1f}%" if d else "N/A"


def _fmt_price(value):
    if value is None:
        return "-"
    try:
        num = float(value)
    except Exception:
        return "-"
    if abs(num) >= 1000:
        return f"{num:,.2f}"
    if abs(num) >= 1:
        return f"{num:.4f}"
    return f"{num:.6f}"


def _canonical_symbol(raw_symbol: str) -> str:
    text = str(raw_symbol or "").strip().upper()
    text = text.replace("_FUTURES", "")
    text = text.replace("/USDT:USDT", "")
    text = text.replace("-USDT", "")
    text = text.replace("/", "")
    return text


def _is_crypto_symbol(raw_symbol: str) -> bool:
    s = str(raw_symbol or "").upper().replace("_FUTURES", "")
    return "/USDT" in s or "-USDT" in s or s.endswith("USDT")


def _resolve_units_per_lot(raw_symbol: str) -> Tuple[Optional[float], str]:
    key = _canonical_symbol(raw_symbol)
    units_per_lot = CONTRACT_UNITS_PER_1_LOT.get(key)
    if units_per_lot is None and _is_crypto_symbol(raw_symbol):
        units_per_lot = CRYPTO_UNITS_PER_1_LOT
    return units_per_lot, key


def _estimate_trade_pnl_for_lot(
    *,
    symbol: str,
    side: str,
    entry: Optional[float],
    close_price: Optional[float],
    lot_size: float,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Return:
      (pnl_usd, units_used, note)
    """
    if entry is None or close_price is None:
        return None, None, "missing_price"

    try:
        entry_f = float(entry)
        close_f = float(close_price)
    except Exception:
        return None, None, "bad_price"

    if entry_f <= 0:
        return None, None, "bad_entry"

    side_u = str(side or "").upper()
    if side_u == "LONG":
        move_per_unit = close_f - entry_f
    elif side_u == "SHORT":
        move_per_unit = entry_f - close_f
    else:
        return None, None, "bad_side"

    units_per_lot, key = _resolve_units_per_lot(symbol)
    if units_per_lot is None:
        return None, None, f"missing_contract:{key}"

    qty = float(lot_size) * float(units_per_lot)
    pnl = move_per_unit * qty
    return pnl, qty, "ok"


def _estimate_spread_cost_for_lot(
    *,
    symbol: str,
    entry: Optional[float],
    lot_size: float,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Return:
      (spread_cost_usd, units_used, note)
    """
    if entry is None:
        return None, None, "missing_entry"
    try:
        entry_f = float(entry)
    except Exception:
        return None, None, "bad_entry"
    if entry_f <= 0:
        return None, None, "bad_entry"

    units_per_lot, key = _resolve_units_per_lot(symbol)
    if units_per_lot is None:
        return None, None, f"missing_contract:{key}"

    qty = float(lot_size) * float(units_per_lot)
    spread_price = SPREAD_PRICE_BY_SYMBOL.get(key)
    note = "configured"
    if spread_price is None:
        if _is_crypto_symbol(symbol):
            spread_price = entry_f * (DEFAULT_SPREAD_PCT_CRYPTO / 100.0)
            note = f"default_pct_crypto:{DEFAULT_SPREAD_PCT_CRYPTO}"
        else:
            spread_price = entry_f * (DEFAULT_SPREAD_PCT_MACRO / 100.0)
            note = f"default_pct_macro:{DEFAULT_SPREAD_PCT_MACRO}"

    spread_cost = float(spread_price) * qty
    return spread_cost, qty, note


def _wr(rows):
    wins = sum(1 for r in rows if r[0] in WIN_OUTCOMES and (r[1] or 0) > 0)
    total = len(rows)
    avg_pnl = sum((r[1] or 0) for r in rows) / total if total else 0
    net_avg = avg_pnl - COST_PCT
    return wins, total, wins/total*100 if total else 0, avg_pnl, net_avg


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    now_utc = datetime.now(timezone.utc)

    print(f"\n{'='*65}")
    print(f"  📊 CRYPTO SIGNAL BOT — PERFORMANCE DASHBOARD")
    print(f"  Generated: {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*65}\n")

    # ── 1. Overall Overview ──────────────────────────────────────────
    print(f"{SEP}")
    print("  1. OVERALL PERFORMANCE (all time)")
    print(SEP)

    rows_all = q(conn, """
        SELECT outcome, pnl_pct FROM signal_outcomes
        WHERE outcome NOT IN ('OPEN','EXITED') AND pnl_pct IS NOT NULL
    """)
    wins, total, wr, avg_pnl, net_avg = _wr([(r['outcome'], r['pnl_pct']) for r in rows_all])
    open_cnt = q(conn, "SELECT COUNT(*) FROM signal_outcomes WHERE outcome='OPEN'")[0][0]
    avg_win  = sum(r['pnl_pct'] for r in rows_all if r['outcome'] in WIN_OUTCOMES and r['pnl_pct'] > 0) / max(1, wins)
    losses   = total - wins
    avg_loss = sum(r['pnl_pct'] for r in rows_all if r['outcome'] in LOSS_OUTCOMES and r['pnl_pct'] is not None) / max(1, losses)
    rr_actual = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    expectancy = (wr/100 * avg_win) + ((1-wr/100) * avg_loss)

    print(f"  Total closed:     {total:,}")
    print(f"  Open (tracking):  {open_cnt}")
    print(f"  Wins:             {wins:,}  |  Losses: {losses:,}")
    print(f"  Win Rate:         {wr:.1f}%")
    print(f"  Avg Winner:       +{avg_win:.2f}%")
    print(f"  Avg Loser:        {avg_loss:.2f}%")
    print(f"  Actual R:R:       1:{rr_actual:.2f}")
    print(f"  Avg Gross PnL:    {avg_pnl:+.3f}% per trade")
    print(f"  Avg Net PnL:      {net_avg:+.3f}% per trade (after ~{COST_PCT}% fees)")
    print(f"  Expectancy:       {expectancy:+.3f}% per trade")
    print(f"  {'✅ POSITIVE EDGE' if expectancy > COST_PCT else '⚠️  EDGE TOO THIN' if expectancy > 0 else '❌ NEGATIVE EDGE'}")

    # ── 2. Pre vs Post Comparison ────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  2. PRE vs POST IMPROVEMENTS (cutoff: {IMPROVEMENT_DATE[:10]})")
    print(SEP)

    for label, op in [("BEFORE", "<"), ("AFTER", ">=")]:
        rows = q(conn, f"""
            SELECT outcome, pnl_pct FROM signal_outcomes
            WHERE outcome NOT IN ('OPEN','EXITED') AND pnl_pct IS NOT NULL
              AND closed_at {op} ?
        """, (IMPROVEMENT_DATE,))
        w, t, wr2, avg2, net2 = _wr([(r['outcome'], r['pnl_pct']) for r in rows])
        print(f"  {label:6}  trades={t:,}  WR={wr2:.1f}%  avg_pnl={avg2:+.3f}%  net={net2:+.3f}%")

    # ── 3. Hour-of-Day Analysis ──────────────────────────────────────
    print(f"\n{SEP}")
    print("  3. HOUR-OF-DAY WIN RATE (UTC) — Session Filter Validation")
    print(SEP)
    print(f"  {'Hour':>5}  {'Trades':>7}  {'WinRate':>8}  {'Avg PnL':>9}  {'Signal'}")

    hour_rows = q(conn, """
        SELECT CAST(strftime('%H', closed_at) AS INTEGER) as hr,
               outcome, pnl_pct
        FROM signal_outcomes
        WHERE outcome NOT IN ('OPEN','EXITED') AND pnl_pct IS NOT NULL
          AND closed_at >= datetime('now','-90 days')
    """)
    by_hour = {}
    for r in hour_rows:
        h = r['hr']
        by_hour.setdefault(h, []).append((r['outcome'], r['pnl_pct']))

    for h in range(24):
        items = by_hour.get(h, [])
        if not items:
            continue
        w2, t2, wr3, avg3, _ = _wr(items)
        flag = "⚠️  Dead zone" if 0 <= h < 6 else ("🟢 Strong" if wr3 >= 55 else ("🔴 Weak" if wr3 < 45 else ""))
        print(f"  {h:02d}:00  {t2:>7,}  {wr3:>7.1f}%  {avg3:>+8.3f}%  {flag}")

    # ── 4. Per-Symbol Breakdown ──────────────────────────────────────
    print(f"\n{SEP}")
    print("  4. PER-SYMBOL BREAKDOWN (last 90 days)")
    print(SEP)
    print(f"  {'Symbol':22}  {'Trades':>7}  {'WinRate':>8}  {'Avg PnL':>9}  {'Net PnL':>9}")

    sym_rows = q(conn, """
        SELECT symbol, outcome, pnl_pct FROM signal_outcomes
        WHERE outcome NOT IN ('OPEN','EXITED') AND pnl_pct IS NOT NULL
          AND closed_at >= datetime('now','-90 days')
    """)
    by_sym = {}
    for r in sym_rows:
        sym = r['symbol'].replace('_futures','').replace('/USDT:USDT','').replace('-USDT','')
        by_sym.setdefault(sym, []).append((r['outcome'], r['pnl_pct']))

    for sym, items in sorted(by_sym.items(), key=lambda x: -len(x[1])):
        w2, t2, wr4, avg4, net4 = _wr(items)
        flag = "✅" if wr4 >= 52 else ("⚠️" if wr4 >= 47 else "❌")
        print(f"  {flag} {sym:20}  {t2:>7,}  {wr4:>7.1f}%  {avg4:>+8.3f}%  {net4:>+8.3f}%")

    # ── 5. Score Bucket Analysis ─────────────────────────────────────
    print(f"\n{SEP}")
    print("  5. SCORE BUCKET ANALYSIS — Over-scoring Validation")
    print(SEP)
    print(f"  {'Score':>8}  {'Trades':>7}  {'WinRate':>8}  {'Avg PnL':>9}")

    score_rows = q(conn, """
        SELECT s.score, so.outcome, so.pnl_pct
        FROM signal_outcomes so
        JOIN signals s ON s.id = so.signal_id
        WHERE so.outcome NOT IN ('OPEN','EXITED') AND so.pnl_pct IS NOT NULL
          AND so.closed_at >= datetime('now','-90 days')
    """)
    buckets = {}
    for r in score_rows:
        sc = r['score']
        bucket = f"{(sc//2)*2}-{(sc//2)*2+1}" if sc is not None else "?"
        buckets.setdefault(bucket, []).append((r['outcome'], r['pnl_pct']))

    for bucket, items in sorted(buckets.items()):
        w2, t2, wr5, avg5, _ = _wr(items)
        bar = "█" * int(wr5 / 5)
        print(f"  Score {bucket:>5}  {t2:>7,}  {wr5:>7.1f}%  {avg5:>+8.3f}%  {bar}")

    # ── 6. R:R Analysis by Month ─────────────────────────────────────
    print(f"\n{SEP}")
    print("  6. MONTHLY PnL SUMMARY")
    print(SEP)
    print(f"  {'Month':>10}  {'Trades':>7}  {'WinRate':>8}  {'Total PnL':>10}  {'Net PnL':>10}")

    month_rows = q(conn, """
        SELECT strftime('%Y-%m', closed_at) as month, outcome, pnl_pct
        FROM signal_outcomes
        WHERE outcome NOT IN ('OPEN','EXITED') AND pnl_pct IS NOT NULL
        ORDER BY closed_at
    """)
    by_month = {}
    for r in month_rows:
        by_month.setdefault(r['month'], []).append((r['outcome'], r['pnl_pct']))

    for month, items in sorted(by_month.items())[-6:]:
        w2, t2, wr6, avg6, _ = _wr(items)
        total_pnl = sum(i[1] for i in items)
        net_total  = total_pnl - t2 * COST_PCT
        flag = "✅" if net_total > 0 else "❌"
        print(f"  {flag} {month:>10}  {t2:>7,}  {wr6:>7.1f}%  {total_pnl:>+9.2f}%  {net_total:>+9.2f}%")

    # ── 7. Last 24h Quick View ───────────────────────────────────────
    print(f"\n{SEP}")
    print("  7. LAST 24 HOURS")
    print(SEP)
    rows_24h = q(conn, """
        SELECT outcome, pnl_pct, symbol, side, entry, close_price FROM signal_outcomes
        WHERE outcome NOT IN ('OPEN','EXITED') AND pnl_pct IS NOT NULL
          AND closed_at >= datetime('now','-24 hours')
        ORDER BY closed_at
    """)
    if rows_24h:
        w2, t2, wr7, avg7, net7 = _wr([(r['outcome'], r['pnl_pct']) for r in rows_24h])
        total_24 = sum(r['pnl_pct'] for r in rows_24h)
        print(f"  Trades: {t2}  |  Win Rate: {wr7:.1f}%  |  Total PnL: {total_24:+.2f}%  |  Net: {total_24 - t2*COST_PCT:+.2f}%")

        lot_pnls = []
        lot_pnls_after_spread = []
        spread_costs = []
        spread_fallback_count = 0
        missing_contract = set()
        for r in rows_24h:
            lot_pnl, _, note = _estimate_trade_pnl_for_lot(
                symbol=r["symbol"],
                side=r["side"],
                entry=r["entry"],
                close_price=r["close_price"],
                lot_size=LOT_SIZE,
            )
            if lot_pnl is not None:
                lot_pnls.append(lot_pnl)
            elif note.startswith("missing_contract:"):
                missing_contract.add(note.split(":", 1)[1])

            spread_cost, _, spread_note = _estimate_spread_cost_for_lot(
                symbol=r["symbol"],
                entry=r["entry"],
                lot_size=LOT_SIZE,
            )
            if spread_cost is not None and lot_pnl is not None:
                spread_costs.append(spread_cost)
                lot_pnls_after_spread.append(lot_pnl - spread_cost)
                if spread_note.startswith("default_pct_"):
                    spread_fallback_count += 1
            elif spread_note.startswith("missing_contract:"):
                missing_contract.add(spread_note.split(":", 1)[1])

        if lot_pnls:
            lot_profit = sum(x for x in lot_pnls if x > 0)
            lot_loss = sum(x for x in lot_pnls if x < 0)
            lot_net = sum(lot_pnls)
            print(
                f"  PnL @{LOT_SIZE:.1f} lot: "
                f"gross profit {lot_profit:+.2f} USD | gross loss {lot_loss:+.2f} USD | net {lot_net:+.2f} USD"
            )
            if spread_costs:
                spread_total = sum(spread_costs)
                net_profit_spread = sum(x for x in lot_pnls_after_spread if x > 0)
                net_loss_spread = sum(x for x in lot_pnls_after_spread if x < 0)
                net_total_spread = sum(lot_pnls_after_spread)
                print(
                    f"  After spread ({SPREAD_PROFILE_NAME}): "
                    f"gross profit {net_profit_spread:+.2f} USD | gross loss {net_loss_spread:+.2f} USD | "
                    f"net {net_total_spread:+.2f} USD | spread cost {-spread_total:+.2f} USD"
                )
                if spread_fallback_count:
                    print(
                        f"  Note: spread fallback % used on {spread_fallback_count} trades "
                        f"(crypto {DEFAULT_SPREAD_PCT_CRYPTO}%, macro {DEFAULT_SPREAD_PCT_MACRO}%)."
                    )
        else:
            print(f"  PnL @{LOT_SIZE:.1f} lot: N/A (missing prices/contracts)")

        if missing_contract:
            missing_list = ", ".join(sorted(missing_contract))
            print(f"  Note: set contract size for symbols in CONTRACT_UNITS_PER_1_LOT: {missing_list}")
        print()
        for r in rows_24h:
            sym = r['symbol'].replace('_futures','').replace('/USDT:USDT','')
            ico = "✅" if r['outcome'] in WIN_OUTCOMES else "❌"
            entry_txt = _fmt_price(r['entry'])
            close_txt = _fmt_price(r['close_price'])
            lot_pnl, qty_used, note = _estimate_trade_pnl_for_lot(
                symbol=r["symbol"],
                side=r["side"],
                entry=r["entry"],
                close_price=r["close_price"],
                lot_size=LOT_SIZE,
            )
            spread_cost, _, _ = _estimate_spread_cost_for_lot(
                symbol=r["symbol"],
                entry=r["entry"],
                lot_size=LOT_SIZE,
            )
            if lot_pnl is None:
                gross_txt = "n/a"
                spread_txt = "n/a"
                net_txt = "n/a"
            else:
                gross_txt = f"{lot_pnl:+.2f}$"
                if spread_cost is None:
                    spread_txt = "n/a"
                    net_txt = "n/a"
                else:
                    spread_txt = f"-{spread_cost:.2f}$"
                    net_txt = f"{(lot_pnl - spread_cost):+.2f}$"
            print(
                f"  {ico} {sym:12} {r['side']:5} {r['outcome']:15} {r['pnl_pct']:+.2f}%  "
                f"entry={entry_txt}  close={close_txt}  "
                f"gross@{LOT_SIZE:.1f}lot={gross_txt}  spread={spread_txt}  net={net_txt}"
            )
    else:
        print("  No closed trades in last 24h.")

    print(f"\n{'='*65}\n")
    conn.close()


if __name__ == "__main__":
    main()
