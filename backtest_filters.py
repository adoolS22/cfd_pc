#!/usr/bin/env python3
"""
Retrospective Filter Impact Analysis
=====================================
Takes all historical closed trades from signals.db and simulates
what performance would have looked like if each new filter had been active.

Filters tested:
  F1  - Session filter: block 00:00–06:00 UTC
  F2  - Symbol filter: remove ORDI
  F3  - Correlation filter: max 2 same-direction crypto signals per 30 min
  F4  - Min R:R filter: minimum TP1 R:R >= 1.3
  F5  - Hour filter: extended dead zones (06:00–07:00, 14:00, 16:00)

Usage: python3 backtest_filters.py
"""

import sqlite3
from datetime import datetime, timezone, timedelta
from collections import defaultdict

DB_PATH   = "signals.db"
COST_PCT  = 0.08   # estimated round-trip fee %
SEP       = "─" * 72

WIN_OUTCOMES  = {"TP_NEAR_HIT", "TP1_HIT", "TP2_HIT", "TRAIL_HIT", "BE_HIT"}
LOSS_OUTCOMES = {"SL_HIT"}

# ── Macro symbol list (same as in bot code) ──────────────────────────────────
MACRO_KEYS = ("XAU", "XAG", "OIL", "WTI", "BRENT", "SNP500", "SPX500",
              "EURUSD", "EUR/USD")

def is_macro(sym: str) -> bool:
    return any(k in sym.upper() for k in MACRO_KEYS)

def is_crypto(sym: str) -> bool:
    return not is_macro(sym)


# ── Stats helpers ─────────────────────────────────────────────────────────────
def stats(trades):
    """Return (count, wins, losses, wr_pct, avg_gross_pnl, avg_net_pnl, total_net)."""
    closed = [t for t in trades if t["outcome"] not in ("OPEN", "EXITED")]
    if not closed:
        return 0, 0, 0, 0.0, 0.0, 0.0, 0.0
    wins   = sum(1 for t in closed if t["outcome"] in WIN_OUTCOMES and (t["pnl"] or 0) > 0)
    losses = len(closed) - wins
    wr     = wins / len(closed) * 100
    avg_gross = sum(t["pnl"] or 0 for t in closed) / len(closed)
    avg_net   = avg_gross - COST_PCT
    total_net = avg_net * len(closed)
    return len(closed), wins, losses, wr, avg_gross, avg_net, total_net


def print_stats(label: str, trades, baseline_total_net=None):
    n, w, l, wr, avg_g, avg_n, tot_n = stats(trades)
    if n == 0:
        print(f"  {label:40s}  No data")
        return
    delta = ""
    if baseline_total_net is not None:
        d = tot_n - baseline_total_net
        delta = f"  Δ net={d:+.1f}%"
    flag = "✅" if avg_n > 0 else ("⚠️ " if avg_n > -0.02 else "❌")
    print(
        f"  {flag} {label:38s}  "
        f"n={n:5,}  WR={wr:5.1f}%  "
        f"avg={avg_g:+.3f}%  net={avg_n:+.3f}%  "
        f"tot_net={tot_n:+6.1f}%{delta}"
    )


# ── Load all closed outcomes ──────────────────────────────────────────────────
def load_trades(conn) -> list:
    rows = conn.execute("""
        SELECT
            so.id, so.symbol, so.side, so.outcome,
            COALESCE(so.pnl_pct, 0.0)             AS pnl,
            so.entry, so.stop_loss, so.take_profit_1,
            so.closed_at,
            s.timestamp                             AS sig_ts,
            s.score
        FROM signal_outcomes so
        LEFT JOIN signals s ON s.id = so.signal_id
        WHERE so.outcome NOT IN ('OPEN','EXITED')
          AND so.pnl_pct IS NOT NULL
        ORDER BY COALESCE(so.closed_at, s.timestamp) ASC
    """).fetchall()

    trades = []
    for r in rows:
        sym = str(r["symbol"]).replace("_futures","")
        closed_raw = r["closed_at"] or r["sig_ts"]
        try:
            ts = datetime.fromisoformat(str(closed_raw))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)

        sig_ts_raw = r["sig_ts"]
        try:
            sig_ts = datetime.fromisoformat(str(sig_ts_raw))
            if sig_ts.tzinfo is None:
                sig_ts = sig_ts.replace(tzinfo=timezone.utc)
        except Exception:
            sig_ts = ts

        # R:R to TP1
        entry = float(r["entry"] or 0)
        sl    = float(r["stop_loss"] or 0)
        tp1   = float(r["take_profit_1"] or 0)
        rr_tp1 = 0.0
        if entry > 0 and sl > 0 and tp1 > 0:
            risk   = abs(entry - sl)
            reward = abs(tp1 - entry)
            rr_tp1 = reward / risk if risk > 0 else 0.0

        trades.append({
            "id":       r["id"],
            "symbol":   sym,
            "side":     r["side"],
            "outcome":  r["outcome"],
            "pnl":      float(r["pnl"]),
            "ts":       ts,       # close time
            "sig_ts":   sig_ts,   # signal sent time
            "hour_utc": sig_ts.hour,
            "score":    r["score"] or 0,
            "rr_tp1":   rr_tp1,
            "is_crypto": is_crypto(sym),
            "is_macro":  is_macro(sym),
        })
    return trades


# ── Filter definitions ────────────────────────────────────────────────────────
def apply_session_filter(trades, blocked_hours=(0,1,2,3,4,5)):
    """Block signals sent in Asia dead zone 00:00–05:59 UTC."""
    return [t for t in trades
            if not (t["is_crypto"] and t["hour_utc"] in blocked_hours)]


def apply_ordi_filter(trades):
    """Remove ORDI signals entirely."""
    return [t for t in trades if "ORDI" not in t["symbol"].upper()]


def apply_extended_dead_zones(trades, weak_hours=(6,7,14,16)):
    """Also block 06:00–07:59 and 14:00–16:59 UTC (shown as weak in analytics)."""
    return [t for t in trades
            if not (t["is_crypto"] and t["hour_utc"] in weak_hours)]


def apply_min_rr_filter(trades, min_rr=1.3):
    """Block signals where TP1 R:R < 1.3 (requires entry/SL/TP1 data)."""
    no_rr  = [t for t in trades if t["rr_tp1"] == 0.0]
    result = [t for t in trades if t["rr_tp1"] == 0.0 or t["rr_tp1"] >= min_rr]
    blocked = len(trades) - len(result)
    return result, blocked, len(no_rr)


def apply_correlation_filter(trades, max_same_dir=2, window_minutes=30):
    """
    Block crypto signals if >= max_same_dir signals of same direction
    were already sent within the last window_minutes.
    Simulates the altcoin correlation filter.
    """
    allowed = []
    # Keep a deque of recent (sig_ts, side) for crypto signals
    recent_crypto = []  # list of (ts, side)

    for t in trades:
        if not t["is_crypto"]:
            allowed.append(t)
            continue

        cutoff = t["sig_ts"] - timedelta(minutes=window_minutes)
        # Prune old entries
        recent_crypto = [(ts, side) for (ts, side) in recent_crypto if ts >= cutoff]

        same_dir_count = sum(1 for (_, side) in recent_crypto if side == t["side"])

        if same_dir_count >= max_same_dir:
            # Would have been blocked by correlation filter
            continue

        allowed.append(t)
        recent_crypto.append((t["sig_ts"], t["side"]))

    return allowed


def apply_score_percentile_filter(trades, percentile_threshold=0.40, min_history=50):
    """
    Block signals whose score is in the bottom X% of recent score history.
    """
    score_history = []
    allowed = []
    for t in trades:
        score = t["score"]
        if len(score_history) >= min_history:
            n_below = sum(1 for s in score_history[-500:] if s <= score)
            pct = n_below / min(len(score_history), 500)
            if pct < percentile_threshold:
                score_history.append(score)
                continue  # blocked
        score_history.append(score)
        allowed.append(t)
    return allowed


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    trades = load_trades(conn)
    conn.close()

    n_tot, w, l, wr, avg_g, avg_n, tot_n = stats(trades)
    baseline_net = tot_n

    print(f"\n{'='*72}")
    print(f"  🔬 RETROSPECTIVE FILTER IMPACT ANALYSIS")
    print(f"  Simulating new filters on {n_tot:,} historical closed trades")
    print(f"{'='*72}\n")

    # ── Baseline ──────────────────────────────────────────────────────────────
    print(f"{SEP}")
    print("  BASELINE (all historical trades, no filters)")
    print(SEP)
    print_stats("Baseline (all trades)", trades)

    crypto_trades = [t for t in trades if t["is_crypto"]]
    macro_trades  = [t for t in trades if t["is_macro"]]
    print_stats("  → Crypto only", crypto_trades)
    print_stats("  → Macro only",  macro_trades)

    # ── Individual filter impact ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  INDIVIDUAL FILTER IMPACT (each filter applied alone)")
    print(f"  {'Filter':40s}  {'n':>5}  {'WR':>6}  {'avg PnL':>8}  {'net/trade':>9}  {'Total Net':>10}  {'Delta':>8}")
    print(SEP)

    # F1: Session filter
    f1 = apply_session_filter(trades)
    print_stats("F1: Block 00:00–05:59 UTC (crypto)", f1, baseline_net)

    # F2: ORDI block
    f2 = apply_ordi_filter(trades)
    print_stats("F2: Remove ORDI", f2, baseline_net)

    # F3: Extended dead zones
    f3 = apply_extended_dead_zones(trades)
    print_stats("F3: Also block 06:00–07:00, 14:00, 16:00", f3, baseline_net)

    # F4: Min R:R
    f4, blocked_rr, no_rr = apply_min_rr_filter(trades, min_rr=1.3)
    print_stats(f"F4: Min R:R >= 1.3 (blocked {blocked_rr:,})", f4, baseline_net)

    # F5: Correlation filter
    f5 = apply_correlation_filter(trades)
    print_stats(f"F5: Correlation (max 2 crypto/{30}min)", f5, baseline_net)

    # F6: Score percentile
    f6 = apply_score_percentile_filter(trades)
    print_stats("F6: Score percentile > 40%", f6, baseline_net)

    # ── Combined filters ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  COMBINED — All filters applied together")
    print(SEP)

    # Apply in sequence
    combined = trades
    combined = apply_ordi_filter(combined)               # F2 — most obvious
    combined = apply_session_filter(combined)            # F1 — session hours
    combined = apply_extended_dead_zones(combined)       # F3 — extended dead zones
    combined = apply_correlation_filter(combined)        # F5 — correlation
    combined_rr, _, _ = apply_min_rr_filter(combined)   # F4 — R:R
    combined = combined_rr

    print_stats("Combined (F1+F2+F3+F4+F5)", combined, baseline_net)

    # Combined crypto only
    combined_crypto = [t for t in combined if t["is_crypto"]]
    combined_macro  = [t for t in combined if t["is_macro"]]
    print_stats("  → Crypto (filtered)", combined_crypto)
    print_stats("  → Macro (unchanged)", combined_macro)

    # ── What got blocked ──────────────────────────────────────────────────────
    combined_ids = {t["id"] for t in combined}
    blocked_all  = [t for t in trades if t["id"] not in combined_ids]

    print(f"\n{SEP}")
    print(f"  WHAT THE FILTERS WOULD HAVE BLOCKED ({len(blocked_all):,} trades)")
    print(SEP)
    print_stats("Blocked signals (would NOT have traded)", blocked_all)

    # Was blocking good or bad?
    blocked_wins   = sum(1 for t in blocked_all if t["outcome"] in WIN_OUTCOMES)
    blocked_losses = sum(1 for t in blocked_all if t["outcome"] in LOSS_OUTCOMES)
    if blocked_all:
        blocked_wr = blocked_wins / len(blocked_all) * 100
        blocked_pnl = sum(t["pnl"] for t in blocked_all) / len(blocked_all)
        verdict = "✅ Good to block (below avg)" if blocked_wr < wr else "⚠️  Mixed (above avg WR)"
        print(f"\n  Blocked trades WR: {blocked_wr:.1f}%  avg PnL: {blocked_pnl:+.3f}%")
        print(f"  Verdict: {verdict}")
        print(f"  (If WR < baseline {wr:.1f}% → blocking them improves performance)")

    # ── Per-symbol impact ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PER-SYMBOL: BEFORE vs AFTER COMBINED FILTER")
    print(f"  {'Symbol':20s}  {'Before n':>8}  {'Before WR':>9}  {'After n':>7}  {'After WR':>8}  {'PnL Δ':>8}")
    print(SEP)

    syms = sorted({t["symbol"].split("/")[0].split("-")[0] for t in trades})
    for sym in syms:
        b_trades = [t for t in trades    if sym in t["symbol"]]
        a_trades = [t for t in combined  if sym in t["symbol"]]
        if not b_trades:
            continue
        bn, bw, bl, bwr, bavg, bnet, btot = stats(b_trades)
        an, aw, al, awr, aavg, anet, atot = stats(a_trades)
        delta_pnl = anet - bnet
        flag = "✅" if delta_pnl > 0.005 else ("⚠️ " if delta_pnl > -0.005 else "❌")
        print(
            f"  {flag} {sym:18s}  {bn:>8,}  {bwr:>8.1f}%  {an:>7,}  {awr:>8.1f}%  {delta_pnl:>+7.3f}%"
        )

    # ── Session hour deep dive ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  HOUR-OF-DAY DEEP DIVE (signal sent hour, crypto only)")
    print(f"  {'Hour':>6}  {'n':>5}  {'WR':>7}  {'avg PnL':>9}  {'net/trade':>10}  {'Status'}")
    print(SEP)

    by_hour = defaultdict(list)
    for t in trades:
        if t["is_crypto"]:
            by_hour[t["hour_utc"]].append(t)

    for h in range(24):
        items = by_hour.get(h, [])
        if not items:
            continue
        n2, w2, l2, wr2, avg2, net2, tot2 = stats(items)
        if h in range(0, 6):
            flag = "❌ BLOCKED (F1)"
        elif h in (6, 7, 14, 16):
            flag = "⚠️  BLOCKED (F3)"
        elif wr2 >= 55:
            flag = "🟢 Strong"
        elif wr2 < 45:
            flag = "🔴 Weak"
        else:
            flag = ""
        print(f"  {h:02d}:00  {n2:>5,}  {wr2:>6.1f}%  {avg2:>+8.3f}%  {net2:>+9.3f}%  {flag}")

    # ── Summary recommendation ────────────────────────────────────────────────
    cn, cw, cl, cwr, cavg, cnet, ctot = stats(combined)
    n_removed = n_tot - cn

    print(f"\n{'='*72}")
    print(f"  📊 SUMMARY")
    print(f"{'='*72}")
    print(f"  Baseline:        {n_tot:,} trades  WR={wr:.1f}%  net/trade={avg_n:+.3f}%  total_net={baseline_net:+.1f}%")
    print(f"  After filters:   {cn:,} trades  WR={cwr:.1f}%  net/trade={cnet:+.3f}%  total_net={ctot:+.1f}%")
    print(f"  Trades removed:  {n_removed:,} ({n_removed/n_tot*100:.1f}% of all signals)")
    print(f"  Net improvement: {ctot - baseline_net:+.1f}% total  ({cnet - avg_n:+.3f}% per trade)")
    print()

    if cnet > avg_n + 0.01:
        print(f"  ✅ VERDICT: Filters IMPROVE performance by {cnet - avg_n:+.3f}% per trade")
        print(f"     Worth enabling — less signals, better quality")
    elif cnet > avg_n - 0.005:
        print(f"  ⚠️  VERDICT: Filters have NEUTRAL impact ({cnet - avg_n:+.3f}% per trade)")
        print(f"     Still useful for reducing noise, monitor closely")
    else:
        print(f"  ❌ VERDICT: Filters may HURT performance ({cnet - avg_n:+.3f}% per trade)")
        print(f"     Review which filter is blocking good signals")

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
