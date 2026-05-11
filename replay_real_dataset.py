#!/usr/bin/env python3
"""
Generate "replay-real" closed outcomes from historical OHLCV and strategy signals.

This script replays historical candles, runs `analyze_symbol` (technical core),
resolves TP/SL outcomes on forward candles, then inserts closed trades into
`signals` + `signal_outcomes` with a dedicated tag:
  HIST_REPLAY_REAL_V1
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from bot.exchange import get_exchange
from bot.signals import analyze_symbol
from bot.utils import Config, load_config, setup_logging
from backtest_expert import fetch_ohlcv_range


REPLAY_TAG = "HIST_REPLAY_REAL_V1"


@dataclass
class ReplayRow:
    symbol_key: str
    side: str
    signal_ts: datetime
    score: int
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    outcome: str
    close_price: float
    pnl_pct: float
    closed_at: datetime
    regime: str


def _pnl_pct(side: str, entry: float, close_price: float) -> float:
    if side == "LONG":
        return ((close_price - entry) / entry) * 100.0
    return ((entry - close_price) / entry) * 100.0


def _resolve_outcome_with_tp2(
    side: str,
    entry: float,
    sl: float,
    tp1: float,
    tp2: Optional[float],
    future_df: pd.DataFrame,
) -> Optional[Tuple[str, float, float, datetime]]:
    """
    Returns: (outcome, close_price, pnl_pct, closed_at)
    outcomes: TP2_HIT / TP1_HIT / SL_HIT
    """
    if future_df.empty:
        return None

    for ts, row in future_df.iterrows():
        high = float(row["high"])
        low = float(row["low"])

        if side == "LONG":
            hit_tp2 = bool(tp2 is not None and high >= tp2)
            hit_tp1 = high >= tp1
            hit_sl = low <= sl
        else:
            hit_tp2 = bool(tp2 is not None and low <= tp2)
            hit_tp1 = low <= tp1
            hit_sl = high >= sl

        # Ambiguous same-candle touches are skipped to reduce optimistic bias.
        touched = int(hit_tp2) + int(hit_tp1) + int(hit_sl)
        if touched >= 2:
            return None

        if hit_tp2 and tp2 is not None:
            return "TP2_HIT", float(tp2), float(_pnl_pct(side, entry, float(tp2))), ts.to_pydatetime()
        if hit_tp1:
            return "TP1_HIT", float(tp1), float(_pnl_pct(side, entry, float(tp1))), ts.to_pydatetime()
        if hit_sl:
            return "SL_HIT", float(sl), float(_pnl_pct(side, entry, float(sl))), ts.to_pydatetime()

    # Timeout mapping (conservative threshold)
    last_ts = future_df.index[-1].to_pydatetime()
    last_close = float(future_df["close"].iloc[-1])
    timeout_pnl = _pnl_pct(side, entry, last_close)
    if timeout_pnl >= 0.15:
        return "TP1_HIT", last_close, timeout_pnl, last_ts
    if timeout_pnl <= -0.15:
        return "SL_HIT", last_close, timeout_pnl, last_ts
    return None


def _is_duplicate_seed(
    cursor: sqlite3.Cursor,
    symbol_key: str,
    signal_ts_iso: str,
) -> bool:
    row = cursor.execute(
        """
        SELECT 1
        FROM signals
        WHERE symbol = ?
          AND timestamp = ?
          AND reasons LIKE ?
        LIMIT 1
        """,
        (symbol_key, signal_ts_iso, f"%{REPLAY_TAG}%"),
    ).fetchone()
    return bool(row)


def _insert_rows(db_path: str, rows: List[ReplayRow], dry_run: bool = False) -> int:
    if dry_run or not rows:
        return 0

    inserted = 0
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for r in rows:
            ts_iso = r.signal_ts.isoformat()
            if _is_duplicate_seed(cur, r.symbol_key, ts_iso):
                continue

            reasons = json.dumps(
                [
                    REPLAY_TAG,
                    "source:real_ohlcv_replay",
                    "policy:technical",
                    f"regime:{r.regime}",
                ]
            )
            cur.execute(
                """
                INSERT INTO signals
                (symbol, side, timestamp, score, entry, stop_loss, take_profit_1, take_profit_2, reasons)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.symbol_key,
                    r.side,
                    ts_iso,
                    int(r.score),
                    float(r.entry),
                    float(r.stop_loss),
                    float(r.take_profit_1),
                    float(r.take_profit_2),
                    reasons,
                ),
            )
            signal_id = int(cur.lastrowid)
            cur.execute(
                """
                INSERT INTO signal_outcomes
                (signal_id, symbol, side, entry, stop_loss, take_profit_1, take_profit_2,
                 outcome, close_price, pnl_pct, closed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal_id,
                    r.symbol_key,
                    r.side,
                    float(r.entry),
                    float(r.stop_loss),
                    float(r.take_profit_1),
                    float(r.take_profit_2),
                    r.outcome,
                    float(r.close_price),
                    float(r.pnl_pct),
                    r.closed_at.isoformat(),
                ),
            )
            inserted += 1
        conn.commit()
    return inserted


def _summarize(rows: List[ReplayRow]) -> Dict[str, float]:
    if not rows:
        return {}

    df = pd.DataFrame(
        [
            {
                "side": r.side,
                "regime": r.regime,
                "outcome": r.outcome,
                "pnl_pct": r.pnl_pct,
            }
            for r in rows
        ]
    )
    wins = int(df["outcome"].isin(["TP1_HIT", "TP2_HIT"]).sum())
    losses = int(df["outcome"].eq("SL_HIT").sum())
    return {
        "rows": int(len(df)),
        "wins": wins,
        "losses": losses,
        "winrate_pct": round((wins / (wins + losses) * 100.0), 2) if (wins + losses) > 0 else 0.0,
        "avg_pnl_pct": round(float(df["pnl_pct"].mean()), 4),
        "total_pnl_pct": round(float(df["pnl_pct"].sum()), 4),
        "tp2_hits": int(df["outcome"].eq("TP2_HIT").sum()),
        "uptrend": int(df["regime"].eq("uptrend").sum()),
        "downtrend": int(df["regime"].eq("downtrend").sum()),
        "sideways": int(df["regime"].eq("sideways").sum()),
        "longs": int(df["side"].eq("LONG").sum()),
        "shorts": int(df["side"].eq("SHORT").sum()),
    }


def _count_replay_real_rows(db_path: str) -> int:
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            row = cur.execute(
                """
                SELECT COUNT(*)
                FROM signal_outcomes so
                JOIN signals s ON s.id = so.signal_id
                WHERE so.closed_at IS NOT NULL
                  AND so.outcome != 'OPEN'
                  AND s.reasons LIKE ?
                """,
                (f"%{REPLAY_TAG}%",),
            ).fetchone()
        return int(row[0] if row else 0)
    except sqlite3.OperationalError:
        return 0


def _count_total_closed(db_path: str) -> int:
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            row = cur.execute(
                """
                SELECT COUNT(*)
                FROM signal_outcomes
                WHERE closed_at IS NOT NULL AND outcome != 'OPEN'
                """
            ).fetchone()
        return int(row[0] if row else 0)
    except sqlite3.OperationalError:
        return 0


def _regime_from_result(result) -> str:
    trend = str(result.trend or "").lower()
    if trend == "up":
        return "uptrend"
    if trend == "down":
        return "downtrend"
    return "sideways"


def run(args: argparse.Namespace) -> Dict:
    cfg = load_config(args.config)
    bt_cfg: Config = copy.deepcopy(cfg)
    bt_cfg.openai.enabled = False
    bt_cfg.time_analysis.enabled = False
    bt_cfg.time_analysis.expert_advisor = False

    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list(bt_cfg.symbols)
    symbols = [s for s in symbols if "/" in s and ":" in s]
    if not symbols:
        raise RuntimeError("No valid futures symbols for replay-real.")

    exchange = get_exchange(
        bt_cfg.exchange_name,
        bt_cfg.market_type,
        getattr(bt_cfg, "mt5_bridge", None),
        getattr(bt_cfg, "mt5", None),
    )
    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(days=max(0, int(args.end_days_ago)))
    start_dt = end_dt - timedelta(days=max(3, int(args.days)))
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    all_rows: List[ReplayRow] = []
    inserted_target = max(1, int(args.target_additional))

    for symbol in symbols:
        logger.info(
            "Replay start symbol={} days={} step={} lookahead={} warmup={} target={}",
            symbol,
            int(args.days),
            int(args.step),
            int(args.lookahead),
            int(args.warmup),
            inserted_target,
        )
        trend_df = fetch_ohlcv_range(exchange, symbol, bt_cfg.trend_tf, since_ms, until_ms)
        entry_df = fetch_ohlcv_range(exchange, symbol, bt_cfg.entry_tf, since_ms, until_ms)
        sr_df = fetch_ohlcv_range(exchange, symbol, bt_cfg.sr_tf, since_ms, until_ms)
        if trend_df.empty or entry_df.empty or sr_df.empty:
            logger.warning("Skipping {} due to empty OHLCV", symbol)
            continue

        warmup = max(260, int(args.warmup))
        if len(entry_df) <= warmup + int(args.lookahead):
            logger.warning("Skipping {} due to insufficient entry candles {}", symbol, len(entry_df))
            continue

        last_by_side: Dict[str, Optional[datetime]] = {"LONG": None, "SHORT": None}
        sym_key = f"{symbol}_futures"
        scanned = 0
        analyzed = 0
        valid_signals = 0
        resolved_rows = 0
        max_scans = max(0, int(args.max_scans_per_symbol))
        progress_every = max(0, int(args.progress_every))

        for i in range(warmup, len(entry_df) - int(args.lookahead)):
            if (i - warmup) % max(1, int(args.step)) != 0:
                continue

            if max_scans > 0 and scanned >= max_scans:
                logger.info("Replay cap reached symbol={} scanned_cap={}", symbol, max_scans)
                break

            scanned += 1
            ts = entry_df.index[i]
            trend_slice = trend_df[trend_df.index <= ts].tail(bt_cfg.limit)
            entry_slice = entry_df.iloc[max(0, i - bt_cfg.limit + 1): i + 1]
            sr_slice = sr_df[sr_df.index <= ts].tail(bt_cfg.limit)
            if len(entry_slice) < 220 or len(trend_slice) < 60 or len(sr_slice) < 60:
                continue

            ticker = {"last": float(entry_slice["close"].iloc[-1])}
            try:
                result = analyze_symbol(
                    symbol=symbol,
                    df_trend=trend_slice,
                    df_entry=entry_slice,
                    df_sr=sr_slice,
                    ticker=ticker,
                    config=bt_cfg,
                )
            except Exception:
                continue
            analyzed += 1

            if progress_every > 0 and (scanned % progress_every == 0):
                logger.info(
                    "Replay progress symbol={} scanned={} analyzed={} valid={} resolved={} candidates={}/{}",
                    symbol,
                    scanned,
                    analyzed,
                    valid_signals,
                    resolved_rows,
                    len(all_rows),
                    inserted_target,
                )

            if not (result.is_valid and result.side in ("LONG", "SHORT") and result.risk_levels):
                continue
            valid_signals += 1

            if float(result.total_score) < float(result.threshold) + float(args.min_score_buffer):
                continue

            prev = last_by_side.get(result.side)
            if prev is not None and (ts - prev) < timedelta(minutes=bt_cfg.cooldown_minutes):
                continue
            last_by_side[result.side] = ts

            rl = result.risk_levels
            future_df = entry_df.iloc[i + 1: i + 1 + int(args.lookahead)]
            resolved = _resolve_outcome_with_tp2(
                side=result.side,
                entry=float(rl.entry),
                sl=float(rl.stop_loss),
                tp1=float(rl.take_profit_1),
                tp2=float(rl.take_profit_2) if rl.take_profit_2 else None,
                future_df=future_df,
            )
            if not resolved:
                continue

            outcome, close_price, pnl_pct, closed_at = resolved
            if closed_at.tzinfo is None:
                closed_at = closed_at.replace(tzinfo=timezone.utc)
            resolved_rows += 1

            all_rows.append(
                ReplayRow(
                    symbol_key=sym_key,
                    side=result.side,
                    signal_ts=ts.to_pydatetime().replace(tzinfo=timezone.utc),
                    score=int(round(float(result.total_score))),
                    entry=float(rl.entry),
                    stop_loss=float(rl.stop_loss),
                    take_profit_1=float(rl.take_profit_1),
                    take_profit_2=float(rl.take_profit_2),
                    outcome=str(outcome),
                    close_price=float(close_price),
                    pnl_pct=float(pnl_pct),
                    closed_at=closed_at,
                    regime=_regime_from_result(result),
                )
            )

            if len(all_rows) >= inserted_target:
                break

        logger.info(
            "Replay done symbol={} scanned={} analyzed={} valid={} resolved={} cumulative_candidates={}",
            symbol,
            scanned,
            analyzed,
            valid_signals,
            resolved_rows,
            len(all_rows),
        )

        if len(all_rows) >= inserted_target:
            break

    selected = all_rows[:inserted_target]
    summary = _summarize(selected)
    inserted = _insert_rows(args.db_path, selected, dry_run=bool(args.dry_run))

    return {
        "status": ("dry_run" if args.dry_run else "inserted"),
        "tag": REPLAY_TAG,
        "symbols_used": symbols,
        "days": int(args.days),
        "end_days_ago": int(args.end_days_ago),
        "target_additional": inserted_target,
        "generated_candidates": len(all_rows),
        "selected_rows": len(selected),
        "inserted_rows": int(inserted),
        "summary": summary,
        "total_closed_now": _count_total_closed(args.db_path),
        "replay_real_rows_now": _count_replay_real_rows(args.db_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate replay-real closed outcomes into signals.db")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--db-path", default="signals.db")
    p.add_argument("--symbols", default="")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--end-days-ago", type=int, default=0)
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--lookahead", type=int, default=120)
    p.add_argument("--warmup", type=int, default=260)
    p.add_argument("--min-score-buffer", type=float, default=0.5)
    p.add_argument("--target-additional", type=int, default=1200)
    p.add_argument("--progress-every", type=int, default=120)
    p.add_argument("--max-scans-per-symbol", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level, log_file=None)
    out = run(args)
    print("\n=== REPLAY REAL SUMMARY ===")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
