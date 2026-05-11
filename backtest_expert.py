#!/usr/bin/env python3
"""
Historical backtest for Expert Advisor decisions.

This script replays historical candles, builds strategy context, asks the
OpenAI expert for a decision (BUY/SELL/WAIT), then evaluates that decision
against subsequent price action.
"""

from __future__ import annotations

import argparse
import copy
import math
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from bot.exchange import get_exchange
from bot.expert_advisor import get_expert_trade_opinion
from bot.learning_engine import evaluate_learning_signal
from bot.signals import analyze_symbol
from bot.utils import Config, OpenAIConfig, load_config, setup_logging


def _tf_to_ms(exchange, timeframe: str) -> int:
    return int(exchange.parse_timeframe(timeframe) * 1000)


def fetch_ohlcv_range(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch OHLCV between two timestamps using paginated ccxt calls."""
    step_ms = _tf_to_ms(exchange, timeframe)
    cursor = since_ms
    rows: List[List[float]] = []
    guard = 0

    while cursor < until_ms and guard < 10000:
        guard += 1
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break

        filtered = [r for r in batch if r and r[0] < until_ms]
        if not filtered:
            break

        rows.extend(filtered)
        last_ts = int(filtered[-1][0])
        next_cursor = last_ts + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        if len(batch) < limit and last_ts + step_ms >= until_ms:
            break

        # Respect exchange rate limits in pagination loops.
        time.sleep(max(0.05, getattr(exchange, "rateLimit", 200) / 1000.0))

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _valid_levels(side: str, entry: float, sl: Optional[float], tp: Optional[float]) -> bool:
    if sl is None or tp is None:
        return False
    if side == "LONG":
        return sl < entry < tp
    return tp < entry < sl


def choose_levels(
    decision: str,
    entry: float,
    expert_opinion: Dict,
    strategy_levels,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Choose SL/TP with fallback:
    1) Expert SL/TP1
    2) Expert SL/TP2
    3) Strategy SL/TP1
    """
    side = "LONG" if decision == "BUY" else "SHORT"
    esl = expert_opinion.get("stop_loss")
    etp1 = expert_opinion.get("take_profit_1")
    etp2 = expert_opinion.get("take_profit_2")

    try:
        esl = float(esl) if esl is not None else None
        etp1 = float(etp1) if etp1 is not None else None
        etp2 = float(etp2) if etp2 is not None else None
    except Exception:
        esl, etp1, etp2 = None, None, None

    if _valid_levels(side, entry, esl, etp1):
        return esl, etp1, "expert_tp1"
    if _valid_levels(side, entry, esl, etp2):
        return esl, etp2, "expert_tp2"

    ssl = float(strategy_levels.stop_loss) if strategy_levels and strategy_levels.stop_loss else None
    stp1 = float(strategy_levels.take_profit_1) if strategy_levels and strategy_levels.take_profit_1 else None
    if _valid_levels(side, entry, ssl, stp1):
        return ssl, stp1, "strategy_tp1"

    return None, None, "none"


def resolve_outcome(
    side: str,
    entry: float,
    sl: float,
    tp: float,
    future_df: pd.DataFrame,
) -> Tuple[str, float]:
    """
    Resolve trade outcome from future candles.

    Rules:
    - First hit of TP/SL wins outcome.
    - If both hit in same candle => AMBIGUOUS (excluded from win-rate).
    - If none hit by horizon => timeout by last close direction.
    """
    for _, row in future_df.iterrows():
        high = float(row["high"])
        low = float(row["low"])

        if side == "LONG":
            hit_tp = high >= tp
            hit_sl = low <= sl
        else:
            hit_tp = low <= tp
            hit_sl = high >= sl

        if hit_tp and hit_sl:
            return "AMBIGUOUS", 0.0
        if hit_tp:
            pnl = ((tp - entry) / entry) * 100.0 if side == "LONG" else ((entry - tp) / entry) * 100.0
            return "WIN", pnl
        if hit_sl:
            pnl = ((sl - entry) / entry) * 100.0 if side == "LONG" else ((entry - sl) / entry) * 100.0
            return "LOSS", pnl

    last_close = float(future_df["close"].iloc[-1])
    timeout_pnl = ((last_close - entry) / entry) * 100.0 if side == "LONG" else ((entry - last_close) / entry) * 100.0
    if timeout_pnl > 0:
        return "WIN_TIMEOUT", timeout_pnl
    if timeout_pnl < 0:
        return "LOSS_TIMEOUT", timeout_pnl
    return "FLAT_TIMEOUT", 0.0


def _init_replay_learning_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                outcome TEXT NOT NULL,
                pnl_pct REAL,
                closed_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_replay_outcomes_symbol_side_time
            ON signal_outcomes(symbol, side, closed_at DESC)
            """
        )
        conn.commit()


def _seed_replay_learning_db(source_db: str, target_db: Path, lookback_days: int) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))).isoformat()
    inserted = 0
    with sqlite3.connect(source_db) as src_conn, sqlite3.connect(target_db) as tgt_conn:
        src_cur = src_conn.cursor()
        tgt_cur = tgt_conn.cursor()
        rows = src_cur.execute(
            """
            SELECT symbol, side, outcome, pnl_pct, closed_at
            FROM signal_outcomes
            WHERE closed_at IS NOT NULL
              AND closed_at >= ?
              AND outcome != 'OPEN'
            """,
            (cutoff,),
        ).fetchall()
        for symbol, side, outcome, pnl_pct, closed_at in rows:
            tgt_cur.execute(
                """
                INSERT INTO signal_outcomes(symbol, side, outcome, pnl_pct, closed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (symbol, side, outcome, pnl_pct, closed_at),
            )
            inserted += 1
        tgt_conn.commit()
    return inserted


def _map_replay_outcome(outcome: Optional[str]) -> Optional[str]:
    if outcome in {"WIN", "WIN_TIMEOUT"}:
        return "TP1_HIT"
    if outcome in {"LOSS", "LOSS_TIMEOUT"}:
        return "SL_HIT"
    return None


def _append_replay_outcome(
    replay_db_path: Path,
    symbol: str,
    side: str,
    outcome: str,
    pnl_pct: float,
    closed_at: datetime,
) -> None:
    if closed_at.tzinfo is None:
        closed_at = closed_at.replace(tzinfo=timezone.utc)
    with sqlite3.connect(replay_db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO signal_outcomes(symbol, side, outcome, pnl_pct, closed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (symbol, side, outcome, float(pnl_pct), closed_at.isoformat()),
        )
        conn.commit()


def build_expert_context(config: Config, result, df_entry_slice: pd.DataFrame) -> Dict:
    candle_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df_entry_slice.columns]
    recent_candles = []
    for _, row in df_entry_slice[candle_cols].tail(5).iterrows():
        candle = {}
        for col in candle_cols:
            v = row[col]
            if pd.isna(v):
                continue
            candle[col] = float(v)
        if candle:
            recent_candles.append(candle)

    zone_info = result.zone_info or {}
    pattern_info = result.pattern_info or {}
    return {
        "trend_timeframe": config.trend_tf,
        "entry_timeframe": config.entry_tf,
        "trend_adx": float(zone_info.get("adx", 0.0)),
        "trend_rsi": float(zone_info.get("rsi_trend", 50.0)),
        "ichimoku_signal": str(zone_info.get("ichimoku_signal", "neutral")),
        "pattern": pattern_info.get("pattern"),
        "pattern_strength": int(pattern_info.get("pattern_strength") or 0),
        "recent_candles": recent_candles,
    }


def run_backtest(args: argparse.Namespace) -> Dict:
    config = load_config(args.config)
    if getattr(config.ollama, "enabled", False):
        expert_openai_config = OpenAIConfig(
            enabled=True,
            api_key="ollama",  # dummy key
            model=args.model or getattr(config.ollama, "model", "llama3.2:3b"),
            base_url=str(getattr(config.ollama, "base_url", "http://127.0.0.1:11434/v1")).rstrip("/") + "/v1" if "v1" not in str(getattr(config.ollama, "base_url", "")) else str(getattr(config.ollama, "base_url", "")),
        )
    else:
        if not config.openai.enabled or not config.openai.api_key:
            raise RuntimeError("OpenAI is not enabled or OPENAI_API_KEY is missing. Enable Ollama if local.")

        expert_openai_config = OpenAIConfig(
            enabled=True,
            api_key=config.openai.api_key,
            model=args.model or config.openai.model,
        )

    # Backtest analysis config:
    # - Always disable embedded expert call (we call it explicitly and evaluate it).
    # - By default, disable timing/news to avoid present-time leakage.
    bt_config = copy.deepcopy(config)
    bt_config.time_analysis.expert_advisor = False
    if args.use_live_timing_news:
        bt_config.openai.enabled = True
    else:
        bt_config.openai.enabled = False
        bt_config.time_analysis.enabled = False

    exchange = get_exchange(
        "binance",
        "linear",
        getattr(bt_config, "mt5_bridge", None),
        getattr(bt_config, "mt5", None),
    )

    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(hours=args.hours)
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    logger.info(
        "Fetching OHLCV | symbol={} | trend_tf={} | entry_tf={} | sr_tf={} | from={} | to={}",
        args.symbol,
        bt_config.trend_tf,
        bt_config.entry_tf,
        bt_config.sr_tf,
        start_dt.isoformat(),
        end_dt.isoformat(),
    )

    trend_df = fetch_ohlcv_range(exchange, args.symbol, bt_config.trend_tf, since_ms, until_ms)
    entry_df = fetch_ohlcv_range(exchange, args.symbol, bt_config.entry_tf, since_ms, until_ms)
    sr_df = fetch_ohlcv_range(exchange, args.symbol, bt_config.sr_tf, since_ms, until_ms)

    if trend_df.empty or entry_df.empty or sr_df.empty:
        raise RuntimeError("Failed to fetch enough OHLCV data for backtest.")

    logger.info("Fetched bars | trend={} | entry={} | sr={}", len(trend_df), len(entry_df), len(sr_df))

    warmup = max(args.warmup, 220)
    if len(entry_df) <= warmup + args.lookahead:
        raise RuntimeError("Not enough entry candles for warmup + lookahead. Increase --hours.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbol_tag = args.symbol.replace("/", "_").replace(":", "_")
    file_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    replay_learning_db: Optional[Path] = None
    replay_learning_seeded_rows = 0
    walkforward_blocks = 0
    walkforward_updates = 0
    if args.walkforward_learning:
        replay_learning_db = out_dir / f"replay_learning_{symbol_tag}_{file_tag}.db"
        _init_replay_learning_db(replay_learning_db)
        if args.seed_learning_from_real_db:
            replay_learning_seeded_rows = _seed_replay_learning_db(
                source_db=args.seed_db_path,
                target_db=replay_learning_db,
                lookback_days=config.learning.lookback_days,
            )
            logger.info(
                "Replay learning seed rows inserted: {} from {}",
                replay_learning_seeded_rows,
                args.seed_db_path,
            )

    records: List[Dict] = []
    last_signal_time: Dict[str, Optional[datetime]] = {"LONG": None, "SHORT": None}
    analyzed_points = 0
    expert_calls = 0

    for i in range(warmup, len(entry_df) - args.lookahead):
        if (i - warmup) % args.step != 0:
            continue

        ts = entry_df.index[i]
        analyzed_points += 1

        entry_slice = entry_df.iloc[max(0, i - bt_config.limit + 1): i + 1]
        trend_slice = trend_df[trend_df.index <= ts].tail(bt_config.limit)
        sr_slice = sr_df[sr_df.index <= ts].tail(bt_config.limit)

        if len(entry_slice) < 220 or len(trend_slice) < 60 or len(sr_slice) < 60:
            continue

        ticker = {"last": float(entry_slice["close"].iloc[-1])}
        try:
            result = analyze_symbol(
                symbol=args.symbol,
                df_trend=trend_slice,
                df_entry=entry_slice,
                df_sr=sr_slice,
                ticker=ticker,
                config=bt_config,
            )
        except Exception as e:
            logger.debug("analyze_symbol failed at {}: {}", ts, e)
            continue

        if not (result.is_valid and result.side in ("LONG", "SHORT") and result.risk_levels):
            continue

        # Mimic basic live cooldown by side.
        prev_ts = last_signal_time.get(result.side)
        if prev_ts is not None and (ts - prev_ts) < timedelta(minutes=bt_config.cooldown_minutes):
            continue
        last_signal_time[result.side] = ts

        extra_context = build_expert_context(bt_config, result, entry_slice)
        try:
            expert = get_expert_trade_opinion(
                symbol=args.symbol,
                side=result.side,
                current_price=float(result.current_price),
                trend=result.trend,
                technical_score=float(result.technical_score),
                timing_score=float(result.timing_score),
                total_score=float(result.total_score),
                threshold=float(result.threshold),
                reasons=result.reasons,
                timing_reasons=result.timing_reasons,
                vpa_info=result.vpa_info,
                risk_levels=result.risk_levels,
                extra_context=extra_context,
                openai_config=expert_openai_config,
                timeout_seconds=args.expert_timeout,
            )
        except Exception as e:
            logger.debug("Expert call failed at {}: {}", ts, e)
            expert = None

        if not expert:
            continue

        expert_calls += 1
        decision = str(expert.get("decision", "WAIT")).upper()
        try:
            confidence_val = int(float(expert.get("confidence", 0) or 0))
        except Exception:
            confidence_val = 0
        if decision in {"BUY", "SELL"} and confidence_val < args.min_confidence:
            decision = "WAIT"

        learning_allow: Optional[bool] = None
        learning_adjustment = 0.0
        learning_reason = ""
        learning_samples = 0
        if decision in {"BUY", "SELL"} and args.walkforward_learning and replay_learning_db:
            trade_side = "LONG" if decision == "BUY" else "SHORT"
            learning_decision = evaluate_learning_signal(
                db_path=str(replay_learning_db),
                symbol=args.symbol,
                side=trade_side,
                base_score=float(result.total_score),
                threshold=float(result.threshold),
                config=config.learning,
            )
            learning_allow = learning_decision.allow
            learning_adjustment = float(learning_decision.score_adjustment)
            learning_reason = learning_decision.reason
            learning_samples = int(learning_decision.sample_size)

            adjusted_score = float(result.total_score) + learning_adjustment
            min_local_for_hard_gate = max(4, int(config.learning.min_symbol_side_trades) // 2)
            if (not learning_decision.allow) or (
                learning_samples >= min_local_for_hard_gate and adjusted_score < float(result.threshold)
            ):
                decision = "WAIT"
                walkforward_blocks += 1

        entry_price = float(result.current_price)
        direction_match = None
        outcome = None
        pnl_pct = None
        level_source = "none"
        sl = None
        tp = None

        if decision in ("BUY", "SELL"):
            trade_side = "LONG" if decision == "BUY" else "SHORT"
            direction_match = trade_side == result.side
            sl, tp, level_source = choose_levels(decision, entry_price, expert, result.risk_levels)
            if sl is not None and tp is not None:
                future_df = entry_df.iloc[i + 1: i + 1 + args.lookahead]
                outcome, pnl_pct = resolve_outcome(trade_side, entry_price, sl, tp, future_df)
                mapped = _map_replay_outcome(outcome)
                if (
                    mapped
                    and args.walkforward_learning
                    and replay_learning_db is not None
                    and pnl_pct is not None
                ):
                    closed_at = future_df.index[-1].to_pydatetime()
                    _append_replay_outcome(
                        replay_db_path=replay_learning_db,
                        symbol=args.symbol,
                        side=trade_side,
                        outcome=mapped,
                        pnl_pct=float(pnl_pct),
                        closed_at=closed_at,
                    )
                    walkforward_updates += 1

        records.append(
            {
                "timestamp": ts.isoformat(),
                "symbol": args.symbol,
                "base_side": result.side,
                "base_score": float(result.total_score),
                "base_threshold": float(result.threshold),
                "expert_decision": decision,
                "expert_confidence": confidence_val,
                "direction_match": direction_match,
                "entry_price": entry_price,
                "stop_loss": sl,
                "take_profit": tp,
                "level_source": level_source,
                "outcome": outcome,
                "pnl_pct": pnl_pct,
                "expert_rationale": str(expert.get("rationale", "")),
                "walkforward_learning": bool(args.walkforward_learning),
                "learning_allow": learning_allow,
                "learning_score_adjustment": learning_adjustment,
                "learning_samples": learning_samples,
                "learning_reason": learning_reason,
            }
        )

        if expert_calls >= args.max_expert_calls:
            logger.info("Reached max expert calls: {}", args.max_expert_calls)
            break

    if not records:
        raise RuntimeError("No expert decisions generated in this run.")

    df = pd.DataFrame(records)
    csv_path = out_dir / f"backtest_expert_{symbol_tag}_{file_tag}.csv"
    df.to_csv(csv_path, index=False)

    trade_df = df[df["outcome"].notna()].copy()
    decisive_df = trade_df[trade_df["outcome"].isin(["WIN", "LOSS", "WIN_TIMEOUT", "LOSS_TIMEOUT"])].copy()

    wins = int(decisive_df["outcome"].isin(["WIN", "WIN_TIMEOUT"]).sum())
    losses = int(decisive_df["outcome"].isin(["LOSS", "LOSS_TIMEOUT"]).sum())
    ambiguous = int(trade_df["outcome"].eq("AMBIGUOUS").sum()) if not trade_df.empty else 0

    pnl_series = decisive_df["pnl_pct"].dropna().astype(float) if not decisive_df.empty else pd.Series(dtype=float)
    avg_pnl = float(pnl_series.mean()) if not pnl_series.empty else 0.0
    median_pnl = float(pnl_series.median()) if not pnl_series.empty else 0.0
    total_pnl = float(pnl_series.sum()) if not pnl_series.empty else 0.0

    gross_win = float(decisive_df.loc[decisive_df["pnl_pct"] > 0, "pnl_pct"].sum()) if not decisive_df.empty else 0.0
    gross_loss = abs(float(decisive_df.loc[decisive_df["pnl_pct"] < 0, "pnl_pct"].sum())) if not decisive_df.empty else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else math.inf

    directional_subset = df[df["direction_match"].notna()]
    direction_acc = float(directional_subset["direction_match"].mean() * 100.0) if not directional_subset.empty else 0.0

    summary = {
        "symbol": args.symbol,
        "period_start_utc": start_dt.isoformat(),
        "period_end_utc": end_dt.isoformat(),
        "hours_tested": args.hours,
        "entry_timeframe": bt_config.entry_tf,
        "trend_timeframe": bt_config.trend_tf,
        "sr_timeframe": bt_config.sr_tf,
        "step_candles": args.step,
        "lookahead_candles": args.lookahead,
        "min_confidence_filter": args.min_confidence,
        "analyzed_points": analyzed_points,
        "expert_calls": expert_calls,
        "walkforward_learning": bool(args.walkforward_learning),
        "walkforward_blocks": int(walkforward_blocks),
        "walkforward_learning_updates": int(walkforward_updates),
        "replay_learning_seeded_rows": int(replay_learning_seeded_rows),
        "replay_learning_db_path": str(replay_learning_db.resolve()) if replay_learning_db else "",
        "records_total": int(len(df)),
        "decision_buy": int((df["expert_decision"] == "BUY").sum()),
        "decision_sell": int((df["expert_decision"] == "SELL").sum()),
        "decision_wait": int((df["expert_decision"] == "WAIT").sum()),
        "direction_match_pct_vs_base": round(direction_acc, 2),
        "trades_evaluated": int(len(trade_df)),
        "decisive_trades": int(len(decisive_df)),
        "wins": wins,
        "losses": losses,
        "ambiguous": ambiguous,
        "winrate_pct": round((wins / (wins + losses) * 100.0), 2) if (wins + losses) > 0 else 0.0,
        "avg_pnl_pct": round(avg_pnl, 4),
        "median_pnl_pct": round(median_pnl, 4),
        "total_pnl_pct": round(total_pnl, 4),
        "profit_factor": ("inf" if math.isinf(profit_factor) else round(profit_factor, 4)),
        "csv_path": str(csv_path.resolve()),
        "notes": (
            "Live timing/news enabled (uses current-time context during replay)."
            if args.use_live_timing_news
            else "Timing/news were disabled for clean historical replay (avoid present-time leakage)."
        ),
    }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest Expert Advisor decisions on historical candles.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Symbol to backtest")
    parser.add_argument("--hours", type=int, default=24, help="History window in hours")
    parser.add_argument("--step", type=int, default=5, help="Evaluate every Nth entry candle")
    parser.add_argument("--lookahead", type=int, default=90, help="Future candles for outcome evaluation")
    parser.add_argument("--warmup", type=int, default=260, help="Warmup candles before evaluations")
    parser.add_argument("--max-expert-calls", type=int, default=25, help="Cap OpenAI expert calls for one run")
    parser.add_argument("--expert-timeout", type=int, default=45, help="Timeout (seconds) per expert call")
    parser.add_argument("--min-confidence", type=int, default=55, help="Treat BUY/SELL below this confidence as WAIT")
    parser.add_argument(
        "--walkforward-learning",
        action="store_true",
        help="Enable replay learning: update a local learning DB from replay outcomes during the run.",
    )
    parser.add_argument(
        "--seed-learning-from-real-db",
        action="store_true",
        help="Seed replay learning DB from existing real outcomes before replay starts.",
    )
    parser.add_argument(
        "--seed-db-path",
        default="signals.db",
        help="Path to real outcomes database used for replay-learning seed.",
    )
    parser.add_argument("--model", default="", help="Override OpenAI model for this run")
    parser.add_argument(
        "--use-live-timing-news",
        action="store_true",
        help="Keep timing/news logic enabled during replay (less clean historically, but mirrors live flow).",
    )
    parser.add_argument("--output-dir", default="backtests", help="Directory to write CSV results")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level, log_file=None)
    summary = run_backtest(args)
    print("\n=== EXPERT BACKTEST SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
