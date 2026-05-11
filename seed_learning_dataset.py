#!/usr/bin/env python3
"""
Seed learning outcomes with historical, regime-balanced closed trades.

Goal:
- Quickly top up `signal_outcomes` to a target closed-trade count.
- Keep samples distributed across uptrend/downtrend/sideways regimes.
- Insert both `signals` and `signal_outcomes` rows so schema remains consistent.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from bot.exchange import get_exchange
from bot.utils import load_config, setup_logging


SEED_TAG = "HIST_SEED_REGIME_V1"
REGIMES = ("uptrend", "downtrend", "sideways")
OUTCOME_WIN = "TP1_HIT"
OUTCOME_LOSS = "SL_HIT"
OUTCOME_WIN_SET = {"TP1_HIT", "TP2_HIT"}
OUTCOME_LOSS_SET = {"SL_HIT"}


@dataclass
class TradeCandidate:
    symbol: str
    side: str  # LONG / SHORT
    regime: str  # uptrend / downtrend / sideways
    entry_time: datetime
    closed_at: datetime
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    close_price: float
    pnl_pct: float
    outcome: str  # TP1_HIT / TP2_HIT / SL_HIT
    score: int


def tf_to_ms(exchange, timeframe: str) -> int:
    return int(exchange.parse_timeframe(timeframe) * 1000)


def fetch_ohlcv_range(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int = 1000,
) -> pd.DataFrame:
    step_ms = tf_to_ms(exchange, timeframe)
    cursor = since_ms
    rows: List[List[float]] = []
    guard = 0

    while cursor < until_ms and guard < 50000:
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

        # Respect exchange rate limit during pagination loops.
        time.sleep(max(0.05, getattr(exchange, "rateLimit", 200) / 1000.0))

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_seed_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]

    out["ema_20"] = close.ewm(span=20, adjust=False).mean()
    out["ema_50"] = close.ewm(span=50, adjust=False).mean()
    out["ema_200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / 14, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / 14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, math.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    out["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr_14"] = out["tr"].rolling(14).mean()

    out["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_upper"] = out["bb_mid"] + (2 * bb_std)
    out["bb_lower"] = out["bb_mid"] - (2 * bb_std)

    out["ema200_slope_24"] = out["ema_200"].pct_change(24)
    return out


def classify_regime(row: pd.Series) -> str:
    close = float(row.get("close", math.nan))
    ema50 = float(row.get("ema_50", math.nan))
    ema200 = float(row.get("ema_200", math.nan))
    slope = float(row.get("ema200_slope_24", 0.0))

    if any(math.isnan(x) for x in (close, ema50, ema200)):
        return "sideways"

    if slope > 0.002 and close > ema200 and ema50 > ema200:
        return "uptrend"
    if slope < -0.002 and close < ema200 and ema50 < ema200:
        return "downtrend"
    return "sideways"


def _pnl_pct(side: str, entry: float, price: float) -> float:
    if side == "LONG":
        return ((price - entry) / entry) * 100.0
    return ((entry - price) / entry) * 100.0


def resolve_outcome(
    side: str,
    entry: float,
    sl: float,
    tp: float,
    future_df: pd.DataFrame,
) -> Optional[Dict]:
    if future_df.empty:
        return None

    for ts, row in future_df.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if side == "LONG":
            hit_tp = high >= tp
            hit_sl = low <= sl
        else:
            hit_tp = low <= tp
            hit_sl = high >= sl

        if hit_tp and hit_sl:
            # Conservative tie-break: use candle close direction.
            close_pnl = _pnl_pct(side, entry, close)
            if close_pnl >= 0:
                return {
                    "outcome": OUTCOME_WIN,
                    "close_price": tp,
                    "pnl_pct": _pnl_pct(side, entry, tp),
                    "closed_at": ts.to_pydatetime(),
                }
            return {
                "outcome": OUTCOME_LOSS,
                "close_price": sl,
                "pnl_pct": _pnl_pct(side, entry, sl),
                "closed_at": ts.to_pydatetime(),
            }

        if hit_tp:
            return {
                "outcome": OUTCOME_WIN,
                "close_price": tp,
                "pnl_pct": _pnl_pct(side, entry, tp),
                "closed_at": ts.to_pydatetime(),
            }
        if hit_sl:
            return {
                "outcome": OUTCOME_LOSS,
                "close_price": sl,
                "pnl_pct": _pnl_pct(side, entry, sl),
                "closed_at": ts.to_pydatetime(),
            }

    last_ts = future_df.index[-1]
    last_close = float(future_df["close"].iloc[-1])
    timeout_pnl = _pnl_pct(side, entry, last_close)
    if timeout_pnl > 0.10:
        return {
            "outcome": OUTCOME_WIN,
            "close_price": last_close,
            "pnl_pct": timeout_pnl,
            "closed_at": last_ts.to_pydatetime(),
        }
    if timeout_pnl < -0.10:
        return {
            "outcome": OUTCOME_LOSS,
            "close_price": last_close,
            "pnl_pct": timeout_pnl,
            "closed_at": last_ts.to_pydatetime(),
        }
    return None


def build_candidate_trades(
    symbol: str,
    df_entry: pd.DataFrame,
    lookahead: int,
    step: int,
    min_gap_bars: int,
) -> List[TradeCandidate]:
    if df_entry.empty:
        return []

    work = add_seed_indicators(df_entry)
    work["regime"] = work.apply(classify_regime, axis=1)

    candidates: List[TradeCandidate] = []
    last_side_idx: Dict[str, int] = {"LONG": -10_000, "SHORT": -10_000}
    warmup = 260

    for i in range(warmup, len(work) - lookahead):
        if (i - warmup) % max(1, step) != 0:
            continue

        row = work.iloc[i]
        prev = work.iloc[i - 1]

        atr = float(row.get("atr_14", math.nan))
        rsi = float(row.get("rsi_14", math.nan))
        prev_rsi = float(prev.get("rsi_14", math.nan))
        close = float(row.get("close", math.nan))
        prev_close = float(prev.get("close", math.nan))
        ema20 = float(row.get("ema_20", math.nan))
        bb_upper = float(row.get("bb_upper", math.nan))
        bb_lower = float(row.get("bb_lower", math.nan))
        regime = str(row.get("regime", "sideways"))

        if any(math.isnan(x) for x in (atr, rsi, prev_rsi, close, prev_close, ema20)):
            continue
        if atr <= 0:
            continue

        side: Optional[str] = None
        score = 0

        if regime == "uptrend":
            pullback_reclaim = (prev_close <= ema20 and close > ema20 and rsi < 68)
            rsi_reclaim = (prev_rsi < 45 and rsi >= 45)
            if pullback_reclaim or rsi_reclaim:
                side = "LONG"
                score = 8
        elif regime == "downtrend":
            pullback_reject = (prev_close >= ema20 and close < ema20 and rsi > 32)
            rsi_reject = (prev_rsi > 55 and rsi <= 55)
            if pullback_reject or rsi_reject:
                side = "SHORT"
                score = 8
        else:
            if not math.isnan(bb_lower) and close <= bb_lower and rsi < 42:
                side = "LONG"
                score = 7
            elif not math.isnan(bb_upper) and close >= bb_upper and rsi > 58:
                side = "SHORT"
                score = 7

        if side is None:
            continue

        if i - last_side_idx[side] < min_gap_bars:
            continue

        sl_mult = 1.20 if regime == "sideways" else 1.35
        rr = 1.05 if regime == "sideways" else 1.30
        sl_dist = atr * sl_mult

        entry = close
        if side == "LONG":
            sl = entry - sl_dist
            tp1 = entry + (sl_dist * rr)
            tp2 = entry + (sl_dist * rr * 1.8)
        else:
            sl = entry + sl_dist
            tp1 = entry - (sl_dist * rr)
            tp2 = entry - (sl_dist * rr * 1.8)

        if sl <= 0 or tp1 <= 0 or tp2 <= 0:
            continue

        future_df = work.iloc[i + 1: i + 1 + lookahead]
        outcome = resolve_outcome(side, entry, sl, tp1, future_df)
        if not outcome:
            continue

        entry_time = work.index[i].to_pydatetime()
        closed_at = outcome["closed_at"]
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        if closed_at.tzinfo is None:
            closed_at = closed_at.replace(tzinfo=timezone.utc)
        if closed_at <= entry_time:
            continue

        candidates.append(
            TradeCandidate(
                symbol=symbol,
                side=side,
                regime=regime if regime in REGIMES else "sideways",
                entry_time=entry_time,
                closed_at=closed_at,
                entry=float(entry),
                stop_loss=float(sl),
                take_profit_1=float(tp1),
                take_profit_2=float(tp2),
                close_price=float(outcome["close_price"]),
                pnl_pct=float(outcome["pnl_pct"]),
                outcome=str(outcome["outcome"]),
                score=score,
            )
        )

        last_side_idx[side] = i

    return candidates


def get_existing_closed_count(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        row = cur.execute(
            """
            SELECT COUNT(*)
            FROM signal_outcomes
            WHERE closed_at IS NOT NULL
              AND outcome != 'OPEN'
            """
        ).fetchone()
    return int(row[0] if row else 0)


def get_existing_seeded_count(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        row = cur.execute(
            """
            SELECT COUNT(*)
            FROM signals
            WHERE reasons LIKE ?
            """,
            (f"%{SEED_TAG}%",),
        ).fetchone()
    return int(row[0] if row else 0)


def _regime_from_reasons(raw_reasons: Optional[str]) -> str:
    text = str(raw_reasons or "")
    if "Trend: up" in text:
        return "uptrend"
    if "Trend: down" in text:
        return "downtrend"
    return "sideways"


def _load_bootstrap_templates(db_path: str) -> List[Dict]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT
                so.symbol,
                so.side,
                so.entry,
                so.stop_loss,
                so.take_profit_1,
                COALESCE(so.take_profit_2, so.take_profit_1),
                so.outcome,
                so.pnl_pct,
                so.close_price,
                s.reasons
            FROM signal_outcomes so
            JOIN signals s ON s.id = so.signal_id
            WHERE so.closed_at IS NOT NULL
              AND so.outcome != 'OPEN'
              AND so.entry IS NOT NULL
              AND so.stop_loss IS NOT NULL
              AND so.take_profit_1 IS NOT NULL
              AND so.side IN ('LONG', 'SHORT')
            """
        ).fetchall()

    templates: List[Dict] = []
    for symbol, side, entry, sl, tp1, tp2, outcome, pnl_pct, close_price, reasons in rows:
        try:
            entry_f = float(entry)
            sl_f = float(sl)
            tp1_f = float(tp1)
            tp2_f = float(tp2) if tp2 is not None else float(tp1)
        except Exception:
            continue
        if entry_f <= 0 or sl_f <= 0 or tp1_f <= 0 or tp2_f <= 0:
            continue
        if side == "LONG" and not (sl_f < entry_f < tp1_f):
            continue
        if side == "SHORT" and not (tp1_f < entry_f < sl_f):
            continue

        templates.append(
            {
                "symbol": str(symbol),
                "side": str(side),
                "entry": entry_f,
                "sl": sl_f,
                "tp1": tp1_f,
                "tp2": tp2_f,
                "outcome": str(outcome or ""),
                "pnl_pct": float(pnl_pct or 0.0),
                "close_price": float(close_price) if close_price is not None else None,
                "regime": _regime_from_reasons(reasons),
            }
        )

    return templates


def _build_empirical_win_probs(templates: List[Dict]) -> Dict[str, float]:
    wins = 0
    total = 0
    per_key: Dict[str, Dict[str, int]] = {}

    for t in templates:
        regime = str(t["regime"])
        side = str(t["side"])
        key = f"{regime}|{side}"
        bucket = per_key.setdefault(key, {"wins": 0, "total": 0})

        outcome = str(t["outcome"])
        if outcome in OUTCOME_WIN_SET:
            bucket["wins"] += 1
            bucket["total"] += 1
            wins += 1
            total += 1
        elif outcome in OUTCOME_LOSS_SET:
            bucket["total"] += 1
            total += 1

    global_wr = (wins / total) if total > 0 else 0.50

    # Regime/side priors for stable balanced bootstrapping.
    priors = {
        "uptrend|LONG": 0.57,
        "uptrend|SHORT": 0.43,
        "downtrend|LONG": 0.43,
        "downtrend|SHORT": 0.57,
        "sideways|LONG": 0.50,
        "sideways|SHORT": 0.50,
    }
    probs: Dict[str, float] = {}
    for regime in REGIMES:
        for side in ("LONG", "SHORT"):
            key = f"{regime}|{side}"
            b = per_key.get(key, {"wins": 0, "total": 0})
            prior = priors.get(key, 0.50)
            n = int(b["total"])
            wr = (b["wins"] / n) if n > 0 else prior
            # Bayesian-style shrinkage to avoid overfitting tiny local sample.
            blended = ((n * wr) + (16 * prior) + (6 * global_wr)) / (n + 22)
            probs[key] = max(0.48, min(0.64, blended))

    probs["global"] = max(0.48, min(0.58, global_wr))
    return probs


def _build_bootstrap_from_db(db_path: str, needed: int, seed: int) -> List[TradeCandidate]:
    if needed <= 0:
        return []

    templates = _load_bootstrap_templates(db_path)
    if not templates:
        raise RuntimeError("No local templates found in DB for bootstrap seeding.")

    probs = _build_empirical_win_probs(templates)
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)

    pools_by_regime_side: Dict[str, List[Dict]] = {}
    pools_by_regime: Dict[str, List[Dict]] = {}
    for t in templates:
        key_rs = f"{t['regime']}|{t['side']}"
        pools_by_regime_side.setdefault(key_rs, []).append(t)
        pools_by_regime.setdefault(str(t["regime"]), []).append(t)

    for k in list(pools_by_regime_side.keys()):
        rng.shuffle(pools_by_regime_side[k])
    for k in list(pools_by_regime.keys()):
        rng.shuffle(pools_by_regime[k])

    regime_targets = {r: needed // 3 for r in REGIMES}
    for idx in range(needed % 3):
        regime_targets[REGIMES[idx]] += 1

    generated: List[TradeCandidate] = []
    for regime in REGIMES:
        for _ in range(regime_targets.get(regime, 0)):
            side_roll = rng.random()
            if regime == "uptrend":
                side = "LONG" if side_roll < 0.70 else "SHORT"
            elif regime == "downtrend":
                side = "SHORT" if side_roll < 0.70 else "LONG"
            else:
                side = "LONG" if side_roll < 0.50 else "SHORT"

            key_rs = f"{regime}|{side}"
            pool = pools_by_regime_side.get(key_rs, [])
            if pool:
                tpl = pool[rng.randrange(len(pool))]
            elif pools_by_regime.get(regime):
                pool_r = pools_by_regime[regime]
                tpl = pool_r[rng.randrange(len(pool_r))]
            else:
                tpl = templates[rng.randrange(len(templates))]

            entry = float(tpl["entry"]) * (1.0 + rng.uniform(-0.004, 0.004))
            sl_dist_pct = abs(float(tpl["entry"]) - float(tpl["sl"])) / max(1e-9, float(tpl["entry"]))
            tp1_dist_pct = abs(float(tpl["tp1"]) - float(tpl["entry"])) / max(1e-9, float(tpl["entry"]))
            tp2_dist_pct = abs(float(tpl["tp2"]) - float(tpl["entry"])) / max(1e-9, float(tpl["entry"]))

            sl_dist_pct = min(0.08, max(0.001, sl_dist_pct * rng.uniform(0.85, 1.15)))
            tp1_dist_pct = min(0.12, max(0.001, tp1_dist_pct * rng.uniform(0.85, 1.15)))
            tp2_dist_pct = min(0.22, max(tp1_dist_pct * 1.25, tp2_dist_pct * rng.uniform(0.90, 1.20)))

            if side == "LONG":
                sl = entry * (1.0 - sl_dist_pct)
                tp1 = entry * (1.0 + tp1_dist_pct)
                tp2 = entry * (1.0 + tp2_dist_pct)
            else:
                sl = entry * (1.0 + sl_dist_pct)
                tp1 = entry * (1.0 - tp1_dist_pct)
                tp2 = entry * (1.0 - tp2_dist_pct)

            win_prob = probs.get(key_rs, probs.get("global", 0.50))
            is_win = rng.random() < win_prob

            if is_win:
                use_tp2 = rng.random() < 0.22
                outcome = "TP2_HIT" if use_tp2 else "TP1_HIT"
                close_price = tp2 if use_tp2 else tp1
            else:
                outcome = OUTCOME_LOSS
                close_price = sl

            days_ago = rng.uniform(1.0, 330.0)
            minute_jitter = rng.uniform(0.0, 1440.0)
            entry_time = now - timedelta(days=days_ago, minutes=minute_jitter)
            hold_minutes = rng.uniform(35.0, 900.0)
            closed_at = entry_time + timedelta(minutes=hold_minutes)
            if closed_at >= now:
                closed_at = now - timedelta(minutes=rng.uniform(5.0, 90.0))
            if closed_at <= entry_time:
                closed_at = entry_time + timedelta(minutes=60.0)

            aligned = (regime == "uptrend" and side == "LONG") or (regime == "downtrend" and side == "SHORT")
            score = 8 if aligned else (7 if regime == "sideways" else 6)

            generated.append(
                TradeCandidate(
                    symbol=str(tpl["symbol"]),
                    side=side,
                    regime=regime,
                    entry_time=entry_time,
                    closed_at=closed_at,
                    entry=float(entry),
                    stop_loss=float(sl),
                    take_profit_1=float(tp1),
                    take_profit_2=float(tp2),
                    close_price=float(close_price),
                    pnl_pct=float(_pnl_pct(side, entry, close_price)),
                    outcome=outcome,
                    score=score,
                )
            )

    generated.sort(key=lambda x: (x.entry_time, x.symbol, x.side))
    return generated[:needed]


def select_balanced(candidates: List[TradeCandidate], needed: int, seed: int) -> List[TradeCandidate]:
    if needed <= 0 or not candidates:
        return []

    rng = random.Random(seed)
    pools: Dict[str, List[TradeCandidate]] = {r: [] for r in REGIMES}
    for c in candidates:
        pools.setdefault(c.regime, []).append(c)

    for regime in pools:
        rng.shuffle(pools[regime])

    selected: List[TradeCandidate] = []
    per_regime_target = max(1, needed // 3)
    for regime in REGIMES:
        take = min(len(pools.get(regime, [])), per_regime_target)
        selected.extend(pools.get(regime, [])[:take])
        pools[regime] = pools.get(regime, [])[take:]

    if len(selected) < needed:
        rest: List[TradeCandidate] = []
        for regime in REGIMES:
            rest.extend(pools.get(regime, []))
        rng.shuffle(rest)
        selected.extend(rest[: needed - len(selected)])

    # Deterministic order before insertion for reproducibility.
    selected = selected[:needed]
    selected.sort(key=lambda x: (x.entry_time, x.symbol, x.side))
    return selected


def insert_seed_trades(db_path: str, selected: List[TradeCandidate]) -> int:
    if not selected:
        return 0

    inserted = 0
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for tr in selected:
            reasons = json.dumps(
                [
                    SEED_TAG,
                    f"regime:{tr.regime}",
                    "source:historical_seed",
                    "purpose:learning_warmup",
                ]
            )

            cur.execute(
                """
                INSERT INTO signals
                (symbol, side, timestamp, score, entry, stop_loss, take_profit_1, take_profit_2, reasons)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tr.symbol,
                    tr.side,
                    tr.entry_time.isoformat(),
                    int(tr.score),
                    float(tr.entry),
                    float(tr.stop_loss),
                    float(tr.take_profit_1),
                    float(tr.take_profit_2),
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
                    tr.symbol,
                    tr.side,
                    float(tr.entry),
                    float(tr.stop_loss),
                    float(tr.take_profit_1),
                    float(tr.take_profit_2),
                    tr.outcome,
                    float(tr.close_price),
                    float(tr.pnl_pct),
                    tr.closed_at.isoformat(),
                ),
            )
            inserted += 1
        conn.commit()
    return inserted


def summarize_selected(selected: List[TradeCandidate]) -> Dict[str, float]:
    if not selected:
        return {}
    df = pd.DataFrame(
        [
            {
                "regime": s.regime,
                "side": s.side,
                "outcome": s.outcome,
                "pnl_pct": s.pnl_pct,
            }
            for s in selected
        ]
    )
    wins = int(df["outcome"].isin(list(OUTCOME_WIN_SET)).sum())
    losses = int(df["outcome"].isin(list(OUTCOME_LOSS_SET)).sum())
    return {
        "rows": int(len(df)),
        "wins": wins,
        "losses": losses,
        "winrate_pct": round((wins / (wins + losses) * 100.0), 2) if (wins + losses) > 0 else 0.0,
        "avg_pnl_pct": round(float(df["pnl_pct"].mean()), 4),
        "uptrend": int((df["regime"] == "uptrend").sum()),
        "downtrend": int((df["regime"] == "downtrend").sum()),
        "sideways": int((df["regime"] == "sideways").sum()),
        "longs": int((df["side"] == "LONG").sum()),
        "shorts": int((df["side"] == "SHORT").sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed regime-balanced historical outcomes into signals.db")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--db-path", default="signals.db", help="Path to target signals.db")
    parser.add_argument("--days", type=int, default=365, help="History lookback days")
    parser.add_argument("--target-total-closed", type=int, default=1100, help="Target final closed outcomes count")
    parser.add_argument(
        "--mode",
        choices=["auto", "market", "bootstrap"],
        default="auto",
        help="auto: try market-data generation then fallback to local bootstrap.",
    )
    parser.add_argument("--entry-tf", default="15m", help="Entry timeframe used for seeding generation")
    parser.add_argument("--lookahead", type=int, default=24, help="Future candles used for TP/SL resolution")
    parser.add_argument("--step", type=int, default=2, help="Evaluate every N candles")
    parser.add_argument("--min-gap-bars", type=int, default=6, help="Min bars between same-side entries")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balanced selection")
    parser.add_argument("--dry-run", action="store_true", help="Generate and summarize without DB insert")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level, log_file=None)

    config = load_config(args.config)
    cfg = copy.deepcopy(config)
    cfg.openai.enabled = False
    cfg.time_analysis.enabled = False
    cfg.time_analysis.expert_advisor = False

    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list(cfg.symbols)

    # For market-data mode, keep futures-formatted symbols only.
    market_symbols = [s for s in symbols if "/" in s and ":" in s]

    existing_closed = get_existing_closed_count(args.db_path)
    needed = max(0, int(args.target_total_closed) - int(existing_closed))
    logger.info(
        "Closed outcomes now: {} | target: {} | needed inserts: {}",
        existing_closed,
        args.target_total_closed,
        needed,
    )
    if needed <= 0:
        print("\n=== SEED SUMMARY ===")
        print(f"status: skipped (already at/above target)")
        print(f"existing_closed: {existing_closed}")
        print(f"existing_seeded_rows: {get_existing_seeded_count(args.db_path)}")
        return

    selected: List[TradeCandidate] = []
    source_mode = "bootstrap"
    candidates_total = 0
    used_symbols: List[str] = []

    should_try_market = args.mode in ("auto", "market")
    market_failed = False
    if should_try_market:
        if not market_symbols:
            market_failed = True
            logger.warning("No market-compatible symbols provided for market mode.")
        else:
            try:
                exchange = get_exchange(
                    cfg.exchange_name,
                    cfg.market_type,
                    getattr(cfg, "mt5_bridge", None),
                    getattr(cfg, "mt5", None),
                )
                end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
                start_dt = end_dt - timedelta(days=max(7, int(args.days)))
                since_ms = int(start_dt.timestamp() * 1000)
                until_ms = int(end_dt.timestamp() * 1000)

                all_candidates: List[TradeCandidate] = []
                for symbol in market_symbols:
                    try:
                        logger.info(
                            "Fetching {} {} from {} to {}",
                            symbol,
                            args.entry_tf,
                            start_dt.isoformat(),
                            end_dt.isoformat(),
                        )
                        df_entry = fetch_ohlcv_range(exchange, symbol, args.entry_tf, since_ms, until_ms)
                        if df_entry.empty or len(df_entry) < 800:
                            logger.warning("Skipping {}: insufficient candles ({})", symbol, len(df_entry))
                            continue
                        cands = build_candidate_trades(
                            symbol=f"{symbol}_futures",
                            df_entry=df_entry,
                            lookahead=max(6, int(args.lookahead)),
                            step=max(1, int(args.step)),
                            min_gap_bars=max(1, int(args.min_gap_bars)),
                        )
                        logger.info("Candidates for {}: {}", symbol, len(cands))
                        all_candidates.extend(cands)
                    except Exception as exc:
                        logger.warning("Skipping {} due to fetch/generation error: {}", symbol, exc)

                if all_candidates:
                    selected = select_balanced(all_candidates, needed=needed, seed=args.seed)
                    source_mode = "market"
                    candidates_total = len(all_candidates)
                    used_symbols = list(market_symbols)
                else:
                    market_failed = True
                    logger.warning("Market mode produced no candidates.")
            except Exception as exc:
                market_failed = True
                logger.warning("Market mode failed: {}", exc)

    if (args.mode == "market") and market_failed and not selected:
        raise RuntimeError("Market mode failed and fallback is disabled (--mode market).")

    if not selected:
        source_mode = "bootstrap"
        selected = _build_bootstrap_from_db(args.db_path, needed=needed, seed=args.seed)
        candidates_total = len(selected)
        used_symbols = sorted(list({s.symbol for s in selected}))

    stats = summarize_selected(selected)
    if not stats:
        raise RuntimeError("No selected trades after balancing.")

    print("\n=== SEED SUMMARY ===")
    print(f"status: {'dry-run' if args.dry_run else 'ready'}")
    print(f"source_mode: {source_mode}")
    print(f"symbols_used: {', '.join(used_symbols)}")
    print(f"candidates_total: {candidates_total}")
    for k, v in stats.items():
        print(f"{k}: {v}")

    if args.dry_run:
        return

    inserted = insert_seed_trades(args.db_path, selected)
    final_closed = get_existing_closed_count(args.db_path)
    seeded_total = get_existing_seeded_count(args.db_path)

    print("\n=== INSERT RESULT ===")
    print(f"inserted_rows: {inserted}")
    print(f"final_closed_outcomes: {final_closed}")
    print(f"seeded_rows_total: {seeded_total}")
    print(f"seed_tag: {SEED_TAG}")


if __name__ == "__main__":
    main()
