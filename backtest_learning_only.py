#!/usr/bin/env python3
"""
Backtest learning policies versus technical baseline.

This replay does not call OpenAI expert decisions.
It evaluates:
- technical policy: take analyze_symbol direction as-is
- hybrid policy: technical direction + adaptive-learning gate/adjustment
- learning_only policy: choose LONG/SHORT from adaptive learning profile
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from bot.exchange import get_exchange
from bot.indicators import add_all_indicators
from bot.learning_engine import evaluate_learning_signal
from bot.risk import RiskLevels, calculate_risk_levels
from bot.signals import analyze_symbol
from bot.utils import Config, load_config, setup_logging
from bot.zones import build_zones, get_nearest_resistance, get_nearest_support, is_price_in_zone

from backtest_expert import fetch_ohlcv_range, resolve_outcome


DECISIVE = {"WIN", "LOSS", "WIN_TIMEOUT", "LOSS_TIMEOUT"}
WIN_SET = {"WIN", "WIN_TIMEOUT"}


@dataclass
class PolicyRecord:
    timestamp: str
    policy: str
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    outcome: str
    pnl_pct: float
    learning_reason: str = ""
    learning_samples: int = 0
    learning_expected_wr: float = 0.0
    learning_expected_pnl: float = 0.0


def _recalc_risk_for_side(
    config: Config,
    side: str,
    ticker: Dict,
    sr_slice: pd.DataFrame,
    entry_slice: pd.DataFrame,
) -> Optional[RiskLevels]:
    zones = build_zones(sr_slice) if not sr_slice.empty else []
    current_price = float(ticker.get("last", 0.0) or 0.0)
    if current_price <= 0:
        return None

    zone_result = is_price_in_zone(current_price, zones)
    nearest_zone = zone_result.zone if zone_result.in_zone else None
    next_zone = (
        get_nearest_resistance(current_price, zones)
        if side == "LONG"
        else get_nearest_support(current_price, zones)
    )

    atr_val = None
    try:
        e = add_all_indicators(entry_slice.copy())
        if "atr_14" in e.columns:
            raw = e["atr_14"].iloc[-1]
            if raw is not None and not pd.isna(raw):
                atr_val = float(raw)
    except Exception:
        atr_val = None

    return calculate_risk_levels(
        ticker=ticker,
        side=side,
        zone=nearest_zone,
        next_zone=next_zone,
        buffer_pct=config.risk.buffer_pct,
        rr_tp1=config.risk.rr_tp1,
        rr_tp2=config.risk.rr_tp2,
        atr=atr_val,
    )


def _pick_learning_side(
    config: Config,
    db_path: str,
    symbol_key: str,
    base_score: float,
    threshold: float,
) -> Tuple[Optional[str], Dict]:
    long_d = evaluate_learning_signal(
        db_path=db_path,
        symbol=symbol_key,
        side="LONG",
        base_score=base_score,
        threshold=threshold,
        config=config.learning,
    )
    short_d = evaluate_learning_signal(
        db_path=db_path,
        symbol=symbol_key,
        side="SHORT",
        base_score=base_score,
        threshold=threshold,
        config=config.learning,
    )

    preferred_side = "LONG"
    preferred = long_d
    other = short_d
    if (
        short_d.expected_winrate > long_d.expected_winrate
        or (
            short_d.expected_winrate == long_d.expected_winrate
            and short_d.expected_pnl_pct > long_d.expected_pnl_pct
        )
    ):
        preferred_side = "SHORT"
        preferred = short_d
        other = long_d

    min_samples = max(1, int(getattr(config.learning, "decision_min_samples", 40)))
    min_wr = float(getattr(config.learning, "decision_min_winrate", 0.50))
    min_pnl = float(getattr(config.learning, "decision_min_pnl_pct", 0.0))
    min_edge = float(getattr(config.learning, "decision_min_edge_pct", 0.002))
    wr_edge = preferred.expected_winrate - other.expected_winrate

    allow = (
        preferred.allow
        and preferred.sample_size >= min_samples
        and preferred.expected_winrate >= min_wr
        and preferred.expected_pnl_pct >= min_pnl
        and wr_edge >= min_edge
    )

    return (preferred_side if allow else None), {
        "reason": preferred.reason,
        "samples": int(preferred.sample_size),
        "exp_wr": float(preferred.expected_winrate),
        "exp_pnl": float(preferred.expected_pnl_pct),
        "wr_edge": float(wr_edge),
        "long_reason": long_d.reason,
        "short_reason": short_d.reason,
    }


def _hybrid_decision_meta(
    config: Config,
    db_path: str,
    symbol_key: str,
    side: str,
    base_score: float,
    threshold: float,
) -> Tuple[bool, float, Dict]:
    learning = evaluate_learning_signal(
        db_path=db_path,
        symbol=symbol_key,
        side=side,
        base_score=base_score,
        threshold=threshold,
        config=config.learning,
    )

    min_local_for_hard_gate = max(4, int(config.learning.min_symbol_side_trades) // 2)
    applied_adjustment = float(learning.score_adjustment)
    if learning.sample_size < min_local_for_hard_gate:
        applied_adjustment *= 0.35
    adjusted_score = float(base_score) + applied_adjustment

    allow = True
    if not learning.allow:
        allow = False
    elif adjusted_score < float(threshold) and learning.sample_size >= min_local_for_hard_gate:
        allow = False

    return allow, adjusted_score, {
        "reason": learning.reason,
        "samples": int(learning.sample_size),
        "exp_wr": float(learning.expected_winrate),
        "exp_pnl": float(learning.expected_pnl_pct),
        "adj": float(applied_adjustment),
        "allow": bool(learning.allow),
    }


def _summary(df: pd.DataFrame, policy: str) -> Dict:
    part = df[df["policy"] == policy].copy()
    if part.empty:
        return {
            "policy": policy,
            "signals": 0,
            "decisive": 0,
            "wins": 0,
            "losses": 0,
            "winrate_pct": 0.0,
            "avg_pnl_pct": 0.0,
            "total_pnl_pct": 0.0,
            "profit_factor": "inf",
        }

    decisive = part[part["outcome"].isin(list(DECISIVE))].copy()
    wins = int(decisive["outcome"].isin(list(WIN_SET)).sum())
    losses = int(decisive["outcome"].isin(["LOSS", "LOSS_TIMEOUT"]).sum())
    pnl = decisive["pnl_pct"].astype(float) if not decisive.empty else pd.Series(dtype=float)
    avg_pnl = float(pnl.mean()) if not pnl.empty else 0.0
    total_pnl = float(pnl.sum()) if not pnl.empty else 0.0
    gross_win = float(decisive.loc[decisive["pnl_pct"] > 0, "pnl_pct"].sum()) if not decisive.empty else 0.0
    gross_loss = abs(float(decisive.loc[decisive["pnl_pct"] < 0, "pnl_pct"].sum())) if not decisive.empty else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else math.inf

    return {
        "policy": policy,
        "signals": int(len(part)),
        "decisive": int(len(decisive)),
        "wins": wins,
        "losses": losses,
        "winrate_pct": round((wins / (wins + losses) * 100.0), 2) if (wins + losses) > 0 else 0.0,
        "avg_pnl_pct": round(avg_pnl, 4),
        "total_pnl_pct": round(total_pnl, 4),
        "profit_factor": ("inf" if math.isinf(pf) else round(pf, 4)),
    }


def run_backtest(args: argparse.Namespace) -> Dict:
    config = load_config(args.config)
    bt_config = copy.deepcopy(config)
    bt_config.openai.enabled = False
    bt_config.time_analysis.enabled = False
    bt_config.time_analysis.expert_advisor = False

    exchange = get_exchange(
        bt_config.exchange_name,
        bt_config.market_type,
        getattr(bt_config, "mt5_bridge", None),
        getattr(bt_config, "mt5", None),
    )

    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(hours=args.hours)
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    trend_df = fetch_ohlcv_range(exchange, args.symbol, bt_config.trend_tf, since_ms, until_ms)
    entry_df = fetch_ohlcv_range(exchange, args.symbol, bt_config.entry_tf, since_ms, until_ms)
    sr_df = fetch_ohlcv_range(exchange, args.symbol, bt_config.sr_tf, since_ms, until_ms)

    if trend_df.empty or entry_df.empty or sr_df.empty:
        raise RuntimeError("Not enough OHLCV data for replay.")

    warmup = max(args.warmup, 260)
    if len(entry_df) <= warmup + args.lookahead:
        raise RuntimeError("Insufficient entry candles for warmup+lookahead.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbol_tag = args.symbol.replace("/", "_").replace(":", "_")
    file_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    records: List[PolicyRecord] = []
    last_by_policy_side: Dict[str, Dict[str, Optional[datetime]]] = {
        "technical": {"LONG": None, "SHORT": None},
        "hybrid": {"LONG": None, "SHORT": None},
        "learning_only": {"LONG": None, "SHORT": None},
    }

    tested_points = 0
    symbol_key = f"{args.symbol}_futures"
    policies = ["technical", "hybrid", "learning_only"] if args.policy == "all" else [args.policy]

    for i in range(warmup, len(entry_df) - args.lookahead):
        if (i - warmup) % max(1, args.step) != 0:
            continue
        ts = entry_df.index[i]
        tested_points += 1

        trend_slice = trend_df[trend_df.index <= ts].tail(bt_config.limit)
        entry_slice = entry_df.iloc[max(0, i - bt_config.limit + 1): i + 1]
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
        except Exception:
            continue

        if not (result.is_valid and result.side in ("LONG", "SHORT") and result.risk_levels):
            continue

        future_df = entry_df.iloc[i + 1: i + 1 + args.lookahead]
        if future_df.empty:
            continue

        for policy in policies:
            chosen_side: Optional[str] = None
            chosen_risk: Optional[RiskLevels] = None
            learning_meta = {}

            if policy == "technical":
                chosen_side = result.side
                chosen_risk = result.risk_levels
            elif policy == "hybrid":
                allow, _adj_score, learning_meta = _hybrid_decision_meta(
                    config=bt_config,
                    db_path=args.learning_db_path,
                    symbol_key=symbol_key,
                    side=result.side,
                    base_score=float(result.total_score),
                    threshold=float(result.threshold),
                )
                if allow:
                    chosen_side = result.side
                    chosen_risk = result.risk_levels
            elif policy == "learning_only":
                chosen_side, learning_meta = _pick_learning_side(
                    config=bt_config,
                    db_path=args.learning_db_path,
                    symbol_key=symbol_key,
                    base_score=float(result.total_score),
                    threshold=float(result.threshold),
                )
                if chosen_side:
                    if chosen_side == result.side:
                        chosen_risk = result.risk_levels
                    else:
                        chosen_risk = _recalc_risk_for_side(
                            config=bt_config,
                            side=chosen_side,
                            ticker=ticker,
                            sr_slice=sr_slice,
                            entry_slice=entry_slice,
                        )

            if not chosen_side or not chosen_risk:
                continue

            last_ts = last_by_policy_side[policy].get(chosen_side)
            if last_ts is not None and (ts - last_ts) < timedelta(minutes=bt_config.cooldown_minutes):
                continue
            last_by_policy_side[policy][chosen_side] = ts

            outcome, pnl_pct = resolve_outcome(
                side=chosen_side,
                entry=float(chosen_risk.entry),
                sl=float(chosen_risk.stop_loss),
                tp=float(chosen_risk.take_profit_1),
                future_df=future_df,
            )

            records.append(
                PolicyRecord(
                    timestamp=ts.isoformat(),
                    policy=policy,
                    symbol=args.symbol,
                    side=chosen_side,
                    entry_price=float(chosen_risk.entry),
                    stop_loss=float(chosen_risk.stop_loss),
                    take_profit=float(chosen_risk.take_profit_1),
                    outcome=str(outcome),
                    pnl_pct=float(pnl_pct),
                    learning_reason=str(learning_meta.get("reason", "")),
                    learning_samples=int(learning_meta.get("samples", 0)),
                    learning_expected_wr=float(learning_meta.get("exp_wr", 0.0)),
                    learning_expected_pnl=float(learning_meta.get("exp_pnl", 0.0)),
                )
            )

            if args.max_trades > 0:
                total_for_policy = sum(1 for r in records if r.policy == policy)
                if total_for_policy >= args.max_trades:
                    break

    if not records:
        raise RuntimeError("No trades generated for selected policy.")

    df = pd.DataFrame([r.__dict__ for r in records])
    csv_path = out_dir / f"backtest_learning_only_{symbol_tag}_{file_tag}.csv"
    df.to_csv(csv_path, index=False)

    policy_summaries: Dict[str, Dict] = {}
    for pol in sorted(set(df["policy"])):
        policy_summaries[pol] = _summary(df, pol)

    compare: Dict[str, Dict] = {}
    tech = policy_summaries.get("technical")
    hyb = policy_summaries.get("hybrid")
    lo = policy_summaries.get("learning_only")
    if tech and hyb:
        compare["hybrid_minus_technical"] = {
            "delta_signals": int(hyb["signals"] - tech["signals"]),
            "delta_decisive": int(hyb["decisive"] - tech["decisive"]),
            "delta_winrate_pct": round(float(hyb["winrate_pct"]) - float(tech["winrate_pct"]), 2),
            "delta_total_pnl_pct": round(float(hyb["total_pnl_pct"]) - float(tech["total_pnl_pct"]), 4),
        }
    if lo and hyb:
        compare["learning_only_minus_hybrid"] = {
            "delta_signals": int(lo["signals"] - hyb["signals"]),
            "delta_decisive": int(lo["decisive"] - hyb["decisive"]),
            "delta_winrate_pct": round(float(lo["winrate_pct"]) - float(hyb["winrate_pct"]), 2),
            "delta_total_pnl_pct": round(float(lo["total_pnl_pct"]) - float(hyb["total_pnl_pct"]), 4),
        }

    return {
        "symbol": args.symbol,
        "period_start_utc": start_dt.isoformat(),
        "period_end_utc": end_dt.isoformat(),
        "hours_tested": args.hours,
        "entry_timeframe": bt_config.entry_tf,
        "trend_timeframe": bt_config.trend_tf,
        "sr_timeframe": bt_config.sr_tf,
        "step_candles": args.step,
        "lookahead_candles": args.lookahead,
        "tested_points": tested_points,
        "policy": args.policy,
        "policy_summaries": policy_summaries,
        "comparisons": compare,
        "csv_path": str(csv_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest learning-only decision policy.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--lookahead", type=int, default=120)
    p.add_argument("--warmup", type=int, default=260)
    p.add_argument("--max-trades", type=int, default=0, help="0 = unlimited")
    p.add_argument("--policy", choices=["technical", "hybrid", "learning_only", "all"], default="all")
    p.add_argument("--learning-db-path", default="signals.db")
    p.add_argument("--output-dir", default="backtests")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level, log_file=None)
    summary = run_backtest(args)
    print("\n=== LEARNING POLICY BACKTEST SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
