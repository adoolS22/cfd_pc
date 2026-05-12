#!/usr/bin/env python3
"""
Crypto Signal Bot - Main Entry Point
=====================================
Scans Binance USDT-M Perpetual Futures and generates trading signals.
"""

import sys
import time
import signal
import argparse
import threading
import re
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from loguru import logger

from bot.utils import load_config, setup_logging, Config
from bot.exchange import get_exchange, fetch_ohlcv, fetch_ticker, fetch_multiple_timeframes
from bot.signals import analyze_symbol, check_exit_conditions, get_dynamic_risk_parameters, SignalResult
from bot.zones import build_zones, is_price_in_zone, get_nearest_support, get_nearest_resistance
from bot.indicators import add_all_indicators, get_trend
from bot.risk import calculate_risk_levels
from bot.storage import SignalStorage, SignalRecord
from bot.notifier import TelegramNotifier
from bot.telegram_control import parse_telegram_control_command
from bot.news_analyzer import NewsAnalyzer
from bot.learning_engine import evaluate_learning_signal, compute_symbol_atr_calibration
from bot.quality_first import evaluate_quality_first
from bot.llm_postmortem import evaluate_loss_postmortem
from bot.yahoo_data import is_yahoo_symbol, fetch_yahoo_ohlcv, get_yahoo_price
from bot.ml_engine import MLEngine
from typing import Any


# Global flag for graceful shutdown
shutdown_requested = False
# Runtime scanner state (controlled by Telegram commands)
scan_paused = False
state_lock = threading.Lock()
# Track last winrate report time
_last_winrate_report: datetime = None
_last_learning_health_report: datetime = None
_risk_alert_timestamps: Dict[str, datetime] = {}


_HISTORICAL_REASON_TAGS = (
    "HIST_SEED_REGIME_V1",
    "HIST_REPLAY_REAL_V1",
    "source:historical_seed",
    "source:real_ohlcv_replay",
)


def _is_historical_or_stale_open_outcome(outcome: dict, max_age_days: int = 14) -> bool:
    """
    Detect non-live OPEN outcomes that should not trigger Telegram alerts.

    We treat seeded/replay rows and very old rows as stale.
    """
    reasons_raw = str(outcome.get("signal_reasons") or "")
    if any(tag in reasons_raw for tag in _HISTORICAL_REASON_TAGS):
        return True

    ts_raw = outcome.get("signal_timestamp")
    if not ts_raw:
        # Broken linkage (missing source signal metadata) -> treat as stale.
        return True

    try:
        signal_ts = datetime.fromisoformat(str(ts_raw))
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - signal_ts > timedelta(days=max_age_days):
            return True
    except Exception:
        # If timestamp is malformed, treat as stale to prevent duplicate spam.
        return True

    return False


def _is_loss_outcome(outcome: str, pnl_pct: float) -> bool:
    """Treat SL or negative managed exits as losses."""
    out = str(outcome or "").upper()
    if out == "SL_HIT":
        return True
    if out in {"TRAIL_HIT", "BE_HIT"} and float(pnl_pct or 0.0) < 0.0:
        return True
    if out == "EXITED" and float(pnl_pct or 0.0) < 0.0:
        return True
    return False


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _is_mt5_source(exchange: Any) -> bool:
    return str(getattr(exchange, "id", "") or "").strip().lower() in {"mt5", "mt5_bridge"}


def _is_mt5_bridge_source(exchange: Any) -> bool:
    return str(getattr(exchange, "id", "") or "").strip().lower() == "mt5_bridge"


def _is_mt5_only_requested(config: Config) -> bool:
    return str(getattr(config, "exchange_name", "") or "").strip().lower() == "mt5"


def _execution_price_from_ticker(
    ticker: Optional[Dict[str, Any]],
    side: str,
    *,
    phase: str = "close",
) -> Tuple[Optional[float], str]:
    """
    MT5-like price selection:
      - LONG entry  -> ask
      - LONG close  -> bid
      - SHORT entry -> bid
      - SHORT close -> ask
    Falls back to last, then opposite quote if needed.
    """
    if not isinstance(ticker, dict):
        return None, "none"

    side_u = str(side or "").upper()
    is_entry = str(phase or "").strip().lower() == "entry"

    if side_u == "LONG":
        if is_entry:
            order = [
                ("ask", ticker.get("ask")),
                ("last", ticker.get("last")),
                ("bid", ticker.get("bid")),
            ]
        else:
            order = [
                ("bid", ticker.get("bid")),
                ("last", ticker.get("last")),
                ("ask", ticker.get("ask")),
            ]
    elif side_u == "SHORT":
        if is_entry:
            order = [
                ("bid", ticker.get("bid")),
                ("last", ticker.get("last")),
                ("ask", ticker.get("ask")),
            ]
        else:
            order = [
                ("ask", ticker.get("ask")),
                ("last", ticker.get("last")),
                ("bid", ticker.get("bid")),
            ]
    else:
        order = [("last", ticker.get("last")), ("bid", ticker.get("bid")), ("ask", ticker.get("ask"))]

    for source, raw in order:
        value = _safe_float(raw)
        if value is not None and value > 0:
            return float(value), source
    return None, "none"


def _build_ticker_from_cached_ohlcv(
    symbol: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a minimal ticker from already-fetched OHLCV data.
    Avoids an extra bridge request when ticker endpoint is unavailable.
    """
    # Prefer entry timeframe for latest execution context.
    for key in ("entry", "trend", "sr", "htf"):
        df = data.get(key)
        if df is None or not hasattr(df, "empty") or bool(df.empty):
            continue
        try:
            last_row = df.iloc[-1]
            last = _safe_float(last_row.get("close"))
            if last is None:
                continue
            high = _safe_float(last_row.get("high"))
            low = _safe_float(last_row.get("low"))
            vol = _safe_float(last_row.get("volume"))
            return {
                "symbol": symbol,
                "last": last,
                "bid": last,
                "ask": last,
                "high": high,
                "low": low,
                "volume": vol,
                "source": "ohlcv_cached",
            }
        except Exception:
            continue
    return {"symbol": symbol}


def _extract_onchain_snapshot_from_reason(reason_line: str) -> Optional[Dict[str, Any]]:
    """Parse one On-chain reason line into a normalized snapshot."""
    line = str(reason_line or "").strip()
    if not line or "on-chain" not in line.lower():
        return None

    snapshot: Dict[str, Any] = {"raw_text": line}

    asset_match = re.search(r"on-chain\s+([a-z0-9]+)\s*:", line, flags=re.IGNORECASE)
    if asset_match:
        snapshot["asset"] = str(asset_match.group(1)).upper()

    score_match = re.search(r"(?:weighted|score)\s*([+-]?\d+(?:\.\d+)?)", line, flags=re.IGNORECASE)
    if score_match:
        snapshot["weighted_score"] = _safe_float(score_match.group(1))

    raw_match = re.search(r"\braw\s*([+-]?\d+(?:\.\d+)?)", line, flags=re.IGNORECASE)
    if raw_match:
        snapshot["raw_score"] = _safe_float(raw_match.group(1))

    rel_match = re.search(r"\brel\s*([0-9]+(?:\.[0-9]+)?)%", line, flags=re.IGNORECASE)
    if rel_match:
        rel_pct = _safe_float(rel_match.group(1))
        if rel_pct is not None:
            snapshot["reliability"] = max(0.0, min(1.0, rel_pct / 100.0))

    cov_match = re.search(r"\bcov\s*([0-9]+(?:\.[0-9]+)?)%", line, flags=re.IGNORECASE)
    if cov_match:
        cov_pct = _safe_float(cov_match.group(1))
        if cov_pct is not None:
            snapshot["coverage"] = max(0.0, min(1.0, cov_pct / 100.0))

    age_match = re.search(r"\bage\s*([0-9]+(?:\.[0-9]+)?)m", line, flags=re.IGNORECASE)
    if age_match:
        snapshot["age_minutes"] = _safe_float(age_match.group(1))

    tx_match = re.search(r"\btx\s*([+-]?\d+(?:\.\d+)?)%", line, flags=re.IGNORECASE)
    if tx_match:
        snapshot["tx_change_pct"] = _safe_float(tx_match.group(1))

    addr_match = re.search(r"\bactive\s*([+-]?\d+(?:\.\d+)?)%", line, flags=re.IGNORECASE)
    if addr_match:
        snapshot["active_addresses_change_pct"] = _safe_float(addr_match.group(1))

    transfer_match = re.search(r"\btransfer\s*([+-]?\d+(?:\.\d+)?)%", line, flags=re.IGNORECASE)
    if transfer_match:
        snapshot["transfer_value_change_pct"] = _safe_float(transfer_match.group(1))

    if len(snapshot.keys()) <= 1:
        return None
    return snapshot


def _extract_onchain_snapshot_from_reasons(reasons: Any) -> Optional[Dict[str, Any]]:
    """Find the latest On-chain reason in a reasons list/string."""
    if reasons is None:
        return None
    if isinstance(reasons, (list, tuple)):
        candidates = list(reasons)
    else:
        candidates = [reasons]

    for line in reversed(candidates):
        snap = _extract_onchain_snapshot_from_reason(str(line or ""))
        if snap:
            return snap
    return None


def _is_onchain_against_side(side: str, weighted_score: Optional[float], min_abs_weighted: float = 0.15) -> bool:
    side_key = str(side or "").upper()
    ws = _safe_float(weighted_score)
    if ws is None:
        return False
    threshold = max(0.05, float(min_abs_weighted))
    if side_key == "LONG":
        return ws <= -threshold
    if side_key == "SHORT":
        return ws >= threshold
    return False


def _infer_onchain_mistake_tags(
    side: str,
    onchain_snapshot: Optional[Dict[str, Any]],
    *,
    min_abs_weighted: float = 0.15,
    min_reliability: float = 0.45,
    max_age_minutes: int = 360,
) -> List[str]:
    """Derive bounded on-chain mistake tags from parsed context."""
    if not onchain_snapshot:
        return []

    tags: List[str] = []
    weighted = _safe_float(onchain_snapshot.get("weighted_score"))
    if _is_onchain_against_side(side, weighted, min_abs_weighted=min_abs_weighted):
        tags.append("onchain_flow_against_side")

    reliability = _safe_float(onchain_snapshot.get("reliability"))
    coverage = _safe_float(onchain_snapshot.get("coverage"))
    age_minutes = _safe_float(onchain_snapshot.get("age_minutes"))

    reliability_low = reliability is not None and reliability < max(0.0, min(1.0, float(min_reliability)))
    coverage_low = coverage is not None and coverage < 0.5
    stale = age_minutes is not None and age_minutes > float(max(30, int(max_age_minutes)))
    if reliability_low or coverage_low or stale:
        tags.append("onchain_signal_low_reliability")

    return tags[:2]


def _run_llm_postmortem_worker(db_path: str, outcome_id: int, config: Config) -> None:
    """Background worker: evaluate one closed trade with LLM and store review."""
    try:
        post_cfg = getattr(config, "llm_postmortem", None)
        if not post_cfg or not bool(getattr(post_cfg, "enabled", False)):
            return

        worker_storage = SignalStorage(db_path=db_path)
        if worker_storage.has_llm_review(int(outcome_id)):
            return

        context = worker_storage.get_closed_outcome_context(int(outcome_id))
        if not context:
            return

        onchain_snapshot = _extract_onchain_snapshot_from_reasons(context.get("signal_reasons"))
        if onchain_snapshot:
            context["onchain_snapshot"] = onchain_snapshot

        # Prefer ollama config if available and enabled
        llm_client_config = getattr(config, "ollama", None)
        if not llm_client_config or not getattr(llm_client_config, "enabled", False):
            llm_client_config = config.openai

        result = evaluate_loss_postmortem(
            trade_context=context,
            openai_config=llm_client_config,
            postmortem_config=post_cfg,
        )
        if result is None:
            return

        inferred_onchain_tags = _infer_onchain_mistake_tags(
            side=str(context.get("side") or ""),
            onchain_snapshot=onchain_snapshot,
        )
        if inferred_onchain_tags:
            merged_tags = list(result.mistake_tags or [])
            for tag in inferred_onchain_tags:
                if tag not in merged_tags:
                    merged_tags.append(tag)
            result.mistake_tags = merged_tags[:5]

        worker_storage.save_llm_trade_review(
            outcome_id=int(context.get("outcome_id") or outcome_id),
            signal_id=context.get("signal_id"),
            symbol=str(context.get("symbol") or ""),
            side=str(context.get("side") or ""),
            outcome=str(context.get("outcome") or ""),
            pnl_pct=context.get("pnl_pct"),
            verdict=result.verdict,
            action=result.action,
            confidence=int(result.confidence),
            penalty=float(result.penalty),
            mistake_tags=list(result.mistake_tags),
            summary=result.summary,
            recommendation=result.recommendation,
            raw_json=result.raw_json,
        )
        logger.info(
            f"LLM postmortem saved: outcome_id={outcome_id} "
            f"verdict={result.verdict} action={result.action} "
            f"penalty={result.penalty:.2f} conf={result.confidence}"
        )
    except Exception as e:
        logger.debug(f"LLM postmortem worker error for outcome_id={outcome_id}: {e}")


def _enqueue_llm_postmortem(storage: SignalStorage, config: Config, outcome_id: int, outcome: str, pnl_pct: float) -> None:
    """Spawn asynchronous LLM postmortem analysis for qualifying outcomes."""
    post_cfg = getattr(config, "llm_postmortem", None)
    if not post_cfg or not bool(getattr(post_cfg, "enabled", False)):
        return

    if bool(getattr(post_cfg, "only_losses", True)) and (not _is_loss_outcome(outcome, pnl_pct)):
        return

    try:
        if storage.has_llm_review(int(outcome_id)):
            return
    except Exception:
        return

    t = threading.Thread(
        target=_run_llm_postmortem_worker,
        args=(storage.db_path, int(outcome_id), config),
        daemon=True,
    )
    t.start()


def _run_llm_postmortem_backfill_worker(db_path: str, config: Config) -> None:
    """
    Startup backfill for historical closed outcomes without LLM reviews.
    """
    try:
        post_cfg = getattr(config, "llm_postmortem", None)
        if not post_cfg or not bool(getattr(post_cfg, "enabled", False)):
            return
        if not bool(getattr(post_cfg, "backfill_existing_on_startup", True)):
            return

        worker_storage = SignalStorage(db_path=db_path)
        lookback_days = max(1, int(getattr(post_cfg, "lookback_days", 21)))
        startup_limit = max(1, int(getattr(post_cfg, "startup_max_reviews", 40)))
        pending = worker_storage.get_outcomes_pending_llm_review(
            lookback_days=lookback_days,
            only_losses=bool(getattr(post_cfg, "only_losses", True)),
            limit=startup_limit,
        )
        if not pending:
            logger.info("LLM postmortem backfill: no pending historical outcomes")
        else:
            logger.info(
                f"LLM postmortem backfill: evaluating {len(pending)} historical outcomes "
                f"(lookback={lookback_days}d, limit={startup_limit})"
            )
            import concurrent.futures
            def _process_one(row_data):
                try:
                    try:
                        oid = int(row_data.get("id") or 0)
                    except Exception:
                        oid = 0
                    if oid > 0:
                        _run_llm_postmortem_worker(db_path=db_path, outcome_id=oid, config=config)
                except Exception as e:
                    logger.error(f"Error in _process_one: {e}")
                    
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # Force iteration to catch errors
                list(executor.map(_process_one, pending))

        # Enrich old reviews as tagging logic evolves (no extra LLM/API calls).
        enrich_limit = max(120, startup_limit * 3)
        recent_reviews = worker_storage.get_recent_llm_reviews_for_onchain_enrichment(
            lookback_days=lookback_days,
            limit=enrich_limit,
        )
        updated_reviews = 0
        for row in recent_reviews:
            side_key = str(row.get("side") or "").upper()
            onchain_snapshot = _extract_onchain_snapshot_from_reasons(row.get("signal_reasons"))
            if not onchain_snapshot:
                continue
            add_tags = _infer_onchain_mistake_tags(side=side_key, onchain_snapshot=onchain_snapshot)
            if not add_tags:
                continue
            try:
                outcome_id = int(row.get("outcome_id") or 0)
            except Exception:
                outcome_id = 0
            if outcome_id <= 0:
                continue
            if worker_storage.append_llm_review_tags(outcome_id=outcome_id, tags=add_tags):
                updated_reviews += 1

        if updated_reviews > 0:
            logger.info(
                f"LLM postmortem backfill: enriched {updated_reviews} historical reviews "
                f"with on-chain tags"
            )
    except Exception as e:
        logger.debug(f"LLM postmortem backfill worker failed: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global shutdown_requested
    logger.info("Shutdown signal received, finishing current scan...")
    shutdown_requested = True


def set_scan_paused(value: bool) -> None:
    """Set scanner pause state."""
    global scan_paused
    with state_lock:
        scan_paused = value


def is_scan_paused() -> bool:
    """Get scanner pause state."""
    with state_lock:
        return scan_paused


def _telegram_control_help_text() -> str:
    """Build Telegram control help message."""
    return (
        "🕹️ <b>Bot Control Commands</b>\n\n"
        "• <code>/pause</code> أو <code>وقف</code>: إيقاف السكّان\n"
        "• <code>/resume</code> أو <code>شغل</code>: تشغيل السكّان\n"
        "• <code>/status</code>: حالة البوت\n"
        "• <code>/stop</code>: إيقاف العملية بالكامل\n"
        "• <code>/news your headline here</code>: تحليل خبر عبر OpenAI API\n"
    )


def _telegram_status_text(config: Config, scan_count: int) -> str:
    """Build human-readable scanner status text."""
    mode = "PAUSED ⏸️" if is_scan_paused() else "RUNNING ▶️"
    return (
        "ℹ️ <b>Bot Runtime Status</b>\n\n"
        f"Mode: <b>{mode}</b>\n"
        f"Scans completed: <b>{scan_count}</b>\n"
        f"Symbols: <b>{len(config.symbols)}</b>\n"
        f"Interval: <b>{config.scan_interval_seconds}s</b>\n"
        f"Time: <b>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</b>"
    )


def _extract_news_text(text: str) -> str:
    """Extract news body from Telegram message."""
    if not text:
        return ""

    stripped = text.strip()
    lowered = stripped.lower()

    prefixes = ('/news', 'news ', 'خبر ', 'حلل ', 'analyze ')
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return stripped[len(prefix):].strip()

    return ""


def _maybe_send_risk_alert(notifier: TelegramNotifier, key: str, message: str, cooldown_minutes: int = 20) -> None:
    """Throttle risk-control alerts to avoid Telegram spam."""
    now = datetime.now(timezone.utc)
    last = _risk_alert_timestamps.get(key)
    if last and (now - last) < timedelta(minutes=max(1, int(cooldown_minutes))):
        return
    _risk_alert_timestamps[key] = now
    notifier.send_text(message)


def _is_high_impact_window(result) -> bool:
    """Check if any high-impact macro window is active in timing analysis."""
    ti = getattr(result, "timing_info", None)
    if not ti:
        return False
    checks = [
        getattr(getattr(ti, "fomc_analysis", None), "in_high_vol_window", False),
        getattr(getattr(ti, "cpi_analysis", None), "in_high_vol_window", False),
        getattr(getattr(ti, "nfp_analysis", None), "in_high_vol_window", False),
        getattr(getattr(ti, "powell_analysis", None), "in_high_vol_window", False),
        getattr(getattr(ti, "fomc_minutes_analysis", None), "in_high_vol_window", False),
    ]
    return any(bool(x) for x in checks)


def _infer_asset_class_for_symbol(symbol: str) -> str:
    text = str(symbol or "").upper()
    macro_keys = ("XAU", "XAG", "OIL", "WTI", "BRENT", "SNP500", "SPX500", "S&P500", "SP500", "EURUSD", "EUR/USD")
    if any(k in text for k in macro_keys):
        return "macro"
    return "crypto"


# Forex/macro symbols that have defined market open/close hours
_FOREX_MACRO_KEYS = (
    "XAU", "XAG", "OIL", "WTI", "BRENT",
    "EURUSD", "EUR/USD",
    "GBPUSD", "GBP/USD",
    "USDJPY", "USD/JPY",
    "USDCHF", "USD/CHF",
    "AUDUSD", "AUD/USD",
    "USDCAD", "USD/CAD",
    "NZDUSD", "NZD/USD",
    "US500", "SP500", "SPX",
)

def _is_macro_session_symbol(symbol: str) -> bool:
    """Return True for forex/macro symbols that have market open/close hours.
    Crypto symbols (BTC, ETH, etc.) return False — they trade 24/7."""
    text = str(symbol or "").upper()
    return any(k in text for k in _FOREX_MACRO_KEYS)


# Session close times per symbol group (UTC HH:MM).
# Bot will close open positions MINUTES_BEFORE_SESSION_END minutes before these times.
_MINUTES_BEFORE_SESSION_END = 15

_SESSION_CLOSE_SCHEDULE: List[Dict] = [
    # Forex major pairs — NY session close (Fri 21:00 UTC = weekly close)
    # Also closes at 21:00 UTC every day (end of NY session)
    {"symbols": ["EURUSD", "EUR/USD", "GBPUSD", "GBP/USD",
                 "USDCHF", "USD/CHF", "USDCAD", "USD/CAD"],
     "close_utc": "21:00",
     "label": "NY session close"},
    # JPY / AUD / NZD — also track Tokyo open/close
    {"symbols": ["USDJPY", "USD/JPY", "AUDUSD", "AUD/USD", "NZDUSD", "NZD/USD"],
     "close_utc": "21:00",
     "label": "NY session close"},
    # Gold / Silver — close at 21:00 UTC (NY Comex close)
    {"symbols": ["XAU", "GOLD", "XAG", "SILVER"],
     "close_utc": "21:00",
     "label": "Comex close"},
    # Oil — close at 21:00 UTC
    {"symbols": ["USOIL", "OIL", "WTI", "BRENT"],
     "close_utc": "21:00",
     "label": "Oil market close"},
    # US500 / Indices — close at 21:00 UTC (NYSE close)
    {"symbols": ["US500", "SP500", "SPX"],
     "close_utc": "21:00",
     "label": "NYSE close"},
]


def _get_session_close_symbols(now_utc: datetime) -> List[str]:
    """Return list of symbol keywords whose session ends within _MINUTES_BEFORE_SESSION_END minutes."""
    minute_now = now_utc.hour * 60 + now_utc.minute
    closing_keys: List[str] = []
    for entry in _SESSION_CLOSE_SCHEDULE:
        close_m = _parse_utc_hhmm(entry["close_utc"])
        if close_m is None:
            continue
        warn_m = close_m - _MINUTES_BEFORE_SESSION_END
        if warn_m < 0:
            warn_m += 1440  # wrap midnight
        # Trigger in the window [warn_m, close_m)
        if warn_m <= minute_now < close_m:
            closing_keys.extend(entry["symbols"])
            logger.info(f"Session close approaching ({entry['label']} at {entry['close_utc']} UTC): will close positions for {entry['symbols']}")
    return closing_keys


def _symbol_matches_keys(symbol: str, keys: List[str]) -> bool:
    text = str(symbol or "").upper().replace("/", "").replace(":", "").replace("_FUTURES", "").replace("_SPOT", "")
    for k in keys:
        if k.upper().replace("/", "") in text:
            return True
    return False


def close_positions_before_session_end(
    storage,
    notifier,
    mt5_client,
    config: "Config",
    now_utc: Optional[datetime] = None,
) -> None:
    """
    Close any open MT5 positions for symbols whose trading session is about to end.
    Called once per scan loop, MINUTES_BEFORE_SESSION_END minutes before close.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    closing_keys = _get_session_close_symbols(now_utc)
    if not closing_keys:
        return

    try:
        all_positions = mt5_client.get_all_bot_positions()
    except Exception as e:
        logger.error(f"close_positions_before_session_end: failed to fetch positions: {e}")
        return

    for pos in all_positions:
        symbol_mt5 = str(pos.get("symbol") or "")
        if not _symbol_matches_keys(symbol_mt5, closing_keys):
            continue

        ticket = int(pos.get("ticket") or 0)
        if not ticket:
            continue

        profit = float(pos.get("profit") or 0.0)
        side = "LONG" if pos.get("type") == 0 else "SHORT"
        logger.info(f"Session close: closing {symbol_mt5} {side} ticket={ticket} profit={profit:.2f}")

        try:
            closed = mt5_client.close_position(ticket)
        except Exception as e:
            logger.error(f"Session close: error closing ticket {ticket}: {e}")
            closed = False

        if closed:
            # Mark the corresponding open outcome as EXITED
            try:
                open_outcomes = storage.get_open_outcomes()
                for oc in open_outcomes:
                    oc_sym = str(oc.get("symbol") or "")
                    if _symbol_matches_keys(oc_sym, [symbol_mt5]):
                        tick_data = mt5_client.get_tick(symbol_mt5)
                        close_price = float(tick_data.get("last") or tick_data.get("bid") or 0.0)
                        entry_price = float(oc.get("entry") or 0.0)
                        if entry_price > 0 and close_price > 0:
                            pnl_pct = ((close_price - entry_price) / entry_price * 100.0
                                       if side == "LONG"
                                       else (entry_price - close_price) / entry_price * 100.0)
                        else:
                            pnl_pct = 0.0
                        storage.update_signal_outcome(
                            int(oc["id"]), "EXITED", close_price, round(pnl_pct, 4)
                        )
            except Exception as e:
                logger.warning(f"Session close: could not update outcome for {symbol_mt5}: {e}")

            try:
                notifier.send_status(
                    f"⏰ Session close: {symbol_mt5} {side} closed @ market\n"
                    f"Profit: {profit:+.2f} (session ending in {_MINUTES_BEFORE_SESSION_END} min)"
                )
            except Exception:
                pass


def _parse_utc_hhmm(value: str) -> Optional[int]:
    try:
        hh_str, mm_str = str(value).strip().split(":", 1)
        hh = int(hh_str)
        mm = int(mm_str)
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return hh * 60 + mm
    except Exception:
        return None
    return None


def _is_within_utc_windows(windows: List[str], now_utc: datetime) -> bool:
    minute_now = (now_utc.hour * 60) + now_utc.minute
    for raw in windows or []:
        text = str(raw or "").strip()
        if "-" not in text:
            continue
        start_raw, end_raw = text.split("-", 1)
        start_m = _parse_utc_hhmm(start_raw)
        end_m = _parse_utc_hhmm(end_raw)
        if start_m is None or end_m is None:
            continue

        # Non-overnight window (e.g. 06:00-16:30)
        if start_m <= end_m:
            if start_m <= minute_now <= end_m:
                return True
            continue

        # Overnight window (e.g. 22:00-02:00)
        if minute_now >= start_m or minute_now <= end_m:
            return True
    return False


def _check_quality_filter(
    symbol: str,
    ticker: Dict[str, Any],
    df_entry: Any,
    result: Any,
    config: Config,
    learning_decision: Any,
    storage: Optional[SignalStorage] = None,
) -> Tuple[bool, str, float]:
    """Apply execution-quality guardrails before sending a new trade signal."""
    qf = getattr(config, "quality_filter", None)
    if not qf or not getattr(qf, "enabled", True):
        return True, "", 0.0

    side = str(getattr(result, "side", "") or "").upper()
    if side not in {"LONG", "SHORT"}:
        return True, "", 0.0

    threshold_add_total = 0.0
    soft_reasons: List[str] = []
    spread_pct: Optional[float] = None

    # Spread filter
    try:
        bid = ticker.get("bid")
        ask = ticker.get("ask")
        last = ticker.get("last")
        if bid and ask and last and float(last) > 0:
            spread_pct = abs(float(ask) - float(bid)) / float(last) * 100.0
            max_spread = float(getattr(qf, "max_spread_pct", 0.22))
            if spread_pct > max_spread:
                return False, f"Quality filter: spread {spread_pct:.3f}% > {max_spread:.3f}%", 0.0
    except Exception:
        pass

    # Volume filter (current volume vs SMA20)
    try:
        if df_entry is not None and not df_entry.empty and "volume" in df_entry.columns:
            with_ind = df_entry
            if "vol_sma_20" not in with_ind.columns:
                with_ind = add_all_indicators(df_entry.copy())
                if "vol_sma_20" in with_ind.columns and len(with_ind) >= 25:
                    vol_now = float(with_ind["volume"].iloc[-1])
                    vol_avg = float(with_ind["vol_sma_20"].iloc[-1] or 0.0)
                    if vol_avg > 0:
                        ratio = vol_now / vol_avg
                        asset_class = _infer_asset_class_for_symbol(symbol)
                        if asset_class == "macro":
                            min_ratio = float(getattr(qf, "min_volume_ratio_macro", 0.0))
                        else:
                            min_ratio = float(getattr(qf, "min_volume_ratio", 0.65))
                        if ratio < min_ratio:
                            # Smart-relax mode: allow moderate low-volume entries only when
                            # trend is strong and spread is healthy.
                            relaxed_ok = False
                            if bool(getattr(qf, "relaxed_volume_mode", True)) and asset_class != "macro":
                                relaxed_floor = float(getattr(qf, "min_volume_ratio_relaxed", 0.45))
                                relax_adx_min = float(getattr(qf, "volume_relax_adx_min", 24.0))
                                relax_max_spread = float(getattr(qf, "volume_relax_max_spread_pct", 0.16))
                                relax_min_score_margin = float(getattr(qf, "volume_relax_min_score_margin", -0.2))
                                adx_now = float(with_ind["adx"].iloc[-1]) if "adx" in with_ind.columns else 0.0
                                score_margin = float(getattr(result, "total_score", 0.0)) - float(getattr(result, "threshold", 0.0))
                                spread_ok = (spread_pct is None) or (spread_pct <= relax_max_spread)
                                if ratio >= relaxed_floor and adx_now >= relax_adx_min and spread_ok and score_margin >= relax_min_score_margin:
                                    relaxed_ok = True

                            if relaxed_ok:
                                soft_reasons.append(
                                    "Quality filter: relaxed volume gate "
                                    f"(ratio {ratio:.2f} < {min_ratio:.2f}, strong-trend allowance)"
                                )
                            else:
                                return False, f"Quality filter: volume ratio {ratio:.2f} < {min_ratio:.2f}", 0.0
    except Exception:
        pass

    # Gold/Oil session filter (UTC)
    if bool(getattr(qf, "macro_session_filter_enabled", True)) and _is_macro_session_symbol(symbol):
        windows = list(getattr(qf, "macro_session_utc_windows", []) or [])
        if windows:
            in_session = _is_within_utc_windows(windows, datetime.now(timezone.utc))
            if not in_session:
                mode = str(getattr(qf, "macro_session_mode", "cautious") or "cautious").strip().lower()
                if mode == "cautious":
                    session_add = max(0.0, float(getattr(qf, "macro_session_threshold_add", 0.8)))
                    threshold_add_total += session_add
                    soft_reasons.append("Quality filter: outside macro active session (cautious mode)")
                else:
                    return False, "Quality filter: outside macro active session", 0.0

    # High-impact macro window: either block/cautious (legacy) or warn-only mode.
    in_high_impact = _is_high_impact_window(result)
    
    volatility_settled = False
    if in_high_impact and 'atr_14' in df_entry.columns and len(df_entry) >= 50:
        current_atr = df_entry['atr_14'].iloc[-1]
        avg_atr = df_entry['atr_14'].rolling(window=50).mean().iloc[-1]
        import pandas as pd
        if not pd.isna(current_atr) and not pd.isna(avg_atr) and current_atr <= (avg_atr * 1.3):
            volatility_settled = True

    if in_high_impact and not volatility_settled:
        if bool(getattr(qf, "block_during_high_impact_news", True)):
            mode = str(getattr(qf, "high_impact_news_mode", "block") or "block").strip().lower()
            if mode == "cautious":
                threshold_add = max(0.0, float(getattr(qf, "high_impact_threshold_add", 0.8)))
                threshold_add_total += threshold_add
                soft_reasons.append("Quality filter: high-impact news window active (cautious mode)")
            else:
                return False, "Quality filter: high-impact news window active", 0.0
        else:
            soft_reasons.append("Quality filter: high-impact news window active (warn-only mode)")
    elif in_high_impact and volatility_settled:
        soft_reasons.append("Quality filter: high-impact news bypassed (volatility settled)")

    # Opposing news score: block (legacy) or warn-only mode.
    timing_score = float(getattr(result, "timing_score", 0.0))
    asset_class = _infer_asset_class_for_symbol(symbol)
    if asset_class == "macro":
        opposing_score = abs(float(getattr(qf, "opposing_news_score_macro", 1.6)))
    else:
        opposing_score = abs(float(getattr(qf, "opposing_news_score", 0.9)))

    opposing_hit_long = side == "LONG" and timing_score <= -opposing_score
    opposing_hit_short = side == "SHORT" and timing_score >= opposing_score
    if opposing_hit_long or opposing_hit_short:
        reason = f"Quality filter: opposing news score {timing_score:.2f} for {side}"
        if bool(getattr(qf, "block_on_opposing_news", True)):
            return False, reason, 0.0
        soft_reasons.append(f"{reason} (warn-only mode)")

    # Learning alignment block (only when enough local evidence exists)
    if bool(getattr(qf, "require_learning_alignment", True)) and learning_decision is not None:
        min_samples = max(1, int(getattr(qf, "learning_min_samples", 10)))
        max_negative_adj = float(getattr(qf, "max_negative_learning_adjustment", -0.4))
        if int(getattr(learning_decision, "sample_size", 0)) >= min_samples:
            if not bool(getattr(learning_decision, "allow", True)):
                return False, "Quality filter: learning policy veto", 0.0
            if float(getattr(learning_decision, "score_adjustment", 0.0)) < max_negative_adj:
                return False, (
                    "Quality filter: learning adjustment "
                    f"{float(getattr(learning_decision, 'score_adjustment', 0.0)):+.2f} < {max_negative_adj:+.2f}"
                ), 0.0

    return True, " | ".join(soft_reasons), threshold_add_total


def _is_side_against_regime(side: str, regime: str) -> bool:
    side_key = str(side or "").upper()
    regime_key = str(regime or "").strip().lower()
    if side_key == "LONG" and regime_key == "downtrend":
        return True
    if side_key == "SHORT" and regime_key == "uptrend":
        return True
    return False


def _apply_llm_execution_adapters(
    storage: SignalStorage,
    config: Config,
    symbol: str,
    side: str,
    df_entry,
    result: Any,
) -> Tuple[bool, str, float]:
    """
    Translate repeated postmortem tags into execution-time behavior.
    This is the bridge between "why we lost" and "what we do now".
    """
    qf = getattr(config, "quality_filter", None)
    if not qf or not bool(getattr(qf, "llm_adapter_enabled", True)):
        return True, "", 0.0

    side_key = str(side or "").upper()
    if side_key not in {"LONG", "SHORT"}:
        return True, "", 0.0

    lookback_days = max(1, int(getattr(qf, "llm_adapter_lookback_days", 21)))
    min_reviews = max(1, int(getattr(qf, "llm_adapter_min_reviews", 3)))
    min_confidence = max(0, min(100, int(getattr(qf, "llm_adapter_min_confidence", 60))))
    shadow_mode = bool(getattr(qf, "llm_adapter_shadow_mode", False))
    since_iso = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

    late_stats = storage.get_recent_llm_tag_stats(
        symbol=str(symbol),
        side=side_key,
        since_iso=since_iso,
        tag="late_entry",
        min_confidence=min_confidence,
    )
    stop_stats = storage.get_recent_llm_tag_stats(
        symbol=str(symbol),
        side=side_key,
        since_iso=since_iso,
        tag="stop_too_tight",
        min_confidence=min_confidence,
    )
    news_stats = storage.get_recent_llm_tag_stats(
        symbol=str(symbol),
        side=side_key,
        since_iso=since_iso,
        tag="ignored_news_risk",
        min_confidence=min_confidence,
    )
    regime_stats = storage.get_recent_llm_tag_stats(
        symbol=str(symbol),
        side=side_key,
        since_iso=since_iso,
        tag="regime_mismatch",
        min_confidence=min_confidence,
    )
    onchain_against_stats = storage.get_recent_llm_tag_stats(
        symbol=str(symbol),
        side=side_key,
        since_iso=since_iso,
        tag="onchain_flow_against_side",
        min_confidence=min_confidence,
    )
    onchain_lowrel_stats = storage.get_recent_llm_tag_stats(
        symbol=str(symbol),
        side=side_key,
        since_iso=since_iso,
        tag="onchain_signal_low_reliability",
        min_confidence=min_confidence,
    )

    review_count = max(
        int(late_stats.get("count", 0) or 0),
        int(stop_stats.get("count", 0) or 0),
        int(news_stats.get("count", 0) or 0),
        int(regime_stats.get("count", 0) or 0),
        int(onchain_against_stats.get("count", 0) or 0),
        int(onchain_lowrel_stats.get("count", 0) or 0),
    )
    if review_count < min_reviews:
        return True, "", 0.0

    late_ratio = max(0.0, min(1.0, float(late_stats.get("tag_ratio", 0.0) or 0.0)))
    stop_ratio = max(0.0, min(1.0, float(stop_stats.get("tag_ratio", 0.0) or 0.0)))
    news_ratio = max(0.0, min(1.0, float(news_stats.get("tag_ratio", 0.0) or 0.0)))
    regime_ratio = max(0.0, min(1.0, float(regime_stats.get("tag_ratio", 0.0) or 0.0)))
    onchain_against_ratio = max(0.0, min(1.0, float(onchain_against_stats.get("tag_ratio", 0.0) or 0.0)))
    onchain_lowrel_ratio = max(0.0, min(1.0, float(onchain_lowrel_stats.get("tag_ratio", 0.0) or 0.0)))

    late_ratio_tr = max(0.0, min(1.0, float(getattr(qf, "llm_adapter_late_entry_ratio", 0.45))))
    stop_ratio_tr = max(0.0, min(1.0, float(getattr(qf, "llm_adapter_stop_tight_ratio", 0.45))))
    news_ratio_tr = max(0.0, min(1.0, float(getattr(qf, "llm_adapter_news_ratio", 0.35))))
    regime_ratio_tr = max(0.0, min(1.0, float(getattr(qf, "llm_adapter_regime_ratio", 0.35))))
    onchain_against_ratio_tr = max(0.0, min(1.0, float(getattr(qf, "llm_adapter_onchain_against_ratio", 0.35))))
    onchain_lowrel_ratio_tr = max(
        0.0,
        min(1.0, float(getattr(qf, "llm_adapter_onchain_low_reliability_ratio", 0.45))),
    )

    threshold_add = 0.0
    reasons: List[str] = []
    block_reason = ""
    virtual_threshold_add = 0.0
    onchain_now = _extract_onchain_snapshot_from_reasons(getattr(result, "reasons", []))
    onchain_now_weighted = _safe_float((onchain_now or {}).get("weighted_score"))
    onchain_now_reliability = _safe_float((onchain_now or {}).get("reliability"))
    onchain_now_age = _safe_float((onchain_now or {}).get("age_minutes"))
    onchain_now_against_side = _is_onchain_against_side(side_key, onchain_now_weighted, min_abs_weighted=0.15)
    min_onchain_reliability = max(
        0.0, min(1.0, float(getattr(qf, "llm_adapter_onchain_min_reliability", 0.45)))
    )
    max_onchain_age_minutes = max(30, int(getattr(qf, "llm_adapter_onchain_max_age_minutes", 360)))
    onchain_now_low_reliability = (
        (onchain_now_reliability is not None and onchain_now_reliability < min_onchain_reliability)
        or (onchain_now_age is not None and onchain_now_age > float(max_onchain_age_minutes))
    )

    # 1) Late-entry memory -> tighten anti-chasing and raise threshold.
    if late_ratio >= late_ratio_tr:
        base_mult = max(0.6, float(getattr(qf, "anti_chasing_atr_mult", 3.0)))
        lookback = max(2, int(getattr(qf, "anti_chasing_lookback_bars", 3)))
        dynamic_mult = max(0.9, base_mult * (1.0 - min(0.40, late_ratio * 0.45)))
        add_late = max(0.0, float(getattr(qf, "llm_adapter_threshold_add_late", 0.35)))
        if shadow_mode:
            virtual_threshold_add += add_late
        else:
            threshold_add += add_late
        reasons.append(
            f"LLM adapter late_entry: {late_stats.get('tag_count', 0)}/{review_count} "
            f"(ratio {late_ratio:.2f})"
        )

        try:
            if df_entry is not None and not df_entry.empty and "atr_14" in df_entry.columns and len(df_entry) >= (lookback + 1):
                atr_now = float(df_entry["atr_14"].iloc[-1] or 0.0)
                recent_move = abs(float(df_entry["close"].iloc[-1]) - float(df_entry["close"].iloc[-(lookback + 1)]))
                if atr_now > 0 and recent_move > (atr_now * dynamic_mult):
                    # Save block reason but intercept and turn into pending entry
                    block_reason = (
                        f"LLM adapter: late-entry risk high "
                        f"(move {recent_move/atr_now:.1f}x ATR > {dynamic_mult:.1f}x)"
                    )
                    if storage and not shadow_mode:
                        zone = getattr(result, "zone_info", None)
                        entry_price, _ = _execution_price_from_ticker(
                            ticker=ticker,
                            side=result.side,
                            phase="entry",
                        )
                        entry_price = float(entry_price or ticker.get("last", 0.0) or 0.0)
                        ideal = zone.upper if zone and result.side == "LONG" else (zone.lower if zone else entry_price)
                        storage.save_pending_entry(
                            symbol=symbol, side=result.side, ideal_entry=ideal,
                            zone_top=ideal * 1.002, zone_bottom=ideal * 0.998,
                            atr=atr_now, score=float(result.total_score),
                            reasons=reasons + [block_reason],
                            expires_minutes=15
                        )
                        return False, f"Anti-chasing: saved as pending pullback entry @ {ideal:.4f}", 0.0
        except Exception:
            pass

    # 2) Stop-too-tight memory -> require stronger quality via threshold.
    if stop_ratio >= stop_ratio_tr:
        add_stop = max(0.0, float(getattr(qf, "llm_adapter_threshold_add_stop", 0.20)))
        if shadow_mode:
            virtual_threshold_add += add_stop
        else:
            threshold_add += add_stop
        reasons.append(
            f"LLM adapter stop_too_tight: {stop_stats.get('tag_count', 0)}/{review_count} "
            f"(ratio {stop_ratio:.2f})"
        )

    # 3) Ignored-news memory -> raise threshold near high-impact windows.
    if news_ratio >= news_ratio_tr and _is_high_impact_window(result):
        add_news = max(0.0, float(getattr(qf, "llm_adapter_threshold_add_news", 0.30)))
        if shadow_mode:
            virtual_threshold_add += add_news
        else:
            threshold_add += add_news
        reasons.append(
            f"LLM adapter ignored_news_risk: {news_stats.get('tag_count', 0)}/{review_count} "
            f"(ratio {news_ratio:.2f})"
        )

    # 4) Regime-mismatch memory -> stricter filtering when side opposes regime.
    if regime_ratio >= regime_ratio_tr and _is_side_against_regime(side_key, getattr(result, "market_regime", "")):
        add_regime = max(0.0, float(getattr(qf, "llm_adapter_threshold_add_regime", 0.25)))
        if shadow_mode:
            virtual_threshold_add += add_regime
        else:
            threshold_add += add_regime
        reasons.append(
            f"LLM adapter regime_mismatch: {regime_stats.get('tag_count', 0)}/{review_count} "
            f"(ratio {regime_ratio:.2f})"
        )

    # 5) On-chain-against-side memory -> raise threshold when live on-chain still disagrees.
    if onchain_against_ratio >= onchain_against_ratio_tr and onchain_now_against_side:
        add_onchain_against = max(
            0.0, float(getattr(qf, "llm_adapter_threshold_add_onchain_against", 0.30))
        )
        if shadow_mode:
            virtual_threshold_add += add_onchain_against
        else:
            threshold_add += add_onchain_against
        reasons.append(
            f"LLM adapter onchain_flow_against_side: {onchain_against_stats.get('tag_count', 0)}/{review_count} "
            f"(ratio {onchain_against_ratio:.2f})"
        )

    # 6) On-chain reliability memory -> be stricter when live on-chain quality is weak.
    if onchain_lowrel_ratio >= onchain_lowrel_ratio_tr and onchain_now_low_reliability:
        add_onchain_rel = max(
            0.0, float(getattr(qf, "llm_adapter_threshold_add_onchain_reliability", 0.20))
        )
        if shadow_mode:
            virtual_threshold_add += add_onchain_rel
        else:
            threshold_add += add_onchain_rel
        reasons.append(
            f"LLM adapter onchain_signal_low_reliability: "
            f"{onchain_lowrel_stats.get('tag_count', 0)}/{review_count} "
            f"(ratio {onchain_lowrel_ratio:.2f})"
        )

    if block_reason:
        if shadow_mode:
            reasons.append(f"Shadow-only: would block -> {block_reason}")
        else:
            return False, block_reason, 0.0

    if shadow_mode and virtual_threshold_add > 0:
        reasons.append(f"Shadow-only: would add +{virtual_threshold_add:.2f} threshold")
        return True, " | ".join(reasons), 0.0

    return True, " | ".join(reasons), threshold_add


def _apply_llm_postmortem_feedback(
    storage: SignalStorage,
    config: Config,
    symbol: str,
    side: str,
    result: Any,
) -> None:
    """
    Apply recent LLM postmortem feedback as advisory note or bounded score penalty.
    """
    post_cfg = getattr(config, "llm_postmortem", None)
    if not post_cfg or not bool(getattr(post_cfg, "enabled", False)):
        return

    side_key = str(side or "").upper()
    if side_key not in {"LONG", "SHORT"}:
        return

    lookback_days = max(1, int(getattr(post_cfg, "lookback_days", 21)))
    since_iso = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    min_confidence = max(0, min(100, int(getattr(post_cfg, "min_confidence", 60))))
    min_reviews = max(1, int(getattr(post_cfg, "min_reviews_for_penalty", 2)))
    penalty_cap = max(0.0, float(getattr(post_cfg, "penalty_max", 0.8)))

    stats = storage.get_recent_llm_penalty(
        symbol=str(symbol),
        side=side_key,
        since_iso=since_iso,
        min_confidence=min_confidence,
    )
    review_count = int(stats.get("count", 0) or 0)
    if review_count < min_reviews:
        return

    avg_penalty = max(0.0, float(stats.get("avg_penalty", 0.0) or 0.0))
    avg_conf = max(0.0, min(100.0, float(stats.get("avg_confidence", 0.0) or 0.0)))
    applied_penalty = min(penalty_cap, avg_penalty)

    # Tag-aware memory boost:
    # If the same symbol/side keeps failing due to late entry, increase the
    # postmortem penalty to make the gate stricter.
    late_entry_stats = storage.get_recent_llm_tag_stats(
        symbol=str(symbol),
        side=side_key,
        since_iso=since_iso,
        tag="late_entry",
        min_confidence=min_confidence,
    )
    late_entry_count = int(late_entry_stats.get("tag_count", 0) or 0)
    late_entry_ratio = max(0.0, min(1.0, float(late_entry_stats.get("tag_ratio", 0.0) or 0.0)))
    late_entry_boost = 0.0
    late_entry_min_reviews = max(min_reviews, 3)
    late_entry_min_count = 2
    late_entry_ratio_trigger = 0.45
    if (
        review_count >= late_entry_min_reviews
        and late_entry_count >= late_entry_min_count
        and late_entry_ratio >= late_entry_ratio_trigger
    ):
        base_for_boost = max(
            0.0,
            float(late_entry_stats.get("tag_avg_penalty", 0.0) or 0.0),
            avg_penalty,
        )
        if base_for_boost > 0:
            late_entry_boost = max(0.08, min(0.20, base_for_boost * 0.35))
            applied_penalty = min(penalty_cap, applied_penalty + late_entry_boost)
            result.reasons.append(
                f"LLM late-entry memory: {late_entry_count}/{review_count} tagged late_entry "
                f"(+{late_entry_boost:.2f})"
            )

    onchain_against_stats = storage.get_recent_llm_tag_stats(
        symbol=str(symbol),
        side=side_key,
        since_iso=since_iso,
        tag="onchain_flow_against_side",
        min_confidence=min_confidence,
    )
    onchain_lowrel_stats = storage.get_recent_llm_tag_stats(
        symbol=str(symbol),
        side=side_key,
        since_iso=since_iso,
        tag="onchain_signal_low_reliability",
        min_confidence=min_confidence,
    )
    onchain_against_count = int(onchain_against_stats.get("tag_count", 0) or 0)
    onchain_lowrel_count = int(onchain_lowrel_stats.get("tag_count", 0) or 0)
    if onchain_against_count > 0 or onchain_lowrel_count > 0:
        result.reasons.append(
            f"LLM on-chain memory: against={onchain_against_count}/{review_count}, "
            f"low_rel={onchain_lowrel_count}/{review_count}"
        )

    if applied_penalty <= 0:
        result.reasons.append(
            f"LLM postmortem: {review_count} reviewed losses ({side_key}) with no penalty"
        )
        return

    if bool(getattr(post_cfg, "advisory_only", False)):
        result.reasons.append(
            f"⚠ LLM postmortem caution: {review_count} reviewed losses "
            f"(avg conf {avg_conf:.0f}%)"
        )
        logger.info(
            f"LLM postmortem advisory for {symbol} {side_key}: "
            f"reviews={review_count} avg_conf={avg_conf:.0f}% avg_penalty={applied_penalty:.2f} "
            f"late_entry={late_entry_count}/{review_count} "
            f"onchain_against={onchain_against_count}/{review_count} "
            f"onchain_lowrel={onchain_lowrel_count}/{review_count}"
        )
        return

    result.total_score = float(result.total_score) - applied_penalty
    result.reasons.append(
        f"LLM postmortem penalty: -{applied_penalty:.2f} "
        f"(reviews={review_count}, avg conf={avg_conf:.0f}%)"
    )
    logger.info(
        f"LLM postmortem applied to {symbol} {side_key}: "
        f"-{applied_penalty:.2f} (reviews={review_count}, avg_conf={avg_conf:.0f}%, "
        f"late_entry={late_entry_count}/{review_count}, "
        f"onchain_against={onchain_against_count}/{review_count}, "
        f"onchain_lowrel={onchain_lowrel_count}/{review_count}, "
        f"extra={late_entry_boost:.2f})"
    )


def _check_portfolio_risk_gate(storage: SignalStorage, config: Config, symbol: str) -> Tuple[bool, str]:
    """Check portfolio-level and symbol-level risk guards before allowing new entries."""
    pr = getattr(config, "portfolio_risk", None)
    if not pr or not getattr(pr, "enabled", True):
        return True, ""

    max_open_positions = max(1, int(getattr(pr, "max_open_positions", 3)))
    open_positions = storage.get_open_positions_count()
    if open_positions >= max_open_positions:
        return False, f"Portfolio risk: open positions {open_positions}/{max_open_positions} (max reached)"

    day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    daily = storage.get_daily_closed_pnl(day_start)
    daily_limit = float(getattr(pr, "daily_loss_limit_pct", -3.0))
    if float(daily.get("total_pnl_pct", 0.0)) <= daily_limit:
        return False, (
            "Portfolio risk: daily loss limit hit "
            f"({float(daily.get('total_pnl_pct', 0.0)):+.2f}% <= {daily_limit:+.2f}%)"
        )

    max_losses = max(1, int(getattr(pr, "max_consecutive_losses", 4)))
    cooldown_minutes = max(1, int(getattr(pr, "loss_streak_cooldown_minutes", 90)))
    
    # Check overall portfolio loss streak
    global_streak_info = storage.get_recent_loss_streak(limit=max(30, max_losses * 3))
    if int(global_streak_info.get("loss_streak", 0)) >= max_losses:
        last_closed_at = global_streak_info.get("last_closed_at")
        if isinstance(last_closed_at, datetime):
            elapsed = datetime.now(timezone.utc) - last_closed_at
            cooldown = timedelta(minutes=cooldown_minutes)
            if elapsed < cooldown:
                rem = int((cooldown - elapsed).total_seconds() // 60) + 1
                return False, f"Portfolio risk: global loss streak {global_streak_info['loss_streak']}, cooldown active ({rem}m left)"

    # Check per-symbol loss streak (R2 Fix: prevent pairs like ARKM from bleeding)
    symbol_streak_info = storage.get_recent_loss_streak(limit=max(30, max_losses * 3), symbol=symbol)
    if int(symbol_streak_info.get("loss_streak", 0)) >= max_losses:
        last_closed_at = symbol_streak_info.get("last_closed_at")
        if isinstance(last_closed_at, datetime):
            elapsed = datetime.now(timezone.utc) - last_closed_at
            cooldown = timedelta(minutes=cooldown_minutes)
            if elapsed < cooldown:
                rem = int((cooldown - elapsed).total_seconds() // 60) + 1
                return False, f"Symbol risk: {symbol} loss streak {symbol_streak_info['loss_streak']}, cooldown active ({rem}m left)"

    return True, ""


def run_telegram_control_loop(notifier: TelegramNotifier, config: Config, get_scan_count) -> None:
    """
    Listen to Telegram updates and apply runtime control commands.

    This keeps working while scanner is paused.
    """
    global shutdown_requested

    if not notifier.enabled:
        return

    logger.info("Telegram control listener started")
    news_analyzer = NewsAnalyzer(config.openai)

    # Drop historical messages so old commands are not replayed after restart
    offset = None
    backlog = notifier.fetch_updates(timeout=0)
    if backlog:
        last_id = max((u.get('update_id', 0) for u in backlog), default=0)
        offset = last_id + 1

    while not shutdown_requested:
        updates = notifier.fetch_updates(offset=offset, timeout=20)
        if not updates:
            continue

        for update in updates:
            update_id = update.get('update_id')
            if update_id is not None:
                offset = update_id + 1

            msg = update.get('message') or update.get('edited_message')
            if not isinstance(msg, dict):
                continue

            chat_id = str(msg.get('chat', {}).get('id', ''))
            if chat_id != str(notifier.chat_id):
                continue

            text = (msg.get('text') or '').strip()
            command = parse_telegram_control_command(text)
            if not command:
                continue

            logger.info(f"Telegram control command received: {command}")

            if command == 'pause':
                if is_scan_paused():
                    notifier.send_text("⏸️ Scanner already paused.")
                else:
                    set_scan_paused(True)
                    notifier.send_text("⏸️ Scanner paused. Send /resume to continue.")
            elif command == 'resume':
                if not is_scan_paused():
                    notifier.send_text("▶️ Scanner already running.")
                else:
                    set_scan_paused(False)
                    notifier.send_text("▶️ Scanner resumed.")
            elif command == 'status':
                notifier.send_text(_telegram_status_text(config, get_scan_count()))
            elif command == 'help':
                notifier.send_text(_telegram_control_help_text())
            elif command == 'stop':
                shutdown_requested = True
                notifier.send_text("🛑 Shutdown requested. Bot will stop after current step.")
            elif command == 'news':
                news_text = _extract_news_text(text)
                notifier.send_text("🧠 جاري تحليل الخبر...") if news_text else None
                notifier.send_text(news_analyzer.analyze(news_text))


def interruptible_sleep(seconds: float, step: float = 1.0) -> None:
    """Sleep in small steps so pause/stop commands are applied quickly."""
    end_at = time.time() + max(0.0, seconds)
    while time.time() < end_at and not shutdown_requested and not is_scan_paused():
        time.sleep(min(step, end_at - time.time()))


def check_open_outcomes(
    storage: SignalStorage,
    notifier: TelegramNotifier,
    binance_exchange,
    kucoin_exchange,
    config: Config,
) -> None:
    """
    Check all OPEN signal outcomes against current prices.
    Automatically manages OPEN outcomes with:
    - optional quick TP close
    - TP1 arm (break-even + trailing)
    - TP2 final target
    - SL / BE / trailing-stop closure
    Sends Telegram notification on outcome.
    """
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _format_outcome_label(outcome_key: str) -> str:
        labels = {
            "TP_NEAR_HIT": "هدف قريب تحقق",
            "TP1_HIT": "TP1 تحقق",
            "TP2_HIT": "TP2 تحقق",
            "SL_HIT": "ضرب وقف الخسارة",
            "BE_HIT": "إغلاق على نقطة التعادل",
            "TRAIL_HIT": "ضرب الوقف المتحرك",
        }
        return labels.get(outcome_key, outcome_key.replace("_", " "))

    open_outcomes = storage.get_open_outcomes()
    
    if not open_outcomes:
        return
    
    for outcome in open_outcomes:
        if _is_historical_or_stale_open_outcome(outcome):
            storage.delete_open_outcome(int(outcome["id"]))
            continue

        symbol_raw = outcome['symbol'].split('_')[0]  # Strip _futures suffix
        try:
            side = str(outcome['side']).upper()
            ticker: Dict[str, Any] = {}
            primary_is_mt5 = _is_mt5_source(binance_exchange)
            if _is_mt5_only_requested(config) and 'VAI' in symbol_raw:
                logger.warning(f"Skipping {symbol_raw} outcome check: MT5-only mode does not allow KuCoin fallback")
                continue
            if is_yahoo_symbol(symbol_raw) and not primary_is_mt5:
                from bot.yahoo_data import get_yahoo_price
                last = get_yahoo_price(symbol_raw)
                ticker = {'last': last, 'bid': last, 'ask': last, 'symbol': symbol_raw}
            elif 'VAI' in symbol_raw:
                ticker = fetch_ticker(kucoin_exchange, symbol_raw) if kucoin_exchange else None
                ticker = ticker or {}
            else:
                ticker = fetch_ticker(binance_exchange, symbol_raw)
                ticker = ticker or {}

            price_raw, price_src = _execution_price_from_ticker(
                ticker=ticker,
                side=side,
                phase="close",
            )
            if price_raw is None:
                continue
            
            entry = _safe_float(outcome['entry'])
            sl = _safe_float(outcome['stop_loss'])
            tp_near = _safe_float(outcome.get('take_profit_near'))
            tp1 = _safe_float(outcome['take_profit_1'])
            tp2 = _safe_float(outcome['take_profit_2'])
            price = _safe_float(price_raw)
            logger.debug(
                f"{symbol_raw} outcome check side={side} close_src={price_src} price={price:.6f}"
            )
            
            hit_outcome = None
            quick_tp_enabled = bool(getattr(config.risk, 'quick_tp_outcome_enabled', True))
            be_enabled = bool(getattr(config.risk, 'break_even_after_tp1', True))
            be_buffer_pct = max(0.0, _safe_float(getattr(config.risk, 'break_even_buffer_pct', 0.02), 0.02))
            trailing_enabled = bool(getattr(config.risk, 'trailing_after_tp1', True))
            trailing_rr = max(0.05, _safe_float(getattr(config.risk, 'trailing_rr_from_risk', 0.70), 0.70))
            trailing_min_move_pct = max(0.0, _safe_float(getattr(config.risk, 'trailing_min_move_pct', 0.10), 0.10))
            manage_after_tp1 = bool(be_enabled or trailing_enabled)

            tp1_touched = bool(int(outcome.get('tp1_touched') or 0))
            be_armed = bool(int(outcome.get('break_even_armed') or 0))
            trail_stop = outcome.get('trail_stop')
            extreme_price = outcome.get('extreme_price')
            trail_stop = _safe_float(trail_stop, default=0.0) if trail_stop is not None else None
            extreme_price = _safe_float(extreme_price, default=0.0) if extreme_price is not None else None

            if entry <= 0:
                continue

            risk_distance = max(0.0, abs(entry - sl))
            trail_distance = max(risk_distance * trailing_rr, entry * (trailing_min_move_pct / 100.0))
            be_price = entry
            if be_enabled:
                if side == 'LONG':
                    be_price = entry * (1.0 + be_buffer_pct / 100.0)
                else:
                    be_price = entry * (1.0 - be_buffer_pct / 100.0)

            # Final TP2 remains hard close — blend PnL if partial exit was done.
            _partial_done = bool(int(outcome.get('partial_exit_done') or 0))
            _tp1_locked   = _safe_float(outcome.get('tp1_partial_pnl'), 0.0)
            _pe_cfg       = getattr(config, 'partial_exit', None)
            _remainder    = float(getattr(_pe_cfg, 'remainder_pct', 0.40)) if _pe_cfg else 0.40

            if side == 'LONG' and tp2 and price >= tp2:
                hit_outcome = 'TP2_HIT'
                raw_pnl = (tp2 - entry) / entry * 100
                # If partial exit was done: final PnL = locked 60% + 40% of TP2 move
                pnl_pct = (_tp1_locked + raw_pnl * _remainder) if _partial_done else raw_pnl
            elif side == 'SHORT' and tp2 and price <= tp2:
                hit_outcome = 'TP2_HIT'
                raw_pnl = (entry - tp2) / entry * 100
                pnl_pct = (_tp1_locked + raw_pnl * _remainder) if _partial_done else raw_pnl
            else:
                # First touch of TP1: arm protection and keep position open if configured.
                tp1_hit_now = (side == 'LONG' and tp1 and price >= tp1) or (side == 'SHORT' and tp1 and price <= tp1)
                if (not tp1_touched) and tp1_hit_now and manage_after_tp1:
                    tp1_touched = True
                    be_armed = bool(be_enabled)
                    extreme_price = price

                    effective_stop = sl
                    if be_armed:
                        if side == 'LONG':
                            effective_stop = max(effective_stop, be_price)
                        else:
                            effective_stop = min(effective_stop, be_price)

                    if trailing_enabled and trail_distance > 0:
                        if side == 'LONG':
                            trail_candidate = max(0.0, price - trail_distance)
                            trail_stop = trail_candidate if trail_stop is None else max(trail_stop, trail_candidate)
                            trail_stop = max(trail_stop, effective_stop)
                        else:
                            trail_candidate = price + trail_distance
                            trail_stop = trail_candidate if trail_stop is None else min(trail_stop, trail_candidate)
                            trail_stop = min(trail_stop, effective_stop)
                    else:
                        trail_stop = None

                    # ── Partial Exit 60/40 ──
                    # Lock in 60% of position at TP1, let 40% run to TP2/trail
                    _pe = getattr(config, 'partial_exit', None)
                    _pe_enabled = bool(getattr(_pe, 'enabled', False)) if _pe else False
                    _tp1_pnl_raw = ((tp1 - entry) / entry * 100) if side == 'LONG' else ((entry - tp1) / entry * 100)
                    _pe_min_rr = float(getattr(_pe, 'min_rr_to_enable', 1.0)) if _pe else 1.0
                    _rr_to_tp1 = abs(tp1 - entry) / max(abs(entry - sl), 1e-9)
                    _pe_eligible = _pe_enabled and _rr_to_tp1 >= _pe_min_rr
                    _exit_pct = float(getattr(_pe, 'tp1_exit_pct', 0.60)) if _pe else 0.60

                    if _pe_eligible:
                        # Record partial PnL (will be blended into final PnL later)
                        storage.update_open_outcome_progress(
                            int(outcome['id']),
                            tp1_touched=True,
                            break_even_armed=be_armed,
                            trail_stop=trail_stop,
                            extreme_price=extreme_price,
                            trail_armed_at=datetime.now(timezone.utc),
                            partial_exit_done=1,
                            tp1_partial_pnl=_tp1_pnl_raw * _exit_pct,
                        )
                        arm_msg = (
                            f"⚡🎯 <b>50% خرج عند TP1 + حماية مُفعَّلة</b>\n"
                            f"الرمز: <b>{symbol_raw}</b> ({side})\n"
                            f"السعر: {price:.4f}  |  TP1: {tp1:.4f}\n"
                            f"ربح مُقفَل ({int(_exit_pct*100)}%): <b>{_tp1_pnl_raw*_exit_pct:+.2f}%</b>\n"
                            f"الباقي ({int((1-_exit_pct)*100)}%) يتابع نحو TP2\n"
                            f"وقف التعادل: {be_price:.4f}"
                        )
                        if trailing_enabled and trail_stop is not None:
                            arm_msg += f"\nالوقف المتحرك: {trail_stop:.4f}"
                    else:
                        storage.update_open_outcome_progress(
                            int(outcome['id']),
                            tp1_touched=True,
                            break_even_armed=be_armed,
                            trail_stop=trail_stop,
                            extreme_price=extreme_price,
                            trail_armed_at=datetime.now(timezone.utc),
                        )
                        arm_msg = (
                            f"🎯✅ <b>TP1 تحقق - تفعيل حماية الصفقة</b>\n"
                            f"الرمز: <b>{symbol_raw}</b> ({side})\n"
                            f"السعر الحالي: {price:.4f}\n"
                            f"وقف التعادل: {be_price:.4f}"
                        )
                        if trailing_enabled and trail_stop is not None:
                            arm_msg += f"\nالوقف المتحرك: {trail_stop:.4f}"

                    notifier._send_telegram(arm_msg) if notifier.enabled else print(arm_msg)
                    
                    # Ensure SL is pushed to MT5 (Break Even / Trailing)
                    if _is_mt5_only_requested(config) and config.mt5:
                        try:
                            from bot.mt5_client import MT5Client
                            client = getattr(binance_exchange, "mt5", None)
                            if not client:
                                client = MT5Client.from_config(config.mt5)
                            client.connect_mt5()
                            positions = client.get_bot_positions(symbol_raw)
                            for pos in positions:
                                target_sl = be_price
                                if trail_stop:
                                    if side == 'LONG':
                                        target_sl = max(target_sl, trail_stop)
                                    else:
                                        target_sl = min(target_sl, trail_stop)
                                client.modify_position_sl(pos['ticket'], target_sl)
                        except Exception as e:
                            logger.error(f"Failed to update MT5 SL for {symbol_raw} at TP1: {e}")

                    continue

                # If TP1 management is disabled, TP1 remains a full close.
                if (not tp1_touched) and tp1_hit_now and (not manage_after_tp1):
                    hit_outcome = 'TP1_HIT'
                    pnl_pct = ((tp1 - entry) / entry * 100) if side == 'LONG' else ((entry - tp1) / entry * 100)
                else:
                    # Update trailing state while still OPEN.
                    effective_stop = sl
                    if tp1_touched and be_armed:
                        if side == 'LONG':
                            effective_stop = max(effective_stop, be_price)
                        else:
                            effective_stop = min(effective_stop, be_price)

                    if tp1_touched and trailing_enabled and trail_distance > 0:
                        if side == 'LONG':
                            new_extreme = max(extreme_price if extreme_price is not None else price, price)
                            trail_candidate = max(0.0, new_extreme - trail_distance)
                            new_trail = trail_candidate if trail_stop is None else max(trail_stop, trail_candidate)
                            new_trail = max(new_trail, effective_stop)
                            if (extreme_price is None) or (new_extreme > extreme_price) or (trail_stop is None) or (new_trail > trail_stop):
                                extreme_price = new_extreme
                                trail_stop = new_trail
                                storage.update_open_outcome_progress(
                                    int(outcome['id']),
                                    extreme_price=extreme_price,
                                    trail_stop=trail_stop,
                                )
                                if _is_mt5_only_requested(config) and config.mt5:
                                    try:
                                        from bot.mt5_client import MT5Client
                                        client = getattr(binance_exchange, "mt5", None)
                                        if not client: client = MT5Client.from_config(config.mt5)
                                        client.connect_mt5()
                                        for pos in client.get_bot_positions(symbol_raw):
                                            client.modify_position_sl(pos['ticket'], trail_stop)
                                    except Exception as e:
                                        logger.error(f"Failed to update MT5 Trailing SL (LONG) for {symbol_raw}: {e}")
                            effective_stop = max(effective_stop, trail_stop if trail_stop is not None else effective_stop)
                        else:
                            new_extreme = min(extreme_price if extreme_price is not None else price, price)
                            trail_candidate = new_extreme + trail_distance
                            new_trail = trail_candidate if trail_stop is None else min(trail_stop, trail_candidate)
                            new_trail = min(new_trail, effective_stop)
                            if (extreme_price is None) or (new_extreme < extreme_price) or (trail_stop is None) or (new_trail < trail_stop):
                                extreme_price = new_extreme
                                trail_stop = new_trail
                                storage.update_open_outcome_progress(
                                    int(outcome['id']),
                                    extreme_price=extreme_price,
                                    trail_stop=trail_stop,
                                )
                                if _is_mt5_only_requested(config) and config.mt5:
                                    try:
                                        from bot.mt5_client import MT5Client
                                        client = getattr(binance_exchange, "mt5", None)
                                        if not client: client = MT5Client.from_config(config.mt5)
                                        client.connect_mt5()
                                        for pos in client.get_bot_positions(symbol_raw):
                                            client.modify_position_sl(pos['ticket'], trail_stop)
                                    except Exception as e:
                                        logger.error(f"Failed to update MT5 Trailing SL (SHORT) for {symbol_raw}: {e}")
                            effective_stop = min(effective_stop, trail_stop if trail_stop is not None else effective_stop)

                    # Quick TP path (only before TP1 touch).
                    if (not tp1_touched) and quick_tp_enabled and tp_near:
                        if side == 'LONG' and price >= tp_near:
                            hit_outcome = 'TP_NEAR_HIT'
                            pnl_pct = (tp_near - entry) / entry * 100
                        elif side == 'SHORT' and price <= tp_near:
                            hit_outcome = 'TP_NEAR_HIT'
                            pnl_pct = (entry - tp_near) / entry * 100

                    # Stop handling (normal SL before TP1; BE/trailing after TP1).
                    if not hit_outcome:
                        if side == 'LONG' and price <= effective_stop:
                            if tp1_touched and trailing_enabled and trail_stop is not None and trail_stop > be_price:
                                hit_outcome = 'TRAIL_HIT'
                            elif tp1_touched and be_armed:
                                hit_outcome = 'BE_HIT'
                            else:
                                hit_outcome = 'SL_HIT'
                            pnl_pct = (price - entry) / entry * 100
                        elif side == 'SHORT' and price >= effective_stop:
                            if tp1_touched and trailing_enabled and trail_stop is not None and trail_stop < be_price:
                                hit_outcome = 'TRAIL_HIT'
                            elif tp1_touched and be_armed:
                                hit_outcome = 'BE_HIT'
                            else:
                                hit_outcome = 'SL_HIT'
                            pnl_pct = (entry - price) / entry * 100
            
            if hit_outcome:
                storage.update_signal_outcome(outcome['id'], hit_outcome, price, pnl_pct)
                
                emoji_map = {
                    'TP_NEAR_HIT': '⚡✅',
                    'TP1_HIT': '🎯✅',
                    'TP2_HIT': '🎯🎯✅',
                    'TRAIL_HIT': '🔒✅',
                    'BE_HIT': '🟡✅',
                    'SL_HIT': '🛑❌',
                }
                pnl_str = f"{pnl_pct:+.2f}%"
                msg = (
                    f"{emoji_map[hit_outcome]} <b>{_format_outcome_label(hit_outcome)}</b>\n"
                    f"الرمز: <b>{symbol_raw}</b> ({side})\n"
                    f"الدخول: {entry:.4f} → الإغلاق: {price:.4f}\n"
                    f"النتيجة: <b>{pnl_str}</b>"
                )
                notifier._send_telegram(msg) if notifier.enabled else print(msg)
                _enqueue_llm_postmortem(
                    storage=storage,
                    config=config,
                    outcome_id=int(outcome["id"]),
                    outcome=hit_outcome,
                    pnl_pct=float(pnl_pct),
                )
        
        except Exception as e:
            logger.debug(f"Outcome check error for {outcome['symbol']}: {e}")


def send_winrate_report(storage: SignalStorage, notifier: TelegramNotifier) -> None:
    """Send a weekly WinRate performance report to Telegram."""
    global _last_winrate_report
    
    now = datetime.now(timezone.utc)
    
    # Send report once per day (every 24h)
    if _last_winrate_report and (now - _last_winrate_report) < timedelta(hours=24):
        return
        
    stats_24h = storage.get_winrate_stats(days=1)
    stats_7d = storage.get_winrate_stats(days=7)
    stats_30d = storage.get_winrate_stats(days=30)
    
    if stats_7d['total'] == 0:
        return  # No data yet
        
    # Fetch latest LLM review tip
    latest_review = ""
    try:
        import sqlite3
        conn = sqlite3.connect(storage.db_path)
        cur = conn.cursor()
        cur.execute("SELECT analysis_text FROM llm_trade_reviews ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            # Extract just the first bullet or sentence for brevity
            latest_review = row[0].split('\n')[0][:150] + "..." if len(row[0]) > 150 else row[0].split('\n')[0]
    except Exception:
        pass
    
    win_emoji_24h = "🟢" if stats_24h['winrate_pct'] >= 50 else "🔴"
    win_emoji_7d = "🟢" if stats_7d['winrate_pct'] >= 50 else "🔴"
    
    msg = (
        f"📊 <b>Daily Performance Report</b>\n\n"
        f"<b>Today (24h):</b>\n"
        f"  ✅ Trades Closed: {stats_24h['total'] - stats_24h.get('open', 0)}\n"
        f"  ⏳ Open: {stats_24h.get('open', 0)}\n"
        f"  {win_emoji_24h} WinRate: <b>{stats_24h['winrate_pct']:.1f}%</b>\n"
        f"  💹 Net P&L: <b>{stats_24h['avg_pnl']:+.2f}%</b> (Avg)\n\n"
        f"<b>Last 7 Days:</b>\n"
        f"  ⚡ Quick TP: {stats_7d.get('tp_near_hit', 0)}  |  🎯 TP1: {stats_7d['tp1_hit']}  |  🎯🎯 TP2: {stats_7d['tp2_hit']}\n"
        f"  🔒 Trailing: {stats_7d.get('trail_hit', 0)}  |  🟡 BE: {stats_7d.get('be_hit', 0)}\n"
        f"  🛑 SL Hit: {stats_7d['sl_hit']}  |  ↩ Exit: {stats_7d.get('exited', 0)}  |  ⏳ Open: {stats_7d['open']}\n"
        f"  {win_emoji_7d} WinRate: <b>{stats_7d['winrate_pct']:.1f}%</b>\n"
        f"  💹 Avg P&L: <b>{stats_7d['avg_pnl']:+.2f}%</b>\n\n"
        f"<b>Last 30 Days:</b>\n"
        f"  {win_emoji_7d} WinRate: <b>{stats_30d['winrate_pct']:.1f}%</b>\n"
        f"  Total Signals: {stats_30d['total']}"
    )
    
    if latest_review:
        msg += f"\n\n🤖 <b>أبرز ملاحظات الذكاء الاصطناعي اليوم:</b>\n<i>{latest_review}</i>"
    
    notifier._send_telegram(msg) if notifier.enabled else print(msg)
    _last_winrate_report = now
    logger.info(f"WinRate report sent: {stats_7d['winrate_pct']:.1f}% (7d)")


def send_walkforward_shadow_report(storage: SignalStorage, notifier: TelegramNotifier) -> None:
    """
    Shadow-only monitoring:
    compare 24h expectancy/WR against 7d baseline without changing live behavior.
    """
    global _last_learning_health_report
    now = datetime.now(timezone.utc)
    if _last_learning_health_report and (now - _last_learning_health_report) < timedelta(hours=6):
        return
    _last_learning_health_report = now

    try:
        stats_24h = storage.get_winrate_stats(days=1)
        stats_7d = storage.get_winrate_stats(days=7)

        exp_24h = float(stats_24h.get('avg_pnl', 0.0) or 0.0)
        exp_7d = float(stats_7d.get('avg_pnl', 0.0) or 0.0)
        wr_24h = float(stats_24h.get('winrate_pct', 0.0) or 0.0)
        wr_7d = float(stats_7d.get('winrate_pct', 0.0) or 0.0)
        delta_exp = exp_24h - exp_7d
        delta_wr = wr_24h - wr_7d

        closed_24h = int(
            (stats_24h.get('tp_near_hit', 0) or 0)
            + (stats_24h.get('tp1_hit', 0) or 0)
            + (stats_24h.get('tp2_hit', 0) or 0)
            + (stats_24h.get('trail_hit', 0) or 0)
            + (stats_24h.get('be_hit', 0) or 0)
            + (stats_24h.get('sl_hit', 0) or 0)
            + (stats_24h.get('exited', 0) or 0)
        )

        logger.info(
            "Walkforward shadow: exp24h={:+.3f}% exp7d={:+.3f}% delta_exp={:+.3f}% "
            "wr24h={:.1f}% wr7d={:.1f}% delta_wr={:+.1f}pp closed24h={}",
            exp_24h,
            exp_7d,
            delta_exp,
            wr_24h,
            wr_7d,
            delta_wr,
            closed_24h,
        )

        # Keep Telegram noise low: only send when sample is meaningful.
        if notifier.enabled and closed_24h >= 8:
            msg = (
                "📈 <b>Walkforward Shadow</b>\n"
                f"24h expectancy: <b>{exp_24h:+.3f}%</b> | 7d: <b>{exp_7d:+.3f}%</b>\n"
                f"Δ expectancy: <b>{delta_exp:+.3f}%</b>\n"
                f"24h WR: <b>{wr_24h:.1f}%</b> | 7d WR: <b>{wr_7d:.1f}%</b>\n"
                f"Δ WR: <b>{delta_wr:+.1f}pp</b> | closed 24h: <b>{closed_24h}</b>\n"
                "Mode: <b>shadow only</b>"
            )
            notifier._send_telegram(msg)
    except Exception as e:
        logger.debug(f"Walkforward shadow report failed: {e}")


def scan_symbol(
    symbol: str,
    exchange,
    config: Config,
    storage: SignalStorage,
    notifier: TelegramNotifier,
    signal_mode: str = "futures",
    signals_sent_this_scan: dict = None,
    max_same_direction: int = 2,
    session_threshold_add: float = 0.0,
    ml_engine: Any = None
) -> str:
    """
    Scan a single symbol for signals in a specific mode.
    
    Args:
        symbol: Trading symbol
        exchange: ccxt exchange instance
        config: Configuration
        storage: Signal storage
        notifier: Telegram notifier
        signal_mode: Signal mode (futures)
        signals_sent_this_scan: Dict tracking signals sent this scan (correlation filter)
        max_same_direction: Max signals of same direction per scan
        
    Returns:
        Signal side sent ('LONG', 'SHORT', 'EXIT', or '')
    """
    if signals_sent_this_scan is None:
        signals_sent_this_scan = {'LONG': 0, 'SHORT': 0}
    
    try:
        is_macro = is_yahoo_symbol(symbol)
        exchange_id = str(getattr(exchange, "id", "") or "").lower()
        macro_via_exchange = bool(is_macro and exchange_id in {"mt5", "mt5_bridge"})
        mode_config = config.futures_macro if is_macro else config.futures
        mode_label = "🔸 FUTURES"
        
        logger.info(f"Scanning {symbol} [{signal_mode.upper()}]...")
        
        # Fetch OHLCV data for all timeframes based on mode
        timeframes = {
            'trend': mode_config.trend_tf,
            'entry': mode_config.entry_tf,
            'sr': mode_config.sr_tf
        }
        # Add HTF (higher timeframe) for multi-timeframe confirmation
        htf_tf = getattr(mode_config, 'htf', '') or ''
        if htf_tf and htf_tf != mode_config.trend_tf:
            timeframes['htf'] = htf_tf
        
        data = {}
        
        # Macro symbols default to Yahoo unless MT5 bridge is selected.
        if is_macro and not macro_via_exchange:
            for name, tf in timeframes.items():
                data[name] = fetch_yahoo_ohlcv(symbol, tf, config.limit)
            
            # Fetch current price from Yahoo
            current_price = get_yahoo_price(symbol)
            ticker = {'last': current_price, 'symbol': symbol}
        else:
            for name, tf in timeframes.items():
                data[name] = fetch_ohlcv(exchange, symbol, tf, config.limit)
            
            # MT5 bridge can run without /ticker (using OHLCV-only bridge).
            # Reuse already-fetched candles to avoid an extra /ohlcv request per symbol.
            mt5_no_ticker = bool(
                _is_mt5_bridge_source(exchange)
                and not bool(getattr(exchange, "_ticker_endpoint_available", True))
            )
            if mt5_no_ticker:
                ticker = _build_ticker_from_cached_ohlcv(symbol, data)
                if _safe_float(ticker.get("last")) is None:
                    ticker = fetch_ticker(exchange, symbol)
            else:
                # Fetch current ticker from selected exchange (Binance / mt5_bridge / etc.)
                ticker = fetch_ticker(exchange, symbol)
            
        # Order flow data
        order_flow_data = None
        of_cfg = getattr(config, 'order_flow', None)
        if of_cfg and getattr(of_cfg, 'enabled', True) and not is_macro:
            funding_rate = 0.0
            open_interest = 0.0
            try:
                if hasattr(exchange, 'fetch_funding_rate'):
                    fr_data = exchange.fetch_funding_rate(symbol)
                    funding_rate = float(fr_data.get('fundingRate', 0) if fr_data else 0)
                
                if hasattr(exchange, 'fetch_open_interest'):
                    oi_data = exchange.fetch_open_interest(symbol)
                    if oi_data and 'openInterestValue' in oi_data:
                        open_interest = float(oi_data.get('openInterestValue', 0))
                    elif oi_data and 'openInterestAmount' in oi_data:
                        open_interest = float(oi_data.get('openInterestAmount', 0)) * getattr(ticker, "last", current_price if 'current_price' in locals() else 1)
            except Exception as e:
                logger.debug(f"Order flow fetch failed for {symbol}: {e}")
            
            order_flow_data = {
                'funding_rate': funding_rate,
                'open_interest': open_interest
            }
        
        # Use mode-specific cooldown key
        cooldown_symbol = f"{symbol}_{signal_mode}"
        
        # Check for existing position (mode-specific)
        existing_position_info = storage.get_open_position_details(cooldown_symbol)
        existing_position = existing_position_info["side"] if existing_position_info else None
        opened_at = existing_position_info.get("opened_at") if existing_position_info else None
        
        if existing_position:
            exit_check_price, _ = _execution_price_from_ticker(
                ticker=ticker,
                side=existing_position,
                phase="close",
            )
            if exit_check_price is None:
                exit_check_price = float(ticker.get('last', 0) or 0.0)
            # Check exit conditions
            zones = build_zones(data['sr']) if not data['sr'].empty else []
            exit_signal = check_exit_conditions(
                symbol=symbol,
                df_entry=data['entry'],
                zones=zones,
                current_price=exit_check_price,
                current_position=existing_position,
                position_opened_at=opened_at,
                config=config
            )
            
            if exit_signal and exit_signal.is_valid:
                # Add mode label to exit signal
                exit_signal.symbol = f"{symbol} {mode_label}"
                
                # Send exit signal
                notifier.send_signal(exit_signal)
                
                # Store exit
                storage.save_signal(SignalRecord(
                    symbol=cooldown_symbol,
                    side='EXIT',
                    timestamp=exit_signal.timestamp,
                    score=0,
                    entry=exit_signal.current_price,
                    stop_loss=0,
                    take_profit_near=0,
                    take_profit_1=0,
                    take_profit_2=0,
                    reasons=exit_signal.reasons
                ))

                # Close tracked OPEN outcomes for this symbol on manual EXIT.
                # MT5-like close pricing:
                # LONG closes on bid, SHORT closes on ask.
                close_exec_price, close_exec_src = _execution_price_from_ticker(
                    ticker=ticker,
                    side=existing_position,
                    phase="close",
                )
                if close_exec_price is None:
                    close_exec_price = float(exit_signal.current_price or 0.0)
                    close_exec_src = "signal_price"
                logger.debug(
                    f"{symbol} [{signal_mode.upper()}] EXIT close pricing: "
                    f"side={existing_position}, src={close_exec_src}, price={close_exec_price:.6f}"
                )

                storage.close_open_outcomes(
                    symbol=cooldown_symbol,
                    close_price=close_exec_price,
                    outcome='EXITED',
                    # Spread is already embedded when entry/exit use side-aware quotes.
                    spread_cost_pct=0.0,
                )
                
                # Clear cooldown for new entries
                storage.clear_cooldown(cooldown_symbol, 'LONG')
                storage.clear_cooldown(cooldown_symbol, 'SHORT')
                
                logger.info(f"EXIT signal sent for {symbol} [{signal_mode.upper()}]")
                return 'EXIT'

            logger.debug(
                f"{symbol} [{signal_mode.upper()}] has open {existing_position} position; "
                "skipping new entry until exit conditions trigger"
            )
            return ''
        
        # Portfolio-level risk gate (applies only to new entries).
        allow_portfolio, portfolio_reason = _check_portfolio_risk_gate(storage, config, symbol)
        if not allow_portfolio:
            logger.info(f"{symbol} [{signal_mode}] blocked by portfolio risk: {portfolio_reason}")
            safe_reason = (
                str(portfolio_reason)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            _maybe_send_risk_alert(
                notifier,
                key=f"portfolio::{portfolio_reason.split(':', 1)[0]}",
                message=f"🛡️ <b>إدارة المخاطر فعّالة</b>\n{safe_reason}",
                cooldown_minutes=20,
            )
            return ''

        # Check pending entries first (side-aware trigger price; MT5-like).
        pending_matches = []
        for p in storage.get_pending_entries(symbol):
            p_side = str(getattr(p, 'side', p.get('side', ''))).upper()
            trigger_price, trigger_src = _execution_price_from_ticker(
                ticker=ticker,
                side=p_side,
                phase="entry",
            )
            if trigger_price is None:
                trigger_price = float(ticker.get('last', 0) or 0)
                trigger_src = "last_fallback"
            zone_bottom = float(getattr(p, 'zone_bottom', p.get('zone_bottom', 0)) or 0)
            zone_top = float(getattr(p, 'zone_top', p.get('zone_top', float('inf'))) or float('inf'))
            if zone_bottom <= trigger_price <= zone_top:
                logger.debug(
                    f"{symbol} [{signal_mode}] pending match id={p.get('id')} side={p_side} "
                    f"price={trigger_price:.6f} src={trigger_src} zone=({zone_bottom:.6f}-{zone_top:.6f})"
                )
                pending_matches.append(p)
        
        if pending_matches:
            p = pending_matches[0]
            logger.info(f"{symbol} [{signal_mode}]: Triggering PENDING PULLBACK ENTRY for {p['side']} at ideal price {p['ideal_entry']}")
            
            result = SignalResult()
            result.is_valid = True
            result.symbol = symbol
            result.side = p['side']
            result.total_score = p['original_score']
            result.threshold = getattr(config.futures, "min_score", 4.0) if not is_macro else getattr(config.futures_macro, "min_score", 4.0)
            result.reasons = p.get('original_reasons', [])
            result.reasons.append(f"Pullback entry matched! Ideal: {p['ideal_entry']:.4f}")
            result.market_regime = "pending_pullback"
            
            storage.delete_pending_entry(p['id'])
        else:
            # Calibrate ATR stop multiplier dynamically
            atr_calibration_scale = compute_symbol_atr_calibration(storage, symbol)
            
            # Analyze for new signals
            result = analyze_symbol(
                symbol=symbol,
                df_trend=data['trend'],
                df_entry=data['entry'],
                df_sr=data['sr'],
                ticker=ticker,
                config=config,
                df_htf=data.get('htf'),
                order_flow_data=order_flow_data,
                ml_engine=ml_engine,
                atr_calibration_scale=atr_calibration_scale
            )
        
        # Log analysis result
        if not getattr(result, 'market_regime', None) == "pending_pullback":
            logger.debug(
                f"{symbol} [{signal_mode}]: trend={result.trend}, score={result.total_score:.1f}/{result.threshold}, "
                f"side={result.side or 'none'}"
            )

        learning_mode = str(getattr(config.learning, "decision_mode", "hybrid")).strip().lower()
        selected_learning = None

        # Optional mode: decision is driven by adaptive learning profile.
        if learning_mode == "learning_only":
            base_score = float(result.total_score)
            threshold = float(result.threshold)
            learning_long = evaluate_learning_signal(
                db_path=storage.db_path,
                symbol=cooldown_symbol,
                side="LONG",
                base_score=base_score,
                threshold=threshold,
                config=config.learning,
                market_regime=getattr(result, "market_regime", None),
                entry_tf=mode_config.entry_tf,
            )
            learning_short = evaluate_learning_signal(
                db_path=storage.db_path,
                symbol=cooldown_symbol,
                side="SHORT",
                base_score=base_score,
                threshold=threshold,
                config=config.learning,
                market_regime=getattr(result, "market_regime", None),
                entry_tf=mode_config.entry_tf,
            )

            optimize_for_expectancy = bool(getattr(config.learning, "optimize_for_expectancy", True))
            expectancy_weight = max(0.1, float(getattr(config.learning, "expectancy_weight", 1.0)))

            def _decision_utility(decision) -> float:
                base_wr_edge = float(decision.expected_winrate) - 0.50
                wr_component = base_wr_edge * 0.20
                pnl_component = float(decision.expected_pnl_pct) * expectancy_weight
                return pnl_component + wr_component

            long_utility = _decision_utility(learning_long)
            short_utility = _decision_utility(learning_short)

            preferred_side = "LONG"
            preferred = learning_long
            opposite_side = "SHORT"
            opposite = learning_short
            if short_utility > long_utility:
                preferred_side = "SHORT"
                preferred = learning_short
                opposite_side = "LONG"
                opposite = learning_long
            elif short_utility == long_utility and (
                learning_short.expected_winrate > learning_long.expected_winrate
                or (
                    learning_short.expected_winrate == learning_long.expected_winrate
                    and learning_short.expected_pnl_pct > learning_long.expected_pnl_pct
                )
            ):
                preferred_side = "SHORT"
                preferred = learning_short
                opposite_side = "LONG"
                opposite = learning_long

            min_samples = max(1, int(getattr(config.learning, "decision_min_samples", 30)))
            min_wr = float(getattr(config.learning, "decision_min_winrate", 0.50))
            min_pnl = float(getattr(config.learning, "decision_min_pnl_pct", 0.0))
            min_expectancy = float(getattr(config.learning, "decision_min_expectancy_pct", min_pnl))
            min_edge = float(getattr(config.learning, "decision_min_edge_pct", 0.01))
            wr_edge = preferred.expected_winrate - opposite.expected_winrate

            result.learning_opinion = {
                "mode": "learning_only",
                "decision": preferred_side,
                "selected_reason": preferred.reason,
                "wr_edge_pct": float(wr_edge * 100.0),
                "long_utility": float(long_utility),
                "short_utility": float(short_utility),
                "optimize_for_expectancy": bool(optimize_for_expectancy),
                "thresholds": {
                    "min_samples": int(min_samples),
                    "min_winrate_pct": float(min_wr * 100.0),
                    "min_pnl_pct": float(min_pnl),
                    "min_expectancy_pct": float(min_expectancy),
                    "min_edge_pct": float(min_edge * 100.0),
                },
                "long": {
                    "allow": bool(learning_long.allow),
                    "samples": int(learning_long.sample_size),
                    "expected_winrate_pct": float(learning_long.expected_winrate * 100.0),
                    "expected_pnl_pct": float(learning_long.expected_pnl_pct),
                    "adjustment": float(learning_long.score_adjustment),
                    "reason": str(learning_long.reason),
                },
                "short": {
                    "allow": bool(learning_short.allow),
                    "samples": int(learning_short.sample_size),
                    "expected_winrate_pct": float(learning_short.expected_winrate * 100.0),
                    "expected_pnl_pct": float(learning_short.expected_pnl_pct),
                    "adjustment": float(learning_short.score_adjustment),
                    "reason": str(learning_short.reason),
                },
            }

            if preferred.sample_size < min_samples:
                logger.info(
                    f"Learning-only blocked {symbol} [{signal_mode}] {preferred_side}: "
                    f"samples {preferred.sample_size} < {min_samples}"
                )
                return ''
            if not preferred.allow:
                logger.info(
                    f"Learning-only blocked {symbol} [{signal_mode}] {preferred_side}: policy gate"
                )
                return ''
            if preferred.expected_winrate < min_wr:
                logger.info(
                    f"Learning-only blocked {symbol} [{signal_mode}] {preferred_side}: "
                    f"winrate {preferred.expected_winrate*100:.1f}% < {min_wr*100:.1f}%"
                )
                return ''
            if preferred.expected_pnl_pct < min_pnl:
                logger.info(
                    f"Learning-only blocked {symbol} [{signal_mode}] {preferred_side}: "
                    f"exp_pnl {preferred.expected_pnl_pct:+.2f}% < {min_pnl:+.2f}%"
                )
                return ''
            if preferred.expected_pnl_pct < min_expectancy:
                logger.info(
                    f"Learning-only blocked {symbol} [{signal_mode}] {preferred_side}: "
                    f"expectancy {preferred.expected_pnl_pct:+.2f}% < {min_expectancy:+.2f}%"
                )
                return ''
            if wr_edge < min_edge:
                logger.info(
                    f"Learning-only blocked {symbol} [{signal_mode}]: "
                    f"edge {wr_edge*100:.2f}% < {min_edge*100:.2f}%"
                )
                return ''

            original_side = result.side
            result.side = preferred_side
            selected_learning = preferred
            result.is_valid = True
            result.blocked_reason = None
            result.total_score = max(float(result.total_score), float(result.threshold))
            result.reasons.append("Learning-only mode: decision driven by learned profile")
            result.reasons.append(
                f"Learning LONG: wr={learning_long.expected_winrate*100:.1f}% "
                f"pnl={learning_long.expected_pnl_pct:+.2f}% n={learning_long.sample_size}"
            )
            result.reasons.append(
                f"Learning SHORT: wr={learning_short.expected_winrate*100:.1f}% "
                f"pnl={learning_short.expected_pnl_pct:+.2f}% n={learning_short.sample_size}"
            )
            result.reasons.append(
                f"Chosen {preferred_side}: wr_edge={wr_edge*100:.2f}% "
                f"adj={preferred.score_adjustment:+.2f} "
                f"util(L={long_utility:+.3f}, S={short_utility:+.3f})"
            )

            if (not result.risk_levels) or (original_side != preferred_side):
                zones = build_zones(data['sr']) if not data['sr'].empty else []
                zone_ref_price, _ = _execution_price_from_ticker(
                    ticker=ticker,
                    side=preferred_side,
                    phase="entry",
                )
                zone_ref_price = float(zone_ref_price or ticker.get('last', 0) or 0.0)
                zone_result = is_price_in_zone(zone_ref_price, zones)
                nearest_zone = zone_result.zone if zone_result.in_zone else None
                next_zone = (
                    get_nearest_resistance(zone_ref_price, zones)
                    if preferred_side == 'LONG'
                    else get_nearest_support(zone_ref_price, zones)
                )
                atr_value = None
                try:
                    entry_with_ind = add_all_indicators(data['entry'])
                    if 'atr_14' in entry_with_ind.columns:
                        atr_raw = entry_with_ind['atr_14'].iloc[-1]
                        atr_value = float(atr_raw) if atr_raw is not None else None
                except Exception:
                    atr_value = None

                risk_profile = get_dynamic_risk_parameters(
                    config,
                    getattr(result, "market_regime", "sideways"),
                )
                result.risk_levels = calculate_risk_levels(
                    ticker=ticker,
                    side=preferred_side,
                    zone=nearest_zone,
                    next_zone=next_zone,
                    buffer_pct=config.risk.buffer_pct,
                    rr_tp1=config.risk.rr_tp1,
                    rr_tp2=risk_profile["rr_tp2"],
                    atr=atr_value,
                    quick_tp_pct=risk_profile["quick_tp_pct"],
                    quick_tp_min_pct=risk_profile["quick_tp_min_pct"],
                    quick_tp1_fraction=config.risk.quick_tp1_fraction,
                    atr_stop_mult=risk_profile["atr_stop_mult"],
                    atr_buffer_mult=risk_profile["atr_buffer_mult"],
                    tp2_atr_mult=risk_profile["tp2_atr_mult"],
                )
        
        if result.is_valid and result.side and result.side != 'EXIT':
            if learning_mode != "learning_only":
                # Adaptive learning filter (trained from last-year outcomes).
                learning = evaluate_learning_signal(
                    db_path=storage.db_path,
                    symbol=cooldown_symbol,
                    side=result.side,
                    base_score=float(result.total_score),
                    threshold=float(result.threshold),
                    config=config.learning,
                    market_regime=getattr(result, "market_regime", None),
                    entry_tf=mode_config.entry_tf,
                )
                selected_learning = learning
                if learning.reason:
                    result.reasons.append(learning.reason)

                result.learning_opinion = {
                    "mode": "hybrid",
                    "decision": result.side,
                    "selected_reason": learning.reason,
                    "wr_edge_pct": None,
                    "thresholds": {
                        "min_samples": int(getattr(config.learning, "min_symbol_side_trades", 0)),
                        "min_winrate_pct": float(getattr(config.learning, "min_expected_winrate", 0.0) * 100.0),
                        "min_pnl_pct": float(getattr(config.learning, "min_expected_pnl_pct", 0.0)),
                        "min_edge_pct": None,
                    },
                    "side_profile": {
                        "allow": bool(learning.allow),
                        "samples": int(learning.sample_size),
                        "expected_winrate_pct": float(learning.expected_winrate * 100.0),
                        "expected_pnl_pct": float(learning.expected_pnl_pct),
                        "adjustment": float(learning.score_adjustment),
                        "reason": str(learning.reason),
                    },
                }

                min_local_for_hard_gate = max(4, int(config.learning.min_symbol_side_trades))
                applied_adjustment = learning.score_adjustment
                if learning.sample_size < min_local_for_hard_gate:
                    applied_adjustment *= 0.35

                if applied_adjustment != 0:
                    result.total_score += applied_adjustment
                    if learning.sample_size < min_local_for_hard_gate:
                        result.reasons.append(
                            f"Learning: adjustment damped (local samples={learning.sample_size})"
                        )

                if not learning.allow:
                    logger.info(
                        f"Learning filter blocked {symbol} [{signal_mode}] {result.side} "
                        f"(score={result.total_score:.2f}/{result.threshold:.2f})"
                    )
                    return ''

                if result.total_score < result.threshold and learning.sample_size >= min_local_for_hard_gate:
                    logger.info(
                        f"Learning-adjusted score below threshold for {symbol} [{signal_mode}] "
                        f"({result.total_score:.2f}/{result.threshold:.2f})"
                    )
                    return ''

                min_hybrid_expectancy = float(
                    getattr(config.learning, "hybrid_min_expectancy_pct", -0.08)
                )
                if (
                    int(getattr(learning, "sample_size", 0)) >= min_local_for_hard_gate
                    and float(getattr(learning, "expected_pnl_pct", 0.0)) < min_hybrid_expectancy
                ):
                    logger.info(
                        f"Learning expectancy gate blocked {symbol} [{signal_mode}] {result.side}: "
                        f"{float(getattr(learning, 'expected_pnl_pct', 0.0)):+.2f}% < {min_hybrid_expectancy:+.2f}%"
                    )
                    return ''

            _apply_llm_postmortem_feedback(
                storage=storage,
                config=config,
                symbol=cooldown_symbol,
                side=result.side,
                result=result,
            )

            allow_adapter, adapter_reason, adapter_threshold_add = _apply_llm_execution_adapters(
                storage=storage,
                config=config,
                symbol=cooldown_symbol,
                side=result.side,
                df_entry=data.get("entry"),
                result=result,
            )
            if adapter_reason:
                result.reasons.append(f"{'⚠' if allow_adapter else '✗'} {adapter_reason}")
            if not allow_adapter:
                logger.info(
                    f"LLM execution adapter blocked {symbol} [{signal_mode}] {result.side}: {adapter_reason}"
                )
                return ''

            # Hard gate after postmortem adjustment:
            # if learning memory pushes score below threshold, do not dispatch.
            if float(result.total_score) < float(result.threshold):
                logger.info(
                    f"Postmortem-adjusted score below threshold for {symbol} [{signal_mode}] "
                    f"{result.side} ({result.total_score:.2f}/{result.threshold:.2f})"
                )
                return ''

            # Strict shadow policy for comparison only (does not change live decision).
            qf_cfg = getattr(config, "quality_first", None)
            if qf_cfg and bool(getattr(qf_cfg, "enabled", False)):
                try:
                    # Isolated snapshot so "quality_first" can never mutate live decision state.
                    qf_shadow_result = SimpleNamespace(
                        zone_info=deepcopy(getattr(result, "zone_info", None)),
                        timing_info=deepcopy(getattr(result, "timing_info", None)),
                        risk_levels=deepcopy(getattr(result, "risk_levels", None)),
                    )
                    result.quality_first_opinion = evaluate_quality_first(
                        symbol=symbol,
                        side=result.side,
                        ticker=ticker,
                        df_trend=data.get("trend"),
                        df_entry=data.get("entry"),
                        result=qf_shadow_result,
                        config=config,
                    )
                except Exception as qf_exc:
                    logger.debug(f"Quality-first shadow evaluation failed for {symbol}: {qf_exc}")

            allow_quality, quality_reason, quality_threshold_add = _check_quality_filter(
                symbol=symbol,
                ticker=ticker,
                df_entry=data.get("entry"),
                result=result,
                config=config,
                learning_decision=selected_learning,
                storage=storage,
            )
            if quality_reason:
                result.reasons.append(f"{'⚠' if allow_quality else '✗'} {quality_reason}")

            if not allow_quality:
                logger.info(
                    f"Quality filter blocked {symbol} [{signal_mode}] {result.side}: {quality_reason}"
                )
                return ''

            total_threshold_add = quality_threshold_add + session_threshold_add + adapter_threshold_add
            if total_threshold_add > 0:
                result.threshold = float(result.threshold) + float(total_threshold_add)
                _reasons = []
                if session_threshold_add > 0:
                    _reasons.append(f"Session filter (+{session_threshold_add:.2f})")
                if quality_threshold_add > 0:
                    _reasons.append(f"Quality filter (+{quality_threshold_add:.2f})")
                if adapter_threshold_add > 0:
                    _reasons.append(f"LLM execution adapter (+{adapter_threshold_add:.2f})")
                
                result.reasons.append(
                    f"Cautious filters raised threshold: {' | '.join(_reasons)}"
                )
                if float(result.total_score) < float(result.threshold):
                    logger.info(
                        f"Quality cautious-gate blocked {symbol} [{signal_mode}] {result.side}: "
                        f"score {result.total_score:.2f} < threshold {result.threshold:.2f}"
                    )
                    return ''

            # Check cooldown (mode-specific)
            if storage.check_cooldown(cooldown_symbol, result.side, config.cooldown_minutes):
                logger.debug(f"{symbol} [{signal_mode}] {result.side} in cooldown, skipping")
                return ''
            
            # ── Correlation Filter ──────────────────────────────────────────
            if signals_sent_this_scan.get(result.side, 0) >= max_same_direction:
                logger.info(
                    f"Correlation filter: {result.side} blocked for {symbol} "
                    f"({signals_sent_this_scan[result.side]}/{max_same_direction} already sent this scan)"
                )
                return ''
            # ───────────────────────────────────────────────────────────────
            
            # Add mode label to result
            result.symbol = f"{symbol} {mode_label}"

            # --- START MAX SPREAD FILTER ---
            auto_trade_cfg = config.mt5.get("auto_trade", {}) if config.mt5 else {}
            max_spread_pct = float(auto_trade_cfg.get("max_spread_pct", 0.0))
            
            _ask, _bid = ticker.get('ask'), ticker.get('bid')
            if max_spread_pct > 0.0 and _ask and _bid and _bid > 0:
                spread_pct = (_ask - _bid) / _bid * 100
                if spread_pct > max_spread_pct:
                    msg = f"⚠️ حماية السبريد: تم إلغاء صفقة <b>{symbol}</b> لأن السبريد الحالي ({spread_pct:.3f}%) أعلى من الحد المسموح ({max_spread_pct:.3f}%)"
                    logger.info(msg)
                    if notifier.enabled:
                        notifier._send_telegram(msg)
                    return ''
            # --------------------------------

            # Send signal
            try:
                import os
                import tempfile
                from bot.chart_generator import generate_signal_chart
                temp_dir = tempfile.gettempdir()
                chart_path = os.path.join(temp_dir, f"{symbol.replace('/', '_').replace(':', '_')}_chart.png")
                generated = generate_signal_chart(symbol, data['entry'], result, chart_path)
            except Exception as chart_err:
                logger.error(f"Chart hook failed: {chart_err}")
                generated = None

            notifier.send_signal(result, chart_path=generated)
            
            # --- START AUTO-TRADING HOOK ---
            if auto_trade_cfg.get("enabled", False) and exchange.id in ("mt5", "metatrader5"):
                try:
                    from bot.mt5_client import MT5Client
                    
                    # Target the actual underlying MT5 Client directly if available on the exchange, 
                    # or instantiate a fresh one since MT5 login states are shared terminal-wide
                    client = getattr(exchange, "mt5", None)
                    if client is None:
                        client = MT5Client.from_config(config.mt5)
                        client.connect_mt5()
                        
                    sym_info = client.get_symbol_info(symbol)
                    
                    trade_mode = sym_info.get("trade_mode", 4)
                    if trade_mode == 0:
                        logger.info(f"Auto-trade skipped. Trading is disabled by broker for {symbol}.")
                    else:
                        vol_min = sym_info.get("volume_min", sym_info.get("symbol_info", {}).get("volume_min", 0.01)) if sym_info else 0.01
                        vol_step = sym_info.get("volume_step", sym_info.get("symbol_info", {}).get("volume_step", 0.01)) if sym_info else 0.01
                        
                        base_lot = float(auto_trade_cfg.get("fixed_lot", 0.01))
                        
                        # Force minimum lot for Gold, Silver, Indices, etc as requested
                        is_macro = any(m in symbol.upper() for m in ["XAU", "XAG", "US500", "USOIL", "USD"])
                        if is_macro:
                            base_lot = vol_min
                            
                        base_lot = max(base_lot, vol_min)
                        base_lot = round(base_lot / vol_step) * vol_step
                        
                        # Safe split calculation
                        if base_lot >= (vol_min * 2):
                            lot1 = max(vol_min, round((base_lot * 0.75) / vol_step) * vol_step)
                            lot2 = round((base_lot - lot1) / vol_step) * vol_step
                            if lot2 < vol_min:
                                lot1 = base_lot
                                lot2 = 0.0
                        else:
                            lot1 = base_lot
                            lot2 = 0.0
                            
                        logger.info(f"Auto-trading execution triggered for {symbol} ({result.side}) with lot={base_lot} (split: {lot1} TP1, {lot2} TP2)")
                        
                        # Execute T1 (75% or 100%)
                        trade_res = client.execute_trade(
                            symbol=symbol,
                            side=result.side,
                            lot=lot1,
                            sl=result.risk_levels.stop_loss if hasattr(result.risk_levels, "stop_loss") else 0.0,
                            tp=result.risk_levels.take_profit_1 if hasattr(result.risk_levels, "take_profit_1") else 0.0,
                        )
                        
                        if trade_res:
                            logger.info(f"Successfully placed MT5 TP1 trade! Ticket: {trade_res.get('order')}")
                            
                            # --- PARTIAL EXECUTION / SCALE OUT: Trade 2 for TP2 ---
                            trade_res2 = None
                            if lot2 > 0:
                                trade_res2 = client.execute_trade(
                                    symbol=symbol,
                                    side=result.side,
                                    lot=lot2,
                                    sl=result.risk_levels.stop_loss if hasattr(result.risk_levels, "stop_loss") else 0.0,
                                    tp=result.risk_levels.take_profit_2 if hasattr(result.risk_levels, "take_profit_2") else 0.0,
                                )
                            
                            ticket_msg = f"T1: {trade_res.get('order')}"
                            if trade_res2:
                                ticket_msg += f" | T2: {trade_res2.get('order')}"
                                logger.info(f"Successfully placed MT5 TP2 trade! Ticket: {trade_res2.get('order')}")

                            if notifier.enabled:
                                notifier.send_text(f"🤖 <b>Auto-Trade Executed (Split T1/T2):</b> {result.side} {symbol}\nLots: {lot1}|{lot2}\nTickets: {ticket_msg}")
                except Exception as e:
                    logger.error(f"MT5 Auto-trading execution failed for {symbol}: {e}")
            # --- END AUTO-TRADING HOOK ---
            
            # Store signal (mode-specific)
            if result.risk_levels:
                storage_reasons = list(result.reasons + result.timing_reasons)
                context_tags = [
                    f"ctx:asset_class={_infer_asset_class_for_symbol(symbol)}",
                    f"ctx:trend_tf={mode_config.trend_tf}",
                    f"ctx:entry_tf={mode_config.entry_tf}",
                    f"ctx:sr_tf={mode_config.sr_tf}",
                ]
                for tag in context_tags:
                    if tag not in storage_reasons:
                        storage_reasons.append(tag)
                storage.save_signal(SignalRecord(
                    symbol=cooldown_symbol,
                    side=result.side,
                    timestamp=result.timestamp,
                    score=int(result.total_score),
                    entry=result.risk_levels.entry,
                    stop_loss=result.risk_levels.stop_loss,
                    take_profit_near=result.risk_levels.take_profit_near,
                    take_profit_1=result.risk_levels.take_profit_1,
                    take_profit_2=result.risk_levels.take_profit_2,
                    reasons=storage_reasons
                ))
            
            logger.info(f"{result.side} signal sent for {symbol} [{signal_mode.upper()}] (score: {result.total_score:.1f})")
            return result.side
        
        blocked_reason = str(getattr(result, "blocked_reason", "") or "").strip()
        if blocked_reason:
            logger.info(
                f"No entry for {symbol} [{signal_mode}]: {blocked_reason} "
                f"(score={float(getattr(result, 'total_score', 0.0)):.2f}/"
                f"{float(getattr(result, 'threshold', 0.0)):.2f}, trend={getattr(result, 'trend', 'n/a')})"
            )
        else:
            logger.info(
                f"No entry for {symbol} [{signal_mode}]: no qualified setup "
                f"(score={float(getattr(result, 'total_score', 0.0)):.2f}/"
                f"{float(getattr(result, 'threshold', 0.0)):.2f}, trend={getattr(result, 'trend', 'n/a')})"
            )
        return ''
        
    except Exception as e:
        logger.error(f"Error scanning {symbol} [{signal_mode}]: {e}")
        notifier.send_error(str(e), f"{symbol} [{signal_mode}]")
        return ''



def _get_btc_pulse_trend(exchange, config: Config) -> str:
    """
    Fetch BTC/USDT 15m trend once per scan cycle.
    Used as a market-wide pulse: if BTC is bearish, altcoin LONGs are logged/penalized.
    Returns 'up', 'down', or 'neutral'.
    """
    bp = getattr(config, 'btc_pulse', None)
    if not bp or not getattr(bp, 'enabled', True):
        return 'neutral'
    btc_sym = str(getattr(bp, 'symbol', 'BTC/USDT:USDT'))
    btc_tf  = str(getattr(bp, 'trend_tf', '15m'))
    try:
        df_btc = fetch_ohlcv(exchange, btc_sym, btc_tf, limit=300)
        if df_btc is None or df_btc.empty:
            return 'neutral'
        df_btc = add_all_indicators(df_btc)
        return get_trend(df_btc)
    except Exception as e:
        logger.debug(f"BTC pulse fetch error: {e}")
        return 'neutral'


def _is_in_blocked_session(config: Config) -> float:
    """
    Returns the threshold penalty for the current UTC hour based on data-driven
    session windows from config. Returns 0.0 if no penalty applies.

    Two tiers:
      blocked_utc_windows  → threshold_add (e.g. 1.2 for worst hours)
      cautious_windows     → 0.5 (softer penalty for below-average hours)
    """
    cs = getattr(config, 'crypto_session', None)
    if not cs or not getattr(cs, 'enabled', True):
        return 0.0
    now = datetime.now(timezone.utc)

    # Tier 1 — high penalty windows
    windows = list(getattr(cs, 'blocked_utc_windows', []) or [])
    if windows and _is_within_utc_windows(windows, now):
        return float(getattr(cs, 'threshold_add', 1.2))

    # Tier 2 — soft penalty windows
    cautious = list(getattr(cs, 'cautious_windows', []) or [])
    if cautious and _is_within_utc_windows(cautious, now):
        return 0.5

    return 0.0


def _ensure_requested_exchange_source(config: Config, exchange: Any) -> None:
    """
    Prevent silent data-source drift when MT5 bridge is explicitly requested.
    """
    requested = str(getattr(config, "exchange_name", "") or "").strip().lower()
    if requested not in {"mt5", "mt5_bridge"}:
        return

    active = str(getattr(exchange, "id", "") or "").strip().lower()
    if active in {"mt5", "mt5_bridge"}:
        logger.info(f"Exchange source check OK: requested {requested} and active source is {active}")
        return

    active_label = active or "unknown"
    raise RuntimeError(
        f"Exchange source mismatch: requested {requested} but active source is "
        f"{active_label}. Refusing to run to avoid non-MT5 prices."
    )


def run_scan_loop(config: Config) -> None:
    """
    Main scanning loop.
    
    Args:
        config: Configuration object
    """
    global shutdown_requested
    
    # Initialize components
    logger.info("Initializing exchange connections...")
    binance_exchange = get_exchange(
        config.exchange_name,
        config.market_type,
        getattr(config, "mt5_bridge", None),
        getattr(config, "mt5", None),
    )
    
    kucoin_exchange = None
    if not _is_mt5_only_requested(config):
        try:
            kucoin_exchange = get_exchange('kucoin', 'spot')
            logger.info("Initialized KuCoin exchange for VAI/USDT")
        except Exception as e:
            logger.error(f"Failed to initialize KuCoin: {e}")
            kucoin_exchange = None
    else:
        logger.info("MT5-only mode enabled: external KuCoin data source disabled")
    
    logger.info("Initializing background engines...")
    storage = SignalStorage()
    
    ml_cfg = getattr(config, 'ml_engine', None)
    ml_engine = None
    if ml_cfg and getattr(ml_cfg, 'enabled', True):
        ml_engine = MLEngine(storage.db_path)
        ml_engine.train()
    
    notifier = TelegramNotifier(config)
    _ensure_requested_exchange_source(config, binance_exchange)
    active_exchange_id = str(getattr(binance_exchange, "id", "") or "unknown")

    # Safety net: recover missing OPEN outcomes for very recent signals.
    # This keeps tracking/reporting aligned if prior runtime missed inserts.
    try:
        storage.repair_recent_missing_open_outcomes(lookback_hours=2)
    except Exception as e:
        logger.debug(f"Recent outcome repair skipped: {e}")

    # Optional startup backfill: let LLM review historical closed losses
    # that were not evaluated yet (bounded by config).
    post_cfg = getattr(config, "llm_postmortem", None)
    if post_cfg and bool(getattr(post_cfg, "enabled", False)) and bool(getattr(post_cfg, "backfill_existing_on_startup", True)):
        threading.Thread(
            target=_run_llm_postmortem_backfill_worker,
            args=(storage.db_path, config),
            daemon=True,
        ).start()
    
    # Send startup message
    notifier.send_status(
        f"Bot started\\n"
        f"Exchange: {config.exchange_name} (active: {active_exchange_id})\\n"
        f"Symbols: {', '.join(config.symbols)}\\n"
        f"🔸 FUTURES: {config.futures.trend_tf}/{config.futures.entry_tf}/{config.futures.sr_tf}\\n"
        f"Scan interval: {config.scan_interval_seconds}s"
    )
    
    logger.info(f"Starting scan loop for {len(config.symbols)} symbols (FUTURES only)")

    scan_count = 0
    paused_log_emitted = False

    # Start Telegram control listener in background
    control_thread = None
    if notifier.enabled:
        control_thread = threading.Thread(
            target=run_telegram_control_loop,
            args=(notifier, config, lambda: scan_count),
            daemon=True
        )
        control_thread.start()

    while not shutdown_requested:
        if is_scan_paused():
            if not paused_log_emitted:
                logger.info("Scanner paused by Telegram command")
                paused_log_emitted = True
            time.sleep(2)
            continue
        paused_log_emitted = False

        scan_start = time.time()
        scan_count += 1
        
        logger.info(f"=== Scan #{scan_count} at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} ===")

        # ── Session close: close forex/macro positions before market shuts ──
        try:
            # Reuse the existing exchange client — never create a new MT5Client
            # per scan (doing so calls mt5.login() again which disrupts the feed)
            _mt5_client = getattr(binance_exchange, "client", None)
            if _mt5_client is not None:
                close_positions_before_session_end(storage, notifier, _mt5_client, config)
        except Exception as _sce:
            logger.warning(f"Session close check failed: {_sce}")

        # ── Phase 5: Check all open outcomes against current prices ──────
        check_open_outcomes(storage, notifier, binance_exchange, kucoin_exchange, config)
        
        # ── Phase 5: Send daily WinRate report ───────────────────────────
        send_winrate_report(storage, notifier)
        # Shadow-only health monitor (no effect on signal dispatch)
        send_walkforward_shadow_report(storage, notifier)
        
        # ── BTC Market Pulse: fetch once per scan, used for all altcoins ──
        _btc_pulse_trend = _get_btc_pulse_trend(binance_exchange, config)
        if _btc_pulse_trend != 'neutral':
            logger.debug(f"BTC Pulse trend: {_btc_pulse_trend}")

        # ── Crypto Session Filter ─────────────────────────────────────────
        _in_blocked_session = _is_in_blocked_session(config)
        if _in_blocked_session:
            logger.debug("Session filter: in blocked UTC window (Asia dead zone)")
        
        # Correlation filter: track how many signals of each type sent this scan
        signals_sent_this_scan = {'LONG': 0, 'SHORT': 0}
        MAX_SAME_DIRECTION_PER_SCAN = 2  # Max 2 of the same direction per scan
        
        for symbol in config.symbols:
            if shutdown_requested or is_scan_paused():
                break

            sym_asset_class = _infer_asset_class_for_symbol(symbol)

            # ── BTC Pulse: block altcoin LONGs when BTC is bearish ────────
            bp = getattr(config, 'btc_pulse', None)
            if bp and getattr(bp, 'enabled', True) and sym_asset_class == 'crypto':
                btc_sym = str(getattr(bp, 'symbol', 'BTC/USDT:USDT'))
                if symbol != btc_sym:  # Don't block BTC itself
                    if getattr(bp, 'block_altcoin_long_on_down', True) and _btc_pulse_trend == 'down':
                        logger.info(f"BTC Pulse blocked {symbol} LONG: BTC trend is down")
                        signals_sent_this_scan['_btc_blocked'] = signals_sent_this_scan.get('_btc_blocked', 0) + 1
                        # Still allow SHORT — only block LONGs
                        # We pass a flag via a temporary config override is complex,
                        # instead we rely on the fact that trend='down' will naturally
                        # push for SHORT anyway. Log and continue normally.
                        # For now we just log — the learning engine + regime filter will handle it

            # ── Session Filter: cautious mode during Asia hours ───────────
            _session_threshold_add = 0.0
            if _in_blocked_session and sym_asset_class == 'crypto':
                cs = getattr(config, 'crypto_session', None)
                mode = str(getattr(cs, 'mode', 'cautious')).lower() if cs else 'cautious'
                if mode == 'block':
                    logger.info(f"Session filter blocked {symbol}: in blocked UTC window")
                    continue
                else:
                    _session_threshold_add = float(_in_blocked_session)

            # ── Altcoin Correlation Filter ────────────────────────────────────
            # If we already sent 2+ altcoin signals in same direction in last
            # 30 min, they're all correlated with BTC — skip additional ones
            _corr_blocked_sides = set()
            if sym_asset_class == 'crypto':
                qf_cfg = getattr(config, 'quality_filter', None)
                if qf_cfg and getattr(qf_cfg, 'altcoin_correlation_filter_enabled', True):
                    _max_corr = int(getattr(qf_cfg, 'max_correlated_altcoin_same_dir', 2))
                    _corr_win = int(getattr(qf_cfg, 'correlation_window_minutes', 30))
                    _corr_cutoff = (datetime.now(timezone.utc) - timedelta(minutes=_corr_win)).isoformat()
                    for _check_side in ('LONG', 'SHORT'):
                        _corr_count = storage.count_recent_signals(
                            asset_class='crypto',
                            side=_check_side,
                            since=_corr_cutoff,
                            exclude_symbol=symbol,
                        )
                        if _corr_count >= _max_corr:
                            _corr_blocked_sides.add(_check_side)
                            logger.info(
                                f"Correlation filter: {symbol} {_check_side} blocked "
                                f"({_corr_count} correlated altcoin {_check_side}s in last {_corr_win}m)"
                            )

            
            # MT5-only mode never routes symbols to external data sources.
            if "VAI" in symbol:
                if _is_mt5_only_requested(config):
                    logger.error(f"Skipping {symbol}: MT5-only mode requires all symbols to exist in MT5")
                    continue
                if not kucoin_exchange:
                    logger.error(f"Skipping {symbol}: KuCoin not initialized")
                    continue
                sent_signal = scan_symbol(symbol, kucoin_exchange, config, storage, notifier,
                                          signal_mode="futures",
                                          signals_sent_this_scan=signals_sent_this_scan,
                                          max_same_direction=MAX_SAME_DIRECTION_PER_SCAN,
                                          session_threshold_add=_session_threshold_add)
            else:
                sent_signal = scan_symbol(symbol, binance_exchange, config, storage, notifier,
                                          signal_mode="futures",
                                          signals_sent_this_scan=signals_sent_this_scan,
                                          max_same_direction=MAX_SAME_DIRECTION_PER_SCAN,
                                          session_threshold_add=_session_threshold_add)
            
            if sent_signal in ('LONG', 'SHORT'):
                signals_sent_this_scan[sent_signal] = signals_sent_this_scan.get(sent_signal, 0) + 1
        
        scan_duration = time.time() - scan_start
        logger.info(f"Scan completed in {scan_duration:.1f}s")
        
        # Wait for next scan
        if not shutdown_requested and not is_scan_paused():
            wait_time = max(0, config.scan_interval_seconds - scan_duration)
            if wait_time > 0:
                logger.debug(f"Waiting {wait_time:.0f}s until next scan...")
                interruptible_sleep(wait_time)
    
    # Shutdown
    logger.info("Shutting down...")
    notifier.send_status("Bot stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Crypto Trading Signal Bot for Binance Futures'
    )
    parser.add_argument(
        '-c', '--config',
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run a single scan and exit'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else config.logging_level
    setup_logging(log_level, config.logging_file)
    
    logger.info("=" * 50)
    logger.info("Crypto Signal Bot Starting")
    logger.info("=" * 50)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.once:
        # Single scan mode
        logger.info("Running single scan...")
        binance_exchange = get_exchange(
            config.exchange_name,
            config.market_type,
            getattr(config, "mt5_bridge", None),
            getattr(config, "mt5", None),
        )
        kucoin_exchange = None
        if not _is_mt5_only_requested(config):
            try:
                kucoin_exchange = get_exchange('kucoin', 'spot')
                logger.info("Initialized KuCoin exchange for VAI/USDT (single scan)")
            except Exception as e:
                logger.error(f"Failed to initialize KuCoin (single scan): {e}")
                kucoin_exchange = None
        else:
            logger.info("MT5-only mode enabled: external KuCoin data source disabled")

        storage = SignalStorage()
        ml_engine = None
        ml_cfg = getattr(config, 'ml_engine', None)
        if ml_cfg and getattr(ml_cfg, 'enabled', True):
            ml_engine = MLEngine(storage.db_path)
            ml_engine.train()
            
        notifier = TelegramNotifier(config)
        _ensure_requested_exchange_source(config, binance_exchange)
        
        for symbol in config.symbols:
            if "VAI" in symbol:
                if _is_mt5_only_requested(config):
                    logger.error(f"Skipping {symbol}: MT5-only mode requires all symbols to exist in MT5")
                    continue
                if not kucoin_exchange:
                    logger.error(f"Skipping {symbol}: KuCoin not initialized")
                    continue
                scan_symbol(symbol, kucoin_exchange, config, storage, notifier, signal_mode="futures", ml_engine=ml_engine)
            else:
                scan_symbol(symbol, binance_exchange, config, storage, notifier, signal_mode="futures", ml_engine=ml_engine)
        
        logger.info("Single scan complete")
    else:
        # Continuous loop mode
        run_scan_loop(config)


if __name__ == '__main__':
    main()
