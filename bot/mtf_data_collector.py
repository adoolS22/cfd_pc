"""
Multi-Timeframe Data Collector
==============================
Collects OHLCV from MT5 across multiple timeframes and runs
code-first SMC detection on each, producing a structured JSON
dict ready for the LLM pending-order planner.

The LLM receives pre-computed structures (FVGs, OBs, sweeps, etc.),
NOT raw candle data. This ensures accuracy and speed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from loguru import logger

from .smc import (
    detect_unmitigated_fvgs,
    detect_ifvgs,
    detect_order_blocks,
    detect_liquidity_sweeps,
    detect_structure_breaks,
    detect_displacement,
    detect_displacements,
    get_dealing_range,
    FVG,
    OrderBlock,
    LiquiditySweep,
    StructureBreak,
    Displacement,
    DealingRange,
)
from .indicators import add_all_indicators, get_trend


# Timeframes to collect and analyze
_TIMEFRAMES = [
    ("1d", 60, "daily"),
    ("4h", 100, "h4"),
    ("1h", 200, "h1"),
    ("15m", 200, "m15"),
    ("5m", 200, "m5"),
    ("1m", 200, "m1"),
]


def _rates_to_dataframe(rates: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert MT5 rate dicts to a pandas DataFrame with standard columns."""
    if not rates:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    # Ensure standard column names
    rename_map = {}
    for src, dst in [("tick_volume", "volume")]:
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    # Ensure datetime index
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
    elif "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    if "datetime" in df.columns:
        df.set_index("datetime", inplace=True, drop=False)
    return df


def _compute_htf_liquidity(daily_df: pd.DataFrame) -> Dict[str, float]:
    """Previous-day and previous-week high/low — the most obvious resting
    liquidity (PDH/PDL buy-side, PWH/PWL sell-side analogues) that price draws to.

    Computed from the daily dataframe (datetime indexed). The last bar is the
    still-forming period, so the previous COMPLETED day/week is index -2.
    """
    out: Dict[str, float] = {}
    if daily_df is None or daily_df.empty or "datetime" not in getattr(daily_df, "columns", []):
        return out
    try:
        d = daily_df.resample("D").agg({"high": "max", "low": "min"}).dropna()
        if len(d) >= 2:
            out["pdh"] = float(d["high"].iloc[-2])
            out["pdl"] = float(d["low"].iloc[-2])
    except Exception:
        pass
    try:
        w = daily_df.resample("W").agg({"high": "max", "low": "min"}).dropna()
        if len(w) >= 2:
            out["pwh"] = float(w["high"].iloc[-2])
            out["pwl"] = float(w["low"].iloc[-2])
    except Exception:
        pass
    return out


def _detect_swing_levels(df: pd.DataFrame, lookback: int = 5) -> Dict[str, List[float]]:
    """Detect swing high/low levels for BSL/SSL identification."""
    bsl: List[float] = []
    ssl: List[float] = []
    if len(df) < lookback * 2 + 1:
        return {"bsl": bsl, "ssl": ssl}

    highs = df["high"].values
    lows = df["low"].values

    for i in range(lookback, len(df) - lookback):
        # Swing high: higher than `lookback` bars on each side
        if all(highs[i] >= highs[i - j] for j in range(1, lookback + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, lookback + 1)):
            bsl.append(float(highs[i]))
        # Swing low: lower than `lookback` bars on each side
        if all(lows[i] <= lows[i - j] for j in range(1, lookback + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, lookback + 1)):
            ssl.append(float(lows[i]))

    # Sort and deduplicate, keep only the top 5 most relevant
    bsl = sorted(set(bsl), reverse=True)[:5]
    ssl = sorted(set(ssl))[:5]
    return {"bsl": bsl, "ssl": ssl}


def _fvg_to_dict(fvg: FVG) -> Dict[str, Any]:
    return {
        "direction": fvg.direction,
        "top": round(float(fvg.top), 5),
        "bottom": round(float(fvg.bottom), 5),
        "midpoint": round((float(fvg.top) + float(fvg.bottom)) / 2, 5),
        "mitigated": fvg.mitigated,
        "index": int(fvg.index),
    }


def _ob_to_dict(ob: OrderBlock) -> Dict[str, Any]:
    return {
        "direction": ob.direction,
        "top": round(float(ob.top), 5),
        "bottom": round(float(ob.bottom), 5),
        "midpoint": round((float(ob.top) + float(ob.bottom)) / 2, 5),
        "index": int(ob.index),
    }


def _sweep_to_dict(sweep: LiquiditySweep) -> Dict[str, Any]:
    return {
        "direction": sweep.direction,
        "swept_price": round(float(sweep.swept_price), 5),
        "killzone": sweep.killzone,
        "index": int(sweep.index),
    }


def _sb_to_dict(sb: StructureBreak) -> Dict[str, Any]:
    # detect_structure_breaks is called with require_body_close=True, so every
    # break here is a body-close break (body_close is implied True).
    return {
        "break_type": sb.break_type,
        "direction": sb.direction,
        "price": round(float(sb.broken_price), 5),
        "body_close": True,
        "index": int(sb.index),
    }


def _displacement_to_dict(disp: Optional[Displacement]) -> Optional[Dict[str, Any]]:
    if disp is None:
        return None
    return {
        "direction": disp.direction,
        "atr_multiple": round(float(disp.magnitude_atr), 2),
        "body_ratio": round(float(disp.body_ratio), 2),
        "start_index": int(disp.start_index),
        "end_index": int(disp.end_index),
    }


def _dealing_range_to_dict(dr: Optional[DealingRange], current_price: float) -> Dict[str, Any]:
    if dr is None:
        return {
            "high": None, "low": None, "equilibrium": None,
            "location": "unknown", "trend": "unknown",
        }
    eq = dr.equilibrium
    if current_price > eq:
        location = "premium"
    elif current_price < eq:
        location = "discount"
    else:
        location = "equilibrium"
    return {
        "high": round(float(dr.top), 5),
        "low": round(float(dr.bottom), 5),
        "equilibrium": round(float(eq), 5),
        "location": location,
        "trend": dr.trend,
    }


def _analyze_single_timeframe(
    df: pd.DataFrame,
    current_price: float,
    tf_label: str,
) -> Dict[str, Any]:
    """Run all SMC detections on a single-timeframe DataFrame.

    Returns a structured dict of detected SMC concepts — NOT raw candles.
    """
    result: Dict[str, Any] = {
        "timeframe": tf_label,
        "bars": len(df),
    }

    if df.empty or len(df) < 20:
        result["error"] = "insufficient_data"
        return result

    try:
        # Add technical indicators
        df_ind = add_all_indicators(df.copy())
    except Exception as e:
        logger.debug(f"MTF indicator error [{tf_label}]: {e}")
        df_ind = df.copy()

    # Trend
    try:
        result["trend"] = get_trend(df_ind)
    except Exception:
        result["trend"] = "unknown"

    # Key indicators
    try:
        result["atr"] = round(float(df_ind["atr_14"].iloc[-1]), 5) if "atr_14" in df_ind.columns else None
        result["rsi"] = round(float(df_ind["rsi_14"].iloc[-1]), 2) if "rsi_14" in df_ind.columns else None
        result["adx"] = round(float(df_ind["adx"].iloc[-1]), 2) if "adx" in df_ind.columns else None
    except Exception:
        result["atr"] = None
        result["rsi"] = None
        result["adx"] = None

    # Latest OHLCV summary (not full candles)
    try:
        last = df_ind.iloc[-1]
        result["last_candle"] = {
            "open": round(float(last["open"]), 5),
            "high": round(float(last["high"]), 5),
            "low": round(float(last["low"]), 5),
            "close": round(float(last["close"]), 5),
        }
    except Exception:
        pass

    # Dealing Range + Premium/Discount
    try:
        dr = get_dealing_range(df_ind, lookback=min(150, len(df_ind)))
        result["dealing_range"] = _dealing_range_to_dict(dr, current_price)
    except Exception:
        result["dealing_range"] = _dealing_range_to_dict(None, current_price)

    # BSL / SSL levels
    try:
        swing_lookback = 3 if tf_label in ("m1", "m5") else 5
        levels = _detect_swing_levels(df_ind, lookback=swing_lookback)
        result["bsl_levels"] = levels["bsl"]
        result["ssl_levels"] = levels["ssl"]
    except Exception:
        result["bsl_levels"] = []
        result["ssl_levels"] = []

    # FVGs (unmitigated only)
    try:
        fvgs = detect_unmitigated_fvgs(df_ind, lookback=50)
        result["fvgs"] = [_fvg_to_dict(f) for f in fvgs[:5]]
    except Exception:
        result["fvgs"] = []

    # Order Blocks
    try:
        obs = detect_order_blocks(df_ind, lookback=50)
        result["order_blocks"] = [_ob_to_dict(ob) for ob in obs[:5]]
    except Exception:
        result["order_blocks"] = []

    # Inversion FVGs (breaker-style POIs: a mitigated FVG that flipped polarity)
    try:
        ifvgs = detect_ifvgs(df_ind, lookback=100)
        result["ifvgs"] = [_fvg_to_dict(f) for f in ifvgs[:5]]
    except Exception:
        result["ifvgs"] = []

    # Structure Breaks (BOS, CHOCH) — body-close only, per SMC methodology
    try:
        breaks = detect_structure_breaks(df_ind, lookback=50, require_body_close=True)
        result["structure_breaks"] = [_sb_to_dict(sb) for sb in breaks[:5]]
    except Exception:
        result["structure_breaks"] = []

    # Displacement — keep the latest (for summaries) plus the full recent list
    # so the confluence can find the bias-direction impulse behind a pullback.
    try:
        disp = detect_displacement(df_ind, lookback=30)
        result["displacement"] = _displacement_to_dict(disp)
    except Exception:
        result["displacement"] = None
    try:
        disps = detect_displacements(df_ind, lookback=30)
        result["displacements"] = [_displacement_to_dict(d) for d in disps]
    except Exception:
        result["displacements"] = []

    # Liquidity Sweeps — scan a recent window (not just the last candle) so the
    # confluence chain can reference a sweep that happened several bars ago.
    try:
        sweeps = detect_liquidity_sweeps(df_ind, lookback=30, scan_last=60)
        result["liquidity_sweeps"] = [_sweep_to_dict(s) for s in sweeps[-10:]]
    except Exception:
        result["liquidity_sweeps"] = []

    return result


def collect_mtf_analysis(
    symbol: str,
    mt5_client: Any,
    existing_pending_orders: Optional[List[Dict[str, Any]]] = None,
    existing_positions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Collect multi-timeframe data from MT5 and run code-first SMC analysis.

    Args:
        symbol: Trading symbol (e.g. 'XAUUSD', 'BTC/USDT:USDT')
        mt5_client: Connected MT5Client instance
        existing_pending_orders: Currently active pending orders for this symbol
        existing_positions: Currently open positions for this symbol

    Returns:
        Structured dict with per-timeframe SMC analysis, ready for LLM consumption.
    """
    # Get current price
    try:
        tick = mt5_client.get_tick(symbol)
        bid = float(tick.get("bid", 0))
        ask = float(tick.get("ask", 0))
        current_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        spread = float(tick.get("spread", 0))
    except Exception as e:
        logger.warning(f"MTF collector: cannot get tick for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}

    if current_price <= 0:
        return {"symbol": symbol, "error": "no_price_data"}

    result: Dict[str, Any] = {
        "symbol": symbol,
        "current_price": round(current_price, 5),
        "bid": round(bid, 5),
        "ask": round(ask, 5),
        "spread": round(spread, 5),
    }

    # Collect and analyze each timeframe
    daily_df = None
    for tf, count, key in _TIMEFRAMES:
        try:
            rates = mt5_client.get_rates(symbol, tf, count)
            df = _rates_to_dataframe(rates)
            if key == "daily":
                daily_df = df
            result[key] = _analyze_single_timeframe(df, current_price, key)
        except Exception as e:
            logger.debug(f"MTF collector [{symbol}] {key}: {e}")
            result[key] = {"timeframe": key, "error": str(e)}

    # Previous day/week high & low — explicit draw-on-liquidity targets.
    try:
        result["htf_liquidity"] = _compute_htf_liquidity(daily_df)
    except Exception:
        result["htf_liquidity"] = {}

    # Attach existing orders and positions context
    result["existing_pending_orders"] = _simplify_orders(existing_pending_orders or [])
    result["existing_positions"] = _simplify_positions(existing_positions or [])

    return result


def _simplify_orders(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simplify MT5 order dicts to essential fields for LLM context."""
    simplified = []
    for o in orders:
        order_type_int = o.get("type", -1)
        if order_type_int == 2:
            ot = "BUY_LIMIT"
        elif order_type_int == 3:
            ot = "SELL_LIMIT"
        elif order_type_int == 4:
            ot = "BUY_STOP"
        elif order_type_int == 5:
            ot = "SELL_STOP"
        else:
            ot = f"TYPE_{order_type_int}"
        simplified.append({
            "ticket": o.get("ticket"),
            "type": ot,
            "price": round(float(o.get("price_open", 0)), 5),
            "sl": round(float(o.get("sl", 0)), 5),
            "tp": round(float(o.get("tp", 0)), 5),
            "volume": float(o.get("volume_current", o.get("volume_initial", 0))),
        })
    return simplified


def _simplify_positions(positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simplify MT5 position dicts to essential fields for LLM context."""
    simplified = []
    for p in positions:
        side = "LONG" if p.get("type", 0) == 0 else "SHORT"
        simplified.append({
            "ticket": p.get("ticket"),
            "side": side,
            "entry_price": round(float(p.get("price_open", 0)), 5),
            "current_price": round(float(p.get("price_current", 0)), 5),
            "sl": round(float(p.get("sl", 0)), 5),
            "tp": round(float(p.get("tp", 0)), 5),
            "volume": float(p.get("volume", 0)),
            "profit": round(float(p.get("profit", 0)), 2),
        })
    return simplified
