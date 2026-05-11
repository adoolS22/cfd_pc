"""
Quality-First Shadow Evaluator
==============================
Strict setup evaluator used for side-by-side comparison.
It does NOT change the live signal flow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .patterns import get_pattern_for_direction


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        if isinstance(value, float) and pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _has_high_impact_window(timing_info: Any) -> bool:
    if not timing_info:
        return False
    checks = [
        getattr(getattr(timing_info, "fomc_analysis", None), "in_high_vol_window", False),
        getattr(getattr(timing_info, "cpi_analysis", None), "in_high_vol_window", False),
        getattr(getattr(timing_info, "nfp_analysis", None), "in_high_vol_window", False),
        getattr(getattr(timing_info, "powell_analysis", None), "in_high_vol_window", False),
        getattr(getattr(timing_info, "fomc_minutes_analysis", None), "in_high_vol_window", False),
    ]
    return any(bool(x) for x in checks)


def _push_check(
    checks: List[Dict[str, Any]],
    *,
    key: str,
    label: str,
    ok: bool,
    details: str,
    required: bool = True,
) -> None:
    checks.append(
        {
            "key": key,
            "label": label,
            "ok": bool(ok),
            "required": bool(required),
            "details": str(details),
        }
    )


def evaluate_quality_first(
    symbol: str,
    side: str,
    ticker: Dict[str, Any],
    df_trend: pd.DataFrame,
    df_entry: pd.DataFrame,
    result: Any,
    config: Any,
) -> Dict[str, Any]:
    """
    Evaluate strict "quality-first" entry policy for comparison only.

    Returns a structured dict that can be displayed in Telegram.
    """
    qf = getattr(config, "quality_first", None)
    section_name = str(getattr(qf, "name_ar", "الجودة أهم")) if qf else "الجودة أهم"

    payload: Dict[str, Any] = {
        "mode": "quality_first",
        "name": section_name,
        "enabled": bool(getattr(qf, "enabled", False)) if qf else False,
        "decision": "WAIT",
        "allow": False,
        "score_pct": 0.0,
        "passed_checks": 0,
        "total_checks": 0,
        "checks": [],
        "failed_required": [],
    }
    if not qf or not getattr(qf, "enabled", False):
        return payload

    side_u = str(side or "").strip().upper()
    if side_u not in {"LONG", "SHORT"}:
        payload["checks"] = [
            {
                "key": "direction",
                "label": "direction",
                "ok": False,
                "required": True,
                "details": "No tradable side",
            }
        ]
        payload["failed_required"] = ["direction"]
        payload["total_checks"] = 1
        return payload

    checks: List[Dict[str, Any]] = []
    trend_last = df_trend.iloc[-1] if not df_trend.empty else None
    entry_last = df_entry.iloc[-1] if not df_entry.empty else None
    zone_info = getattr(result, "zone_info", None) or {}

    # 1) Trend strength (ADX)
    adx_val = _safe_float(zone_info.get("adx"))
    if adx_val is None and trend_last is not None:
        adx_val = _safe_float(trend_last.get("adx"), 0.0)
    adx_val = float(adx_val or 0.0)
    min_adx = float(getattr(qf, "htf_adx_min", 22.0))
    _push_check(
        checks,
        key="adx",
        label="ADX",
        ok=adx_val >= min_adx,
        details=f"{adx_val:.1f} >= {min_adx:.1f}",
        required=True,
    )

    # 2) EMA trend alignment
    ema50 = _safe_float(trend_last.get("ema_50")) if trend_last is not None else None
    ema200 = _safe_float(trend_last.get("ema_200")) if trend_last is not None else None
    require_ema_alignment = bool(getattr(qf, "require_ema_alignment", True))
    allow_counter_trend = bool(getattr(qf, "allow_counter_trend", False))
    ema_ok = False
    if ema50 is not None and ema200 is not None:
        ema_ok = (ema50 > ema200) if side_u == "LONG" else (ema50 < ema200)
    _push_check(
        checks,
        key="ema_alignment",
        label="EMA alignment",
        ok=ema_ok or (allow_counter_trend and not require_ema_alignment),
        details=(
            f"EMA50={ema50:.6f}, EMA200={ema200:.6f}" if ema50 is not None and ema200 is not None else "EMA missing"
        ),
        required=bool(require_ema_alignment and not allow_counter_trend),
    )

    # 3) Zone proximity and side alignment
    require_zone = bool(getattr(qf, "require_zone_proximity", True))
    max_zone_distance_pct = max(0.01, float(getattr(qf, "max_zone_distance_pct", 0.25)))
    distance_raw = _safe_float(zone_info.get("distance_pct"), float("inf"))
    distance_pct = float(distance_raw * 100.0) if distance_raw is not None else float("inf")
    in_zone = bool(zone_info.get("in_zone", False))
    zone_type = str(zone_info.get("zone_type") or "").strip().lower()
    near_enough = in_zone or (distance_pct <= max_zone_distance_pct)
    aligned_zone = (side_u == "LONG" and zone_type == "support") or (side_u == "SHORT" and zone_type == "resistance")
    zone_ok = near_enough and aligned_zone
    _push_check(
        checks,
        key="zone",
        label="Zone proximity",
        ok=zone_ok,
        details=f"{zone_type or 'none'} | dist={distance_pct:.3f}% <= {max_zone_distance_pct:.3f}%",
        required=require_zone,
    )

    # 4) Candle confirmation
    require_candle = bool(getattr(qf, "require_candle_confirmation", True))
    min_strength = max(1, int(getattr(qf, "min_pattern_strength", 2)))
    pattern_dir = "bullish" if side_u == "LONG" else "bearish"
    pattern = get_pattern_for_direction(df_entry, pattern_dir) if not df_entry.empty else None
    pattern_strength = int(getattr(pattern, "strength", 0) or 0)
    pattern_name = str(getattr(pattern, "name", "")) if pattern is not None else "none"
    pattern_ok = pattern is not None and pattern_strength >= min_strength
    _push_check(
        checks,
        key="candle",
        label="Candle confirmation",
        ok=pattern_ok,
        details=f"{pattern_name} (strength={pattern_strength})",
        required=require_candle,
    )

    # 5) Volume quality
    min_volume_ratio = float(getattr(qf, "min_volume_ratio", 1.10))
    vol_now = _safe_float(entry_last.get("volume")) if entry_last is not None else None
    vol_avg = _safe_float(entry_last.get("vol_sma_20")) if entry_last is not None else None
    ratio = (float(vol_now) / float(vol_avg)) if (vol_now is not None and vol_avg and vol_avg > 0) else 0.0
    _push_check(
        checks,
        key="volume",
        label="Volume ratio",
        ok=ratio >= min_volume_ratio,
        details=f"{ratio:.2f} >= {min_volume_ratio:.2f}",
        required=True,
    )

    # 6) Spread quality
    max_spread = float(getattr(qf, "max_spread_pct", 0.18))
    bid = _safe_float(ticker.get("bid"))
    ask = _safe_float(ticker.get("ask"))
    last = _safe_float(ticker.get("last"))
    if bid is not None and ask is not None and last and last > 0:
        spread_pct = abs(ask - bid) / last * 100.0
        spread_ok = spread_pct <= max_spread
        spread_details = f"{spread_pct:.3f}% <= {max_spread:.3f}%"
    else:
        spread_ok = True
        spread_details = "Spread unavailable"
    _push_check(
        checks,
        key="spread",
        label="Spread",
        ok=spread_ok,
        details=spread_details,
        required=True,
    )

    # 7) RR quality
    rr1_min = float(getattr(qf, "min_rr_tp1", 1.0))
    rr2_min = float(getattr(qf, "min_rr_tp2", 1.8))
    levels = getattr(result, "risk_levels", None)
    if levels:
        entry = _safe_float(getattr(levels, "entry", None))
        sl = _safe_float(getattr(levels, "stop_loss", None))
        tp1 = _safe_float(getattr(levels, "take_profit_1", None))
        tp2 = _safe_float(getattr(levels, "take_profit_2", None))
        risk = abs((entry or 0.0) - (sl or 0.0))
        if risk > 0 and entry is not None and tp1 is not None and tp2 is not None:
            rr1 = abs(tp1 - entry) / risk
            rr2 = abs(tp2 - entry) / risk
            rr_ok = rr1 >= rr1_min and rr2 >= rr2_min
            rr_details = f"RR1={rr1:.2f} (>= {rr1_min:.2f}) | RR2={rr2:.2f} (>= {rr2_min:.2f})"
        else:
            rr_ok = False
            rr_details = "Invalid risk levels"
    else:
        rr_ok = False
        rr_details = "Risk levels missing"
    _push_check(
        checks,
        key="rr",
        label="Risk/Reward",
        ok=rr_ok,
        details=rr_details,
        required=True,
    )

    # 8) High-impact news window
    avoid_high_impact_news = bool(getattr(qf, "avoid_high_impact_news", True))
    in_high_impact = _has_high_impact_window(getattr(result, "timing_info", None))
    _push_check(
        checks,
        key="news_window",
        label="News window",
        ok=not in_high_impact,
        details="No high-impact window" if not in_high_impact else "High-impact window active",
        required=avoid_high_impact_news,
    )

    total_checks = len(checks)
    passed_checks = sum(1 for c in checks if c.get("ok"))
    failed_required = [c["key"] for c in checks if c.get("required") and not c.get("ok")]
    allow = len(failed_required) == 0

    payload.update(
        {
            "decision": side_u if allow else "WAIT",
            "allow": allow,
            "score_pct": (passed_checks / total_checks * 100.0) if total_checks > 0 else 0.0,
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "checks": checks,
            "failed_required": failed_required,
        }
    )
    return payload

