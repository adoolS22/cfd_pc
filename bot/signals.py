"""
Signal Scoring Engine
=====================
Combines all analysis components into scored trading signals.
"""

import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from loguru import logger

from .indicators import add_all_indicators, get_trend, check_rsi_divergence, get_ichimoku_signal, analyze_vpa
from .zones import build_zones, is_price_in_zone, get_nearest_support, get_nearest_resistance, Zone
from .patterns import (
    get_pattern_for_direction, get_advisory_pattern_for_direction, detect_wave3_setup, CandlePattern,
    detect_trendline_break, detect_structure
)
from .smc import (
    detect_unmitigated_fvgs, detect_order_blocks, detect_liquidity_sweeps,
    get_nearest_unmitigated_fvg, get_nearest_order_block,
    detect_structure_breaks, get_latest_structure_break, StructureBreak
)
from .gann import calculate_gann_angles, analyze_square9, GannAnalysis, Square9Analysis
from .time_cycles import analyze_52_cycle, analyze_lunar, CycleAnalysis, LunarAnalysis
from .calendar_events import (
    analyze_fomc, FOMCAnalysis,
    analyze_cpi, CPIAnalysis,
    analyze_nfp, NFPAnalysis,
    analyze_powell_speeches, PowellAnalysis,
    analyze_fomc_minutes, FOMCMinutesAnalysis,
    analyze_market_sentiment, SentimentAnalysis,
    analyze_social_sentiment, SocialSentimentAnalysis,
)
from .risk import calculate_risk_levels, RiskLevels
from .utils import Config
from .fibonacci import calculate_fib_levels, FibAnalysis
from .llm_market_news import get_market_news_signal
from .onchain_analysis import analyze_onchain, OnChainAnalysis
from .expert_advisor import get_expert_trade_opinion, get_manager_opinion


_ASSET_MACRO_KEYS = ("XAU", "XAG", "OIL", "WTI", "BRENT", "SNP500", "SPX500", "S&P500", "SP500", "EURUSD", "EUR/USD")

# Rolling score history for percentile filtering (last 500 signals, in-memory)
# Keeps track of score distribution so we only trade when score is genuinely above average
_score_history: deque = deque(maxlen=500)


def _infer_asset_class(symbol: str) -> str:
    text = str(symbol or "").upper()
    if any(k in text for k in _ASSET_MACRO_KEYS):
        return "macro"
    return "crypto"


def _base_threshold_for_asset(config: Config, symbol: str) -> Tuple[float, str]:
    asset_class = _infer_asset_class(symbol)
    scoring_cfg = getattr(config, "scoring", None)
    fallback = float(getattr(scoring_cfg, "base_threshold", 6.0))
    if asset_class == "macro":
        return float(getattr(scoring_cfg, "base_threshold_macro", fallback)), asset_class
    return float(getattr(scoring_cfg, "base_threshold_crypto", fallback)), asset_class


def _side_to_structure_direction(side: str) -> str:
    return "bullish" if str(side or "").upper() == "LONG" else "bearish"


def _zone_aligned_with_side(zone_type: str, side: str) -> bool:
    z = str(zone_type or "").strip().lower()
    s = str(side or "").strip().upper()
    return (s == "LONG" and z == "support") or (s == "SHORT" and z == "resistance")


def _chart_price_from_ticker(ticker: Dict[str, Any]) -> float:
    """
    MT5-like chart reference:
    prefer Bid (chart candles), then Last, then Ask.
    """
    if not isinstance(ticker, dict):
        return 0.0
    for key in ("bid", "last", "ask"):
        try:
            value = float(ticker.get(key))
            if value > 0:
                return value
        except Exception:
            continue
    return 0.0


@dataclass
class TimingScore:
    """Container for all timing analysis scores."""
    gann_analysis: Optional[GannAnalysis] = None
    square9_analysis: Optional[Square9Analysis] = None
    cycle_analysis: Optional[CycleAnalysis] = None
    lunar_analysis: Optional[LunarAnalysis] = None
    fomc_analysis: Optional[FOMCAnalysis] = None
    cpi_analysis: Optional[CPIAnalysis] = None
    nfp_analysis: Optional[NFPAnalysis] = None
    powell_analysis: Optional[PowellAnalysis] = None
    fomc_minutes_analysis: Optional[FOMCMinutesAnalysis] = None
    sentiment_analysis: Optional[SentimentAnalysis] = None
    social_sentiment_analysis: Optional[SocialSentimentAnalysis] = None
    onchain_analysis: Optional[OnChainAnalysis] = None
    fib_analysis: Optional[FibAnalysis] = None
    llm_news_score: float = 0.0
    llm_news_sources: List[str] = field(default_factory=list)
    llm_news_categories: List[str] = field(default_factory=list)
    llm_news_headlines_count: int = 0
    
    @property
    def total_score(self) -> float:
        """Calculate total timing score."""
        score = 0.0
        if self.gann_analysis:
            score += self.gann_analysis.angle_confluence_score
        if self.square9_analysis:
            score += self.square9_analysis.square9_score
        if self.cycle_analysis:
            score += self.cycle_analysis.cycle_score
        if self.lunar_analysis:
            score += self.lunar_analysis.lunar_score
        if self.fomc_analysis:
            score += self.fomc_analysis.fomc_score  # Can be negative
        if self.cpi_analysis:
            score += self.cpi_analysis.cpi_score  # Can be negative
        if self.nfp_analysis:
            score += self.nfp_analysis.nfp_score  # Can be negative
        if self.powell_analysis:
            score += self.powell_analysis.powell_score  # Can be negative
        if self.fomc_minutes_analysis:
            score += self.fomc_minutes_analysis.minutes_score  # Can be negative
        if self.sentiment_analysis:
            score += self.sentiment_analysis.sentiment_score  # Can be negative
        if self.social_sentiment_analysis:
            score += self.social_sentiment_analysis.social_score  # Can be negative
        if self.onchain_analysis:
            score += self.onchain_analysis.score  # can be negative
        if self.fib_analysis and self.fib_analysis.nearest_level:
            # Add up to 2 points based on proximity
            if self.fib_analysis.distance_pct < 0.2:
                score += 2.0
            elif self.fib_analysis.distance_pct < 0.5:
                score += 1.0
        score += self.llm_news_score
        return score


@dataclass
class SignalResult:
    """Complete signal analysis result."""
    symbol: str
    timestamp: datetime
    side: Optional[str]  # 'LONG', 'SHORT', 'EXIT', or None
    
    # Scoring
    technical_score: int
    timing_score: float
    total_score: float
    threshold: int
    
    # Analysis details
    trend: str
    market_regime: str
    current_price: float
    reasons: List[str] = field(default_factory=list)
    timing_reasons: List[str] = field(default_factory=list)
    
    # Risk levels (if signal generated)
    risk_levels: Optional[RiskLevels] = None
    
    # Detailed analysis
    zone_info: Optional[Dict] = None
    pattern_info: Optional[Dict] = None
    timing_info: Optional[TimingScore] = None
    
    # Status
    is_valid: bool = False
    blocked_reason: Optional[str] = None

    # VPA Analysis (separate from technical score)
    vpa_score: float = 0.0
    vpa_info: Optional[Dict] = None

    # Optional expert advisor output (OpenAI)
    expert_opinion: Optional[Dict] = None

    # Optional manager advisor output (HTF/LTF structure opinion)
    manager_opinion: Optional[Dict] = None

    # Optional adaptive-learning output (for Telegram section/reporting)
    learning_opinion: Optional[Dict] = None

    # Optional strict shadow policy output (comparison only)
    quality_first_opinion: Optional[Dict] = None

    # Optional "Schools" section (advisory only, does NOT affect execution)
    schools_opinion: Optional[Dict] = None

    # Order flow data (Open Interest, Funding Rate, Liquidation markers)
    order_flow: Optional[Dict] = None

    # Machine Learning prediction probability
    ml_prediction: Optional[Dict] = None


def _clamp_confidence(value: float) -> int:
    """Clamp confidence to [0, 100] as integer."""
    try:
        num = float(value)
    except Exception:
        num = 50.0
    return int(max(0.0, min(100.0, round(num))))


def _school_card(key: str, name_ar: str, decision: str, confidence: float, reason: str) -> Dict[str, Any]:
    side = str(decision or "WAIT").upper()
    if side not in {"LONG", "SHORT", "WAIT"}:
        side = "WAIT"
    return {
        "key": key,
        "name_ar": name_ar,
        "decision": side,
        "confidence": _clamp_confidence(confidence),
        "reason": str(reason or "").strip(),
    }


def _pattern_name_ar(name: str) -> str:
    """Human-friendly Arabic label for candlestick pattern names."""
    key = str(name or "").strip().lower()
    labels = {
        "bullish_engulfing": "ابتلاع شرائي",
        "bearish_engulfing": "ابتلاع بيعي",
        "hammer": "شمعة المطرقة",
        "shooting_star": "الشهاب",
        "pin_bar": "بن بار",
        "morning_star": "نجمة الصباح",
        "evening_star": "نجمة المساء",
        "three_white_soldiers": "ثلاثة جنود بيض",
        "three_black_crows": "ثلاثة غربان سود",
        "doji": "دوجي",
        "dragonfly_doji": "دوجي اليعسوب",
        "gravestone_doji": "دوجي شاهد القبر",
        "bullish_harami": "هارامي شرائي",
        "bearish_harami": "هارامي بيعي",
        "piercing_line": "خط الاختراق الشرائي",
        "dark_cloud_cover": "غطاء السحابة الداكنة",
    }
    if key in labels:
        return labels[key]
    return key.replace("_", " ") if key else "نموذج شموعي"


def _build_schools_opinion(
    trend: str,
    potential_side: str,
    df_trend: pd.DataFrame,
    zone_result,
    pattern: Optional[CandlePattern],
    wave_result: Dict[str, Any],
    trendline_result: Optional[Dict[str, Any]],
    structure_result: Optional[Dict[str, Any]],
    vpa_info: Optional[Dict[str, Any]],
    timing_info: Optional[TimingScore],
) -> Dict[str, Any]:
    """
    Build advisory-only "Schools" snapshot.
    This output is informational and must not change signal execution.
    """
    cards: List[Dict[str, Any]] = []

    # 1) Dow
    ema50 = None
    ema200 = None
    adx_now = 0.0
    try:
        if "ema_50" in df_trend.columns:
            raw = df_trend["ema_50"].iloc[-1]
            ema50 = float(raw) if not pd.isna(raw) else None
        if "ema_200" in df_trend.columns:
            raw = df_trend["ema_200"].iloc[-1]
            ema200 = float(raw) if not pd.isna(raw) else None
        if "adx" in df_trend.columns:
            raw = df_trend["adx"].iloc[-1]
            adx_now = float(raw) if not pd.isna(raw) else 0.0
    except Exception:
        ema50 = ema50 if isinstance(ema50, (int, float)) else None
        ema200 = ema200 if isinstance(ema200, (int, float)) else None
        adx_now = 0.0

    dow_decision = "WAIT"
    dow_reason = "لا يوجد اتجاه واضح حسب داو"
    dow_conf = 52
    if trend == "up" and ema50 is not None and ema200 is not None and ema50 > ema200:
        dow_decision = "LONG"
        dow_conf = 66 + min(18, max(0, int((adx_now - 18.0) * 1.2)))
        dow_reason = f"اتجاه صاعد + EMA50 فوق EMA200 (ADX={adx_now:.1f})"
    elif trend == "down" and ema50 is not None and ema200 is not None and ema50 < ema200:
        dow_decision = "SHORT"
        dow_conf = 66 + min(18, max(0, int((adx_now - 18.0) * 1.2)))
        dow_reason = f"اتجاه هابط + EMA50 تحت EMA200 (ADX={adx_now:.1f})"
    cards.append(_school_card("dow", "داو", dow_decision, dow_conf, dow_reason))

    # 2) Classical (support/resistance + pattern + structure + trendline)
    classical_long = 0.0
    classical_short = 0.0
    classical_tags_long: List[str] = []
    classical_tags_short: List[str] = []

    zt = str(getattr(zone_result, "zone_type", "") or "").strip().lower()
    if zt == "support":
        classical_long += 1.1
        classical_tags_long.append("قرب دعم")
    elif zt == "resistance":
        classical_short += 1.1
        classical_tags_short.append("قرب مقاومة")

    if pattern is not None:
        pdir = str(getattr(pattern, "direction", "")).lower()
        pstrength = int(getattr(pattern, "strength", 1) or 1)
        pname = _pattern_name_ar(str(getattr(pattern, "name", "") or ""))
        bonus = 0.5 + (0.25 * max(1, min(3, pstrength)))
        if pdir == "bullish":
            classical_long += bonus
            classical_tags_long.append(f"{pname} x{pstrength}")
        elif pdir == "bearish":
            classical_short += bonus
            classical_tags_short.append(f"{pname} x{pstrength}")

    if trendline_result:
        tl_type = str(trendline_result.get("type", "")).lower()
        if tl_type == "bullish_break":
            classical_long += 1.0
            classical_tags_long.append("كسر ترند صاعد")
        elif tl_type == "bearish_break":
            classical_short += 1.0
            classical_tags_short.append("كسر ترند هابط")

    if structure_result:
        sdir = str(structure_result.get("direction", "")).lower()
        sstrength = int(structure_result.get("strength", 1) or 1)
        sbonus = 0.5 + (0.2 * max(1, min(3, sstrength)))
        if sdir == "bullish":
            classical_long += sbonus
            classical_tags_long.append("هيكلية HH/HL")
        elif sdir == "bearish":
            classical_short += sbonus
            classical_tags_short.append("هيكلية LH/LL")

    if trend == "up":
        classical_long += 0.5
    elif trend == "down":
        classical_short += 0.5

    classical_diff = classical_long - classical_short
    classical_decision = "WAIT"
    classical_conf = 50 + int(min(20.0, max(classical_long, classical_short) * 5.0))
    classical_reason = "إشارات كلاسيكية مختلطة"
    if classical_diff >= 0.7:
        classical_decision = "LONG"
        classical_conf = 58 + int(min(30.0, classical_diff * 14.0))
        classical_reason = " + ".join(classical_tags_long[:3]) or "كفة الشراء أقوى"
    elif classical_diff <= -0.7:
        classical_decision = "SHORT"
        classical_conf = 58 + int(min(30.0, abs(classical_diff) * 14.0))
        classical_reason = " + ".join(classical_tags_short[:3]) or "كفة البيع أقوى"
    cards.append(_school_card("classical", "الكلاسيكية", classical_decision, classical_conf, classical_reason))

    # 3) Elliott (wave trigger)
    wave_triggered = bool((wave_result or {}).get("triggered", False))
    wave_score = float((wave_result or {}).get("score", 0.0) or 0.0)
    wave_reasons = list((wave_result or {}).get("reasons", []) or [])
    if wave_triggered and potential_side in {"LONG", "SHORT"}:
        elliott_decision = potential_side
        elliott_conf = 62 + min(26, int((wave_score * 10.0) + (len(wave_reasons) * 4.0)))
        elliott_reason = "موجة دافعة مكتملة: " + ", ".join(str(x) for x in wave_reasons[:2])
    else:
        cond_count = len((wave_result or {}).get("conditions", {}) or {})
        elliott_decision = "WAIT"
        elliott_conf = 48 + min(16, cond_count * 3)
        elliott_reason = "شروط الموجة 3 غير مكتملة"
    cards.append(_school_card("elliott", "إليوت الموجي", elliott_decision, elliott_conf, elliott_reason))

    # 4) Wyckoff (accumulation/distribution + structure context)
    wy_long = 0.0
    wy_short = 0.0
    wy_tags_long: List[str] = []
    wy_tags_short: List[str] = []
    vpa_sig = str((vpa_info or {}).get("vpa_signal", "neutral")).lower()

    if vpa_sig in {"accumulation", "confirmed_breakout_up", "double_bottom"}:
        wy_long += 1.2
        wy_tags_long.append("قراءة تجميع")
    elif vpa_sig in {"distribution", "confirmed_breakout_down", "double_top"}:
        wy_short += 1.2
        wy_tags_short.append("قراءة تصريف")

    if structure_result:
        sdir = str(structure_result.get("direction", "")).lower()
        if sdir == "bullish":
            wy_long += 0.9
            wy_tags_long.append("هيكلية داعمة للصعود")
        elif sdir == "bearish":
            wy_short += 0.9
            wy_tags_short.append("هيكلية داعمة للهبوط")

    if zt == "support":
        wy_long += 0.5
    elif zt == "resistance":
        wy_short += 0.5

    wy_diff = wy_long - wy_short
    wy_decision = "WAIT"
    wy_conf = 50 + int(min(16.0, max(wy_long, wy_short) * 7.0))
    wy_reason = "لا يوجد تجميع/تصريف واضح"
    if wy_diff >= 0.8:
        wy_decision = "LONG"
        wy_conf = 57 + int(min(28.0, wy_diff * 16.0))
        wy_reason = " + ".join(wy_tags_long[:3]) or "أولوية للتجميع"
    elif wy_diff <= -0.8:
        wy_decision = "SHORT"
        wy_conf = 57 + int(min(28.0, abs(wy_diff) * 16.0))
        wy_reason = " + ".join(wy_tags_short[:3]) or "أولوية للتصريف"
    cards.append(_school_card("wyckoff", "وايكوف (تجميع/تصريف)", wy_decision, wy_conf, wy_reason))

    # 5) Volume / VPA
    vpa_score = float((vpa_info or {}).get("score", 0.0) or 0.0)
    vpa_alignment = str((vpa_info or {}).get("signal_alignment", "neutral") or "neutral").lower()
    vol_decision = "WAIT"
    vol_conf = 50 + int(min(20.0, abs(vpa_score) * 8.0))
    vol_reason = "الحجم محايد"
    if vpa_score >= 1.5:
        vol_decision = "LONG"
        vol_conf = 56 + int(min(34.0, abs(vpa_score) * 12.0))
        vol_reason = f"VPA داعم للشراء ({vpa_sig or 'bullish'})"
    elif vpa_score <= -1.5:
        vol_decision = "SHORT"
        vol_conf = 56 + int(min(34.0, abs(vpa_score) * 12.0))
        vol_reason = f"VPA داعم للبيع ({vpa_sig or 'bearish'})"
    if vpa_alignment in {"opposing", "weak_opposing"}:
        vol_reason += " مع تعارض جزئي"
        vol_conf = max(45, vol_conf - 8)
    cards.append(_school_card("volume", "التحليل الحجمي", vol_decision, vol_conf, vol_reason))

    # 6) Gann (price-time relation)
    gann_decision = "WAIT"
    gann_conf = 50
    gann_reason = "لا يوجد توافق زمني/سعري واضح"
    if timing_info and getattr(timing_info, "gann_analysis", None):
        ga = timing_info.gann_analysis
        gann_score = float(getattr(ga, "angle_confluence_score", 0.0) or 0.0)
        relation = str(getattr(ga, "price_relation", "unknown") or "unknown").lower()
        if relation == "above_angles" and trend == "up":
            gann_decision = "LONG"
            gann_conf = 58 + int(min(30.0, gann_score * 18.0))
            gann_reason = f"السعر فوق زوايا جان (score={gann_score:.1f})"
        elif relation == "below_angles" and trend == "down":
            gann_decision = "SHORT"
            gann_conf = 58 + int(min(30.0, gann_score * 18.0))
            gann_reason = f"السعر تحت زوايا جان (score={gann_score:.1f})"
        elif relation == "at_angle" and potential_side in {"LONG", "SHORT"} and gann_score >= 1.0:
            gann_decision = potential_side
            gann_conf = 55 + int(min(28.0, gann_score * 16.0))
            gann_reason = f"السعر عند زاوية محورية (score={gann_score:.1f})"
        else:
            gann_conf = 50 + int(min(20.0, gann_score * 10.0))
            gann_reason = f"قراءة جان محايدة ({relation})"
    cards.append(_school_card("gann", "جان الزمني", gann_decision, gann_conf, gann_reason))

    long_votes = sum(1 for c in cards if c.get("decision") == "LONG")
    short_votes = sum(1 for c in cards if c.get("decision") == "SHORT")
    total = max(1, len(cards))
    agreement_votes = max(long_votes, short_votes)
    agreement_pct = (agreement_votes / total) * 100.0

    consensus = "WAIT"
    if long_votes > short_votes and long_votes >= 2:
        consensus = "LONG"
    elif short_votes > long_votes and short_votes >= 2:
        consensus = "SHORT"

    aligned_cards = [c for c in cards if c.get("decision") == consensus] if consensus != "WAIT" else []
    if aligned_cards:
        avg_conf = sum(float(c.get("confidence", 50)) for c in aligned_cards) / max(1, len(aligned_cards))
        consensus_conf = 48 + int(min(45.0, (agreement_pct * 0.35) + (avg_conf * 0.30)))
    else:
        consensus_conf = 48 + int(min(20.0, agreement_pct * 0.22))

    return {
        "name_ar": "المدارس",
        "mode": "advisory_only",
        "read_only": True,
        "consensus": {
            "decision": consensus,
            "confidence": _clamp_confidence(consensus_conf),
            "agreement_pct": round(agreement_pct, 1),
            "long_votes": int(long_votes),
            "short_votes": int(short_votes),
            "total_schools": int(total),
        },
        "schools": cards,
    }


# =============================================================================
# Scoring Logic
# =============================================================================

def calculate_technical_score(
    trend: str,
    side: str,
    zone_result,
    pattern: Optional[CandlePattern],
    volume_spike: bool,
    wave_result: Dict,
    rsi_divergence: Optional[str],
    adx_value: float = 0.0,
    di_plus: float = 0.0,
    di_minus: float = 0.0,
) -> Tuple[int, List[str]]:
    """
    Calculate technical score for a signal.

    Scoring:
    - Trend match: +2
    - Price in S/R zone: +2
    - Candle pattern confirmation: +2
    - Volume spike: +2
    - Wave trigger: +2
    - RSI divergence against trade: -2
    - ADX weak market: -2 (below 20)
    - ADX strong trend confirmation: +1 (above 35 + DI direction)

    Returns:
        Tuple of (score, reasons_list)
    """
    score = 0
    reasons = []

    # ADX Filter
    if adx_value > 0:
        if adx_value < 20:
            score -= 2
            reasons.append(f"✗ Ranging market (ADX: {adx_value:.1f})")
        elif adx_value > 35:
            if (side == 'LONG' and di_plus > di_minus) or (side == 'SHORT' and di_minus > di_plus):
                score += 1
                reasons.append(f"✓ Strong trend (ADX: {adx_value:.1f})")

    # Trend match (+2)
    if (side == 'LONG' and trend == 'up') or (side == 'SHORT' and trend == 'down'):
        score += 2
        reasons.append(f"✓ Trend: {trend}")
    elif trend == 'neutral':
        score += 1
        reasons.append("○ Trend: neutral")

    # Zone proximity (+2)
    if zone_result.in_zone:
        score += 2
        reasons.append(f"✓ At {zone_result.zone_type} zone")
    elif zone_result.distance_pct < 0.5:
        score += 1
        reasons.append(f"○ Near {zone_result.zone_type} zone ({zone_result.distance_pct:.2f}%)")

    # Candle pattern (+2)
    if pattern:
        direction = 'bullish' if side == 'LONG' else 'bearish'
        if pattern.direction == direction:
            score += 2
            reasons.append(f"✓ {pattern.name.replace('_', ' ').title()}")

    # Volume spike (+2)
    if volume_spike:
        score += 2
        reasons.append("✓ Volume spike")

    # Wave trigger (+2)
    if wave_result.get('triggered'):
        score += 2
        reasons.append("✓ Wave 3 setup")

    # RSI divergence penalty (-2)
    if rsi_divergence:
        if (side == 'LONG' and rsi_divergence == 'bearish') or \
           (side == 'SHORT' and rsi_divergence == 'bullish'):
            score -= 2
            reasons.append(f"✗ RSI {rsi_divergence} divergence")

    return score, reasons


def calculate_vpa_score(
    df_entry: pd.DataFrame,
    side: str,
    current_price: float,
) -> Tuple[float, Dict]:
    """
    Standalone Volume Price Analysis (VPA) score.
    Fully independent — does NOT affect the main technical score or signal threshold.
    Displayed as a separate section in Telegram notifications.

    Scoring (max +4, min -4):
    - VPA signal confirms direction (breakout, accumulation, climax_sell, etc.): +2
    - VPA signal opposes direction (distribution, climax_buy, etc.):             -2
    - Weak-move caution against direction (weak_move_down for LONG, etc.):       -1
    - VWAP confirms direction:                                                   +1
    - VWAP opposes direction:                                                    -1
    - CMF confirms direction (>0.05 for LONG / <-0.05 for SHORT):               +1
    - CMF opposes direction:                                                     -1

    Returns:
        Tuple of (vpa_score, vpa_info_dict)
    """
    vpa_info: Dict = {
        'vpa_signal': 'neutral',
        'vpa_description': 'No notable VPA signal',
        'signal_alignment': 'neutral',
        'raw_confirms_side': False,
        'raw_opposes_side': False,
        'vwap': None,
        'cmf': None,
        'obv_trend': None,
        'score': 0.0,
        'reasons': [],
    }

    score = 0.0
    reasons: List[str] = []

    # --- VPA candle / OBV analysis ---
    vpa_result = analyze_vpa(df_entry)
    vpa_info['vpa_signal'] = vpa_result['signal']
    vpa_info['vpa_description'] = vpa_result['description']

    confirming = {
        'LONG':  ['confirmed_breakout_up', 'accumulation', 'effort_no_result_down', 'climax_sell'],
        'SHORT': ['confirmed_breakout_down', 'distribution', 'effort_no_result_up', 'climax_buy'],
    }
    opposing = {
        'LONG':  ['confirmed_breakout_down', 'distribution', 'effort_no_result_up', 'climax_buy'],
        'SHORT': ['confirmed_breakout_up', 'accumulation', 'effort_no_result_down', 'climax_sell'],
    }
    weak_opp = {
        'LONG':  ['weak_move_down'],
        'SHORT': ['weak_move_up'],
    }

    sig = vpa_result['signal']
    signal_alignment = 'neutral'
    raw_confirms_side = False
    raw_opposes_side = False
    if sig != 'neutral':
        if sig in confirming.get(side, []):
            score += 2
            signal_alignment = 'confirming'
            raw_confirms_side = True
            reasons.append(f"✓ {sig.replace('_', ' ')}")
        elif sig in opposing.get(side, []):
            score -= 2
            signal_alignment = 'opposing'
            raw_opposes_side = True
            reasons.append(f"✗ {sig.replace('_', ' ')}")
        elif sig in weak_opp.get(side, []):
            score -= 1
            signal_alignment = 'weak_opposing'
            raw_opposes_side = True
            reasons.append(f"⚠ {sig.replace('_', ' ')}")
        else:
            if (side == 'LONG' and vpa_result.get('bullish_bias')) or \
               (side == 'SHORT' and vpa_result.get('bearish_bias')):
                score += 1
                signal_alignment = 'confirming'
                raw_confirms_side = True
                reasons.append(f"○ {sig.replace('_', ' ')}")

    # --- VWAP bias ---
    vwap_val = df_entry['vwap'].iloc[-1] if 'vwap' in df_entry.columns else None
    if vwap_val is not None and not pd.isna(vwap_val):
        vpa_info['vwap'] = float(vwap_val)
        if side == 'LONG' and current_price > vwap_val:
            score += 1
            reasons.append(f"✓ Price above VWAP ({vwap_val:.4f})")
        elif side == 'SHORT' and current_price < vwap_val:
            score += 1
            reasons.append(f"✓ Price below VWAP ({vwap_val:.4f})")
        elif side == 'LONG':
            score -= 1
            reasons.append(f"✗ Price below VWAP ({vwap_val:.4f})")
        elif side == 'SHORT':
            score -= 1
            reasons.append(f"✗ Price above VWAP ({vwap_val:.4f})")

    # --- CMF confirmation ---
    cmf_val = df_entry['cmf'].iloc[-1] if 'cmf' in df_entry.columns else None
    if cmf_val is not None and not pd.isna(cmf_val):
        vpa_info['cmf'] = float(cmf_val)
        if side == 'LONG' and cmf_val > 0.05:
            score += 1
            reasons.append(f"✓ CMF buying pressure ({cmf_val:.2f})")
        elif side == 'SHORT' and cmf_val < -0.05:
            score += 1
            reasons.append(f"✓ CMF selling pressure ({cmf_val:.2f})")
        elif side == 'LONG' and cmf_val < -0.05:
            score -= 1
            reasons.append(f"✗ CMF selling pressure ({cmf_val:.2f})")
        elif side == 'SHORT' and cmf_val > 0.05:
            score -= 1
            reasons.append(f"✗ CMF buying pressure ({cmf_val:.2f})")

    vpa_info['score'] = score
    vpa_info['signal_alignment'] = signal_alignment
    vpa_info['raw_confirms_side'] = raw_confirms_side
    vpa_info['raw_opposes_side'] = raw_opposes_side
    vpa_info['reasons'] = reasons
    return score, vpa_info


def classify_market_regime(
    df_trend: pd.DataFrame,
    config: Config
) -> Tuple[str, Dict[str, float]]:
    """
    Classify market regime from trend structure + volatility.

    Returns:
        (regime, metrics)
        regime in {'uptrend', 'downtrend', 'sideways', 'high_volatility'}
    """
    if df_trend.empty:
        return "sideways", {"adx": 0.0, "atr_pct": 0.0, "ema200_slope_24": 0.0}

    last = df_trend.iloc[-1]
    close = float(last.get("close", 0.0) or 0.0)
    ema50 = float(last.get("ema_50", close) or close)
    ema200 = float(last.get("ema_200", close) or close)
    adx_val = float(last.get("adx", 0.0) or 0.0)
    atr_val = float(last.get("atr_14", 0.0) or 0.0)
    atr_pct = (atr_val / close * 100.0) if close > 0 else 0.0

    if len(df_trend) >= 25 and "ema_200" in df_trend.columns:
        ema200_prev = float(df_trend["ema_200"].iloc[-25] or ema200)
        ema200_slope = ((ema200 - ema200_prev) / ema200_prev) if ema200_prev else 0.0
    else:
        ema200_slope = 0.0

    rg = getattr(config, "regime", None)
    if not rg or not getattr(rg, "enabled", True):
        return "sideways", {"adx": adx_val, "atr_pct": atr_pct, "ema200_slope_24": ema200_slope}

    slope_min = float(getattr(rg, "ema200_slope_abs_min", 0.0015))
    adx_trend_min = float(getattr(rg, "adx_trend_min", 22.0))
    high_vol_atr_pct = float(getattr(rg, "high_vol_atr_pct", 1.2))

    if atr_pct >= high_vol_atr_pct and adx_val < adx_trend_min:
        regime = "high_volatility"
    elif ema200_slope >= slope_min and close >= ema200 and ema50 >= ema200:
        regime = "uptrend"
    elif ema200_slope <= -slope_min and close <= ema200 and ema50 <= ema200:
        regime = "downtrend"
    elif adx_val >= adx_trend_min:
        regime = "uptrend" if close >= ema200 else "downtrend"
    else:
        regime = "sideways"

    return regime, {"adx": adx_val, "atr_pct": atr_pct, "ema200_slope_24": ema200_slope}


def get_scaled_quick_tp_pct(config: Config, market_regime: str) -> float:
    """Scale quick TP target percent by detected regime."""
    quick_tp = float(getattr(config.risk, "quick_tp_pct", 0.35))
    rg = getattr(config, "regime", None)
    if not rg or not getattr(rg, "enabled", True):
        return max(0.05, quick_tp)

    if market_regime == "sideways":
        quick_tp *= float(getattr(rg, "quick_tp_scale_sideways", 0.8))
    elif market_regime == "high_volatility":
        quick_tp *= float(getattr(rg, "quick_tp_scale_high_vol", 0.75))
    return max(0.05, quick_tp)


def _timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe notation (e.g. 1m/15m/1h/4h/1d) to minutes."""
    tf = str(timeframe or "").strip().lower()
    if not tf:
        return 15
    try:
        unit = tf[-1]
        value = int(tf[:-1])
    except Exception:
        return 15
    if value <= 0:
        return 15
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    return 15


def _get_timeframe_scale(scale_map: Dict[str, float], timeframe: str, default: float = 1.0) -> float:
    """Resolve timeframe scale from config map using exact or nearest timeframe key."""
    if not isinstance(scale_map, dict) or not scale_map:
        return float(default)

    tf = str(timeframe or "").strip().lower()
    if tf in scale_map:
        try:
            return float(scale_map[tf])
        except Exception:
            return float(default)

    target_min = _timeframe_to_minutes(tf)
    best_key = None
    best_gap = None
    for key, val in scale_map.items():
        key_tf = str(key or "").strip().lower()
        if not key_tf:
            continue
        gap = abs(_timeframe_to_minutes(key_tf) - target_min)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_key = key
    if best_key is None:
        return float(default)
    try:
        return float(scale_map.get(best_key, default))
    except Exception:
        return float(default)


def get_dynamic_risk_parameters(config: Config, market_regime: str) -> Dict[str, float]:
    """
    Build dynamic risk parameters using regime + entry timeframe profile.
    """
    risk_cfg = getattr(config, "risk", None)
    if not risk_cfg:
        return {
            "quick_tp_pct": 0.35,
            "quick_tp_min_pct": 0.12,
            "rr_tp2": 2.0,
            "atr_stop_mult": 1.8,
            "atr_buffer_mult": 0.30,
            "tp2_atr_mult": 4.0,
        }

    quick_tp_pct = get_scaled_quick_tp_pct(config, market_regime)
    quick_tf_scale = _get_timeframe_scale(
        getattr(risk_cfg, "timeframe_quick_tp_scale", {}) or {},
        getattr(config, "entry_tf", "15m"),
        default=1.0,
    )
    tp2_tf_scale = _get_timeframe_scale(
        getattr(risk_cfg, "timeframe_tp2_rr_scale", {}) or {},
        getattr(config, "entry_tf", "15m"),
        default=1.0,
    )

    quick_tp_pct *= max(0.4, quick_tf_scale)
    quick_tp_min_pct = max(
        0.03,
        float(getattr(risk_cfg, "quick_tp_min_pct", 0.12)),
    )

    rr_tp2 = float(getattr(risk_cfg, "rr_tp2", 2.0)) * max(0.6, tp2_tf_scale)
    rr_tp2 = max(1.0, rr_tp2)

    return {
        "quick_tp_pct": max(quick_tp_min_pct, quick_tp_pct),
        "quick_tp_min_pct": quick_tp_min_pct,
        "rr_tp2": rr_tp2,
        "atr_stop_mult": float(getattr(risk_cfg, "atr_stop_mult", 1.8)),
        "atr_buffer_mult": float(getattr(risk_cfg, "atr_buffer_mult", 0.30)),
        "tp2_atr_mult": float(getattr(risk_cfg, "tp2_atr_mult", 4.0)),
    }


def calculate_timing_score(
    df_trend: pd.DataFrame,
    df_entry: pd.DataFrame,
    symbol: str,
    trend: str,
    side: str,
    current_price: float,
    config: Config
) -> Tuple[TimingScore, List[str]]:
    """
    Calculate timing score from all timing components.
    
    Args:
        df_trend: Trend timeframe DataFrame with indicators
        df_entry: Entry timeframe DataFrame with indicators
        symbol: Trading symbol (used for asset-aware news analysis)
        trend: Current trend
        current_price: Current price
        config: Configuration object
        
    Returns:
        Tuple of (TimingScore, timing_reasons)
    """
    timing = TimingScore()
    reasons = []
    
    time_cfg = config.time_analysis
    
    if not time_cfg.enabled:
        return timing, reasons
    
    # Gann Angles
    if time_cfg.gann_angles and 'atr_14' in df_trend.columns:
        atr_value = df_trend['atr_14'].iloc[-1]
        timing.gann_analysis = calculate_gann_angles(df_trend, trend, atr_value)
        if timing.gann_analysis.angle_confluence_score > 0:
            reasons.append(f"Gann: {timing.gann_analysis.price_relation} (score: {timing.gann_analysis.angle_confluence_score:.1f})")
    
    # Square of 9
    if time_cfg.square9:
        timing.square9_analysis = analyze_square9(current_price, time_cfg.zone_proximity_pct)
        if timing.square9_analysis.square9_score > 0:
            nl = timing.square9_analysis.nearest_level
            if nl:
                reasons.append(f"Sq9: {nl.level:.2f} ({nl.distance_pct:.2f}%, score: {timing.square9_analysis.square9_score:.1f})")
    
    # 52 Cycle
    if time_cfg.cycle_52:
        timing.cycle_analysis = analyze_52_cycle(df_entry)
        if timing.cycle_analysis.in_cycle_window:
            reasons.append(f"Cycle52: IN WINDOW (pos: {timing.cycle_analysis.cycle_position})")
    
    # Lunar
    if time_cfg.lunar:
        timing.lunar_analysis = analyze_lunar(time_cfg.lunar_window_hours)
        if timing.lunar_analysis.in_event_window:
            reasons.append(f"Lunar: {timing.lunar_analysis.phase} moon window")
    
    # FOMC
    if time_cfg.fomc_filter:
        timing.fomc_analysis = analyze_fomc(time_cfg.fomc_high_vol_days)
        if timing.fomc_analysis.in_high_vol_window:
            reasons.append(f"⚠️ FOMC in {timing.fomc_analysis.days_to_next} days")

    # CPI
    if time_cfg.cpi_filter:
        timing.cpi_analysis = analyze_cpi(time_cfg.cpi_high_vol_days)
        if timing.cpi_analysis.in_high_vol_window and timing.cpi_analysis.next_release:
            reasons.append(f"⚠️ CPI in {timing.cpi_analysis.hours_to_next:.1f}h")

    # NFP
    if time_cfg.nfp_filter:
        timing.nfp_analysis = analyze_nfp(time_cfg.nfp_high_vol_days)
        if timing.nfp_analysis.in_high_vol_window and timing.nfp_analysis.next_release:
            reasons.append(f"⚠️ NFP in {timing.nfp_analysis.hours_to_next:.1f}h")

    # Powell speeches
    if time_cfg.powell_filter:
        timing.powell_analysis = analyze_powell_speeches(time_cfg.powell_high_vol_hours)
        if timing.powell_analysis.in_high_vol_window and timing.powell_analysis.next_event:
            reasons.append(f"⚠️ Powell speech in {timing.powell_analysis.hours_to_next:.1f}h")

    # FOMC minutes
    if time_cfg.fomc_minutes_filter:
        timing.fomc_minutes_analysis = analyze_fomc_minutes(time_cfg.fomc_minutes_high_vol_days)
        if timing.fomc_minutes_analysis.in_high_vol_window and timing.fomc_minutes_analysis.next_release:
            reasons.append(f"⚠️ FOMC minutes in {timing.fomc_minutes_analysis.hours_to_next:.1f}h")

    # Crowd sentiment (Fear & Greed index)
    if getattr(time_cfg, 'sentiment_filter', False):
        timing.sentiment_analysis = analyze_market_sentiment(
            time_cfg.sentiment_extreme_fear,
            time_cfg.sentiment_extreme_greed
        )
        if timing.sentiment_analysis.value is not None:
            if timing.sentiment_analysis.in_extreme_zone:
                reasons.append(
                    f"⚠️ Sentiment EXTREME ({timing.sentiment_analysis.classification}: {timing.sentiment_analysis.value})"
                )
            elif timing.sentiment_analysis.sentiment_score < 0:
                reasons.append(
                    f"Sentiment caution ({timing.sentiment_analysis.classification}: {timing.sentiment_analysis.value})"
                )

    # Social sentiment (Reddit crowd)
    if getattr(time_cfg, 'social_sentiment_filter', False):
        timing.social_sentiment_analysis = analyze_social_sentiment(
            min_posts=time_cfg.social_sentiment_min_posts,
            caution_ratio=time_cfg.social_sentiment_caution_ratio,
            extreme_ratio=time_cfg.social_sentiment_extreme_ratio
        )
        sa = timing.social_sentiment_analysis
        if sa.posts_scanned > 0 and sa.social_score < 0:
            bull_pct = sa.bullish_ratio * 100
            bear_pct = sa.bearish_ratio * 100
            if sa.in_extreme_zone:
                reasons.append(f"⚠️ Reddit EXTREME {sa.dominant_side} (B {bull_pct:.0f}% / S {bear_pct:.0f}%)")
            else:
                reasons.append(f"Reddit caution {sa.dominant_side} (B {bull_pct:.0f}% / S {bear_pct:.0f}%)")

    # On-chain flow and activity (crypto only)
    if _infer_asset_class(symbol) == "crypto":
        oc_cfg = getattr(config, "onchain", None)
        if oc_cfg and getattr(oc_cfg, "enabled", False):
            timing.onchain_analysis = analyze_onchain(
                symbol=symbol,
                side=side,
                config=oc_cfg,
            )
            if timing.onchain_analysis.valid and timing.onchain_analysis.reason:
                reasons.append(timing.onchain_analysis.reason)

    # LLM automatic market news (headlines fetched automatically + OpenAI analysis)
    if config.openai.enabled and config.openai.api_key:
        llm_signal = get_market_news_signal(side=side, symbol=symbol, openai_config=config.openai)
        if llm_signal:
            timing.llm_news_score = llm_signal.score
            timing.llm_news_sources = llm_signal.sources
            timing.llm_news_categories = llm_signal.categories
            timing.llm_news_headlines_count = llm_signal.headlines_count
            reasons.append(llm_signal.reason)

    # Fibonacci
    if time_cfg.fibonacci:
        timing.fib_analysis = calculate_fib_levels(df_trend, trend)
        if timing.fib_analysis and timing.fib_analysis.nearest_level:
            nl = timing.fib_analysis.nearest_level
            if timing.fib_analysis.distance_pct < 0.5:
                reasons.append(f"Fib: {nl.ratio} {nl.type} @ {nl.price:.2f} ({timing.fib_analysis.distance_pct:.2f}%)")
    
    return timing, reasons


# =============================================================================
# Signal Generation
# =============================================================================

def analyze_symbol(
    symbol: str,
    df_trend: pd.DataFrame,
    df_entry: pd.DataFrame,
    df_sr: pd.DataFrame,
    ticker: Dict,
    config: Config,
    df_htf: Optional[pd.DataFrame] = None,
    order_flow_data: Optional[Dict] = None,
    ml_engine: Any = None,
    atr_calibration_scale: float = 1.0,
) -> SignalResult:
    """
    Perform complete signal analysis for a symbol.
    
    Args:
        symbol: Trading symbol
        df_trend: Trend timeframe DataFrame
        df_entry: Entry timeframe DataFrame
        df_sr: S/R timeframe DataFrame
        ticker: Current ticker data
        config: Configuration object
        
    Returns:
        SignalResult with complete analysis
    """
    now = datetime.now(timezone.utc)
    current_price = _chart_price_from_ticker(ticker)
    
    base_threshold_for_asset, asset_class = _base_threshold_for_asset(config, symbol)

    # Initialize result
    result = SignalResult(
        symbol=symbol,
        timestamp=now,
        side=None,
        technical_score=0,
        timing_score=0,
        total_score=0,
        threshold=base_threshold_for_asset,
        trend='neutral',
        market_regime='sideways',
        current_price=current_price
    )
    
    if df_trend.empty or df_entry.empty:
        result.blocked_reason = "Insufficient data"
        return result
    
    # Add indicators to all DataFrames
    df_trend = add_all_indicators(df_trend)
    df_entry = add_all_indicators(df_entry)
    df_sr = add_all_indicators(df_sr) if not df_sr.empty else df_sr

    # Prepare HTF context once (structure / POI / raid).
    df_htf_ind = None
    htf_trend = "neutral"
    htf_structure_result: Optional[Dict[str, Any]] = None
    htf_zone_result = None
    htf_resistance_levels: List[float] = []
    htf_support_levels: List[float] = []
    htf_sweeps = []
    if df_htf is not None and isinstance(df_htf, pd.DataFrame) and not df_htf.empty:
        try:
            df_htf_ind = add_all_indicators(df_htf.copy())
            htf_trend = get_trend(df_htf_ind)
            _qf_cfg = getattr(config, "quality_filter", None)
            _htf_structure_lookback = max(20, int(getattr(_qf_cfg, "htf_structure_lookback", 60)))
            _htf_raid_lookback = max(10, int(getattr(_qf_cfg, "htf_raid_lookback", 40)))

            htf_structure_result = detect_structure(df_htf_ind, lookback=_htf_structure_lookback)
            htf_zones = build_zones(df_htf_ind, window=4, zone_width_pct=0.0035)
            if htf_zones:
                htf_zone_result = is_price_in_zone(current_price, htf_zones)
                htf_resistance_levels = sorted(
                    [float(z.level) for z in htf_zones if z.type == 'resistance' and z.level >= current_price]
                )[:3]
                htf_support_levels = sorted(
                    [float(z.level) for z in htf_zones if z.type == 'support' and z.level <= current_price],
                    reverse=True,
                )[:3]
            htf_sweeps = detect_liquidity_sweeps(df_htf_ind, lookback=_htf_raid_lookback)
        except Exception as htf_exc:
            logger.debug(f"HTF context prep error for {symbol}: {htf_exc}")
    
    # Determine trend from trend timeframe
    trend = get_trend(df_trend)
    result.trend = trend
    market_regime, regime_metrics = classify_market_regime(df_trend, config)
    result.market_regime = market_regime
    
    # Build S/R zones
    zones = build_zones(df_sr, window=3, zone_width_pct=0.003) if not df_sr.empty else []
    zone_result = is_price_in_zone(current_price, zones)
    resistance_levels = sorted(
        [float(z.level) for z in zones if z.type == 'resistance' and z.level >= current_price]
    )[:3]
    support_levels = sorted(
        [float(z.level) for z in zones if z.type == 'support' and z.level <= current_price],
        reverse=True
    )[:3]
    
    result.zone_info = {
        'in_zone': zone_result.in_zone,
        'zone_type': zone_result.zone_type,
        'distance_pct': zone_result.distance_pct,
        'adx': df_trend['adx'].iloc[-1] if 'adx' in df_trend.columns else 0.0,
        'rsi_trend': df_trend['rsi_14'].iloc[-1] if 'rsi_14' in df_trend.columns else 50.0,
        'ichimoku_signal': get_ichimoku_signal(df_trend),
        'market_regime': market_regime,
        'atr_pct': regime_metrics.get('atr_pct', 0.0),
        'ema200_slope_24': regime_metrics.get('ema200_slope_24', 0.0),
        'resistance_levels': resistance_levels,
        'support_levels': support_levels,
        'htf_trend': htf_trend,
        'htf_structure': (htf_structure_result or {}).get('type', '') if htf_structure_result else '',
        'htf_in_zone': bool(getattr(htf_zone_result, 'in_zone', False)) if htf_zone_result is not None else False,
        'htf_zone_type': getattr(htf_zone_result, 'zone_type', '') if htf_zone_result is not None else '',
        'htf_distance_pct': float(getattr(htf_zone_result, 'distance_pct', 0.0) or 0.0) if htf_zone_result is not None else 0.0,
        'htf_resistance_levels': htf_resistance_levels,
        'htf_support_levels': htf_support_levels,
    }
    
    # Get volume spike status
    volume_spike = df_entry['volume_spike'].iloc[-1] if 'volume_spike' in df_entry.columns else False
    
    # Check RSI divergence
    rsi_div = check_rsi_divergence(df_entry)
    
    # Determine potential signal direction
    if trend == 'up':
        potential_side = 'LONG'
    elif trend == 'down':
        potential_side = 'SHORT'
    else:
        # Neutral trend — use broader EMA structure for bias
        ema50_t = df_trend['ema_50'].iloc[-1] if 'ema_50' in df_trend.columns else None
        ema200_t = df_trend['ema_200'].iloc[-1] if 'ema_200' in df_trend.columns else None
        if ema50_t is not None and ema200_t is not None and not (pd.isna(ema50_t) or pd.isna(ema200_t)):
            if ema50_t > ema200_t:
                potential_side = 'LONG'
            elif ema50_t < ema200_t:
                potential_side = 'SHORT'
            elif zone_result.zone_type == 'support':
                potential_side = 'LONG'
            elif zone_result.zone_type == 'resistance':
                potential_side = 'SHORT'
            else:
                potential_side = None  # No clear direction
        elif zone_result.zone_type == 'support':
            potential_side = 'LONG'
        elif zone_result.zone_type == 'resistance':
            potential_side = 'SHORT'
        else:
            potential_side = None  # No clear direction

    if potential_side is None:
        # No directional bias — skip this signal
        result.side = None
        result.is_valid = False
        result.total_score = 0
        return result

    
    # Get relevant pattern
    pattern_dir = 'bullish' if potential_side == 'LONG' else 'bearish'
    pattern = get_pattern_for_direction(df_entry, pattern_dir)
    
    result.pattern_info = {
        'pattern': pattern.name if pattern else None,
        'pattern_strength': pattern.strength if pattern else 0
    }
    
    # Wave 3 detection
    wave_result = detect_wave3_setup(df_trend, df_entry, potential_side.lower())

    # ── Trendline & Structure Analysis (visual chart reading) ──
    trendline_result = detect_trendline_break(df_entry, lookback=20)
    ltf_structure_result = detect_structure(df_entry, lookback=30)
    side_structure_dir = _side_to_structure_direction(potential_side)

    # LTF confirmation stack used after HTF raid (pattern / trendline / structure).
    ltf_confirmation_count = 0
    ltf_confirmation_tags: List[str] = []
    if pattern is not None and str(getattr(pattern, "direction", "")).lower() == side_structure_dir:
        ltf_confirmation_count += 1
        ltf_confirmation_tags.append("pattern")
    if trendline_result is not None:
        tl_type = str(trendline_result.get("type", "")).lower()
        if (potential_side == "LONG" and tl_type == "bullish_break") or \
           (potential_side == "SHORT" and tl_type == "bearish_break"):
            ltf_confirmation_count += 1
            ltf_confirmation_tags.append("trendline")
    if ltf_structure_result is not None:
        ltf_dir = str(ltf_structure_result.get("direction", "")).lower()
        if ltf_dir == side_structure_dir:
            ltf_confirmation_count += 1
            ltf_confirmation_tags.append("structure")

    # ── Smart Money Concepts (SMC) ──
    # NOTE: SMC is calculated on df_trend (15m) instead of df_entry (1m).
    # Order Blocks and FVGs on 1m are too short-lived and noisy — 15m zones
    # represent actual institutional accumulation/distribution areas.
    fvgs = detect_unmitigated_fvgs(df_trend, lookback=50)
    obs = detect_order_blocks(df_trend, lookback=50)
    sweeps = detect_liquidity_sweeps(df_trend, lookback=30)
    logger.debug(
        f"SMC [{symbol}]: OrderBlocks={len(obs)}, Sweeps={len(sweeps)}, FVGs={len(fvgs)} "
        f"| sweeps={[f'{s.direction}@{s.swept_price:.4f}' for s in sweeps]} "
        f"| obs={[f'{o.direction}@{o.top:.4f}-{o.bottom:.4f}' for o in obs[-3:]]}"
    )

    # ── Anti-Chasing Filter ──
    # Block entries where price already moved too far too fast.
    # Threshold is configurable because some sessions trend in bursts.
    _qf = getattr(config, 'quality_filter', None)
    _anti_mult = float(getattr(_qf, 'anti_chasing_atr_mult', 3.0))
    _anti_lookback = max(2, int(getattr(_qf, 'anti_chasing_lookback_bars', 3)))
    if 'atr_14' in df_entry.columns and len(df_entry) >= (_anti_lookback + 1):
        _atr_entry = float(df_entry['atr_14'].iloc[-1] or 0.0)
        _recent_move = abs(
            float(df_entry['close'].iloc[-1])
            - float(df_entry['close'].iloc[-(_anti_lookback + 1)])
        )
        if _atr_entry > 0 and _recent_move > _atr_entry * _anti_mult:
            result.side = None
            result.is_valid = False
            result.blocked_reason = (
                f"Anti-chasing: price moved {_recent_move:.4f} "
                f"({_recent_move/_atr_entry:.1f}x ATR) in last {_anti_lookback} candles"
            )
            logger.info(f"Anti-chasing filter blocked {symbol} {potential_side}: "
                        f"move={_recent_move:.4f} ATR={_atr_entry:.4f} "
                        f"(limit={_anti_mult:.1f}x/{_anti_lookback} bars)")
            return result

    # ── Order Flow Analysis (Funding Rate & Open Interest Squeezes) ──
    order_flow_score = 0.0
    of_cfg = getattr(config, 'order_flow', None)
    if of_cfg and getattr(of_cfg, 'enabled', True) and order_flow_data:
        fr = float(order_flow_data.get('funding_rate', 0.0))
        oi = float(order_flow_data.get('open_interest', 0.0))
        min_oi = float(getattr(of_cfg, 'min_open_interest_usd', 5000000))
        
        result.order_flow = {
            'funding_rate': fr,
            'open_interest': oi
        }
        
        # Only consider squeeze if Open Interest is large enough
        if oi >= min_oi:
            short_th = float(getattr(of_cfg, 'short_squeeze_funding_threshold', -0.05))
            long_th = float(getattr(of_cfg, 'long_squeeze_funding_threshold', 0.05))
            boost = float(getattr(of_cfg, 'score_boost', 2.0))
            
            # If funding is highly negative, majority is shorting → high risk of short squeeze (LONG edge)
            if fr <= short_th and potential_side == 'LONG':
                order_flow_score = boost
                result.reasons.append(f"Short Squeeze Setup (+{boost:.1f}): FR={fr:.3f}% OI=${oi/1e6:.1f}M")
            # If funding is highly positive, majority is longing → high risk of long squeeze (SHORT edge)
            elif fr >= long_th and potential_side == 'SHORT':
                order_flow_score = boost
                result.reasons.append(f"Long Squeeze Setup (+{boost:.1f}): FR={fr:.3f}% OI=${oi/1e6:.1f}M")

    # Extract ADX values from trend timeframe
    adx_val = df_trend['adx'].iloc[-1] if 'adx' in df_trend.columns else 0.0
    di_plus_val = df_trend['di_plus'].iloc[-1] if 'di_plus' in df_trend.columns else 0.0
    di_minus_val = df_trend['di_minus'].iloc[-1] if 'di_minus' in df_trend.columns else 0.0
    
    # Calculate technical score
    tech_score, tech_reasons = calculate_technical_score(
        trend=trend,
        side=potential_side,
        zone_result=zone_result,
        pattern=pattern,
        volume_spike=volume_spike,
        wave_result=wave_result,
        rsi_divergence=rsi_div,
        adx_value=adx_val,
        di_plus=di_plus_val,
        di_minus=di_minus_val
    )

    # Regime shaping: encourage aligned trades, penalize counter-trend/range risk.
    rg = getattr(config, "regime", None)
    if rg and getattr(rg, "enabled", True):
        aligned_bonus = float(getattr(rg, "aligned_score_bonus", 0.6))
        counter_penalty = float(getattr(rg, "countertrend_score_penalty", 1.0))

        if market_regime == "uptrend":
            if potential_side == "LONG":
                tech_score += aligned_bonus
                tech_reasons.append("✓ Regime uptrend aligned with LONG")
            else:
                tech_score -= counter_penalty
                tech_reasons.append("✗ Regime uptrend opposes SHORT")
        elif market_regime == "downtrend":
            if potential_side == "SHORT":
                tech_score += aligned_bonus
                tech_reasons.append("✓ Regime downtrend aligned with SHORT")
            else:
                tech_score -= counter_penalty
                tech_reasons.append("✗ Regime downtrend opposes LONG")
        elif market_regime == "sideways":
            tech_reasons.append("⚠ Regime sideways: higher fakeout risk")
        elif market_regime == "high_volatility":
            tech_reasons.append("⚠ Regime high volatility: spread/noise risk")

    # HTF workflow: structure + POI + raid + LTF confirmation
    if bool(getattr(_qf, "htf_context_enabled", True)) and df_htf_ind is not None:
        try:
            if potential_side == 'LONG' and htf_trend == 'down':
                tech_score -= 2.0
                tech_reasons.append("✗ HTF trend opposes LONG")
            elif potential_side == 'SHORT' and htf_trend == 'up':
                tech_score -= 2.0
                tech_reasons.append("✗ HTF trend opposes SHORT")
            elif (potential_side == 'LONG' and htf_trend == 'up') or \
                 (potential_side == 'SHORT' and htf_trend == 'down'):
                tech_score += 1.0
                tech_reasons.append(f"✓ HTF trend confirms {potential_side}")

            htf_structure_bonus = float(getattr(_qf, "htf_structure_bonus", 1.0))
            htf_structure_penalty = abs(float(getattr(_qf, "htf_structure_penalty", 1.3)))
            htf_poi_bonus = float(getattr(_qf, "htf_poi_bonus", 1.2))
            htf_poi_penalty = abs(float(getattr(_qf, "htf_poi_penalty", 1.2)))
            htf_raid_bonus = float(getattr(_qf, "htf_raid_bonus", 0.8))
            htf_raid_penalty = abs(float(getattr(_qf, "htf_raid_penalty", 0.6)))
            ltf_confirmation_min = max(1, int(getattr(_qf, "ltf_confirmation_min_signals", 1)))
            ltf_confirmation_bonus = float(getattr(_qf, "ltf_confirmation_bonus", 0.8))
            ltf_no_confirmation_penalty = abs(float(getattr(_qf, "ltf_no_confirmation_penalty", 1.0)))
            poi_max_distance_pct = max(0.05, float(getattr(_qf, "htf_poi_max_distance_pct", 0.35)))

            if htf_structure_result:
                htf_s_type = str(htf_structure_result.get("type", ""))
                htf_s_dir = str(htf_structure_result.get("direction", "")).lower()
                if htf_s_dir == side_structure_dir:
                    tech_score += htf_structure_bonus
                    tech_reasons.append(
                        f"✓ HTF structure confirms {potential_side} ({htf_s_type})"
                    )
                elif htf_s_dir in {"bullish", "bearish"}:
                    tech_score -= htf_structure_penalty
                    tech_reasons.append(
                        f"✗ HTF structure opposes {potential_side} ({htf_s_type})"
                    )

            if htf_zone_result is not None:
                htf_zone_type = str(getattr(htf_zone_result, "zone_type", "") or "").lower()
                htf_in_zone = bool(getattr(htf_zone_result, "in_zone", False))
                htf_dist_pct = float(getattr(htf_zone_result, "distance_pct", 1.0) or 1.0) * 100.0
                htf_near_zone = htf_in_zone or (htf_dist_pct <= poi_max_distance_pct)
                htf_zone_aligned = _zone_aligned_with_side(htf_zone_type, potential_side)

                if htf_near_zone and htf_zone_aligned:
                    tech_score += htf_poi_bonus
                    tag = "in-zone" if htf_in_zone else "near-zone"
                    tech_reasons.append(
                        f"✓ HTF POI aligned ({htf_zone_type}, {tag}, dist={htf_dist_pct:.2f}%)"
                    )
                elif htf_near_zone and htf_zone_type in {"support", "resistance"}:
                    tech_score -= htf_poi_penalty
                    tech_reasons.append(
                        f"✗ HTF POI opposes {potential_side} ({htf_zone_type}, dist={htf_dist_pct:.2f}%)"
                    )

            htf_raid_aligned = False
            if htf_sweeps:
                last_htf_sweep = htf_sweeps[-1]
                sweep_dir = str(getattr(last_htf_sweep, "direction", "")).lower()
                if (potential_side == "LONG" and sweep_dir == "bullish") or \
                   (potential_side == "SHORT" and sweep_dir == "bearish"):
                    htf_raid_aligned = True
                    tech_score += htf_raid_bonus
                    tech_reasons.append(f"✓ HTF raid aligned ({sweep_dir} liquidity sweep)")
                elif sweep_dir in {"bullish", "bearish"}:
                    tech_score -= htf_raid_penalty
                    tech_reasons.append(f"✗ HTF raid opposes {potential_side} ({sweep_dir} sweep)")

            if htf_raid_aligned:
                if ltf_confirmation_count >= ltf_confirmation_min:
                    tech_score += ltf_confirmation_bonus
                    tech_reasons.append(
                        f"✓ LTF confirmation after HTF raid "
                        f"({ltf_confirmation_count} signals: {','.join(ltf_confirmation_tags)})"
                    )
                else:
                    tech_score -= ltf_no_confirmation_penalty
                    tech_reasons.append(
                        f"✗ HTF raid without enough LTF confirmation "
                        f"({ltf_confirmation_count}/{ltf_confirmation_min})"
                    )
        except Exception as e:
            logger.debug(f"HTF workflow scoring error: {e}")

    # ── Trendline break scoring (what traders see when drawing lines) ──
    if trendline_result:
        tl_type = trendline_result.get('type', '')
        if (potential_side == 'LONG' and tl_type == 'bullish_break') or \
           (potential_side == 'SHORT' and tl_type == 'bearish_break'):
            tech_score += 2.0
            tech_reasons.append(f"✓ Trendline {tl_type.replace('_', ' ')}")
        elif (potential_side == 'LONG' and tl_type == 'bearish_break') or \
             (potential_side == 'SHORT' and tl_type == 'bullish_break'):
            tech_score -= 1.0
            tech_reasons.append(f"✗ Trendline {tl_type.replace('_', ' ')} opposes")

    # ── LTF market structure scoring (confirmation layer) ──
    if ltf_structure_result:
        s_type = ltf_structure_result.get('type', '')
        s_dir = ltf_structure_result.get('direction', '')
        s_strength = ltf_structure_result.get('strength', 1)
        if s_dir == 'bullish' and potential_side == 'LONG':
            bonus = 2.0 if s_type in ('double_bottom',) else 1.5
            tech_score += bonus
            label = {'bullish_structure': 'HH/HL structure', 'double_bottom': 'Double Bottom'}.get(s_type, s_type)
            tech_reasons.append(f"✓ {label}")
        elif s_dir == 'bearish' and potential_side == 'SHORT':
            bonus = 2.0 if s_type in ('double_top',) else 1.5
            tech_score += bonus
            label = {'bearish_structure': 'LH/LL structure', 'double_top': 'Double Top'}.get(s_type, s_type)
            tech_reasons.append(f"✓ {label}")
        elif s_dir != '' and s_dir != ('bullish' if potential_side == 'LONG' else 'bearish'):
            tech_score -= 1.0
            tech_reasons.append(f"✗ Structure opposes {potential_side}")

    # ── Smart Money Concepts (SMC) Scoring ── (Confluence Multiplier)
    # Instead of flat additive scores, we count how many SMC signals align
    # at the same price level. 2+ signals = exponential bonus (institutions).
    _smc_confluence = 0
    _smc_reasons = []

    # 1. Order Blocks (OB)
    nearest_ob = get_nearest_order_block(obs, current_price, potential_side)
    _ob_near = False
    if nearest_ob:
        ob_distance = abs(current_price - (nearest_ob.top if potential_side == 'LONG' else nearest_ob.bottom)) / current_price
        if ob_distance < 0.01:  # Within 1% of OB
            _ob_near = True
            _smc_confluence += 1
            _smc_reasons.append(f"✓ {nearest_ob.direction.title()} Order Block")

    # 2. Liquidity Sweeps
    _sweep_near = False
    if sweeps:
        last_sweep = sweeps[-1]
        if (last_sweep.direction == 'bullish' and potential_side == 'LONG') or \
           (last_sweep.direction == 'bearish' and potential_side == 'SHORT'):
            _sweep_near = True
            
            last_kz = getattr(last_sweep, 'killzone', 'None')
            if last_kz in ['London', 'NewYork']:
                _smc_confluence += 2  # Stronger institutional bonus
                _smc_reasons.append(f"✓ {last_kz} Killzone {last_sweep.direction.title()} Sweep")
            else:
                _smc_confluence += 1
                _smc_reasons.append(f"✓ {last_sweep.direction.title()} Liquidity Sweep")

    # 3. Fair Value Gaps (FVG) — magnet check
    nearest_fvg_target = get_nearest_unmitigated_fvg(fvgs, current_price, potential_side)
    if nearest_fvg_target:
        result.extra_data = getattr(result, "extra_data", {})
        result.extra_data['fvg_target'] = nearest_fvg_target

    opposing_fvg = get_nearest_unmitigated_fvg(fvgs, current_price, "SHORT" if potential_side == "LONG" else "LONG")
    if opposing_fvg:
        opp_distance = abs(current_price - (opposing_fvg.top if potential_side == 'LONG' else opposing_fvg.bottom)) / current_price
        if opp_distance < 0.015:  # Within 1.5% is dangerous
            tech_score -= 3.0
            tech_reasons.append("✗ Unfilled FVG opposes entry (Magnet Risk)")

    # 4. BOS / CHoCH — market structure confirmation
    _struct_breaks = detect_structure_breaks(df_trend, lookback=50)
    _latest_break = _struct_breaks[-1] if _struct_breaks else None
    if _latest_break:
        _side_dir = 'bullish' if potential_side == 'LONG' else 'bearish'
        if _latest_break.direction == _side_dir:
            _smc_confluence += 1
            _label = f"{'BOS' if _latest_break.break_type == 'BOS' else 'CHoCH'} {_latest_break.direction.title()}"
            _tag = "trend continuation" if _latest_break.break_type == 'BOS' else "reversal signal"
            _smc_reasons.append(f"✓ {_label} ({_tag})")
        else:
            tech_score -= 1.5
            tech_reasons.append(f"✗ {_latest_break.break_type} {_latest_break.direction} opposes {potential_side}")

    # Apply confluence-multiplied SMC score
    if _smc_confluence >= 2:
        # Multiple SMC signals at same level = institutional zone — strong bonus
        smc_bonus = _smc_confluence * 2.5
        tech_score += smc_bonus
        for r in _smc_reasons:
            tech_reasons.append(r)
        tech_reasons.append(f"✓ SMC Confluence x{_smc_confluence} (sniper entry +{smc_bonus:.1f})")
    elif _smc_confluence == 1:
        tech_score += 2.0
        for r in _smc_reasons:
            tech_reasons.append(r)

    # Store OB for SL placement
    if nearest_ob and _ob_near:
        result.extra_data = getattr(result, "extra_data", {})
        result.extra_data['support_ob'] = nearest_ob

    # ── CVD (Cumulative Volume Delta) Scoring ──
    # CVD divergence: if CVD is moving opposite to price → danger signal
    if 'cvd' in df_entry.columns and len(df_entry) >= 10:
        cvd_now  = float(df_entry['cvd'].iloc[-1])
        cvd_prev = float(df_entry['cvd'].iloc[-10])
        close_now  = float(df_entry['close'].iloc[-1])
        close_prev = float(df_entry['close'].iloc[-10])
        cvd_rising = cvd_now > cvd_prev
        price_rising = close_now > close_prev
        if potential_side == 'LONG':
            if cvd_rising and price_rising:
                tech_score += 1.0
                tech_reasons.append(f"✓ CVD confirms bullish (buyers in control)")
            elif not cvd_rising and price_rising:
                tech_score -= 1.5
                tech_reasons.append(f"✗ CVD bearish divergence (price up, delta falling)")
        elif potential_side == 'SHORT':
            if not cvd_rising and not price_rising:
                tech_score += 1.0
                tech_reasons.append(f"✓ CVD confirms bearish (sellers in control)")
            elif cvd_rising and not price_rising:
                tech_score -= 1.5
                tech_reasons.append(f"✗ CVD bullish divergence (price down, delta rising)")

    # Diminishing returns: scores above 10 historically underperform (overconfidence).
    # Soft-cap excess above 10 at 50% effectiveness.
    if tech_score > 10:
        tech_score = 10.0 + (tech_score - 10.0) * 0.5

    result.technical_score = tech_score
    result.reasons = tech_reasons

    # Calculate VPA score (independent — does NOT affect signal decision)
    vpa_score, vpa_info = calculate_vpa_score(df_entry, potential_side, current_price)
    result.vpa_score = vpa_score
    result.vpa_info = vpa_info
    
    # Calculate timing/news score
    timing_score, timing_reasons = calculate_timing_score(
        df_trend, df_entry, symbol, trend, potential_side, current_price, config
    )

    result.timing_info = timing_score
    result.timing_reasons = timing_reasons
    result.timing_score = timing_score.total_score

    # Optionally blend timing/news into final signal score.
    # This uses existing config knobs:
    # - scoring.add_timing_to_score
    # - scoring.max_timing_points_used
    timing_contribution = 0.0
    if config.scoring.add_timing_to_score:
        cap = max(0, int(config.scoring.max_timing_points_used))
        raw_timing = timing_score.total_score
        timing_contribution = max(-float(cap), min(float(cap), raw_timing))
        result.total_score = tech_score + timing_contribution
        if timing_contribution != 0:
            result.timing_reasons.append(
                f"Timing contribution applied to final score: {timing_contribution:+.1f}"
            )
    else:
        result.total_score = tech_score

    # Macro safety guard: raise entry threshold during high-impact news windows.
    # When quality_filter.block_during_high_impact_news is disabled, run warn-only
    # mode and avoid threshold uplift from calendar/sentiment windows.
    threshold = float(base_threshold_for_asset)
    default_base = float(getattr(config.scoring, "base_threshold", threshold))
    if abs(threshold - default_base) > 1e-9:
        result.reasons.append(f"Threshold: asset base {asset_class}={threshold:.2f} (default={default_base:.2f})")
    news_guard_enabled = bool(
        getattr(getattr(config, "quality_filter", None), "block_during_high_impact_news", True)
    )
    
    # Adaptive Volatility Check for Post-News Recovery
    volatility_settled = False
    if 'atr_14' in df_entry.columns and len(df_entry) >= 50:
        current_atr = df_entry['atr_14'].iloc[-1]
        avg_atr = df_entry['atr_14'].rolling(window=50).mean().iloc[-1]
        if not pd.isna(current_atr) and not pd.isna(avg_atr) and current_atr <= avg_atr * 1.3:
            volatility_settled = True
            if news_guard_enabled:
                result.reasons.append(f"✓ Volatility settled (ATR {current_atr:.4f} <= Avg ATR {avg_atr:.4f} * 1.3). Bypassing news block.")
                news_guard_enabled = False

    high_impact_hits: List[str] = []
    if timing_score.fomc_analysis and timing_score.fomc_analysis.in_high_vol_window:
        high_impact_hits.append("FOMC")
        if news_guard_enabled:
            threshold += 2
    if timing_score.cpi_analysis and timing_score.cpi_analysis.in_high_vol_window:
        high_impact_hits.append("CPI")
        if news_guard_enabled:
            threshold += 2
    if timing_score.nfp_analysis and timing_score.nfp_analysis.in_high_vol_window:
        high_impact_hits.append("NFP")
        if news_guard_enabled:
            threshold += 2
    if timing_score.powell_analysis and timing_score.powell_analysis.in_high_vol_window:
        high_impact_hits.append("POWELL")
        if news_guard_enabled:
            threshold += 1
    if timing_score.fomc_minutes_analysis and timing_score.fomc_minutes_analysis.in_high_vol_window:
        high_impact_hits.append("FOMC_MINUTES")
        if news_guard_enabled:
            threshold += 1
    if timing_score.sentiment_analysis and timing_score.sentiment_analysis.in_extreme_zone:
        high_impact_hits.append("SENTIMENT")
        if news_guard_enabled:
            threshold += 1
    if timing_score.social_sentiment_analysis and timing_score.social_sentiment_analysis.in_extreme_zone:
        high_impact_hits.append("SOCIAL")
        if news_guard_enabled:
            threshold += 1
    if high_impact_hits and not news_guard_enabled and not volatility_settled:
        result.reasons.append("⚠ News window active: warn-only mode (no threshold block)")
    if rg and getattr(rg, "enabled", True):
        sideways_add = float(getattr(rg, "sideways_threshold_add", 1))
        high_vol_add = float(getattr(rg, "high_vol_threshold_add", 1))
        if asset_class == "macro":
            # Keep macro entries selective, but avoid over-penalizing them.
            sideways_add *= 0.55
            high_vol_add *= 0.65
        if market_regime == "sideways":
            threshold += sideways_add
        elif market_regime == "high_volatility":
            threshold += high_vol_add

    if bool(getattr(config.scoring, "dynamic_threshold_enabled", True)):
        atr_pct = float((result.zone_info or {}).get("atr_pct", 0.0) or 0.0)
        adx_val = float((result.zone_info or {}).get("adx", 0.0) or 0.0)
        low_vol_atr = float(getattr(config.scoring, "dynamic_low_vol_atr_pct", 0.45))
        high_vol_atr = float(getattr(config.scoring, "dynamic_high_vol_atr_pct", 1.25))
        low_vol_adj = float(getattr(config.scoring, "dynamic_low_vol_adjust", -0.5))
        high_vol_adj = float(getattr(config.scoring, "dynamic_high_vol_adjust", 1.0))
        strong_trend_min_adx = float(getattr(config.scoring, "dynamic_strong_trend_adx_min", 28.0))
        strong_trend_adj = float(getattr(config.scoring, "dynamic_strong_trend_adjust", -0.5))
        if asset_class == "macro":
            high_vol_adj *= 0.75

        if atr_pct >= high_vol_atr:
            threshold += high_vol_adj
            result.reasons.append(
                f"Threshold: dynamic high-vol ATR {atr_pct:.2f}% >= {high_vol_atr:.2f}% ({high_vol_adj:+.2f})"
            )
        elif atr_pct <= low_vol_atr:
            threshold += low_vol_adj
            result.reasons.append(
                f"Threshold: dynamic low-vol ATR {atr_pct:.2f}% <= {low_vol_atr:.2f}% ({low_vol_adj:+.2f})"
            )

        if trend in {"up", "down"} and adx_val >= strong_trend_min_adx and strong_trend_adj != 0:
            threshold += strong_trend_adj
            result.reasons.append(
                f"Threshold: dynamic strong-trend ADX {adx_val:.1f} >= {strong_trend_min_adx:.1f} ({strong_trend_adj:+.2f})"
            )

    # Soft guard against obvious VPA contradiction:
    # we don't hard-block, but we make borderline entries harder to pass.
    if bool((vpa_info or {}).get("raw_opposes_side", False)):
        raw_sig = str((vpa_info or {}).get("vpa_signal", "neutral"))
        contradiction_add = 0.45
        if float(vpa_score) <= -2.0:
            contradiction_add = 0.80
        threshold += contradiction_add
        result.reasons.append(
            f"Threshold: VPA contradiction {raw_sig} opposes {potential_side} ({contradiction_add:+.2f})"
        )

    min_threshold = float(getattr(config.scoring, "min_threshold", 3.0))
    if threshold < min_threshold:
        result.reasons.append(f"Threshold: floor applied {threshold:.2f} -> {min_threshold:.2f}")
    threshold = max(min_threshold, threshold)
    result.threshold = threshold
        
    # RSI Safety Filter on Trend Timeframe
    rsi_trend = df_trend['rsi_14'].iloc[-1] if 'rsi_14' in df_trend.columns else 50
    if potential_side == 'SHORT':
        if rsi_trend > 65 and trend == 'up':
            # Too strong upward momentum, highly likely a trap to short
            result.total_score -= 3
            result.reasons.append(f"✗ Blocked: Strong bullish momentum (RSI {rsi_trend:.1f})")
        elif rsi_trend < 35:
            # Already oversold, bad risk/reward
            result.total_score -= 2
            result.reasons.append(f"✗ Penalty: Already oversold (RSI {rsi_trend:.1f})")
            
    elif potential_side == 'LONG':
        if rsi_trend < 35 and trend == 'down':
            # Too strong downward momentum, catching a falling knife
            result.total_score -= 3
            result.reasons.append(f"✗ Blocked: Strong bearish momentum (RSI {rsi_trend:.1f})")
        elif rsi_trend > 65:
            # Already overbought, bad risk/reward
            result.total_score -= 2
            result.reasons.append(f"✗ Penalty: Already overbought (RSI {rsi_trend:.1f})")
    
    # ── Score Percentile Filter ──
    # Track score distribution in memory. Penalize signals in the bottom 40%
    # of recent scores — if many setups are forming, only trade the best ones.
    if len(_score_history) >= 50:
        _n_below = sum(1 for s in _score_history if s <= result.total_score)
        _percentile = _n_below / len(_score_history)
        if _percentile < 0.40:
            # Score is in the bottom 40% of recent history — raise threshold slightly
            _percentile_penalty = 0.5
            threshold += _percentile_penalty
            result.reasons.append(
                f"Threshold: score percentile {_percentile*100:.0f}% < 40% ({_percentile_penalty:+.1f})"
            )
    _score_history.append(float(result.total_score))

    # ── Fee-Aware Expectancy Filter ──
    # Block trades where the expected net PnL (after fees) is below minimum
    # and we have enough historical data to be confident in the estimate.
    _ld = getattr(result, 'learning_decision', None)
    if _ld is not None:
        _expected_pnl  = float(getattr(_ld, 'expected_pnl_pct', 0.0))
        _sample_size   = int(getattr(_ld, 'sample_size', 0))
        _learning_cfg = getattr(config, 'learning', None)
        if asset_class == "macro":
            _est_cost_pct = float(getattr(_learning_cfg, 'estimated_roundtrip_cost_pct_macro', 0.03))
        else:
            _est_cost_pct = float(getattr(_learning_cfg, 'estimated_roundtrip_cost_pct_crypto', 0.08))
        _est_cost_pct = max(0.0, _est_cost_pct)

        # If live spread is available, ensure cost floor reflects current market spread.
        _live_spread_pct = None
        try:
            _ask = float(ticker.get('ask'))
            _bid = float(ticker.get('bid'))
            _ref_price = float(_chart_price_from_ticker(ticker))
            if _ask > 0 and _bid > 0 and _ref_price > 0:
                _live_spread_pct = abs(_ask - _bid) / _ref_price * 100.0
        except Exception:
            _live_spread_pct = None
        if _live_spread_pct is not None:
            _est_cost_pct = max(_est_cost_pct, _live_spread_pct)

        _net_expected  = _expected_pnl - _est_cost_pct
        _min_edge      = -0.05
        if _sample_size >= 20 and _net_expected < _min_edge:
            threshold += 0.6
            result.reasons.append(
                f"Threshold: fee-aware expectancy {_net_expected:+.3f}% < {_min_edge:+.3f}% "
                f"(cost {_est_cost_pct:.3f}%) "
                f"(n={_sample_size}) (+0.6)"
            )

    # ── 7. Machine Learning Engine Validation ──
    ml_cfg = getattr(config, 'ml_engine', None)
    if ml_cfg and getattr(ml_cfg, 'enabled', True) and ml_engine and ml_engine.is_trained:
        fr = result.order_flow.get('funding_rate', 0.0) if result.order_flow else 0.0
        prob = ml_engine.predict_probability(result.total_score, result.timestamp, result.reasons, fr)
        result.ml_prediction = {'win_probability': prob}
        
        min_boost_th = float(getattr(ml_cfg, 'min_win_probability_boost', 0.55))
        max_pen_th = float(getattr(ml_cfg, 'max_win_probability_penalty', 0.45))
        
        if prob >= min_boost_th:
            boost = float(getattr(ml_cfg, 'ml_score_boost', 1.5))
            result.total_score += boost
            result.reasons.append(f"ML Model Confirmed (+{boost}): Win Probability {prob*100:.1f}%")
        elif prob <= max_pen_th:
            penalty = float(getattr(ml_cfg, 'ml_threshold_penalty', 1.5))
            threshold += penalty
            result.threshold = threshold
            result.reasons.append(f"ML Model Warning (Threshold +{penalty}): Win Probability {prob*100:.1f}%")

    # Advisory-only "Schools" section (separate display, no execution impact).
    schools_cfg = dict(getattr(config, "schools", {}) or {})
    schools_enabled = bool(schools_cfg.get("enabled", True))
    if schools_enabled:
        try:
            advisory_pattern = get_advisory_pattern_for_direction(df_entry, pattern_dir) or pattern
            result.schools_opinion = _build_schools_opinion(
                trend=trend,
                potential_side=potential_side,
                df_trend=df_trend,
                zone_result=zone_result,
                pattern=advisory_pattern,
                wave_result=wave_result,
                trendline_result=trendline_result,
                structure_result=ltf_structure_result,
                vpa_info=result.vpa_info,
                timing_info=result.timing_info,
            )
        except Exception as schools_exc:
            logger.debug(f"Schools advisory snapshot failed for {symbol}: {schools_exc}")
            result.schools_opinion = None
    else:
        result.schools_opinion = None

    # Check if signal qualifies
    if result.total_score >= threshold:
        # Signal qualifies!
        result.side = potential_side
        result.is_valid = True
        
        # Calculate risk levels
        # Use a direction-aware stop reference zone:
        # LONG -> nearest support below price
        # SHORT -> nearest resistance above price
        if potential_side == 'LONG':
            if zone_result.in_zone and zone_result.zone is not None and zone_result.zone.type == 'support':
                stop_zone = zone_result.zone
            else:
                stop_zone = get_nearest_support(current_price, zones)
            next_zone = get_nearest_resistance(current_price, zones)
        else:
            if zone_result.in_zone and zone_result.zone is not None and zone_result.zone.type == 'resistance':
                stop_zone = zone_result.zone
            else:
                stop_zone = get_nearest_resistance(current_price, zones)
            next_zone = get_nearest_support(current_price, zones)
        
        atr_value = df_entry['atr_14'].iloc[-1] if 'atr_14' in df_entry.columns else None
        risk_profile = get_dynamic_risk_parameters(config, market_regime)
        if atr_calibration_scale != 1.0:
            risk_profile["atr_stop_mult"] *= atr_calibration_scale
        
        result.risk_levels = calculate_risk_levels(
            ticker=ticker,
            side=potential_side,
            zone=stop_zone,
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
            support_ob=result.extra_data.get('support_ob') if hasattr(result, 'extra_data') else None,
            fvg_target=result.extra_data.get('fvg_target') if hasattr(result, 'extra_data') else None,
        )

        # ── Maximum Stop Distance Filter ──
        # Prevent oversized stops (especially on thin/volatile altcoins) from being traded live.
        if result.risk_levels and result.risk_levels.entry > 0:
            _sl_pct = (float(result.risk_levels.risk_amount) / float(result.risk_levels.entry)) * 100.0
            _max_sl_pct = float(
                getattr(
                    config.risk,
                    'max_sl_pct_macro' if asset_class == 'macro' else 'max_sl_pct_crypto',
                    0.0
                )
            )
            if _max_sl_pct > 0 and _sl_pct > _max_sl_pct:
                result.side = None
                result.is_valid = False
                result.blocked_reason = (
                    f"Max SL filter: SL distance {_sl_pct:.2f}% > {_max_sl_pct:.2f}%"
                )
                logger.info(
                    f"Max SL blocked {symbol} {potential_side}: "
                    f"sl_pct={_sl_pct:.2f}% > {_max_sl_pct:.2f}%"
                )
                return result

        # ── Minimum R:R Filter ──
        # Block signals where the actual R:R to TP1 is below the configured minimum.
        # Even a great setup isn't worth trading if the math doesn't work out.
        _min_rr = float(getattr(config.risk, 'min_rr_tp1', 1.0))

        if result.risk_levels:
            _rr_raw = float(getattr(result.risk_levels, 'rr_ratio_tp1', 0.0))
            _rr_net = float(getattr(result.risk_levels, 'rr_ratio_tp1_net', _rr_raw))
            _spread_cost = float(getattr(result.risk_levels, 'spread_cost', 0.0))
        else:
            _rr_raw = 0.0
            _rr_net = 0.0
            _spread_cost = 0.0

        if result.risk_levels and _rr_net < _min_rr:
            _spread_txt = f"{_spread_cost:.6f}" if _spread_cost < 1 else f"{_spread_cost:.2f}"
            result.side = None
            result.is_valid = False
            result.blocked_reason = (
                f"Min R:R net filter: TP1 R:R net {_rr_net:.2f} < {_min_rr:.2f} "
                f"(raw {_rr_raw:.2f}, spread {_spread_txt})"
            )
            logger.info(f"Min R:R blocked {symbol} {potential_side}: "
                        f"R:R_net={_rr_net:.2f} (raw={_rr_raw:.2f}, spread={_spread_txt}) < {_min_rr:.2f}")
            return result

        # Optional expert opinion from OpenAI (decision + SL/TP sanity).
        if config.openai.enabled and config.openai.api_key and getattr(config.time_analysis, 'expert_advisor', True):
            logger.info(f"Expert advisor block reached for {symbol} {potential_side} (score={result.total_score:.1f})")
            try:
                candle_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df_entry.columns]
                recent_candles: List[Dict[str, float]] = []
                if candle_cols:
                    for _, row in df_entry[candle_cols].tail(5).iterrows():
                        candle: Dict[str, float] = {}
                        for col in candle_cols:
                            val = row[col]
                            if pd.isna(val):
                                continue
                            candle[col] = float(val)
                        if candle:
                            recent_candles.append(candle)

                adx_val_ctx = result.zone_info.get('adx', 0.0) if result.zone_info else 0.0
                rsi_val_ctx = result.zone_info.get('rsi_trend', 50.0) if result.zone_info else 50.0
                ichimoku_ctx = result.zone_info.get('ichimoku_signal', 'neutral') if result.zone_info else 'neutral'
                pattern_name = result.pattern_info.get('pattern') if result.pattern_info else None
                pattern_strength = result.pattern_info.get('pattern_strength') if result.pattern_info else 0

                result.expert_opinion = get_expert_trade_opinion(
                    symbol=symbol,
                    side=potential_side,
                    current_price=current_price,
                    trend=trend,
                    technical_score=float(result.technical_score),
                    timing_score=float(result.timing_score),
                    total_score=float(result.total_score),
                    threshold=float(result.threshold),
                    reasons=result.reasons,
                    timing_reasons=result.timing_reasons,
                    vpa_info=result.vpa_info,
                    risk_levels=result.risk_levels,
                    extra_context={
                        "trend_timeframe": config.trend_tf,
                        "entry_timeframe": config.entry_tf,
                        "trend_adx": float(adx_val_ctx),
                        "trend_rsi": float(rsi_val_ctx),
                        "ichimoku_signal": str(ichimoku_ctx),
                        "pattern": pattern_name,
                        "pattern_strength": int(pattern_strength or 0),
                        "recent_candles": recent_candles,
                    },
                    openai_config=config.openai,
                    timeout_seconds=getattr(config.time_analysis, 'expert_advisor_timeout_seconds', 75),
                )

                # Optional hard filter: expert can veto weak/conflicting entries.
                if result.expert_opinion and getattr(config.time_analysis, 'expert_use_as_filter', False):
                    op = result.expert_opinion
                    decision = str(op.get("decision", "WAIT")).upper()
                    confidence_raw = op.get("confidence", 0)
                    try:
                        confidence = int(float(confidence_raw))
                    except Exception:
                        confidence = 0
                    confidence = max(0, min(100, confidence))

                    expected_decision = "BUY" if potential_side == "LONG" else "SELL"
                    min_conf = max(0, min(100, int(getattr(config.time_analysis, 'expert_min_confidence', 60))))
                    block_reason = None

                    if decision not in {"BUY", "SELL", "WAIT"}:
                        block_reason = "Expert filter: invalid decision"
                    elif decision == "WAIT" and getattr(config.time_analysis, 'expert_block_on_wait', True):
                        block_reason = f"Expert filter: WAIT ({confidence}%)"
                    elif decision in {"BUY", "SELL"} and confidence < min_conf:
                        block_reason = f"Expert filter: low confidence {confidence}% < {min_conf}%"
                    elif (
                        decision in {"BUY", "SELL"}
                        and getattr(config.time_analysis, 'expert_require_side_alignment', True)
                        and decision != expected_decision
                    ):
                        block_reason = f"Expert filter: {decision} opposes {expected_decision}"

                    if block_reason:
                        result.is_valid = False
                        result.side = None
                        result.blocked_reason = block_reason
                        result.reasons.append(f"✗ {block_reason}")
            except Exception as e:
                logger.warning(f"Expert advisor outer error for {symbol}: {e}")

        # Optional manager opinion from OpenAI (market-structure scenario).
        if result.is_valid and config.openai.enabled and config.openai.api_key and getattr(config.time_analysis, 'manager_advisor', True):
            try:
                candle_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df_entry.columns]
                recent_candles: List[Dict[str, float]] = []
                if candle_cols:
                    for _, row in df_entry[candle_cols].tail(12).iterrows():
                        candle: Dict[str, float] = {}
                        for col in candle_cols:
                            val = row[col]
                            if pd.isna(val):
                                continue
                            candle[col] = float(val)
                        if candle:
                            recent_candles.append(candle)

                result.manager_opinion = get_manager_opinion(
                    symbol=symbol,
                    current_price=current_price,
                    trend=trend,
                    extra_context={
                        "candidate_side": potential_side,
                        "technical_score": float(result.technical_score),
                        "timing_score": float(result.timing_score),
                        "total_score": float(result.total_score),
                        "threshold": float(result.threshold),
                        "market_regime": market_regime,
                        "zone_info": result.zone_info or {},
                        "pattern_info": result.pattern_info or {},
                        "vpa_info": result.vpa_info or {},
                        "technical_reasons": list(result.reasons[:10]),
                        "timing_reasons": list(result.timing_reasons[:10]),
                        "recent_candles": recent_candles,
                    },
                    risk_levels=result.risk_levels,
                    openai_config=config.openai,
                    timeout_seconds=getattr(config.time_analysis, 'manager_advisor_timeout_seconds', getattr(config.time_analysis, 'expert_advisor_timeout_seconds', 75)),
                )
            except Exception as e:
                logger.warning(f"Manager advisor failed for {symbol}: {e}")
    else:
        result.blocked_reason = f"Score {result.total_score:.1f} below threshold {threshold}"
    
    return result


def check_exit_conditions(
    symbol: str,
    df_entry: pd.DataFrame,
    zones: List[Zone],
    current_price: float,
    current_position: str,  # 'LONG' or 'SHORT'
    position_opened_at: Optional[datetime],
    config: Config
) -> Optional[SignalResult]:
    """
    Check if exit conditions are met for an existing position.
    
    Exit triggers:
    - Minimum hold time has passed
    - Strong reversal candle appears at the correct opposite zone
    
    Args:
        symbol: Trading symbol
        df_entry: Entry timeframe DataFrame
        zones: Current S/R zones
        current_price: Current price
        current_position: Current position side
        position_opened_at: Timestamp when current position was opened
        config: Configuration
        
    Returns:
        SignalResult with EXIT if conditions met, None otherwise
    """
    now = datetime.now(timezone.utc)

    min_hold_minutes = max(0, int(getattr(config.time_analysis, 'exit_min_hold_minutes', 5)))
    if position_opened_at is not None and min_hold_minutes > 0:
        opened_at = position_opened_at
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        held_minutes = (now - opened_at).total_seconds() / 60.0
        if held_minutes < min_hold_minutes:
            logger.debug(
                f"{symbol} exit skipped: held {held_minutes:.1f}m < min {min_hold_minutes}m"
            )
            return None
    
    # Check for reversal pattern
    exit_direction = 'bearish' if current_position == 'LONG' else 'bullish'
    reversal_pattern = get_pattern_for_direction(df_entry, exit_direction)
    
    if not reversal_pattern:
        return None
    
    # Check if at zone
    zone_result = is_price_in_zone(current_price, zones)
    
    if not zone_result.in_zone:
        return None

    # Exit should happen at the opposite protective zone:
    # LONG -> resistance, SHORT -> support.
    required_zone = 'resistance' if current_position == 'LONG' else 'support'
    if zone_result.zone_type != required_zone:
        logger.debug(
            f"{symbol} exit skipped: zone={zone_result.zone_type}, required={required_zone}"
        )
        return None

    min_strength = max(1, int(getattr(config.time_analysis, 'exit_min_pattern_strength', 3)))
    if reversal_pattern.strength < min_strength:
        logger.debug(
            f"{symbol} exit skipped: pattern={reversal_pattern.name} strength={reversal_pattern.strength} < {min_strength}"
        )
        return None

    # Reverse-close guard: require consecutive opposite candle closes.
    confirm_bars = max(1, int(getattr(config.time_analysis, 'exit_reverse_confirmation_bars', 2)))
    if len(df_entry) < confirm_bars + 2:
        logger.debug(f"{symbol} exit skipped: insufficient bars for reverse confirmation")
        return None

    confirm_ok = True
    for idx in range(2, 2 + confirm_bars):
        row = df_entry.iloc[-idx]
        o = row.get('open')
        c = row.get('close')
        if pd.isna(o) or pd.isna(c):
            confirm_ok = False
            break
        if current_position == 'LONG':
            if not (float(c) < float(o)):
                confirm_ok = False
                break
        else:
            if not (float(c) > float(o)):
                confirm_ok = False
                break

    if not confirm_ok:
        logger.debug(
            f"{symbol} exit skipped: reverse confirmation bars ({confirm_bars}) not satisfied"
        )
        return None

    return SignalResult(
        symbol=symbol,
        timestamp=now,
        side='EXIT',
        technical_score=0,
        timing_score=0,
        total_score=0,
        threshold=0,
        trend='',
        market_regime='sideways',
        current_price=current_price,
        reasons=[f"Exit: {reversal_pattern.name} at {zone_result.zone_type}"],
        is_valid=True
    )
