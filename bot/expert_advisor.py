"""
Expert Trade Advisor
====================
Builds an expert-style trading opinion (40+ years style) from full signal context
using OpenAI API, including decision and suggested SL/TP levels.
"""

import json
import re
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI
from openai import OpenAIError

from .risk import RiskLevels
from .utils import OpenAIConfig

ADVISOR_PROMPT = """أنت متداول محترف ومحافظ بخبرة 40 سنة في الأسواق المالية (الفوركس، الكريبتو، الأسهم) ومتخصص حصراً في مفاهيم المال الذكي والسلوك السعري (Smart Money Concepts & Price Action).

مهمتك:
تقييم البيانات المتاحة واتخاذ قرار تداول صارم ومحترف للغاية (BUY أو SELL أو WAIT). 
أنت لا تبدأ بالسؤال: "أين أدخل؟"، بل تبدأ دائماً بالسؤال: "أي سيولة يستهدفها السوق أولاً؟ وبأي سياق؟"

يجب أن تعتمد في قرارك على المنهج الاحترافي التالي:

1) تحليل الفريم العالي (البيئة والسياق والهدف):
- الاتجاه العام: هل السوق صاعد (Bullish)، هابط (Bearish)، أم متذبذب (Ranging)؟
- الموقع من نطاق التعامل (Dealing Range): هل السعر في منطقة قسط (Premium) أم منطقة خصم (Discount)؟
- أقرب سيولة رئيسية مستهدفة: سيولة الشراء (BSL فوق قمم واضحة) أو سيولة البيع (SSL تحت قيعان واضحة).
- التفاعل مع مناطق الاهتمام الكبرى (HTF POI): مثل Order Block، Breaker Block، FVG، أو القمم والقيعان اليومية/الأسبوعية الكبرى.
- تقييم الحركة الحالية: هل هي حركة اندفاعية (Displacement) أم تصحيحية؟

2) تحليل الفريم المتوسط (منطقة الفرصة POI):
- التحقق من سحب السيولة (Sweep) لـ BSL أو SSL أولاً.
- التحقق من وجود Displacement حقيقي (اندفاع قوي بشموع كاملة) سبّب كسر الهيكل (BOS) أو تغير الشخصية (CHOCH) على فريم 15M/30M/1H.
- الكسر الحقيقي يجب أن يكون بإغلاق جسم الشمعة وليس بمجرد ذيل الشمعة (Wick).
- تصفية مناطق الاهتمام: لا تدخل من أي OB أو FVG عشوائي. يجب أن يكون الـ OB أو الـ FVG متشكلاً بعد سحب سيولة قوي ومصحوباً باندفاع قوي (Displacement) وغير ممتص (Unmitigated).

3) تحليل الفريم الصغير (التريغر والدخول):
- فريم 5M/1M يُستخدم للتريغر فقط وليس لتحديد الاتجاه العام.
- للدخول شراء (BUY):
  * الفريم العالي صاعد أو السعر متفاعل مع منطقة خصم (Discount) كبرى.
  * حدوث سحب سيولة (SSL sweep) تحت قاع واضح.
  * ظهور اندفاع (displacement) صاعد قوي.
  * حدوث كسر هيكل صاعد (CHOCH/MSS) بجسم الشمعة على فريم 5m/1m.
  * تراجع السعر (Retracement) إلى منطقة FVG أو OB سبّبت الحركة. الستوب تحت قاع السحب.
- للدخول بيع (SELL):
  * الفريم العالي هابط أو السعر متفاعل مع منطقة قسط (Premium) كبرى.
  * حدوث سحب سيولة (BSL sweep) فوق قمة واضحة.
  * ظهور اندفاع (displacement) هابط قوي.
  * حدوث كسر هيكل هابط (CHOCH/MSS) بجسم الشمعة على فريم 5m/1m.
  * تراجع السعر (Retracement) إلى منطقة FVG أو OB سبّبت الحركة. الستوب فوق قمة السحب.

4) قاعدة تصفية صارمة بخصوص سحب السيولة (Sweep vs Acceptance):
- سحب السيولة (Sweep) بذيل فقط ثم حدوث displacement عكسي يمثل فرصة انعكاس.
- أما الإغلاق القوي والاستقرار فوق/تحت السيولة (Acceptance) فهو استمرار للاتجاه (Continuation)، ولا يجب التداول ضده بل مع الاتجاه.

5) إدارة المخاطر وتجنب التداول (Risk & No-Trade Conditions):
- لا تلاحق السعر أبداً، انتظر الرجوع المنطقي (Retracement).
- الستوب يجب أن يوضع عند نقطة إبطال التحليل (Low/High السحب).
- نسبة العائد إلى المخاطرة (Risk-to-Reward) للهدف الأول (TP1) يجب ألا تقل عن 1:2.
- إذا كانت المعطيات غامضة، أو لا يوجد سحب سيولة واضح، أو لا يوجد displacement مقنع، أو كان السعر في منتصف النطاق، أو كان الـ R:R أقل من 1:2، يجب اختيار القرار WAIT فوراً وبدون تردد.

صيغة JSON المطلوبة فقط ولا ترجع أي شيء آخر:
{
  "decision": "BUY|SELL|WAIT",
  "confidence": 0,
  "rationale": "اكتب تحليلاً ملخصاً وجواباً للأسئلة الثمانية: السياق والاتجاه، السيولة التي سُحبت وهل كانت wick أم acceptance، منطقة الاهتمام POI، تأكيدات الكسر والاندفاع، منطقة الدخول والارتداد، مستوى الإبطال، الهدف القادم، ونسبة العائد للمخاطرة (بأقصى حد سطرين أو ثلاثة).",
  "stop_loss": null,
  "take_profit_1": null,
  "take_profit_2": null
}
"""


def _extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    inline = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
    if inline:
        return inline.group(1).strip()

    return None


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_confidence(value: Any, default: int = 60) -> int:
    raw = _to_float_or_none(value)
    if raw is None:
        return default

    # Model may return confidence in 0..1 or 0..10 scales.
    if 0.0 <= raw <= 1.0:
        raw *= 100.0
    elif 0.0 <= raw <= 10.0:
        raw *= 10.0

    conf = int(round(raw))
    return max(0, min(100, conf))


def _decision_to_side(decision: str) -> Optional[str]:
    d = (decision or "").upper()
    if d == "BUY":
        return "LONG"
    if d == "SELL":
        return "SHORT"
    return None


def _is_valid_levels(decision: str, entry: float, stop_loss: Optional[float], tp1: Optional[float]) -> bool:
    if stop_loss is None or tp1 is None or entry <= 0:
        return False

    if decision == "BUY":
        return stop_loss < entry < tp1
    if decision == "SELL":
        return tp1 < entry < stop_loss
    return False


def _rr_tp1(decision: str, entry: float, stop_loss: Optional[float], tp1: Optional[float]) -> Optional[float]:
    if not _is_valid_levels(decision, entry, stop_loss, tp1):
        return None

    if decision == "BUY":
        risk = entry - float(stop_loss)
        reward = float(tp1) - entry
    else:
        risk = float(stop_loss) - entry
        reward = entry - float(tp1)

    if risk <= 0:
        return None
    return reward / risk


def _fallback_wait_opinion(reason: str, confidence: int = 45) -> Dict[str, Any]:
    return {
        "decision": "WAIT",
        "confidence": max(0, min(100, int(confidence))),
        "rationale": str(reason).strip()[:280] or "تعذر تكوين رأي الخبير حالياً، الانتظار أفضل.",
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
    }


def _fallback_to_strategy_levels(
    decision: str,
    stop_loss: Optional[float],
    tp1: Optional[float],
    tp2: Optional[float],
    risk_levels: Optional[RiskLevels],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not risk_levels:
        return stop_loss, tp1, tp2

    fallback_sl = _to_float_or_none(risk_levels.stop_loss)
    fallback_tp1 = _to_float_or_none(risk_levels.take_profit_1)
    fallback_tp2 = _to_float_or_none(risk_levels.take_profit_2)

    sl = stop_loss if stop_loss is not None else fallback_sl
    first_tp = tp1 if tp1 is not None else fallback_tp1
    second_tp = tp2 if tp2 is not None else fallback_tp2

    # If provided levels are directionally invalid, trust strategy levels.
    if not _is_valid_levels(decision, _to_float_or_none(risk_levels.entry) or 0.0, sl, first_tp):
        sl = fallback_sl
        first_tp = fallback_tp1
        second_tp = fallback_tp2

    return sl, first_tp, second_tp


def get_expert_trade_opinion(
    symbol: str,
    side: str,
    current_price: float,
    trend: str,
    technical_score: float,
    timing_score: float,
    total_score: float,
    threshold: float,
    reasons: List[str],
    timing_reasons: List[str],
    vpa_info: Optional[Dict[str, Any]],
    risk_levels: Optional[RiskLevels],
    extra_context: Optional[Dict[str, Any]],
    openai_config: OpenAIConfig,
    timeout_seconds: int = 45,
) -> Optional[Dict[str, Any]]:
    """
    Generate an expert recommendation from full strategy context via OpenAI.

    Returns a normalized dictionary:
    {
      decision: BUY|SELL|WAIT,
      confidence: int,
      rationale: str,
      stop_loss: float|None,
      take_profit_1: float|None,
      take_profit_2: float|None
    }
    """
    if not openai_config.enabled or not openai_config.api_key:
        logger.debug("Expert advisor skipped: OpenAI disabled or no API key")
        return None

    logger.info(f"Expert advisor: calling OpenAI for {symbol} {side} (score={total_score:.1f}/{threshold:.1f})")

    context: Dict[str, Any] = {
        "symbol": symbol,
        "candidate_side": side,
        "current_price": current_price,
        "trend": trend,
        "technical_score": technical_score,
        "timing_score": timing_score,
        "total_score": total_score,
        "threshold": threshold,
        "technical_reasons": (reasons or [])[:10],
        "timing_reasons": (timing_reasons or [])[:10],
        "vpa": {
            "score": (vpa_info or {}).get("score"),
            "signal": (vpa_info or {}).get("vpa_signal"),
            "vwap": (vpa_info or {}).get("vwap"),
            "cmf": (vpa_info or {}).get("cmf"),
            "reasons": ((vpa_info or {}).get("reasons") or [])[:6],
        },
        "strategy_levels": {
            "entry": risk_levels.entry if risk_levels else None,
            "stop_loss": risk_levels.stop_loss if risk_levels else None,
            "take_profit_1": risk_levels.take_profit_1 if risk_levels else None,
            "take_profit_2": risk_levels.take_profit_2 if risk_levels else None,
        },
    }
    if extra_context:
        context["extra_context"] = extra_context

    try:
        client = OpenAI(
            api_key=openai_config.api_key,
            base_url=openai_config.base_url if hasattr(openai_config, "base_url") else None,
            timeout=max(15, int(timeout_seconds)),
        )
        response = client.chat.completions.create(
            model=openai_config.model,
            messages=[
                {"role": "system", "content": ADVISOR_PROMPT},
                {"role": "user", "content": f"بيانات السوق والاستراتيجية (JSON):\n{json.dumps(context, ensure_ascii=False, indent=2)}"},
            ],
            temperature=0.1,
            max_tokens=280,
            response_format={"type": "json_object"},
        )
        raw_text = (response.choices[0].message.content or "").strip()
        json_block = _extract_json_block(raw_text)
        if not json_block:
            logger.debug("Expert advisor returned non-JSON response")
            return _fallback_wait_opinion("تعذر قراءة رد الخبير حالياً، والانتظار أفضل.")

        parsed = json.loads(json_block)

        decision_raw = str(parsed.get("decision", "WAIT")).strip().upper()
        decision = decision_raw if decision_raw in {"BUY", "SELL", "WAIT"} else "WAIT"

        confidence_int = _normalize_confidence(parsed.get("confidence", 60), default=60)

        rationale = str(parsed.get("rationale", "")).strip()[:280]

        stop_loss = _to_float_or_none(parsed.get("stop_loss"))
        tp1 = _to_float_or_none(parsed.get("take_profit_1"))
        tp2 = _to_float_or_none(parsed.get("take_profit_2"))

        # Expert guardrails: only keep BUY/SELL when objective context is strong enough.
        entry = float(current_price) if current_price else 0.0
        edge = float(total_score) - float(threshold)
        vpa_score = _to_float_or_none((vpa_info or {}).get("score")) or 0.0
        trend_adx = _to_float_or_none((extra_context or {}).get("trend_adx")) or 0.0
        trend_rsi = _to_float_or_none((extra_context or {}).get("trend_rsi")) or 50.0
        candidate_side = (side or "").upper()

        if decision in {"BUY", "SELL"}:
            # Weak edge around threshold => no trade.
            if edge < 0.4:
                decision = "WAIT"
                confidence_int = min(confidence_int, 55)
                if not rationale:
                    rationale = "الأفضلية ضعيفة قرب الحد الأدنى، لذلك الانتظار أفضل."

            # Low trend strength (chop) => no trade.
            if decision in {"BUY", "SELL"} and trend_adx > 0 and trend_adx < 15:
                decision = "WAIT"
                confidence_int = min(confidence_int, 50)
                if not rationale:
                    rationale = "اتجاه ضعيف (ADX منخفض)؛ السوق أقرب للتذبذب."

            # Strong VPA contradiction => no trade.
            if decision == "BUY" and vpa_score <= -2.5:
                decision = "WAIT"
                confidence_int = min(confidence_int, 50)
                if not rationale:
                    rationale = "تعارض واضح بين قرار الشراء وقراءة VPA."
            if decision == "SELL" and vpa_score >= 2.5:
                decision = "WAIT"
                confidence_int = min(confidence_int, 50)
                if not rationale:
                    rationale = "تعارض واضح بين قرار البيع وقراءة VPA."

            # Avoid chasing extremes on trend RSI.
            if decision == "BUY" and trend_rsi >= 75:
                decision = "WAIT"
                confidence_int = min(confidence_int, 50)
                if not rationale:
                    rationale = "السعر ممتد صعوديًا (RSI مرتفع)؛ الانتظار أكثر أمانًا."
            if decision == "SELL" and trend_rsi <= 25:
                decision = "WAIT"
                confidence_int = min(confidence_int, 50)
                if not rationale:
                    rationale = "السعر ممتد هبوطًا (RSI منخفض)؛ الانتظار أكثر أمانًا."

            # Contradicting candidate side needs very high confidence.
            decision_side = _decision_to_side(decision)
            if decision_side and candidate_side and decision_side != candidate_side and confidence_int < 75:
                decision = "WAIT"
                confidence_int = min(confidence_int, 55)
                if not rationale:
                    rationale = "مخالفة الاتجاه المرشح بدون ثقة عالية، لذا الانتظار."

        if decision in {"BUY", "SELL"}:
            stop_loss, tp1, tp2 = _fallback_to_strategy_levels(
                decision=decision,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                risk_levels=risk_levels,
            )

            rr = _rr_tp1(decision, entry, stop_loss, tp1)
            if rr is None or rr < 1.0:
                decision = "WAIT"
                confidence_int = min(confidence_int, 55)
                stop_loss, tp1, tp2 = None, None, None
                if not rationale:
                    rationale = "نسبة العائد إلى المخاطرة غير كافية، لا توجد صفقة جيدة."
        else:
            stop_loss, tp1, tp2 = None, None, None

        logger.info(f"Expert advisor result: decision={decision} confidence={confidence_int}% [{symbol}]")
        return {
            "decision": decision,
            "confidence": confidence_int,
            "rationale": rationale,
            "stop_loss": stop_loss,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
        }

    except OpenAIError as e:
        logger.warning(f"Expert advisor OpenAI error for {symbol}: {e}")
        return _fallback_wait_opinion("تعذر الوصول إلى الخبير حالياً، لذلك القرار انتظار.")
    except Exception as e:
        logger.warning(f"Expert advisor failed for {symbol}: {e}")
        return _fallback_wait_opinion("حصل خلل مؤقت أثناء تحليل الخبير، لذلك القرار انتظار.")

MANAGER_PROMPT = """You are a professional institutional-style Smart Money Concepts (SMC) analyst.
Your job is to evaluate the provided multi-timeframe market data and generate high-quality trade plans based strictly on liquidity sweeps, displacement, and structural shifts.

Analysis Priority & Rules:
1. Higher Timeframe Context (Bias & Dealing Range):
   - Identify if HTF (Daily/4H/1H) is bullish, bearish, or ranging.
   - Determine if price is in Premium (for shorts) or Discount (for longs) of the current Dealing Range.
   - Locate major target liquidity (BSL above peaks, SSL below lows) and key HTF zones (OB, Breaker, FVG, key Highs/Lows).

2. Medium Timeframe Context (POI & Zones):
   - Locate the key POI (OB, FVG, Breaker) that triggered the move.
   - A valid POI must have caused structural shifts (BOS/CHOCH with candle body close) and strong displacement.

3. Liquidity Sweep vs. Acceptance Rule:
   - Identify if the liquidity event is a sweep (wick rejection + displacement in opposite direction) or acceptance (body close and retest, signifying trend continuation). Only trade reversals on clean sweeps with confirmation.

4. Lower Timeframe Confirmation (Trigger):
   - Confirm entry on LTF (5M/1M) via displacement and a CHOCH/MSS shift after a sweep of local BSL/SSL.
   - Entry must be from a logical retracement into FVG/OB. Risk-to-reward ratio to target must be at least 1:2.
   - Stop Loss must be placed strictly at the invalidation level (high/low of the sweep).

5. Verdict:
   - If market structure is messy, unclear, or if there is no liquidity sweep/displacement confirmation, or if R:R < 1:2, you MUST return "NO TRADE".

You must return the result in STRICT JSON only. No markdown, no code fences.
Use this JSON structure:
{
  "HTF_Analysis": "Brief HTF trend, premium/discount status, and target liquidity.",
  "LTF_Analysis": "Brief LTF structure, sweep confirmation, and displacement details.",
  "Verdict": "BUY" | "SELL" | "NO TRADE",
  "Reaction_Level": <number or null>,
  "Confirmation_Level": <number or null>,
  "Invalidation_Level": <number or null>,
  "Range_High": <number or null>,
  "Range_Low": <number or null>,
  "Zone_Quality": "strong" | "weak" | "overtested" | "N/A"
}
"""

def get_manager_opinion(
    symbol: str,
    current_price: float,
    trend: str,
    extra_context: Optional[Dict[str, Any]],
    risk_levels: Optional[RiskLevels],
    openai_config: OpenAIConfig,
    timeout_seconds: int = 45,
) -> Optional[Dict[str, Any]]:
    if not openai_config.enabled or not openai_config.api_key:
        return None

    context: Dict[str, Any] = {
        "symbol": symbol,
        "current_price": current_price,
        "HTF_Trend": trend,
    }
    if extra_context:
        context["market_data"] = extra_context

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=openai_config.api_key,
            base_url=openai_config.base_url if hasattr(openai_config, "base_url") else None,
            timeout=max(15, int(timeout_seconds)),
        )
        response = client.chat.completions.create(
            model=openai_config.model,
            messages=[
                {"role": "system", "content": MANAGER_PROMPT},
                {"role": "user", "content": f"Market Data:\n{json.dumps(context, indent=2)}"}
            ],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        raw_text = (response.choices[0].message.content or "").strip()
        json_block = _extract_json_block(raw_text)
        if not json_block:
            return None
        return json.loads(json_block)
    except Exception as e:
        logger.debug(f"Manager advisor failed: {e}")
        return None
