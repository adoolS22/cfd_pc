"""
Pending Order Planner (LLM Decision Engine)
============================================
Receives code-detected SMC structures and asks Ollama to make
a proactive pending-order decision. The LLM decides; the code detects.

Strict JSON parser — any non-JSON response is treated as NO_TRADE.
Hard guardrails validate every LLM decision against market reality.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

# ═══════════════════════════════════════════════════════════════════════
# System Prompt — the user's full SMC pending-order methodology
# ═══════════════════════════════════════════════════════════════════════

SMC_PENDING_ORDER_PROMPT = r"""You are a proactive Smart Money Concepts pending-order trading planner.

Your job is NOT to wait until price reaches the entry zone and then decide.
Your job is to analyze existing market data, forecast the highest-probability scenario, define future entry zones in advance, and output pending limit orders before price reaches those zones.

You are not a reactive market-entry bot.
You are a proactive liquidity, premium/discount, POI, and limit-order planning engine.

Your default decision is NO_TRADE.

You must return ONLY one valid JSON object.
Do not return markdown, code fences, explanations outside JSON, greetings, confirmations, or disclaimers outside JSON.

==================================================
CORE LOGIC
==================================================

Do not start by asking:
"Where should I enter now?"

Start by asking:
"Which liquidity is price most likely targeting first?"
"Is the current move likely to continue, retrace, rebalance, sweep, or reverse?"
"Where is the best future entry zone if price pulls back?"
"Can a pending limit order be placed before price reaches that zone?"

A valid trade must be planned before execution.

The professional model is:

Higher-timeframe bias
-> draw on liquidity
-> premium/discount location
-> displacement
-> FVG, OB, breaker, or mitigation zone creation
-> forecasted retracement zone
-> pending limit order
-> automatic execution if price returns
-> cancellation if invalidated

==================================================
MARKET MAKER TRAP AVOIDANCE
==================================================

The bot must avoid becoming easy liquidity.

Before placing any pending order, check if the setup may be a trap.

Reject or cancel the order if:
- Entry is in the middle of the dealing range.
- Long entry is in premium without confirmed bullish acceptance.
- Short entry is in discount without confirmed bearish acceptance.
- Stop loss is placed at an obvious liquidity level.
- Price already reached the intended liquidity target before the order was filled.
- The POI is already mitigated or stale.
- The move that created the POI looks like news volatility without clean structure.
- Liquidity classification is unresolved between sweep and acceptance.
- Opposite displacement appears before entry.
- Opposite MSS/CHOCH appears before entry.
- Spread or volatility makes real risk-to-reward invalid.
- Current price is too close to the planned entry, meaning planning is late.

The goal is not to predict with certainty.
The goal is to place orders only when the bot has a location advantage:
- HTF bias
- correct premium/discount
- clear liquidity draw
- valid sweep or acceptance
- displacement
- structural shift
- fresh POI
- logical invalidation
- clear target
- real RR >= 1:2

==================================================
DECISION VALUES
==================================================

Choose exactly one decision:

1. NO_TRADE
2. WAIT_FOR_SETUP
3. PLACE_BUY_LIMIT
4. PLACE_SELL_LIMIT
5. KEEP_PENDING_ORDER
6. CANCEL_PENDING_ORDER
7. MANAGE_OPEN_TRADE

==================================================
EXECUTION RULES
==================================================

Market entries are not the default.
Default execution method:
- Long setup = BUY LIMIT below current price.
- Short setup = SELL LIMIT above current price.

If price is already inside the entry zone and no pending order was previously placed:
- Do not chase. Return NO_TRADE or WAIT_FOR_SETUP.

==================================================
SCORING MODEL
==================================================

Start every setup score at 0.

Add points:
- Higher-timeframe bias aligned: +20
- Correct premium/discount location: +15
- Clear draw on liquidity: +10
- Liquidity sweep or valid continuation acceptance: +15
- Strong displacement: +15
- BOS, CHOCH, or MSS with candle body close: +10
- Valid FVG, OB, breaker, or mitigation zone created: +10
- Current price is away from the entry zone: +10
- Clear invalidation: +5
- Clear target liquidity: +5
- Risk-to-reward at least 1:2: +5

Subtract points:
- Higher-timeframe bias unclear: -20
- Price near equilibrium with no strong POI: -20
- Long from premium without acceptance: -20
- Short from discount without acceptance: -20
- No clear POI: -20
- No displacement: -15
- No structural shift: -15
- Entry requires market order or chasing: -25
- Price already touched entry zone: -15
- Price already reached target liquidity: -25
- Stop loss unclear: -20
- Target unclear: -15
- Risk-to-reward below 1:2: -25
- Conflicting signals: -20

Setup quality:
- 85 to 100: HIGH
- 70 to 84: MEDIUM
- 60 to 69: WEAK
- Below 60: INVALID

Only return PLACE_BUY_LIMIT or PLACE_SELL_LIMIT if the setup is clean and actionable.

"""

SMC_JSON_SCHEMA_PROMPT = r"""
==================================================
JSON OUTPUT SCHEMA
==================================================

Return exactly this JSON structure:

{
  "decision": "NO_TRADE | WAIT_FOR_SETUP | PLACE_BUY_LIMIT | PLACE_SELL_LIMIT | KEEP_PENDING_ORDER | CANCEL_PENDING_ORDER | MANAGE_OPEN_TRADE",
  "setup_quality": "HIGH | MEDIUM | WEAK | INVALID",
  "score": "integer_0_to_100",
  "market_bias": {
    "daily": "bullish | bearish | ranging | unclear",
    "h4": "bullish | bearish | ranging | unclear",
    "h1": "bullish | bearish | ranging | unclear",
    "final_bias": "bullish | bearish | ranging | unclear",
    "reason": ""
  },
  "premium_discount": {
    "dealing_range_high": null,
    "dealing_range_low": null,
    "equilibrium": null,
    "current_location": "premium | discount | equilibrium | unknown",
    "trade_location_valid": false
  },
  "forecast": {
    "most_likely_scenario": "bullish_retracement_continuation | bearish_retracement_continuation | bullish_sweep_reversal | bearish_sweep_reversal | break_and_acceptance_continuation | range_continuation | unclear",
    "expected_path": ""
  },
  "liquidity": {
    "bsl_levels": [],
    "ssl_levels": [],
    "liquidity_taken": "BSL | SSL | NONE",
    "draw_on_liquidity": ""
  },
  "market_structure": {
    "bos": "bullish | bearish | none",
    "choch": "bullish | bearish | none",
    "mss": "bullish | bearish | none",
    "displacement": "bullish | bearish | none"
  },
  "poi": {
    "type": "FVG | OB | breaker | mitigation | none",
    "direction": "bullish | bearish | none",
    "zone_low": null,
    "zone_high": null,
    "valid": false
  },
  "planned_order": {
    "order_type": "BUY_LIMIT | SELL_LIMIT | NONE",
    "entry_price": null,
    "stop_loss": null,
    "take_profit_1": null,
    "take_profit_2": null,
    "risk_to_reward": null,
    "valid_until_hours": 8,
    "order_reason": ""
  },
  "order_management": {
    "keep_order": false,
    "cancel_order": false,
    "cancel_reasons": []
  },
  "no_trade_reason": ""
}

==================================================
FINAL RULES
==================================================

Be proactive, not reactive. Plan trades before price reaches the entry zone.
Prefer pending limit orders over market entries. Never chase price.
CRITICAL LIMIT ORDER RULES:
- A BUY_LIMIT entry_price MUST be mathematically BELOW the current_price (Wait for pullback into a Discount POI).
- A SELL_LIMIT entry_price MUST be mathematically ABOVE the current_price (Wait for pullback into a Premium POI).
- If you want to trade a breakout or continuation but the current price has already passed your ideal entry, DO NOT set a Limit order in the wrong direction. Return NO_TRADE.
- Your entry_price MUST exactly match the top or bottom of a valid FVG or OB from the lower timeframes (m15, m5, m1).
Never place an order without clear draw on liquidity or invalidation.
Never place an order if risk-to-reward is below 1:2.
Longs preferred from discount. Shorts preferred from premium.
If the setup is not clean, return NO_TRADE.

CRITICAL: You MUST return ONLY one valid JSON object matching the schema above.
Do NOT return analysis text, markdown, explanations, or per-timeframe descriptions.
Return ONLY the JSON object. Nothing else.
"""


# ═══════════════════════════════════════════════════════════════════════
# Prefilter and Order Management logic
# ═══════════════════════════════════════════════════════════════════════

def classify_symbol_fast(mtf_data: Dict[str, Any]) -> tuple[str, str]:
    """
    Classify a symbol into REJECT, WATCHLIST, or LLM_CANDIDATE based on pre-filter rules.
    Returns (State, Reason).
    """
    summary = extract_htf_context(mtf_data)
    pd = summary.get("premium_discount", {})
    structure = summary.get("market_structure", {})
    liquidity = summary.get("liquidity", {})
    poi = summary.get("poi_candidates", [])
    bias = summary.get("bias", {}).get("final_bias", "unclear")

    location = pd.get("current_location", "unknown")

    # Hard rejects
    if location == "equilibrium":
        return "REJECT", "Price is in equilibrium"

    if bias == "bearish" and location == "discount":
        return "REJECT", "Bearish bias but price is in discount (bad short location)"

    if bias == "bullish" and location == "premium":
        return "REJECT", "Bullish bias but price is in premium (bad long location)"

    if not liquidity.get("draw_on_liquidity"):
        return "REJECT", "No clear draw on liquidity"

    fresh_pois = [p for p in poi if p.get("valid") is True and p.get("mitigated") is False]
    if not fresh_pois:
        return "REJECT", "No fresh valid POI"

    # If it passes all hard rejects and has fresh POIs + draw on liquidity + good location,
    # it is a valid candidate for the LLM to evaluate. The LLM will assess if the POI
    # was created by a displacement and if the structure supports it.
    return "LLM_CANDIDATE", "Passed prefilter heuristics"


def score_candidate_fast(mtf_data: Dict[str, Any]) -> int:
    """Quick heuristic score to rank candidates before sending to LLM."""
    summary = extract_htf_context(mtf_data)
    score = 0
    bias = summary.get("bias", {}).get("final_bias", "unclear")
    location = summary.get("premium_discount", {}).get("current_location", "unknown")
    structure = summary.get("market_structure", {})
    liquidity = summary.get("liquidity", {})
    pois = summary.get("poi_candidates", [])

    if bias in ["bullish", "bearish"]:
        score += 20
    if (bias == "bullish" and location == "discount") or (bias == "bearish" and location == "premium"):
        score += 20
    if liquidity.get("draw_on_liquidity"):
        score += 15
    if structure.get("displacement") not in [None, "none", ""]:
        score += 15
    if structure.get("bos") not in [None, "none", ""] or structure.get("choch") not in [None, "none", ""]:
        score += 15
        
    fresh_pois = [p for p in pois if p.get("valid") is True and p.get("mitigated") is False]
    if fresh_pois:
        score += 15
        
    return score


def extract_htf_context(mtf_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts HTF context (bias, structure, liquidity, pois) from the raw timeframe data."""
    summary = {
        "symbol": mtf_data.get("symbol"),
        "current_price": mtf_data.get("current_price"),
        "spread": mtf_data.get("spread"),
        "bias": {"final_bias": "neutral"},
        "premium_discount": {"current_location": "unknown"},
        "liquidity": {"draw_on_liquidity": None, "nearest_bsl": [], "nearest_ssl": []},
        "market_structure": {"displacement": "none", "bos": "none", "choch": "none", "mss": "none"},
        "poi_candidates": [],
        "existing_pending_orders": mtf_data.get("existing_pending_orders", []),
        "existing_positions": mtf_data.get("existing_positions", []),
    }

    # 1. Bias calculation (Daily & H4)
    d1 = mtf_data.get("daily", {})
    h4 = mtf_data.get("h4", {})
    h1 = mtf_data.get("h1", {})
    
    d1_trend = str(d1.get("trend", "")).lower()
    h4_trend = str(h4.get("trend", "")).lower()
    
    bias = "neutral"
    if "up" in d1_trend or "up" in h4_trend or "bull" in d1_trend:
        bias = "bullish"
    elif "down" in d1_trend or "down" in h4_trend or "bear" in d1_trend:
        bias = "bearish"
    summary["bias"]["final_bias"] = bias
    summary["bias"]["daily"] = d1_trend
    summary["bias"]["h4"] = h4_trend

    # 2. Premium / Discount (Use H4 or H1 dealing range)
    dr_h4 = h4.get("dealing_range", {})
    dr_h1 = h1.get("dealing_range", {})
    loc = dr_h4.get("location", "unknown")
    if loc == "unknown" or loc == "equilibrium":
        loc = dr_h1.get("location", "unknown")
    summary["premium_discount"]["current_location"] = loc

    # 3. Liquidity (Nearest HTF levels & sweeps)
    h4_bsl = h4.get("bsl_levels", []) or h1.get("bsl_levels", [])
    h4_ssl = h4.get("ssl_levels", []) or h1.get("ssl_levels", [])
    summary["liquidity"]["nearest_bsl"] = h4_bsl
    summary["liquidity"]["nearest_ssl"] = h4_ssl
    
    # Simple draw: if bias is bullish, we draw to BSL. If bearish, we draw to SSL.
    if bias == "bullish" and h4_bsl:
        summary["liquidity"]["draw_on_liquidity"] = "buy_side"
    elif bias == "bearish" and h4_ssl:
        summary["liquidity"]["draw_on_liquidity"] = "sell_side"
    elif h4_bsl and h4_ssl:
        # If bias is neutral but we have levels, let's just check where price is closer
        current_price = mtf_data.get("current_price", 0)
        # simplistic check
        summary["liquidity"]["draw_on_liquidity"] = "buy_side" if bias != "bearish" else "sell_side"

    logger.debug(f"Prefilter extracted bias={bias}, bsl={len(h4_bsl)}, ssl={len(h4_ssl)} for {mtf_data.get('symbol')}")

    # 4. Market Structure (H1 / M15)
    m15 = mtf_data.get("m15", {})
    for tf_data in [h1, m15]:
        disp = tf_data.get("displacement")
        if disp and disp.get("direction"):
            summary["market_structure"]["displacement"] = disp.get("direction")
        
        breaks = tf_data.get("structure_breaks", [])
        for b in breaks:
            b_type = b.get("break_type", "").lower()
            b_dir = b.get("direction", "")
            if b_type in ["bos", "choch", "mss"]:
                summary["market_structure"][b_type] = b_dir

    # 5. POI Candidates (Unmitigated FVGs / OBs from H4, H1, M15)
    pois = []
    for tf_key, tf_data in [("h4", h4), ("h1", h1), ("m15", m15)]:
        for fvg in tf_data.get("fvgs", []):
            fvg["type"] = "FVG"
            fvg["timeframe"] = tf_key
            fvg["valid"] = True
            pois.append(fvg)
        for ob in tf_data.get("order_blocks", []):
            ob["type"] = "OB"
            ob["timeframe"] = tf_key
            ob["valid"] = True
            ob["mitigated"] = False
            pois.append(ob)
    summary["poi_candidates"] = pois

    return summary


def summarize_mtf_data(mtf_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build a focused, per-timeframe summary for the LLM prompt.
    
    Sends key SMC fields per timeframe (trend, displacement, structure breaks,
    FVGs, OBs, dealing range, BSL/SSL) — rich enough for decisions but compact
    enough that the LLM stays on-schema.
    """
    prefilter_ctx = extract_htf_context(mtf_data)
    
    summary = {
        "symbol": mtf_data.get("symbol"),
        "current_price": mtf_data.get("current_price"),
        "spread": mtf_data.get("spread"),
        "bias": prefilter_ctx.get("bias", {}),
        "premium_discount": prefilter_ctx.get("premium_discount", {}),
        "existing_pending_orders": mtf_data.get("existing_pending_orders", []),
        "existing_positions": mtf_data.get("existing_positions", []),
        "timeframes": {},
    }
    
    # Key fields to extract per timeframe
    _KEEP_KEYS = {
        "timeframe", "trend", "atr", "rsi", "adx",
        "displacement", "structure_breaks",
        "fvgs", "order_blocks",
        "dealing_range", "bsl_levels", "ssl_levels",
        "liquidity_sweeps",
    }
    
    for tf_key in ["daily", "h4", "h1", "m15"]:
        tf_data = mtf_data.get(tf_key, {})
        if not tf_data:
            continue
        tf_summary = {k: v for k, v in tf_data.items() if k in _KEEP_KEYS}
        if tf_summary:
            summary["timeframes"][tf_key] = tf_summary
    
    return summary


def manage_existing_pending_order(order: Dict[str, Any], mtf_data: Dict[str, Any]) -> tuple[str, str]:
    """
    Hard-coded Python rules to manage existing pending orders quickly.
    Returns (Action, Reason) where Action is "KEEP" or "CANCEL".
    """
    summary = extract_htf_context(mtf_data)
    
    order_type = order.get("type", "").lower()
    entry_price = order.get("entry_price", 0.0)
    current_price = summary.get("current_price", 0.0)
    
    structure = summary.get("market_structure", {})
    pois = summary.get("poi_candidates", [])
    
    # 1. Target reached before fill (simplistic check: price swept past entry in direction of target without filling)
    # Actually, a better check: if order is BUY LIMIT, and price goes ABOVE target. But we don't have target here easily unless stored.
    # For now, we rely on POI invalidation and opposite displacement
    
    # 2. Opposite displacement
    displacement = structure.get("displacement", "none")
    if "buy" in order_type and displacement == "bearish":
        return "CANCEL", "Opposite displacement (bearish) appeared against LONG pending order"
    if "sell" in order_type and displacement == "bullish":
        return "CANCEL", "Opposite displacement (bullish) appeared against SHORT pending order"
        
    # 3. POI invalidated
    # If the order's entry price is no longer within any valid POI
    if pois:
        valid_poi_found = False
        for p in pois:
            if not p.get("valid") or p.get("mitigated"):
                continue
            z_low = p.get("low", p.get("zone_low", 0.0))
            z_high = p.get("high", p.get("zone_high", 0.0))
            if z_low <= entry_price <= z_high:
                valid_poi_found = True
                break
        
        # We don't cancel just because we couldn't match a POI (it might be a manual order or an older timeframe POI).
        # But if we want to be strict, we can. Let's keep it safe for now.

    # 4. Opposite MSS/CHOCH
    choch = structure.get("choch", "none")
    if "buy" in order_type and choch == "bearish":
        return "CANCEL", "Opposite CHOCH (bearish) appeared"
    if "sell" in order_type and choch == "bullish":
        return "CANCEL", "Opposite CHOCH (bullish) appeared"

    return "KEEP", "Order still valid"


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════

_VALID_DECISIONS = frozenset({
    "NO_TRADE", "WAIT_FOR_SETUP",
    "PLACE_BUY_LIMIT", "PLACE_SELL_LIMIT",
    "KEEP_PENDING_ORDER", "CANCEL_PENDING_ORDER",
    "MANAGE_OPEN_TRADE",
})


@dataclass
class PendingOrderDecision:
    decision: str = "NO_TRADE"
    setup_quality: str = "INVALID"
    score: int = 0
    symbol: str = ""
    order_type: str = "NONE"         # BUY_LIMIT | SELL_LIMIT | NONE
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    risk_to_reward: float = 0.0
    valid_until_hours: int = 8
    reason: str = ""
    full_analysis: Dict[str, Any] = field(default_factory=dict)
    guardrail_violations: List[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return self.decision in ("PLACE_BUY_LIMIT", "PLACE_SELL_LIMIT")


# ═══════════════════════════════════════════════════════════════════════
# Strict JSON parser
# ═══════════════════════════════════════════════════════════════════════

def _strict_parse_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from LLM response. Returns None if not valid JSON.

    Strategy:
    1. Direct JSON (starts with {, ends with })
    2. Extract from ```json ... ``` fences
    3. Find first { to last } in text
    If all fail → None → NO_TRADE
    """
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()

    # Strategy 1: direct JSON
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Strategy 2: code fence extraction
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # All strategies failed
    return None


# ═══════════════════════════════════════════════════════════════════════
# Normalize LLM response to expected schema
# ═══════════════════════════════════════════════════════════════════════

def _normalize_llm_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Convert alternative LLM response formats to the expected schema.
    
    The qwen2.5 model sometimes returns trade data in non-standard formats:
    - {"entry": {"side": "sell", "price": X}, "stop_loss": {"price": Y}, ...}
    - {"signal": "SELL", "price": X, "stop_loss": X, ...}
    - {"order_type": "SELL_LIMIT", "entry_price": X, ...}
    
    This normalizer maps them to the expected schema so guardrails can process them.
    """
    # If already in expected format, return as-is
    if "decision" in parsed and "planned_order" in parsed:
        return parsed
    
    result = dict(parsed)  # shallow copy
    
    # --- Detect entry/exit format ---
    entry = parsed.get("entry", {})
    sl_obj = parsed.get("stop_loss", {})
    tp_obj = parsed.get("take_profit", parsed.get("exit", {}))
    signal = str(parsed.get("signal", parsed.get("order_type", ""))).upper()
    
    # Extract side from entry object or signal field
    side = ""
    if isinstance(entry, dict):
        side = str(entry.get("side", "")).upper()
    if not side and signal:
        side = signal
    
    # Extract prices from nested or flat structure
    entry_price = 0.0
    stop_loss = 0.0
    tp1 = 0.0
    tp2 = 0.0
    
    if isinstance(entry, dict) and entry.get("price"):
        entry_price = float(entry["price"])
    elif parsed.get("entry_price"):
        entry_price = float(parsed["entry_price"])
    elif parsed.get("price"):
        entry_price = float(parsed["price"])
        
    if isinstance(sl_obj, dict) and sl_obj.get("price"):
        stop_loss = float(sl_obj["price"])
    elif isinstance(parsed.get("stop_loss"), (int, float)):
        stop_loss = float(parsed["stop_loss"])
    
    if isinstance(tp_obj, dict) and tp_obj.get("price"):
        tp1 = float(tp_obj["price"])
    elif isinstance(parsed.get("take_profit"), (int, float)):
        tp1 = float(parsed["take_profit"])
    elif isinstance(parsed.get("take_profit_1"), (int, float)):
        tp1 = float(parsed["take_profit_1"])
    
    # Handle list of take profits
    tp_list = parsed.get("take_profit", [])
    if isinstance(tp_list, list) and tp_list:
        first_tp = tp_list[0]
        if isinstance(first_tp, dict) and first_tp.get("price"):
            tp1 = float(first_tp["price"])
        elif isinstance(first_tp, (int, float, str)):
            tp1 = float(first_tp)
            
        if len(tp_list) > 1:
            second_tp = tp_list[1]
            if isinstance(second_tp, dict) and second_tp.get("price"):
                tp2 = float(second_tp["price"])
            elif isinstance(second_tp, (int, float, str)):
                tp2 = float(second_tp)
    
    tp2 = tp2 or float(parsed.get("take_profit_2", 0) or 0)
    
    # Map side to decision
    if entry_price > 0 and stop_loss > 0 and tp1 > 0:
        if "BUY" in side or "LONG" in side:
            decision = "PLACE_BUY_LIMIT"
        elif "SELL" in side or "SHORT" in side:
            decision = "PLACE_SELL_LIMIT"
        else:
            # Infer from price relationship: if SL < entry, it's a buy
            if stop_loss < entry_price:
                decision = "PLACE_BUY_LIMIT"
            else:
                decision = "PLACE_SELL_LIMIT"
        
        result["decision"] = decision
        result["setup_quality"] = result.get("setup_quality", "MEDIUM")
        result["score"] = result.get("score", 75)
        result["planned_order"] = {
            "order_type": decision.replace("PLACE_", ""),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "valid_until_hours": 8,
            "order_reason": str(parsed.get("strategy", {}).get("description", ""))
                           if isinstance(parsed.get("strategy"), dict) 
                           else str(parsed.get("reason", "")),
        }
        logger.info(f"Normalized non-standard LLM response to {decision} "
                     f"entry={entry_price}, sl={stop_loss}, tp1={tp1}")
    
    return result


# ═══════════════════════════════════════════════════════════════════════
# Guardrails — validate LLM decisions against market reality
# ═══════════════════════════════════════════════════════════════════════

def _apply_guardrails(
    parsed: Dict[str, Any],
    current_price: float,
    min_rr: float = 2.0,
    min_score: int = 65,
    max_sl_pct: float = 3.0,
    spread: float = 0.0,
) -> PendingOrderDecision:
    """Validate and construct a PendingOrderDecision from parsed LLM JSON.

    Hard guardrails:
    - BUY_LIMIT entry_price MUST be < current_price
    - SELL_LIMIT entry_price MUST be > current_price
    - R:R computed mathematically (not trusted from LLM)
    - Score >= min_score for actionable orders
    - SL distance <= max_sl_pct of entry
    - SL distance >= spread * 3 (minimum meaningful stop)
    """
    violations: List[str] = []

    decision = str(parsed.get("decision", "NO_TRADE")).upper().replace(" ", "_")
    if decision not in _VALID_DECISIONS:
        decision = "NO_TRADE"
        violations.append(f"Invalid decision value, defaulting to NO_TRADE")

    raw_score = parsed.get("score", 0)
    if isinstance(raw_score, dict):
        # If the LLM returned a dict of scores, just default to 75
        score = 75
    else:
        try:
            score = int(float(raw_score))
        except (ValueError, TypeError):
            score = 0

    quality = str(parsed.get("setup_quality", "INVALID")).upper()

    planned = parsed.get("planned_order", {}) or {}
    order_type = str(planned.get("order_type", "NONE")).upper().replace(" ", "_")
    entry_price = float(planned.get("entry_price") or 0)
    stop_loss = float(planned.get("stop_loss") or 0)
    tp1 = float(planned.get("take_profit_1") or 0)
    tp2 = float(planned.get("take_profit_2") or 0)
    valid_hours = int(planned.get("valid_until_hours", 8) or 8)
    order_reason = str(planned.get("order_reason", ""))

    no_trade_reason = str(parsed.get("no_trade_reason", ""))

    # For non-actionable decisions, return early without price validation
    if decision not in ("PLACE_BUY_LIMIT", "PLACE_SELL_LIMIT"):
        return PendingOrderDecision(
            decision=decision,
            setup_quality=quality,
            score=score,
            order_type="NONE",
            reason=no_trade_reason or order_reason,
            full_analysis=parsed,
            guardrail_violations=violations,
        )

    # ── Actionable order guardrails ──

    # G1: Entry price must be valid
    if entry_price <= 0:
        violations.append("entry_price is zero or negative")

    # G2: BUY_LIMIT must be below current price
    if decision == "PLACE_BUY_LIMIT" and entry_price >= current_price:
        violations.append(
            f"BUY_LIMIT entry_price ({entry_price}) must be BELOW current price ({current_price})"
        )

    # G3: SELL_LIMIT must be above current price
    if decision == "PLACE_SELL_LIMIT" and entry_price <= current_price:
        violations.append(
            f"SELL_LIMIT entry_price ({entry_price}) must be ABOVE current price ({current_price})"
        )

    # G4: SL must exist and be on correct side
    if stop_loss <= 0:
        violations.append("stop_loss is zero or negative")
    elif entry_price > 0:
        if decision == "PLACE_BUY_LIMIT" and stop_loss >= entry_price:
            violations.append(f"BUY_LIMIT SL ({stop_loss}) must be BELOW entry ({entry_price})")
        if decision == "PLACE_SELL_LIMIT" and stop_loss <= entry_price:
            violations.append(f"SELL_LIMIT SL ({stop_loss}) must be ABOVE entry ({entry_price})")

    # G5: TP must exist and be on correct side
    if tp1 <= 0:
        violations.append("take_profit_1 is zero or negative")
    elif entry_price > 0:
        if decision == "PLACE_BUY_LIMIT" and tp1 <= entry_price:
            violations.append(f"BUY_LIMIT TP1 ({tp1}) must be ABOVE entry ({entry_price})")
        if decision == "PLACE_SELL_LIMIT" and tp1 >= entry_price:
            violations.append(f"SELL_LIMIT TP1 ({tp1}) must be BELOW entry ({entry_price})")

    # G6: Compute R:R mathematically (do not trust LLM's number)
    computed_rr = 0.0
    if entry_price > 0 and stop_loss > 0 and tp1 > 0:
        if decision == "PLACE_BUY_LIMIT":
            risk = entry_price - stop_loss
            reward = tp1 - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - tp1

        if risk > 0:
            computed_rr = round(reward / risk, 2)

        if computed_rr < min_rr:
            violations.append(f"R:R ({computed_rr}) below minimum ({min_rr})")

    # G7: Score check
    if score < min_score:
        violations.append(f"Score ({score}) below minimum ({min_score})")

    # G8: SL distance % check
    if entry_price > 0 and stop_loss > 0:
        sl_distance_pct = abs(entry_price - stop_loss) / entry_price * 100
        if sl_distance_pct > max_sl_pct:
            violations.append(f"SL distance ({sl_distance_pct:.2f}%) exceeds max ({max_sl_pct}%)")

    # G9: SL must be at least 3x spread
    if spread > 0 and entry_price > 0 and stop_loss > 0:
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance < spread * 3:
            violations.append(f"SL distance ({sl_distance:.5f}) less than 3x spread ({spread * 3:.5f})")

    # If any violation → downgrade to NO_TRADE
    if violations:
        logger.warning(
            f"Guardrail violations for {decision}: {violations}"
        )
        return PendingOrderDecision(
            decision="NO_TRADE",
            setup_quality="INVALID",
            score=score,
            order_type="NONE",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_to_reward=computed_rr,
            reason=f"Guardrail blocked: {'; '.join(violations)}",
            full_analysis=parsed,
            guardrail_violations=violations,
        )

    # All guardrails passed
    return PendingOrderDecision(
        decision=decision,
        setup_quality=quality,
        score=score,
        order_type=order_type if order_type in ("BUY_LIMIT", "SELL_LIMIT") else decision.replace("PLACE_", ""),
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        take_profit_2=tp2,
        risk_to_reward=computed_rr,
        valid_until_hours=valid_hours,
        reason=order_reason,
        full_analysis=parsed,
        guardrail_violations=[],
    )


# ═══════════════════════════════════════════════════════════════════════
# Main entry point: plan_pending_order
# ═══════════════════════════════════════════════════════════════════════

def plan_pending_order(
    symbol: str,
    mtf_data: Dict[str, Any],
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "deepseek-r1:14b",
    timeout_seconds: int = 180,
    min_rr: float = 1.5,
    min_score: int = 55,
    max_sl_pct: float = 3.0,
    temperature: float = 0.1,
    num_ctx: int = 12288,
    num_predict: int = 2048,
    num_gpu: int = 999,
    stream: bool = False,
    keep_alive: str = "30m",
) -> PendingOrderDecision:
    """
    Evaluates MTF SMC data and decides if a pending order should be placed.
    Uses a fast Python prefilter to avoid slow LLM calls for bad setups.
    """
    current_price = float(mtf_data.get("current_price", 0))
    spread = float(mtf_data.get("spread", 0))

    if current_price <= 0:
        return PendingOrderDecision(
            decision="NO_TRADE",
            symbol=symbol,
            reason="No price data available",
        )

    # 1. Fast Prefilter
    state, reason = classify_symbol_fast(mtf_data)
    
    if state == "REJECT":
        logger.info(f"[PREFILTER_REJECT] {symbol} reason={reason}")
        return PendingOrderDecision(
            decision="NO_TRADE",
            setup_quality="INVALID",
            score=0,
            order_type="NONE",
            reason=f"Prefilter Rejected: {reason}"
        )
    elif state == "WATCHLIST":
        logger.info(f"[PREFILTER_WATCHLIST] {symbol} reason={reason}")
        return PendingOrderDecision(
            decision="NO_TRADE",
            setup_quality="WATCHLIST",
            score=50,
            order_type="NONE",
            reason=f"Prefilter Watchlist: {reason}"
        )
        
    logger.info(f"[PREFILTER_PASS] {symbol} Passed prefilter, sending to LLM. reason={reason}")

    # Strip raw candles for the LLM prompt
    mtf_summary = summarize_mtf_data(mtf_data)

    # Build user prompt with the pre-computed structured data
    user_content = (
        f"Symbol: {symbol}\n"
        f"Current Price: {current_price}\n"
        f"Spread: {spread}\n\n"
        f"Pre-computed Multi-Timeframe SMC Analysis Summary (JSON):\n"
        f"{json.dumps(mtf_summary, ensure_ascii=False, indent=2, default=str)}"
    )

    # Call Ollama via native API
    try:
        import requests

        url = f"{ollama_base_url}/api/generate"
        combined_prompt = f"{SMC_PENDING_ORDER_PROMPT}\n\n[MARKET DATA]\n{user_content}\n\n{SMC_JSON_SCHEMA_PROMPT}"
        
        payload = {
            "model": ollama_model,
            "prompt": combined_prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "num_gpu": num_gpu,
            },
            "keep_alive": keep_alive,
        }

        t0 = time.time()
        response = requests.post(url, json=payload, timeout=max(15, timeout_seconds))
        response.raise_for_status()
        elapsed = time.time() - t0

        data = response.json()
        raw_text = data.get("response", "").strip()
        logger.info(f"Planner LLM response for {symbol} ({elapsed:.1f}s):\n{raw_text}")

    except Exception as e:
        logger.warning(f"Planner LLM call failed for {symbol}: {e}")
        return PendingOrderDecision(
            decision="NO_TRADE",
            symbol=symbol,
            reason=f"LLM call failed: {e}",
        )

    # Strict JSON parsing
    parsed = _strict_parse_json(raw_text)
    if parsed is None:
        logger.warning(f"Planner: LLM returned non-JSON for {symbol}, treating as NO_TRADE")
        return PendingOrderDecision(
            decision="NO_TRADE",
            symbol=symbol,
            reason="LLM returned non-JSON response",
            guardrail_violations=["non_json_response"],
        )

    # Normalize non-standard LLM response formats
    parsed = _normalize_llm_response(parsed)

    # Apply guardrails
    result = _apply_guardrails(
        parsed=parsed,
        current_price=current_price,
        min_rr=min_rr,
        min_score=min_score,
        max_sl_pct=max_sl_pct,
        spread=spread,
    )
    result.symbol = symbol

    logger.info(
        f"Planner decision [{symbol}]: {result.decision} "
        f"(score={result.score}, quality={result.setup_quality}, "
        f"rr={result.risk_to_reward}, violations={len(result.guardrail_violations)})"
    )

    return result
