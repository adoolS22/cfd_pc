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

    # Removed faulty prefilter blocks for "bearish and discount" and "bullish and premium"
    # because a pending order planner *should* analyze these setups to place limits ahead of time.

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


# ═══════════════════════════════════════════════════════════════════════
# Code-first pricing: the code owns the numbers, the LLM only vetoes
# ═══════════════════════════════════════════════════════════════════════

SMC_VETO_PROMPT = """You are a strict SMC (Smart Money Concepts) trade reviewer.
The trading system has ALREADY computed a pending-order proposal deterministically
from multi-timeframe analysis (POI zones, liquidity targets, ATR-buffered stop).
Your ONLY job is to APPROVE or REJECT this exact proposal.
You must NOT propose different prices, levels, or order types.

Evaluate:
1. Does the HTF bias genuinely support the trade direction?
2. Is the POI high quality (created by displacement, aligned with structure)?
3. Is the draw on liquidity logical for the take-profit target?
4. Premium/Discount location. IMPORTANT SMC RULE:
   - A SELL (bearish) entry SHOULD be in PREMIUM (above equilibrium) — selling
     high is correct. Premium + bearish is GOOD, not a conflict.
   - A BUY (bullish) entry SHOULD be in DISCOUNT (below equilibrium) — buying
     low is correct. Discount + bullish is GOOD, not a conflict.
   Only flag a location problem if the entry is on the WRONG side (e.g. a SELL
   placed in deep discount, or a BUY placed in deep premium).
5. Are there conflicting signals across timeframes that make this setup unreliable?
6. If a PAST PERFORMANCE & LESSONS section is present: weigh it, and be more
   cautious when this setup closely resembles recent losing trades.

Respond with your VERDICT as JSON, exactly this schema:
{
  "decision": "APPROVE" or "REJECT",
  "score": 0-100,
  "setup_quality": "HIGH" or "MEDIUM" or "WEAK",
  "reason": "one short sentence"
}
"""

# Ollama structured-output schema: forces the model to emit exactly these keys.
VETO_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["APPROVE", "REJECT"]},
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "setup_quality": {"type": "string", "enum": ["HIGH", "MEDIUM", "WEAK"]},
        "reason": {"type": "string"},
    },
    "required": ["decision", "score", "setup_quality", "reason"],
}


# Appended at the very END of the prompt so it is the last thing the model
# reads before generating. Without this, format:json makes deepseek-r1 echo the
# proposal JSON it was given instead of producing the verdict.
SMC_VETO_REQUEST = """

[YOUR TASK]
Do NOT repeat or echo the proposed order above. Output ONLY your own verdict
about whether to take that trade, as a JSON object with EXACTLY these four keys:
{"decision": "APPROVE" or "REJECT", "score": 0-100, "setup_quality": "HIGH" or "MEDIUM" or "WEAK", "reason": "one short sentence"}
"""


def _evaluate_confluence(
    tf_data: Dict[str, Any],
    current_price: float,
    direction: str,
    min_disp_atr: float,
    min_disp_body: float,
) -> Optional[Dict[str, Any]]:
    """Validate the SMC confluence chain on one timeframe for a trade direction.

    Required sequence (enforced by bar index ordering):
      liquidity sweep (in trade direction) -> displacement (trade direction,
      after the sweep) -> body-close structure break (trade direction, after the
      sweep) -> a fresh FVG/OB created by that move, sitting in the retracement
      zone between the swept extreme and current price.

    direction: 'bullish' (long) or 'bearish' (short).
    Returns the validated setup (poi + swept extreme + evidence) or None.
    """
    prefix = "bull" if direction == "bullish" else "bear"
    sweeps = tf_data.get("liquidity_sweeps", []) or []
    breaks = tf_data.get("structure_breaks", []) or []
    fvgs = tf_data.get("fvgs", []) or []
    obs = tf_data.get("order_blocks", []) or []

    # Displacement list (fall back to the single 'displacement' for callers/tests
    # that don't provide the list).
    disps = tf_data.get("displacements")
    if disps is None:
        d1 = tf_data.get("displacement")
        disps = [d1] if d1 else []

    # 1) A strong displacement (impulse) in the trade direction. Use the most
    #    recent qualifying one — at a pullback entry the LATEST move is often the
    #    counter-bias retracement, so we must search the whole window, not just
    #    the last displacement.
    dir_disps = [
        d for d in disps
        if d and str(d.get("direction", "")) == direction
        and float(d.get("atr_multiple") or 0) >= min_disp_atr
        and float(d.get("body_ratio") or 0) >= min_disp_body
    ]
    if not dir_disps:
        return None
    disp = max(dir_disps, key=lambda d: int(d.get("end_index", 0)))
    disp_end = int(disp.get("end_index", -1))

    # 2) A liquidity sweep in the trade direction that PRECEDES the displacement.
    #    Bullish sweep took out the lows (SSL) -> fuel for a long; bearish sweep
    #    took out the highs (BSL) -> fuel for a short. Pick the most recent sweep
    #    at or before the displacement so a later wick doesn't void the setup.
    dir_sweeps = [
        s for s in sweeps
        if str(s.get("direction", "")) == direction
        and float(s.get("swept_price") or 0) > 0
        and int(s.get("index", 10**9)) <= disp_end
    ]
    if not dir_sweeps:
        return None
    sweep = max(dir_sweeps, key=lambda s: int(s.get("index", 0)))
    sweep_idx = int(sweep.get("index", 0))
    sweep_price = float(sweep["swept_price"])

    # 3) Body-close structure break (BOS/CHoCH) in the trade direction, after the sweep.
    dir_breaks = [
        b for b in breaks
        if str(b.get("direction", "")) == direction and int(b.get("index", -1)) >= sweep_idx
    ]
    if not dir_breaks:
        return None

    # 4) Fresh POI (FVG / OB / breaker iFVG) in the trade direction, created by
    #    the post-sweep move, sitting in the retracement zone vs current price.
    ifvgs = tf_data.get("ifvgs", []) or []
    candidates: List[Dict[str, Any]] = []
    for f in fvgs:
        if str(f.get("direction", "")).startswith(prefix) and not f.get("mitigated"):
            candidates.append({**f, "type": "FVG"})
    for o in obs:
        if str(o.get("direction", "")).startswith(prefix):
            candidates.append({**o, "type": "OB"})
    for f in ifvgs:
        # iFVG already carries the FLIPPED direction (acts from the opposite side)
        if str(f.get("direction", "")).startswith(prefix):
            candidates.append({**f, "type": "breaker"})

    valid: List[Dict[str, Any]] = []
    for p in candidates:
        top = float(p.get("top") or 0)
        bottom = float(p.get("bottom") or 0)
        if not (top > bottom > 0):
            continue
        if int(p.get("index", -1)) < sweep_idx:
            continue  # POI must come from the post-sweep move
        if direction == "bullish":
            if top < current_price and bottom > sweep_price:
                valid.append(p)
        else:
            if bottom > current_price and top < sweep_price:
                valid.append(p)
    if not valid:
        return None

    # Shallowest pullback = best location (closest to current price)
    poi = max(valid, key=lambda p: float(p["top"])) if direction == "bullish" \
        else min(valid, key=lambda p: float(p["bottom"]))

    return {
        "direction": direction,
        "poi": poi,
        "sweep_price": sweep_price,
        "sweep_index": sweep_idx,
        "displacement": disp,
        "structure_break": dir_breaks[-1],
    }


def build_candidate_order(
    mtf_data: Dict[str, Any],
    min_rr: float = 2.0,
    atr_buffer_mult: float = 0.25,
    min_stop_atr_mult: float = 0.5,
    max_rr: float = 10.0,
    max_sl_pct: float = 5.0,
    setup_timeframes: tuple = ("m15", "m5"),
    min_disp_atr: float = 1.5,
    min_disp_body: float = 0.65,
    enforce_premium_discount: bool = True,
) -> Optional[Dict[str, Any]]:
    """Deterministically price a pending-order candidate that satisfies the full
    SMC confluence story (sweep -> displacement -> structure shift -> retracement
    into a fresh POI), aligned with HTF bias and premium/discount.

    Entry  = near edge of the validated POI (pullback into the zone).
    SL     = behind the swept extreme (the real invalidation), ATR-buffered and
             floored, capped at max_sl_pct.
    TP1    = nearest HTF liquidity giving min_rr <= RR <= max_rr.
    Returns None when no setup satisfies every gate (code-side rejection).
    """
    ctx = extract_htf_context(mtf_data)
    current_price = float(mtf_data.get("current_price", 0) or 0)
    if current_price <= 0:
        return None

    bias = str(ctx.get("bias", {}).get("final_bias", "neutral"))
    if bias not in ("bullish", "bearish"):
        return None

    # HTF dealing-range equilibrium for the premium/discount check (applied to the
    # ENTRY, not the current price — a pending order fills at the POI, so its
    # location is what must be in discount (buy) / premium (sell)).
    equilibrium = 0.0
    for tf in ("h4", "h1"):
        dr = (mtf_data.get(tf, {}) or {}).get("dealing_range", {}) or {}
        eq = dr.get("equilibrium")
        if eq:
            equilibrium = float(eq)
            break

    # Find a confluence setup on the configured setup timeframes (finest first).
    setup = None
    setup_tf = None
    for tf in setup_timeframes:
        tf_data = mtf_data.get(tf) or {}
        if not tf_data:
            continue
        s = _evaluate_confluence(tf_data, current_price, bias, min_disp_atr, min_disp_body)
        if s:
            setup, setup_tf = s, tf
            break
    if not setup:
        return None

    poi = setup["poi"]
    sweep_price = float(setup["sweep_price"])

    # ATR for the SL buffer: prefer the setup timeframe, then fall back.
    atr = 0.0
    for tf in (setup_tf, "m15", "h1", "h4"):
        try:
            v = float(mtf_data.get(tf, {}).get("atr") or 0)
        except (TypeError, ValueError):
            v = 0.0
        if v > 0:
            atr = v
            break
    if atr <= 0:
        atr = current_price * 0.001

    liquidity = ctx.get("liquidity", {})
    min_stop_dist = atr * min_stop_atr_mult

    if bias == "bullish":
        order_type = "BUY_LIMIT"
        entry = float(poi["top"])
        location = "discount" if (equilibrium and entry <= equilibrium) else "premium"
        stop_loss = sweep_price - atr * atr_buffer_mult  # behind real invalidation
        if entry - stop_loss < min_stop_dist:
            stop_loss = entry - min_stop_dist
        risk = entry - stop_loss
        if risk <= 0:
            return None
        targets = sorted(
            float(x) for x in liquidity.get("nearest_bsl", []) if float(x or 0) > entry
        )
    else:
        order_type = "SELL_LIMIT"
        entry = float(poi["bottom"])
        location = "premium" if (equilibrium and entry >= equilibrium) else "discount"
        stop_loss = sweep_price + atr * atr_buffer_mult
        if stop_loss - entry < min_stop_dist:
            stop_loss = entry + min_stop_dist
        risk = stop_loss - entry
        if risk <= 0:
            return None
        targets = sorted(
            (float(x) for x in liquidity.get("nearest_ssl", []) if 0 < float(x or 0) < entry),
            reverse=True,
        )

    # Premium/Discount gate on the ENTRY: buy only in discount, sell only in
    # premium. Skipped when the HTF equilibrium is unavailable.
    if enforce_premium_discount and equilibrium > 0:
        if order_type == "BUY_LIMIT" and entry > equilibrium:
            return None
        if order_type == "SELL_LIMIT" and entry < equilibrium:
            return None

    # Reject setups whose real-invalidation stop is too wide.
    if entry > 0 and (risk / entry) * 100.0 > max_sl_pct:
        return None

    # TP1 = nearest liquidity target with min_rr <= RR <= max_rr.
    tp1 = None
    tp2 = None
    for i, target in enumerate(targets):
        rr = abs(target - entry) / risk
        if rr >= min_rr:
            tp1 = (entry + risk * max_rr if order_type == "BUY_LIMIT" else entry - risk * max_rr) \
                if rr > max_rr else target
            if i + 1 < len(targets):
                tp2 = targets[i + 1]
            break
    if tp1 is None:
        return None
    if tp2 is None:
        tp2 = entry + risk * 3.0 if order_type == "BUY_LIMIT" else entry - risk * 3.0

    sb = setup.get("structure_break", {}) or {}
    return {
        "order_type": order_type,
        "entry_price": round(entry, 5),
        "stop_loss": round(stop_loss, 5),
        "take_profit_1": round(float(tp1), 5),
        "take_profit_2": round(float(tp2), 5),
        "risk_to_reward": round(abs(tp1 - entry) / risk, 2),
        "bias": bias,
        "location": location,
        "poi_type": str(poi.get("type", "")),
        "poi_timeframe": str(setup_tf or ""),
        "poi_zone_low": round(float(poi["bottom"]), 5),
        "poi_zone_high": round(float(poi["top"]), 5),
        "sweep_price": round(sweep_price, 5),
        "structure_break": f"{sb.get('break_type', '')} {sb.get('direction', '')}".strip(),
        "atr_used": round(atr, 5),
    }


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

    # 3. Liquidity (swing levels + explicit previous day/week high/low).
    #    PDH/PWH are buy-side liquidity (BSL targets); PDL/PWL are sell-side (SSL).
    h4_bsl = list(h4.get("bsl_levels", []) or h1.get("bsl_levels", []))
    h4_ssl = list(h4.get("ssl_levels", []) or h1.get("ssl_levels", []))
    htf_liq = mtf_data.get("htf_liquidity", {}) or {}
    for k in ("pdh", "pwh"):
        v = htf_liq.get(k)
        if v:
            h4_bsl.append(float(v))
    for k in ("pdl", "pwl"):
        v = htf_liq.get(k)
        if v:
            h4_ssl.append(float(v))
    # De-duplicate while keeping order
    h4_bsl = sorted(set(round(float(x), 6) for x in h4_bsl if x))
    h4_ssl = sorted(set(round(float(x), 6) for x in h4_ssl if x))
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
    min_rr: float = 2.0,
    min_score: int = 55,
    max_sl_pct: float = 5.0,
    temperature: float = 0.1,
    num_ctx: int = 12288,
    num_predict: int = 2048,
    num_gpu: int = 999,
    stream: bool = False,
    keep_alive: str = "30m",
    learning_context: str = "",
    confluence_cfg: Optional[Dict[str, Any]] = None,
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
        
    logger.info(f"[PREFILTER_PASS] {symbol} Passed prefilter. reason={reason}")

    # Code-first pricing: the code computes the order (full SMC confluence chain),
    # the LLM only approves/rejects.
    cc = confluence_cfg or {}
    proposal = build_candidate_order(
        mtf_data,
        min_rr=min_rr,
        max_sl_pct=max_sl_pct,
        atr_buffer_mult=float(cc.get("sl_buffer_atr_mult", 0.25)),
        min_stop_atr_mult=float(cc.get("min_stop_atr_mult", 0.5)),
        max_rr=float(cc.get("max_rr", 10.0)),
        setup_timeframes=tuple(cc.get("setup_timeframes", ("m15", "m5"))),
        min_disp_atr=float(cc.get("min_displacement_atr", 1.5)),
        min_disp_body=float(cc.get("min_displacement_body_ratio", 0.65)),
        enforce_premium_discount=bool(cc.get("enforce_premium_discount", True)),
    )
    if proposal is None:
        logger.info(f"[CODE_PRICER] {symbol}: no confluence setup (sweep->displacement->shift->POI), skipping LLM")
        return PendingOrderDecision(
            decision="NO_TRADE",
            symbol=symbol,
            setup_quality="INVALID",
            reason="No confluence setup: needs sweep + displacement + structure shift + fresh POI in bias direction",
        )
    logger.info(
        f"[CODE_PRICER] {symbol}: {proposal['order_type']} entry={proposal['entry_price']} "
        f"SL={proposal['stop_loss']} TP1={proposal['take_profit_1']} RR={proposal['risk_to_reward']} "
        f"(POI {proposal['poi_type']}/{proposal['poi_timeframe']}) — sending to LLM for veto"
    )

    # Compact context for the veto (NOT the full market dump). With code-first
    # pricing the model only needs bias/structure/location to judge — sending the
    # whole multi-TF JSON made R1 inference exceed the 180s timeout.
    _ctx = extract_htf_context(mtf_data)
    _bias = _ctx.get("bias", {})
    _struct = _ctx.get("market_structure", {})
    veto_context = (
        f"Symbol: {symbol} | Current Price: {current_price} | Spread: {spread}\n"
        f"Bias: final={_bias.get('final_bias')} daily={_bias.get('daily')} h4={_bias.get('h4')}\n"
        f"Location: {_ctx.get('premium_discount', {}).get('current_location')}\n"
        f"Structure: displacement={_struct.get('displacement')} bos={_struct.get('bos')} "
        f"choch={_struct.get('choch')} mss={_struct.get('mss')}\n"
        f"Draw on liquidity: {_ctx.get('liquidity', {}).get('draw_on_liquidity')}"
    )

    user_content = veto_context

    # Call Ollama via native API
    try:
        import requests

        url = f"{ollama_base_url}/api/generate"
        learning_block = ""
        if learning_context:
            learning_block = (
                f"\n\n[PAST PERFORMANCE & LESSONS]\n"
                f"{learning_context[:1200]}\n"
                f"Weigh this context in your judgment. It is informational, not a "
                f"command to reject — judge this setup on its own technical merits.\n"
            )
        # Render the proposal as plain text (NOT JSON) so the model cannot simply
        # echo a JSON object back; with format:json that echo was losing setups.
        proposal_block = (
            f"\n\n[PROPOSED ORDER — review only, do NOT echo these numbers]\n"
            f"Direction: {proposal['order_type']}\n"
            f"Entry: {proposal['entry_price']} | Stop: {proposal['stop_loss']} | "
            f"Target: {proposal['take_profit_1']} | R:R: {proposal['risk_to_reward']}\n"
            f"Setup: {proposal['poi_type']} on {proposal.get('poi_timeframe')} timeframe, "
            f"bias is {proposal['bias']}, location {proposal.get('location')}\n"
            f"Confluence: swept liquidity at {proposal.get('sweep_price')}, "
            f"structure break {proposal.get('structure_break')} (stop sits behind the swept extreme)\n"
        )
        combined_prompt = f"{SMC_VETO_PROMPT}\n\n[MARKET DATA]\n{user_content}{proposal_block}{learning_block}{SMC_VETO_REQUEST}"
        
        payload = {
            "model": ollama_model,
            "prompt": combined_prompt,
            "stream": stream,
            # Constrain output to the EXACT verdict schema via Ollama structured
            # outputs. Plain "json" let deepseek-r1 echo a JSON blob from the
            # prompt (market data / proposal); a full schema makes the grammar
            # only permit these four keys, so echoing is impossible and every
            # response carries a real APPROVE/REJECT decision.
            "format": VETO_RESPONSE_SCHEMA,
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

    # Detect echo: model returned the proposal JSON instead of a verdict
    if "decision" not in parsed:
        logger.warning(
            f"[LLM_VETO] {symbol}: response has no 'decision' key "
            f"(model likely echoed the proposal); treating as REJECT. keys={list(parsed.keys())}"
        )

    # Veto verdict: APPROVE places the code-priced order, anything else is NO_TRADE
    verdict = str(parsed.get("decision", "REJECT")).strip().upper()
    try:
        veto_score = int(float(parsed.get("score", 0) or 0))
    except (TypeError, ValueError):
        veto_score = 0
    veto_quality = str(parsed.get("setup_quality", "WEAK")).strip().upper()
    veto_reason = str(parsed.get("reason", "")).strip()[:300]

    if verdict != "APPROVE":
        logger.info(f"[LLM_VETO] {symbol}: REJECT (score={veto_score}) — {veto_reason}")
        return PendingOrderDecision(
            decision="NO_TRADE",
            symbol=symbol,
            setup_quality=veto_quality or "WEAK",
            score=veto_score,
            reason=f"LLM veto: {veto_reason or 'rejected'}",
            full_analysis={"proposal": proposal, "veto": parsed},
        )

    # Approved: run guardrails on the CODE-priced numbers (never LLM numbers)
    guard_input = {
        "decision": "PLACE_BUY_LIMIT" if proposal["order_type"] == "BUY_LIMIT" else "PLACE_SELL_LIMIT",
        "score": veto_score,
        "setup_quality": veto_quality or "MEDIUM",
        "planned_order": {
            "order_type": proposal["order_type"],
            "entry_price": proposal["entry_price"],
            "stop_loss": proposal["stop_loss"],
            "take_profit_1": proposal["take_profit_1"],
            "take_profit_2": proposal["take_profit_2"],
            "valid_until_hours": 8,
            "order_reason": veto_reason
                or f"{proposal['poi_type']} {proposal['poi_timeframe']} retest ({proposal['bias']})",
        },
        "proposal": proposal,
        "veto": parsed,
    }
    result = _apply_guardrails(
        parsed=guard_input,
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
