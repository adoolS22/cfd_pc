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

Only return PLACE_BUY_LIMIT or PLACE_SELL_LIMIT if score >= 75.

==================================================
JSON OUTPUT SCHEMA
==================================================

Return exactly this JSON structure:

{
  "decision": "NO_TRADE | WAIT_FOR_SETUP | PLACE_BUY_LIMIT | PLACE_SELL_LIMIT | KEEP_PENDING_ORDER | CANCEL_PENDING_ORDER | MANAGE_OPEN_TRADE",
  "setup_quality": "HIGH | MEDIUM | WEAK | INVALID",
  "score": 0,
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
Never place an order without clear draw on liquidity or invalidation.
Never place an order if risk-to-reward is below 1:2.
Longs preferred from discount. Shorts preferred from premium.
If the setup is not clean, return NO_TRADE.
"""


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
# Guardrails — validate LLM decisions against market reality
# ═══════════════════════════════════════════════════════════════════════

def _apply_guardrails(
    parsed: Dict[str, Any],
    current_price: float,
    min_rr: float = 2.0,
    min_score: int = 75,
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

    score = int(parsed.get("score", 0))
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
    ollama_model: str = "qwen2.5:7b",
    timeout_seconds: int = 60,
    min_rr: float = 2.0,
    min_score: int = 75,
    max_sl_pct: float = 3.0,
) -> PendingOrderDecision:
    """Send structured MTF analysis to Ollama and get a pending-order decision.

    Args:
        symbol: Trading symbol
        mtf_data: Output from collect_mtf_analysis() — pre-computed SMC structures
        ollama_base_url: Ollama API base URL
        ollama_model: Model name (e.g. qwen2.5:7b)
        timeout_seconds: Request timeout
        min_rr: Minimum risk-to-reward ratio (default 2.0)
        min_score: Minimum score for actionable orders (default 75)
        max_sl_pct: Maximum SL distance as % of entry price (default 3.0)

    Returns:
        PendingOrderDecision with the validated decision
    """
    current_price = float(mtf_data.get("current_price", 0))
    spread = float(mtf_data.get("spread", 0))

    if current_price <= 0:
        return PendingOrderDecision(
            decision="NO_TRADE",
            symbol=symbol,
            reason="No price data available",
        )

    # Build user prompt with the pre-computed structured data
    user_content = (
        f"Symbol: {symbol}\n"
        f"Current Price: {current_price}\n"
        f"Spread: {spread}\n\n"
        f"Pre-computed Multi-Timeframe SMC Analysis (JSON):\n"
        f"{json.dumps(mtf_data, ensure_ascii=False, indent=2, default=str)}"
    )

    # Call Ollama via OpenAI-compatible API
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key="ollama",
            base_url=f"{ollama_base_url}/v1",
            timeout=max(15, timeout_seconds),
        )

        t0 = time.time()
        response = client.chat.completions.create(
            model=ollama_model,
            messages=[
                {"role": "system", "content": SMC_PENDING_ORDER_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.05,
            max_tokens=2000,
        )
        elapsed = time.time() - t0

        raw_text = (response.choices[0].message.content or "").strip()
        logger.debug(f"Planner LLM response for {symbol} ({elapsed:.1f}s): {raw_text[:200]}...")

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
