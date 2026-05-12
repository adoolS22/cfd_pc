"""
LLM Trade Postmortem
====================
Analyzes losing trades after closure and suggests bounded learning penalties.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI
from openai import OpenAIError

from .utils import LLMPostmortemConfig, OpenAIConfig


POSTMORTEM_PROMPT = """You are a strict trading postmortem auditor.
You receive one CLOSED losing trade from an algorithmic strategy.
Your job is to evaluate why it lost and whether the same setup should be penalized.

Rules:
1) Use only the provided trade context.
2) Be conservative: do not overfit one trade.
3) Pick one verdict from:
   - thesis_invalid
   - timing_error
   - risk_management_error
   - news_or_event_risk
   - normal_variance
4) mistake_tags must be chosen from:
   late_entry, chased_move, weak_confirmation, counter_trend, high_spread_cost,
   low_liquidity, stop_too_tight, ignored_news_risk, poor_rr_after_costs, regime_mismatch,
   onchain_flow_against_side, onchain_signal_low_reliability
5) action must be one of:
   - keep (no penalty)
   - soft_penalty
   - hard_penalty
6) penalty must be numeric in range [0, MAX_PENALTY].
7) Return JSON only.

JSON schema:
{
  "verdict": "thesis_invalid|timing_error|risk_management_error|news_or_event_risk|normal_variance",
  "confidence": 0,
  "mistake_tags": ["late_entry"],
  "summary": "short reason",
  "action": "keep|soft_penalty|hard_penalty",
  "penalty": 0.0,
  "recommendation": "short practical fix"
}
"""

ALLOWED_VERDICTS = {
    "thesis_invalid",
    "timing_error",
    "risk_management_error",
    "news_or_event_risk",
    "normal_variance",
}
ALLOWED_TAGS = {
    "late_entry",
    "chased_move",
    "weak_confirmation",
    "counter_trend",
    "high_spread_cost",
    "low_liquidity",
    "stop_too_tight",
    "ignored_news_risk",
    "poor_rr_after_costs",
    "regime_mismatch",
    "onchain_flow_against_side",
    "onchain_signal_low_reliability",
}
ALLOWED_ACTIONS = {"keep", "soft_penalty", "hard_penalty"}


@dataclass
class LLMPostmortemResult:
    """Normalized postmortem result."""

    verdict: str
    confidence: int
    mistake_tags: List[str]
    summary: str
    action: str
    penalty: float
    recommendation: str
    raw_json: str = ""


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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _normalize_confidence(value: Any, default: int = 60) -> int:
    raw = _to_float(value, default=float(default))
    if raw <= 0:
        raw = float(default)
    # Model may return 0..1 or 0..10 scales.
    if 0.0 <= raw <= 1.0:
        raw *= 100.0
    elif 0.0 <= raw <= 10.0:
        raw *= 10.0
    return max(0, min(100, int(round(raw))))


def _normalize_result(parsed: Dict[str, Any], max_penalty: float) -> LLMPostmortemResult:
    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in ALLOWED_VERDICTS:
        verdict = "normal_variance"

    confidence = _normalize_confidence(parsed.get("confidence", 60), default=60)

    raw_tags = parsed.get("mistake_tags", [])
    tags: List[str] = []
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            t = str(tag or "").strip().lower()
            if t in ALLOWED_TAGS and t not in tags:
                tags.append(t)
    tags = tags[:5]

    summary = str(parsed.get("summary", "")).strip()[:280]

    action = str(parsed.get("action", "keep")).strip().lower()
    if action not in ALLOWED_ACTIONS:
        action = "keep"

    penalty = _to_float(parsed.get("penalty", 0.0), default=0.0)
    penalty = max(0.0, min(float(max_penalty), penalty))
    if action == "keep":
        penalty = 0.0

    recommendation = str(parsed.get("recommendation", "")).strip()[:200]

    return LLMPostmortemResult(
        verdict=verdict,
        confidence=confidence,
        mistake_tags=tags,
        summary=summary,
        action=action,
        penalty=penalty,
        recommendation=recommendation,
        raw_json=json.dumps(parsed, ensure_ascii=False),
    )


def evaluate_loss_postmortem(
    trade_context: Dict[str, Any],
    openai_config: OpenAIConfig,
    postmortem_config: LLMPostmortemConfig,
) -> Optional[LLMPostmortemResult]:
    """
    Run LLM postmortem on one closed losing trade.
    Returns None when feature is disabled/unavailable or call fails.
    """
    if not getattr(postmortem_config, "enabled", False):
        return None
    # Support both OpenAI and Ollama (Ollama doesn't strictly need an api_key, just base_url)
    is_ollama = bool(getattr(openai_config, "base_url", None)) and "localhost" in str(getattr(openai_config, "base_url", ""))
    has_api_key = bool(getattr(openai_config, "api_key", None))
    
    if not getattr(openai_config, "enabled", False) or (not has_api_key and not is_ollama):
        logger.debug("LLM postmortem skipped: AI disabled or missing API key/Ollama config")
        return None

    max_penalty = max(0.0, float(getattr(postmortem_config, "penalty_max", 0.8)))
    timeout_seconds = max(10, int(getattr(postmortem_config, "timeout_seconds", 45)))

    payload = {
        "MAX_PENALTY": max_penalty,
        "trade": trade_context,
    }

    try:
        # For Ollama, the api_key can be anything, e.g. "ollama"
        api_key = getattr(openai_config, "api_key", None) or "ollama"
        base_url = openai_config.base_url if hasattr(openai_config, "base_url") else None
        
        # Ensure Ollama base_url gets the /v1 suffix required by OpenAI library
        if base_url and not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
            
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )
        response = client.chat.completions.create(
            model=openai_config.model,
            messages=[
                {"role": "system", "content": POSTMORTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Trade context JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2),
                },
            ],
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
        json_block = _extract_json_block(raw)
        if not json_block:
            logger.debug("LLM postmortem returned non-JSON response")
            return None
        parsed = json.loads(json_block)
        return _normalize_result(parsed, max_penalty=max_penalty)
    except (OpenAIError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e:
        logger.debug(f"LLM postmortem failed: {e}")
        return None
    except Exception as e:
        logger.debug(f"LLM postmortem unexpected error: {e}")
        return None
