"""
LLM Market News Engine
======================
Fetches crypto market headlines, analyzes them via OpenAI, and returns
an actionable score contribution for LONG/SHORT decisions.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import requests
from loguru import logger

from .news_analyzer import NewsAnalyzer
from .utils import OpenAIConfig


RSS_SOURCES = [
    # Crypto
    ("CoinDesk", "crypto", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "crypto", "https://cointelegraph.com/rss"),
    ("Decrypt", "crypto", "https://decrypt.co/feed"),
    ("Google News Crypto", "crypto", "https://news.google.com/rss/search?q=bitcoin+crypto+etf+regulation+hacks+institutional+flows&hl=en-US&gl=US&ceid=US:en"),
    # Macro
    ("Reuters Business", "macro", "https://feeds.reuters.com/reuters/businessNews"),
    ("CNBC Top", "macro", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("Google News Macro", "macro", "https://news.google.com/rss/search?q=Federal+Reserve+CPI+NFP+DXY+US+Treasury+yields&hl=en-US&gl=US&ceid=US:en"),
    # Metals
    ("Google News Gold", "metals", "https://news.google.com/rss/search?q=gold+XAUUSD+safe+haven+central+bank+buying&hl=en-US&gl=US&ceid=US:en"),
    ("Google News Silver", "metals", "https://news.google.com/rss/search?q=silver+XAGUSD+industrial+demand+safe+haven&hl=en-US&gl=US&ceid=US:en"),
    # Oil
    ("Google News Oil", "oil", "https://news.google.com/rss/search?q=WTI+Brent+OPEC+oil+supply+demand&hl=en-US&gl=US&ceid=US:en"),
    # Geopolitical risk
    ("Reuters World", "geopolitics", "https://feeds.reuters.com/Reuters/worldNews"),
    ("Google News Geopolitics", "geopolitics", "https://news.google.com/rss/search?q=Middle+East+war+energy+supply+risk+shipping&hl=en-US&gl=US&ceid=US:en"),
]

CACHE_FILE = "llm_market_news_cache.json"
CACHE_TTL_MINUTES = 20
LOOKBACK_HOURS = 12
MAX_HEADLINES = 14
MAX_HEADLINES_PER_CATEGORY = 4
REQUEST_TIMEOUT_SECONDS = 20
DEFAULT_SCORE_WEIGHT = 2.0


@dataclass
class LLMMarketNewsSignal:
    """LLM-based market news output mapped to side-aware score."""

    decision: str  # buy | sell | wait
    confidence: int
    score: float
    headlines_count: int
    sources: List[str]
    categories: List[str]
    analysis: str
    reason: str


def _normalize_symbol(symbol: Optional[str]) -> str:
    return str(symbol or "").strip().upper().replace(" ", "")


def _infer_asset_profile(symbol: Optional[str]) -> Dict[str, object]:
    text = _normalize_symbol(symbol)

    if any(k in text for k in ("XAU", "GOLD")):
        return {
            "asset_key": "gold",
            "label_ar": "الذهب",
            "focus_categories": ["metals", "macro", "geopolitics", "oil"],
            "keywords": ["gold", "xau", "bullion", "central bank", "safe haven"],
            "decision_scope_ar": "تداول الذهب (XAUUSD)",
        }
    if any(k in text for k in ("XAG", "SILVER")):
        return {
            "asset_key": "silver",
            "label_ar": "الفضة",
            "focus_categories": ["metals", "macro", "geopolitics", "oil"],
            "keywords": ["silver", "xag", "industrial demand", "safe haven"],
            "decision_scope_ar": "تداول الفضة (XAGUSD)",
        }
    if any(k in text for k in ("OIL", "WTI", "BRENT", "USOIL", "CRUDE")):
        return {
            "asset_key": "oil",
            "label_ar": "النفط",
            "focus_categories": ["oil", "geopolitics", "macro"],
            "keywords": ["oil", "wti", "brent", "opec", "crude", "inventory", "shipping"],
            "decision_scope_ar": "تداول النفط (WTI/Brent)",
        }
    if any(k in text for k in ("SNP500", "SPX500", "S&P500", "SP500", "ES=F")):
        return {
            "asset_key": "equity_index",
            "label_ar": "مؤشر S&P 500",
            "focus_categories": ["macro", "geopolitics", "oil"],
            "keywords": ["s&p", "sp500", "equity", "fed", "yields", "earnings"],
            "decision_scope_ar": "تداول مؤشر S&P 500",
        }

    # Default to crypto profile.
    return {
        "asset_key": "crypto",
        "label_ar": "الكريبتو",
        "focus_categories": ["crypto", "macro", "geopolitics"],
        "keywords": [
            "bitcoin",
            "btc",
            "ethereum",
            "eth",
            "crypto",
            "etf",
            "exchange",
            "stablecoin",
            "regulation",
            "hack",
        ],
        "decision_scope_ar": "تداول سوق الكريبتو",
    }


def _category_weight_map(asset_key: str) -> Dict[str, float]:
    if asset_key in {"gold", "silver"}:
        return {"metals": 1.0, "macro": 0.85, "geopolitics": 0.80, "oil": 0.45, "crypto": 0.15, "other": 0.20}
    if asset_key == "oil":
        return {"oil": 1.0, "geopolitics": 0.90, "macro": 0.80, "metals": 0.20, "crypto": 0.10, "other": 0.15}
    if asset_key == "equity_index":
        return {"macro": 1.0, "geopolitics": 0.70, "oil": 0.45, "metals": 0.25, "crypto": 0.20, "other": 0.20}
    return {"crypto": 1.0, "macro": 0.75, "geopolitics": 0.65, "oil": 0.20, "metals": 0.20, "other": 0.20}


def _headline_relevance_score(item: Dict, profile: Dict[str, object]) -> float:
    asset_key = str(profile.get("asset_key", "crypto"))
    weights = _category_weight_map(asset_key)
    cat = str(item.get("category", "other"))
    base = float(weights.get(cat, weights.get("other", 0.20)))

    title = str(item.get("title", "")).lower()
    boost = 0.0
    for kw in profile.get("keywords", []) or []:
        if str(kw).lower() in title:
            boost = max(boost, 0.20)
    return min(1.25, base + boost)


def _select_headlines_for_profile(headlines: List[Dict], profile: Dict[str, object]) -> Tuple[List[Dict], float]:
    if not headlines:
        return [], 0.0

    enriched: List[Tuple[float, datetime, Dict]] = []
    for item in headlines:
        relevance = _headline_relevance_score(item, profile)
        published = _parse_datetime(item.get("published_at", "")) or datetime(2000, 1, 1, tzinfo=timezone.utc)
        enriched.append((relevance, published, item))

    enriched.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected: List[Dict] = []
    per_cat: Dict[str, int] = {}
    focus = {str(c) for c in (profile.get("focus_categories", []) or [])}
    for relevance, _, item in enriched:
        cat = str(item.get("category", "other"))
        # Keep stronger diversity caps for non-focus categories.
        cat_cap = MAX_HEADLINES_PER_CATEGORY if cat in focus else max(2, MAX_HEADLINES_PER_CATEGORY - 1)
        if per_cat.get(cat, 0) >= cat_cap:
            continue
        item_copy = dict(item)
        item_copy["relevance"] = round(float(relevance), 3)
        selected.append(item_copy)
        per_cat[cat] = per_cat.get(cat, 0) + 1
        if len(selected) >= MAX_HEADLINES:
            break

    if not selected:
        return [], 0.0

    relevance_avg = sum(float(x.get("relevance", 0.0)) for x in selected) / len(selected)
    relevance_norm = max(0.0, min(1.0, relevance_avg / 1.0))
    return selected, relevance_norm


def _get_cache_path(cache_key: str = "global") -> Path:
    base_dir = Path(__file__).parent.parent
    key = re.sub(r"[^a-z0-9_]+", "_", str(cache_key or "global").lower()).strip("_") or "global"
    if key == "global":
        return base_dir / CACHE_FILE
    return base_dir / f"llm_market_news_cache_{key}.json"


def _parse_datetime(raw: str) -> Optional[datetime]:
    if not raw:
        return None

    raw = raw.strip()
    if not raw:
        return None

    # RFC2822 style: "Mon, 01 Jan 2026 10:00:00 GMT"
    try:
        dt = parsedate_to_datetime(raw)
        if dt:
            return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # ISO style
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _extract_decision_and_confidence(text: str) -> tuple[str, int]:
    clean = (text or "").replace("*", " ").strip()
    lowered = clean.lower()

    decision = "wait"

    decision_match = re.search(
        r"القرار(?:\s+التداولي(?:\s+النهائي)?)?\s*[:：]\s*(شراء|بيع|انتظار)",
        clean,
        flags=re.IGNORECASE,
    )
    if decision_match:
        val = decision_match.group(1).strip()
        if val == "شراء":
            decision = "buy"
        elif val == "بيع":
            decision = "sell"
        else:
            decision = "wait"
    else:
        if "buy" in lowered or "شراء" in clean:
            decision = "buy"
        elif "sell" in lowered or "بيع" in clean:
            decision = "sell"
        elif "انتظار" in clean or "wait" in lowered:
            decision = "wait"

    confidence = 60
    conf_match = re.search(
        r"درجة\s+الثقة\s*[:：]\s*(\d{1,3})",
        clean,
        flags=re.IGNORECASE,
    )
    if conf_match:
        confidence = int(conf_match.group(1))
    else:
        # Fallback to first percentage in response
        pct_match = re.search(r"(\d{1,3})\s*%", clean)
        if pct_match:
            confidence = int(pct_match.group(1))

    confidence = max(0, min(100, confidence))
    return decision, confidence


def _compute_side_score(decision: str, confidence: int, side: str, weight: float) -> float:
    if decision == "wait":
        return 0.0

    confidence_factor = max(0.3, min(1.0, confidence / 100.0))
    magnitude = round(weight * confidence_factor, 2)

    side_upper = (side or "").upper()
    if decision == "buy":
        return magnitude if side_upper == "LONG" else -magnitude
    if decision == "sell":
        return magnitude if side_upper == "SHORT" else -magnitude
    return 0.0


def _load_cache(cache_key: str) -> Optional[Dict]:
    path = _get_cache_path(cache_key)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text())
        cached_at = datetime.fromisoformat(payload.get("cached_at", "2000-01-01T00:00:00+00:00"))
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)

        age = datetime.now(timezone.utc) - cached_at
        if age <= timedelta(minutes=CACHE_TTL_MINUTES):
            return payload
    except Exception as e:
        logger.debug(f"Failed to load LLM news cache: {e}")

    return None


def _load_cache_any_age(cache_key: str) -> Optional[Dict]:
    """Load cache payload even if TTL expired (fallback for feed outages)."""
    path = _get_cache_path(cache_key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.debug(f"Failed to load stale LLM news cache: {e}")
        return None


def _save_cache(cache_key: str, payload: Dict) -> None:
    path = _get_cache_path(cache_key)
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.debug(f"Failed to save LLM news cache: {e}")


def _fetch_headlines() -> List[Dict]:
    headlines: List[Dict] = []
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=LOOKBACK_HOURS)

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CryptoSignalBot/1.0)"
    }

    for source_name, category, url in RSS_SOURCES:
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            root = ET.fromstring(response.content)

            for entry in root.iter():
                tag_name = entry.tag.split("}")[-1].lower()
                if tag_name not in {"item", "entry"}:
                    continue

                title = ""
                link = ""
                date_text = ""

                for child in list(entry):
                    child_tag = child.tag.split("}")[-1].lower()
                    child_text = (child.text or "").strip()

                    if child_tag == "title" and child_text:
                        title = child_text
                    elif child_tag == "link":
                        # Atom feeds usually store link in href attribute.
                        link = (child.attrib.get("href") or child_text or "").strip()
                    elif child_tag in {"pubdate", "published", "updated"} and child_text:
                        if not date_text:
                            date_text = child_text

                if not title:
                    continue

                published_at = _parse_datetime(date_text)
                if published_at and published_at < cutoff:
                    continue

                headlines.append(
                    {
                        "title": title.strip(),
                        "source": source_name,
                        "category": category,
                        "link": link.strip(),
                        "published_at": published_at.isoformat() if published_at else "",
                    }
                )

        except Exception as e:
            logger.debug(f"News feed fetch failed for {source_name}: {e}")

    # Deduplicate by normalized title and keep the newest version.
    dedup: Dict[str, Dict] = {}
    for item in headlines:
        key = re.sub(r"\s+", " ", item["title"].strip().lower())
        if not key:
            continue
        existing = dedup.get(key)
        if not existing:
            dedup[key] = item
            continue
        existing_dt = _parse_datetime(existing.get("published_at", ""))
        new_dt = _parse_datetime(item.get("published_at", ""))
        if new_dt and (not existing_dt or new_dt > existing_dt):
            dedup[key] = item

    unique = list(dedup.values())
    unique.sort(
        key=lambda x: _parse_datetime(x.get("published_at", "")) or datetime(2000, 1, 1, tzinfo=timezone.utc),
        reverse=True,
    )
    # Keep broad market coverage by capping each category.
    selected: List[Dict] = []
    per_cat: Dict[str, int] = {}
    selected_keys = set()

    for item in unique:
        cat = str(item.get("category", "other"))
        if per_cat.get(cat, 0) >= MAX_HEADLINES_PER_CATEGORY:
            continue
        key = f"{item.get('source','')}|{item.get('title','')}"
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
        per_cat[cat] = per_cat.get(cat, 0) + 1
        if len(selected) >= MAX_HEADLINES:
            return selected

    for item in unique:
        if len(selected) >= MAX_HEADLINES:
            break
        key = f"{item.get('source','')}|{item.get('title','')}"
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)

    return selected


def get_market_news_signal(
    side: str,
    symbol: str,
    openai_config: OpenAIConfig,
    score_weight: float = DEFAULT_SCORE_WEIGHT,
) -> Optional[LLMMarketNewsSignal]:
    """
    Return a side-aware score from automatically fetched + LLM-analyzed headlines.

    Args:
        side: LONG or SHORT (candidate direction)
        symbol: Trading symbol (for asset-aware news weighting)
        openai_config: OpenAI config for NewsAnalyzer
        score_weight: Maximum absolute contribution from LLM news
    """
    if not openai_config.enabled or not openai_config.api_key:
        return None

    profile = _infer_asset_profile(symbol)
    cache_key = str(profile.get("asset_key", "crypto"))
    cache = _load_cache(cache_key)
    stale_fallback_used = False
    if cache and cache.get("provider") == "openai" and cache.get("model") == openai_config.model:
        decision = cache.get("decision", "wait")
        confidence = int(cache.get("confidence", 60))
        analysis = cache.get("analysis", "")
        headlines_count = int(cache.get("headlines_count", 0))
        sources = [str(s) for s in cache.get("sources", [])]
        categories = [str(c) for c in cache.get("categories", [])]
        relevance_ratio = float(cache.get("relevance_ratio", 0.70))
    else:
        headlines_all = _fetch_headlines()
        selected_headlines, relevance_ratio = _select_headlines_for_profile(headlines_all, profile)
        if not selected_headlines:
            stale = _load_cache_any_age(cache_key)
            if not stale or stale.get("provider") != "openai" or stale.get("model") != openai_config.model:
                return None
            stale_fallback_used = True
            decision = stale.get("decision", "wait")
            confidence = int(stale.get("confidence", 60))
            analysis = stale.get("analysis", "")
            headlines_count = int(stale.get("headlines_count", 0))
            sources = [str(s) for s in stale.get("sources", [])]
            categories = [str(c) for c in stale.get("categories", [])]
            relevance_ratio = float(stale.get("relevance_ratio", 0.65))
            effective_weight = max(0.6, float(score_weight) * (0.65 + (0.70 * relevance_ratio)))
            score = _compute_side_score(decision, confidence, side, effective_weight)
            decision_ar = {
                "buy": "شراء",
                "sell": "بيع",
                "wait": "انتظار",
            }.get(decision, "انتظار")
            categories_ar = {
                "crypto": "كريبتو",
                "macro": "ماكرو",
                "metals": "معادن",
                "oil": "نفط",
                "geopolitics": "جيوسياسي",
                "other": "أخرى",
            }
            cats = [categories_ar.get(c, c) for c in categories]
            source_preview = ", ".join(sources[:3]) + ("..." if len(sources) > 3 else "")
            cat_preview = ", ".join(cats[:4]) + ("..." if len(cats) > 4 else "")
            reason = (
                f"LLM News: قرار {decision_ar} | ثقة {confidence}% | "
                f"الأصل: {profile.get('label_ar', 'السوق')} | "
                f"أخبار محللة: {headlines_count} | فئات: {cat_preview or 'غير محدد'} | "
                f"مصادر: {source_preview or 'غير متاح'} | "
                f"ملاءمة: {int(relevance_ratio * 100)}% | أثر: {score:+.2f} (cache)"
            )
            return LLMMarketNewsSignal(
                decision=decision,
                confidence=confidence,
                score=score,
                headlines_count=headlines_count,
                sources=sources,
                categories=categories,
                analysis=analysis,
                reason=reason,
            )

        news_blob_lines = []
        for idx, item in enumerate(selected_headlines, start=1):
            src = item.get("source", "Unknown")
            cat = item.get("category", "other")
            title = item.get("title", "")
            news_blob_lines.append(f"{idx}. {title} [المصدر: {src} | الفئة: {cat}]")

        news_blob = "\n".join(news_blob_lines)
        prompt_news_text = (
            f"الأصل المستهدف: {profile.get('decision_scope_ar', 'السوق')} "
            f"({symbol or profile.get('label_ar', 'asset')}).\n"
            "هذه أبرز عناوين السوق خلال الساعات الأخيرة (مختارة حسب صلتها بالأصل):\n\n"
            f"{news_blob}\n\n"
            "حلّل المزاج العام كمتداول خبير، مع إعطاء وزن أعلى للأخبار الأكثر ارتباطًا بالأصل المستهدف.\n"
            "لا تعطِ قرارًا عامًا على الكريبتو إذا الأصل المستهدف ليس كريبتو.\n"
            "القرار التداولي النهائي المطلوب يكون للتداول القصير على الأصل المستهدف: شراء / بيع / انتظار."
        )

        analyzer = NewsAnalyzer(openai_config)
        analysis = analyzer.analyze(prompt_news_text)
        decision, confidence = _extract_decision_and_confidence(analysis)
        headlines_count = len(selected_headlines)
        sources = sorted({str(h.get("source", "Unknown")) for h in selected_headlines})
        categories = sorted({str(h.get("category", "other")) for h in selected_headlines})

        _save_cache(
            cache_key,
            {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "provider": "openai",
                "model": openai_config.model,
                "decision": decision,
                "confidence": confidence,
                "analysis": analysis,
                "headlines_count": headlines_count,
                "sources": sources,
                "categories": categories,
                "relevance_ratio": relevance_ratio,
                "symbol_hint": str(symbol or ""),
            }
        )

    effective_weight = max(0.6, float(score_weight) * (0.65 + (0.70 * relevance_ratio)))
    score = _compute_side_score(decision, confidence, side, effective_weight)
    decision_ar = {
        "buy": "شراء",
        "sell": "بيع",
        "wait": "انتظار",
    }.get(decision, "انتظار")
    categories_ar = {
        "crypto": "كريبتو",
        "macro": "ماكرو",
        "metals": "معادن",
        "oil": "نفط",
        "geopolitics": "جيوسياسي",
        "other": "أخرى",
    }
    cats = [categories_ar.get(c, c) for c in categories]
    source_preview = ", ".join(sources[:3]) + ("..." if len(sources) > 3 else "")
    cat_preview = ", ".join(cats[:4]) + ("..." if len(cats) > 4 else "")
    reason = (
        f"LLM News: قرار {decision_ar} | ثقة {confidence}% | "
        f"الأصل: {profile.get('label_ar', 'السوق')} | "
        f"أخبار محللة: {headlines_count} | فئات: {cat_preview or 'غير محدد'} | "
        f"مصادر: {source_preview or 'غير متاح'} | "
        f"ملاءمة: {int(relevance_ratio * 100)}% | أثر: {score:+.2f}"
    )
    if stale_fallback_used:
        reason += " (cache)"

    return LLMMarketNewsSignal(
        decision=decision,
        confidence=confidence,
        score=score,
        headlines_count=headlines_count,
        sources=sources,
        categories=categories,
        analysis=analysis,
        reason=reason,
    )
