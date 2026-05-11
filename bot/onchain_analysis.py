"""
On-chain Analysis
=================
Fetches lightweight on-chain metrics and turns them into a directional score.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from loguru import logger

from .utils import OnChainConfig

CACHE_FILE = "onchain_cache.json"
COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
_MACRO_KEYS = ("XAU", "XAG", "OIL", "WTI", "BRENT", "SNP500", "SPX500", "SP500", "EURUSD", "EUR/USD")


@dataclass
class OnChainAnalysis:
    """On-chain analysis output."""
    asset: str
    score: float
    raw_score: float
    source: str
    reason: str
    tx_change_pct: Optional[float] = None
    active_addresses_change_pct: Optional[float] = None
    transfer_value_change_pct: Optional[float] = None
    data_points: int = 0
    reliability: float = 0.0
    coverage: float = 0.0
    freshness_minutes: float = 0.0
    valid: bool = False


def _cache_path() -> Path:
    return Path(__file__).parent.parent / CACHE_FILE


def _load_cache() -> Dict[str, Any]:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception as e:
        logger.debug(f"On-chain cache load failed: {e}")
    return {}


def _save_cache(payload: Dict[str, Any]) -> None:
    path = _cache_path()
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.debug(f"On-chain cache save failed: {e}")


def _extract_base_asset(symbol: str) -> str:
    text = str(symbol or "").upper().replace("_FUTURES", "")
    text = text.replace("/USDT:USDT", "").replace("-USDT", "").replace("USDT", "")
    for sep in ("/", "-", ":"):
        if sep in text:
            text = text.split(sep)[0]
    return text.strip()


def _is_macro_symbol(symbol: str) -> bool:
    s = str(symbol or "").upper()
    return any(k in s for k in _MACRO_KEYS)


def _metric_change_pct(rows: list, metric_name: str) -> Optional[float]:
    vals = []
    for row in rows:
        raw = row.get(metric_name)
        if raw in (None, ""):
            continue
        try:
            vals.append(float(raw))
        except Exception:
            continue

    if len(vals) < 4:
        return None

    latest = vals[-1]
    history = vals[:-1][-7:]
    if not history:
        return None
    avg = sum(history) / len(history)
    if avg <= 0:
        return None
    return ((latest - avg) / avg) * 100.0


def _fetch_coinmetrics_snapshot(asset: str, config: OnChainConfig) -> Optional[Dict[str, Any]]:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=max(8, int(config.lookback_days)) + 2)
    metric_sets = [
        "TxCnt,AdrActCnt,TxTfrValAdjUSD",
        "TxCnt,AdrActCnt",
        "TxCnt",
    ]

    last_error = None
    for metric_set in metric_sets:
        try:
            r = requests.get(
                COINMETRICS_URL,
                params={
                    "assets": asset.lower(),
                    "metrics": metric_set,
                    "frequency": "1d",
                    "start_time": start.isoformat(),
                    "end_time": end.isoformat(),
                    "page_size": 1000,
                },
                timeout=max(2, int(config.request_timeout_seconds)),
            )
            r.raise_for_status()
            payload = r.json()
            rows = payload.get("data") or []
            if not rows:
                continue

            tx_change = _metric_change_pct(rows, "TxCnt")
            addr_change = _metric_change_pct(rows, "AdrActCnt")
            transfer_change = _metric_change_pct(rows, "TxTfrValAdjUSD")
            if tx_change is None and addr_change is None and transfer_change is None:
                continue

            return {
                "asset": asset.lower(),
                "tx_change_pct": tx_change,
                "active_addresses_change_pct": addr_change,
                "transfer_value_change_pct": transfer_change,
                "data_points": len(rows),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "source": "coinmetrics",
            }
        except Exception as e:
            last_error = e
            continue

    if last_error is not None:
        logger.debug(f"On-chain fetch failed for {asset}: {last_error}")
    return None


def _load_snapshot_from_cache(asset: str, ttl_minutes: int) -> Optional[Dict[str, Any]]:
    payload = _load_cache()
    assets = payload.get("assets", {})
    item = assets.get(asset.lower())
    if not isinstance(item, dict):
        return None
    try:
        ts = datetime.fromisoformat(str(item.get("fetched_at")))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except Exception:
        return None
    age = datetime.now(timezone.utc) - ts
    if age > timedelta(minutes=max(1, int(ttl_minutes))):
        return None
    return item


def _save_snapshot_to_cache(asset: str, snapshot: Dict[str, Any]) -> None:
    payload = _load_cache()
    if not isinstance(payload, dict):
        payload = {}
    assets = payload.get("assets")
    if not isinstance(assets, dict):
        assets = {}
    assets[asset.lower()] = snapshot
    payload["assets"] = assets
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_cache(payload)


def _build_reason(
    asset: str,
    tx_change_pct: Optional[float],
    active_addresses_change_pct: Optional[float],
    transfer_value_change_pct: Optional[float],
    raw_score: float,
    effective_raw_score: float,
    applied_score: float,
    reliability: float,
    coverage: float,
    freshness_minutes: float,
    advisory_only: bool,
) -> str:
    tx_txt = "n/a" if tx_change_pct is None else f"{tx_change_pct:+.1f}%"
    addr_txt = "n/a" if active_addresses_change_pct is None else f"{active_addresses_change_pct:+.1f}%"
    val_txt = "n/a" if transfer_value_change_pct is None else f"{transfer_value_change_pct:+.1f}%"
    rel_txt = f"{reliability*100:.0f}%"
    cov_txt = f"{coverage*100:.0f}%"
    fresh_txt = f"{freshness_minutes:.0f}m"
    if advisory_only:
        return (
            f"On-chain {asset.upper()}: tx {tx_txt}, active {addr_txt}, transfer {val_txt} "
            f"(raw {raw_score:+.2f}, weighted {effective_raw_score:+.2f}, rel {rel_txt}, cov {cov_txt}, age {fresh_txt})"
        )
    return (
        f"On-chain {asset.upper()}: tx {tx_txt}, active {addr_txt}, transfer {val_txt} "
        f"(score {applied_score:+.2f}, rel {rel_txt}, cov {cov_txt}, age {fresh_txt})"
    )


def analyze_onchain(symbol: str, side: str, config: OnChainConfig) -> OnChainAnalysis:
    """
    Analyze on-chain metrics for crypto symbols.
    """
    if not getattr(config, "enabled", False):
        return OnChainAnalysis(asset="", score=0.0, raw_score=0.0, source="disabled", reason="", valid=False)

    if _is_macro_symbol(symbol):
        return OnChainAnalysis(
            asset="",
            score=0.0,
            raw_score=0.0,
            source=str(getattr(config, "provider", "coinmetrics")),
            reason="On-chain skipped: macro asset.",
            valid=False,
        )

    asset = _extract_base_asset(symbol)
    if not asset:
        return OnChainAnalysis(asset="", score=0.0, raw_score=0.0, source="unknown", reason="", valid=False)

    if str(getattr(config, "provider", "coinmetrics")) != "coinmetrics":
        return OnChainAnalysis(
            asset=asset.lower(),
            score=0.0,
            raw_score=0.0,
            source=str(getattr(config, "provider", "coinmetrics")),
            reason=f"On-chain provider not supported: {getattr(config, 'provider', '')}",
            valid=False,
        )

    snapshot = _load_snapshot_from_cache(asset, int(getattr(config, "cache_ttl_minutes", 20)))
    if snapshot is None:
        snapshot = _fetch_coinmetrics_snapshot(asset, config)
        if snapshot is not None:
            _save_snapshot_to_cache(asset, snapshot)

    if snapshot is None:
        return OnChainAnalysis(
            asset=asset.lower(),
            score=0.0,
            raw_score=0.0,
            source="coinmetrics",
            reason=f"On-chain {asset.upper()}: data unavailable.",
            valid=False,
        )

    tx_change = snapshot.get("tx_change_pct")
    addr_change = snapshot.get("active_addresses_change_pct")
    transfer_change = snapshot.get("transfer_value_change_pct")
    data_points = int(snapshot.get("data_points", 0) or 0)
    freshness_minutes = 0.0
    try:
        ts = datetime.fromisoformat(str(snapshot.get("fetched_at", "")))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        freshness_minutes = max(
            0.0,
            (datetime.now(timezone.utc) - ts).total_seconds() / 60.0,
        )
    except Exception:
        freshness_minutes = float(getattr(config, "cache_ttl_minutes", 20))

    pos_th = max(0.1, float(getattr(config, "positive_change_threshold_pct", 5.0)))
    w_tx, w_addr, w_val = 0.45, 0.35, 0.20
    weighted_sum = 0.0
    used_weight = 0.0
    metrics_present = 0

    for change, w in ((tx_change, w_tx), (addr_change, w_addr), (transfer_change, w_val)):
        if change is None:
            continue
        metrics_present += 1
        norm = max(-1.0, min(1.0, float(change) / pos_th))
        weighted_sum += norm * w
        used_weight += w

    if used_weight <= 0:
        return OnChainAnalysis(
            asset=asset.lower(),
            score=0.0,
            raw_score=0.0,
            source="coinmetrics",
            reason=f"On-chain {asset.upper()}: insufficient metrics.",
            valid=False,
        )

    bullish_signal = weighted_sum / used_weight
    raw_score = bullish_signal * float(getattr(config, "score_boost_max", 1.2))
    if str(side).upper() == "SHORT":
        raw_score *= -1.0
    elif str(side).upper() != "LONG":
        raw_score = 0.0

    # Confidence weighting: reduce impact when on-chain coverage, depth, or freshness is weak.
    use_reliability = bool(getattr(config, "use_reliability_weighting", True))
    min_rel_to_score = max(0.0, min(1.0, float(getattr(config, "min_reliability_to_score", 0.45))))
    min_data_points = max(4, int(getattr(config, "min_data_points", 8)))
    half_life_minutes = max(10, int(getattr(config, "freshness_half_life_minutes", 180)))

    coverage = max(0.0, min(1.0, float(metrics_present) / 3.0))
    data_quality = max(0.0, min(1.0, float(data_points) / float(min_data_points)))
    freshness_factor = max(0.0, min(1.0, 0.5 ** (float(freshness_minutes) / float(half_life_minutes))))
    reliability = coverage * data_quality * freshness_factor
    if not use_reliability:
        reliability = 1.0

    effective_raw_score = raw_score * reliability
    if use_reliability and reliability < min_rel_to_score:
        effective_raw_score = 0.0

    applied_score = 0.0 if bool(getattr(config, "advisory_only", True)) else effective_raw_score
    reason = _build_reason(
        asset=asset,
        tx_change_pct=tx_change,
        active_addresses_change_pct=addr_change,
        transfer_value_change_pct=transfer_change,
        raw_score=raw_score,
        effective_raw_score=effective_raw_score,
        applied_score=applied_score,
        reliability=reliability,
        coverage=coverage,
        freshness_minutes=freshness_minutes,
        advisory_only=bool(getattr(config, "advisory_only", True)),
    )

    return OnChainAnalysis(
        asset=asset.lower(),
        score=applied_score,
        raw_score=raw_score,
        source=str(snapshot.get("source", "coinmetrics")),
        reason=reason,
        tx_change_pct=tx_change,
        active_addresses_change_pct=addr_change,
        transfer_value_change_pct=transfer_change,
        data_points=data_points,
        reliability=reliability,
        coverage=coverage,
        freshness_minutes=freshness_minutes,
        valid=True,
    )
