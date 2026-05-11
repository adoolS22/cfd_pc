"""
Exchange Layer
==============
CCXT-based exchange interface plus optional MT5 bridge adapter.
Uses public endpoints only unless a private MT5 bridge is configured.
"""

import ccxt
import pandas as pd
import time
import random
import re
import os
import json
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
from loguru import logger
import requests

from bot.mt5_client import MT5Client, MT5ClientError


# Rate limiting and retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds
RATE_LIMIT_DELAY = 0.5  # seconds between requests
MAX_BACKOFF = 18.0  # cap for exponential backoff
JITTER_MAX = 0.35  # random jitter to avoid retry bursts
_EXCHANGE_COOLDOWN_UNTIL: Dict[str, float] = {}


class ExchangeError(Exception):
    """Custom exception for exchange-related errors."""
    pass


def _exchange_key(exchange: ccxt.Exchange) -> str:
    """Build a stable key for per-exchange cooldown tracking."""
    return str(getattr(exchange, "id", None) or exchange.__class__.__name__).lower()


def _extract_retry_after_seconds(err: Exception) -> Optional[float]:
    """Extract retry-after hint from exchange error text when present."""
    text = str(err or "")
    # Common forms: "retry after 2", "retry_after=2", "wait 2s"
    patterns = (
        r"retry[\s_-]*after[:=\s]+(\d+(?:\.\d+)?)",
        r"retryafter[:=\s]+(\d+(?:\.\d+)?)",
        r"wait[:=\s]+(\d+(?:\.\d+)?)s",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                continue
    return None


def _is_rate_limit_error(err: Exception) -> bool:
    """Detect rate-limit style errors (including KuCoin-specific 429 codes)."""
    if isinstance(err, ccxt.RateLimitExceeded):
        return True
    text = str(err or "").lower()
    hints = (
        "429",
        "429000",
        "too many request",
        "too many requests",
        "rate limit",
        "ratelimit",
        "throttle",
    )
    return any(h in text for h in hints)


def _compute_backoff_seconds(attempt: int, err: Optional[Exception] = None) -> float:
    """Compute retry backoff with optional retry-after hint and jitter."""
    retry_after = _extract_retry_after_seconds(err) if err is not None else None
    base = retry_after if retry_after is not None else (BASE_DELAY * (2 ** attempt))
    base = min(MAX_BACKOFF, max(BASE_DELAY, float(base)))
    return base + random.uniform(0.0, JITTER_MAX)


def _wait_for_exchange_cooldown(exchange: ccxt.Exchange) -> None:
    """Respect shared cooldown to avoid hammering an exchange after 429."""
    key = _exchange_key(exchange)
    until = float(_EXCHANGE_COOLDOWN_UNTIL.get(key, 0.0) or 0.0)
    now = time.time()
    if until > now:
        wait_sec = min(MAX_BACKOFF, until - now)
        if wait_sec > 0:
            logger.debug(f"{key}: shared cooldown active ({wait_sec:.2f}s)")
            time.sleep(wait_sec)


def _register_exchange_cooldown(exchange: ccxt.Exchange, delay_sec: float) -> None:
    """Set per-exchange cooldown after a throttling response."""
    key = _exchange_key(exchange)
    _EXCHANGE_COOLDOWN_UNTIL[key] = max(float(_EXCHANGE_COOLDOWN_UNTIL.get(key, 0.0) or 0.0), time.time() + delay_sec)


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_api_path(path: Any, default: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        raw = default
    if not raw.startswith("/"):
        raw = f"/{raw}"
    if len(raw) > 1:
        raw = raw.rstrip("/")
    return raw


def _normalize_symbol_key(symbol: str) -> str:
    """
    Canonical symbol key for symbol_map matching.
    Accepts forms like:
      - BTC/USDT:USDT
      - BTC/USDT
      - BTCUSDT
      - BTC-USDT
      - XAU/USD, XAUUSD
    """
    raw = str(symbol or "").strip().upper().replace("_FUTURES", "").replace("_SPOT", "")
    if ":" in raw:
        raw = raw.split(":", 1)[0]
    return raw.replace("/", "").replace("-", "").replace("_", "")


def _load_symbol_map(cfg: Dict[str, Any]) -> Dict[str, str]:
    raw = dict(cfg.get("symbol_map", {}) or {})
    out: Dict[str, str] = {}
    for k, v in raw.items():
        key = _normalize_symbol_key(k)
        val = str(v or "").strip()
        if key and val:
            out[key] = val

    map_file = str(cfg.get("symbol_map_file", "") or "").strip()
    if map_file:
        try:
            with open(map_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                for k, v in data.items():
                    key = _normalize_symbol_key(k)
                    val = str(v or "").strip()
                    if key and val and key not in out:
                        out[key] = val
        except Exception as e:
            logger.warning(f"MT5 bridge symbol_map_file ignored ({map_file}): {e}")

    return out


class MT5BridgeExchange:
    """
    Lightweight exchange adapter that proxies ticker/OHLCV via an external MT5 bridge HTTP API.
    Expected endpoints:
      GET /health
      GET /ticker?symbol=...
      GET /ohlcv?symbol=...&timeframe=...&limit=...
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, market_type: str = "futures") -> None:
        self.id = "mt5_bridge"
        self.market_type = str(market_type or "futures")
        config = dict(cfg or {})

        self.base_url = str(config.get("base_url") or os.getenv("MT5_BRIDGE_BASE_URL", "")).strip().rstrip("/")
        if not self.base_url:
            raise ExchangeError("MT5 bridge base_url is required (config.mt5_bridge.base_url)")

        timeout_raw = config.get("timeout_seconds", os.getenv("MT5_BRIDGE_TIMEOUT_SECONDS", 6))
        try:
            self.timeout_seconds = max(1.0, float(timeout_raw))
        except Exception:
            self.timeout_seconds = 6.0

        self.verify_ssl = _to_bool(config.get("verify_ssl", os.getenv("MT5_BRIDGE_VERIFY_SSL", "false")), default=False)
        token_env = str(config.get("token_env", "MT5_BRIDGE_TOKEN")).strip() or "MT5_BRIDGE_TOKEN"
        self.token = str(config.get("token") or os.getenv(token_env, "")).strip()
        self.account_label = str(config.get("account_label") or os.getenv("MT5_BRIDGE_ACCOUNT_LABEL", "")).strip()
        self.symbol_suffix = str(config.get("symbol_suffix", "") or "").strip()
        self.symbol_map = _load_symbol_map(config)
        self.health_path = _normalize_api_path(config.get("health_path"), default="/health")
        self.ticker_path = _normalize_api_path(config.get("ticker_path"), default="/ticker")
        self.ohlcv_path = _normalize_api_path(config.get("ohlcv_path"), default="/ohlcv")
        self.validate_endpoints = _to_bool(config.get("validate_endpoints", True), default=True)
        self.ticker_from_ohlcv = _to_bool(config.get("ticker_from_ohlcv", True), default=True)
        self._ticker_endpoint_available = True

    def load_markets(self) -> None:
        try:
            self._request_json(self.health_path, params={})
            if self.validate_endpoints:
                self._assert_endpoint_exists("ohlcv", self.ohlcv_path)
                self._ticker_endpoint_available = self._endpoint_exists(self.ticker_path)
                if not self._ticker_endpoint_available:
                    if self.ticker_from_ohlcv:
                        logger.warning(
                            f"MT5 bridge ticker endpoint missing ({self.ticker_path}); "
                            f"falling back to last close from {self.ohlcv_path}"
                        )
                    else:
                        raise ExchangeError(
                            f"MT5 bridge endpoint '{self.ticker_path}' for ticker is missing (HTTP 404). "
                            f"Configured base_url={self.base_url}. Check bridge deployment or set custom ticker_path."
                        )
            logger.info(
                "Initialized mt5_bridge exchange "
                f"(health check OK, ticker={self.ticker_path}, ohlcv={self.ohlcv_path}, "
                f"ticker_available={self._ticker_endpoint_available})"
            )
        except Exception as e:
            raise ExchangeError(f"Failed to initialize mt5_bridge: {e}")

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
            headers["X-API-Key"] = self.token
        return headers

    def _assert_endpoint_exists(self, name: str, path: str) -> None:
        """
        Fast path-level check to catch bridge mismatches early.
        Any status other than 404 means the route likely exists (may still fail validation/auth).
        """
        url = f"{self.base_url}{path}"
        try:
            response = requests.get(
                url,
                params={},
                headers=self._headers(),
                timeout=self.timeout_seconds,
                verify=self.verify_ssl,
            )
        except requests.RequestException as e:
            raise ExchangeError(f"MT5 bridge endpoint probe failed for {name} ({path}): {e}")

        if response.status_code == 404:
            raise ExchangeError(
                f"MT5 bridge endpoint '{path}' for {name} is missing (HTTP 404). "
                f"Configured base_url={self.base_url}. Check bridge deployment or set custom {name}_path."
            )

    def _endpoint_exists(self, path: str) -> bool:
        """
        Returns False only for explicit 404 (route missing), otherwise True.
        """
        url = f"{self.base_url}{path}"
        try:
            response = requests.get(
                url,
                params={},
                headers=self._headers(),
                timeout=self.timeout_seconds,
                verify=self.verify_ssl,
            )
            return response.status_code != 404
        except requests.RequestException:
            # Keep runtime resilient when probe fails transiently.
            return True

    def _request_json(self, path: str, params: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = requests.get(
                url,
                params=params,
                headers=self._headers(),
                timeout=self.timeout_seconds,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("ok") is False:
                raise ExchangeError(str(payload.get("error") or payload.get("message") or "bridge error"))
            return payload
        except requests.RequestException as e:
            raise ExchangeError(f"MT5 bridge request failed: {e}")
        except ValueError as e:
            raise ExchangeError(f"MT5 bridge invalid JSON: {e}")

    def _resolve_symbol(self, symbol: str) -> str:
        raw = str(symbol or "").strip()
        norm = _normalize_symbol_key(raw)
        if norm in self.symbol_map:
            target = self.symbol_map[norm]
        else:
            core = norm
            raw_u = raw.upper()
            if "/" in raw_u and "USDT" in raw_u:
                base = raw_u.split("/", 1)[0].replace(":", "").replace("-", "").replace("_", "")
                target = f"{base}USD"
            elif core in {"XAUUSD"}:
                target = "XAUUSD"
            elif core in {"XAGUSD"}:
                target = "XAGUSD"
            elif core in {"OILUSD", "WTIUSD", "USOIL"}:
                target = "USOIL"
            elif core in {"SNP500", "US500", "SPX500"}:
                target = "US500"
            else:
                target = core
        if self.symbol_suffix and not target.endswith(self.symbol_suffix):
            target = f"{target}{self.symbol_suffix}"
        return target

    @staticmethod
    def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _to_ms_timestamp(raw: Any) -> Optional[int]:
        if raw is None:
            return None
        try:
            if isinstance(raw, (int, float)):
                value = float(raw)
                if value > 1e12:
                    return int(value)
                if value > 1e10:
                    return int(value)
                return int(value * 1000.0)
            text = str(raw).strip()
            if not text:
                return None
            if text.isdigit():
                num = int(text)
                if num > 1e12:
                    return num
                if num > 1e10:
                    return num
                return num * 1000
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000.0)
        except Exception:
            return None

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        mt5_symbol = self._resolve_symbol(symbol)
        if not self._ticker_endpoint_available and self.ticker_from_ohlcv:
            return self._build_ticker_from_ohlcv(symbol, mt5_symbol)

        params: Dict[str, Any] = {"symbol": mt5_symbol}
        if self.account_label:
            params["account"] = self.account_label
        try:
            payload = self._request_json(self.ticker_path, params=params)
        except ExchangeError as e:
            if self.ticker_from_ohlcv and "404" in str(e):
                self._ticker_endpoint_available = False
                return self._build_ticker_from_ohlcv(symbol, mt5_symbol)
            raise

        body = payload.get("data") if isinstance(payload, dict) and isinstance(payload.get("data"), dict) else payload
        if not isinstance(body, dict):
            raise ExchangeError(f"MT5 bridge ticker invalid payload for {symbol}")

        bid = self._as_float(body.get("bid"))
        ask = self._as_float(body.get("ask"))
        last = self._as_float(body.get("last"))
        if last is None and bid is not None and ask is not None:
            last = (bid + ask) / 2.0
        if bid is None and last is not None:
            bid = last
        if ask is None and last is not None:
            ask = last

        if last is None:
            raise ExchangeError(f"MT5 bridge ticker missing last for {symbol} ({mt5_symbol})")

        return {
            "symbol": symbol,
            "bridge_symbol": mt5_symbol,
            "last": last,
            "bid": bid,
            "ask": ask,
            "high": self._as_float(body.get("high")),
            "low": self._as_float(body.get("low")),
            "baseVolume": self._as_float(body.get("volume")),
            "timestamp": datetime.now(timezone.utc),
        }

    def _build_ticker_from_ohlcv(self, symbol: str, mt5_symbol: str) -> Dict[str, Any]:
        """
        Fallback ticker builder for bridges that expose OHLCV but not /ticker.
        Uses latest close as last/bid/ask.
        """
        raw = self.fetch_ohlcv(symbol, timeframe="1m", limit=2)
        if not raw:
            raise ExchangeError(f"MT5 bridge ticker fallback failed (no OHLCV) for {symbol} ({mt5_symbol})")
        ts_ms, _o, h, l, c, v = raw[-1]
        last = float(c)
        return {
            "symbol": symbol,
            "bridge_symbol": mt5_symbol,
            "last": last,
            "bid": last,
            "ask": last,
            "high": float(h),
            "low": float(l),
            "baseVolume": float(v),
            "timestamp": datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc),
        }

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> List[List[float]]:
        mt5_symbol = self._resolve_symbol(symbol)
        params: Dict[str, Any] = {
            "symbol": mt5_symbol,
            "timeframe": str(timeframe),
            "limit": int(limit),
        }
        if self.account_label:
            params["account"] = self.account_label
        payload = self._request_json(self.ohlcv_path, params=params)

        candles = payload
        if isinstance(payload, dict):
            if isinstance(payload.get("candles"), list):
                candles = payload.get("candles")
            elif isinstance(payload.get("data"), list):
                candles = payload.get("data")

        if not isinstance(candles, list) or not candles:
            raise ExchangeError(f"MT5 bridge returned no candles for {symbol} {timeframe}")

        out: List[List[float]] = []
        for item in candles:
            if isinstance(item, dict):
                ts = self._to_ms_timestamp(item.get("timestamp") or item.get("time") or item.get("t"))
                o = self._as_float(item.get("open"))
                h = self._as_float(item.get("high"))
                l = self._as_float(item.get("low"))
                c = self._as_float(item.get("close"))
                v = self._as_float(item.get("volume"), 0.0) or 0.0
            elif isinstance(item, (list, tuple)) and len(item) >= 5:
                ts = self._to_ms_timestamp(item[0])
                o = self._as_float(item[1])
                h = self._as_float(item[2])
                l = self._as_float(item[3])
                c = self._as_float(item[4])
                v = self._as_float(item[5], 0.0) if len(item) >= 6 else 0.0
            else:
                continue

            if ts is None or None in {o, h, l, c}:
                continue
            out.append([int(ts), float(o), float(h), float(l), float(c), float(v or 0.0)])

        if not out:
            raise ExchangeError(f"MT5 bridge candles parse failed for {symbol} {timeframe}")
        return out


class MT5Exchange:
    """Direct MetaTrader 5 exchange adapter used by the existing bot data layer."""
    def parse_timeframe(self, tf): 
        units = {"m": 60, "h": 3600, "d": 86400}
        return int(tf[:-1]) * units.get(tf[-1], 1)

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, market_type: str = "futures") -> None:
        self.id = "mt5"
        self.market_type = str(market_type or "futures")
        self.config = dict(cfg or {})
        self.client = MT5Client.from_config(self.config)
        self.default_symbol = str(self.config.get("default_symbol", "") or "").strip()
        self._ticker_endpoint_available = True

    def load_markets(self) -> None:
        try:
            self.client.connect_mt5()
            if self.default_symbol:
                self.client.ensure_symbol(self.default_symbol)
        except MT5ClientError as e:
            raise ExchangeError(str(e)) from e

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        try:
            tick = self.client.get_tick(symbol)
            return {
                "symbol": symbol,
                "mt5_symbol": tick.get("mt5_symbol"),
                "last": tick.get("last"),
                "bid": tick.get("bid"),
                "ask": tick.get("ask"),
                "spread": tick.get("spread"),
                "spread_points": tick.get("spread_points"),
                "tick_time": tick.get("tick_time"),
                "symbol_info": tick.get("symbol_info"),
                "baseVolume": tick.get("volume"),
                "timestamp": datetime.now(timezone.utc),
            }
        except MT5ClientError as e:
            raise ExchangeError(str(e)) from e

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> List[List[float]]:
        try:
            candles = self.client.get_rates(symbol, timeframe, limit)
            return [
                [
                    int(float(item["time"]) * 1000.0),
                    float(item["open"]),
                    float(item["high"]),
                    float(item["low"]),
                    float(item["close"]),
                    float(item.get("tick_volume") or item.get("real_volume") or 0.0),
                ]
                for item in candles
            ]
        except MT5ClientError as e:
            raise ExchangeError(str(e)) from e

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        try:
            return self.client.get_symbol_info(symbol)
        except MT5ClientError as e:
            raise ExchangeError(str(e)) from e

    def close(self) -> None:
        self.client.shutdown_mt5()


def get_exchange(
    exchange_name: str = "binance",
    market_type: str = "futures",
    mt5_bridge_config: Optional[Dict[str, Any]] = None,
    mt5_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Initialize and return a ccxt exchange instance.
    
    Args:
        exchange_name: Name of the exchange (default: binance)
        market_type: Market type - 'futures' or 'spot'
        
    Returns:
        Configured ccxt exchange instance
    """
    name = str(exchange_name or "").strip().lower()
    if name == "mt5_bridge":
        bridge_cfg = dict(mt5_bridge_config or {})
        strict_mode = _to_bool(bridge_cfg.get("strict_mode", True), default=True)
        fallback_exchange = str(bridge_cfg.get("fallback_exchange", "binance") or "binance").strip().lower()
        try:
            exchange = MT5BridgeExchange(bridge_cfg, market_type=market_type)
            exchange.load_markets()
            return exchange
        except Exception as e:
            if strict_mode:
                raise ExchangeError(str(e))
            if fallback_exchange in {"mt5", "mt5_bridge"}:
                raise ExchangeError(f"mt5_bridge unavailable and invalid fallback_exchange={fallback_exchange}")
            logger.warning(
                f"mt5_bridge unavailable ({e}); falling back to {fallback_exchange} {market_type}"
            )
            return get_exchange(fallback_exchange, market_type, mt5_bridge_config=None)

    if name == "mt5":
        direct_cfg = dict(mt5_config or {})
        strict_mode = _to_bool(direct_cfg.get("strict_mode", True), default=True)
        fallback_exchange = str(direct_cfg.get("fallback_exchange", "") or "").strip().lower()
        try:
            exchange = MT5Exchange(direct_cfg, market_type=market_type)
            exchange.load_markets()
            return exchange
        except Exception as e:
            if strict_mode:
                raise ExchangeError(str(e))
            if not fallback_exchange:
                raise ExchangeError(f"mt5 unavailable and no fallback_exchange is configured: {e}")
            if fallback_exchange in {"mt5", "mt5_bridge"}:
                raise ExchangeError(f"mt5 unavailable and invalid fallback_exchange={fallback_exchange}")
            logger.warning(
                f"mt5 unavailable ({e}); falling back to {fallback_exchange} {market_type}"
            )
            return get_exchange(fallback_exchange, market_type, mt5_bridge_config=None, mt5_config=None)

    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if market_type == 'futures' else 'spot',
                'adjustForTimeDifference': True,
            }
        })
        
        # Load markets to ensure connection works
        exchange.load_markets()
        logger.info(f"Initialized {exchange_name} exchange for {market_type} markets")
        
        return exchange
        
    except AttributeError:
        raise ExchangeError(f"Exchange '{exchange_name}' not found in ccxt")
    except Exception as e:
        raise ExchangeError(f"Failed to initialize exchange: {str(e)}")


def fetch_ohlcv(
    exchange: Any,
    symbol: str,
    timeframe: str,
    limit: int = 500
) -> pd.DataFrame:
    """
    Fetch OHLCV data with retry logic and exponential backoff.
    
    Args:
        exchange: ccxt exchange instance
        symbol: Trading pair symbol (e.g., 'BTC/USDT:USDT')
        timeframe: Candle timeframe (e.g., '4h', '15m', '1h')
        limit: Number of candles to fetch
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
        Index is datetime in UTC
    """
    if hasattr(exchange, "id") and str(getattr(exchange, "id", "")).lower() in {"mt5", "mt5_bridge"}:
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not raw:
                raise ExchangeError(f"No OHLCV data returned for {symbol}")
            df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.debug(
                f"Fetched {len(df)} candles for {symbol} {timeframe} via {str(getattr(exchange, 'id', 'mt5')).lower()}"
            )
            return df
        except Exception as e:
            raise ExchangeError(f"MT5 OHLCV error for {symbol}: {e}")

    for attempt in range(MAX_RETRIES):
        try:
            _wait_for_exchange_cooldown(exchange)
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                raise ExchangeError(f"No OHLCV data returned for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except ccxt.NetworkError as e:
            delay = _compute_backoff_seconds(attempt, e)
            logger.warning(
                f"Network error while fetching OHLCV {symbol} {timeframe}: {e}; "
                f"retry in {delay:.2f}s (attempt {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(delay)
            
        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection, ccxt.ExchangeNotAvailable, ccxt.ExchangeError) as e:
            if _is_rate_limit_error(e):
                delay = _compute_backoff_seconds(attempt, e)
                _register_exchange_cooldown(exchange, delay)
                logger.warning(
                    f"Rate limit on {getattr(exchange, 'id', 'exchange')} while fetching {symbol} {timeframe}: {e}; "
                    f"backoff {delay:.2f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(delay)
                continue
            if isinstance(e, ccxt.ExchangeError):
                logger.error(f"Exchange error fetching OHLCV for {symbol}: {e}")
                raise ExchangeError(f"Exchange error: {e}")
            delay = _compute_backoff_seconds(attempt, e)
            logger.warning(
                f"Temporary exchange error fetching OHLCV for {symbol}: {e}; "
                f"retry in {delay:.2f}s (attempt {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(delay)
            
        except Exception as e:
            logger.error(f"Unexpected error fetching OHLCV for {symbol}: {e}")
            raise ExchangeError(f"Unexpected error: {e}")
    
    raise ExchangeError(f"Failed to fetch OHLCV after {MAX_RETRIES} attempts")


def fetch_ticker(exchange: Any, symbol: str) -> Dict:
    """
    Fetch current ticker data for a symbol.
    
    Args:
        exchange: ccxt exchange instance
        symbol: Trading pair symbol
        
    Returns:
        Dictionary with 'last', 'bid', 'ask' prices
    """
    if hasattr(exchange, "id") and str(getattr(exchange, "id", "")).lower() in {"mt5", "mt5_bridge"}:
        try:
            ticker = exchange.fetch_ticker(symbol)
            result = {
                'last': ticker.get('last'),
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'high': ticker.get('high'),
                'low': ticker.get('low'),
                'volume': ticker.get('baseVolume') if ticker.get('baseVolume') is not None else ticker.get('volume'),
                'spread': ticker.get('spread'),
                'tick_time': ticker.get('tick_time'),
                'symbol_info': ticker.get('symbol_info'),
                'timestamp': datetime.now(timezone.utc)
            }
            logger.debug(
                f"Ticker via {str(getattr(exchange, 'id', 'mt5')).lower()} for {symbol}: "
                f"last={result['last']}, bid={result['bid']}, ask={result['ask']}"
            )
            return result
        except Exception as e:
            raise ExchangeError(f"MT5 ticker error for {symbol}: {e}")

    for attempt in range(MAX_RETRIES):
        try:
            _wait_for_exchange_cooldown(exchange)
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
            ticker = exchange.fetch_ticker(symbol)
            
            result = {
                'last': ticker.get('last'),
                'bid': ticker.get('bid'),
                'ask': ticker.get('ask'),
                'high': ticker.get('high'),
                'low': ticker.get('low'),
                'volume': ticker.get('baseVolume'),
                'timestamp': datetime.now(timezone.utc)
            }
            
            logger.debug(f"Ticker for {symbol}: last={result['last']}, bid={result['bid']}, ask={result['ask']}")
            return result
            
        except ccxt.NetworkError as e:
            delay = _compute_backoff_seconds(attempt, e)
            logger.warning(
                f"Network error fetching ticker for {symbol}: {e}; "
                f"retry in {delay:.2f}s (attempt {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(delay)

        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection, ccxt.ExchangeNotAvailable, ccxt.ExchangeError) as e:
            if _is_rate_limit_error(e):
                delay = _compute_backoff_seconds(attempt, e)
                _register_exchange_cooldown(exchange, delay)
                logger.warning(
                    f"Rate limit on {getattr(exchange, 'id', 'exchange')} fetching ticker {symbol}: {e}; "
                    f"backoff {delay:.2f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(delay)
                continue
            if isinstance(e, ccxt.ExchangeError):
                logger.error(f"Exchange error fetching ticker for {symbol}: {e}")
                raise ExchangeError(f"Failed to fetch ticker: {e}")
            delay = _compute_backoff_seconds(attempt, e)
            logger.warning(
                f"Temporary exchange error fetching ticker {symbol}: {e}; "
                f"retry in {delay:.2f}s (attempt {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(delay)
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise ExchangeError(f"Failed to fetch ticker: {e}")
    
    raise ExchangeError(f"Failed to fetch ticker after {MAX_RETRIES} attempts")


def fetch_multiple_timeframes(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframes: List[str],
    limit: int = 500
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple timeframes.
    
    Args:
        exchange: ccxt exchange instance
        symbol: Trading pair symbol
        timeframes: List of timeframes to fetch
        limit: Number of candles per timeframe
        
    Returns:
        Dictionary mapping timeframe to DataFrame
    """
    result = {}
    
    for tf in timeframes:
        try:
            result[tf] = fetch_ohlcv(exchange, symbol, tf, limit)
        except ExchangeError as e:
            logger.error(f"Failed to fetch {tf} data for {symbol}: {e}")
            result[tf] = pd.DataFrame()
    
    return result


def get_symbol_info(exchange: ccxt.Exchange, symbol: str) -> Optional[Dict]:
    """
    Get market information for a symbol.
    
    Args:
        exchange: ccxt exchange instance
        symbol: Trading pair symbol
        
    Returns:
        Market info dictionary or None
    """
    try:
        if hasattr(exchange, "id") and str(getattr(exchange, "id", "")).lower() in {"mt5", "mt5_bridge"}:
            info = exchange.get_symbol_info(symbol) if hasattr(exchange, 'get_symbol_info') else {}
            return {
                'symbol': symbol,
                'base': info.get('currency_base'),
                'quote': info.get('currency_profit') or info.get('currency_margin'),
                'tick_size': info.get('point'),
                'min_qty': info.get('volume_min'),
                'contract_size': info.get('trade_contract_size', 1),
                'spread': info.get('spread'),
                'digits': info.get('digits'),
            }
        if symbol in exchange.markets:
            market = exchange.markets[symbol]
            return {
                'symbol': symbol,
                'base': market.get('base'),
                'quote': market.get('quote'),
                'tick_size': market.get('precision', {}).get('price'),
                'min_qty': market.get('limits', {}).get('amount', {}).get('min'),
                'contract_size': market.get('contractSize', 1)
            }
        return None
    except Exception as e:
        logger.error(f"Error getting symbol info for {symbol}: {e}")
        return None
