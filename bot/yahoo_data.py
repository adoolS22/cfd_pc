"""
Yahoo Finance Data Fetcher
==========================
Fetches macro/commodity OHLCV data from Yahoo Finance.
"""

import pandas as pd
import yfinance as yf
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from loguru import logger


# Yahoo Finance symbols
YAHOO_SYMBOLS = {
    # Metals
    "XAU/USD": "GC=F",      # Gold Futures
    "XAUUSD": "GC=F",
    "XAG/USD": "SI=F",      # Silver Futures
    "XAGUSD": "SI=F",
    # Oil
    "OIL/USD": "CL=F",      # WTI Crude Futures
    "WTI/USD": "CL=F",
    "USOIL": "CL=F",
    "USOILM": "CL=F",
    "CRUDEOIL": "CL=F",
    "CRUDE": "CL=F",
    "BRENT/USD": "BZ=F",    # Brent Crude Futures
    # US equities index
    "SNP500": "ES=F",       # E-mini S&P 500 Futures
    "SPX500": "ES=F",
    "S&P500": "ES=F",
    "SP500": "ES=F",
    "US500": "ES=F",
    "USTEC": "NQ=F",        # Nasdaq 100 Futures
    "NASDAQ": "NQ=F",
    "US30": "YM=F",         # Dow Jones Futures
    "DOW": "YM=F",
    # FX
    "EURUSD": "EURUSD=X",   # EUR/USD spot FX
    "EUR/USD": "EURUSD=X",
    "EUR-USD": "EURUSD=X",
}

# Timeframe mapping: our format -> Yahoo Finance interval
TIMEFRAME_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "1h",  # Yahoo doesn't have 4h, we'll resample
    "1d": "1d",
}


_CACHE_LOCK = threading.Lock()
_OHLCV_CACHE: Dict[str, Dict[str, Any]] = {}
_PRICE_CACHE: Dict[str, Dict[str, Any]] = {}


def _timeframe_minutes(timeframe: str) -> int:
    return {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }.get(str(timeframe or "1h").strip().lower(), 60)


def _ohlcv_cache_ttl_seconds(timeframe: str) -> int:
    tf = str(timeframe or "").strip().lower()
    if tf == "1m":
        return 18
    if tf == "5m":
        return 45
    if tf == "15m":
        return 90
    if tf == "30m":
        return 140
    if tf in {"1h", "4h"}:
        return 300
    if tf == "1d":
        return 1800
    return 120


def _get_cached_ohlcv(key: str, max_age_sec: int) -> Optional[pd.DataFrame]:
    now = time.time()
    with _CACHE_LOCK:
        item = _OHLCV_CACHE.get(key)
        if not item:
            return None
        age = now - float(item.get("ts", 0.0) or 0.0)
        if age > float(max_age_sec):
            return None
        cached_df = item.get("df")
        if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
            return cached_df.copy()
    return None


def _set_cached_ohlcv(key: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    with _CACHE_LOCK:
        _OHLCV_CACHE[key] = {"ts": time.time(), "df": df.copy()}


def _get_cached_price(key: str, max_age_sec: int) -> Optional[float]:
    now = time.time()
    with _CACHE_LOCK:
        item = _PRICE_CACHE.get(key)
        if not item:
            return None
        age = now - float(item.get("ts", 0.0) or 0.0)
        if age > float(max_age_sec):
            return None
        price = item.get("price")
        try:
            return float(price)
        except Exception:
            return None


def _set_cached_price(key: str, price: Optional[float]) -> None:
    if price is None:
        return
    try:
        value = float(price)
    except Exception:
        return
    with _CACHE_LOCK:
        _PRICE_CACHE[key] = {"ts": time.time(), "price": value}


def _get_latest_cached_ohlcv_close(yahoo_ticker: str, max_age_sec: int = 1800) -> Optional[float]:
    """Best-effort fallback: read last close from freshest cached OHLCV payload."""
    prefix = f"{str(yahoo_ticker).strip()}|"
    now = time.time()
    latest_ts = 0.0
    latest_price: Optional[float] = None

    with _CACHE_LOCK:
        for key, item in _OHLCV_CACHE.items():
            if not str(key).startswith(prefix):
                continue

            ts = float(item.get("ts", 0.0) or 0.0)
            if (now - ts) > float(max_age_sec):
                continue

            df = item.get("df")
            if not isinstance(df, pd.DataFrame) or df.empty or "close" not in df.columns:
                continue

            close_value = df["close"].iloc[-1]
            if pd.isna(close_value):
                continue

            try:
                price = float(close_value)
            except Exception:
                continue

            if ts >= latest_ts:
                latest_ts = ts
                latest_price = price

    return latest_price


def is_yahoo_symbol(symbol: str) -> bool:
    """Check if symbol should be fetched from Yahoo Finance."""
    if not symbol:
        return False
    key = str(symbol).strip().upper().replace(" ", "")
    return key in YAHOO_SYMBOLS


def get_yahoo_ticker(symbol: str) -> str:
    """Convert our symbol format to Yahoo Finance ticker."""
    key = str(symbol).strip().upper().replace(" ", "")
    return YAHOO_SYMBOLS.get(key, symbol)


def fetch_yahoo_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 500
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.
    
    Args:
        symbol: Trading symbol (e.g., "XAU/USD")
        timeframe: Timeframe string (e.g., "1h", "4h", "15m")
        limit: Number of bars to fetch
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    cache_key = f"{get_yahoo_ticker(symbol)}|{timeframe}|{int(limit)}"
    fresh_ttl = _ohlcv_cache_ttl_seconds(timeframe)
    cached = _get_cached_ohlcv(cache_key, fresh_ttl)
    if cached is not None:
        return cached

    attempts = 2
    try:
        yahoo_ticker = get_yahoo_ticker(symbol)
        tf_minutes = _timeframe_minutes(timeframe)

        for attempt in range(attempts):
            logger.debug(
                f"Fetching {symbol} ({yahoo_ticker}) from Yahoo Finance, tf={timeframe}, "
                f"attempt={attempt + 1}/{attempts}"
            )

            # Calculate period based on timeframe and limit
            minutes_needed = tf_minutes * int(limit)
            days_needed = max(7, int(minutes_needed / 1440) + 5)

            # Yahoo Finance limits
            if timeframe in ["1m", "5m"]:
                days_needed = min(days_needed, 7)
            elif timeframe in ["15m", "30m"]:
                days_needed = min(days_needed, 60)
            else:
                days_needed = min(days_needed, 730)

            ticker = yf.Ticker(yahoo_ticker)
            yahoo_interval = TIMEFRAME_MAP.get(timeframe, "1h")

            # For 4h, fetch 1h and resample
            if timeframe == "4h":
                yahoo_interval = "1h"
                days_needed = min(days_needed * 4, 730)

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_needed)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=yahoo_interval,
                auto_adjust=True
            )

            if df.empty:
                if attempt < attempts - 1:
                    time.sleep(0.8 + (attempt * 0.5))
                    continue
                raise ValueError(f"No data returned from Yahoo Finance for {symbol}")

            # Rename columns to match our format
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # Handle datetime column name (can be 'date' or 'datetime')
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            elif 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'timestamp'})

            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Remove timezone info if present
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            # Select and order columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'volume':
                        df['volume'] = 0
                    else:
                        raise ValueError(f"Missing required column: {col}")

            df = df[required_cols]

            # Resample to 4h if needed
            if timeframe == "4h":
                df = df.set_index('timestamp')
                df = df.resample('4h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna().reset_index()

            # Limit rows
            df = df.tail(limit)
            df.set_index('timestamp', inplace=True)
            _set_cached_ohlcv(cache_key, df)
            logger.info(f"Fetched {len(df)} bars for {symbol} from Yahoo Finance")
            return df

        raise ValueError(f"No data returned from Yahoo Finance for {symbol}")
    except Exception as e:
        logger.error(f"Error fetching {symbol} from Yahoo Finance: {e}")
        fallback = _get_cached_ohlcv(cache_key, max(120, fresh_ttl * 10))
        if fallback is not None:
            logger.warning(f"Using cached Yahoo OHLCV for {symbol} tf={timeframe} due to fetch error")
            return fallback
        return pd.DataFrame()


def get_yahoo_price(symbol: str) -> Optional[float]:
    """
    Get current price for a Yahoo Finance symbol.
    
    Args:
        symbol: Trading symbol (e.g., "XAU/USD")
        
    Returns:
        Current price or None if error
    """
    cache_key = get_yahoo_ticker(symbol)
    fresh_ttl = 8
    cached_price = _get_cached_price(cache_key, fresh_ttl)
    if cached_price is not None:
        return cached_price

    yahoo_ticker = get_yahoo_ticker(symbol)
    ticker = yf.Ticker(yahoo_ticker)
    failures = []

    def _history_last_close(period: str, interval: str) -> Optional[float]:
        try:
            hist = ticker.history(period=period, interval=interval, auto_adjust=False)
        except Exception as exc:
            failures.append(f"history({period},{interval}) failed: {exc}")
            return None

        if hist is None or hist.empty:
            failures.append(f"history({period},{interval}) empty")
            return None

        close_col = "Close" if "Close" in hist.columns else ("close" if "close" in hist.columns else None)
        if close_col is None:
            failures.append(f"history({period},{interval}) missing close")
            return None

        last_close = hist[close_col].iloc[-1]
        if pd.isna(last_close):
            failures.append(f"history({period},{interval}) last close NaN")
            return None

        try:
            return float(last_close)
        except Exception as exc:
            failures.append(f"history({period},{interval}) cast failed: {exc}")
            return None

    def _mapping_price(data: Any, source: str) -> Optional[float]:
        if data is None or not hasattr(data, "get"):
            failures.append(f"{source} unavailable")
            return None

        for key in ("lastPrice", "last_price", "regularMarketPrice", "previousClose"):
            try:
                value = data.get(key)
            except Exception:
                value = None

            if value is None or pd.isna(value):
                continue

            try:
                return float(value)
            except Exception:
                continue

        failures.append(f"{source} missing price fields")
        return None

    # Primary source: high-resolution intraday feed
    for period, interval in (("1d", "1m"), ("5d", "1h")):
        price = _history_last_close(period, interval)
        if price is not None:
            _set_cached_price(cache_key, price)
            return price

    # Secondary source: lightweight ticker metadata
    try:
        price = _mapping_price(ticker.fast_info, "fast_info")
        if price is not None:
            _set_cached_price(cache_key, price)
            return price
    except Exception as exc:
        failures.append(f"fast_info failed: {exc}")

    # Tertiary source: full ticker info (slower and less reliable)
    try:
        price = _mapping_price(ticker.info, "info")
        if price is not None:
            _set_cached_price(cache_key, price)
            return price
    except Exception as exc:
        failures.append(f"info failed: {exc}")

    # Final fallbacks: cached values should keep the bot operating.
    stale = _get_cached_price(cache_key, max(120, fresh_ttl * 75))
    if stale is not None:
        logger.warning(f"Using stale cached Yahoo price for {symbol}; live fetch unavailable")
        return stale

    ohlcv_price = _get_latest_cached_ohlcv_close(yahoo_ticker, max_age_sec=1800)
    if ohlcv_price is not None:
        _set_cached_price(cache_key, ohlcv_price)
        logger.warning(f"Using cached OHLCV close as Yahoo price fallback for {symbol}")
        return ohlcv_price

    if failures:
        summary = "; ".join(failures[:3])
        if len(failures) > 3:
            summary = f"{summary}; +{len(failures) - 3} more"
        logger.error(f"Error getting price for {symbol}: {summary}")
    else:
        logger.error(f"Error getting price for {symbol}: no price sources available")
    return None
