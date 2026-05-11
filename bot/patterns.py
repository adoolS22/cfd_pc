"""
Pattern Detection
=================
Fractal and candlestick pattern detection.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass
from loguru import logger


@dataclass
class Fractal:
    """Represents a fractal point."""
    type: str  # 'high' or 'low'
    price: float
    timestamp: pd.Timestamp
    index: int


@dataclass
class CandlePattern:
    """Represents a detected candlestick pattern."""
    name: str
    direction: str  # 'bullish' or 'bearish'
    strength: int  # 1-3
    timestamp: pd.Timestamp
    details: Dict


# =============================================================================
# Fractal Detection
# =============================================================================

def detect_fractals(df: pd.DataFrame, window: int = 2) -> List[Fractal]:
    """
    Detect Williams fractals.
    
    Fractal High: high is higher than 'window' bars before and after
    Fractal Low: low is lower than 'window' bars before and after
    
    Args:
        df: DataFrame with 'high' and 'low' columns
        window: Number of bars on each side (default: 2)
        
    Returns:
        List of Fractal objects
    """
    fractals = []
    
    # Need full left window and at least one bar on the right.
    if len(df) < (window + 2):
        return fractals
    
    # Allow provisional right-edge fractals using the available right bars.
    # This keeps latest swing detection useful in live scanning.
    for i in range(window, len(df) - 1):
        left = df.iloc[max(0, i - window):i]
        right = df.iloc[i + 1:min(len(df), i + window + 1)]

        if right.empty:
            continue

        # Check fractal high
        high_val = df['high'].iloc[i]
        is_fractal_high = (high_val > left['high']).all() and (high_val > right['high']).all()
        if is_fractal_high:
            fractals.append(Fractal(
                type='high',
                price=high_val,
                timestamp=df.index[i],
                index=i
            ))
        
        # Check fractal low
        low_val = df['low'].iloc[i]
        is_fractal_low = (low_val < left['low']).all() and (low_val < right['low']).all()
        if is_fractal_low:
            fractals.append(Fractal(
                type='low',
                price=low_val,
                timestamp=df.index[i],
                index=i
            ))
    
    return fractals


def get_last_fractal(df: pd.DataFrame, fractal_type: str = 'any', window: int = 2) -> Optional[Fractal]:
    """
    Get the most recent fractal of specified type.
    
    Args:
        df: DataFrame with OHLCV data
        fractal_type: 'high', 'low', or 'any'
        window: Fractal detection window
        
    Returns:
        Most recent Fractal or None
    """
    fractals = detect_fractals(df, window)
    
    if not fractals:
        return None
    
    if fractal_type == 'any':
        return fractals[-1]
    
    filtered = [f for f in fractals if f.type == fractal_type]
    return filtered[-1] if filtered else None


# =============================================================================
# Candlestick Patterns
# =============================================================================

def _candle_body(open_: float, close: float) -> float:
    """Calculate candle body size."""
    return abs(close - open_)


def _candle_range(high: float, low: float) -> float:
    """Calculate candle range."""
    return high - low


def _upper_wick(high: float, open_: float, close: float) -> float:
    """Calculate upper wick size."""
    return high - max(open_, close)


def _lower_wick(low: float, open_: float, close: float) -> float:
    """Calculate lower wick size."""
    return min(open_, close) - low


def _is_bullish(open_: float, close: float) -> bool:
    """Check if candle is bullish."""
    return close > open_


def _is_bearish(open_: float, close: float) -> bool:
    """Check if candle is bearish."""
    return close < open_


def detect_bullish_engulfing(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Detect bullish engulfing pattern.
    
    Rules:
    - Previous candle is bearish
    - Current candle is bullish
    - Current body completely engulfs previous body
    - Current body is larger than previous body
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        CandlePattern if detected, None otherwise
    """
    if len(df) < 2:
        return None
    
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    prev_bullish = _is_bullish(prev['open'], prev['close'])
    curr_bullish = _is_bullish(curr['open'], curr['close'])
    
    if prev_bullish or not curr_bullish:
        return None
    
    # Current body must engulf previous body
    prev_body_low = min(prev['open'], prev['close'])
    prev_body_high = max(prev['open'], prev['close'])
    curr_body_low = min(curr['open'], curr['close'])
    curr_body_high = max(curr['open'], curr['close'])
    
    if curr_body_low <= prev_body_low and curr_body_high >= prev_body_high:
        curr_body = _candle_body(curr['open'], curr['close'])
        prev_body = _candle_body(prev['open'], prev['close'])
        
        if curr_body > prev_body:
            return CandlePattern(
                name='bullish_engulfing',
                direction='bullish',
                strength=2,
                timestamp=df.index[-1],
                details={
                    'prev_body': prev_body,
                    'curr_body': curr_body,
                    'engulf_ratio': curr_body / prev_body if prev_body > 0 else float('inf')
                }
            )
    
    return None


def detect_bearish_engulfing(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Detect bearish engulfing pattern.
    
    Rules:
    - Previous candle is bullish
    - Current candle is bearish
    - Current body completely engulfs previous body
    - Current body is larger than previous body
    """
    if len(df) < 2:
        return None
    
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    prev_bearish = _is_bearish(prev['open'], prev['close'])
    curr_bearish = _is_bearish(curr['open'], curr['close'])
    
    if prev_bearish or not curr_bearish:
        return None
    
    prev_body_low = min(prev['open'], prev['close'])
    prev_body_high = max(prev['open'], prev['close'])
    curr_body_low = min(curr['open'], curr['close'])
    curr_body_high = max(curr['open'], curr['close'])
    
    if curr_body_low <= prev_body_low and curr_body_high >= prev_body_high:
        curr_body = _candle_body(curr['open'], curr['close'])
        prev_body = _candle_body(prev['open'], prev['close'])
        
        if curr_body > prev_body:
            return CandlePattern(
                name='bearish_engulfing',
                direction='bearish',
                strength=2,
                timestamp=df.index[-1],
                details={
                    'prev_body': prev_body,
                    'curr_body': curr_body,
                    'engulf_ratio': curr_body / prev_body if prev_body > 0 else float('inf')
                }
            )
    
    return None


def detect_harami(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Detect Harami (inside-body reversal pattern, 2-candle).

    Bullish Harami:
    - Previous candle bearish with sizable body
    - Current candle bullish with smaller body fully inside previous body

    Bearish Harami:
    - Previous candle bullish with sizable body
    - Current candle bearish with smaller body fully inside previous body
    """
    if len(df) < 2:
        return None

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_open = float(prev["open"])
    prev_close = float(prev["close"])
    curr_open = float(curr["open"])
    curr_close = float(curr["close"])

    prev_range = _candle_range(float(prev["high"]), float(prev["low"]))
    if prev_range <= 0:
        return None

    prev_body = _candle_body(prev_open, prev_close)
    curr_body = _candle_body(curr_open, curr_close)
    if prev_body <= 0 or curr_body <= 0:
        return None

    # Require a meaningful first candle and a clearly smaller second candle.
    if (prev_body / prev_range) < 0.45:
        return None
    if curr_body > (prev_body * 0.60):
        return None

    prev_low = min(prev_open, prev_close)
    prev_high = max(prev_open, prev_close)
    curr_low = min(curr_open, curr_close)
    curr_high = max(curr_open, curr_close)

    if curr_low < prev_low or curr_high > prev_high:
        return None

    if _is_bearish(prev_open, prev_close) and _is_bullish(curr_open, curr_close):
        return CandlePattern(
            name="bullish_harami",
            direction="bullish",
            strength=2,
            timestamp=df.index[-1],
            details={
                "prev_body": prev_body,
                "curr_body": curr_body,
                "inside_ratio": curr_body / prev_body if prev_body > 0 else 0.0,
            },
        )

    if _is_bullish(prev_open, prev_close) and _is_bearish(curr_open, curr_close):
        return CandlePattern(
            name="bearish_harami",
            direction="bearish",
            strength=2,
            timestamp=df.index[-1],
            details={
                "prev_body": prev_body,
                "curr_body": curr_body,
                "inside_ratio": curr_body / prev_body if prev_body > 0 else 0.0,
            },
        )

    return None


def detect_piercing_line(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Detect Piercing Line (bullish 2-candle reversal).

    Adapted for markets with weak/no gaps:
    - Candle 1 bearish with strong body
    - Candle 2 bullish
    - Candle 2 opens near/below prior close
    - Candle 2 closes above midpoint of candle 1 body
    - Candle 2 closes below candle 1 open (otherwise it becomes engulfing-like)
    """
    if len(df) < 2:
        return None

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_open = float(prev["open"])
    prev_close = float(prev["close"])
    curr_open = float(curr["open"])
    curr_close = float(curr["close"])

    if not _is_bearish(prev_open, prev_close) or not _is_bullish(curr_open, curr_close):
        return None

    prev_range = _candle_range(float(prev["high"]), float(prev["low"]))
    prev_body = _candle_body(prev_open, prev_close)
    if prev_range <= 0 or prev_body <= 0:
        return None
    if (prev_body / prev_range) < 0.50:
        return None

    midpoint = (prev_open + prev_close) / 2.0

    near_or_below_prev_close = curr_open <= (prev_close + (prev_body * 0.20))
    pierced_midpoint = curr_close > midpoint
    not_full_engulf = curr_close < prev_open

    if near_or_below_prev_close and pierced_midpoint and not_full_engulf:
        return CandlePattern(
            name="piercing_line",
            direction="bullish",
            strength=2,
            timestamp=df.index[-1],
            details={
                "midpoint_prev_body": midpoint,
                "close_above_midpoint_pct": ((curr_close - midpoint) / max(prev_body, 1e-10)) * 100.0,
            },
        )

    return None


def detect_dark_cloud_cover(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Detect Dark Cloud Cover (bearish 2-candle reversal).

    Adapted for markets with weak/no gaps:
    - Candle 1 bullish with strong body
    - Candle 2 bearish
    - Candle 2 opens near/above prior close
    - Candle 2 closes below midpoint of candle 1 body
    - Candle 2 closes above candle 1 open (otherwise it becomes engulfing-like)
    """
    if len(df) < 2:
        return None

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_open = float(prev["open"])
    prev_close = float(prev["close"])
    curr_open = float(curr["open"])
    curr_close = float(curr["close"])

    if not _is_bullish(prev_open, prev_close) or not _is_bearish(curr_open, curr_close):
        return None

    prev_range = _candle_range(float(prev["high"]), float(prev["low"]))
    prev_body = _candle_body(prev_open, prev_close)
    if prev_range <= 0 or prev_body <= 0:
        return None
    if (prev_body / prev_range) < 0.50:
        return None

    midpoint = (prev_open + prev_close) / 2.0

    near_or_above_prev_close = curr_open >= (prev_close - (prev_body * 0.20))
    closed_below_midpoint = curr_close < midpoint
    not_full_engulf = curr_close > prev_open

    if near_or_above_prev_close and closed_below_midpoint and not_full_engulf:
        return CandlePattern(
            name="dark_cloud_cover",
            direction="bearish",
            strength=2,
            timestamp=df.index[-1],
            details={
                "midpoint_prev_body": midpoint,
                "close_below_midpoint_pct": ((midpoint - curr_close) / max(prev_body, 1e-10)) * 100.0,
            },
        )

    return None


def detect_hammer(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Detect hammer / pin bar pattern (bullish).
    
    Rules:
    - Small body (< 30% of range)
    - Long lower wick (> 60% of range)
    - Small upper wick (< 20% of range)
    - Should appear after downtrend
    """
    if len(df) < 3:
        return None
    
    curr = df.iloc[-1]
    
    range_ = _candle_range(curr['high'], curr['low'])
    if range_ == 0:
        return None
    
    body = _candle_body(curr['open'], curr['close'])
    upper = _upper_wick(curr['high'], curr['open'], curr['close'])
    lower = _lower_wick(curr['low'], curr['open'], curr['close'])
    
    body_ratio = body / range_
    upper_ratio = upper / range_
    lower_ratio = lower / range_
    
    # Check hammer criteria
    if body_ratio < 0.30 and lower_ratio > 0.60 and upper_ratio < 0.20:
        # Check for preceding downtrend
        prev_closes = df['close'].iloc[-4:-1]
        if prev_closes.iloc[0] > prev_closes.iloc[-1]:  # Downtrend
            return CandlePattern(
                name='hammer',
                direction='bullish',
                strength=2,
                timestamp=df.index[-1],
                details={
                    'body_ratio': body_ratio,
                    'lower_wick_ratio': lower_ratio,
                    'upper_wick_ratio': upper_ratio
                }
            )
    
    return None


def detect_shooting_star(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Detect shooting star pattern (bearish).
    
    Rules:
    - Small body (< 30% of range)
    - Long upper wick (> 60% of range)
    - Small lower wick (< 20% of range)
    - Should appear after uptrend
    """
    if len(df) < 3:
        return None
    
    curr = df.iloc[-1]
    
    range_ = _candle_range(curr['high'], curr['low'])
    if range_ == 0:
        return None
    
    body = _candle_body(curr['open'], curr['close'])
    upper = _upper_wick(curr['high'], curr['open'], curr['close'])
    lower = _lower_wick(curr['low'], curr['open'], curr['close'])
    
    body_ratio = body / range_
    upper_ratio = upper / range_
    lower_ratio = lower / range_
    
    # Check shooting star criteria
    if body_ratio < 0.30 and upper_ratio > 0.60 and lower_ratio < 0.20:
        # Check for preceding uptrend
        prev_closes = df['close'].iloc[-4:-1]
        if prev_closes.iloc[0] < prev_closes.iloc[-1]:  # Uptrend
            return CandlePattern(
                name='shooting_star',
                direction='bearish',
                strength=2,
                timestamp=df.index[-1],
                details={
                    'body_ratio': body_ratio,
                    'upper_wick_ratio': upper_ratio,
                    'lower_wick_ratio': lower_ratio
                }
            )
    
    return None


def detect_pin_bar(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Detect pin bar pattern (bidirectional).
    
    Rules:
    - Small body (< 25% of range)
    - One wick > 66% of range
    - Opposite wick < 15% of range
    """
    if len(df) < 2:
        return None
    
    curr = df.iloc[-1]
    
    range_ = _candle_range(curr['high'], curr['low'])
    if range_ == 0:
        return None
    
    body = _candle_body(curr['open'], curr['close'])
    upper = _upper_wick(curr['high'], curr['open'], curr['close'])
    lower = _lower_wick(curr['low'], curr['open'], curr['close'])
    
    body_ratio = body / range_
    upper_ratio = upper / range_
    lower_ratio = lower / range_
    
    if body_ratio > 0.25:
        return None
    
    # Bullish pin bar (long lower wick)
    if lower_ratio > 0.66 and upper_ratio < 0.15:
        return CandlePattern(
            name='pin_bar',
            direction='bullish',
            strength=2,
            timestamp=df.index[-1],
            details={
                'body_ratio': body_ratio,
                'tail_ratio': lower_ratio,
                'nose_ratio': upper_ratio
            }
        )
    
    # Bearish pin bar (long upper wick)
    if upper_ratio > 0.66 and lower_ratio < 0.15:
        return CandlePattern(
            name='pin_bar',
            direction='bearish',
            strength=2,
            timestamp=df.index[-1],
            details={
                'body_ratio': body_ratio,
                'tail_ratio': upper_ratio,
                'nose_ratio': lower_ratio
            }
        )
    
    return None


# =============================================================================
# Additional Candlestick Patterns
# =============================================================================

def detect_morning_star(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Detect Morning Star pattern (bullish reversal, 3-candle).

    Rules:
    - Candle 1: large bearish body
    - Candle 2: small body (indecision) that gaps down or has small range
    - Candle 3: large bullish body that closes above midpoint of candle 1
    """
    if len(df) < 4:
        return None

    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    r1 = _candle_range(c1['high'], c1['low'])
    r2 = _candle_range(c2['high'], c2['low'])
    r3 = _candle_range(c3['high'], c3['low'])
    if r1 == 0 or r3 == 0:
        return None

    b1 = _candle_body(c1['open'], c1['close'])
    b2 = _candle_body(c2['open'], c2['close'])
    b3 = _candle_body(c3['open'], c3['close'])

    if not _is_bearish(c1['open'], c1['close']):
        return None
    if not _is_bullish(c3['open'], c3['close']):
        return None
    if b1 / r1 < 0.5:
        return None
    if b2 / r2 > 0.4 if r2 > 0 else False:
        return None
    midpoint_c1 = (c1['open'] + c1['close']) / 2.0
    if c3['close'] < midpoint_c1:
        return None

    # Check preceding downtrend
    if len(df) >= 5:
        prior = df['close'].iloc[-5:-3]
        if prior.iloc[0] < prior.iloc[-1]:
            return None

    return CandlePattern(
        name='morning_star', direction='bullish', strength=3,
        timestamp=df.index[-1],
        details={'body1': b1, 'body2': b2, 'body3': b3}
    )


def detect_evening_star(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Detect Evening Star pattern (bearish reversal, 3-candle).
    Mirror of Morning Star.
    """
    if len(df) < 4:
        return None

    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    r1 = _candle_range(c1['high'], c1['low'])
    r2 = _candle_range(c2['high'], c2['low'])
    r3 = _candle_range(c3['high'], c3['low'])
    if r1 == 0 or r3 == 0:
        return None

    b1 = _candle_body(c1['open'], c1['close'])
    b2 = _candle_body(c2['open'], c2['close'])
    b3 = _candle_body(c3['open'], c3['close'])

    if not _is_bullish(c1['open'], c1['close']):
        return None
    if not _is_bearish(c3['open'], c3['close']):
        return None
    if b1 / r1 < 0.5:
        return None
    if b2 / r2 > 0.4 if r2 > 0 else False:
        return None
    midpoint_c1 = (c1['open'] + c1['close']) / 2.0
    if c3['close'] > midpoint_c1:
        return None

    if len(df) >= 5:
        prior = df['close'].iloc[-5:-3]
        if prior.iloc[0] > prior.iloc[-1]:
            return None

    return CandlePattern(
        name='evening_star', direction='bearish', strength=3,
        timestamp=df.index[-1],
        details={'body1': b1, 'body2': b2, 'body3': b3}
    )


def detect_three_white_soldiers(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Detect Three White Soldiers (strong bullish continuation).
    Three consecutive bullish candles, each closing higher with decent bodies.
    """
    if len(df) < 4:
        return None

    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    for c in [c1, c2, c3]:
        if not _is_bullish(c['open'], c['close']):
            return None

    if not (c3['close'] > c2['close'] > c1['close']):
        return None
    if not (c3['open'] > c2['open'] > c1['open']):
        return None

    for c in [c1, c2, c3]:
        rng = _candle_range(c['high'], c['low'])
        body = _candle_body(c['open'], c['close'])
        if rng > 0 and body / rng < 0.4:
            return None

    return CandlePattern(
        name='three_white_soldiers', direction='bullish', strength=3,
        timestamp=df.index[-1],
        details={'pattern': '3 consecutive bullish with higher closes'}
    )


def detect_three_black_crows(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Detect Three Black Crows (strong bearish continuation).
    Three consecutive bearish candles, each closing lower with decent bodies.
    """
    if len(df) < 4:
        return None

    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    for c in [c1, c2, c3]:
        if not _is_bearish(c['open'], c['close']):
            return None

    if not (c3['close'] < c2['close'] < c1['close']):
        return None
    if not (c3['open'] < c2['open'] < c1['open']):
        return None

    for c in [c1, c2, c3]:
        rng = _candle_range(c['high'], c['low'])
        body = _candle_body(c['open'], c['close'])
        if rng > 0 and body / rng < 0.4:
            return None

    return CandlePattern(
        name='three_black_crows', direction='bearish', strength=3,
        timestamp=df.index[-1],
        details={'pattern': '3 consecutive bearish with lower closes'}
    )


def detect_doji(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Detect Doji (indecision candle).
    Body < 10% of range. Can signal reversal at key zones.
    Returns bullish if preceded by downtrend, bearish if preceded by uptrend.
    """
    if len(df) < 4:
        return None

    curr = df.iloc[-1]
    rng = _candle_range(curr['high'], curr['low'])
    if rng == 0:
        return None

    body = _candle_body(curr['open'], curr['close'])
    if body / rng > 0.10:
        return None

    prev_closes = df['close'].iloc[-4:-1]
    if prev_closes.iloc[0] > prev_closes.iloc[-1]:
        direction = 'bullish'
    elif prev_closes.iloc[0] < prev_closes.iloc[-1]:
        direction = 'bearish'
    else:
        return None

    upper = _upper_wick(curr['high'], curr['open'], curr['close'])
    lower = _lower_wick(curr['low'], curr['open'], curr['close'])

    # Dragonfly doji (long lower wick, bullish)
    if rng > 0 and lower / rng > 0.65 and upper / rng < 0.10:
        return CandlePattern(
            name='dragonfly_doji', direction='bullish', strength=2,
            timestamp=df.index[-1],
            details={'body_ratio': body / rng, 'lower_ratio': lower / rng}
        )

    # Gravestone doji (long upper wick, bearish)
    if rng > 0 and upper / rng > 0.65 and lower / rng < 0.10:
        return CandlePattern(
            name='gravestone_doji', direction='bearish', strength=2,
            timestamp=df.index[-1],
            details={'body_ratio': body / rng, 'upper_ratio': upper / rng}
        )

    return CandlePattern(
        name='doji', direction=direction, strength=1,
        timestamp=df.index[-1],
        details={'body_ratio': body / rng}
    )


# =============================================================================
# Trendline & Structure Detection
# =============================================================================

def detect_trendline_break(df: pd.DataFrame, lookback: int = 20) -> Optional[Dict]:
    """
    Detect trendline breaks by connecting recent swing highs/lows.

    This is what traders do visually: draw a line connecting 2+ swing highs
    or swing lows, then watch for price to break through.

    Returns:
        Dict with 'type' ('bullish_break'/'bearish_break'), 'strength', 'trendline_slope'
        or None if no break detected.
    """
    if len(df) < lookback:
        return None

    window = df.iloc[-lookback:]
    close = float(window['close'].iloc[-1])

    # Find swing highs and swing lows (simplified: local max/min over 3 bars)
    swing_highs = []
    swing_lows = []
    for i in range(2, len(window) - 1):
        h = window['high'].iloc[i]
        if h > window['high'].iloc[i-1] and h > window['high'].iloc[i-2] and \
           h > window['high'].iloc[i+1 if i+1 < len(window) else i]:
            swing_highs.append((i, float(h)))
        l = window['low'].iloc[i]
        if l < window['low'].iloc[i-1] and l < window['low'].iloc[i-2] and \
           l < window['low'].iloc[i+1 if i+1 < len(window) else i]:
            swing_lows.append((i, float(l)))

    # Descending trendline break (bullish): connect 2 recent swing highs,
    # if close is above the projected trendline = breakout
    if len(swing_highs) >= 2:
        sh1, sh2 = swing_highs[-2], swing_highs[-1]
        if sh2[0] > sh1[0] and sh2[1] < sh1[1]:  # descending highs
            bars_since = len(window) - 1 - sh2[0]
            slope = (sh2[1] - sh1[1]) / max(1, sh2[0] - sh1[0])
            projected = sh2[1] + slope * bars_since
            if close > projected and close > sh2[1]:
                return {
                    'type': 'bullish_break',
                    'strength': 2,
                    'slope': slope,
                    'projected_level': projected,
                    'break_pct': (close - projected) / projected * 100 if projected > 0 else 0,
                }

    # Ascending trendline break (bearish): connect 2 recent swing lows,
    # if close is below the projected trendline = breakdown
    if len(swing_lows) >= 2:
        sl1, sl2 = swing_lows[-2], swing_lows[-1]
        if sl2[0] > sl1[0] and sl2[1] > sl1[1]:  # ascending lows
            bars_since = len(window) - 1 - sl2[0]
            slope = (sl2[1] - sl1[1]) / max(1, sl2[0] - sl1[0])
            projected = sl2[1] + slope * bars_since
            if close < projected and close < sl2[1]:
                return {
                    'type': 'bearish_break',
                    'strength': 2,
                    'slope': slope,
                    'projected_level': projected,
                    'break_pct': (projected - close) / projected * 100 if projected > 0 else 0,
                }

    return None


def detect_structure(df: pd.DataFrame, lookback: int = 30) -> Optional[Dict]:
    """
    Detect market structure: Higher Highs/Higher Lows (bullish) or
    Lower Highs/Lower Lows (bearish).

    This is the foundation of how traders "read" a chart — they identify
    if the structure is making HH/HL (buy) or LH/LL (sell).

    Returns:
        Dict with 'type' ('bullish_structure'/'bearish_structure'/'double_top'/'double_bottom'),
        'strength', and details, or None.
    """
    if len(df) < lookback:
        return None

    window = df.iloc[-lookback:]

    # Identify swing points
    swing_highs = []
    swing_lows = []
    for i in range(2, len(window) - 1):
        h = float(window['high'].iloc[i])
        if h > float(window['high'].iloc[i-1]) and h > float(window['high'].iloc[i+1 if i+1 < len(window) else i]):
            swing_highs.append(h)
        l = float(window['low'].iloc[i])
        if l < float(window['low'].iloc[i-1]) and l < float(window['low'].iloc[i+1 if i+1 < len(window) else i]):
            swing_lows.append(l)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    last_2h = swing_highs[-2:]
    last_2l = swing_lows[-2:]

    # Double Top: two highs within 0.3% of each other, bearish
    if abs(last_2h[1] - last_2h[0]) / max(last_2h[0], 1e-10) < 0.003:
        close = float(window['close'].iloc[-1])
        if close < min(last_2l):
            return {
                'type': 'double_top',
                'direction': 'bearish',
                'strength': 3,
                'levels': last_2h,
            }

    # Double Bottom: two lows within 0.3% of each other, bullish
    if abs(last_2l[1] - last_2l[0]) / max(last_2l[0], 1e-10) < 0.003:
        close = float(window['close'].iloc[-1])
        if close > max(last_2h):
            return {
                'type': 'double_bottom',
                'direction': 'bullish',
                'strength': 3,
                'levels': last_2l,
            }

    # Higher High + Higher Low = bullish structure
    hh = last_2h[1] > last_2h[0]
    hl = last_2l[1] > last_2l[0]
    if hh and hl:
        return {
            'type': 'bullish_structure',
            'direction': 'bullish',
            'strength': 2,
            'highs': last_2h,
            'lows': last_2l,
        }

    # Lower High + Lower Low = bearish structure
    lh_flag = last_2h[1] < last_2h[0]
    ll_flag = last_2l[1] < last_2l[0]
    if lh_flag and ll_flag:
        return {
            'type': 'bearish_structure',
            'direction': 'bearish',
            'strength': 2,
            'highs': last_2h,
            'lows': last_2l,
        }

    return None


def detect_all_patterns(df: pd.DataFrame, include_advisory: bool = False) -> List[CandlePattern]:
    """
    Detect all candlestick patterns.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        List of detected CandlePattern objects
    """
    patterns = []

    detectors = [
        detect_bullish_engulfing,
        detect_bearish_engulfing,
        detect_hammer,
        detect_shooting_star,
        detect_pin_bar,
        detect_morning_star,
        detect_evening_star,
        detect_three_white_soldiers,
        detect_three_black_crows,
        detect_doji,
    ]
    if include_advisory:
        detectors.extend([
            detect_harami,
            detect_piercing_line,
            detect_dark_cloud_cover,
        ])

    for detector in detectors:
        pattern = detector(df)
        if pattern:
            patterns.append(pattern)

    return patterns


def get_pattern_for_direction(df: pd.DataFrame, direction: str) -> Optional[CandlePattern]:
    """
    Get the strongest pattern matching the desired direction.
    
    Args:
        df: DataFrame with OHLCV data
        direction: 'bullish' or 'bearish'
        
    Returns:
        Strongest matching pattern or None
    """
    patterns = detect_all_patterns(df)
    matching = [p for p in patterns if p.direction == direction]
    
    if not matching:
        return None
    
    return max(matching, key=lambda p: p.strength)


def get_advisory_pattern_for_direction(df: pd.DataFrame, direction: str) -> Optional[CandlePattern]:
    """
    Advisory-only pattern picker.

    Includes additional classical patterns used for display sections
    without affecting execution scoring/filters.
    """
    patterns = detect_all_patterns(df, include_advisory=True)
    matching = [p for p in patterns if p.direction == direction]
    if not matching:
        return None
    return max(matching, key=lambda p: p.strength)


# =============================================================================
# Wave Logic (Simplified Wave 3 Detection)
# =============================================================================

def detect_wave3_setup(
    df_trend: pd.DataFrame,
    df_entry: pd.DataFrame,
    direction: str = 'long'
) -> Dict:
    """
    Detect simplified Wave 3 start conditions.
    
    Long Wave 3 Trigger:
    - Trend up (EMA50 > EMA200 on trend_tf)
    - Breakout above recent swing high on entry_tf
    - RSI(14) > 55
    - Volume spike

    Short Wave 3 is symmetrical.
    
    Args:
        df_trend: Trend timeframe DataFrame with indicators
        df_entry: Entry timeframe DataFrame with indicators
        direction: 'long' or 'short'
        
    Returns:
        Dict with 'triggered', 'reasons', 'score'
    """
    result = {
        'triggered': False,
        'reasons': [],
        'score': 0,
        'conditions': {}
    }
    
    if df_trend.empty or df_entry.empty:
        return result
    
    # Get current values
    ema50_trend = df_trend['ema_50'].iloc[-1] if 'ema_50' in df_trend else None
    ema200_trend = df_trend['ema_200'].iloc[-1] if 'ema_200' in df_trend else None
    rsi_entry = df_entry['rsi_14'].iloc[-1] if 'rsi_14' in df_entry else None
    volume_spike = df_entry['volume_spike'].iloc[-1] if 'volume_spike' in df_entry else False
    
    conditions_met = 0
    
    if direction == 'long':
        # Trend condition
        if ema50_trend and ema200_trend and ema50_trend > ema200_trend:
            conditions_met += 1
            result['reasons'].append("Trend up (EMA50>EMA200)")
            result['conditions']['trend'] = True
        
        # RSI condition
        if rsi_entry and rsi_entry > 55:
            conditions_met += 1
            result['reasons'].append(f"RSI={rsi_entry:.1f} > 55")
            result['conditions']['rsi'] = True
        
        # Volume spike
        if volume_spike:
            conditions_met += 1
            result['reasons'].append("Volume spike")
            result['conditions']['volume'] = True
        
        # Breakout structure: close breaks above recent 5-bar high
        if len(df_entry) >= 6:
            recent_high = df_entry['high'].iloc[-6:-1].max()
            close_now = df_entry['close'].iloc[-1]
            if close_now > recent_high:
                conditions_met += 1
                result['reasons'].append("Breakout above recent high")
                result['conditions']['breakout'] = True
    
    else:  # short
        # Trend condition
        if ema50_trend and ema200_trend and ema50_trend < ema200_trend:
            conditions_met += 1
            result['reasons'].append("Trend down (EMA50<EMA200)")
            result['conditions']['trend'] = True
        
        # RSI condition
        if rsi_entry and rsi_entry < 45:
            conditions_met += 1
            result['reasons'].append(f"RSI={rsi_entry:.1f} < 45")
            result['conditions']['rsi'] = True
        
        # Volume spike
        if volume_spike:
            conditions_met += 1
            result['reasons'].append("Volume spike")
            result['conditions']['volume'] = True
        
        # Breakdown structure: close breaks below recent 5-bar low
        if len(df_entry) >= 6:
            recent_low = df_entry['low'].iloc[-6:-1].min()
            close_now = df_entry['close'].iloc[-1]
            if close_now < recent_low:
                conditions_met += 1
                result['reasons'].append("Breakdown below recent low")
                result['conditions']['breakout'] = True
    
    # Require at least 3 conditions for a trigger
    if conditions_met >= 3:
        result['triggered'] = True
        result['score'] = 2
    
    return result
