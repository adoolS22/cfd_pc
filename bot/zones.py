"""
Support/Resistance Zones Detection
===================================
Pivot-based zone detection for support and resistance levels.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class Zone:
    """Represents a support or resistance zone."""
    type: str  # 'support' or 'resistance'
    level: float
    upper: float
    lower: float
    strength: int  # Number of touches
    timestamp: pd.Timestamp


@dataclass
class ZoneProximity:
    """Result of zone proximity check."""
    in_zone: bool
    zone: Optional[Zone]
    distance_pct: float
    zone_type: Optional[str]


def detect_pivots(df: pd.DataFrame, window: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Detect pivot highs and lows.
    
    A pivot high is where the high is greater than 'window' bars before and after.
    A pivot low is where the low is lower than 'window' bars before and after.
    
    Args:
        df: DataFrame with 'high' and 'low' columns
        window: Number of bars on each side to compare
        
    Returns:
        Tuple of (pivot_highs, pivot_lows) boolean series
    """
    highs = df['high']
    lows = df['low']
    
    pivot_highs = pd.Series(False, index=df.index)
    pivot_lows = pd.Series(False, index=df.index)
    
    for i in range(window, len(df) - window):
        # Check pivot high
        is_pivot_high = True
        for j in range(1, window + 1):
            if highs.iloc[i] <= highs.iloc[i - j] or highs.iloc[i] <= highs.iloc[i + j]:
                is_pivot_high = False
                break
        pivot_highs.iloc[i] = is_pivot_high
        
        # Check pivot low
        is_pivot_low = True
        for j in range(1, window + 1):
            if lows.iloc[i] >= lows.iloc[i - j] or lows.iloc[i] >= lows.iloc[i + j]:
                is_pivot_low = False
                break
        pivot_lows.iloc[i] = is_pivot_low
    
    return pivot_highs, pivot_lows


def build_zones(
    df: pd.DataFrame,
    window: int = 3,
    zone_width_pct: float = 0.003,
    min_touches: int = 1,
    max_zones: int = 10
) -> List[Zone]:
    """
    Build support and resistance zones from pivot points.
    
    Args:
        df: DataFrame with OHLCV data
        window: Pivot detection window
        zone_width_pct: Zone width as percentage of price (default 0.3%)
        min_touches: Minimum touches to consider a zone valid
        max_zones: Maximum number of zones to return
        
    Returns:
        List of Zone objects sorted by proximity to current price
    """
    if df.empty:
        return []
    
    pivot_highs, pivot_lows = detect_pivots(df, window)
    
    zones = []
    current_price = df['close'].iloc[-1]
    
    # Extract resistance zones from pivot highs
    resistance_prices = df.loc[pivot_highs, 'high']
    for idx, price in resistance_prices.items():
        zone_width = price * zone_width_pct
        zones.append(Zone(
            type='resistance',
            level=price,
            upper=price + zone_width,
            lower=price - zone_width,
            strength=1,
            timestamp=idx
        ))
    
    # Extract support zones from pivot lows
    support_prices = df.loc[pivot_lows, 'low']
    for idx, price in support_prices.items():
        zone_width = price * zone_width_pct
        zones.append(Zone(
            type='support',
            level=price,
            upper=price + zone_width,
            lower=price - zone_width,
            strength=1,
            timestamp=idx
        ))
    
    # Merge overlapping zones and count touches
    merged_zones = _merge_zones(zones, zone_width_pct)
    
    # Filter by minimum touches
    merged_zones = [z for z in merged_zones if z.strength >= min_touches]
    
    # Sort by distance to current price and take top N
    merged_zones.sort(key=lambda z: abs(z.level - current_price))
    
    return merged_zones[:max_zones]


def _merge_zones(zones: List[Zone], merge_threshold_pct: float = 0.005) -> List[Zone]:
    """
    Merge overlapping zones.
    
    Args:
        zones: List of Zone objects
        merge_threshold_pct: Threshold for considering zones overlapping
        
    Returns:
        List of merged zones
    """
    if not zones:
        return []
    
    # Sort by level
    sorted_zones = sorted(zones, key=lambda z: z.level)
    merged = []
    
    current = sorted_zones[0]
    for zone in sorted_zones[1:]:
        # Check if zones overlap
        threshold = current.level * merge_threshold_pct
        if abs(zone.level - current.level) <= threshold:
            # Merge: average the levels, expand bounds, add strength
            new_level = (current.level + zone.level) / 2
            new_upper = max(current.upper, zone.upper)
            new_lower = min(current.lower, zone.lower)
            
            # Determine type based on position relative to average
            zone_type = current.type if current.strength >= zone.strength else zone.type
            
            current = Zone(
                type=zone_type,
                level=new_level,
                upper=new_upper,
                lower=new_lower,
                strength=current.strength + zone.strength,
                timestamp=max(current.timestamp, zone.timestamp)
            )
        else:
            merged.append(current)
            current = zone
    
    merged.append(current)
    return merged


def is_price_in_zone(price: float, zones: List[Zone]) -> ZoneProximity:
    """
    Check if price is within any zone.
    
    Args:
        price: Current price to check
        zones: List of zones
        
    Returns:
        ZoneProximity result with zone info if found
    """
    if not zones:
        return ZoneProximity(
            in_zone=False,
            zone=None,
            distance_pct=float('inf'),
            zone_type=None
        )
    
    closest_zone = None
    min_distance = float('inf')
    
    for zone in zones:
        # Check if price is within zone bounds
        if zone.lower <= price <= zone.upper:
            distance_pct = abs(price - zone.level) / zone.level
            return ZoneProximity(
                in_zone=True,
                zone=zone,
                distance_pct=distance_pct,
                zone_type=zone.type
            )
        
        # Track closest zone
        distance = min(abs(price - zone.upper), abs(price - zone.lower))
        distance_pct = distance / zone.level
        if distance_pct < min_distance:
            min_distance = distance_pct
            closest_zone = zone
    
    return ZoneProximity(
        in_zone=False,
        zone=closest_zone,
        distance_pct=min_distance,
        zone_type=closest_zone.type if closest_zone else None
    )


def get_nearest_support(price: float, zones: List[Zone]) -> Optional[Zone]:
    """
    Get the nearest support zone below current price.
    
    Args:
        price: Current price
        zones: List of zones
        
    Returns:
        Nearest support Zone or None
    """
    support_zones = [z for z in zones if z.type == 'support' and z.level < price]
    if not support_zones:
        return None
    return max(support_zones, key=lambda z: z.level)


def get_nearest_resistance(price: float, zones: List[Zone]) -> Optional[Zone]:
    """
    Get the nearest resistance zone above current price.
    
    Args:
        price: Current price
        zones: List of zones
        
    Returns:
        Nearest resistance Zone or None
    """
    resistance_zones = [z for z in zones if z.type == 'resistance' and z.level > price]
    if not resistance_zones:
        return None
    return min(resistance_zones, key=lambda z: z.level)


def get_swing_high(df: pd.DataFrame, lookback: int = 20) -> Optional[Tuple[float, pd.Timestamp]]:
    """
    Get the most recent swing high.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of bars to look back
        
    Returns:
        Tuple of (price, timestamp) or None
    """
    recent = df.tail(lookback)
    pivot_highs, _ = detect_pivots(recent, window=2)
    
    if pivot_highs.any():
        last_pivot_idx = pivot_highs[pivot_highs].index[-1]
        return (df.loc[last_pivot_idx, 'high'], last_pivot_idx)
    
    return None


def get_swing_low(df: pd.DataFrame, lookback: int = 20) -> Optional[Tuple[float, pd.Timestamp]]:
    """
    Get the most recent swing low.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of bars to look back
        
    Returns:
        Tuple of (price, timestamp) or None
    """
    recent = df.tail(lookback)
    _, pivot_lows = detect_pivots(recent, window=2)
    
    if pivot_lows.any():
        last_pivot_idx = pivot_lows[pivot_lows].index[-1]
        return (df.loc[last_pivot_idx, 'low'], last_pivot_idx)
    
    return None
