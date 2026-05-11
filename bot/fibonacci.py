"""
Fibonacci Analysis
==================
Fibonacci Retracement and Extension calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class FibLevel:
    """Represents a Fibonacci level."""
    ratio: float
    price: float
    type: str  # 'retracement' or 'extension'


@dataclass
class FibAnalysis:
    """Complete Fibonacci analysis result."""
    swing_high: float
    swing_low: float
    levels: List[FibLevel]
    nearest_level: Optional[FibLevel]
    distance_pct: float  # Distance to nearest level from current price (percentage)


def calculate_fib_levels(
    df: pd.DataFrame,
    trend: str,
    lookback: int = 100
) -> Optional[FibAnalysis]:
    """
    Calculate Fibonacci Retracement and Extension levels based on recent Swing High/Low.
    
    For Uptrend (Swing Low -> Swing High):
    - Retracements (Potential Support): 0.236, 0.382, 0.5, 0.618, 0.786 back down from High
    - Extensions (Potential Resistance/Targets): 1.272, 1.618 projecting up from Low
    
    For Downtrend (Swing High -> Swing Low):
    - Retracements (Potential Resistance): 0.236, 0.382, 0.5, 0.618, 0.786 back up from Low
    - Extensions (Potential Support/Targets): 1.272, 1.618 projecting down from High
    
    Args:
        df: DataFrame with OHLCV data
        trend: Current trend ('up', 'down', or 'neutral')
        lookback: Period to find swing points
        
    Returns:
        FibAnalysis object or None
    """
    if df.empty or len(df) < lookback:
        return None
    
    recent = df.tail(lookback)
    current_price = df['close'].iloc[-1]
    
    # Identify Swing Points in the lookback period
    high_idx = recent['high'].idxmax()
    low_idx = recent['low'].idxmin()
    
    swing_high = recent.loc[high_idx, 'high']
    swing_low = recent.loc[low_idx, 'low']
    
    if swing_high == swing_low:
        return None
        
    diff = swing_high - swing_low
    levels = []
    
    retracement_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    extension_ratios = [1.272, 1.618, 2.0, 2.618]
    
    if trend == 'up':
        # Uptrend: Measure move from Last Major Low to Last Major High
        # Retracements are levels price pulls back TO (below High)
        for r in retracement_ratios:
            price = swing_high - (diff * r)
            levels.append(FibLevel(r, price, 'retracement'))
            
        # Extensions are targets above High
        for e in extension_ratios:
            price = swing_low + (diff * e) 
            levels.append(FibLevel(e, price, 'extension'))
            
    elif trend == 'down':
        # Downtrend: Measure move from Last Major High to Last Major Low
        # Retracements are levels price bounces up TO (above Low)
        for r in retracement_ratios:
            price = swing_low + (diff * r)
            levels.append(FibLevel(r, price, 'retracement'))
            
        # Extensions are targets below Low
        for e in extension_ratios:
            price = swing_high - (diff * e)
            levels.append(FibLevel(e, price, 'extension'))
    
    else: 
        # Neutral - Can calculate retracements of the range for reference
        # Range Retracements: levels inside the range
        for r in retracement_ratios:
            price_up = swing_low + (diff * r) # Levels from bottom up
            levels.append(FibLevel(r, price_up, 'retracement_bearish'))
            
            price_down = swing_high - (diff * r) # Levels from top down
            levels.append(FibLevel(r, price_down, 'retracement_bullish'))
    
    # Find nearest level
    nearest = None
    min_dist = float('inf')
    
    for lvl in levels:
        dist = abs(current_price - lvl.price)
        if dist < min_dist:
            min_dist = dist
            nearest = lvl
            
    dist_pct = (min_dist / current_price) * 100 if nearest else float('inf')
    
    # Filter to only keep relevant levels? Or return all? 
    # For now return all.
    
    return FibAnalysis(
        swing_high=swing_high,
        swing_low=swing_low,
        levels=levels,
        nearest_level=nearest,
        distance_pct=dist_pct
    )
