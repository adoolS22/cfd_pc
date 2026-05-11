"""
Gann Analysis
=============
Gann price angles and Square of 9 calculations.
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class GannAngle:
    """Represents a Gann angle line."""
    name: str  # e.g., "1x1", "2x1", "1x2"
    slope: float
    current_value: float
    relation: str  # 'above', 'below', 'at'


@dataclass
class GannAnalysis:
    """Complete Gann analysis result."""
    pivot_price: float
    pivot_time: pd.Timestamp
    angles: List[GannAngle]
    angle_confluence_score: float  # 0-2
    price_relation: str  # overall relation to angles


@dataclass
class Square9Level:
    """Represents a Square of 9 level."""
    level: float
    degrees: int
    distance_pct: float
    direction: str  # 'above' or 'below'


@dataclass
class Square9Analysis:
    """Complete Square of 9 analysis result."""
    base_price: float
    levels: List[Square9Level]
    nearest_level: Optional[Square9Level]
    square9_score: float  # 0-2


# =============================================================================
# Gann Price Angles
# =============================================================================

def calculate_gann_angles(
    df: pd.DataFrame,
    trend: str,
    atr_value: float,
    lookback: int = 50
) -> GannAnalysis:
    """
    Calculate Gann angle projections from a pivot point.
    
    For uptrend: use last swing low
    For downtrend: use last swing high
    
    Angles use ATR as normalized unit:
    - 1x1 = 1 ATR per bar (45 degrees)
    - 2x1 = 2 ATR per bar (steeper)
    - 1x2 = 0.5 ATR per bar (flatter)
    
    Args:
        df: DataFrame with OHLCV data
        trend: 'up' or 'down'
        atr_value: Current ATR(14) value
        lookback: Bars to look back for pivot
        
    Returns:
        GannAnalysis with angle projections
    """
    if df.empty or len(df) < lookback:
        return GannAnalysis(
            pivot_price=0,
            pivot_time=pd.Timestamp.now(tz='UTC'),
            angles=[],
            angle_confluence_score=0,
            price_relation='unknown'
        )
    
    recent = df.tail(lookback)
    current_price = df['close'].iloc[-1]
    current_idx = len(df) - 1
    
    # Find pivot based on trend
    if trend == 'up':
        pivot_idx_rel = recent['low'].idxmin()
        pivot_price = recent.loc[pivot_idx_rel, 'low']
    else:
        pivot_idx_rel = recent['high'].idxmax()
        pivot_price = recent.loc[pivot_idx_rel, 'high']
    
    pivot_time = pivot_idx_rel
    
    # Calculate bars since pivot
    pivot_position = recent.index.get_loc(pivot_idx_rel)
    bars_since_pivot = len(recent) - pivot_position - 1
    
    if bars_since_pivot <= 0:
        bars_since_pivot = 1
    
    # Define angle slopes (ATR per bar)
    angle_definitions = [
        ("1x1", 1.0),   # 45 degree
        ("2x1", 2.0),   # Steeper
        ("1x2", 0.5),   # Flatter
        ("4x1", 4.0),   # Very steep
        ("1x4", 0.25),  # Very flat
    ]
    
    angles = []
    
    for name, multiplier in angle_definitions:
        slope = atr_value * multiplier
        
        if trend == 'up':
            # Project upward from low
            angle_value = pivot_price + (slope * bars_since_pivot)
        else:
            # Project downward from high
            angle_value = pivot_price - (slope * bars_since_pivot)
        
        # Determine relation to current price
        tolerance = atr_value * 0.1  # 10% of ATR
        
        if abs(current_price - angle_value) <= tolerance:
            relation = 'at'
        elif current_price > angle_value:
            relation = 'above'
        else:
            relation = 'below'
        
        angles.append(GannAngle(
            name=name,
            slope=slope,
            current_value=angle_value,
            relation=relation
        ))
    
    # Calculate confluence score
    score = _calculate_angle_confluence(angles, current_price, trend, atr_value)
    
    # Overall relation
    at_angles = sum(1 for a in angles if a.relation == 'at')
    above = sum(1 for a in angles if a.relation == 'above')
    below = sum(1 for a in angles if a.relation == 'below')
    
    if at_angles > 0:
        overall = 'at_angle'
    elif above > below:
        overall = 'above_angles'
    else:
        overall = 'below_angles'
    
    return GannAnalysis(
        pivot_price=pivot_price,
        pivot_time=pivot_time,
        angles=angles,
        angle_confluence_score=score,
        price_relation=overall
    )


def _calculate_angle_confluence(
    angles: List[GannAngle],
    price: float,
    trend: str,
    atr: float
) -> float:
    """
    Calculate confluence score based on angle alignment.
    
    Score 0-2 based on:
    - Price at key angle (1x1) = +1
    - Price respecting trend angle = +0.5
    - Multiple angles clustering = +0.5
    """
    score = 0.0
    
    # Check 1x1 angle (most important)
    for angle in angles:
        if angle.name == "1x1" and angle.relation == 'at':
            score += 1.0
            break
    
    # Check trend alignment
    if trend == 'up':
        # Price should be above slower angles in uptrend
        for angle in angles:
            if angle.name in ["1x2", "1x4"] and angle.relation == 'above':
                score += 0.25
    else:
        # Price should be below slower angles in downtrend
        for angle in angles:
            if angle.name in ["1x2", "1x4"] and angle.relation == 'below':
                score += 0.25
    
    # Check for clustering (multiple angles near price)
    at_count = sum(1 for a in angles if a.relation == 'at')
    if at_count >= 2:
        score += 0.5
    
    return min(score, 2.0)


# =============================================================================
# Square of 9
# =============================================================================

def square9_levels(
    price: float,
    degrees_list: List[int] = None,
    steps: int = 2
) -> List[Square9Level]:
    """
    Calculate Square of 9 levels around a price.
    
    The Square of 9 uses the formula:
    - sqrt(price) ± (degrees/360) * k
    - Then square the result
    
    Common degrees: 45, 90, 135, 180, 225, 270, 315, 360
    
    Args:
        price: Base price to calculate from
        degrees_list: List of degree values (default: standard set)
        steps: Number of steps above and below
        
    Returns:
        List of Square9Level sorted by distance from price
    """
    if degrees_list is None:
        degrees_list = [45, 90, 135, 180, 225, 270, 315, 360]
    
    levels = []
    sqrt_price = math.sqrt(price)
    
    for step in range(-steps, steps + 1):
        if step == 0:
            continue
            
        for degrees in degrees_list:
            # Calculate k factor from degrees
            k = degrees / 360.0
            
            # Calculate level
            if step > 0:
                # Levels above
                sqrt_level = sqrt_price + (k * step)
                direction = 'above'
            else:
                # Levels below
                sqrt_level = sqrt_price + (k * step)  # step is negative
                direction = 'below'
            
            if sqrt_level <= 0:
                continue
            
            level_price = sqrt_level ** 2
            distance_pct = abs(level_price - price) / price * 100
            
            levels.append(Square9Level(
                level=level_price,
                degrees=degrees,
                distance_pct=distance_pct,
                direction=direction
            ))
    
    # Sort by distance from price
    levels.sort(key=lambda l: l.distance_pct)
    
    return levels


def analyze_square9(
    price: float,
    proximity_threshold_pct: float = 0.15
) -> Square9Analysis:
    """
    Perform complete Square of 9 analysis.
    
    Args:
        price: Current price
        proximity_threshold_pct: Threshold for considering price "at" a level
        
    Returns:
        Square9Analysis with levels and score
    """
    levels = square9_levels(price)
    
    if not levels:
        return Square9Analysis(
            base_price=price,
            levels=[],
            nearest_level=None,
            square9_score=0
        )
    
    nearest = levels[0] if levels else None
    
    # Calculate score based on proximity to nearest level
    score = 0.0
    if nearest:
        if nearest.distance_pct <= proximity_threshold_pct:
            score = 2.0
        elif nearest.distance_pct <= proximity_threshold_pct * 2:
            score = 1.5
        elif nearest.distance_pct <= proximity_threshold_pct * 3:
            score = 1.0
        elif nearest.distance_pct <= proximity_threshold_pct * 5:
            score = 0.5
    
    return Square9Analysis(
        base_price=price,
        levels=levels[:10],  # Return top 10 closest
        nearest_level=nearest,
        square9_score=score
    )


def format_gann_analysis(analysis: GannAnalysis) -> str:
    """Format Gann analysis for display."""
    lines = [f"Pivot: {analysis.pivot_price:.2f}"]
    
    for angle in analysis.angles[:3]:  # Show top 3 angles
        lines.append(f"  {angle.name}: {angle.current_value:.2f} ({angle.relation})")
    
    lines.append(f"Score: {analysis.angle_confluence_score:.1f}/2")
    
    return "\n".join(lines)


def format_square9_analysis(analysis: Square9Analysis) -> str:
    """Format Square of 9 analysis for display."""
    lines = [f"Base: {analysis.base_price:.2f}"]
    
    if analysis.nearest_level:
        nl = analysis.nearest_level
        lines.append(f"Nearest: {nl.level:.2f} ({nl.degrees}° {nl.direction}, {nl.distance_pct:.2f}%)")
    
    lines.append(f"Score: {analysis.square9_score:.1f}/2")
    
    return "\n".join(lines)
