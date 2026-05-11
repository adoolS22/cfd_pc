"""
Time Cycles Analysis
====================
52-cycle and lunar phase timing calculations.
"""

import math
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from loguru import logger

# Try to import ephem for lunar calculations
try:
    import ephem
    EPHEM_AVAILABLE = True
except ImportError:
    EPHEM_AVAILABLE = False
    logger.warning("ephem library not available - lunar analysis disabled")


@dataclass
class CycleAnalysis:
    """Result of 52-cycle analysis."""
    cycle_position: int  # 0-51
    in_cycle_window: bool
    bars_to_window: int
    cycle_score: float  # 0-1


@dataclass
class LunarAnalysis:
    """Result of lunar phase analysis."""
    phase: str  # 'new', 'full', 'waxing', 'waning'
    phase_pct: float  # 0-100
    hours_to_next_new: float
    hours_to_next_full: float
    in_event_window: bool
    lunar_score: float  # 0-1


# =============================================================================
# 52-Cycle Analysis
# =============================================================================

def analyze_52_cycle(
    df: pd.DataFrame,
    anchor_type: str = 'pivot',
    buffer_bars: int = 2
) -> CycleAnalysis:
    """
    Analyze 52-bar cycle position.
    
    The 52-cycle theory suggests market turns occur near:
    - Bar 0-2 (cycle start)
    - Bar 50-51 (cycle end)
    
    Args:
        df: DataFrame with OHLCV data
        anchor_type: 'pivot' (last major low/high) or 'year_start'
        buffer_bars: Buffer around cycle windows
        
    Returns:
        CycleAnalysis with cycle position and score
    """
    if df.empty:
        return CycleAnalysis(
            cycle_position=0,
            in_cycle_window=False,
            bars_to_window=26,
            cycle_score=0
        )
    
    # Find anchor point
    if anchor_type == 'pivot':
        # Find the most significant pivot in the data
        lookback = min(len(df), 100)
        recent = df.tail(lookback)
        
        # Find lowest low and highest high
        low_idx = recent['low'].idxmin()
        high_idx = recent['high'].idxmax()
        
        # Use the more recent one as anchor
        if low_idx > high_idx:
            anchor_idx = df.index.get_loc(low_idx)
        else:
            anchor_idx = df.index.get_loc(high_idx)
    else:
        # Use start of year or first bar
        anchor_idx = 0
    
    # Calculate position in cycle
    bars_since_anchor = len(df) - anchor_idx - 1
    cycle_position = bars_since_anchor % 52
    
    # Determine if in cycle window
    start_window = list(range(0, buffer_bars + 1))
    end_window = list(range(52 - buffer_bars, 52))
    cycle_windows = start_window + end_window
    
    in_window = cycle_position in cycle_windows
    
    # Calculate bars to next window
    if in_window:
        bars_to_window = 0
    elif cycle_position < 52 - buffer_bars:
        bars_to_window = (52 - buffer_bars) - cycle_position
    else:
        bars_to_window = (52 + buffer_bars + 1) - cycle_position
    
    # Calculate score
    if in_window:
        # Higher score when exactly at window edges
        if cycle_position in [0, 51]:
            score = 1.0
        else:
            score = 0.7
    else:
        score = 0.0
    
    return CycleAnalysis(
        cycle_position=cycle_position,
        in_cycle_window=in_window,
        bars_to_window=bars_to_window,
        cycle_score=score
    )


# =============================================================================
# Lunar Timing Analysis
# =============================================================================

def get_lunar_phase(dt: Optional[datetime] = None) -> Tuple[str, float]:
    """
    Calculate current lunar phase.
    
    Args:
        dt: Datetime to check (default: now)
        
    Returns:
        Tuple of (phase_name, phase_percentage)
    """
    if not EPHEM_AVAILABLE:
        return ('unknown', 0.0)
    
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    # Convert to ephem date
    date = ephem.Date(dt)
    
    # Calculate phase
    moon = ephem.Moon(date)
    
    # Phase is 0-1 where:
    # 0 = new moon, 0.25 = first quarter, 0.5 = full moon, 0.75 = last quarter
    phase_pct = moon.phase  # 0-100
    
    # Determine phase name
    if phase_pct < 5 or phase_pct > 95:
        phase = 'new'
    elif 45 < phase_pct < 55:
        phase = 'full'
    elif phase_pct < 50:
        phase = 'waxing'
    else:
        phase = 'waning'
    
    return (phase, phase_pct)


def get_next_lunar_events(dt: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """
    Get next new moon and full moon dates.
    
    Args:
        dt: Starting datetime (default: now)
        
    Returns:
        Tuple of (next_new_moon, next_full_moon) as datetime objects
    """
    if not EPHEM_AVAILABLE:
        # Return far future dates if ephem not available
        far_future = datetime.now(timezone.utc) + timedelta(days=365)
        return (far_future, far_future)
    
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    date = ephem.Date(dt)
    
    # Get next new moon
    next_new = ephem.next_new_moon(date)
    next_new_dt = ephem.Date(next_new).datetime().replace(tzinfo=timezone.utc)
    
    # Get next full moon
    next_full = ephem.next_full_moon(date)
    next_full_dt = ephem.Date(next_full).datetime().replace(tzinfo=timezone.utc)
    
    return (next_new_dt, next_full_dt)


def analyze_lunar(
    window_hours: int = 24,
    dt: Optional[datetime] = None
) -> LunarAnalysis:
    """
    Perform complete lunar analysis.
    
    Markets often show increased volatility around new and full moons.
    
    Args:
        window_hours: Hours before/after event to consider "in window"
        dt: Datetime to analyze (default: now)
        
    Returns:
        LunarAnalysis with phase info and score
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    if not EPHEM_AVAILABLE:
        return LunarAnalysis(
            phase='unknown',
            phase_pct=0,
            hours_to_next_new=float('inf'),
            hours_to_next_full=float('inf'),
            in_event_window=False,
            lunar_score=0
        )
    
    phase, phase_pct = get_lunar_phase(dt)
    next_new, next_full = get_next_lunar_events(dt)
    
    hours_to_new = (next_new - dt).total_seconds() / 3600
    hours_to_full = (next_full - dt).total_seconds() / 3600
    
    # Also check previous events (we might be in window after event)
    prev_new = ephem.previous_new_moon(ephem.Date(dt))
    prev_full = ephem.previous_full_moon(ephem.Date(dt))
    prev_new_dt = ephem.Date(prev_new).datetime().replace(tzinfo=timezone.utc)
    prev_full_dt = ephem.Date(prev_full).datetime().replace(tzinfo=timezone.utc)
    
    hours_since_new = (dt - prev_new_dt).total_seconds() / 3600
    hours_since_full = (dt - prev_full_dt).total_seconds() / 3600
    
    # Check if in event window
    in_window = (
        hours_to_new <= window_hours or 
        hours_to_full <= window_hours or
        hours_since_new <= window_hours or
        hours_since_full <= window_hours
    )
    
    # Calculate score
    score = 0.0
    if in_window:
        # Higher score for new moon events
        if hours_to_new <= window_hours or hours_since_new <= window_hours:
            score = 1.0
        elif hours_to_full <= window_hours or hours_since_full <= window_hours:
            score = 0.7
    
    return LunarAnalysis(
        phase=phase,
        phase_pct=phase_pct,
        hours_to_next_new=hours_to_new,
        hours_to_next_full=hours_to_full,
        in_event_window=in_window,
        lunar_score=score
    )


def format_cycle_analysis(analysis: CycleAnalysis) -> str:
    """Format cycle analysis for display."""
    window_status = "IN WINDOW" if analysis.in_cycle_window else f"{analysis.bars_to_window} bars to window"
    return f"Position: {analysis.cycle_position}/52 | {window_status} | Score: {analysis.cycle_score:.1f}"


def format_lunar_analysis(analysis: LunarAnalysis) -> str:
    """Format lunar analysis for display."""
    window_status = "IN WINDOW" if analysis.in_event_window else "Outside window"
    return (f"Phase: {analysis.phase} ({analysis.phase_pct:.0f}%) | "
            f"Next New: {analysis.hours_to_next_new:.0f}h | "
            f"Next Full: {analysis.hours_to_next_full:.0f}h | "
            f"{window_status} | Score: {analysis.lunar_score:.1f}")
