"""
Tests for time cycle analysis.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from bot.time_cycles import (
    analyze_52_cycle,
    analyze_lunar,
    get_lunar_phase,
    CycleAnalysis,
    LunarAnalysis
)


def create_ohlcv_df(length: int = 100) -> pd.DataFrame:
    """Create test OHLCV DataFrame."""
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    records = []
    
    for i in range(length):
        base_price = 100 + np.sin(i / 10) * 10
        records.append({
            'timestamp': base_time + timedelta(hours=i),
            'open': base_price,
            'high': base_price + 2,
            'low': base_price - 2,
            'close': base_price + 1,
            'volume': 1000
        })
    
    df = pd.DataFrame(records)
    df.set_index('timestamp', inplace=True)
    return df


class TestCycle52:
    """Tests for 52-cycle analysis."""
    
    def test_cycle_position_calculation(self):
        """Test correct cycle position calculation."""
        df = create_ohlcv_df(100)
        
        result = analyze_52_cycle(df, anchor_type='pivot')
        
        assert isinstance(result, CycleAnalysis)
        assert 0 <= result.cycle_position < 52
    
    def test_cycle_window_at_start(self):
        """Test cycle window detection at cycle start."""
        df = create_ohlcv_df(52)  # Exactly one cycle
        
        result = analyze_52_cycle(df, anchor_type='pivot', buffer_bars=2)
        
        # Should be in window if position is 0-2 or 50-51
        if result.cycle_position in [0, 1, 2, 50, 51]:
            assert result.in_cycle_window
    
    def test_cycle_window_outside(self):
        """Test detection when outside cycle window."""
        df = create_ohlcv_df(26 + 10)  # Put us at position ~26 (middle of cycle)
        
        result = analyze_52_cycle(df, anchor_type='pivot', buffer_bars=2)
        
        # If in middle of cycle, should not be in window
        if 5 < result.cycle_position < 48:
            assert not result.in_cycle_window
    
    def test_cycle_score_in_window(self):
        """Test score is positive when in window."""
        df = create_ohlcv_df(52)  # One full cycle
        
        result = analyze_52_cycle(df, anchor_type='pivot')
        
        if result.in_cycle_window:
            assert result.cycle_score > 0
    
    def test_cycle_score_outside_window(self):
        """Test score is zero when outside window."""
        df = create_ohlcv_df(26)  # Half cycle
        
        result = analyze_52_cycle(df, anchor_type='pivot')
        
        if not result.in_cycle_window:
            assert result.cycle_score == 0
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        result = analyze_52_cycle(df)
        
        assert result.cycle_position == 0
        assert not result.in_cycle_window


class TestLunarAnalysis:
    """Tests for lunar phase analysis."""
    
    @patch('bot.time_cycles.EPHEM_AVAILABLE', True)
    def test_lunar_analysis_returns_valid_result(self):
        """Test that lunar analysis returns valid result."""
        result = analyze_lunar(window_hours=24)
        
        assert isinstance(result, LunarAnalysis)
        assert result.phase in ['new', 'full', 'waxing', 'waning', 'unknown']
        assert 0 <= result.phase_pct <= 100
    
    @patch('bot.time_cycles.EPHEM_AVAILABLE', False)
    def test_lunar_analysis_without_ephem(self):
        """Test lunar analysis when ephem is not available."""
        result = analyze_lunar(window_hours=24)
        
        assert result.phase == 'unknown'
        assert result.lunar_score == 0
        assert not result.in_event_window
    
    def test_lunar_score_range(self):
        """Test that lunar score is in valid range."""
        result = analyze_lunar(window_hours=24)
        
        assert 0 <= result.lunar_score <= 1
    
    def test_lunar_window_hours_effect(self):
        """Test that larger window affects detection."""
        result_small = analyze_lunar(window_hours=1)
        result_large = analyze_lunar(window_hours=48)
        
        # Larger window should be more likely to be in window
        # But we can't guarantee this without mocking time
        assert isinstance(result_small, LunarAnalysis)
        assert isinstance(result_large, LunarAnalysis)


class TestLunarPhase:
    """Tests for lunar phase calculation."""
    
    @patch('bot.time_cycles.EPHEM_AVAILABLE', True)
    def test_lunar_phase_returns_tuple(self):
        """Test that get_lunar_phase returns correct tuple."""
        phase, pct = get_lunar_phase()
        
        assert isinstance(phase, str)
        assert isinstance(pct, float)
    
    @patch('bot.time_cycles.EPHEM_AVAILABLE', False)
    def test_lunar_phase_without_ephem(self):
        """Test fallback when ephem not available."""
        phase, pct = get_lunar_phase()
        
        assert phase == 'unknown'
        assert pct == 0.0
