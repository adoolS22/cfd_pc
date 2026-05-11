"""
Tests for Gann analysis.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from bot.gann import (
    calculate_gann_angles,
    square9_levels,
    analyze_square9,
    GannAnalysis,
    Square9Analysis,
    Square9Level
)


def create_ohlcv_df(length: int = 100, trend: str = 'up') -> pd.DataFrame:
    """Create test OHLCV DataFrame with trend."""
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    records = []
    
    for i in range(length):
        if trend == 'up':
            base_price = 100 + i * 0.5
        elif trend == 'down':
            base_price = 150 - i * 0.5
        else:
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


class TestGannAngles:
    """Tests for Gann angle calculations."""
    
    def test_gann_angles_uptrend(self):
        """Test Gann angles in uptrend."""
        df = create_ohlcv_df(100, trend='up')
        atr = 5.0
        
        result = calculate_gann_angles(df, 'up', atr, lookback=50)
        
        assert isinstance(result, GannAnalysis)
        assert len(result.angles) > 0
        assert result.pivot_price > 0
    
    def test_gann_angles_downtrend(self):
        """Test Gann angles in downtrend."""
        df = create_ohlcv_df(100, trend='down')
        atr = 5.0
        
        result = calculate_gann_angles(df, 'down', atr, lookback=50)
        
        assert isinstance(result, GannAnalysis)
        assert len(result.angles) > 0
    
    def test_gann_angle_names(self):
        """Test that expected angle names are present."""
        df = create_ohlcv_df(100, trend='up')
        atr = 5.0
        
        result = calculate_gann_angles(df, 'up', atr)
        
        angle_names = [a.name for a in result.angles]
        assert '1x1' in angle_names
        assert '2x1' in angle_names
        assert '1x2' in angle_names
    
    def test_gann_confluence_score_range(self):
        """Test confluence score is in valid range."""
        df = create_ohlcv_df(100, trend='up')
        atr = 5.0
        
        result = calculate_gann_angles(df, 'up', atr)
        
        assert 0 <= result.angle_confluence_score <= 2
    
    def test_gann_price_relation(self):
        """Test price relation is valid."""
        df = create_ohlcv_df(100, trend='up')
        atr = 5.0
        
        result = calculate_gann_angles(df, 'up', atr)
        
        valid_relations = ['above_angles', 'below_angles', 'at_angle', 'unknown']
        assert result.price_relation in valid_relations
    
    def test_gann_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        result = calculate_gann_angles(df, 'up', 5.0)
        
        assert len(result.angles) == 0
        assert result.angle_confluence_score == 0


class TestSquare9:
    """Tests for Square of 9 calculations."""
    
    def test_square9_levels_generation(self):
        """Test Square of 9 levels are generated."""
        price = 100
        
        levels = square9_levels(price)
        
        assert len(levels) > 0
        assert all(isinstance(l, Square9Level) for l in levels)
    
    def test_square9_levels_sorted_by_distance(self):
        """Test levels are sorted by distance from price."""
        price = 100
        
        levels = square9_levels(price)
        
        for i in range(len(levels) - 1):
            assert levels[i].distance_pct <= levels[i + 1].distance_pct
    
    def test_square9_levels_both_directions(self):
        """Test levels are generated above and below price."""
        price = 100
        
        levels = square9_levels(price)
        
        above = [l for l in levels if l.direction == 'above']
        below = [l for l in levels if l.direction == 'below']
        
        assert len(above) > 0
        assert len(below) > 0
    
    def test_square9_levels_monotonicity(self):
        """Test that levels increase in steps."""
        price = 100
        
        levels = square9_levels(price, degrees_list=[360], steps=5)
        
        above_levels = sorted([l for l in levels if l.direction == 'above'], key=lambda x: x.level)
        
        # Each level should be higher than previous
        for i in range(len(above_levels) - 1):
            assert above_levels[i].level < above_levels[i + 1].level
    
    def test_square9_custom_degrees(self):
        """Test custom degree values."""
        price = 100
        degrees = [90, 180, 270, 360]
        
        levels = square9_levels(price, degrees_list=degrees)
        
        # Should have multiple levels per degree
        assert len(levels) > len(degrees)
    
    def test_analyze_square9(self):
        """Test complete Square of 9 analysis."""
        price = 100
        
        result = analyze_square9(price, proximity_threshold_pct=0.5)
        
        assert isinstance(result, Square9Analysis)
        assert result.base_price == price
        assert len(result.levels) > 0
    
    def test_analyze_square9_score_range(self):
        """Test Square of 9 score is in valid range."""
        price = 100
        
        result = analyze_square9(price)
        
        assert 0 <= result.square9_score <= 2
    
    def test_analyze_square9_nearest_level(self):
        """Test nearest level is identified."""
        price = 100
        
        result = analyze_square9(price)
        
        assert result.nearest_level is not None
        # Nearest level should be close to price
        assert result.nearest_level.distance_pct < 10  # Within 10%
    
    def test_square9_different_prices(self):
        """Test Square of 9 works with different price scales."""
        prices = [1, 10, 100, 1000, 50000]
        
        for price in prices:
            result = analyze_square9(price)
            
            assert result.base_price == price
            assert len(result.levels) > 0
