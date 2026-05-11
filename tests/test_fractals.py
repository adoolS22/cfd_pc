"""
Tests for fractal detection.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from bot.patterns import detect_fractals, get_last_fractal, Fractal


def create_ohlcv_df(data: list) -> pd.DataFrame:
    """Create OHLCV DataFrame from list of (open, high, low, close) tuples."""
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    records = []
    
    for i, (o, h, l, c) in enumerate(data):
        records.append({
            'timestamp': base_time + timedelta(hours=i),
            'open': o,
            'high': h,
            'low': l,
            'close': c,
            'volume': 1000 + i * 100
        })
    
    df = pd.DataFrame(records)
    df.set_index('timestamp', inplace=True)
    return df


class TestFractalDetection:
    """Tests for fractal detection."""
    
    def test_fractal_high_basic(self):
        """Test basic fractal high detection."""
        # Create data with clear fractal high at index 2
        # Pattern: lower, lower, HIGHEST, lower, lower
        data = [
            (100, 101, 99, 100),   # 0
            (100, 102, 99, 101),   # 1
            (101, 110, 100, 109),  # 2 - Fractal High (110 > 102 and 110 > 103)
            (109, 103, 102, 102),  # 3
            (102, 101, 100, 100),  # 4
        ]
        df = create_ohlcv_df(data)
        
        fractals = detect_fractals(df, window=2)
        
        high_fractals = [f for f in fractals if f.type == 'high']
        assert len(high_fractals) == 1
        assert high_fractals[0].price == 110
    
    def test_fractal_low_basic(self):
        """Test basic fractal low detection."""
        # Create data with clear fractal low at index 2
        # Pattern: higher, higher, LOWEST, higher, higher
        data = [
            (100, 105, 98, 100),   # 0
            (100, 104, 97, 99),    # 1
            (99, 102, 90, 91),     # 2 - Fractal Low (90 < 97 and 90 < 94)
            (91, 101, 94, 100),    # 3
            (100, 103, 96, 102),   # 4
        ]
        df = create_ohlcv_df(data)
        
        fractals = detect_fractals(df, window=2)
        
        low_fractals = [f for f in fractals if f.type == 'low']
        assert len(low_fractals) == 1
        assert low_fractals[0].price == 90
    
    def test_no_fractal_equal_values(self):
        """Test that equal values don't form fractals."""
        # All highs are equal
        data = [
            (100, 105, 99, 100),
            (100, 105, 99, 100),
            (100, 105, 99, 100),
            (100, 105, 99, 100),
            (100, 105, 99, 100),
        ]
        df = create_ohlcv_df(data)
        
        fractals = detect_fractals(df, window=2)
        
        assert len(fractals) == 0
    
    def test_multiple_fractals(self):
        """Test detection of multiple fractals."""
        data = [
            (100, 101, 99, 100),
            (100, 102, 99, 101),
            (101, 110, 100, 109),  # Fractal High
            (109, 103, 95, 96),
            (96, 100, 85, 86),     # Fractal Low
            (86, 95, 84, 94),
            (94, 98, 93, 97),
        ]
        df = create_ohlcv_df(data)
        
        fractals = detect_fractals(df, window=2)
        
        high_fractals = [f for f in fractals if f.type == 'high']
        low_fractals = [f for f in fractals if f.type == 'low']
        
        assert len(high_fractals) >= 1
        assert len(low_fractals) >= 1
    
    def test_get_last_fractal(self):
        """Test getting the last fractal."""
        data = [
            (100, 110, 99, 105),   # 0
            (105, 108, 100, 107),  # 1
            (107, 120, 105, 118),  # 2 - Fractal High
            (118, 115, 110, 112),  # 3
            (112, 113, 90, 91),    # Would be fractal low but at edge
            (91, 100, 88, 98),     # 5
            (98, 102, 95, 100),    # 6
        ]
        df = create_ohlcv_df(data)
        
        last_high = get_last_fractal(df, 'high', window=2)
        
        assert last_high is not None
        assert last_high.type == 'high'
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        data = [
            (100, 105, 99, 100),
            (100, 106, 99, 101),
        ]
        df = create_ohlcv_df(data)
        
        fractals = detect_fractals(df, window=2)
        
        assert len(fractals) == 0
    
    def test_window_size_3(self):
        """Test fractal detection with window size 3."""
        # Need at least 7 bars for window=3
        data = [
            (100, 101, 99, 100),   # 0
            (100, 102, 99, 101),   # 1
            (101, 103, 100, 102),  # 2
            (102, 120, 101, 118),  # 3 - Potential Fractal High
            (118, 110, 108, 109),  # 4
            (109, 105, 104, 104),  # 5
            (104, 103, 100, 101),  # 6
        ]
        df = create_ohlcv_df(data)
        
        fractals = detect_fractals(df, window=3)
        
        high_fractals = [f for f in fractals if f.type == 'high']
        assert len(high_fractals) == 1
        assert high_fractals[0].price == 120
