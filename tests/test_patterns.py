"""
Tests for candlestick pattern detection.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from bot.patterns import (
    detect_bullish_engulfing,
    detect_bearish_engulfing,
    detect_hammer,
    detect_shooting_star,
    detect_pin_bar,
    detect_all_patterns,
    detect_harami,
    detect_piercing_line,
    detect_dark_cloud_cover,
    get_pattern_for_direction,
    get_advisory_pattern_for_direction,
)


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
            'volume': 1000
        })
    
    df = pd.DataFrame(records)
    df.set_index('timestamp', inplace=True)
    return df


class TestBullishEngulfing:
    """Tests for bullish engulfing pattern."""
    
    def test_valid_bullish_engulfing(self):
        """Test detection of valid bullish engulfing."""
        # Previous: bearish (close < open), Current: bullish engulfing
        data = [
            (102, 103, 98, 99),    # Bearish candle
            (98, 105, 97, 104),    # Bullish engulfing (body 98-104 engulfs 99-102)
        ]
        df = create_ohlcv_df(data)
        
        pattern = detect_bullish_engulfing(df)
        
        assert pattern is not None
        assert pattern.name == 'bullish_engulfing'
        assert pattern.direction == 'bullish'
    
    def test_no_engulfing_same_direction(self):
        """Test no pattern when both candles same direction."""
        # Both bullish
        data = [
            (100, 103, 99, 102),
            (102, 108, 101, 106),
        ]
        df = create_ohlcv_df(data)
        
        pattern = detect_bullish_engulfing(df)
        
        assert pattern is None
    
    def test_no_engulfing_smaller_body(self):
        """Test no pattern when current body doesn't engulf."""
        data = [
            (102, 103, 98, 99),    # Bearish
            (99, 102, 98, 101),    # Body doesn't fully engulf
        ]
        df = create_ohlcv_df(data)
        
        pattern = detect_bullish_engulfing(df)
        
        assert pattern is None


class TestBearishEngulfing:
    """Tests for bearish engulfing pattern."""
    
    def test_valid_bearish_engulfing(self):
        """Test detection of valid bearish engulfing."""
        # Previous: bullish, Current: bearish engulfing
        data = [
            (98, 103, 97, 102),    # Bullish candle
            (103, 104, 95, 96),    # Bearish engulfing
        ]
        df = create_ohlcv_df(data)
        
        pattern = detect_bearish_engulfing(df)
        
        assert pattern is not None
        assert pattern.name == 'bearish_engulfing'
        assert pattern.direction == 'bearish'


class TestHammer:
    """Tests for hammer pattern."""
    
    def test_valid_hammer(self):
        """Test detection of valid hammer."""
        # Small body, long lower wick, small upper wick
        # After downtrend
        data = [
            (105, 106, 103, 103),  # Downtrend
            (103, 104, 101, 101),  # Downtrend  
            (101, 102, 99, 99),    # Downtrend
            (99, 100, 90, 99.5),   # Hammer: body ~0.5, lower wick ~9, upper wick ~0.5
        ]
        df = create_ohlcv_df(data)
        
        pattern = detect_hammer(df)
        
        assert pattern is not None
        assert pattern.name == 'hammer'
        assert pattern.direction == 'bullish'
    
    def test_no_hammer_large_body(self):
        """Test no hammer when body is too large."""
        data = [
            (105, 106, 103, 103),
            (103, 104, 101, 101),
            (101, 102, 99, 99),
            (95, 102, 90, 100),   # Body too large relative to range
        ]
        df = create_ohlcv_df(data)
        
        pattern = detect_hammer(df)
        
        assert pattern is None


class TestShootingStar:
    """Tests for shooting star pattern."""
    
    def test_valid_shooting_star(self):
        """Test detection of valid shooting star."""
        # Small body, long upper wick, small lower wick
        # After uptrend
        data = [
            (95, 97, 94, 97),      # Uptrend
            (97, 99, 96, 99),      # Uptrend
            (99, 101, 98, 101),    # Uptrend
            (101, 110, 100, 101.5), # Shooting star
        ]
        df = create_ohlcv_df(data)
        
        pattern = detect_shooting_star(df)
        
        assert pattern is not None
        assert pattern.name == 'shooting_star'
        assert pattern.direction == 'bearish'


class TestPinBar:
    """Tests for pin bar pattern."""
    
    def test_bullish_pin_bar(self):
        """Test detection of bullish pin bar."""
        # Long lower tail
        data = [
            (100, 101, 99, 100),
            (100, 101, 85, 100),   # Bullish pin bar
        ]
        df = create_ohlcv_df(data)
        
        pattern = detect_pin_bar(df)
        
        assert pattern is not None
        assert pattern.name == 'pin_bar'
        assert pattern.direction == 'bullish'
    
    def test_bearish_pin_bar(self):
        """Test detection of bearish pin bar."""
        # Long upper tail
        data = [
            (100, 101, 99, 100),
            (100, 115, 99, 100),   # Bearish pin bar
        ]
        df = create_ohlcv_df(data)
        
        pattern = detect_pin_bar(df)
        
        assert pattern is not None
        assert pattern.name == 'pin_bar'
        assert pattern.direction == 'bearish'


class TestAllPatterns:
    """Tests for combined pattern detection."""
    
    def test_detect_all_patterns(self):
        """Test that detect_all_patterns finds multiple patterns."""
        # Create data that could match multiple patterns
        data = [
            (102, 103, 98, 99),
            (98, 105, 97, 104),    # Bullish engulfing
        ]
        df = create_ohlcv_df(data)
        
        patterns = detect_all_patterns(df)
        
        assert len(patterns) >= 1
        pattern_names = [p.name for p in patterns]
        assert 'bullish_engulfing' in pattern_names
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        patterns = detect_all_patterns(df)
        
        assert len(patterns) == 0


class TestAdvisoryOnlyPatterns:
    """Tests for advisory-only candlestick patterns."""

    def test_bullish_harami(self):
        data = [
            (105, 106, 98, 99),     # bearish strong body
            (100, 102, 99.5, 101.5) # bullish small body inside previous body
        ]
        df = create_ohlcv_df(data)

        pattern = detect_harami(df)
        assert pattern is not None
        assert pattern.name == 'bullish_harami'
        assert pattern.direction == 'bullish'

    def test_piercing_line(self):
        data = [
            (100, 101, 92, 93),      # bearish strong
            (92.8, 98, 92, 96.8),    # bullish close above midpoint, below prev open
        ]
        df = create_ohlcv_df(data)

        pattern = detect_piercing_line(df)
        assert pattern is not None
        assert pattern.name == 'piercing_line'
        assert pattern.direction == 'bullish'

    def test_dark_cloud_cover(self):
        data = [
            (93, 101, 92, 100),      # bullish strong
            (99.2, 100, 95, 96),     # bearish close below midpoint, above prev open
        ]
        df = create_ohlcv_df(data)

        pattern = detect_dark_cloud_cover(df)
        assert pattern is not None
        assert pattern.name == 'dark_cloud_cover'
        assert pattern.direction == 'bearish'

    def test_advisory_patterns_do_not_change_core_pattern_picker(self):
        """
        Core picker remains unchanged (execution path),
        advisory picker includes the extra classical patterns.
        """
        data = [
            (100, 101, 92, 93),      # bearish strong
            (92.8, 98, 92, 96.8),    # piercing line only
        ]
        df = create_ohlcv_df(data)

        core_pattern = get_pattern_for_direction(df, 'bullish')
        advisory_pattern = get_advisory_pattern_for_direction(df, 'bullish')

        assert core_pattern is None
        assert advisory_pattern is not None
        assert advisory_pattern.name == 'piercing_line'
