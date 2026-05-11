
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.fibonacci import calculate_fib_levels, FibAnalysis, FibLevel

def test_calculate_fib_levels_uptrend():
    # Create simple uptrend data
    # Low at 100, High at 200
    # Create valid dataframe structure
    idx = pd.date_range(start='2024-01-01', periods=10, tz='UTC')
    data = {
        'timestamp': idx,
        'open': [100.0]*10,
        'high': [100.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 200.0, 195.0],
        'low':  [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0],
        'close': [150.0]*10,
        'volume': [1000.0]*10
    }
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # Calculate for uptrend
    analysis = calculate_fib_levels(df, trend='up', lookback=10)
    
    assert analysis is not None
    assert analysis.swing_high == 200.0
    assert analysis.swing_low == 100.0
    
    # Check 0.5 Retracement level
    # Range = 200 - 100 = 100
    # Uptrend Retracement = High - (Range * 0.5) = 200 - 50 = 150
    level_05 = next((l for l in analysis.levels if l.ratio == 0.5 and l.type == 'retracement'), None)
    assert level_05 is not None
    assert level_05.price == 150.0
    
    # Check 1.618 Extension level
    # Extension = Low + (Range * 1.618) = 100 + 161.8 = 261.8
    level_ext = next((l for l in analysis.levels if l.ratio == 1.618 and l.type == 'extension'), None)
    assert level_ext is not None
    assert level_ext.price == pytest.approx(261.8)

def test_calculate_fib_levels_downtrend():
    # Create simple downtrend data
    # High at 200, Low at 100
    idx = pd.date_range(start='2024-01-01', periods=10, tz='UTC')
    data = {
        'timestamp': idx,
        'open': [150.0]*10,
        'high': [200.0, 190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0],
        'low':  [190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0, 100.0],
        'close': [105.0]*10,
        'volume': [1000.0]*10
    }
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # Calculate for downtrend
    analysis = calculate_fib_levels(df, trend='down', lookback=10)
    
    assert analysis is not None
    assert analysis.swing_high == 200.0
    assert analysis.swing_low == 100.0
    
    # Check 0.618 Retracement level (Bounce up)
    # Range = 100
    # Downtrend Retracement = Low + (Range * 0.618) = 100 + 61.8 = 161.8
    level_618 = next((l for l in analysis.levels if l.ratio == 0.618 and l.type == 'retracement'), None)
    assert level_618 is not None
    assert level_618.price == pytest.approx(161.8)

def test_calculate_fib_levels_empty():
    df = pd.DataFrame()
    analysis = calculate_fib_levels(df, trend='up')
    assert analysis is None
