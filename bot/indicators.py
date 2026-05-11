"""
Technical Indicators
====================
Pandas-based technical indicator calculations.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from loguru import logger


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.
    
    Args:
        series: Price series (typically close prices)
        period: Lookback period
        
    Returns:
        SMA series
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.
    
    Args:
        series: Price series
        period: Lookback period
        
    Returns:
        EMA series
    """
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    
    Args:
        series: Price series (typically close prices)
        period: Lookback period (default: 14)
        
    Returns:
        RSI series (0-100)
    """
    delta = series.diff()
    
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values.fillna(50)  # Fill NaN with neutral value


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default: 14)
        
    Returns:
        ATR series
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range components
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    # True Range is max of all three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is smoothed TR
    atr_values = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return atr_values


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Volume Simple Moving Average.
    
    Args:
        df: DataFrame with 'volume' column
        period: Lookback period (default: 20)
        
    Returns:
        Volume SMA series
    """
    return sma(df['volume'], period)


def is_volume_spike(df: pd.DataFrame, period: int = 20, multiplier: float = 1.8) -> pd.Series:
    """
    Detect volume spikes.
    
    Args:
        df: DataFrame with 'volume' column
        period: Lookback period for average
        multiplier: Spike threshold multiplier
        
    Returns:
        Boolean series indicating volume spikes
    """
    vol_sma = volume_sma(df, period)
    return df['volume'] > (vol_sma * multiplier)


def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume (OBV).
    Cumulative volume that adds volume on up days and subtracts on down days.
    Used to detect divergence between price and volume (smart money flow).

    Returns:
        OBV series
    """
    close = df['close']
    volume = df['volume']
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price (VWAP).
    Average price weighted by volume — used as institutional reference.
    Price above VWAP = bullish bias; below = bearish bias.

    Returns:
        VWAP series
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_tp_vol = (typical_price * df['volume']).cumsum()
    cumulative_vol = df['volume'].cumsum().replace(0, np.nan)
    return cumulative_tp_vol / cumulative_vol


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Chaikin Money Flow (CMF).
    Measures buying/selling pressure over a rolling window.
    Range: -1 (strong selling) to +1 (strong buying).
    Values > 0.05 = accumulation; < -0.05 = distribution.

    Args:
        df: DataFrame with OHLCV columns
        period: Rolling window (default: 20)

    Returns:
        CMF series
    """
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']

    hl_range = (high - low).replace(0, np.nan)
    money_flow_multiplier = ((close - low) - (high - close)) / hl_range
    money_flow_volume = money_flow_multiplier * volume

    cmf_values = (
        money_flow_volume.rolling(window=period, min_periods=period).sum() /
        volume.rolling(window=period, min_periods=period).sum().replace(0, np.nan)
    )
    return cmf_values.fillna(0)


def cvd(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative Volume Delta (CVD).
    Estimates the net difference between buying and selling volume per candle.

    Formula per candle:
      buy_vol  = ((close - low) / (high - low)) * volume
      sell_vol = ((high - close) / (high - low)) * volume
      delta    = buy_vol - sell_vol
      CVD      = cumsum(delta)

    Interpretation:
      CVD rising  + price rising  = confirmed bullish momentum
      CVD flat/falling + price rising = bearish divergence (distribution)
      CVD rising  + price falling = bullish divergence (accumulation)

    Returns:
        CVD series
    """
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']

    hl_range = (high - low).replace(0, np.nan)
    buy_vol  = ((close - low) / hl_range) * volume
    sell_vol = ((high - close) / hl_range) * volume
    delta = (buy_vol - sell_vol).fillna(0)
    return delta.cumsum()


def analyze_vpa(df: pd.DataFrame, period: int = 20) -> Dict:
    """
    Volume Price Analysis (VPA) — Wyckoff-inspired.
    Detects key VPA signals on the last completed candle:

    - 'climax_buy'   : Wide candle UP + massive volume → potential exhaustion top
    - 'climax_sell'  : Wide candle DOWN + massive volume → potential exhaustion bottom
    - 'effort_no_result_up'   : Strong upward volume but tiny price movement → buying flooded by sellers
    - 'effort_no_result_down' : Strong downward volume but tiny price movement → selling absorbed by buyers
    - 'confirmed_breakout_up'   : Price breaks recent high + high volume → valid breakout
    - 'confirmed_breakout_down' : Price breaks recent low + high volume → valid breakdown
    - 'weak_move_up'   : Price rising but volume below average → suspect rally
    - 'weak_move_down' : Price falling but volume below average → suspect sell-off
    - 'accumulation'   : OBV rising while price flat/down → hidden buying
    - 'distribution'   : OBV falling while price flat/up → hidden selling
    - 'neutral' : No notable VPA signal

    Args:
        df: DataFrame with OHLCV and pre-computed indicators (needs vol_sma_20, obv)
        period: Volume average period (default: 20)

    Returns:
        dict with keys:
            signal (str), description (str), bullish_bias (bool), bearish_bias (bool)
    """
    result = {
        'signal': 'neutral',
        'description': 'No notable VPA signal',
        'bullish_bias': False,
        'bearish_bias': False,
    }

    if len(df) < max(period + 2, 5):
        return result

    # Use second-to-last candle (last closed)
    last = df.iloc[-2]
    prev = df.iloc[-3]

    close  = last['close']
    open_  = last['open']
    high   = last['high']
    low    = last['low']
    vol    = last['volume']

    vol_avg = df['vol_sma_20'].iloc[-2] if 'vol_sma_20' in df.columns else df['volume'].rolling(period).mean().iloc[-2]
    if pd.isna(vol_avg) or vol_avg == 0:
        return result

    candle_range  = high - low
    body          = abs(close - open_)
    avg_range     = (df['high'] - df['low']).rolling(period).mean().iloc[-2]
    is_up_candle  = close > open_
    is_down_candle= close < open_
    high_vol      = vol > vol_avg * 1.8
    low_vol       = vol < vol_avg * 0.7
    wide_range    = candle_range > avg_range * 1.4
    narrow_range  = candle_range < avg_range * 0.4
    recent_high   = df['high'].iloc[-period-2:-2].max()
    recent_low    = df['low'].iloc[-period-2:-2].min()

    # --- OBV Divergence (last 10 bars) ---
    obv_col = 'obv' if 'obv' in df.columns else None
    if obv_col:
        price_slope = df['close'].iloc[-10:].iloc[-1] - df['close'].iloc[-10:].iloc[0]
        obv_slope   = df[obv_col].iloc[-10:].iloc[-1] - df[obv_col].iloc[-10:].iloc[0]

        if price_slope <= 0 and obv_slope > 0:
            result['signal']      = 'accumulation'
            result['description'] = 'OBV rising while price flat/down — hidden buying (accumulation)'
            result['bullish_bias'] = True
            return result

        if price_slope >= 0 and obv_slope < 0:
            result['signal']      = 'distribution'
            result['description'] = 'OBV falling while price flat/up — hidden selling (distribution)'
            result['bearish_bias'] = True
            return result

    # --- Climax candles ---
    if high_vol and wide_range:
        if is_up_candle:
            result['signal']      = 'climax_buy'
            result['description'] = f'Buy Climax: wide UP candle + massive volume ({vol/vol_avg:.1f}x avg) — possible exhaustion top'
            result['bearish_bias'] = True
            return result
        if is_down_candle:
            result['signal']      = 'climax_sell'
            result['description'] = f'Sell Climax: wide DOWN candle + massive volume ({vol/vol_avg:.1f}x avg) — possible exhaustion bottom'
            result['bullish_bias'] = True
            return result

    # --- Effort vs Result ---
    if high_vol and narrow_range:
        if is_up_candle:
            result['signal']      = 'effort_no_result_up'
            result['description'] = f'Effort (high vol) but no result (tiny UP) — sellers absorbing buyers'
            result['bearish_bias'] = True
            return result
        if is_down_candle:
            result['signal']      = 'effort_no_result_down'
            result['description'] = f'Effort (high vol) but no result (tiny DOWN) — buyers absorbing sellers'
            result['bullish_bias'] = True
            return result

    # --- Volume-confirmed breakouts ---
    if high_vol:
        if high > recent_high and is_up_candle:
            result['signal']      = 'confirmed_breakout_up'
            result['description'] = f'Breakout above {recent_high:.4f} confirmed by high volume — bullish'
            result['bullish_bias'] = True
            return result
        if low < recent_low and is_down_candle:
            result['signal']      = 'confirmed_breakout_down'
            result['description'] = f'Breakdown below {recent_low:.4f} confirmed by high volume — bearish'
            result['bearish_bias'] = True
            return result

    # --- Weak moves (low-volume price moves) ---
    if low_vol and wide_range:
        if is_up_candle:
            result['signal']      = 'weak_move_up'
            result['description'] = 'Price rising on low volume — suspect rally, possibly a trap'
            result['bearish_bias'] = True
            return result
        if is_down_candle:
            result['signal']      = 'weak_move_down'
            result['description'] = 'Price falling on low volume — suspect sell-off, possibly oversold'
            result['bullish_bias'] = True
            return result

    return result



def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index (ADX) + DI+ / DI-.
    
    ADX measures TREND STRENGTH (not direction):
    - ADX < 20: Weak/Ranging market (avoid signals)
    - ADX 20-40: Developing trend
    - ADX > 40: Strong trend
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default: 14)
        
    Returns:
        Original DataFrame with added columns:
        - adx: ADX value (0-100)
        - di_plus: Positive Directional Indicator
        - di_minus: Negative Directional Indicator
    """
    df = df.copy()
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    high_diff = high.diff()
    low_diff = low.diff().mul(-1)
    
    plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0), index=df.index)
    
    # Smoothed TR and DM
    atr_s = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_dm_s = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # DI+ and DI-
    di_plus = 100 * (plus_dm_s / atr_s.replace(0, np.nan))
    di_minus = 100 * (minus_dm_s / atr_s.replace(0, np.nan))
    
    # DX and ADX
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx_values = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    df['adx'] = adx_values.fillna(0)
    df['di_plus'] = di_plus.fillna(0)
    df['di_minus'] = di_minus.fillna(0)
    
    return df


def ichimoku(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, 
             senkou_b_period: int = 52, displacement: int = 26) -> pd.DataFrame:
    """
    Ichimoku Cloud .
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        tenkan_period: Period for Tenkan-sen (default: 9)
        kijun_period: Period for Kijun-sen (default: 26)
        senkou_b_period: Period for Senkou Span B (default: 52)
        displacement: Forward displacement for Senkou spans (default: 26)
        
    Returns:
        DataFrame with Ichimoku columns added:
        - ichimoku_tenkan: Tenkan-sen (Conversion Line)
        - ichimoku_kijun: Kijun-sen (Base Line)
        - ichimoku_senkou_a: Senkou Span A (Leading Span A)
        - ichimoku_senkou_b: Senkou Span B (Leading Span B)
        - ichimoku_chikou: Chikou Span (Lagging Span)
    """
    df = df.copy()
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_high = high.rolling(window=tenkan_period, min_periods=tenkan_period).max()
    tenkan_low = low.rolling(window=tenkan_period, min_periods=tenkan_period).min()
    df['ichimoku_tenkan'] = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_high = high.rolling(window=kijun_period, min_periods=kijun_period).max()
    kijun_low = low.rolling(window=kijun_period, min_periods=kijun_period).min()
    df['ichimoku_kijun'] = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted forward
    df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted forward
    senkou_b_high = high.rolling(window=senkou_b_period, min_periods=senkou_b_period).max()
    senkou_b_low = low.rolling(window=senkou_b_period, min_periods=senkou_b_period).min()
    df['ichimoku_senkou_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
    
    # Chikou Span (Lagging Span): Close shifted backward
    df['ichimoku_chikou'] = close.shift(-displacement)
    
    return df


def get_ichimoku_signal(df: pd.DataFrame) -> str:
    """
    Get trading signal based on Ichimoku Cloud analysis.
    
    Args:
        df: DataFrame with Ichimoku indicators
        
    Returns:
        'bullish', 'bearish', or 'neutral'
    """
    if df.empty or len(df) < 2:
        return 'neutral'
    
    # Use last complete candle
    last = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    
    close = last.get('close')
    tenkan = last.get('ichimoku_tenkan')
    kijun = last.get('ichimoku_kijun')
    senkou_a = last.get('ichimoku_senkou_a')
    senkou_b = last.get('ichimoku_senkou_b')
    
    # Check for required values
    if any(pd.isna(v) for v in [close, tenkan, kijun, senkou_a, senkou_b]):
        return 'neutral'
    
    # Cloud top and bottom
    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)
    
    # Strong bullish: Price above cloud + Tenkan above Kijun
    if close > cloud_top and tenkan > kijun:
        return 'bullish'
    
    # Strong bearish: Price below cloud + Tenkan below Kijun
    if close < cloud_bottom and tenkan < kijun:
        return 'bearish'
    
    return 'neutral'


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all standard indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()
    
    # Moving Averages
    df['sma_20'] = sma(df['close'], 20)
    df['sma_50'] = sma(df['close'], 50)
    df['sma_200'] = sma(df['close'], 200)
    
    df['ema_9'] = ema(df['close'], 9)
    df['ema_21'] = ema(df['close'], 21)
    df['ema_50'] = ema(df['close'], 50)
    df['ema_200'] = ema(df['close'], 200)
    
    # Momentum
    df['rsi_14'] = rsi(df['close'], 14)
    
    # Volatility
    df['atr_14'] = atr(df, 14)
    
    # Volume
    df['vol_sma_20'] = volume_sma(df, 20)
    df['volume_spike'] = is_volume_spike(df, 20, 1.8)

    # Volume Price Analysis (VPA)
    df['obv']  = obv(df)
    df['vwap'] = vwap(df)
    df['cmf']  = cmf(df, 20)
    df['cvd']  = cvd(df)

    # Ichimoku Cloud
    df = ichimoku(df)
    
    # ADX - Trend Strength
    df = adx(df)
    
    return df


def get_trend(df: pd.DataFrame) -> str:
    """
    Determine current trend based on EMA crossover.
    
    Args:
        df: DataFrame with EMA indicators
        
    Returns:
        'up', 'down', or 'neutral'
    """
    if df.empty or len(df) < 2:
        return 'neutral'
    
    # Use last complete candle
    last = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    
    ema50 = last.get('ema_50')
    ema200 = last.get('ema_200')
    close = last.get('close')
    
    # Ichimoku Cloud Check
    senkou_a = last.get('ichimoku_senkou_a')
    senkou_b = last.get('ichimoku_senkou_b')
    
    if pd.isna(ema50) or pd.isna(ema200) or pd.isna(close):
        return 'neutral'
        
    cloud_top = max(senkou_a, senkou_b) if not pd.isna(senkou_a) and not pd.isna(senkou_b) else None
    cloud_bottom = min(senkou_a, senkou_b) if not pd.isna(senkou_a) and not pd.isna(senkou_b) else None

    # Trend is strictly UP only if EMA50 > EMA200 AND price is above EMA50 (healthy trend)
    if ema50 > ema200:
        if close < ema200:
            return 'neutral' # Price broke below EMA200 -> Trend is weakening
        return 'up'
        
    # Trend is strictly DOWN only if EMA50 < EMA200 AND price is below EMA50
    elif ema50 < ema200:
        if close > ema200:
            # Price broke above EMA200 but EMAs haven't crossed yet (Potential fakeout / Bull Trap)
            # Check Ichimoku Cloud to confirm
            if cloud_top is not None and close < cloud_top:
                # Still under the cloud -> It's likely a Bull Trap. Keep searching for Shorts.
                return 'down'
            else:
                # Broke above the cloud -> Trend changed, stop shorting.
                return 'neutral'
        return 'down'
        
    else:
        return 'neutral'


def check_rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> Optional[str]:
    """
    Check for RSI divergence.
    
    Args:
        df: DataFrame with price and RSI data
        lookback: Number of bars to check
        
    Returns:
        'bullish', 'bearish', or None
    """
    if len(df) < lookback:
        return None
    
    recent = df.tail(lookback)
    
    # Find price extremes
    price_high_idx = recent['high'].idxmax()
    price_low_idx = recent['low'].idxmin()
    
    # Get RSI at those points
    if 'rsi_14' not in df.columns:
        return None
    
    # Bearish divergence: higher price high but lower RSI high
    try:
        first_half = recent.head(lookback // 2)
        second_half = recent.tail(lookback // 2)
        
        if not first_half.empty and not second_half.empty:
            # Bearish: price making higher highs, RSI making lower highs
            if (second_half['high'].max() > first_half['high'].max() and
                second_half['rsi_14'].max() < first_half['rsi_14'].max()):
                return 'bearish'
            
            # Bullish: price making lower lows, RSI making higher lows
            if (second_half['low'].min() < first_half['low'].min() and
                second_half['rsi_14'].min() > first_half['rsi_14'].min()):
                return 'bullish'
    except Exception:
        pass
    
    return None


