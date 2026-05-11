"""
Smart Money Concepts (SMC) Detection
====================================
Algorithms to identify Institutional Order Flow: Fair Value Gaps (FVG), 
Order Blocks (OB), and Liquidity Sweeps.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class FVG:
    direction: str  # 'bullish' or 'bearish'
    top: float
    bottom: float
    mitigated: bool
    timestamp: pd.Timestamp
    index: int


@dataclass
class OrderBlock:
    direction: str  # 'bullish' or 'bearish'
    top: float
    bottom: float
    timestamp: pd.Timestamp
    index: int


@dataclass
class LiquiditySweep:
    direction: str  # 'bullish' (swept lows, price going up) or 'bearish'
    swept_price: float
    timestamp: pd.Timestamp
    index: int


def detect_fvgs(df: pd.DataFrame, lookback: int = 50) -> List[FVG]:
    """
    Detect Fair Value Gaps (Imbalances) in the recent price action.
    Bullish FVG: Low of candle 3 > High of candle 1.
    Bearish FVG: High of candle 3 < Low of candle 1.
    """
    fvgs = []
    if len(df) < 3:
        return fvgs

    # Only process up to 'lookback' to save on performance
    start_idx = max(2, len(df) - lookback)

    for i in range(start_idx, len(df)):
        c1 = df.iloc[i-2]
        c2 = df.iloc[i-1]
        c3 = df.iloc[i]

        # Bullish FVG Check
        if c3['low'] > c1['high']:
            fvg = FVG(
                direction='bullish',
                top=float(c3['low']),
                bottom=float(c1['high']),
                mitigated=False,
                timestamp=df.index[i-1],  # FVG formed by c2
                index=i-1
            )
            fvgs.append(fvg)

        # Bearish FVG Check
        elif c3['high'] < c1['low']:
            fvg = FVG(
                direction='bearish',
                top=float(c1['low']),
                bottom=float(c3['high']),
                mitigated=False,
                timestamp=df.index[i-1],
                index=i-1
            )
            fvgs.append(fvg)

    # Check for mitigation (if future candles filled the gap)
    for fvg in fvgs:
        future_df = df.iloc[fvg.index + 2:]
        if fvg.direction == 'bullish':
            # Bullish FVG is mitigated if price drops below its top
            if not future_df.empty and future_df['low'].min() <= fvg.bottom:
                fvg.mitigated = True
        else:
            # Bearish FVG is mitigated if price rises above its bottom
            if not future_df.empty and future_df['high'].max() >= fvg.top:
                fvg.mitigated = True

    return [f for f in fvgs if not fvg.mitigated]  # Return mostly unmitigated or recently unmitigated


def detect_unmitigated_fvgs(df: pd.DataFrame, lookback: int = 50) -> List[FVG]:
    """Returns only FVGs that have not been fully filled yet."""
    fvgs = detect_fvgs(df, lookback=lookback)
    return [f for f in fvgs if not f.mitigated]


def detect_order_blocks(df: pd.DataFrame, lookback: int = 50) -> List[OrderBlock]:
    """
    Detect Order Blocks (OB).
    Bullish OB: The last down candle before a strong up move (ideally creating an FVG).
    Bearish OB: The last up candle before a strong down move.
    """
    obs = []
    if len(df) < 5:
        return obs

    start_idx = max(3, len(df) - lookback)
    fvgs = detect_fvgs(df, lookback)

    for fvg in fvgs:
        # OB usually forms right before the imbalance (FVG)
        ob_cand_idx = fvg.index - 1
        if ob_cand_idx < 0:
            continue
            
        c = df.iloc[ob_cand_idx]

        if fvg.direction == 'bullish':
            # Look backwards for the lowest bearish candle in the consolidation before the jump
            search_window = df.iloc[max(0, ob_cand_idx-3):ob_cand_idx+1]
            bearish_cands = search_window[search_window['close'] < search_window['open']]
            
            if not bearish_cands.empty:
                # The lowest bearish candle is our OB
                ob_cand = bearish_cands.loc[bearish_cands['low'].idxmin()]
                obs.append(OrderBlock(
                    direction='bullish',
                    top=float(max(ob_cand['open'], ob_cand['close'])),  # Or High to be safe
                    bottom=float(ob_cand['low']),
                    timestamp=ob_cand.name,
                    index=df.index.get_loc(ob_cand.name)
                ))
        
        elif fvg.direction == 'bearish':
            # Look backwards for the highest bullish candle
            search_window = df.iloc[max(0, ob_cand_idx-3):ob_cand_idx+1]
            bullish_cands = search_window[search_window['close'] > search_window['open']]
            
            if not bullish_cands.empty:
                ob_cand = bullish_cands.loc[bullish_cands['high'].idxmax()]
                obs.append(OrderBlock(
                    direction='bearish',
                    top=float(ob_cand['high']),
                    bottom=float(min(ob_cand['open'], ob_cand['close'])),
                    timestamp=ob_cand.name,
                    index=df.index.get_loc(ob_cand.name)
                ))

    # De-duplicate
    unique_obs = {ob.timestamp: ob for ob in obs}.values()
    return list(sorted(unique_obs, key=lambda x: x.timestamp))


def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 30, min_wick_pct: float = 0.05) -> List[LiquiditySweep]:
    """
    Detect Liquidity Sweeps (Stop hunts / Turtle Soups).
    Occurs when price wicks past a significant previous swing high/low
    but closes back inside the range.

    Filters applied:
    - Swing must be confirmed by 2 candles on each side (stronger structure)
    - Wick beyond the swing level >= min_wick_pct % of price (default 0.05%)
      eliminates insignificant micro-wicks that are just noise
    - Wick size >= 50% of candle body — ensures the sweep was a real aggressive
      move, not a small spike on a large body candle
    """
    sweeps = []
    if len(df) < lookback + 5:
        return sweeps

    # Find swing highs and lows — require 2 candles on EACH side for a stronger swing
    swing_highs = []
    swing_lows = []
    for i in range(2, len(df) - 2):  # -2 so we can check 2 candles ahead
        h = df['high'].iloc[i]
        if h > df['high'].iloc[i-1] and h > df['high'].iloc[i-2] and \
           h > df['high'].iloc[i+1] and h > df['high'].iloc[i+2]:
            swing_highs.append((i, float(h)))

        l = df['low'].iloc[i]
        if l < df['low'].iloc[i-1] and l < df['low'].iloc[i-2] and \
           l < df['low'].iloc[i+1] and l < df['low'].iloc[i+2]:
            swing_lows.append((i, float(l)))

    # Scan the last 3 candles to see if they swept any of these levels
    recent_idx = len(df) - 1
    recent_cand = df.iloc[recent_idx]
    candle_body = abs(float(recent_cand['close']) - float(recent_cand['open']))

    # Bearish Sweep: price wicked above a swing high but closed below it
    for sh_idx, sh_price in swing_highs[-10:]:
        if sh_idx < recent_idx - 3:  # Ensure it's not the same candle complex
            if recent_cand['high'] > sh_price and recent_cand['close'] < sh_price:
                wick_size = float(recent_cand['high']) - sh_price  # pierce above the swing
                min_wick = sh_price * (min_wick_pct / 100)
                # Filter: wick must be meaningful in size AND significant vs candle body
                if wick_size >= min_wick and (candle_body == 0 or wick_size >= candle_body * 0.5):
                    sweeps.append(LiquiditySweep(
                        direction='bearish',
                        swept_price=sh_price,
                        timestamp=df.index[-1],
                        index=recent_idx
                    ))
                    break

    # Bullish Sweep: price wicked below a swing low but closed above it
    for sl_idx, sl_price in swing_lows[-10:]:
        if sl_idx < recent_idx - 3:
            if recent_cand['low'] < sl_price and recent_cand['close'] > sl_price:
                wick_size = sl_price - float(recent_cand['low'])  # pierce below the swing
                min_wick = sl_price * (min_wick_pct / 100)
                # Filter: wick must be meaningful in size AND significant vs candle body
                if wick_size >= min_wick and (candle_body == 0 or wick_size >= candle_body * 0.5):
                    sweeps.append(LiquiditySweep(
                        direction='bullish',
                        swept_price=sl_price,
                        timestamp=df.index[-1],
                        index=recent_idx
                    ))
                    break

    return sweeps


def get_nearest_unmitigated_fvg(fvgs: List[FVG], current_price: float, direction: str) -> Optional[FVG]:
    """
    Get the closest unmitigated FVG in the path of the trade.
    For LONGs ('bullish' targets), we want bearish FVGs that are ABOVE us.
    For SHORTs ('bearish' targets), we want bullish FVGs that are BELOW us.
    """
    valid_fvgs = []
    if direction == 'LONG':
        for f in fvgs:
             if f.bottom > current_price:
                 valid_fvgs.append(f)
        if valid_fvgs:
            return min(valid_fvgs, key=lambda x: x.bottom - current_price)
            
    else:  # SHORT
        for f in fvgs:
             if f.top < current_price:
                 valid_fvgs.append(f)
        if valid_fvgs:
            return min(valid_fvgs, key=lambda x: current_price - x.top)
            
    return None

def get_nearest_order_block(obs: List[OrderBlock], current_price: float, direction: str) -> Optional[OrderBlock]:
    """
    Find the supporting Order Block for the current direction.
    For LONG, we want a bullish OB BELOW the current price to lean our stop loss against.
    For SHORT, we want a bearish OB ABOVE the current price.
    """
    valid_obs = []
    if direction == 'LONG':
        for ob in obs:
            if ob.direction == 'bullish' and ob.top <= current_price * 1.002: # Allow small overlap
                valid_obs.append(ob)
        if valid_obs:
            return min(valid_obs, key=lambda x: current_price - x.top)
            
    else: # SHORT
        for ob in obs:
            if ob.direction == 'bearish' and ob.bottom >= current_price * 0.998:
                valid_obs.append(ob)
        if valid_obs:
            return min(valid_obs, key=lambda x: x.bottom - current_price)
            
    return None


# =============================================================================
# Break of Structure (BOS) & Change of Character (CHoCH)
# =============================================================================

@dataclass
class StructureBreak:
    break_type: str   # 'BOS' or 'CHoCH'
    direction: str    # 'bullish' or 'bearish'
    broken_price: float  # the swing level that was broken
    timestamp: pd.Timestamp
    index: int


def detect_structure_breaks(df: pd.DataFrame, lookback: int = 50) -> List[StructureBreak]:
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).

    BOS  — structural break in the direction of the prevailing swing trend (continuation).
      - Bullish BOS : uptrend + close above previous swing high (new HH)
      - Bearish BOS : downtrend + close below previous swing low (new LL)

    CHoCH — structural break AGAINST the prevailing swing trend (potential reversal).
      - Bullish CHoCH : downtrend + close above previous swing high
      - Bearish CHoCH : uptrend  + close below previous swing low

    Returns list of StructureBreak objects (most recent last).
    """
    breaks: List[StructureBreak] = []
    if len(df) < 10:
        return breaks

    # ── Find swing highs / lows (2 candles confirmation on each side) ──
    swing_highs: List[Tuple[int, float]] = []
    swing_lows:  List[Tuple[int, float]] = []
    for i in range(2, len(df) - 2):
        h = df['high'].iloc[i]
        if (h > df['high'].iloc[i - 1] and h > df['high'].iloc[i - 2] and
                h > df['high'].iloc[i + 1] and h > df['high'].iloc[i + 2]):
            swing_highs.append((i, float(h)))
        l = df['low'].iloc[i]
        if (l < df['low'].iloc[i - 1] and l < df['low'].iloc[i - 2] and
                l < df['low'].iloc[i + 1] and l < df['low'].iloc[i + 2]):
            swing_lows.append((i, float(l)))

    if not swing_highs or not swing_lows:
        return breaks

    # ── Infer prevailing swing trend from last 2 swings ──
    recent_highs = swing_highs[-4:]
    recent_lows  = swing_lows[-4:]
    swing_trend  = 'neutral'
    if len(recent_highs) >= 2 and len(recent_lows) >= 2:
        hh = recent_highs[-1][1] > recent_highs[-2][1]
        hl = recent_lows[-1][1]  > recent_lows[-2][1]
        lh = recent_highs[-1][1] < recent_highs[-2][1]
        ll = recent_lows[-1][1]  < recent_lows[-2][1]
        if hh and hl:
            swing_trend = 'up'
        elif lh and ll:
            swing_trend = 'down'

    last_close = float(df['close'].iloc[-1])
    last_idx   = len(df) - 1
    last_ts    = df.index[-1]

    # ── Check break above a swing high ──
    for sh_idx, sh_price in reversed(swing_highs[-5:]):
        if sh_idx >= last_idx - 1:
            continue
        if last_close > sh_price:
            if swing_trend == 'up':
                breaks.append(StructureBreak('BOS',   'bullish', sh_price, last_ts, last_idx))
            elif swing_trend == 'down':
                breaks.append(StructureBreak('CHoCH', 'bullish', sh_price, last_ts, last_idx))
            break

    # ── Check break below a swing low ──
    for sl_idx, sl_price in reversed(swing_lows[-5:]):
        if sl_idx >= last_idx - 1:
            continue
        if last_close < sl_price:
            if swing_trend == 'down':
                breaks.append(StructureBreak('BOS',   'bearish', sl_price, last_ts, last_idx))
            elif swing_trend == 'up':
                breaks.append(StructureBreak('CHoCH', 'bearish', sl_price, last_ts, last_idx))
            break

    return breaks


def get_latest_structure_break(df: pd.DataFrame, lookback: int = 50) -> Optional[StructureBreak]:
    """Returns the most recent BOS or CHoCH, or None."""
    breaks = detect_structure_breaks(df, lookback=lookback)
    return breaks[-1] if breaks else None
