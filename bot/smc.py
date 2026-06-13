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
class DealingRange:
    top: float
    bottom: float
    equilibrium: float
    trend: str  # 'uptrend' or 'downtrend'

@dataclass
class FVG:
    direction: str  # 'bullish' or 'bearish'
    top: float
    bottom: float
    mitigated: bool
    timestamp: pd.Timestamp
    index: int
    inverted: bool = False  # True = iFVG (mitigated FVG that flipped polarity)


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
    killzone: str = "None"  # e.g., 'London', 'NewYork', 'Asian', 'None'

def get_killzone_session(ts: pd.Timestamp) -> str:
    """
    Identifies the institutional trading killzone (based on UTC usually).
    London Killzone: 07:00 - 10:00
    New York Killzone: 12:00 - 15:00
    London Close Killzone: 15:00 - 17:00
    Asian Session / Accumulation: 00:00 - 06:00
    """
    hour = ts.hour
    if 7 <= hour < 11:
        return 'London'
    elif 12 <= hour < 15:
        return 'NewYork'
    elif 15 <= hour < 17:
        return 'LondonClose'
    elif 0 <= hour < 6:
        return 'Asian'
    return 'None'


@dataclass
class DailyLiquidityLevels:
    pdh: float  # Previous Day High
    pdl: float  # Previous Day Low
    timestamp: pd.Timestamp

def get_volume_profile_poc(df: pd.DataFrame, lookback: int = 200, bins: int = 50) -> float:
    """
    Calculates the Point of Control (POC) using Volume Profile over a lookback period.
    The POC is the price level with the highest traded volume.
    """
    if len(df) < lookback:
        lookback = len(df)
    if lookback < 50:
        return 0.0
        
    recent_df = df.iloc[-lookback:]
    min_price = recent_df['low'].min()
    max_price = recent_df['high'].max()
    
    if min_price == max_price:
        return min_price
        
    # Create price bins
    price_bins = np.linspace(min_price, max_price, bins)
    volume_profile = np.zeros(bins - 1)
    
    for i in range(len(recent_df)):
        c = recent_df.iloc[i]
        hl_range = c['high'] - c['low']
        if hl_range == 0:
            continue
            
        # Distribute volume proportionally across bins that overlap the candle's high/low
        for b in range(bins - 1):
            bin_low = price_bins[b]
            bin_high = price_bins[b+1]
            
            # Check overlap
            overlap_low = max(c['low'], bin_low)
            overlap_high = min(c['high'], bin_high)
            
            if overlap_low < overlap_high:
                weight = (overlap_high - overlap_low) / hl_range
                volume_profile[b] += c['volume'] * weight
                
    max_vol_bin = np.argmax(volume_profile)
    poc_price = (price_bins[max_vol_bin] + price_bins[max_vol_bin + 1]) / 2.0
    return poc_price

def get_previous_day_liquidity(df: pd.DataFrame) -> Optional[DailyLiquidityLevels]:
    """Calculates the Previous Day High (PDH) and Previous Day Low (PDL) from a DataFrame."""
    if len(df) < 100:
        return None
    try:
        # Group by Date (UTC) to find daily highs and lows
        daily_df = df.resample('D').agg({'high': 'max', 'low': 'min'})
        if len(daily_df) < 2:
            return None
        # The row at index -2 is the previous completed day
        prev_day = daily_df.iloc[-2]
        return DailyLiquidityLevels(
            pdh=float(prev_day['high']),
            pdl=float(prev_day['low']),
            timestamp=prev_day.name
        )
    except Exception:
        return None

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

    return fvgs  # Return all FVGs, let the unmitigated function filter them


def get_dealing_range(df: pd.DataFrame, lookback: int = 150) -> Optional[DealingRange]:
    """Finds the recent Premium/Discount boundaries based on major swings."""
    if len(df) < 5:
        return None
        
    start_idx = max(2, len(df) - lookback)
    
    swing_highs = []
    swing_lows = []
    
    for i in range(start_idx, len(df) - 2):
        h = df['high'].iloc[i]
        if (h > df['high'].iloc[i - 1] and h > df['high'].iloc[i - 2] and
                h > df['high'].iloc[i + 1] and h > df['high'].iloc[i + 2]):
            swing_highs.append((i, float(h)))

        l = df['low'].iloc[i]
        if (l < df['low'].iloc[i - 1] and l < df['low'].iloc[i - 2] and
                l < df['low'].iloc[i + 1] and l < df['low'].iloc[i + 2]):
            swing_lows.append((i, float(l)))
            
    if not swing_highs or not swing_lows:
        return None
        
    last_high_idx, last_high_val = swing_highs[-1]
    last_low_idx, last_low_val = swing_lows[-1]
    
    trend = 'uptrend' if last_high_idx > last_low_idx else 'downtrend'
    
    return DealingRange(
        top=max(last_high_val, last_low_val),
        bottom=min(last_high_val, last_low_val),
        equilibrium=(last_high_val + last_low_val) / 2.0,
        trend=trend
    )

def filter_pd_array_fvgs(fvgs: List[FVG], df: pd.DataFrame) -> List[FVG]:
    """
    Applies Premium & Discount Matrix rules:
    - Only keeps Bullish FVGs if they are in the Discount zone (< 50%).
    - Only keeps Bearish FVGs if they are in the Premium zone (> 50%).
    """
    dr = get_dealing_range(df)
    if not dr:
        return fvgs  # Fallback to no filter if no range

    good_fvgs = []
    for f in fvgs:
        if f.direction == 'bullish':
            # Must be inside discount
            if f.top < dr.equilibrium:
                good_fvgs.append(f)
        else:
            # Must be inside premium
            if f.bottom > dr.equilibrium:
                good_fvgs.append(f)
    return good_fvgs


def detect_unmitigated_fvgs(df: pd.DataFrame, lookback: int = 50) -> List[FVG]:
    """Returns only high-probability FVGs inside the correct Premium/Discount zones that have not been filled."""
    fvgs = detect_fvgs(df, lookback=lookback)
    unmitigated = [f for f in fvgs if not f.mitigated]
    return filter_pd_array_fvgs(unmitigated, df)


def detect_ifvgs(df: pd.DataFrame, lookback: int = 100) -> List[FVG]:
    """
    Detect Inversion Fair Value Gaps (iFVG).

    An iFVG forms when a regular FVG gets FULLY mitigated (price passed through
    the entire gap), then the zone FLIPS polarity:
    - Bullish FVG that got fully filled → becomes bearish iFVG (resistance)
    - Bearish FVG that got fully filled → becomes bullish iFVG (support)

    Why it works: the original FVG was an institutional imbalance zone.
    When price returns to fill it completely, the zone often "inverts" and
    the same institution now defends it from the opposite side.

    Premium/Discount filter applied:
    - Bullish iFVG (now acts as support) must be in Discount zone.
    - Bearish iFVG (now acts as resistance) must be in Premium zone.

    المبدأ: FVG تعبّأ بالكامل → انعكاس الدور → iFVG كـ POI عكسي
    """
    ifvgs: List[FVG] = []
    if len(df) < 5:
        return ifvgs

    all_fvgs = detect_fvgs(df, lookback=lookback)

    for fvg in all_fvgs:
        if not fvg.mitigated:
            continue  # Only care about fully mitigated FVGs

        # Check how deeply price entered the zone after formation
        future_df = df.iloc[fvg.index + 2:]
        if future_df.empty:
            continue

        fully_pierced = False
        if fvg.direction == 'bullish':
            # Bullish FVG fully pierced = price closed BELOW the FVG bottom
            # (not just touched it — full close-through)
            if future_df['close'].min() < fvg.bottom:
                fully_pierced = True
            # Create inverse bearish iFVG (zone now acts as resistance)
            if fully_pierced:
                ifvg = FVG(
                    direction='bearish',   # Inverted direction
                    top=fvg.top,
                    bottom=fvg.bottom,
                    mitigated=False,       # Fresh as an iFVG POI
                    timestamp=fvg.timestamp,
                    index=fvg.index,
                    inverted=True,
                )
                ifvgs.append(ifvg)

        elif fvg.direction == 'bearish':
            # Bearish FVG fully pierced = price closed ABOVE the FVG top
            if future_df['close'].max() > fvg.top:
                fully_pierced = True
            # Create inverse bullish iFVG (zone now acts as support)
            if fully_pierced:
                ifvg = FVG(
                    direction='bullish',   # Inverted direction
                    top=fvg.top,
                    bottom=fvg.bottom,
                    mitigated=False,
                    timestamp=fvg.timestamp,
                    index=fvg.index,
                    inverted=True,
                )
                ifvgs.append(ifvg)

    # Apply Premium/Discount filter — same rule as regular FVGs
    return filter_pd_array_fvgs(ifvgs, df)


def filter_pd_array_obs(obs: List[OrderBlock], df: pd.DataFrame) -> List[OrderBlock]:
    """Applies Premium & Discount Matrix rules to Order Blocks."""
    dr = get_dealing_range(df)
    if not dr:
        return obs

    good_obs = []
    for ob in obs:
        if ob.direction == 'bullish':
            if ob.top < dr.equilibrium:
                good_obs.append(ob)
        else:
            if ob.bottom > dr.equilibrium:
                good_obs.append(ob)
    return good_obs


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
    final_obs = list(sorted(unique_obs, key=lambda x: x.timestamp))
    return filter_pd_array_obs(final_obs, df)


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    lookback: int = 30,
    min_wick_pct: float = 0.05,
    scan_last: int = 1,
) -> List[LiquiditySweep]:
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

    scan_last: how many of the most recent candles to test for a sweep. The
    default of 1 only flags a sweep on the very last candle (legacy behaviour).
    A larger window returns every sweep in that window with its bar index, so a
    multi-step confluence (sweep -> displacement -> shift -> retracement) can
    reference a sweep that happened several candles earlier.
    """
    sweeps: List[LiquiditySweep] = []
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

    start = max(2, len(df) - max(1, scan_last))
    for j in range(start, len(df)):
        cand = df.iloc[j]
        candle_body = abs(float(cand['close']) - float(cand['open']))

        # Bearish Sweep: candle wicked above a prior swing high but closed below it
        for sh_idx, sh_price in swing_highs:
            if sh_idx <= j - 3:  # swing confirmed before this candle
                if cand['high'] > sh_price and cand['close'] < sh_price:
                    wick_size = float(cand['high']) - sh_price
                    min_wick = sh_price * (min_wick_pct / 100)
                    if wick_size >= min_wick and (candle_body == 0 or wick_size >= candle_body * 0.5):
                        sweeps.append(LiquiditySweep(
                            direction='bearish', swept_price=sh_price,
                            timestamp=df.index[j], index=j,
                            killzone=get_killzone_session(df.index[j]),
                        ))
                        break

        # Bullish Sweep: candle wicked below a prior swing low but closed above it
        for sl_idx, sl_price in swing_lows:
            if sl_idx <= j - 3:
                if cand['low'] < sl_price and cand['close'] > sl_price:
                    wick_size = sl_price - float(cand['low'])
                    min_wick = sl_price * (min_wick_pct / 100)
                    if wick_size >= min_wick and (candle_body == 0 or wick_size >= candle_body * 0.5):
                        sweeps.append(LiquiditySweep(
                            direction='bullish', swept_price=sl_price,
                            timestamp=df.index[j], index=j,
                            killzone=get_killzone_session(df.index[j]),
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
# Displacement Detection (Strong Impulsive Moves)
# =============================================================================

@dataclass
class Displacement:
    direction: str        # 'bullish' or 'bearish'
    magnitude_atr: float  # size relative to ATR
    body_ratio: float     # body / total range
    start_index: int
    end_index: int
    timestamp: pd.Timestamp


def detect_displacement(
    df: pd.DataFrame,
    lookback: int = 10,
    min_atr_mult: float = 1.5,
    min_body_ratio: float = 0.65,
) -> Optional[Displacement]:
    """
    Detect a strong impulsive price move (displacement) in recent candles.

    A displacement is a large, decisive candle (or 2-3 consecutive candles) where:
    - The total move exceeds min_atr_mult × ATR
    - The body covers at least min_body_ratio of the total range
      (i.e., it's a real move, not a wick-dominated candle)

    This is used to confirm that institutional interest is present after a
    liquidity sweep — without displacement, a sweep alone is meaningless.

    Returns the most recent displacement, or None if none found.
    """
    if len(df) < lookback + 14:
        return None

    # Calculate ATR if not present
    if 'atr_14' in df.columns:
        atr = float(df['atr_14'].iloc[-1])
    else:
        tr_series = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr_series.rolling(14).mean().iloc[-1])

    if atr <= 0:
        return None

    start = max(0, len(df) - lookback)

    # Check single candles first (strongest signal)
    for i in range(len(df) - 1, start - 1, -1):
        c = df.iloc[i]
        body = abs(float(c['close']) - float(c['open']))
        total_range = float(c['high']) - float(c['low'])
        if total_range <= 0:
            continue

        body_ratio = body / total_range
        magnitude = total_range / atr

        if magnitude >= min_atr_mult and body_ratio >= min_body_ratio:
            direction = 'bullish' if float(c['close']) > float(c['open']) else 'bearish'
            return Displacement(
                direction=direction,
                magnitude_atr=round(magnitude, 2),
                body_ratio=round(body_ratio, 2),
                start_index=i,
                end_index=i,
                timestamp=df.index[i],
            )

    # Check 2-candle combinations (still strong)
    for i in range(len(df) - 1, start, -1):
        c1 = df.iloc[i - 1]
        c2 = df.iloc[i]
        combo_open = float(c1['open'])
        combo_close = float(c2['close'])
        combo_high = max(float(c1['high']), float(c2['high']))
        combo_low = min(float(c1['low']), float(c2['low']))
        body = abs(combo_close - combo_open)
        total_range = combo_high - combo_low
        if total_range <= 0:
            continue

        body_ratio = body / total_range
        magnitude = total_range / atr

        if magnitude >= min_atr_mult and body_ratio >= min_body_ratio:
            # Both candles should move in the same direction
            dir1 = float(c1['close']) > float(c1['open'])
            dir2 = float(c2['close']) > float(c2['open'])
            if dir1 == dir2:
                direction = 'bullish' if combo_close > combo_open else 'bearish'
                return Displacement(
                    direction=direction,
                    magnitude_atr=round(magnitude, 2),
                    body_ratio=round(body_ratio, 2),
                    start_index=i - 1,
                    end_index=i,
                    timestamp=df.index[i],
                )

    return None


def detect_displacements(
    df: pd.DataFrame,
    lookback: int = 30,
    min_atr_mult: float = 1.5,
    min_body_ratio: float = 0.65,
) -> List[Displacement]:
    """Return ALL qualifying single-candle displacements in the recent window.

    Unlike detect_displacement (which returns only the most recent), this lets a
    multi-step confluence locate the impulse leg in the bias direction even when
    the latest displacement is the counter-bias pullback into the POI.
    """
    results: List[Displacement] = []
    if len(df) < 14:
        return results

    if 'atr_14' in df.columns:
        atr = float(df['atr_14'].iloc[-1])
    else:
        tr_series = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr_series.rolling(14).mean().iloc[-1])
    if atr <= 0:
        return results

    start = max(0, len(df) - lookback)
    for i in range(start, len(df)):
        c = df.iloc[i]
        body = abs(float(c['close']) - float(c['open']))
        total_range = float(c['high']) - float(c['low'])
        if total_range <= 0:
            continue
        body_ratio = body / total_range
        magnitude = total_range / atr
        if magnitude >= min_atr_mult and body_ratio >= min_body_ratio:
            results.append(Displacement(
                direction='bullish' if float(c['close']) > float(c['open']) else 'bearish',
                magnitude_atr=round(magnitude, 2),
                body_ratio=round(body_ratio, 2),
                start_index=i,
                end_index=i,
                timestamp=df.index[i],
            ))
    return results


# =============================================================================
# Sweep vs Acceptance Classification
# =============================================================================

def classify_sweep_vs_acceptance(
    df: pd.DataFrame,
    sweep: LiquiditySweep,
    candles_after: int = 3,
) -> str:
    """
    After a liquidity sweep, classify whether it was:

    - 'sweep_rejection': Price wicked past the level but closed back inside
      and subsequent candles confirm rejection (wick-only move).
      → Potential reversal setup.

    - 'acceptance': Price closed strongly beyond the swept level and
      subsequent candles held or continued beyond it.
      → Continuation / do NOT fade.

    This is the golden rule: "أخذ BSL = احتمال انعكاس، وليس إشارة بيع وحدها"
    - Sweep + rejection → look for reversal
    - Sweep + acceptance → continuation, do NOT reverse
    """
    sweep_idx = sweep.index
    swept_price = sweep.swept_price

    # Need candles after the sweep to classify
    if sweep_idx + candles_after >= len(df):
        # Not enough data after sweep — conservative: treat as rejection
        # (the sweep candle itself already closed inside by definition)
        return 'sweep_rejection'

    # Check candles AFTER the sweep
    closes_beyond = 0
    for offset in range(1, candles_after + 1):
        idx = sweep_idx + offset
        if idx >= len(df):
            break
        candle_close = float(df.iloc[idx]['close'])

        if sweep.direction == 'bearish':
            # Bearish sweep = swept a high → check if closes stay ABOVE
            if candle_close > swept_price:
                closes_beyond += 1
        else:
            # Bullish sweep = swept a low → check if closes stay BELOW
            if candle_close < swept_price:
                closes_beyond += 1

    # If majority of subsequent candles closed beyond the swept level
    # → acceptance (price accepted the new level, not rejecting)
    if closes_beyond >= max(1, candles_after // 2 + 1):
        return 'acceptance'

    return 'sweep_rejection'


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


def detect_structure_breaks(
    df: pd.DataFrame,
    lookback: int = 50,
    require_body_close: bool = False,
) -> List[StructureBreak]:
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).

    BOS  — structural break in the direction of the prevailing swing trend (continuation).
      - Bullish BOS : uptrend + close above previous swing high (new HH)
      - Bearish BOS : downtrend + close below previous swing low (new LL)

    CHoCH — structural break AGAINST the prevailing swing trend (potential reversal).
      - Bullish CHoCH : downtrend + close above previous swing high
      - Bearish CHoCH : uptrend  + close below previous swing low

    Args:
        df: OHLCV DataFrame
        lookback: Number of candles to scan for swing points
        require_body_close: If True, the break must be confirmed by a candle
            whose BODY (not just wick) closes beyond the swing level.
            This filters out wick-only fakeout breaks.
            القاعدة: "كسر هيكل حقيقي: BOS أو CHOCH بجسم شمعة، مو مجرد ذيل"

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
    last_open  = float(df['open'].iloc[-1])
    last_idx   = len(df) - 1
    last_ts    = df.index[-1]

    # Body boundaries for body-close validation
    body_top    = max(last_close, last_open)
    body_bottom = min(last_close, last_open)

    # ── Check break above a swing high ──
    for sh_idx, sh_price in reversed(swing_highs[-5:]):
        if sh_idx >= last_idx - 1:
            continue
        if last_close > sh_price:
            # Body-close filter: the body bottom must be above the swing
            if require_body_close and body_bottom <= sh_price:
                continue  # Only wick crossed — not a real break
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
            # Body-close filter: the body top must be below the swing
            if require_body_close and body_top >= sl_price:
                continue  # Only wick crossed — not a real break
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
