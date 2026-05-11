"""
Risk Management
===============
Entry, Stop Loss, and Take Profit calculations.
"""

from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .zones import Zone
from .utils import get_decimal_places


@dataclass
class RiskLevels:
    """Container for calculated risk levels."""
    entry: float
    stop_loss: float
    take_profit_near: float
    take_profit_1: float
    take_profit_2: float
    reward_tp_near: float
    risk_amount: float
    reward_tp1: float
    reward_tp2: float
    rr_ratio_tp_near: float
    rr_ratio_tp1: float
    rr_ratio_tp2: float
    spread_cost: float = 0.0
    rr_ratio_tp_near_net: float = 0.0
    rr_ratio_tp1_net: float = 0.0
    rr_ratio_tp2_net: float = 0.0


def calculate_entry(ticker: Dict, side: str) -> float:
    """
    Calculate entry price based on side.
    
    Args:
        ticker: Ticker dict with 'bid', 'ask', 'last'
        side: 'LONG' or 'SHORT'
        
    Returns:
        Entry price
    """
    if side == 'LONG':
        return ticker.get('ask') or ticker.get('last')
    else:  # SHORT
        return ticker.get('bid') or ticker.get('last')


def calculate_spread_cost(ticker: Dict) -> float:
    """
    Estimate spread in price units from ticker.
    """
    ask = ticker.get('ask')
    bid = ticker.get('bid')
    try:
        ask_f = float(ask)
        bid_f = float(bid)
    except Exception:
        return 0.0
    if ask_f <= 0 or bid_f <= 0:
        return 0.0
    return abs(ask_f - bid_f)


def calculate_stop_loss(
    entry: float,
    side: str,
    zone: Optional[Zone],
    buffer_pct: float = 0.15,
    atr: Optional[float] = None,
    atr_stop_mult: float = 1.8,
    atr_buffer_mult: float = 0.30,
) -> float:
    """
    Calculate stop loss level.
    
    For LONG:
    - Below support zone
    - Minus buffer percentage

    For SHORT:
    - Above resistance zone
    - Plus buffer percentage
    
    Args:
        entry: Entry price
        side: 'LONG' or 'SHORT'
        zone: Nearest S/R zone
        buffer_pct: Buffer as percentage (default 0.15%)
        atr: Optional ATR for fallback calculation
        
    Returns:
        Stop loss price
    """
    if side == 'LONG':
        candidates = []

        if zone and zone.type == 'support':
            candidates.append(zone.lower)
        
        if candidates:
            # Use the lowest value (furthest from entry)
            base_sl = min(candidates)
        else:
            # Fallback: use ATR or percentage
            if atr:
                base_sl = entry - (atr * max(0.8, float(atr_stop_mult)))
            else:
                base_sl = entry * 0.98  # 2% default
        
        # Apply buffer
        if atr and atr > 0:
            buffer_amount = max(
                atr * max(0.05, float(atr_buffer_mult)),
                entry * (max(0.01, float(buffer_pct)) / 100.0),
            )
            stop_loss = base_sl - buffer_amount
        else:
            stop_loss = base_sl * (1 - buffer_pct / 100)
        
    else:  # SHORT
        candidates = []

        if zone and zone.type == 'resistance':
            candidates.append(zone.upper)
        
        if candidates:
            # Use the highest value (furthest from entry)
            base_sl = max(candidates)
        else:
            # Fallback: use ATR or percentage
            if atr:
                base_sl = entry + (atr * max(0.8, float(atr_stop_mult)))
            else:
                base_sl = entry * 1.02  # 2% default
        
        # Apply buffer
        if atr and atr > 0:
            buffer_amount = max(
                atr * max(0.05, float(atr_buffer_mult)),
                entry * (max(0.01, float(buffer_pct)) / 100.0),
            )
            stop_loss = base_sl + buffer_amount
        else:
            stop_loss = base_sl * (1 + buffer_pct / 100)

    # R1: Enforce minimum SL distance (0.8%) to avoid being stopped out by noise
    # Raised from 0.5% — fix stop_too_tight: tight SLs were the #2 most common LLM mistake tag
    min_sl_dist = entry * 0.008
    if side == 'LONG':
        if (entry - stop_loss) < min_sl_dist:
            stop_loss = entry - min_sl_dist
    else:  # SHORT
        if (stop_loss - entry) < min_sl_dist:
            stop_loss = entry + min_sl_dist

    return stop_loss


def calculate_take_profits(
    entry: float,
    stop_loss: float,
    side: str,
    rr_tp1: float = 1.0,
    rr_tp2: float = 2.0,
    next_zone: Optional[Zone] = None,
    atr: Optional[float] = None,
    tp2_atr_mult: float = 4.0,
) -> Tuple[float, float]:
    """
    Calculate take profit levels (dynamic R:R).

    Priority:
    1. TP1: Use next S/R zone if it falls between 0.8:1 and 2.5:1 R:R (dynamic)
    2. TP2: ATR*4 based or fixed ratio, whichever is further from entry

    Args:
        entry: Entry price
        stop_loss: Stop loss price
        side: 'LONG' or 'SHORT'
        rr_tp1: Risk-reward ratio for TP1 (default 1:1)
        rr_tp2: Risk-reward ratio for TP2 (default 1:2)
        next_zone: Optional next S/R zone
        atr: Optional ATR value for dynamic TP2

    Returns:
        Tuple of (TP1, TP2) prices
    """
    risk = abs(entry - stop_loss)

    if side == 'LONG':
        tp1_fixed = entry + (risk * rr_tp1)
        tp2_fixed = entry + (risk * rr_tp2)

        # Use next resistance zone for TP1 if R:R is sensible
        if next_zone and next_zone.type == 'resistance':
            zone_tp = next_zone.lower
            zone_rr = (zone_tp - entry) / risk if risk > 0 else 0
            tp1 = zone_tp if 0.8 <= zone_rr <= 2.5 else tp1_fixed
        else:
            tp1 = tp1_fixed

        # TP2: max of fixed and ATR-based
        tp2_atr = entry + (atr * max(1.0, float(tp2_atr_mult))) if atr and atr > 0 else tp2_fixed
        tp2 = max(tp2_fixed, tp2_atr)
        if tp2 <= tp1:
            tp2 = entry + (risk * rr_tp2)

    else:  # SHORT
        tp1_fixed = entry - (risk * rr_tp1)
        tp2_fixed = entry - (risk * rr_tp2)

        # Use next support zone for TP1 if R:R is sensible
        if next_zone and next_zone.type == 'support':
            zone_tp = next_zone.upper
            zone_rr = (entry - zone_tp) / risk if risk > 0 else 0
            tp1 = zone_tp if 0.8 <= zone_rr <= 2.5 else tp1_fixed
        else:
            tp1 = tp1_fixed

        # TP2: min of fixed and ATR-based
        tp2_atr = entry - (atr * max(1.0, float(tp2_atr_mult))) if atr and atr > 0 else tp2_fixed
        tp2 = min(tp2_fixed, tp2_atr)
        if tp2 >= tp1:
            tp2 = entry - (risk * rr_tp2)

    return tp1, tp2


def calculate_near_take_profit(
    entry: float,
    tp1: float,
    side: str,
    near_pct: float = 0.35,
    near_min_pct: float = 0.12,
    tp1_fraction: float = 0.35,
) -> float:
    """
    Calculate a short-term take-profit level for lower timeframes.

    The near target is intentionally conservative:
    - capped by a small fixed % move from entry (default 0.35%)
    - also capped to be a fraction of TP1 distance (default 35%)
    - always stays before TP1 in trade direction
    """
    distance_to_tp1 = abs(tp1 - entry)
    pct_distance = entry * max(near_pct, 0.05) / 100.0
    min_distance = entry * max(near_min_pct, 0.03) / 100.0
    frac_distance = distance_to_tp1 * max(min(tp1_fraction, 0.9), 0.1)
    distance = max(min(pct_distance, frac_distance), min_distance)

    # If TP1 is very close, keep a small but non-zero near distance.
    if distance <= 0:
        distance = max(distance_to_tp1 * 0.3, entry * 0.0005)

    if side == 'LONG':
        target = entry + distance
        if target >= tp1:
            target = entry + (distance_to_tp1 * 0.7)
    else:
        target = entry - distance
        if target <= tp1:
            target = entry - (distance_to_tp1 * 0.7)
    return target



def calculate_risk_levels(
    ticker: Dict,
    side: str,
    zone: Optional[Zone],
    next_zone: Optional[Zone] = None,
    buffer_pct: float = 0.15,
    rr_tp1: float = 1.0,
    rr_tp2: float = 2.0,
    atr: Optional[float] = None,
    quick_tp_pct: float = 0.35,
    quick_tp_min_pct: float = 0.12,
    quick_tp1_fraction: float = 0.35,
    atr_stop_mult: float = 1.8,
    atr_buffer_mult: float = 0.30,
    tp2_atr_mult: float = 4.0,
    support_ob: Optional[Any] = None,
    fvg_target: Optional[Any] = None,
) -> RiskLevels:
    """
    Calculate all risk/reward levels.
    
    Args:
        ticker: Current ticker data
        side: 'LONG' or 'SHORT'
        zone: Nearest zone for SL
        next_zone: Next zone for potential TP adjustment
        buffer_pct: SL buffer percentage
        rr_tp1: RR ratio for TP1
        rr_tp2: RR ratio for TP2
        atr: Optional ATR value
        quick_tp_pct: Max percent move for short-term target
        quick_tp1_fraction: Max TP1-distance fraction for short-term target
        
    Returns:
        RiskLevels dataclass with all calculated values
    """
    entry = calculate_entry(ticker, side)
    stop_loss = calculate_stop_loss(
        entry,
        side,
        zone,
        buffer_pct,
        atr,
        atr_stop_mult=atr_stop_mult,
        atr_buffer_mult=atr_buffer_mult,
    )
    tp1, tp2 = calculate_take_profits(
        entry,
        stop_loss,
        side,
        rr_tp1,
        rr_tp2,
        next_zone,
        atr,
        tp2_atr_mult=tp2_atr_mult,
    )

    # Overrides based on SMC (Smart Money Concepts)
    if support_ob:
        # Stop loss goes slightly behind the Order Block + small buffer
        buffer_amount = entry * 0.0015
        if side == 'LONG':
            stop_loss = support_ob.bottom - buffer_amount
        else:
            stop_loss = support_ob.top + buffer_amount

    if fvg_target:
        # FVG acts as a magnet. If it's close, TP1 goes to the near edge. TP2 to mid/far edge.
        if side == 'LONG':
            # Target is a bearish FVG above us
            tp1_fvg = fvg_target.bottom
            tp2_fvg = (fvg_target.bottom + fvg_target.top) / 2.0
            if tp1_fvg > entry:
                tp1 = tp1_fvg
            if tp2_fvg > tp1:
                tp2 = tp2_fvg
        else:
            # Target is a bullish FVG below us
            tp1_fvg = fvg_target.top
            tp2_fvg = (fvg_target.top + fvg_target.bottom) / 2.0
            if tp1_fvg < entry:
                tp1 = tp1_fvg
            if tp2_fvg < tp1:
                tp2 = tp2_fvg
    tp_near = calculate_near_take_profit(
        entry=entry,
        tp1=tp1,
        side=side,
        near_pct=quick_tp_pct,
        near_min_pct=quick_tp_min_pct,
        tp1_fraction=quick_tp1_fraction,
    )
    
    risk_amount = abs(entry - stop_loss)
    reward_tp_near = abs(tp_near - entry)
    reward_tp1 = abs(tp1 - entry)
    reward_tp2 = abs(tp2 - entry)
    spread_cost = calculate_spread_cost(ticker)

    # Net R:R approximation after spread impact.
    # Reward is reduced by spread; risk is increased by spread.
    risk_amount_net = risk_amount + spread_cost
    reward_tp_near_net = max(0.0, reward_tp_near - spread_cost)
    reward_tp1_net = max(0.0, reward_tp1 - spread_cost)
    reward_tp2_net = max(0.0, reward_tp2 - spread_cost)
    
    rr_ratio_tp_near = reward_tp_near / risk_amount if risk_amount > 0 else 0
    rr_ratio_tp1 = reward_tp1 / risk_amount if risk_amount > 0 else 0
    rr_ratio_tp2 = reward_tp2 / risk_amount if risk_amount > 0 else 0
    rr_ratio_tp_near_net = reward_tp_near_net / risk_amount_net if risk_amount_net > 0 else 0
    rr_ratio_tp1_net = reward_tp1_net / risk_amount_net if risk_amount_net > 0 else 0
    rr_ratio_tp2_net = reward_tp2_net / risk_amount_net if risk_amount_net > 0 else 0
    
    return RiskLevels(
        entry=entry,
        stop_loss=stop_loss,
        take_profit_near=tp_near,
        take_profit_1=tp1,
        take_profit_2=tp2,
        reward_tp_near=reward_tp_near,
        risk_amount=risk_amount,
        reward_tp1=reward_tp1,
        reward_tp2=reward_tp2,
        rr_ratio_tp_near=rr_ratio_tp_near,
        rr_ratio_tp1=rr_ratio_tp1,
        rr_ratio_tp2=rr_ratio_tp2,
        spread_cost=spread_cost,
        rr_ratio_tp_near_net=rr_ratio_tp_near_net,
        rr_ratio_tp1_net=rr_ratio_tp1_net,
        rr_ratio_tp2_net=rr_ratio_tp2_net,
    )


def format_risk_levels(levels: RiskLevels, symbol: str = "") -> Dict[str, str]:
    """
    Format risk levels with appropriate decimal places.
    
    Args:
        levels: RiskLevels to format
        symbol: Optional symbol for context
        
    Returns:
        Dictionary with formatted string values
    """
    decimals = get_decimal_places(levels.entry)
    
    return {
        'entry': f"{levels.entry:.{decimals}f}",
        'stop_loss': f"{levels.stop_loss:.{decimals}f}",
        'take_profit_near': f"{levels.take_profit_near:.{decimals}f}",
        'take_profit_1': f"{levels.take_profit_1:.{decimals}f}",
        'take_profit_2': f"{levels.take_profit_2:.{decimals}f}",
        'risk_pct': f"{(levels.risk_amount / levels.entry * 100):.2f}%",
        'rr_tp_near': f"1:{levels.rr_ratio_tp_near:.1f}",
        'rr_tp1': f"1:{levels.rr_ratio_tp1:.1f}",
        'rr_tp2': f"1:{levels.rr_ratio_tp2:.1f}"
    }
