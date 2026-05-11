import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def manage_open_trades(mt5_client) -> None:
    """Monitors open trades and moves SL to Break Even (BE) when risk is 1:1."""
    if not mt5_client:
        return
        
    try:
        positions = mt5_client.get_all_bot_positions()
    except Exception as e:
        logger.error(f"Failed to fetch bot positions for trade management: {e}")
        return

    for pos in positions:
        ticket = pos.get('ticket')
        pos_type = pos.get('type')  # 0: Buy, 1: Sell
        price_open = pos.get('price_open')
        price_current = pos.get('price_current')
        sl = pos.get('sl')
        symbol = pos.get('symbol')
        
        if not all((ticket, price_open, price_current, sl)):
            continue
            
        sl = float(sl)
        price_open = float(price_open)
        price_current = float(price_current)
        
        if sl == 0:
            continue
            
        # If SL is already better than or equal to open price (BE secured), skip.
        if pos_type == 0:  # BUY
            if sl >= price_open:
                continue
                
            risk_dist = price_open - sl
            profit_dist = price_current - price_open
            
            if profit_dist >= risk_dist and risk_dist > 0:
                logger.info(f"[Trade Manager] {symbol} BUY position #{ticket} hit 1:1 RR. Moving SL to Break Even ({price_open}).")
                mt5_client.modify_position_sl(ticket, price_open)
                
        elif pos_type == 1:  # SELL
            if sl <= price_open:
                continue
                
            risk_dist = sl - price_open
            profit_dist = price_open - price_current
            
            if profit_dist >= risk_dist and risk_dist > 0:
                logger.info(f"[Trade Manager] {symbol} SELL position #{ticket} hit 1:1 RR. Moving SL to Break Even ({price_open}).")
                mt5_client.modify_position_sl(ticket, price_open)

