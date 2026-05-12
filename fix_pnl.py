import re

with open("main.py", "r", encoding="utf-8") as f:
    text = f.read()

old_str = """                    # Stop handling (normal SL before TP1; BE/trailing after TP1).
                    if not hit_outcome:
                        if side == 'LONG' and price <= effective_stop:
                            if tp1_touched and trailing_enabled and trail_stop is not None and trail_stop > be_price:
                                hit_outcome = 'TRAIL_HIT'
                            elif tp1_touched and be_armed:
                                hit_outcome = 'BE_HIT'
                            else:
                                hit_outcome = 'SL_HIT'
                            pnl_pct = (price - entry) / entry * 100                        elif side == 'SHORT' and price >= effective_stop:
                            if tp1_touched and trailing_enabled and trail_stop is not None and trail_stop < be_price:
                                hit_outcome = 'TRAIL_HIT'
                            elif tp1_touched and be_armed:
                                hit_outcome = 'BE_HIT'
                            else:
                                hit_outcome = 'SL_HIT'
                            pnl_pct = (entry - price) / entry * 100            
            if hit_outcome:"""

new_str = """                    # Stop handling (normal SL before TP1; BE/trailing after TP1).
                    if not hit_outcome:
                        if side == 'LONG' and price <= effective_stop:
                            if tp1_touched and trailing_enabled and trail_stop is not None and trail_stop > be_price:
                                hit_outcome = 'TRAIL_HIT'
                                price = trail_stop
                            elif tp1_touched and be_armed:
                                hit_outcome = 'BE_HIT'
                                price = be_price
                            else:
                                hit_outcome = 'SL_HIT'
                                price = sl
                            raw_pnl = (price - entry) / entry * 100
                            pnl_pct = (_tp1_locked + raw_pnl * _remainder) if _partial_done else raw_pnl
                            
                        elif side == 'SHORT' and price >= effective_stop:
                            if tp1_touched and trailing_enabled and trail_stop is not None and trail_stop < be_price:
                                hit_outcome = 'TRAIL_HIT'
                                price = trail_stop
                            elif tp1_touched and be_armed:
                                hit_outcome = 'BE_HIT'
                                price = be_price
                            else:
                                hit_outcome = 'SL_HIT'
                                price = sl
                            raw_pnl = (entry - price) / entry * 100
                            pnl_pct = (_tp1_locked + raw_pnl * _remainder) if _partial_done else raw_pnl            
            
            if hit_outcome:"""

if old_str in text:
    text = text.replace(old_str, new_str)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(text)
    print("Replaced perfectly.")
else:
    print("Could not find string")
