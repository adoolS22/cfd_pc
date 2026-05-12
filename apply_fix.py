import sys

file_path = "y:/trading/trading/crypto_signal_bot/main.py"
with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

old_block = """                    # Stop handling (normal SL before TP1; BE/trailing after TP1).
                    if not hit_outcome:
                        if side == 'LONG' and price <= effective_stop:
                            if tp1_touched and trailing_enabled and trail_stop is not None and trail_stop > be_price:
                                hit_outcome = 'TRAIL_HIT'
                            elif tp1_touched and be_armed:
                                hit_outcome = 'BE_HIT'
                            else:
                                hit_outcome = 'SL_HIT'
                            pnl_pct = (price - entry) / entry * 100
                        elif side == 'SHORT' and price >= effective_stop:
                            if tp1_touched and trailing_enabled and trail_stop is not None and trail_stop < be_price:
                                hit_outcome = 'TRAIL_HIT'
                            elif tp1_touched and be_armed:
                                hit_outcome = 'BE_HIT'
                            else:
                                hit_outcome = 'SL_HIT'
                            pnl_pct = (entry - price) / entry * 100
            
            if hit_outcome:"""

new_block = """                    # Stop handling (normal SL before TP1; BE/trailing after TP1).
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

# Quick fix for Windows newline variance
old_block_normalized = old_block.replace('\r\n', '\n')
new_block_normalized = new_block.replace('\r\n', '\n')
code_normalized = code.replace('\r\n', '\n')

if old_block_normalized in code_normalized:
    new_code = code_normalized.replace(old_block_normalized, new_block_normalized)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_code)
    print("Replaced successfully!")
else:
    print("Could not find exact block. Let's try matching with regex.")
    import re
    # Match the block ignoring exact whitespaces
    pattern = re.compile(r"# Stop handling \(normal SL before TP1.*?(?=if hit_outcome:)", re.DOTALL | re.MULTILINE)
    match = pattern.search(code_normalized)
    if match:
        print("Found with regex, replacing...")
        replaced_code = code_normalized[:match.start()] + new_block_normalized.replace('            if hit_outcome:', '') + code_normalized[match.end():]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(replaced_code)
    else:
        print("Failed regex too.")
