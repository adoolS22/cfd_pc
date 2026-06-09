import pytest
from bot.pending_order_planner import classify_symbol_fast, summarize_mtf_data, score_candidate_fast

def build_mock_mtf(bias, pd_loc, draw, disp, event_type, pois):
    # bias: 'bullish', 'bearish'
    # pd_loc: 'premium', 'discount', 'equilibrium'
    # draw: 'buy_side', 'sell_side', None
    return {
        "symbol": "BTC/USDT",
        "current_price": 50000,
        "daily": {"trend": bias},
        "h4": {
            "trend": bias,
            "dealing_range": {"location": pd_loc},
            "bsl_levels": [51000] if draw == "buy_side" or draw == "both" else [],
            "ssl_levels": [49000] if draw == "sell_side" or draw == "both" else [],
        },
        "h1": {
            "displacement": {"direction": disp} if disp else None,
            "structure_breaks": [{"break_type": event_type, "direction": disp}] if event_type else [],
            "fvgs": [p for p in pois if p.get("type") == "FVG"],
            "order_blocks": [p for p in pois if p.get("type") == "OB"],
        }
    }

def test_prefilter_rejects_equilibrium():
    mtf = build_mock_mtf("bullish", "equilibrium", "buy_side", "bullish", "bos", [{"valid": True, "mitigated": False, "type": "FVG"}])
    mtf["h1"]["dealing_range"] = {"location": "equilibrium"}
    state, reason = classify_symbol_fast(mtf)
    assert state == "REJECT"
    assert "equilibrium" in reason.lower()

def test_prefilter_accepts_valid_bullish_fvg_candidate():
    mtf = build_mock_mtf("bullish", "discount", "buy_side", "bullish", "bos", [{"valid": True, "mitigated": False, "type": "FVG"}])
    state, reason = classify_symbol_fast(mtf)
    assert state == "LLM_CANDIDATE"

def test_prefilter_accepts_valid_bearish_fvg_candidate():
    mtf = build_mock_mtf("bearish", "premium", "sell_side", "bearish", "choch", [{"valid": True, "mitigated": False, "type": "OB"}])
    state, reason = classify_symbol_fast(mtf)
    assert state == "LLM_CANDIDATE"

def test_score_candidate_fast_orders_best_first():
    mtf_good = build_mock_mtf("bullish", "discount", "buy_side", "bullish", "bos", [{"valid": True, "mitigated": False, "type": "FVG"}])
    mtf_bad = build_mock_mtf("bullish", "unknown", None, None, None, [])
    assert score_candidate_fast(mtf_good) > score_candidate_fast(mtf_bad)

def test_summarize_mtf_data_removes_raw_candles():
    mtf = build_mock_mtf("bullish", "discount", "buy_side", "bullish", "bos", [])
    mtf["daily"]["candles"] = [{"close": 50000}, {"close": 49000}]
    summary = summarize_mtf_data(mtf)
    assert "candles" not in summary.get("daily", {})
    assert summary["symbol"] == "BTC/USDT"
    assert summary["bias"]["final_bias"] == "bullish"
