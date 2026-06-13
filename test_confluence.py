"""Tests for the SMC confluence gate in build_candidate_order.

Each gate (sweep, displacement, structure break, fresh POI, premium/discount)
must be required: removing any one must turn a valid setup into None.
Run: python test_confluence.py
"""
import copy
import bot.pending_order_planner as pl

passed, failed = [], []


def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(("  OK  " if cond else " FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))


# ── A full, valid BULLISH confluence setup on m15 ───────────────────────
# bias bullish (daily/h4 up), discount location, m15 has: bullish sweep of SSL
# at idx 10 (price 95), bullish displacement after, bullish CHoCH after, and a
# fresh bullish FVG (idx 15) at 97-98 sitting between the sweep (95) and price (100).
def base_mtf():
    return {
        "symbol": "T", "current_price": 100.0, "spread": 0.1,
        "daily": {"trend": "up"},
        "h4": {"trend": "up", "atr": 1.0,
               "bsl_levels": [104.0, 106.0], "ssl_levels": [],
               "dealing_range": {"location": "discount", "equilibrium": 102.0},
               "fvgs": [], "order_blocks": [], "structure_breaks": [],
               "displacement": None, "liquidity_sweeps": []},
        "h1": {"dealing_range": {"location": "discount", "equilibrium": 102.0}},
        "m15": {
            "atr": 1.0,
            "liquidity_sweeps": [{"direction": "bullish", "swept_price": 95.0, "index": 10, "killzone": "London"}],
            "displacement": {"direction": "bullish", "atr_multiple": 2.0, "body_ratio": 0.8,
                             "start_index": 11, "end_index": 12},
            "structure_breaks": [{"break_type": "CHoCH", "direction": "bullish", "price": 99.0,
                                  "body_close": True, "index": 13}],
            "fvgs": [{"direction": "bullish", "top": 98.0, "bottom": 97.0, "midpoint": 97.5,
                      "mitigated": False, "index": 15}],
            "order_blocks": [],
        },
        "m5": {},
    }


c = pl.build_candidate_order(base_mtf(), min_rr=1.5)
check("full bullish confluence -> candidate", c is not None and c["order_type"] == "BUY_LIMIT", str(c))
if c:
    check("entry at POI top (98)", abs(c["entry_price"] - 98.0) < 1e-6, str(c))
    check("stop behind swept low (<95)", c["stop_loss"] < 95.0, str(c))  # real invalidation
    check("TP at HTF BSL, RR>=1.5", c["take_profit_1"] >= 104.0 and c["risk_to_reward"] >= 1.5, str(c))
    check("structure_break recorded", "CHoCH" in c.get("structure_break", ""), str(c.get("structure_break")))


# ── Each gate is required: remove one element -> None ───────────────────
m = base_mtf(); m["m15"]["liquidity_sweeps"] = []
check("no sweep -> None", pl.build_candidate_order(m, min_rr=1.5) is None)

m = base_mtf(); m["m15"]["displacement"] = None
check("no displacement -> None", pl.build_candidate_order(m, min_rr=1.5) is None)

m = base_mtf(); m["m15"]["displacement"]["atr_multiple"] = 0.9  # below 1.5
check("weak displacement (atr<1.5) -> None", pl.build_candidate_order(m, min_rr=1.5) is None)

m = base_mtf(); m["m15"]["displacement"]["body_ratio"] = 0.4  # below 0.65
check("wick-dominated displacement -> None", pl.build_candidate_order(m, min_rr=1.5) is None)

m = base_mtf(); m["m15"]["structure_breaks"] = []
check("no structure break -> None", pl.build_candidate_order(m, min_rr=1.5) is None)

m = base_mtf(); m["m15"]["structure_breaks"][0]["direction"] = "bearish"  # wrong direction
check("opposite-direction structure break -> None", pl.build_candidate_order(m, min_rr=1.5) is None)

m = base_mtf(); m["m15"]["fvgs"] = []
check("no fresh POI -> None", pl.build_candidate_order(m, min_rr=1.5) is None)

m = base_mtf(); m["m15"]["fvgs"][0]["mitigated"] = True
check("mitigated POI -> None", pl.build_candidate_order(m, min_rr=1.5) is None)

m = base_mtf(); m["m15"]["fvgs"][0]["index"] = 5  # before the sweep (idx 10)
check("POI before sweep (not from the move) -> None", pl.build_candidate_order(m, min_rr=1.5) is None)


# ── Temporal ordering: displacement before sweep -> None ────────────────
m = base_mtf(); m["m15"]["displacement"]["end_index"] = 8  # before sweep idx 10
check("displacement before sweep -> None", pl.build_candidate_order(m, min_rr=1.5) is None)

m = base_mtf(); m["m15"]["structure_breaks"][0]["index"] = 8  # before sweep idx 10
check("structure break before sweep -> None", pl.build_candidate_order(m, min_rr=1.5) is None)


# ── Premium/discount gate: bullish ENTRY in premium -> None ─────────────
# Move equilibrium below the entry (98) so the entry sits in premium.
m = base_mtf()
m["h4"]["dealing_range"]["equilibrium"] = 96.0
m["h1"]["dealing_range"]["equilibrium"] = 96.0
check("bullish entry in premium -> None (enforced)", pl.build_candidate_order(m, min_rr=1.5) is None)
check("bullish entry in premium allowed when enforce off",
      pl.build_candidate_order(m, min_rr=1.5, enforce_premium_discount=False) is not None)


# ── BEARISH mirror: full valid short setup ──────────────────────────────
def base_mtf_short():
    return {
        "symbol": "T", "current_price": 100.0, "spread": 0.1,
        "daily": {"trend": "down"},
        "h4": {"trend": "down", "atr": 1.0,
               "bsl_levels": [], "ssl_levels": [96.0, 94.0],
               "dealing_range": {"location": "premium", "equilibrium": 98.0},
               "fvgs": [], "order_blocks": [], "structure_breaks": [],
               "displacement": None, "liquidity_sweeps": []},
        "h1": {"dealing_range": {"location": "premium", "equilibrium": 98.0}},
        "m15": {
            "atr": 1.0,
            "liquidity_sweeps": [{"direction": "bearish", "swept_price": 105.0, "index": 10, "killzone": "NewYork"}],
            "displacement": {"direction": "bearish", "atr_multiple": 2.0, "body_ratio": 0.8,
                             "start_index": 11, "end_index": 12},
            "structure_breaks": [{"break_type": "CHoCH", "direction": "bearish", "price": 101.0,
                                  "body_close": True, "index": 13}],
            "fvgs": [{"direction": "bearish", "top": 103.0, "bottom": 102.0, "midpoint": 102.5,
                      "mitigated": False, "index": 15}],
            "order_blocks": [],
        },
        "m5": {},
    }


cs = pl.build_candidate_order(base_mtf_short(), min_rr=1.5)
check("full bearish confluence -> SELL_LIMIT", cs is not None and cs["order_type"] == "SELL_LIMIT", str(cs))
if cs:
    check("short entry at POI bottom (102)", abs(cs["entry_price"] - 102.0) < 1e-6, str(cs))
    check("short stop behind swept high (>105)", cs["stop_loss"] > 105.0, str(cs))


# ── Displacements list: latest move is counter-bias, impulse is earlier ─
# A bullish setup where the most recent displacement is the bearish pullback,
# but an earlier bullish impulse exists after the sweep -> must still qualify.
m = base_mtf()
m["m15"].pop("displacement", None)
m["m15"]["displacements"] = [
    {"direction": "bullish", "atr_multiple": 2.0, "body_ratio": 0.8, "start_index": 11, "end_index": 12},
    {"direction": "bearish", "atr_multiple": 1.8, "body_ratio": 0.7, "start_index": 16, "end_index": 17},
]
c_imp = pl.build_candidate_order(m, min_rr=1.5)
check("bias-impulse found despite later counter-bias displacement",
      c_imp is not None and c_imp["order_type"] == "BUY_LIMIT", str(c_imp))

# Only a counter-bias displacement present -> no setup.
m = base_mtf()
m["m15"].pop("displacement", None)
m["m15"]["displacements"] = [
    {"direction": "bearish", "atr_multiple": 2.0, "body_ratio": 0.8, "start_index": 11, "end_index": 12},
]
check("only counter-bias displacement -> None",
      pl.build_candidate_order(m, min_rr=1.5) is None)


# ── Fallback to m5 when m15 has no setup ────────────────────────────────
m = base_mtf()
m["m5"] = copy.deepcopy(m["m15"])
m["m15"] = {"atr": 1.0}  # empty m15
c5 = pl.build_candidate_order(m, min_rr=1.5)
check("falls back to m5 setup", c5 is not None and c5["poi_timeframe"] == "m5", str(c5))


print()
print(f"PASSED: {len(passed)}  FAILED: {len(failed)}")
if failed:
    raise SystemExit(1)
