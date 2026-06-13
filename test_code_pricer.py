"""Functional tests for code-first pricing + LLM veto + news blackout.

Run: python test_code_pricer.py
"""
import types
from datetime import datetime, timedelta, timezone

passed = []
failed = []


def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(("  OK  " if cond else " FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))


import bot.pending_order_planner as pl


# ── 1. build_candidate_order: bullish pullback ──────────────────────────
# current=100, bullish bias, fresh bullish FVG at 97-98 (discount), BSL at 102/105
mtf_bull = {
    "symbol": "TEST",
    "current_price": 100.0,
    "spread": 0.1,
    "daily": {"trend": "up"},
    "h4": {
        "trend": "up", "atr": 1.0,
        "bsl_levels": [102.0, 105.0], "ssl_levels": [],
        "fvgs": [{"direction": "bullish", "top": 98.0, "bottom": 97.0,
                  "midpoint": 97.5, "mitigated": False}],
        "order_blocks": [],
        "dealing_range": {"location": "discount"},
    },
    "h1": {}, "m15": {},
}
c = pl.build_candidate_order(mtf_bull, min_rr=1.5)
check("bullish candidate produced", c is not None and c["order_type"] == "BUY_LIMIT", str(c))
if c:
    # entry=98 (top), SL=97-0.25*1=96.75, risk=1.25, TP1=102 (RR=(102-98)/1.25=3.2)
    check("bullish entry/SL correct", abs(c["entry_price"] - 98.0) < 1e-6 and abs(c["stop_loss"] - 96.75) < 1e-6, str(c))
    check("bullish entry below current price", c["entry_price"] < 100.0)
    check("bullish SL below entry", c["stop_loss"] < c["entry_price"])
    check("bullish TP1 above entry & RR>=1.5", c["take_profit_1"] > c["entry_price"] and c["risk_to_reward"] >= 1.5, str(c))
    check("bullish TP1 = nearest qualifying BSL (102)", abs(c["take_profit_1"] - 102.0) < 1e-6, str(c))


# ── 2. build_candidate_order: bearish pullback ──────────────────────────
mtf_bear = {
    "symbol": "TEST",
    "current_price": 100.0,
    "spread": 0.1,
    "daily": {"trend": "down"},
    "h4": {
        "trend": "down", "atr": 1.0,
        "bsl_levels": [], "ssl_levels": [98.0, 95.0],
        "fvgs": [{"direction": "bearish", "top": 103.0, "bottom": 102.0,
                  "midpoint": 102.5, "mitigated": False}],
        "order_blocks": [],
        "dealing_range": {"location": "premium"},
    },
    "h1": {}, "m15": {},
}
c2 = pl.build_candidate_order(mtf_bear, min_rr=1.5)
check("bearish candidate produced", c2 is not None and c2["order_type"] == "SELL_LIMIT", str(c2))
if c2:
    # entry=102 (bottom), SL=103+0.25=103.25, risk=1.25, TP1=98 RR=(102-98)/1.25=3.2
    check("bearish entry above current price", c2["entry_price"] > 100.0)
    check("bearish SL above entry", c2["stop_loss"] > c2["entry_price"])
    check("bearish TP1 below entry & RR>=1.5", c2["take_profit_1"] < c2["entry_price"] and c2["risk_to_reward"] >= 1.5, str(c2))


# ── 3. No candidate cases ───────────────────────────────────────────────
check("neutral bias -> no candidate",
      pl.build_candidate_order({"current_price": 100, "daily": {}, "h4": {}, "h1": {}, "m15": {}}) is None)

mtf_no_target = dict(mtf_bull)
# entry=98, SL=96.75, risk=1.25 → BSL 99.0 gives RR=0.8 < 1.5 (too close)
mtf_no_target["h4"] = dict(mtf_bull["h4"], bsl_levels=[99.0])
check("bullish but no RR-qualifying target -> None",
      pl.build_candidate_order(mtf_no_target, min_rr=1.5) is None)

mtf_poi_above = dict(mtf_bull)
mtf_poi_above["h4"] = dict(mtf_bull["h4"],
                           fvgs=[{"direction": "bullish", "top": 101.0, "bottom": 100.5,
                                  "midpoint": 100.75, "mitigated": False}])
check("bullish POI above price (not a pullback) -> None",
      pl.build_candidate_order(mtf_poi_above, min_rr=1.5) is None)


# ── 3b. Tiny POI must not create a wick-out stop (min_stop_atr_mult floor) ──
# Microscopic FVG (98.0-97.95) on a 1.0-ATR instrument: natural stop ~0.05 is
# way under 0.5*ATR=0.5, so the stop must be floored to entry-0.5.
mtf_tiny = dict(mtf_bull)
mtf_tiny["h4"] = dict(mtf_bull["h4"], bsl_levels=[105.0],
                      fvgs=[{"direction": "bullish", "top": 98.0, "bottom": 97.95,
                             "midpoint": 97.975, "mitigated": False}])
ct = pl.build_candidate_order(mtf_tiny, min_rr=1.5, min_stop_atr_mult=0.5)
check("tiny POI -> stop floored to >=0.5*ATR",
      ct is not None and abs((ct["entry_price"] - ct["stop_loss"]) - 0.5) < 1e-6, str(ct))
check("tiny POI -> RR no longer absurd (<=10)", ct is not None and ct["risk_to_reward"] <= 10.0, str(ct))

# ── 3c. Far target with degenerate RR is capped at max_rr ──
# entry=98, SL=97 (1.0 ATR floor makes risk=1.0? no: poi 97-98 gives risk=1.0+buffer)
mtf_fartp = dict(mtf_bull)
mtf_fartp["h4"] = dict(mtf_bull["h4"], bsl_levels=[500.0],  # absurdly far
                       fvgs=[{"direction": "bullish", "top": 98.0, "bottom": 97.0,
                              "midpoint": 97.5, "mitigated": False}])
cf = pl.build_candidate_order(mtf_fartp, min_rr=1.5, max_rr=10.0)
check("far target -> RR capped at max_rr", cf is not None and cf["risk_to_reward"] <= 10.0 + 1e-6, str(cf))


# ── 4. Full plan_pending_order veto path ────────────────────────────────
def make_fake_post(response_text):
    def fake_post(url, json=None, timeout=None):
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"response": response_text}
        return R()
    return fake_post


import requests as real_requests
orig_post = real_requests.post
orig_classify = pl.classify_symbol_fast
pl.classify_symbol_fast = lambda mtf: ("LLM_CANDIDATE", "test bypass")

try:
    # 4a. APPROVE -> produces a placed order with CODE prices (not LLM prices)
    real_requests.post = make_fake_post(
        '{"decision":"APPROVE","score":82,"setup_quality":"HIGH","reason":"clean discount FVG"}'
    )
    d = pl.plan_pending_order(symbol="TEST", mtf_data=mtf_bull, min_rr=1.5, min_score=55)
    check("APPROVE -> actionable BUY_LIMIT", d.decision == "PLACE_BUY_LIMIT", f"{d.decision} {d.reason}")
    check("APPROVE uses code entry price (98.0)", abs(d.entry_price - 98.0) < 1e-6, str(d.entry_price))
    check("APPROVE carries score from veto", d.score == 82, str(d.score))

    # 4b. REJECT -> NO_TRADE
    real_requests.post = make_fake_post(
        '{"decision":"REJECT","score":30,"setup_quality":"WEAK","reason":"conflicting H1"}'
    )
    d2 = pl.plan_pending_order(symbol="TEST", mtf_data=mtf_bull, min_rr=1.5, min_score=55)
    check("REJECT -> NO_TRADE", d2.decision == "NO_TRADE" and "veto" in d2.reason.lower(), f"{d2.decision} {d2.reason}")

    # 4c. APPROVE but code can't price (no candidate) -> never reaches LLM, NO_TRADE
    real_requests.post = make_fake_post('{"decision":"APPROVE","score":90}')
    d3 = pl.plan_pending_order(symbol="TEST", mtf_data=mtf_no_target, min_rr=1.5)
    check("no code candidate -> NO_TRADE before LLM", d3.decision == "NO_TRADE", f"{d3.decision} {d3.reason}")
finally:
    real_requests.post = orig_post
    pl.classify_symbol_fast = orig_classify


# ── 5. News blackout window ─────────────────────────────────────────────
import main

cfg = types.SimpleNamespace(pending_order_planner={
    "news_guard_enabled": True,
    "news_guard_minutes_before": 45,
    "news_guard_minutes_after": 30,
})

now = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)

import bot.calendar_events as cal
orig_cpi = cal.get_next_cpi_release
orig_nfp = cal.analyze_nfp
orig_fomc = cal.get_next_fomc_meeting
try:
    # CPI in 20 min -> inside 45-min pre-window -> blocked
    cal.get_next_cpi_release = lambda dt=None: types.SimpleNamespace(release_datetime=now + timedelta(minutes=20))
    cal.analyze_nfp = lambda high_vol_days=1, dt=None: types.SimpleNamespace(next_release=None)
    cal.get_next_fomc_meeting = lambda dt=None: None
    r = main._pending_news_blackout(cfg, now=now)
    check("CPI in 20min -> blackout active", r is not None and "CPI" in r, str(r))

    # CPI in 3 hours -> outside window -> allowed
    cal.get_next_cpi_release = lambda dt=None: types.SimpleNamespace(release_datetime=now + timedelta(hours=3))
    r2 = main._pending_news_blackout(cfg, now=now)
    check("CPI in 3h -> no blackout", r2 is None, str(r2))

    # Disabled guard -> never blocks
    cfg_off = types.SimpleNamespace(pending_order_planner={"news_guard_enabled": False})
    cal.get_next_cpi_release = lambda dt=None: types.SimpleNamespace(release_datetime=now + timedelta(minutes=5))
    check("guard disabled -> no blackout", main._pending_news_blackout(cfg_off, now=now) is None)
finally:
    cal.get_next_cpi_release = orig_cpi
    cal.analyze_nfp = orig_nfp
    cal.get_next_fomc_meeting = orig_fomc


print()
print(f"PASSED: {len(passed)}  FAILED: {len(failed)}")
if failed:
    raise SystemExit(1)
