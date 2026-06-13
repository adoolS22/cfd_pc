"""Functional tests for the LLM veto path + news blackout.

Pricing and the confluence gate are covered by test_confluence.py. This file
checks: a confluence-valid candidate flows through the veto correctly
(APPROVE -> order with code prices, REJECT -> NO_TRADE), no-candidate skips the
LLM, and the news blackout window logic.
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


# A confluence-complete bullish setup (mirrors test_confluence.py base_mtf).
def valid_mtf():
    return {
        "symbol": "TEST", "current_price": 100.0, "spread": 0.1,
        "daily": {"trend": "up"},
        "h4": {"trend": "up", "atr": 1.0, "bsl_levels": [104.0, 106.0], "ssl_levels": [],
               "dealing_range": {"location": "discount", "equilibrium": 102.0},
               "fvgs": [], "order_blocks": [], "structure_breaks": [], "displacement": None,
               "liquidity_sweeps": []},
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
try:
    # APPROVE -> actionable order using the CODE-priced entry (POI top = 98)
    real_requests.post = make_fake_post(
        '{"decision":"APPROVE","score":82,"setup_quality":"HIGH","reason":"clean discount FVG after sweep"}'
    )
    d = pl.plan_pending_order(symbol="TEST", mtf_data=valid_mtf(), min_rr=1.5, min_score=55)
    check("APPROVE -> PLACE_BUY_LIMIT", d.decision == "PLACE_BUY_LIMIT", f"{d.decision} {d.reason}")
    check("APPROVE uses code entry (98.0)", abs(d.entry_price - 98.0) < 1e-6, str(d.entry_price))
    check("APPROVE carries veto score", d.score == 82, str(d.score))

    # REJECT -> NO_TRADE
    real_requests.post = make_fake_post(
        '{"decision":"REJECT","score":30,"setup_quality":"WEAK","reason":"h1 conflict"}'
    )
    d2 = pl.plan_pending_order(symbol="TEST", mtf_data=valid_mtf(), min_rr=1.5, min_score=55)
    check("REJECT -> NO_TRADE", d2.decision == "NO_TRADE" and "veto" in d2.reason.lower(), f"{d2.decision} {d2.reason}")

    # No confluence -> NO_TRADE before the LLM is ever called
    real_requests.post = make_fake_post('{"decision":"APPROVE","score":90}')
    bare = {"symbol": "T", "current_price": 100.0, "spread": 0.1,
            "daily": {"trend": "up"}, "h4": {"trend": "up", "atr": 1.0,
            "dealing_range": {"location": "discount"}}, "h1": {}, "m15": {}, "m5": {}}
    d3 = pl.plan_pending_order(symbol="T", mtf_data=bare, min_rr=1.5)
    check("no confluence -> NO_TRADE (no LLM)", d3.decision == "NO_TRADE", f"{d3.decision}")
finally:
    real_requests.post = orig_post


# ── News blackout window ────────────────────────────────────────────────
import main

cfg = types.SimpleNamespace(pending_order_planner={
    "news_guard_enabled": True, "news_guard_minutes_before": 45, "news_guard_minutes_after": 30,
})
now = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)

import bot.calendar_events as cal
orig_cpi, orig_nfp, orig_fomc = cal.get_next_cpi_release, cal.analyze_nfp, cal.get_next_fomc_meeting
try:
    cal.get_next_cpi_release = lambda dt=None: types.SimpleNamespace(release_datetime=now + timedelta(minutes=20))
    cal.analyze_nfp = lambda high_vol_days=1, dt=None: types.SimpleNamespace(next_release=None)
    cal.get_next_fomc_meeting = lambda dt=None: None
    check("CPI in 20min -> blackout", (main._pending_news_blackout(cfg, now=now) or "").find("CPI") >= 0)

    cal.get_next_cpi_release = lambda dt=None: types.SimpleNamespace(release_datetime=now + timedelta(hours=3))
    check("CPI in 3h -> no blackout", main._pending_news_blackout(cfg, now=now) is None)

    cfg_off = types.SimpleNamespace(pending_order_planner={"news_guard_enabled": False})
    cal.get_next_cpi_release = lambda dt=None: types.SimpleNamespace(release_datetime=now + timedelta(minutes=5))
    check("guard disabled -> no blackout", main._pending_news_blackout(cfg_off, now=now) is None)
finally:
    cal.get_next_cpi_release, cal.analyze_nfp, cal.get_next_fomc_meeting = orig_cpi, orig_nfp, orig_fomc


print()
print(f"PASSED: {len(passed)}  FAILED: {len(failed)}")
if failed:
    raise SystemExit(1)
