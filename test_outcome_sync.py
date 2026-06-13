"""Functional test for the real-trade outcome sync + learning context feed.

Uses a temp SQLite DB and a mocked MT5 module. Run: python test_outcome_sync.py
"""
import os
import tempfile
import types
import time

passed = []
failed = []


def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(("  OK  " if cond else " FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))


# ── 1. Storage: record + stats + lessons ────────────────────────────────
from bot.storage import SignalStorage

db_path = os.path.join(tempfile.mkdtemp(), "test_signals.db")
storage = SignalStorage(db_path=db_path)

order_id = storage.save_pending_order({
    "symbol": "XAUUSD", "order_type": "SELL_LIMIT",
    "entry_price": 2400.0, "stop_loss": 2410.0,
    "take_profit_1": 2380.0, "take_profit_2": 2360.0,
    "lot": 0.02, "mt5_ticket": 111222, "status": "PENDING",
    "score": 85, "setup_quality": "HIGH", "reason": "FVG retest",
    "valid_until_hours": 8, "llm_analysis": {"x": 1},
})
unresolved = storage.get_unresolved_pending_orders()
check("unresolved pending order visible", len(unresolved) == 1 and unresolved[0]["mt5_ticket"] == 111222)

outcome_id = storage.record_closed_real_trade(
    symbol="XAUUSD", side="SHORT", entry=2400.0, stop_loss=2410.0,
    take_profit_1=2380.0, take_profit_2=2360.0, score=85,
    reasons=["planner_real_trade", "setup_quality=HIGH"],
    outcome="TP1_HIT", close_price=2380.0, pnl_pct=0.8333,
    closed_reason="mt5_tp",
    opened_at="2026-06-12T10:00:00+00:00", closed_at="2026-06-12T14:00:00+00:00",
)
check("record_closed_real_trade returns outcome_id", outcome_id > 0)

ctx = storage.get_closed_outcome_context(outcome_id)
check("postmortem context readable for real trade", bool(ctx) and ctx.get("symbol") == "XAUUSD")

storage.record_closed_real_trade(
    symbol="XAUUSD", side="SHORT", entry=2390.0, stop_loss=2398.0,
    take_profit_1=2370.0, take_profit_2=0, score=70,
    reasons=["planner_real_trade"],
    outcome="SL_HIT", close_price=2398.0, pnl_pct=-0.3347,
    closed_reason="mt5_sl",
    opened_at="2026-06-12T16:00:00+00:00", closed_at="2026-06-12T18:00:00+00:00",
)
stats = storage.get_real_trade_stats("XAUUSD", lookback_days=21)
check("real trade stats: SHORT 1W/1L", stats.get("SHORT", {}).get("trades") == 2 and stats.get("SHORT", {}).get("wins") == 1,
      str(stats))

storage.save_llm_trade_review(
    outcome_id=outcome_id + 1, signal_id=None, symbol="XAUUSD", side="SHORT",
    outcome="SL_HIT", pnl_pct=-0.33, verdict="timing_error", action="soft_penalty",
    confidence=70, penalty=0.3, mistake_tags=["late_entry"],
    summary="Entered after move exhausted", recommendation="Wait for displacement",
    raw_json="{}",
)
lessons = storage.get_recent_trade_lessons("XAUUSD", lookback_days=30, limit=4)
check("lessons query returns review", len(lessons) == 1 and lessons[0]["verdict"] == "timing_error")

storage.update_pending_order_status(111222, "CLOSED")
check("status CLOSED removes from unresolved", len(storage.get_unresolved_pending_orders()) == 0)

# ── 2. MT5 deal-history aggregation (mocked mt5) ────────────────────────
import bot.mt5_client as mc


class Deal:
    def __init__(self, position_id, order, symbol, entry, dtype, price, volume, t, profit=0.0, reason=0, magic=202604):
        self.position_id = position_id
        self.order = order
        self.symbol = symbol
        self.entry = entry          # 0=IN, 1=OUT
        self.type = dtype           # 0=BUY, 1=SELL
        self.price = price
        self.volume = volume
        self.time = t
        self.profit = profit
        self.swap = 0.0
        self.commission = -0.1
        self.reason = reason
        self.magic = magic


now = int(time.time())
deals = [
    # Position 555: SHORT filled at 2400, closed at TP 2380 (reason 5 = TP)
    Deal(555, 555, "XAUUSDm", 0, 1, 2400.0, 0.02, now - 7200),
    Deal(555, 600, "XAUUSDm", 1, 0, 2380.0, 0.02, now - 3600, profit=40.0, reason=5),
    # Position 777: still open (IN only) — must NOT appear
    Deal(777, 777, "EURUSDm", 0, 0, 1.1000, 0.02, now - 1800),
    # Position 888: foreign magic — must NOT appear
    Deal(888, 888, "US500m", 0, 0, 7000.0, 0.02, now - 1800, magic=999),
]

mc.mt5 = types.SimpleNamespace(
    history_deals_get=lambda a, b: deals,
    ORDER_TYPE_BUY=0,
    DEAL_REASON_SL=4, DEAL_REASON_TP=5,
    DEAL_ENTRY_IN=0, DEAL_ENTRY_OUT=1, DEAL_ENTRY_OUT_BY=3,
)

client = mc.MT5Client.__new__(mc.MT5Client)
client._lock = __import__("threading").RLock()
client._require_package = lambda: None
client.connect_mt5 = lambda: True

closed = client.get_closed_positions_history(lookback_days=3)
check("only fully closed bot positions returned", len(closed) == 1 and closed[0]["position_id"] == 555, str(closed))
p = closed[0]
check("aggregation: side/prices/reason correct",
      p["side"] == "SHORT" and p["entry_price"] == 2400.0 and p["close_price"] == 2380.0
      and p["close_reason"] == "TP" and abs(p["profit"] - 39.8) < 1e-6, str(p))

# ── 3. Planner prompt: learning context injected ────────────────────────
import bot.pending_order_planner as pl

captured = {}


def fake_post(url, json=None, timeout=None):
    captured["prompt"] = json["prompt"]

    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "not json on purpose"}
    return R()


import requests as real_requests
orig_post = real_requests.post
real_requests.post = fake_post
try:
    # Confluence-complete bearish setup so the code pricer produces a candidate
    # and the LLM (and thus the learning context) is actually reached.
    mtf = {
        "symbol": "XAUUSD", "current_price": 2400.0, "spread": 0.3,
        "daily": {"trend": "down"},
        "h4": {"trend": "down", "atr": 5.0, "bsl_levels": [], "ssl_levels": [2360.0, 2350.0],
               "dealing_range": {"location": "premium", "equilibrium": 2405.0},
               "fvgs": [], "order_blocks": [], "structure_breaks": [], "displacement": None,
               "liquidity_sweeps": []},
        "h1": {"dealing_range": {"location": "premium", "equilibrium": 2405.0}},
        "m15": {
            "atr": 5.0,
            "liquidity_sweeps": [{"direction": "bearish", "swept_price": 2420.0, "index": 10, "killzone": "NewYork"}],
            "displacement": {"direction": "bearish", "atr_multiple": 2.0, "body_ratio": 0.8,
                             "start_index": 11, "end_index": 12},
            "structure_breaks": [{"break_type": "CHoCH", "direction": "bearish", "price": 2405.0,
                                  "body_close": True, "index": 13}],
            "fvgs": [{"direction": "bearish", "top": 2412.0, "bottom": 2410.0, "midpoint": 2411.0,
                      "mitigated": False, "index": 15}],
            "order_blocks": [],
        },
        "m5": {},
    }
    decision = pl.plan_pending_order(
        symbol="XAUUSD",
        mtf_data=mtf,
        learning_context="Real closed trades on this symbol (last 21 days): SHORT: 1W/1L winrate=50% avgPnL=+0.25%",
    )
finally:
    real_requests.post = orig_post

check("learning context appears in LLM prompt",
      "PAST PERFORMANCE & LESSONS" in captured.get("prompt", "")
      and "SHORT: 1W/1L" in captured.get("prompt", ""))
check("non-JSON still degrades to NO_TRADE", decision.decision == "NO_TRADE")

print()
print(f"PASSED: {len(passed)}  FAILED: {len(failed)}")
if failed:
    raise SystemExit(1)
