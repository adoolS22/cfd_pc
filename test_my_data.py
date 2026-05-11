#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test My Data (Dynamic Config Analysis)
=====================================
يقرأ إعداداتك الحالية من config.yaml (الحد الأدنى للجودة، R:R، وغيرها)
يطابقها على صفقاتك السابقة في قاعدة البيانات ويخبرك كيف سيكون الأداء!
"""

import sqlite3
import yaml
import os

DB_PATH   = "signals.db"
CONFIG_PATH = "config.yaml"
COST_PCT  = 0.08   # رسوم المنصة التقريبية لكل صفقة (ذهاب وعودة)

WIN_OUTCOMES  = {"TP_NEAR_HIT", "TP1_HIT", "TP2_HIT", "TRAIL_HIT", "BE_HIT"}
MACRO_KEYS = ("XAU", "XAG", "OIL", "WTI", "BRENT", "SNP500", "SPX500", "EURUSD", "US30", "US500", "USTEC")

def is_macro(sym: str) -> bool:
    return any(k in sym.upper() for k in MACRO_KEYS)

def stats(trades):
    if not trades:
        return 0, 0, 0, 0.0, 0.0, 0.0, 0.0
    wins   = sum(1 for t in trades if t["outcome"] in WIN_OUTCOMES and (t["pnl"] or 0) > 0)
    losses = sum(1 for t in trades if t["outcome"] == "SL_HIT" or (t["pnl"] or 0) <= 0)
    if not wins + losses: losses = 1 # Avoid division by zero occasionally
    wr     = wins / len(trades) * 100
    avg_gross = sum(t["pnl"] or 0 for t in trades) / len(trades)
    avg_net   = avg_gross - COST_PCT
    total_net = avg_net * len(trades)
    return len(trades), wins, losses, wr, avg_gross, avg_net, total_net

def print_stats(label: str, trades):
    n, w, l, wr, avg_g, avg_n, tot_n = stats(trades)
    if n == 0:
        print(f"  {label:35s}  لا توجد أي بيانات سابقة")
        return
    flag = "✅" if avg_n > 0 else ("⚠️ " if avg_n > -0.02 else "❌")
    print(
        f" {flag} {label:35s}  "
        f"الصفقات={n:<5,}  نسبة النجاح={wr:5.1f}%  "
        f"متوسط ربح الصفقة={avg_n:+.3f}%  "
        f"صافي الأرباح={tot_n:+6.1f}%"
    )

def run():
    print("=" * 80)
    print("  🚀 نظام الفحص الذكي (Test My Data)")
    print("  يقرأ إعداداتك الآن من config.yaml ويطبقها على الداتا لمعرفة الأداء")
    print("=" * 80)

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"حدث خطأ أثناء قراءة إعدادات config.yaml: {e}")
        return

    min_rr_tp1 = cfg.get("risk", {}).get("min_rr_tp1", 0.0)
    max_sl_crypto = cfg.get("risk", {}).get("max_sl_pct_crypto", 99.0)
    max_sl_macro = cfg.get("risk", {}).get("max_sl_pct_macro", 99.0)
    
    scoring = cfg.get("scoring", {})
    base_thresh = scoring.get("base_threshold", 0)
    base_thresh_crypto = scoring.get("base_threshold_crypto", base_thresh)
    base_thresh_macro = scoring.get("base_threshold_macro", base_thresh)

    print(f"\n[🎯] إعداداتك المكتوبة حالياً في ملف الكونفيج:")
    print(f"  - الجودة المطلوبة للعملات الرقمية : {base_thresh_crypto}")
    print(f"  - الجودة المطلوبة لأسواق الماكرو   : {base_thresh_macro}")
    print(f"  - الحد الأدنى للهدف مقابل الستوب (R:R) : {min_rr_tp1}")
    print("-" * 80)

    if not os.path.exists(DB_PATH):
        print(f"لم يتم العثور على قاعدة البيانات {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    try:
        rows = conn.execute("""
            SELECT
                so.symbol, so.outcome, COALESCE(so.pnl_pct, 0.0) AS pnl,
                so.entry, so.stop_loss, so.take_profit_1, s.score
            FROM signal_outcomes so
            LEFT JOIN signals s ON s.id = so.signal_id
            WHERE so.outcome NOT IN ('OPEN','EXITED') AND so.pnl_pct IS NOT NULL
        """).fetchall()
    except Exception as e:
        print(f"Error reading DB: {e}")
        return

    all_trades = []
    filtered_trades = []
    blocked_trades = []

    for r in rows:
        sym = str(r["symbol"]).upper()
        pnl = float(r["pnl"])
        score = float(r["score"] or 0)
        entry = float(r["entry"] or 0)
        sl = float(r["stop_loss"] or 0)
        tp1 = float(r["take_profit_1"] or 0)
        
        is_m = is_macro(sym)
        rr = 0.0
        sl_pct = 0.0
        if entry > 0 and sl > 0 and tp1 > 0:
            risk = abs(entry - sl)
            if risk > 0: rr = abs(entry - tp1) / risk
            sl_pct = (risk / entry) * 100.0

        trade = {"pnl": pnl, "outcome": r["outcome"], "is_macro": is_m}
        all_trades.append(trade)

        # Apply Live Settings Filtering
        if rr > 0 and rr < min_rr_tp1:
            blocked_trades.append(trade)
            continue
            
        if sl_pct > 0 and sl_pct > (max_sl_macro if is_m else max_sl_crypto):
            blocked_trades.append(trade)
            continue
            
        if score > 0 and (score < base_thresh_macro if is_m else score < base_thresh_crypto):
            blocked_trades.append(trade)
            continue
            
        filtered_trades.append(trade)

    if not all_trades:
        print("لا توجد داتا فعلية حتى الآن!")
        return

    print("\n[📊] التقييم الشامل (لو تركت البوت يفتح كل الصفقات العشوائية):")
    print_stats("السوق بالكامل (بدون فلاتر)", all_trades)
    
    print("\n[✨] النتيجة الذهبية (الصفقات النخبة المُجتازة للشروط في config.yaml):")
    if filtered_trades:
        print_stats("الصفقات التي اختارها إعدادك", filtered_trades)
        print_stats(" > كريبتو فقط مفلتر", [t for t in filtered_trades if not t["is_macro"]])
        print_stats(" > ماكرو (ذهب/مؤشرات/فوركس)", [t for t in filtered_trades if t["is_macro"]])
    else:
        print("  ❌ شروطك قاسية جداً لدرجة أنها لم توافق على أي صفقة سابقة!")

    print("\n[🗑️] الصفقات الخطرة (سيمنعها البوت ويحمي رصيدك من خسارتها):")
    print_stats("المرفوض بسبب ضعف الجودة/RR", blocked_trades)
    print("\n💡 استنتاج: إذا رأيت علامة صح خضراء بدلاً من إشارة خطر في الحقل المرفوض، فهذا يعني أنك تفاديت خسائر!")
    print("=" * 80)

if __name__ == "__main__":
    run()