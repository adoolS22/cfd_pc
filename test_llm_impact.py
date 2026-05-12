import sqlite3
import pandas as pd

def test_learning():
    db = sqlite3.connect('signals.db')
    
    query = """
    SELECT o.id, o.pnl_pct, o.outcome, r.action 
    FROM signal_outcomes o 
    LEFT JOIN llm_trade_reviews r ON o.id=r.outcome_id 
    WHERE o.pnl_pct IS NOT NULL
    """
    df = pd.read_sql(query, db)
    
    # 1. Without learning
    total_trades = len(df)
    total_wins = len(df[df.pnl_pct > 0])
    total_pnl = df.pnl_pct.sum()
    winrate_raw = (total_wins / total_trades) * 100 if total_trades else 0
    
    print("=" * 60)
    print(" 📊 أداء النظام الأصلي (بدون فلاتر التعلم)")
    print("=" * 60)
    print(f"الصفقات      : {total_trades}")
    print(f"نسبة النجاح  : {winrate_raw:.2f}%")
    print(f"إجمالي الربح : {total_pnl:.2f}%")
    print("-" * 60)
    
    # 2. Block 'hard_penalty'
    passed_hard = df[~df.action.isin(['hard_penalty'])]
    t2 = len(passed_hard)
    w2 = len(passed_hard[passed_hard.pnl_pct > 0])
    pnl2 = passed_hard.pnl_pct.sum()
    wr2 = (w2 / t2) * 100 if t2 else 0
    
    print(" 🛡️ الأداء مع استبعاد الأخطاء الجسيمة (hard_penalty) التي تعلمها")
    print("-" * 60)
    print(f"الصفقات المتبقية : {t2} (تم تفادي {total_trades - t2} صفقة سيئة)")
    print(f"نسبة النجاح الجديدة: {wr2:.2f}%")
    print(f"إجمالي الربح الجديد: {pnl2:.2f}% (الفرق: {pnl2 - total_pnl:+.2f}%)")
    print("-" * 60)
    
    # 3. Strict mode (only 'keep')
    passed_strict = df[df.action == 'keep']
    t3 = len(passed_strict)
    w3 = len(passed_strict[passed_strict.pnl_pct > 0])
    pnl3 = passed_strict.pnl_pct.sum()
    wr3 = (w3 / t3) * 100 if t3 else 0
    
    print(" 🏆 الأداء الصارم جداً (فقط الصفقات الموصى بإبقائها - keep)")
    print("-" * 60)
    print(f"الصفقات المتبقية : {t3} (تم تفادي {total_trades - t3} صفقة)")
    print(f"نسبة النجاح      : {wr3:.2f}%")
    print(f"إجمالي الربح     : {pnl3:.2f}%")
    print("=" * 60)

if __name__ == '__main__':
    test_learning()
