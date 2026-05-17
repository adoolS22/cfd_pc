# Crypto & Macro Trading Signal Bot

بوت إشارات تداول ذكي للعقود الآجلة (Futures) يعمل على الكريبتو والماكرو (ذهب، نفط، فوركس، مؤشرات).
يُولّد إشارات LONG/SHORT/EXIT مع Entry/SL/TP1/TP2 ويرسلها عبر Telegram.

## 🏗️ البنية الحالية

```
crypto_signal_bot/
├── main.py                  # نقطة الدخول الرئيسية + Scan Loop
├── config.yaml              # الإعدادات الرئيسية
├── signals.db               # قاعدة بيانات SQLite
├── bot/
│   ├── exchange.py          # محولات CCXT + MT5 Bridge + MT5 Direct
│   ├── mt5_client.py        # اتصال مباشر بـ MetaTrader 5
│   ├── indicators.py        # SMA, EMA, RSI, ATR, Bollinger, MACD...
│   ├── zones.py             # كشف مناطق الدعم والمقاومة
│   ├── patterns.py          # أنماط الشموع + Fractals
│   ├── gann.py              # زوايا Gann + مربع 9
│   ├── time_cycles.py       # دورة 52 + أطوار القمر
│   ├── calendar_events.py   # FOMC + NFP + CPI + خطابات Powell
│   ├── signals.py           # محرك التقييم (Scoring Engine)
│   ├── risk.py              # حساب Entry/SL/TP
│   ├── storage.py           # تخزين SQLite + تتبع النتائج + Shadow Tracking
│   ├── notifier.py          # إشعارات Telegram
│   ├── telegram_control.py  # أوامر التحكم عن بعد
│   ├── learning_engine.py   # محرك التعلم التكيفي + Shadow Learning
│   ├── quality_first.py     # فلتر الجودة
│   ├── llm_postmortem.py    # تحليل ما بعد الصفقة بالذكاء الاصطناعي
│   ├── ml_engine.py         # نموذج ML للتنبؤ
│   ├── news_analyzer.py     # تحليل الأخبار
│   ├── yahoo_data.py        # بيانات Yahoo Finance للماكرو
│   └── utils.py             # أدوات مساعدة
├── tests/                   # اختبارات Pytest
└── backtests/               # أدوات الباك تست
```

## 🧠 أنظمة الذكاء في البوت

### 1. محرك التعلم التكيفي (Adaptive Learning Engine)
- يتدرب على آخر 21 يوم من نتائج الصفقات
- يحسب winrate متوقعة لكل **رمز + اتجاه + نوع سوق + نظام سوق**
- يعدّل تقييم الإشارات بناءً على الأداء التاريخي
- يحظر الإشارات ذات التوقعات السلبية

### 2. نظام التعلم من المرفوضات (Shadow Learning) 🆕
- **Shadow Tracking:** كل صفقة يرفضها الفلتر تُحفظ ويُتتبع سعرها
- **تقييم القرارات:** يفحص هل الصفقة المرفوضة كانت ستربح أو ستخسر
- **تغذية راجعة:** نتائج الصفقات المرفوضة تُدمج في بيانات التدريب
- **تصحيح ذكي:** لكل رمز/اتجاه يُحسب "معدل الخطأ" ويُصحح تلقائياً
- **تقرير يومي:** يُرسل على Telegram يوضح دقة قرارات الرفض

### 3. تحليل ما بعد الصفقة (LLM Postmortem)
- يستخدم الذكاء الاصطناعي لتحليل الصفقات الخاسرة
- يحدد أخطاء مثل: `late_entry`, `stop_too_tight`, `ignored_news_risk`
- يُطبّق عقوبات تلقائية على أنماط الدخول المتكررة الخاطئة

### 4. نموذج التعلم الآلي (ML Engine)
- يتدرب على ميزات الإشارات السابقة
- يُعطي تقييم إضافي لكل إشارة جديدة

## 📊 قاعدة البيانات

| الجدول | الوظيفة |
|--------|---------|
| `signals` | الإشارات المُرسلة (رمز، اتجاه، سعر، SL، TP) |
| `signal_outcomes` | نتائج الصفقات (TP_HIT, SL_HIT, TRAIL_HIT...) |
| `llm_trade_reviews` | تحليلات AI لما بعد الصفقة |
| `rejected_signals` | 🆕 الصفقات المرفوضة تحت التتبع (Shadow Tracking) |
| `pending_entries` | إشارات تنتظر pullback للدخول |

## ⚙️ مصادر البيانات

| المصدر | الاستخدام |
|--------|-----------|
| **MT5 Direct** | ✅ المصدر الرئيسي الحالي (Exness) |
| **Binance/CCXT** | بيانات كريبتو بديلة |
| **Yahoo Finance** | بيانات ماكرو (XAU, OIL, SNP500) |
| **MT5 Bridge** | خيار بديل عبر HTTP API |

## 🚀 التشغيل

### المتطلبات
- Python 3.11+
- MetaTrader 5 (مثبت ومسجل الدخول)
- اتصال إنترنت

### الإعداد
```bash
# تثبيت المتطلبات
pip install -r requirements.txt

# إعداد ملف البيئة (.env)
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
```

### التشغيل
```bash
# مسح واحد (اختبار)
python main.py --once

# تشغيل مستمر
python main.py

# مع سجلات مفصلة
python main.py -v
```

## 📱 أوامر Telegram

| الأمر | الوظيفة |
|-------|---------|
| `/pause` | إيقاف مؤقت للمسح |
| `/resume` | استئناف المسح |
| `/status` | حالة البوت الحالية |
| `/stop` | إيقاف البوت |
| `/help` | عرض الأوامر |

## 📈 التقارير اليومية على Telegram

1. **تقرير الأداء (WinRate):** نسبة الربح، عدد الصفقات، متوسط PnL
2. **تقرير Shadow:** حالة الإشارات التجريبية
3. **تقرير قرارات الرفض:** 🆕 دقة الرفض، فرص ضائعة، خسائر تم تجنبها

## 🔒 إدارة المخاطر

- **Quick TP:** هدف ربح قريب للخروج السريع
- **TP1 + Break Even:** عند TP1 يُنقل الوقف لنقطة التعادل
- **Trailing Stop:** وقف متحرك بعد TP1
- **Partial Exit (60/40):** خروج 60% عند TP1، 40% يتابع لـ TP2
- **Altcoin Correlation Filter:** يمنع تكديس إشارات متشابهة
- **BTC Pulse:** يحظر LONG على العملات البديلة عندما BTC هابط
- **Session Filter:** حذر إضافي في ساعات آسيا الميتة

## 🔄 دورة التعلم الكاملة

```
إشارة جديدة → Learning Filter → [قبول] → تنفيذ → نتيجة → تدريب
                                → [رفض]  → Shadow Track → تقييم → تصحيح ذكي → تدريب
```

## 📋 الرموز المدعومة حالياً

### كريبتو
BTC, ETH, SOL, BNB, XRP, ADA, DOGE, DOT, AVAX, LINK,
UNI, AAVE, COMP, ENJ, THETA, SNX, XTZ, LTC, FIL

### ماكرو
XAUUSD (ذهب), XAGUSD (فضة), OILUSD/USOIL/WTIUSD (نفط),
EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD, USDCAD,
SNP500/SPX500/US500 (مؤشرات)

## License

MIT License
