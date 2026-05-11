# Crypto Signal Bot - تشغيل بدون Docker

## التشغيل بدون Docker (Windows)

### المتطلبات:
- Python 3.11 أو أحدث
- اتصال بالإنترنت

### الخطوات:

1. **تثبيت Python** (إذا لم يكن مثبتاً):
   - قم بتحميل Python من: https://www.python.org/downloads/
   - تأكد من إضافة Python إلى PATH أثناء التثبيت

2. **تثبيت المتطلبات**:
   ```bash
   pip install -r requirements.txt
   ```

3. **إعداد متغيرات البيئة**:
   - قم بتحرير الملف `.env` وتأكد من وجود:
   ```
   TELEGRAM_BOT_TOKEN=8257131897:AAESCFNiZVCHk6tw_QidkNIeP3S6wEnWTb8
   TELEGRAM_CHAT_ID=-4829541353
   OPENAI_API_KEY=sk-proj-...
   ```

4. **تشغيل البوت**:
   - للمسح الواحد: `python main.py --once`
   - للتشغيل المستمر: `python main.py`
   - أو استخدم `run_bot.bat` للتشغيل السهل

## استكشاف الأخطاء:

### إذا لم يرسل البوت إشارات:
- البوت يرسل إشارات فقط عندما تتوفر إعدادات تداول مناسبة
- تحقق من السجلات لمعرفة سبب عدم إرسال الإشارات
- جرب `python main.py --once` لمسح واحد

### إذا لم يعمل Telegram:
- تأكد من صحة `TELEGRAM_BOT_TOKEN` و `TELEGRAM_CHAT_ID`
- جرب `python test_telegram.py` للاختبار

### إذا ظهرت أخطاء في المتطلبات:
- قم بتحديث pip: `python -m pip install --upgrade pip`
- أعد تثبيت المتطلبات: `pip install -r requirements.txt`

## ملاحظات مهمة:
- البوت يرسل الإشارات إلى Telegram عندما يجد فرص تداول مناسبة
- قد لا يرسل إشارات كل دقيقة - هذا طبيعي
- راقب السجلات لفهم ما يحدث</content>
<parameter name="filePath">c:\Users\Administrator.WIN-J2S4568BP74\Desktop\trading\trading\crypto_signal_bot\README_Windows.md