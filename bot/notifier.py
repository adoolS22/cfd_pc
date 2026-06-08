"""
Telegram Notifier
=================
Sends formatted signals to Telegram.
"""

import html
import re
import requests
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from loguru import logger

from .signals import SignalResult
from .risk import format_risk_levels
from .utils import format_price, Config


def _get_section_decisions(signal: SignalResult) -> Dict[str, str]:
    """Infer buy/sell/neutral decision for each analysis section."""
    technical_decision = 'neutral'
    if signal.side == 'LONG':
        technical_decision = 'buy'
    elif signal.side == 'SHORT':
        technical_decision = 'sell'

    if signal.timing_score > 0.5:
        news_decision = 'buy'
    elif signal.timing_score < -0.5:
        news_decision = 'sell'
    else:
        news_decision = 'neutral'

    if signal.vpa_score > 0:
        vpa_decision = 'buy'
    elif signal.vpa_score < 0:
        vpa_decision = 'sell'
    else:
        vpa_decision = 'neutral'

    return {
        'technical': technical_decision,
        'news': news_decision,
        'vpa': vpa_decision,
    }


def _get_consensus_decision(signal: SignalResult) -> Optional[str]:
    """Return the shared decision when all three sections agree."""
    decisions = _get_section_decisions(signal)
    unique_decisions = set(decisions.values())

    if len(unique_decisions) == 1:
        agreed = unique_decisions.pop()
        if agreed in {'buy', 'sell'}:
            return agreed

    return None


# Token color mapping - each token gets a unique colored emoji
TOKEN_COLORS = {
    "SOL": "🟣",      # Purple
    "ARKM": "🔵",     # Blue
    "ORDI": "🟠",     # Orange
    "PYTH": "🟤",     # Brown
    "SUI": "🩵",      # Light Blue
    "BTC": "🟡",      # Yellow (Gold)
    "BNB": "🟨",      # Yellow Square
    "XRP": "⚫",      # Black
    "DOGE": "🟫",     # Brown Square
    "ADA": "🔷",      # Blue Diamond
    "AVAX": "🔺",     # Red Triangle
    "LINK": "🔹",     # Small Blue Diamond
    "MATIC": "🟪",    # Purple Square
    "DOT": "⬛",      # Black Square
    "XAU": "🥇",      # Gold Medal (Real Gold)
    "XAG": "🥈",      # Silver Medal (Real Silver)
    "OIL": "🛢️",      # Oil barrel
    "WTI": "🛢️",
    "BRENT": "🛢️",
    "SNP500": "📊",   # S&P500
    "SPX500": "📊",
    "S&P500": "📊",
    "SP500": "📊",
    "EUR": "💶",      # Euro FX
    "EURUSD": "💶",
}

def get_token_color(symbol: str) -> str:
    """Get the color emoji for a token symbol."""
    # Extract base token from symbol (e.g., "SOL/USDT:USDT" -> "SOL")
    base_token = symbol.split('/')[0].upper()
    return TOKEN_COLORS.get(base_token, "⬜")  # Default to white square


def _escape_html_text(value: Any) -> str:
    """Escape user/model text for Telegram HTML parse mode."""
    return html.escape(str(value), quote=False)


def _format_zone_levels(levels: Any, symbol: str, max_items: int = 3) -> str:
    """Format numeric levels for compact Telegram display."""
    if not isinstance(levels, list) or not levels:
        return "غير متاح"
    out: List[str] = []
    for v in levels[:max_items]:
        try:
            out.append(format_price(float(v), symbol))
        except Exception:
            continue
    return " | ".join(out) if out else "غير متاح"


def _to_arabic_reason(reason: Any) -> str:
    """Best-effort conversion for common English reason snippets to Arabic."""
    text = str(reason or "").strip()
    if not text:
        return ""

    exact = {
        "✓ At support zone": "✓ السعر ضمن منطقة دعم",
        "✓ At resistance zone": "✓ السعر ضمن منطقة مقاومة",
        "✓ Volume spike": "✓ ارتفاع واضح بالحجم",
        "✓ Wave 3 setup": "✓ تم رصد إعداد موجة 3",
        "✗ RSI bearish divergence": "✗ انحراف RSI هابط",
        "✗ RSI bullish divergence": "✗ انحراف RSI صاعد",
        "✓ Regime uptrend aligned with LONG": "✓ توافق مع سوق صاعد (اتجاه شراء)",
        "✗ Regime uptrend opposes SHORT": "✗ بيع عكس اتجاه سوق صاعد",
        "✓ Regime downtrend aligned with SHORT": "✓ توافق مع سوق هابط (اتجاه بيع)",
        "✗ Regime downtrend opposes LONG": "✗ شراء عكس اتجاه سوق هابط",
        "⚠ Regime sideways: higher fakeout risk": "⚠ السوق عرضي: احتمال الإشارات الكاذبة أعلى",
        "⚠ Regime high volatility: spread/noise risk": "⚠ تقلب عالي: مخاطر سبريد/ضوضاء أعلى",
        "✗ HTF trend opposes LONG": "✗ الإطار الزمني الأكبر هابط — يُعارض الشراء",
        "✗ HTF trend opposes SHORT": "✗ الإطار الزمني الأكبر صاعد — يُعارض البيع",
        "✓ HTF trend confirms LONG": "✓ الإطار الزمني الأكبر يؤكد اتجاه الشراء",
        "✓ HTF trend confirms SHORT": "✓ الإطار الزمني الأكبر يؤكد اتجاه البيع",
        "✓ Morning Star": "✓ نموذج نجمة الصباح",
        "✓ Evening Star": "✓ نموذج نجمة المساء",
        "✓ Three White Soldiers": "✓ ثلاثة جنود بيض",
        "✓ Three Black Crows": "✓ ثلاثة غربان سود",
        "✓ Doji": "✓ شمعة دوجي (حيرة)",
        "✓ Dragonfly Doji": "✓ شمعة دوجي اليعسوب",
        "✓ Gravestone Doji": "✓ شمعة دوجي شاهد القبر",
        "✓ Trendline bullish break": "✓ كسر خط ترند صاعد",
        "✓ Trendline bearish break": "✓ كسر خط ترند هابط",
        "✗ Trendline bullish break opposes": "✗ كسر ترند صاعد يعارض اتجاه البيع",
        "✗ Trendline bearish break opposes": "✗ كسر ترند هابط يعارض اتجاه الشراء",
        "✓ HH/HL structure": "✓ هيكلية قمم وقيعان صاعدة (HH/HL)",
        "✓ LH/LL structure": "✓ هيكلية قمم وقيعان هابطة (LH/LL)",
        "✓ Double Top": "✓ قمة مزدوجة",
        "✓ Double Bottom": "✓ قاع مزدوج",
        "✗ Structure opposes LONG": "✗ الهيكلية الحالية تعارض الشراء",
        "✗ Structure opposes SHORT": "✗ الهيكلية الحالية تعارض البيع",
        "✓ Near Bullish Order Block (Sniper Entry)": "✓ قريب من بلوك أوامر شرائي (دخول قناص 🐋)",
        "✓ Near Bearish Order Block (Sniper Entry)": "✓ قريب من بلوك أوامر بيعي (دخول قناص 🐋)",
        "✓ Bullish Order Block": "✓ بلوك أوامر شرائي قريب (دخول قناص 🐋)",
        "✓ Bearish Order Block": "✓ بلوك أوامر بيعي قريب (دخول قناص 🐋)",
        "✓ Bullish Liquidity Sweep (Retail Trapped)": "✓ صيد سيولة هابط (فخ للمتداولين - الدخول شراء 🛒)",
        "✓ Bearish Liquidity Sweep (Retail Trapped)": "✓ صيد سيولة صاعد (فخ للمتداولين - الدخول بيع 🛒)",
        "✓ Bullish Liquidity Sweep": "✓ صيد سيولة هابط داعم للشراء 🛒",
        "✓ Bearish Liquidity Sweep": "✓ صيد سيولة صاعد داعم للبيع 🛒",
        "✗ Unfilled FVG opposes entry (Magnet Risk)": "✗ فجوة سعرية (FVG) معاكسة (خطر انجذاب السعر 🧲)",
    }
    if text in exact:
        return exact[text]

    if text.startswith("✓ Trend:"):
        direction = text.split(":", 1)[1].strip().lower()
        direction_ar = {"up": "صاعد", "down": "هابط", "neutral": "محايد"}.get(direction, direction)
        return f"✓ الاتجاه: {direction_ar}"

    if text.startswith("○ Trend:"):
        direction = text.split(":", 1)[1].strip().lower()
        direction_ar = {"up": "صاعد", "down": "هابط", "neutral": "محايد"}.get(direction, direction)
        return f"○ الاتجاه: {direction_ar}"

    if text.startswith("○ Near support zone"):
        return text.replace("○ Near support zone", "○ قريب من منطقة دعم")
    if text.startswith("○ Near resistance zone"):
        return text.replace("○ Near resistance zone", "○ قريب من منطقة مقاومة")
    if text.startswith("✗ Ranging market"):
        return text.replace("✗ Ranging market", "✗ سوق عرضي")
    if text.startswith("✓ Strong trend"):
        return text.replace("✓ Strong trend", "✓ اتجاه قوي")
    if text.startswith("✓ SMC Confluence x"):
        return text.replace("sniper entry", "دخول قناص")
    if text.startswith("Timing contribution applied to final score:"):
        val = text.split(":", 1)[1].strip()
        return f"تأثير التوقيت/الأخبار على السكور النهائي: {val}"
    if text.startswith("Learning:"):
        mapped = text.replace("Learning:", "التعلّم:")
        mapped = mapped.replace("samples=", "عينات=").replace("wr=", "نسبة نجاح=")
        mapped = mapped.replace("pnl=", "العائد=").replace("adj=", "التعديل=")
        return mapped
    if text.startswith("Quality filter: high-impact news window active (cautious mode)"):
        return "فلتر الجودة: نافذة خبر قوي مفعّلة (وضع حذر)"
    if text.startswith("Quality filter: high-impact news window active (warn-only mode)"):
        return "فلتر الجودة: نافذة خبر قوي مفعّلة (تحذير فقط)"
    if text.startswith("⚠ Quality filter: high-impact news window active (warn-only mode)"):
        return "⚠ فلتر الجودة: نافذة خبر قوي مفعّلة (تحذير فقط)"
    if text.startswith("⚠ Quality filter: high-impact news window active (cautious mode)"):
        return "⚠ فلتر الجودة: نافذة خبر قوي مفعّلة (وضع حذر)"
    m = re.search(
        r"Quality filter: opposing news score\s+([-+0-9.]+)\s+for\s+(LONG|SHORT)(?:\s+\(warn-only mode\))?",
        text,
        re.IGNORECASE,
    )
    if m:
        side_ar = "الشراء" if m.group(2).upper() == "LONG" else "البيع"
        score = m.group(1)
        if "warn-only mode" in text:
            return f"فلتر الجودة: تعارض أخبار قوي ({score}) عكس اتجاه {side_ar} (تحذير فقط)"
        return f"فلتر الجودة: تعارض أخبار قوي ({score}) عكس اتجاه {side_ar}"
    if text.startswith("Quality filter: outside macro active session (cautious mode)"):
        return "فلتر الجودة: خارج جلسة الذهب/النفط النشطة (وضع حذر)"
    if text.startswith("Quality filter: outside macro active session"):
        return "فلتر الجودة: خارج جلسة الذهب/النفط النشطة"
    if text.startswith("Quality: cautious filters raised threshold"):
        return text.replace(
            "Quality: cautious filters raised threshold",
            "فلتر الجودة: تم رفع عتبة الدخول بسبب وضع الحذر"
        )
    if text.startswith("Threshold: asset base"):
        return text.replace("Threshold: asset base", "العتبة: أساس حسب نوع الأصل")
    if text.startswith("Threshold: dynamic high-vol ATR"):
        return text.replace("Threshold: dynamic high-vol ATR", "العتبة الديناميكية: تقلب مرتفع ATR")
    if text.startswith("Threshold: dynamic low-vol ATR"):
        return text.replace("Threshold: dynamic low-vol ATR", "العتبة الديناميكية: تقلب منخفض ATR")
    if text.startswith("Threshold: dynamic strong-trend ADX"):
        return text.replace("Threshold: dynamic strong-trend ADX", "العتبة الديناميكية: اتجاه قوي ADX")
    if text.startswith("Threshold: VPA contradiction"):
        m = re.match(
            r"^Threshold: VPA contradiction\s+([a-z_]+)\s+opposes\s+(LONG|SHORT)\s+\(([-+0-9.]+)\)$",
            text,
            re.IGNORECASE,
        )
        if m:
            sig = m.group(1).strip().lower()
            side = m.group(2).strip().upper()
            add = m.group(3).strip()
            sig_ar = {
                "distribution": "توزيع",
                "accumulation": "تجميع",
                "confirmed_breakout_down": "اختراق هابط مؤكد",
                "confirmed_breakout_up": "اختراق صاعد مؤكد",
                "climax_buy": "ذروة شراء",
                "climax_sell": "ذروة بيع",
                "effort_no_result_up": "جهد بلا نتيجة صعوداً",
                "effort_no_result_down": "جهد بلا نتيجة هبوطاً",
                "weak_move_up": "صعود ضعيف",
                "weak_move_down": "هبوط ضعيف",
            }.get(sig, sig)
            side_ar = "الشراء" if side == "LONG" else "البيع" if side == "SHORT" else side
            return f"العتبة الديناميكية: تحذير VPA ({sig_ar}) يعاكس {side_ar} ({add})"
        return text.replace("Threshold: VPA contradiction", "العتبة الديناميكية: تعارض VPA")
    if text.startswith("Threshold: floor applied"):
        return text.replace("Threshold: floor applied", "العتبة: تم تطبيق الحد الأدنى")
    if text.startswith("LLM News:"):
        return text.replace("LLM News:", "تحليل الأخبار:")

    m = re.match(r"^Gann:\s*([a-zA-Z0-9_]+)\s*\(score:\s*([^)]+)\)$", text)
    if m:
        rel = m.group(1).strip().lower()
        rel_ar = {
            "above_angles": "فوق زوايا غان",
            "below_angles": "تحت زوايا غان",
            "between_angles": "بين زوايا غان",
            "mixed": "إشارة مختلطة",
        }.get(rel, rel)
        return f"غان: {rel_ar} (سكور: {m.group(2).strip()})"

    m = re.match(r"^Sq9:\s*([0-9.]+)\s*\(([^,]+),\s*score:\s*([^)]+)\)$", text)
    if m:
        return f"مربع 9: {m.group(1)} ({m.group(2)}، سكور: {m.group(3)})"

    m = re.match(r"^Cycle52:\s*IN WINDOW\s*\(pos:\s*([^)]+)\)$", text, re.IGNORECASE)
    if m:
        return f"دورة 52: داخل نافذة الحدث (الموضع: {m.group(1)})"

    m = re.match(r"^Lunar:\s*([a-zA-Z_]+)\s*moon window$", text, re.IGNORECASE)
    if m:
        phase = m.group(1).strip().lower()
        phase_ar = {
            "new": "محاق",
            "waxing": "متزايد",
            "full": "بدر",
            "waning": "متناقص",
        }.get(phase, phase)
        return f"القمر: طور {phase_ar} (نافذة تأثير)"

    m = re.match(r"^⚠️\s*FOMC in\s*([0-9.]+)\s*days$", text, re.IGNORECASE)
    if m:
        return f"⚠️ اجتماع FOMC خلال {m.group(1)} يوم"
    m = re.match(r"^⚠️\s*CPI in\s*([0-9.]+)h$", text, re.IGNORECASE)
    if m:
        return f"⚠️ بيانات CPI خلال {m.group(1)} ساعة"
    m = re.match(r"^⚠️\s*NFP in\s*([0-9.]+)h$", text, re.IGNORECASE)
    if m:
        return f"⚠️ بيانات الوظائف NFP خلال {m.group(1)} ساعة"
    m = re.match(r"^⚠️\s*Powell speech in\s*([0-9.]+)h$", text, re.IGNORECASE)
    if m:
        return f"⚠️ خطاب باول خلال {m.group(1)} ساعة"
    m = re.match(r"^⚠️\s*FOMC minutes in\s*([0-9.]+)h$", text, re.IGNORECASE)
    if m:
        return f"⚠️ محضر FOMC خلال {m.group(1)} ساعة"

    m = re.match(r"^⚠️\s*Sentiment EXTREME \(([^:]+):\s*([^)]+)\)$", text, re.IGNORECASE)
    if m:
        return f"⚠️ معنويات متطرفة ({m.group(1).strip()}: {m.group(2).strip()})"
    m = re.match(r"^Reddit caution\s*([A-Za-z]+)\s*\(B\s*([0-9.]+)%\s*/\s*S\s*([0-9.]+)%\)$", text, re.IGNORECASE)
    if m:
        side = "شراء" if m.group(1).lower() in {"bull", "bullish", "buy"} else "بيع"
        return f"تحذير ريديت ({side}) (شراء {m.group(2)}% / بيع {m.group(3)}%)"
    m = re.match(r"^⚠️\s*Reddit EXTREME\s*([A-Za-z]+)\s*\(B\s*([0-9.]+)%\s*/\s*S\s*([0-9.]+)%\)$", text, re.IGNORECASE)
    if m:
        side = "شراء" if m.group(1).lower() in {"bull", "bullish", "buy"} else "بيع"
        return f"⚠️ ريديت متطرف ({side}) (شراء {m.group(2)}% / بيع {m.group(3)}%)"

    m = re.match(r"^Fib:\s*([0-9.]+)\s*([A-Za-z_]+)\s*@\s*([0-9.]+)\s*\(([^)]+)\)$", text)
    if m:
        kind = m.group(2).strip().lower()
        kind_ar = {"retracement": "تصحيح", "extension": "امتداد"}.get(kind, kind)
        return f"فيبوناتشي: {m.group(1)} {kind_ar} @ {m.group(3)} ({m.group(4)})"

    m = re.match(r"^[✓✗]\s*Price (above|below) VWAP\s*\(([^)]+)\)$", text, re.IGNORECASE)
    if m:
        dir_ar = "فوق" if m.group(1).lower() == "above" else "تحت"
        mark = "✓" if text.strip().startswith("✓") else "✗"
        return f"{mark} السعر {dir_ar} VWAP ({m.group(2)})"

    m = re.match(r"^[✓✗]\s*CMF (buying|selling) pressure\s*\(([^)]+)\)$", text, re.IGNORECASE)
    if m:
        pressure_ar = "ضغط شراء" if m.group(1).lower() == "buying" else "ضغط بيع"
        mark = "✓" if text.strip().startswith("✓") else "✗"
        return f"{mark} {pressure_ar} (CMF: {m.group(2)})"

    m = re.match(r"^([✓✗⚠])\s*(.+)$", text)
    if m:
        mark = m.group(1)
        body = m.group(2).strip()
        body = body.replace("_", " ")
        body = re.sub(r"\bBullish\b", "صاعد", body)
        body = re.sub(r"\bBearish\b", "هابط", body)
        body = re.sub(r"\bBlocked\b", "مرفوض", body)
        body = re.sub(r"\bPenalty\b", "خصم", body)
        body = re.sub(r"\bStrong bullish momentum\b", "زخم صاعد قوي", body, flags=re.IGNORECASE)
        body = re.sub(r"\bStrong bearish momentum\b", "زخم هابط قوي", body, flags=re.IGNORECASE)
        body = re.sub(r"\bAlready oversold\b", "السوق في تشبع بيع مسبقاً", body, flags=re.IGNORECASE)
        body = re.sub(r"\bAlready overbought\b", "السوق في تشبع شراء مسبقاً", body, flags=re.IGNORECASE)
        return f"{mark} {body}"

    # Keep percentages and numeric content; only convert a few common tokens.
    text = re.sub(r"\bLONG\b", "شراء", text)
    text = re.sub(r"\bSHORT\b", "بيع", text)
    text = re.sub(r"\bBUY\b", "شراء", text)
    text = re.sub(r"\bSELL\b", "بيع", text)
    text = re.sub(r"\bWAIT\b", "انتظار", text)
    return text


class TelegramNotifier:
    """Telegram notification handler."""
    
    TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
    TELEGRAM_PHOTO_URL = "https://api.telegram.org/bot{token}/sendPhoto"
    TELEGRAM_UPDATES_URL = "https://api.telegram.org/bot{token}/getUpdates"
    TELEGRAM_MAX_TEXT_LENGTH = 4096
    TELEGRAM_SAFE_CHUNK_LENGTH = 3500
    
    def __init__(self, config: Config):
        """
        Initialize notifier.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.enabled = config.telegram.enabled
        self.bot_token = config.telegram.bot_token
        self.chat_id = config.telegram.chat_id
        
        if self.enabled and (not self.bot_token or not self.chat_id):
            logger.warning("Telegram enabled but credentials missing - will print to console")
            self.enabled = False
    
    def send_signal(self, signal: SignalResult, chart_path: Optional[str] = None) -> bool:
        """
        Send a signal notification.
        
        Args:
            signal: SignalResult to send
            chart_path: Optional path to an image chart to attach to the signal message
            
        Returns:
            True if sent successfully
        """
        if not signal.is_valid or not signal.side:
            return False
        
        sent_at = datetime.now(timezone.utc)
        message = self._format_signal_message(signal, sent_at=sent_at)
        
        if self.enabled:
            if chart_path:
                status = self._send_telegram_photo(chart_path, caption=message)
                if not status:
                    logger.warning(f"Could not submit photo {chart_path}, rolling back to text-only send.")
                    return self._send_telegram(message)
                return True
            else:
                return self._send_telegram(message)
        else:
            print("\n" + "=" * 50)
            if chart_path:
                print(f"[IMAGE ATTACHED: {chart_path}]")
            print(message)
            print("=" * 50 + "\n")
            return True

    def send_text(self, message: str) -> bool:
        """Send plain text message (HTML enabled) or print to console."""
        if self.enabled:
            return self._send_telegram(message)
        print(f"\n{message}\n")
        return True
    
    def _format_signal_message(self, signal: SignalResult, sent_at: Optional[datetime] = None) -> str:
        """Format signal into Telegram message in Arabic."""

        token_color = get_token_color(signal.symbol)
        color_line = token_color * 10

        if signal.side == 'LONG':
            direction_emoji = "📈"
            direction_ar = "شراء"
        elif signal.side == 'SHORT':
            direction_emoji = "📉"
            direction_ar = "بيع"
        else:
            direction_emoji = "⏹️"
            direction_ar = "إغلاق"

        trend_ar = {"up": "صاعد ⬆️", "down": "هابط ⬇️", "neutral": "محايد ↔️"}.get(signal.trend, signal.trend)

        lines = [
            color_line,
            f"{direction_emoji} <b>إشارة {direction_ar} — {signal.symbol}</b>",
            color_line,
            "",
            f"💰 السعر: <code>{format_price(signal.current_price)}</code>",
            f"📊 الاتجاه: {trend_ar}",
            "",
        ]

        # Risk Levels
        if signal.risk_levels:
            levels = format_risk_levels(signal.risk_levels, signal.symbol)
            resistance_text = _format_zone_levels(
                (signal.zone_info or {}).get('resistance_levels'),
                signal.symbol,
            )
            support_text = _format_zone_levels(
                (signal.zone_info or {}).get('support_levels'),
                signal.symbol,
            )
            lines.extend([
                "━━━━ <b>مستويات الدخول</b> ━━━━",
                f"▫️ الدخول:       <code>{levels['entry']}</code>",
                f"🛑 وقف الخسارة: <code>{levels['stop_loss']}</code> ({levels['risk_pct']})",
                f"⚡ الهدف القريب: <code>{levels['take_profit_near']}</code> ({levels['rr_tp_near']})",
                f"🎯 الهدف 1:     <code>{levels['take_profit_1']}</code> ({levels['rr_tp1']})",
                f"🎯 الهدف 2:     <code>{levels['take_profit_2']}</code> ({levels['rr_tp2']})",
                f"🧱 مناطق المقاومة: <code>{resistance_text}</code>",
                f"🟩 مناطق الدعم:   <code>{support_text}</code>",
                "🧾 مرجع التسعير (MT5-like): الشارت Bid | LONG دخول Ask/خروج Bid | SHORT دخول Bid/خروج Ask",
                "",
            ])

        # ── سكور التحليل الفني ──
        lines.append(f"━━━━ <b>⚙️ التحليل الفني</b>  (سكور: {signal.total_score:.0f}/{signal.threshold}) ━━━━")
        if signal.zone_info:
            adx_val = signal.zone_info.get('adx', 0)
            rsi_val = signal.zone_info.get('rsi_trend', 50)
            ichi    = signal.zone_info.get('ichimoku_signal', 'neutral')
            regime = signal.zone_info.get('market_regime', signal.market_regime if hasattr(signal, 'market_regime') else 'sideways')
            adx_ar  = "سوق عرضي ⚠️" if adx_val < 20 else ("اتجاه قوي 💪" if adx_val > 35 else "اتجاه متطور 📊")
            rsi_ar  = ("مشتري بشدة 🔴" if rsi_val > 70 else
                       "صاعد 🟢" if rsi_val > 55 else
                       "مبيع بشدة 🔴" if rsi_val < 30 else
                       "هابط 🔴" if rsi_val < 45 else "محايد ⚪")
            ichi_ar = {"bullish": "فوق السحابة 🟢", "bearish": "تحت السحابة 🔴", "neutral": "داخل السحابة ⚪"}.get(ichi, "⚪")
            regime_ar = {
                'uptrend': 'صاعد',
                'downtrend': 'هابط',
                'sideways': 'عرضي',
                'high_volatility': 'تقلب عالي',
            }.get(str(regime), str(regime))
            lines.append(f"  ADX: {adx_val:.0f} ({adx_ar})")
            lines.append(f"  RSI: {rsi_val:.0f} ({rsi_ar})")
            lines.append(f"  ايشيموكو: {ichi_ar}")
            lines.append(f"  حالة السوق: {regime_ar}")
        for r in (signal.reasons or []):
            rr = _to_arabic_reason(r)
            if rr:
                lines.append(f"  {_escape_html_text(rr)}")
        lines.append("")

        # ── سكور الأخبار ──
        ns = signal.timing_score
        news_label = "🟢 إيجابي" if ns > 0.5 else ("🔴 سلبي" if ns < -0.5 else "⚪ محايد")
        lines.append(f"━━━━ <b>📰 الأخبار والأحداث</b>  (سكور: {ns:.1f} — {news_label}) ━━━━")
        if signal.timing_info:
            ti = signal.timing_info
            if getattr(ti, 'cpi_analysis', None) and ti.cpi_analysis and ti.cpi_analysis.next_release:
                ca = ti.cpi_analysis
                warn = "⚠️ تقلب متوقع" if ca.in_high_vol_window else ""
                lines.append(f"  CPI: بعد {ca.hours_to_next:.0f} ساعة {warn}")
            if getattr(ti, 'nfp_analysis', None) and ti.nfp_analysis and ti.nfp_analysis.next_release:
                na = ti.nfp_analysis
                warn = "⚠️ تقلب متوقع" if na.in_high_vol_window else ""
                lines.append(f"  NFP: بعد {na.hours_to_next:.0f} ساعة {warn}")
            if getattr(ti, 'fomc_analysis', None) and ti.fomc_analysis and ti.fomc_analysis.next_meeting_end:
                fa = ti.fomc_analysis
                warn = "⚠️ تقلب متوقع" if fa.in_high_vol_window else ""
                lines.append(f"  FOMC: بعد {fa.days_to_next} يوم {warn}")
            if getattr(ti, 'powell_analysis', None) and ti.powell_analysis and ti.powell_analysis.next_event:
                pa = ti.powell_analysis
                warn = "⚠️ تقلب متوقع" if pa.in_high_vol_window else ""
                lines.append(f"  باول: بعد {pa.hours_to_next:.0f} ساعة {warn}")
            if getattr(ti, 'llm_news_headlines_count', 0):
                lines.append(f"  تحليل الأخبار: {int(ti.llm_news_headlines_count)} عنوان")
            if getattr(ti, 'llm_news_categories', None):
                cat_map = {
                    'crypto': 'كريبتو',
                    'macro': 'ماكرو',
                    'metals': 'معادن',
                    'oil': 'نفط',
                    'geopolitics': 'جيوسياسي',
                    'other': 'أخرى',
                }
                cats = [cat_map.get(str(c), str(c)) for c in ti.llm_news_categories]
                if cats:
                    lines.append(f"  الفئات: {', '.join(cats[:5])}")
            if getattr(ti, 'llm_news_sources', None):
                srcs = [str(s) for s in ti.llm_news_sources if str(s).strip()]
                if srcs:
                    preview = ", ".join(srcs[:3]) + ("..." if len(srcs) > 3 else "")
                    lines.append(f"  المصادر: {preview}")
            if ti.gann_analysis and ti.gann_analysis.angle_confluence_score > 0:
                lines.append(f"  غان: {ti.gann_analysis.price_relation} ✓")
            if ti.square9_analysis and ti.square9_analysis.square9_score > 0 and ti.square9_analysis.nearest_level:
                nl = ti.square9_analysis.nearest_level
                lines.append(f"  مربع 9: {nl.level:.2f} ({nl.distance_pct:.2f}%) ✓")
            if ti.lunar_analysis and ti.lunar_analysis.in_event_window:
                lines.append(f"  القمر: {ti.lunar_analysis.phase} — نافذة حدث 🌕")
            if ti.fib_analysis and ti.fib_analysis.nearest_level and ti.fib_analysis.distance_pct < 0.5:
                nl = ti.fib_analysis.nearest_level
                lines.append(f"  فيبوناتشي: {nl.ratio} @ {nl.price:.4f} ({ti.fib_analysis.distance_pct:.2f}%) ✓")
        for r in (signal.timing_reasons or []):
            rr = _to_arabic_reason(r)
            if rr:
                lines.append(f"  {_escape_html_text(rr)}")
        lines.append("")

        # ── سكور السعر والحجم (VPA) ──
        if signal.vpa_info:
            vi  = signal.vpa_info
            vs  = vi.get('score', 0.0)
            vpa_label = ("🟢 شراء" if vs >= 2 else "🔴 بيع" if vs <= -2 else "🟡 ميول شراء" if vs > 0 else "🟠 ميول بيع" if vs < 0 else "⚪ محايد")
            lines.append(f"━━━━ <b>💹 السعر والحجم (VPA)</b>  (سكور: {vs:+.0f} — {vpa_label}) ━━━━")
            sig_ar_map = {
                'confirmed_breakout_up':   "اختراق صاعد مؤكد ✅",
                'confirmed_breakout_down': "اختراق هابط مؤكد ✅",
                'accumulation':            "احتمال تجميع (شراء خفي) 🐳",
                'distribution':            "احتمال توزيع (ضغط بيع خفي) 🐳",
                'climax_buy':              "ذروة شراء ⚠️",
                'climax_sell':             "ذروة بيع ⚠️",
                'effort_no_result_up':     "جهد بلا نتيجة صعوداً ⚠️",
                'effort_no_result_down':   "جهد بلا نتيجة هبوطاً ⚠️",
                'weak_move_up':            "صعود بحجم ضعيف ⚠️",
                'weak_move_down':          "هبوط بحجم ضعيف ⚠️",
            }
            vpa_sig = vi.get('vpa_signal', 'neutral')
            if vpa_sig != 'neutral':
                lines.append(f"  الإشارة: {sig_ar_map.get(vpa_sig, vpa_sig)}")
            alignment = str(vi.get('signal_alignment', 'neutral')).strip().lower()
            live_side = str(signal.side or "").upper()
            if live_side in {"LONG", "SHORT"} and alignment in {"opposing", "weak_opposing"}:
                if abs(float(vs)) < 1.0:
                    lines.append("  ⚠️ ملاحظة: إشارة VPA الأولية معاكسة، لكن التأكيدات الأخرى جعلت القراءة مختلطة.")
                else:
                    lines.append("  ⚠️ ملاحظة: VPA يميل لعكس اتجاه الصفقة الحالية (إشارة حذر).")
            if vi.get('vwap') is not None:
                lines.append(f"  VWAP: <code>{format_price(vi['vwap'])}</code>")
            if vi.get('cmf') is not None:
                cmf_val = vi['cmf']
                cmf_ar  = "ضغط شراء 🟢" if cmf_val > 0.05 else ("ضغط بيع 🔴" if cmf_val < -0.05 else "محايد ⚪")
                lines.append(f"  CMF: {cmf_val:.2f} ({cmf_ar})")
            for r in (vi.get('reasons') or []):
                rr = _to_arabic_reason(r)
                if rr:
                    lines.append(f"  {_escape_html_text(rr)}")
            lines.append("")

        # ── رأي الخبير (LLM) ──
        if signal.learning_opinion:
            lo = signal.learning_opinion or {}
            mode = str(lo.get('mode', 'hybrid')).strip().lower()
            decision = str(lo.get('decision', signal.side or 'WAIT')).upper()
            decision_ar = {
                'LONG': "🟢 شراء",
                'SHORT': "🔴 بيع",
                'WAIT': "⚪ انتظار",
                'BUY': "🟢 شراء",
                'SELL': "🔴 بيع",
            }.get(decision, "⚪ انتظار")
            mode_ar = "تعلم فقط" if mode == "learning_only" else "هجين (فني + تعلم)"

            lines.append("━━━━ <b>🧪 قرار التعلم</b> ━━━━")
            lines.append(f"  النمط: {mode_ar}")
            lines.append(f"  القرار: {decision_ar}")

            selected_reason = _escape_html_text(_to_arabic_reason(lo.get('selected_reason', ''))).strip()
            if selected_reason:
                lines.append(f"  السبب: {selected_reason}")

            wr_edge = lo.get('wr_edge_pct')
            if wr_edge is not None:
                try:
                    lines.append(f"  أفضلية الاتجاه: {float(wr_edge):+.2f}%")
                except Exception:
                    pass

            thresholds = lo.get('thresholds') or {}
            min_samples = thresholds.get('min_samples')
            min_wr = thresholds.get('min_winrate_pct')
            min_pnl = thresholds.get('min_pnl_pct')
            min_edge = thresholds.get('min_edge_pct')
            if min_samples is not None or min_wr is not None or min_pnl is not None:
                parts: List[str] = []
                if min_samples is not None:
                    parts.append(f"عينات≥{int(min_samples)}")
                if min_wr is not None:
                    parts.append(f"نجاح≥{float(min_wr):.1f}%")
                if min_pnl is not None:
                    parts.append(f"عائد≥{float(min_pnl):+.2f}%")
                if min_edge is not None:
                    parts.append(f"أفضلية≥{float(min_edge):.2f}%")
                if parts:
                    lines.append(f"  شروط القرار: {' | '.join(parts)}")

            long_prof = lo.get('long')
            short_prof = lo.get('short')
            side_prof = lo.get('side_profile')
            if isinstance(long_prof, dict):
                lines.append(
                    "  ملف الشراء: "
                    f"نجاح {float(long_prof.get('expected_winrate_pct', 0.0)):.1f}% | "
                    f"عائد {float(long_prof.get('expected_pnl_pct', 0.0)):+.2f}% | "
                    f"عينات {int(long_prof.get('samples', 0))}"
                )
            if isinstance(short_prof, dict):
                lines.append(
                    "  ملف البيع: "
                    f"نجاح {float(short_prof.get('expected_winrate_pct', 0.0)):.1f}% | "
                    f"عائد {float(short_prof.get('expected_pnl_pct', 0.0)):+.2f}% | "
                    f"عينات {int(short_prof.get('samples', 0))}"
                )
            if isinstance(side_prof, dict):
                lines.append(
                    "  ملف الاتجاه الحالي: "
                    f"نجاح {float(side_prof.get('expected_winrate_pct', 0.0)):.1f}% | "
                    f"عائد {float(side_prof.get('expected_pnl_pct', 0.0)):+.2f}% | "
                    f"عينات {int(side_prof.get('samples', 0))}"
                )
            lines.append("")

        if signal.schools_opinion:
            so = signal.schools_opinion or {}
            consensus = so.get('consensus') if isinstance(so.get('consensus'), dict) else {}
            schools = so.get('schools') if isinstance(so.get('schools'), list) else []
            mode = str(so.get('mode', 'advisory_only')).strip().lower()
            consensus_decision = str(consensus.get('decision', 'WAIT')).upper()
            consensus_ar = {
                'LONG': "🟢 شراء",
                'SHORT': "🔴 بيع",
                'WAIT': "⚪ انتظار",
                'BUY': "🟢 شراء",
                'SELL': "🔴 بيع",
            }.get(consensus_decision, "⚪ انتظار")
            agreement_pct = float(consensus.get('agreement_pct', 0.0) or 0.0)
            consensus_conf = int(consensus.get('confidence', 0) or 0)
            long_votes = int(consensus.get('long_votes', 0) or 0)
            short_votes = int(consensus.get('short_votes', 0) or 0)
            total_schools = int(consensus.get('total_schools', len(schools)) or len(schools))
            mode_label = "تحليل استشاري فقط" if mode == "advisory_only" else _escape_html_text(mode)

            lines.append("━━━━ <b>🏫 المدارس (منفصل)</b> ━━━━")
            lines.append(f"  الوضع: {mode_label}")
            lines.append(f"  إجماع المدارس: {consensus_ar} (ثقة {consensus_conf}%)")
            lines.append(f"  التوافق: {agreement_pct:.1f}% | شراء {long_votes} / بيع {short_votes} من {total_schools}")

            for item in schools[:6]:
                if not isinstance(item, dict):
                    continue
                school_name = _escape_html_text(str(item.get('name_ar') or item.get('key') or "مدرسة"))
                decision = str(item.get('decision', 'WAIT')).upper()
                decision_ar = {
                    'LONG': "🟢 شراء",
                    'SHORT': "🔴 بيع",
                    'WAIT': "⚪ انتظار",
                    'BUY': "🟢 شراء",
                    'SELL': "🔴 بيع",
                }.get(decision, "⚪ انتظار")
                confidence = int(item.get('confidence', 0) or 0)
                reason = _escape_html_text(str(item.get('reason', '')).strip())
                if reason:
                    lines.append(f"  - {school_name}: {decision_ar} ({confidence}%) — {reason}")
                else:
                    lines.append(f"  - {school_name}: {decision_ar} ({confidence}%)")
            lines.append("")

        if signal.quality_first_opinion:
            qo = signal.quality_first_opinion or {}
            section_name = _escape_html_text(str(qo.get('name') or "الجودة أهم"))
            q_decision = str(qo.get('decision', 'WAIT')).upper()
            q_decision_ar = {
                'LONG': "🟢 شراء",
                'SHORT': "🔴 بيع",
                'WAIT': "⚪ انتظار",
                'BUY': "🟢 شراء",
                'SELL': "🔴 بيع",
            }.get(q_decision, "⚪ انتظار")
            q_allow = bool(qo.get('allow', False))
            status_ar = "✅ مقبول" if q_allow else "❌ مرفوض"

            live_side = str(signal.side or "WAIT").upper()
            if q_decision in {"LONG", "SHORT"} and live_side in {"LONG", "SHORT"}:
                compare_ar = "✅ متوافق مع الإشارة الحالية" if q_decision == live_side else "⚠️ مختلف عن الإشارة الحالية"
            else:
                compare_ar = "⚪ بدون قرار دخول"

            try:
                score_pct = float(qo.get('score_pct', 0.0))
            except Exception:
                score_pct = 0.0
            passed = int(qo.get('passed_checks', 0) or 0)
            total = int(qo.get('total_checks', 0) or 0)

            label_ar = {
                "adx": "قوة الاتجاه (ADX)",
                "ema_alignment": "توافق الاتجاه (EMA50/EMA200)",
                "zone": "القرب من الدعم/المقاومة",
                "candle": "تأكيد الشمعة",
                "volume": "جودة الحجم",
                "spread": "السبريد",
                "rr": "العائد مقابل المخاطرة",
                "news_window": "نافذة الأخبار القوية",
                "direction": "اتجاه الصفقة",
            }
            checks = qo.get('checks') if isinstance(qo.get('checks'), list) else []
            check_by_key: Dict[str, Dict[str, Any]] = {
                str(c.get('key')): c for c in checks if isinstance(c, dict) and c.get('key')
            }
            failed_required = qo.get('failed_required') if isinstance(qo.get('failed_required'), list) else []

            lines.append(f"━━━━ <b>🏅 قسم {section_name} (مقارنة)</b> ━━━━")
            lines.append(f"  الحالة: {status_ar}")
            lines.append(f"  قرار القسم: {q_decision_ar}")
            lines.append(f"  المقارنة: {compare_ar}")
            if total > 0:
                lines.append(f"  الجودة: {passed}/{total} ({score_pct:.0f}%)")

            if failed_required:
                lines.append("  أسباب الرفض:")
                for key in failed_required[:3]:
                    item = check_by_key.get(str(key), {})
                    label = label_ar.get(str(key), str(key))
                    details = _escape_html_text(str(item.get('details', '')).strip())
                    if details:
                        lines.append(f"  - {label}: {details}")
                    else:
                        lines.append(f"  - {label}")
            elif checks:
                lines.append("  نقاط القوة:")
                strong = [c for c in checks if isinstance(c, dict) and c.get('ok')]
                for item in strong[:3]:
                    key = str(item.get('key'))
                    label = label_ar.get(key, key)
                    details = _escape_html_text(str(item.get('details', '')).strip())
                    if details:
                        lines.append(f"  - {label}: {details}")
                    else:
                        lines.append(f"  - {label}")
            lines.append("")

        if signal.expert_opinion:
            op = signal.expert_opinion
            decision = str(op.get('decision', 'WAIT')).upper()
            decision_ar = {
                'BUY': "🟢 شراء",
                'SELL': "🔴 بيع",
                'WAIT': "⚪ انتظار",
            }.get(decision, "⚪ انتظار")

            confidence = op.get('confidence')
            rationale = _escape_html_text(_to_arabic_reason(op.get('rationale', ''))).strip()
            sl = op.get('stop_loss')
            tp1 = op.get('take_profit_1')
            tp2 = op.get('take_profit_2')

            lines.append("━━━━ <b>🧠 رأي الخبير (40 سنة)</b> ━━━━")
            lines.append(f"  القرار: {decision_ar}")
            if confidence is not None:
                try:
                    lines.append(f"  الثقة: {int(float(confidence))}%")
                except Exception:
                    pass
            if rationale:
                lines.append(f"  السبب: {rationale}")
            if sl is not None:
                lines.append(f"  🛑 وقف: <code>{format_price(float(sl))}</code>")
            if tp1 is not None:
                lines.append(f"  🎯 هدف 1: <code>{format_price(float(tp1))}</code>")
            if tp2 is not None:
                lines.append(f"  🎯 هدف 2: <code>{format_price(float(tp2))}</code>")
            lines.append("")

        if getattr(signal, 'manager_opinion', None):
            mo = signal.manager_opinion or {}
            verdict = str(mo.get('Verdict', 'NO TRADE')).upper()
            verdict_ar = {
                'BUY': "🟢 شراء",
                'SELL': "🔴 بيع",
                'NO TRADE': "⚪ لا صفقة",
                'WAIT': "⚪ انتظار",
            }.get(verdict, "⚪ لا صفقة")

            htf_analysis = _escape_html_text(_to_arabic_reason(mo.get('HTF_Analysis', ''))).strip()
            ltf_analysis = _escape_html_text(_to_arabic_reason(mo.get('LTF_Analysis', ''))).strip()
            zone_quality = _escape_html_text(str(mo.get('Zone_Quality', 'N/A'))).strip()

            reaction_level = mo.get('Reaction_Level')
            confirmation_level = mo.get('Confirmation_Level')
            invalidation_level = mo.get('Invalidation_Level')
            range_high = mo.get('Range_High')
            range_low = mo.get('Range_Low')

            lines.append("━━━━ <b>🧭 قسم المدير</b> ━━━━")
            lines.append(f"  الحكم: {verdict_ar}")
            if htf_analysis:
                lines.append(f"  HTF: {htf_analysis}")
            if ltf_analysis:
                lines.append(f"  LTF: {ltf_analysis}")
            if zone_quality and zone_quality.upper() != 'N/A':
                lines.append(f"  جودة المنطقة: {zone_quality}")
            if reaction_level is not None:
                try:
                    lines.append(f"  📍 مستوى ردّة الفعل: <code>{format_price(float(reaction_level))}</code>")
                except Exception:
                    pass
            if confirmation_level is not None:
                try:
                    lines.append(f"  ✅ مستوى التأكيد: <code>{format_price(float(confirmation_level))}</code>")
                except Exception:
                    pass
            if invalidation_level is not None:
                try:
                    lines.append(f"  🛑 مستوى الإبطال: <code>{format_price(float(invalidation_level))}</code>")
                except Exception:
                    pass
            if range_high is not None and range_low is not None:
                try:
                    lines.append(
                        f"  ↔️ النطاق: <code>{format_price(float(range_low))}</code> → <code>{format_price(float(range_high))}</code>"
                    )
                except Exception:
                    pass
            lines.append("")

        consensus_decision = _get_consensus_decision(signal)
        if consensus_decision == 'buy':
            lines.extend([
                "━━━━ <b>🤝 توافق الأقسام</b> ━━━━",
                "  القرار الموحّد: 🟢 شراء",
                "",
            ])
        elif consensus_decision == 'sell':
            lines.extend([
                "━━━━ <b>🤝 توافق الأقسام</b> ━━━━",
                "  القرار الموحّد: 🔴 بيع",
                "",
            ])

        signal_ts = signal.timestamp
        if getattr(signal_ts, "tzinfo", None) is None:
            signal_ts = signal_ts.replace(tzinfo=timezone.utc)
        if sent_at is None:
            sent_at = datetime.now(timezone.utc)
        age_seconds = max(0.0, (sent_at - signal_ts).total_seconds())

        lines.extend([
            f"⏱ الأطر الزمنية: {self.config.trend_tf} / {self.config.entry_tf} / {self.config.sr_tf}",
            f"🕐 وقت الإشارة: {signal_ts.strftime('%Y-%m-%d %H:%M UTC')}",
            f"⏳ عمر الإشارة عند الإرسال: {age_seconds:.1f} ثانية",
        ])

        return "\n".join(lines)

    def _build_telegram_payload(self, message: str, parse_mode: Optional[str] = "HTML") -> Dict[str, Any]:
        """Build Telegram API payload with optional parse mode."""
        payload: Dict[str, Any] = {
            'chat_id': self.chat_id,
            'text': message,
            'disable_web_page_preview': True,
        }
        if parse_mode:
            payload['parse_mode'] = parse_mode
        return payload

    def _post_telegram_payload(self, payload: Dict[str, Any]) -> requests.Response:
        """Post payload to Telegram sendMessage endpoint."""
        url = self.TELEGRAM_API_URL.format(token=self.bot_token)
        return requests.post(url, json=payload, timeout=30)

    def _split_message_chunks(self, message: str, max_chars: int) -> List[str]:
        """Split long message into safe chunks, preferring line boundaries."""
        if len(message) <= max_chars:
            return [message]

        chunks: List[str] = []
        current = ""

        for line in message.splitlines(keepends=True):
            if len(line) > max_chars:
                if current.strip():
                    chunks.append(current.rstrip("\n"))
                    current = ""
                start = 0
                while start < len(line):
                    piece = line[start:start + max_chars]
                    if piece.strip():
                        chunks.append(piece.rstrip("\n"))
                    start += max_chars
                continue

            if len(current) + len(line) > max_chars:
                if current.strip():
                    chunks.append(current.rstrip("\n"))
                current = line
            else:
                current += line

        if current.strip():
            chunks.append(current.rstrip("\n"))

        return chunks or [message[:max_chars]]

    def _send_telegram_chunks(self, message: str) -> bool:
        """Send long message in multiple chunks."""
        chunks = self._split_message_chunks(message, self.TELEGRAM_SAFE_CHUNK_LENGTH)
        total = len(chunks)

        for idx, chunk in enumerate(chunks, start=1):
            body = chunk
            if total > 1 and idx > 1:
                body = f"<b>متابعة {idx}/{total}</b>\n{chunk}"
            body = body[:self.TELEGRAM_MAX_TEXT_LENGTH]

            try:
                response = self._post_telegram_payload(self._build_telegram_payload(body, parse_mode="HTML"))
                if response.status_code == 200:
                    continue

                response_text = (response.text or "").lower()
                # Fallback: if entity parsing fails, send this chunk as plain text.
                if response.status_code == 400 and "can't parse entities" in response_text:
                    plain = re.sub(r"<[^>]+>", "", body)
                    plain = plain[:self.TELEGRAM_MAX_TEXT_LENGTH]
                    plain_resp = self._post_telegram_payload(self._build_telegram_payload(plain, parse_mode=None))
                    if plain_resp.status_code == 200:
                        continue
                    logger.error(
                        f"Telegram API error (chunk plain fallback): {plain_resp.status_code} - {plain_resp.text}"
                    )
                    return False

                logger.error(f"Telegram API error (chunk {idx}/{total}): {response.status_code} - {response.text}")
                return False
            except requests.RequestException as e:
                logger.error(f"Failed to send Telegram chunk {idx}/{total}: {e}")
                return False

        logger.info(f"Telegram long message sent in {total} chunk(s)")
        return True

    def _send_telegram(self, message: str) -> bool:
        """Send message via Telegram Bot API."""
        try:
            payload = self._build_telegram_payload(message, parse_mode="HTML")
            response = self._post_telegram_payload(payload)

            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True

            response_text = (response.text or "").lower()
            if response.status_code == 400 and "message is too long" in response_text:
                logger.warning("Telegram message too long; retrying as chunks")
                return self._send_telegram_chunks(message)

            # Rare fallback for malformed entities from dynamic text.
            if response.status_code == 400 and "can't parse entities" in response_text:
                logger.warning("Telegram HTML parse error; retrying as plain text")
                plain = re.sub(r"<[^>]+>", "", message)[:self.TELEGRAM_MAX_TEXT_LENGTH]
                plain_resp = self._post_telegram_payload(self._build_telegram_payload(plain, parse_mode=None))
                if plain_resp.status_code == 200:
                    logger.info("Telegram plain-text fallback sent successfully")
                    return True
                logger.error(f"Telegram API error (plain fallback): {plain_resp.status_code} - {plain_resp.text}")
                return False

            logger.error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
                
        except requests.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def _send_telegram_photo(self, photo_path: str, caption: str = "") -> bool:
        """Sends an image to Telegram with an optional caption."""
        if not self.enabled or not self.bot_token or not self.chat_id:
            return False
            
        import os
        if not os.path.exists(photo_path):
            logger.error(f"Cannot send photo: File not found {photo_path}")
            return False
            
        try:
            url = self.TELEGRAM_PHOTO_URL.format(token=self.bot_token)
            
            # Telegram caption limit is 1024 chars. Truncate if necessary.
            safe_caption = caption
            if len(safe_caption) > 1020:
                safe_caption = safe_caption[:1020] + "..."
                
            data = {"chat_id": self.chat_id, "caption": safe_caption, "parse_mode": "HTML"}
            with open(photo_path, "rb") as bf:
                files = {"photo": bf}
                response = requests.post(url, data=data, files=files, timeout=45)
            
            if response.status_code == 200:
                logger.info("Telegram photo sent successfully")
                return True
                
            logger.error(f"Telegram API photo error: {response.status_code} - {response.text}")
            return False
        except requests.RequestException as e:
            logger.error(f"Failed to send Telegram photo: {e}")
            return False

    def fetch_updates(self, offset: Optional[int] = None, timeout: int = 20) -> List[Dict[str, Any]]:
        """
        Poll Telegram updates for command handling.

        Args:
            offset: Telegram update_id offset
            timeout: Long-poll timeout in seconds

        Returns:
            List of update dictionaries
        """
        if not self.enabled:
            return []

        try:
            url = self.TELEGRAM_UPDATES_URL.format(token=self.bot_token)
            params: Dict[str, Any] = {
                'timeout': timeout,
                'allowed_updates': ['message', 'edited_message']
            }
            if offset is not None:
                params['offset'] = offset

            response = requests.get(url, params=params, timeout=timeout + 10)
            response.raise_for_status()
            data = response.json()

            if not data.get('ok'):
                logger.error(f"Telegram getUpdates returned not ok: {data}")
                return []

            result = data.get('result', [])
            return result if isinstance(result, list) else []

        except requests.RequestException as e:
            logger.debug(f"Telegram getUpdates request failed: {e}")
            return []
        except Exception as e:
            logger.debug(f"Telegram getUpdates parse failed: {e}")
            return []
    
    def send_status(self, message: str) -> bool:
        """
        Send a status message.
        
        Args:
            message: Status message text
            
        Returns:
            True if sent successfully
        """
        formatted = f"ℹ️ <b>حالة البوت</b>\n\n{_escape_html_text(message)}"
        
        if self.enabled:
            return self._send_telegram(formatted)
        else:
            print(f"\n[STATUS] {message}\n")
            return True
    
    def send_error(self, error: str, symbol: Optional[str] = None) -> bool:
        """
        Send an error notification.
        
        Args:
            error: Error message
            symbol: Optional symbol context
            
        Returns:
            True if sent successfully
        """
        if symbol:
            formatted = f"⚠️ <b>خطأ ({_escape_html_text(symbol)})</b>\n\n{_escape_html_text(error)}"
        else:
            formatted = f"⚠️ <b>خطأ</b>\n\n{_escape_html_text(error)}"
        
        if self.enabled:
            return self._send_telegram(formatted)
        else:
            print(f"\n[ERROR] {symbol or ''}: {error}\n")
            return True


def create_example_message() -> str:
    """
    Generate an example Telegram message for documentation.
    
    Returns:
        Example formatted message string
    """
    return """
🟢 <b>LONG Signal: BTC/USDT:USDT</b>

📊 <b>Score:</b> 9.5/7 (Tech: 8, Timing: 1.5)
📈 <b>Trend:</b> UP
💰 <b>Price:</b> 97,245.50

━━━━ <b>Levels</b> ━━━━
▫️ Entry: <code>97,250.00</code>
🛑 Stop Loss: <code>95,800.00</code> (1.49%)
⚡ Quick TP: <code>97,590.00</code> (1:0.2)
🎯 TP1: <code>98,700.00</code> (1:1)
🎯 TP2: <code>100,150.00</code> (1:2)

━━━━ <b>Technical Analysis</b> ━━━━
  ✓ Trend: up
  ✓ At support zone
  ✓ Fractal low @ 95,920.00
  ✓ Bullish Engulfing
  ✓ Volume spike

━━━━ <b>Timing Analysis</b> ━━━━
  Gann: above_angles (score: 1.0)
  Sq9: 97,344.00 (0.10%, score: 0.5)

━━━━ <b>Timing Details</b> ━━━━
  Gann: above_angles | Score: 1.0/2
  Sq9: 97,344.00 (0.10% above) | Score: 0.5/2
  Cycle52: 48/52 4 bars | Score: 0.0/1
  Lunar: waxing (62%) | Score: 0.0/1
  FOMC: Jan 29 (5d) | Score: 0.0

⏱ Timeframes: 4h/15m/1h
🕐 2026-02-04 19:15 UTC
""".strip()
