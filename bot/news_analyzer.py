"""
OpenAI-powered news impact analyzer for Telegram commands.
"""

from loguru import logger
from openai import OpenAI
from openai import OpenAIError

from .utils import OpenAIConfig

NEWS_ANALYSIS_PROMPT = """أنت محلل أسواق ومتداول محترف بخبرة 30+ سنة في الفوركس، الكريبتو، والأسهم.

المطلوب:
حلّل الخبر بطريقة عملية ومباشرة، وباللغة العربية فقط، ثم أعطِ رأيًا تداوليًا واضحًا.

قواعد صارمة:
1) اكتب بالعربية فقط.
2) لا تعطِ وعودًا أو يقينًا مطلقًا.
3) ميّز بين: شراء / بيع / انتظار.
4) اربط الحكم بالعوامل الواقعية: توقعات السوق، المفاجأة في الخبر، السيولة، والتذبذب.
5) إذا الخبر غير كافٍ لاتخاذ قرار، اذكر ذلك بوضوح واختر "انتظار".

هيكل الجواب (التزم به):
- نوع الخبر:
- السوق/الأصول المتأثرة:
- قوة التأثير: (ضعيف/متوسط/قوي/عنيف)
- قراءة الخبر مقارنة بالتوقعات: (أفضل/أسوأ/مطابق للتوقعات)
- الانعكاس المتوقع على السعر: (صعود/هبوط/تذبذب)
- رؤية المتداول الخبير (30+ سنة): سطرين إلى ثلاثة كحد أقصى.
- القرار التداولي النهائي: (شراء / بيع / انتظار)
- درجة الثقة: (من 100)
- إدارة المخاطر المقترحة: (وقف خسارة، تخفيف حجم الصفقة، انتظار تأكيد)
- ملاحظة تحذيرية قصيرة:
"""


class NewsAnalyzer:
    """Analyze free-form news text using OpenAI."""

    def __init__(self, config: OpenAIConfig):
        self.enabled = config.enabled
        self.model = config.model
        self.api_key = config.api_key
        self.base_url = config.base_url if hasattr(config, "base_url") else None

    def analyze(self, news_text: str) -> str:
        """Analyze a news headline/body and return structured market impact text."""
        if not news_text or not news_text.strip():
            return (
                "اكتب خبرًا أو عنوانًا بعد الأمر مباشرة.\n\n"
                "مثال:\n"
                "/news الفيدرالي خفّض الفائدة بشكل مفاجئ"
            )

        if not self.enabled:
            return (
                "تحليل الأخبار عبر OpenAI غير مفعّل.\n"
                "فعّل قسم openai في config.yaml."
            )

        if not self.api_key:
            return (
                "مفتاح OpenAI غير موجود.\n"
                "أضف OPENAI_API_KEY في ملف .env ثم أعد تشغيل البوت."
            )

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=120,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": NEWS_ANALYSIS_PROMPT},
                    {"role": "user", "content": f"News input:\n{news_text.strip()}"},
                ],
                temperature=0.1,
                max_tokens=700,
            )
            output = (response.choices[0].message.content or "").strip()

            if output:
                return output

            logger.warning("OpenAI returned empty news analysis output")
            return "تم تنفيذ التحليل لكن ما رجع محتوى. جرّب خبر أوضح."

        except OpenAIError as e:
            logger.error(f"OpenAI news analysis failed: {e}")
            return (
                "فشل تحليل الخبر عبر OpenAI.\n"
                "تأكد من OPENAI_API_KEY ومن اسم الموديل في config.yaml."
            )
