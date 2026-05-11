"""
Telegram control command parsing utilities.
"""


def parse_telegram_control_command(text: str) -> str:
    """
    Parse Telegram command text.

    Returns:
        Canonical command name: pause/resume/status/help/stop/news or empty string.
    """
    if not text:
        return ""

    normalized = text.strip().lower()

    # Telegram slash commands (/pause or /pause@botname)
    if normalized.startswith('/'):
        head = normalized.split()[0]
        head = head.split('@')[0]
        mapping = {
            '/pause': 'pause',
            '/off': 'pause',
            '/resume': 'resume',
            '/on': 'resume',
            '/start': 'resume',
            '/status': 'status',
            '/help': 'help',
            '/stop': 'stop',
            '/news': 'news',
        }
        return mapping.get(head, '')

    # Arabic + plain text fallbacks
    if any(token in normalized for token in ('وقف', 'ايقاف', 'اطفي', 'اطف', 'pause')):
        return 'pause'
    if any(token in normalized for token in ('شغل', 'تشغيل', 'resume', 'start')):
        return 'resume'
    if any(token in normalized for token in ('حالة', 'status', 'ستاتس')):
        return 'status'
    if any(token in normalized for token in ('مساعدة', 'help')):
        return 'help'
    if any(token in normalized for token in ('انهي', 'shutdown', 'stop bot')):
        return 'stop'
    if normalized.startswith('news ') or normalized.startswith('خبر ') or normalized.startswith('حلل ') or normalized.startswith('analyze '):
        return 'news'

    return ''
