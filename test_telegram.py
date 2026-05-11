"""Test Telegram connection"""
from bot.notifier import TelegramNotifier
from bot.utils import load_config

config = load_config('config.yaml')
notifier = TelegramNotifier(config)

# Send a test message
result = notifier.send_status('🧪 Test-Nachricht vom Crypto Signal Bot!\n\nDeine Telegram-Verbindung funktioniert einwandfrei ✅')
print(f'Telegram message sent: {result}')
