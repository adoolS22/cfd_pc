"""
Tests for Telegram runtime control command parsing.
"""

from bot.telegram_control import parse_telegram_control_command


def test_parse_slash_commands():
    assert parse_telegram_control_command("/pause") == "pause"
    assert parse_telegram_control_command("/resume") == "resume"
    assert parse_telegram_control_command("/status") == "status"
    assert parse_telegram_control_command("/help") == "help"
    assert parse_telegram_control_command("/stop") == "stop"


def test_parse_slash_commands_with_botname():
    assert parse_telegram_control_command("/pause@mybot") == "pause"
    assert parse_telegram_control_command("/resume@mybot now") == "resume"


def test_parse_arabic_commands():
    assert parse_telegram_control_command("وقف") == "pause"
    assert parse_telegram_control_command("تشغيل") == "resume"
    assert parse_telegram_control_command("حالة") == "status"
    assert parse_telegram_control_command("مساعدة") == "help"


def test_parse_unknown_command():
    assert parse_telegram_control_command("hello bot") == ""
