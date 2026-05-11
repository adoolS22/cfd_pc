@echo off
REM Crypto Signal Bot Runner (Windows)
REM Sets environment variables and runs the bot

cd /d "%~dp0"

REM Set environment variables (Use .env instead of hardcoding)
REM TELEGRAM_BOT_TOKEN and OPENAI_API_KEY should be in your .env file

REM Add Python to PATH if not already there
set "PATH=%PATH%;C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311"

echo Starting Crypto Signal Bot...
python main.py

pause</content>
<parameter name="filePath">c:\Users\Administrator.WIN-J2S4568BP74\Desktop\trading\trading\crypto_signal_bot\run_bot.bat