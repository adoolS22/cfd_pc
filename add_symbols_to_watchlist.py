"""
Run this once to add all required symbols to the MT5 Market Watch.
"""
import os
from dotenv import load_dotenv
import MetaTrader5 as mt5

load_dotenv(".env")

REQUIRED_SYMBOLS = [
    "BTCUSDm", "ETHUSDm",
    "EURUSDm", "GBPUSDm", "USDJPYm", "USDCHFm",
    "AUDUSDm", "USDCADm", "NZDUSDm",
    "XAUUSDm", "XAGUSDm", "USOILm", "US500m",
]

login = int(os.getenv("MT5_LOGIN", "0"))
password = os.getenv("MT5_PASSWORD", "")
server = os.getenv("MT5_SERVER", "")

if not mt5.initialize():
    print(f"initialize() failed: {mt5.last_error()}")
    exit(1)

if not mt5.login(login=login, password=password, server=server):
    print(f"login() failed: {mt5.last_error()}")
    mt5.shutdown()
    exit(1)

print(f"Connected: login={login} server={server}")
print()

for sym in REQUIRED_SYMBOLS:
    info = mt5.symbol_info(sym)
    if info is None:
        print(f"  NOT FOUND: {sym}")
        continue
    ok = mt5.symbol_select(sym, True)
    status = "OK" if ok else "FAILED"
    print(f"  {status}: {sym}")

mt5.shutdown()
print("\nDone. Restart the bot now.")
