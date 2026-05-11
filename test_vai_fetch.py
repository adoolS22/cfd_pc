
import ccxt

def find_vai_symbol(exchange_class, exchange_name):
    try:
        exchange = exchange_class()
        markets = exchange.load_markets()
        print(f"\n🔍 Searching on {exchange_name}...")
        
        found = False
        for symbol in markets:
            if "VAI" in symbol:
                print(f"  - Found: {symbol} ({markets[symbol]['id']})")
                found = True
        
        if not found:
            print(f"  ❌ No symbols containing 'VAI' found on {exchange_name}")
            
    except Exception as e:
        print(f"  ⚠️ Error: {e}")

if __name__ == "__main__":
    find_vai_symbol(ccxt.kucoin, "KuCoin")
    find_vai_symbol(ccxt.gateio, "Gate.io")
