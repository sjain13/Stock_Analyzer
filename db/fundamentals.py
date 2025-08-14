import math
import os
import time
import yfinance as yf
import pandas as pd

CACHE_FILE = "data/fundamental_cache.csv"

def load_cache():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE).set_index("symbol").to_dict("index")
    return {}

def save_cache(cache):
    df = pd.DataFrame.from_dict(cache, orient="index")
    df.index.name = "symbol"
    df.reset_index().to_csv(CACHE_FILE, index=False)

def fetch_fundamentals(symbol, cache, max_retries=3, delay=10):
    if symbol in cache and cache[symbol]["PE"] is not None and cache[symbol]["PB"] is not None:
        return cache[symbol]
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            pe = safe_float(info.get("trailingPE"))
            pb = safe_float(info.get("priceToBook"))
            cache[symbol] = {"PE": pe, "PB": pb}
            if pe is None or pb is None:
                print(f"Warning: Missing PE or PB for {symbol}")
            return cache[symbol]
        except Exception as e:
            print(f"Error fetching fundamentals for {symbol}: {e}")
            if "Too Many Requests" in str(e):
                print(f"Sleeping {delay} seconds due to rate limit...")
                time.sleep(delay)
            else:
                break
    # Fallback
    cache[symbol] = {"PE": None, "PB": None}
    return cache[symbol]

def safe_float(val):
    try:
        if val is None:
            return None
        # Convert 'Infinity', 'inf', 'NaN' etc to None
        if isinstance(val, str):
            if val.lower() in ["inf", "infinity", "nan"]:
                return None
        f = float(val)
        if math.isinf(f) or math.isnan(f):
            return None
        return f
    except Exception:
        return None
    
def get_yahoo_ticker(row):
        exch = str(row.get("exchange", "")).upper()
        symbol = row.get("tradingsymbol", "")
        if exch == "NSE":
            return f"{symbol}.NS"
        elif exch == "BSE":
            return f"{symbol}.BO"
        return symbol or None
    
