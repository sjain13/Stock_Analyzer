import time
import yfinance as yf
import pandas as pd

def fetch_fundamentals(symbol, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            pe = info.get("trailingPE")
            pb = info.get("priceToBook")
            if pe is None or pb is None:
                print(f"Warning: Missing PE or PB for {symbol}")
            return {"PE": pe, "PB": pb}
        except Exception as e:
            print(f"Error fetching fundamentals for {symbol}: {e}")
            if "Too Many Requests" in str(e):
                print(f"Sleeping {delay} seconds due to rate limit...")
                time.sleep(delay)
            else:
                break
    # Fallback if still failing
    return {"PE": None, "PB": None}
    
def get_yahoo_ticker(row):
        exch = str(row.get("exchange", "")).upper()
        symbol = row.get("tradingsymbol", "")
        if exch == "NSE":
            return f"{symbol}.NS"
        elif exch == "BSE":
            return f"{symbol}.BO"
        return symbol or None
    
