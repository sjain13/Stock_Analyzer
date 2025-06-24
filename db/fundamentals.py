import yfinance as yf
import pandas as pd

def fetch_fundamentals(ticker):
    """
    Fetch P/E and P/B ratio for a given ticker symbol from Yahoo Finance.
    Returns a dictionary: {"PE": value, "PB": value}
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        pe = info.get("trailingPE", None)
        pb = info.get("priceToBook", None)
        return {"PE": pe, "PB": pb}
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")
        return {"PE": None, "PB": None}
    
def get_yahoo_ticker(row):
        exch = str(row.get("exchange", "")).upper()
        symbol = row.get("tradingsymbol", "")
        if exch == "NSE":
            return f"{symbol}.NS"
        elif exch == "BSE":
            return f"{symbol}.BO"
        return symbol or None
    
