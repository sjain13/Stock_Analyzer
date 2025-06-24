import pandas as pd
import pandas_ta as ta
from db.fetch_data import get_price_data

def calculate_indicators(df: pd.DataFrame):
    # RSI
    df["RSI_14"] = ta.rsi(df["close"], length=14)

    # DMAs
    df["DMA_20"] = ta.sma(df["close"], length=20)
    df["DMA_50"] = ta.sma(df["close"], length=50)
    df["DMA_100"] = ta.sma(df["close"], length=100)

    # Ichimoku (uses high, low, close)
    #ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
    # These will be tuples; you may want to select the columns you care about
    #df["ICHIMOKU_CONV"] = ichimoku[0]['ITS_9']        # Conversion line (Tenkan-sen)
    #df["ICHIMOKU_BASE"] = ichimoku[0]['KJS_26']       # Base line (Kijun-sen)
    #df["ICHIMOKU_LEAD_A"] = ichimoku[1]['SSA_26']     # Leading Span A (Senkou Span A)
    #df["ICHIMOKU_LEAD_B"] = ichimoku[1]['SSB_52']     # Leading Span B (Senkou Span B)
    #df["ICHIMOKU_LAGGING"] = ichimoku[2]['CLS_26']    # Lagging Span (Chikou Span)

    # Support (20-day rolling minimum)
    df["SUPPORT_20"] = df["close"].rolling(window=20).min()
    # Resistance (20-day rolling maximum)
    df["RESIST_20"] = df["close"].rolling(window=20).max()

    return df

def get_indicators_for_instrument(instrument_id):
    df = get_price_data(instrument_id)
    if df is None or len(df) < 100:
        print("Not enough data for instrument:", instrument_id)
        return None
    df = calculate_indicators(df)
    return df
