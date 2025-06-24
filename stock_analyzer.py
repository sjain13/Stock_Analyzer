# stock_analyzer.py

import pandas as pd
from db.fetch_data import create_table_if_not_exists, save_signal_to_db, load_instruments
from db.fundamentals import fetch_fundamentals, get_yahoo_ticker
from indicators.indicator_engine import get_indicators_for_instrument
from signals.signals_logic import generate_signal


def main(instrument_mapping):
    instrument_id = int(input("Enter instrument ID: "))
    df = get_indicators_for_instrument(instrument_id)

    if df is None or df.empty:
        print("No data or not enough data for this instrument.")
        return
    
    print("\nLast 10 days with indicators:")
    print(df.tail(10)[["date", "close", "RSI_14", "DMA_20", "DMA_50", "DMA_100"]])

    # Generate signal and reason for every row
    signals, reasons = ["HOLD"], ["No previous row"] 
    for i in range(1, len(df)):
        sig, rea = generate_signal(df.iloc[i], df.iloc[i-1])
        signals.append(sig)
        reasons.append(rea)
            
        signal_data = build_dict_for_saving_data(instrument_id, sig, rea, df.iloc[i])
        save_signal_to_db(signal_data)
                
    df['Signal'] = signals
    df['Reason'] = reasons

    # --- Fundamentals Fetch -----
    symbol = instrument_mapping.get(instrument_id)
    pe, pb = None, None
    if symbol:
        fundamentals = fetch_fundamentals(symbol)
        pe = fundamentals.get("PE", None)
        pb = fundamentals.get("PB", None)
    # Assign PE/PB to every row (they are constant for this instrument_id)
    df['PE'] = pe
    df['PB'] = pb

    # Show latest row's signal (for immediate user feedback)
    print(f"\nLatest Signal: {df.iloc[-1]['Signal']} (Reason: {df.iloc[-1]['Reason']})")

    # ML label generation for entire DataFrame
    N = 5  # lookahead window in days
    buy_threshold = 0.02   # 2% up
    sell_threshold = -0.02 # 2% down

    df['future_close'] = df['close'].shift(-N)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']

    def label_row(ret):
        if ret > buy_threshold:
            return 'BUY'
        elif ret < sell_threshold:
            return 'SELL'
        else:
            return 'HOLD'

    df['ml_label'] = df['future_return'].apply(label_row)
    df_ml = df.dropna(subset=['ml_label'])  # Remove rows with NaN (at end of series)

    print("\nSample ML training data (features and labels):")
    print(df_ml.tail(20)[["date", "close", "RSI_14", "DMA_20", "DMA_50", "DMA_100", "SUPPORT_20", "RESIST_20", "PE", "PB", "Signal", "Reason", "ml_label"]])

    df_ml.to_csv('data/df_ml.csv', index=False)

def build_dict_for_saving_data(instrument_id, sig, rea, row):
    
    return {
            "instrument_signal": sig,
            "created_date": row["date"],
            "extra_info": rea,
            "instrument_id": instrument_id,
            "closing_price": float(row["close"]),
            "daily_20_ma": float(row["DMA_20"]) if "DMA_20" in row and pd.notnull(row["DMA_20"]) else None,
            "daily_50_ma": float(row["DMA_50"]) if "DMA_50" in row and pd.notnull(row["DMA_50"]) else None,
            "support_20": float(row["SUPPORT_20"]) if "SUPPORT_20" in row and pd.notnull(row["SUPPORT_20"]) else None,
            "resist_20": float(row["RESIST_20"]) if "RESIST_20" in row and pd.notnull(row["RESIST_20"]) else None,
            # Add more fields if needed
        }


def load_fundamentals():
    """
    Returns: instrument_mapping (dict) and df_instruments (DataFrame)
    """
    df_instruments = load_instruments()
    df_instruments["yahoo_ticker"] = df_instruments.apply(get_yahoo_ticker, axis=1)
    instrument_mapping = dict(zip(df_instruments['id'], df_instruments['yahoo_ticker']))
    return instrument_mapping, df_instruments

if __name__ == "__main__":
    create_table_if_not_exists()
    instrument_mapping, df_instruments = load_fundamentals()
    main(instrument_mapping)

