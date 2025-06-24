# stock_analyzer.py

import pandas as pd
from db.fetch_data import create_table_if_not_exists, save_signal_to_db, load_instruments,get_instruments_for_cal
from db.fundamentals import fetch_fundamentals, get_yahoo_ticker
from indicators.indicator_engine import get_indicators_for_instrument
from signals.signals_logic import generate_signal


def main(instrument_mapping):
    # Step 1: Get instrument list for calculation (IDs, tradingsymbol, name)
    df_instruments, instrument_ids = get_instruments_for_cal(save_to_file=False)
    if df_instruments.empty:
        print("No eligible instruments found. Exiting.")
        return

    # Collect all ML DataFrames here
    ml_dfs = []

    # Step 2: Loop over all instrument IDs
    for idx, row in df_instruments.iterrows():
        instrument_id = row['instrument_id']
        tradingsymbol = row['tradingsymbol']
        name = row['name']
        print(f"\nProcessing {instrument_id}: {tradingsymbol} ({name})")

        df = get_indicators_for_instrument(instrument_id)

        if df is None or df.empty:
            print(f"No data or not enough data for instrument {tradingsymbol}. Skipping.")
            continue

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

        df = add_fundamentals_and_filtered_signals(df, instrument_id, instrument_mapping)

        df_ml = add_ml_labels(df)
        # Add instrument_id, tradingsymbol, and name to the DataFrame for context
        # Insert columns at the start
        df_ml.insert(0, 'instrument_id', instrument_id)
        df_ml.insert(1, 'name', name)
        df_ml.insert(2, 'tradingsymbol', tradingsymbol)
        

        ml_dfs.append(df_ml)

    if ml_dfs:
        final_df_ml = pd.concat(ml_dfs, ignore_index=True)

        # Optional: Ensure these are the first columns (in case .insert isn't enough)
        desired_order = ['instrument_id', 'tradingsymbol', 'name'] + [col for col in final_df_ml.columns if col not in ['instrument_id', 'tradingsymbol', 'name']]
        final_df_ml = final_df_ml[desired_order]

        final_df_ml.to_csv('data/df_ml.csv', index=False)
        print(f"\nSaved combined ML features for all instruments to data/df_ml.csv")
        print(final_df_ml.head(10))
    else:
        print("No ML data to save.")

    print("\nProcessing complete for all eligible instruments.")



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

def add_fundamentals_and_filtered_signals(df, instrument_id, instrument_mapping):
    """
    Adds PE, PB fundamentals and filtered signals (using technical and fundamentals) to the dataframe.
    Returns: DataFrame with added columns ['PE', 'PB', 'Filtered_Signal', 'Filtered_Reason']
    """
    # Fetch fundamentals for this instrument
    symbol = instrument_mapping.get(instrument_id)
    pe, pb = None, None
    if symbol:
        print(f"Instrument ID: {instrument_id}, Yahoo symbol: {symbol}")
        fundamentals = fetch_fundamentals(symbol)
        print(f"Fetched fundamentals for {symbol}: {fundamentals}")
        pe = fundamentals.get("PE", None)
        pb = fundamentals.get("PB", None)
    df['PE'] = pe
    df['PB'] = pb

    # Filter signals using fundamentals
    filtered_signals = []
    filtered_reasons = []
    for i, row in df.iterrows():
        tech_signal = row['Signal']
        this_pe = row['PE']
        this_pb = row['PB']
        filtered_signal, filtered_reason = filtered_buy_signal(tech_signal, this_pe, this_pb)
        filtered_signals.append(filtered_signal)
        filtered_reasons.append(filtered_reason)
    df['Filtered_Signal'] = filtered_signals
    df['Filtered_Reason'] = filtered_reasons
    return df


def filtered_buy_signal(tech_signal, pe, pb, pe_max=40, pb_max=5):
    if tech_signal == "BUY":
        if (pe is not None and pe > pe_max) or (pb is not None and pb > pb_max):
            return "HOLD", "Technical BUY but fundamentals too expensive (PE or PB high)"
        else:
            return "BUY", "Technical BUY confirmed by reasonable fundamentals"
    return tech_signal, "No technical BUY"

def add_ml_labels(df, N=5, buy_threshold=0.02, sell_threshold=-0.02):
    """
    Adds ML labels for stock prediction based on future returns.
    - N: Lookahead window (in days)
    - buy_threshold: Threshold for BUY signal
    - sell_threshold: Threshold for SELL signal

    Returns a new DataFrame (df_ml) with all non-NaN labels.
    """
    df = df.copy()
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
    return df_ml


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

