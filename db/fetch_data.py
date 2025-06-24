from sqlalchemy import inspect
import pandas as pd
from db.db_connect import SessionLocal, engine
from models import InstrumentPrice
from models import InstrumentSignal
from sqlalchemy.exc import SQLAlchemyError
from models import Base


def create_table_if_not_exists():
    # Only needed if you're running outside a migration system (for dev/experiments)
    inspector = inspect(engine)
    if 'instrument_signal_data' not in inspector.get_table_names():
        print("Table instrument_signal not found. Creating table...")
        Base.metadata.create_all(engine)
    else:
        print("Table instrument_signal already exists.")

def get_instruments_for_cal(save_to_file: bool = True, file_path: str = "data/common_instruments.csv"):
    """
    Retrieves all unique instrument IDs (with tradingsymbol and name) that are common
    between 'instrument' and 'instrument_signal' tables (i.e., have signals).

    Args:
        save_to_file (bool): Whether to save the resulting DataFrame to CSV.
        file_path (str): Path to save the CSV file.

    Returns:
        df (pd.DataFrame): DataFrame with instrument_id, tradingsymbol, name.
        instrument_ids (list): List of instrument IDs.
    """
    query = """
        SELECT DISTINCT i.id AS instrument_id, i.tradingsymbol, i.name
        FROM instrument i
        INNER JOIN instrument_signal s ON i.id = s.instrument_id;
    """
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            print("No common instruments found.")
            return df, []
        if save_to_file:
            df.to_csv(file_path, index=False)
            print(f"Saved common instrument list to {file_path}")
        instrument_ids = df['instrument_id'].tolist()
        return df, instrument_ids
    except Exception as e:
        print("Error fetching instruments for calculation:", e)
        return pd.DataFrame(), []
        
def get_price_data(instrument_id, n_days=120):
    """
    Fetches last n_days of price data for a given instrument_id from instrument_price table.
    Returns a pandas DataFrame with columns: date, close.
    """
    session = SessionLocal()
    try:
        prices = (
            session.query(InstrumentPrice)
            .filter(InstrumentPrice.instrument_id == instrument_id)
            .order_by(InstrumentPrice.updated_date_time.desc())
            .limit(n_days)
            .all()
        )
        if not prices:
            return None
        prices = prices[::-1]  # chronological order
        df = pd.DataFrame([{
            "date": p.updated_date_time,
            "close": float(p.closing_price),
            "volume": p.volume if p.volume is not None else 0
        } for p in prices])
        return df
    finally:
        session.close()

def save_signal_to_db(signal_data):
    """
    Saves a signal entry to the instrument_signal table.
    signal_data: dict with keys matching InstrumentSignal columns.
    Example keys: instrument_signal, created_date, extra_info, instrument_id, closing_price, daily_20_ma, daily_50_ma, etc.
    """
    session = SessionLocal()
    try:
        new_signal = InstrumentSignal(**signal_data)
        session.add(new_signal)
        session.commit()
    except SQLAlchemyError as e:
        print("Error saving signal:", e)
        session.rollback()
    finally:
        session.close()

def load_instruments():
    query = "SELECT * FROM instrument"
    df = pd.read_sql(query, engine)
    return df