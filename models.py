from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, BigInteger, String, Integer, DateTime, ForeignKey, DECIMAL, Text, CHAR
from sqlalchemy.orm import relationship

Base = declarative_base()

class Instrument(Base):
    __tablename__ = 'instrument'
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    instrument_token = Column(BigInteger)
    exchange_token = Column(BigInteger)
    tradingsymbol = Column(String(45))
    name = Column(String(255))
    last_price = Column(String(45))
    tick_size = Column(String(45))
    instrument_type = Column(String(255))
    segment = Column(String(45))
    exchange = Column(String(45))
    strike = Column(String(45))
    lot_size = Column(Integer)
    expiry = Column(DateTime)
    buy_sell_signal = Column(String(10))
    buy_sell_signal_ma_cross = Column(String(10))
    watch_list = Column(CHAR(1), default='Y')
    my_buy_price = Column(DECIMAL(10,2))

    prices = relationship("InstrumentPrice", back_populates="instrument")
    signals = relationship("InstrumentSignal", back_populates="instrument")

class InstrumentPrice(Base):
    __tablename__ = 'instrument_price'
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    instrument_id = Column(BigInteger, ForeignKey('instrument.id'), nullable=False, index=True)
    closing_price = Column(DECIMAL(10,2), nullable=False)
    updated_date_time = Column(DateTime, nullable=False)
    volume = Column(BigInteger)
    oi = Column(BigInteger)
    buy_sell_signal = Column(String(45))
    buy_sell_signal_ma_cross = Column(String(45))

    instrument = relationship("Instrument", back_populates="prices")

class InstrumentSignal(Base):
    __tablename__ = 'instrument_signal_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_signal = Column(String(50), nullable=False)
    created_date = Column(DateTime, nullable=False)
    extra_info = Column(Text, nullable=False)
    instrument_id = Column(BigInteger, ForeignKey('instrument.id'), nullable=False, index=True)
    closing_price = Column(DECIMAL(10,2), nullable=False)
    percentage_diff = Column(String(45))
    end_result = Column(String(10))
    end_result_date = Column(DateTime)
    weekly_9_ma = Column(DECIMAL(10,2))
    weekly_26_ma = Column(DECIMAL(10,2))
    daily_20_ma = Column(DECIMAL(10,2))
    daily_50_ma = Column(DECIMAL(10,2))
    support_20 = Column(DECIMAL(10,2))
    resist_20 = Column(DECIMAL(10,2)) 

    instrument = relationship("Instrument", back_populates="signals")
