import numpy as np
import pandas as pd

def generate_signal(latest_row, prev_row=None):
    """
    Rule-based signal generator based on most recent row with indicators.
    Returns: "BUY", "SELL", or "HOLD" and a reason string.
    """
    rsi = latest_row["RSI_14"]
    price = latest_row["close"]
    dma_20 = latest_row["DMA_20"]
    dma_50 = latest_row["DMA_50"]
    dma_100 = latest_row["DMA_100"]
    support = latest_row.get("SUPPORT_20")
    resist = latest_row.get("RESIST_20")

    # NaN checks
    indicators = [rsi, dma_20, dma_50, dma_100, support, resist]
    if any(pd.isnull(x) for x in indicators):
        return "HOLD", "Insufficient indicator data"

    # Rule 1: Classic RSI + Trend
    if rsi < 30 and price > dma_20 and price > dma_50 and price > dma_100:
        return "BUY", "RSI<30 and price above all DMAs"

    if rsi > 70 and price < dma_20 and price < dma_50 and price < dma_100:
        return "SELL", "RSI>70 and price below all DMAs"

    # Rule 2: DMA Crossover (need previous row)
    if prev_row is not None:
        prev_dma20, prev_dma50 = prev_row["DMA_20"], prev_row["DMA_50"]
        # Golden Cross
        if prev_dma20 < prev_dma50 and dma_20 > dma_50:
            return "BUY", "DMA-20 crossed above DMA-50 (Golden Cross)"
        # Death Cross
        if prev_dma20 > prev_dma50 and dma_20 < dma_50:
            return "SELL", "DMA-20 crossed below DMA-50 (Death Cross)"

    # Rule 3: Support/Resistance breakout
    if price > resist:
        return "BUY", "Price broke above 20-day resistance"
    if price < support:
        return "SELL", "Price broke below 20-day support"

    # Rule 4: Trend confirmation
    if price > dma_20 > dma_50 > dma_100:
        return "BUY", "Strong uptrend (price > DMA20 > DMA50 > DMA100)"
    if price < dma_20 < dma_50 < dma_100:
        return "SELL", "Strong downtrend (price < DMA20 < DMA50 < DMA100)"

    return "HOLD", "No strong signal"

