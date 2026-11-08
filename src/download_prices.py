"""
Download daily closing prices of the 8 futures from Yahoo Finance.

Output: data/raw/prices_daily.parquet, one column per commodity.
Run:    python -m src.download_prices
"""

import pandas as pd
import yfinance as yf

from config import COMMODITIES, DATA_RAW, END_DATE, START_DATE


def download_prices() -> pd.DataFrame:
    """Return a DataFrame of daily closes (rows = dates, columns = commodities)."""
    closes = {}
    for name, spec in COMMODITIES.items():
        # auto_adjust=False: futures have no dividends, we want the raw settlement
        data = yf.download(spec["ticker"], start=START_DATE, end=END_DATE,
                           progress=False, auto_adjust=False)
        closes[name] = data["Close"].squeeze()
        print(f"{name:10s} {spec['ticker']:6s} {len(data)} days")
    prices = pd.DataFrame(closes)
    prices.index.name = "date"
    return prices


if __name__ == "__main__":
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    download_prices().to_parquet(DATA_RAW / "prices_daily.parquet")
