"""
Download the Oceanic Nino Index (ONI) from NOAA CPC.

ONI = 3-month running mean of sea surface temperature anomalies in the Nino 3.4
region of the Pacific. Above +0.5 C: El Nino; below -0.5 C: La Nina.
ENSO is the main driver of year-to-year rainfall swings in the tropics, with
well-documented signs (Ropelewski and Halpert, 1987): El Nino brings drought to
India, Thailand, Colombia, and rain to Argentina and the US southern Plains.
We use it as an independent check of the weather data, not as a model input.

Output: data/raw/enso/oni.txt                  (file as published)
        data/processed/oni_monthly.parquet     one value per month, dated on the
                                               central month of the 3-month season
Run:    python -m src.download_enso
"""

import io

import pandas as pd
import requests

from config import DATA_PROCESSED, DATA_RAW

URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
# Each row is a 3-month season; its central month is the one we date it on
SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
CENTRAL_MONTH = {season: i + 1 for i, season in enumerate(SEASONS)}  # DJF -> January


def download_oni() -> pd.Series:
    text = requests.get(URL, timeout=60).text
    out = DATA_RAW / "enso" / "oni.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

    table = pd.read_csv(io.StringIO(text), sep=r"\s+")  # columns SEAS, YR, TOTAL, ANOM
    dates = pd.to_datetime(dict(year=table["YR"], month=table["SEAS"].map(CENTRAL_MONTH), day=1))
    return pd.Series(table["ANOM"].values, index=dates, name="oni").rename_axis("date")


if __name__ == "__main__":
    oni = download_oni()
    oni.to_frame().to_parquet(DATA_PROCESSED / "oni_monthly.parquet")
    print(f"ONI from {oni.index.min().date()} to {oni.index.max().date()}, {len(oni)} months")
