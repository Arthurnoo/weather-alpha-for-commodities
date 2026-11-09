"""
Download the USDA Weekly Weather and Crop Bulletin (WWCB) and keep its text.

The WWCB is written every week by USDA and NOAA agricultural meteorologists.
It describes the weather of the past week and its effect on crops, for the US
and for the main producing regions abroad (South America, Europe, Black Sea,
West Africa, Asia). It is released on Tuesdays.

The archive (ESMIS) lists 10 bulletins per page, newest first. For each
bulletin we download the PDF, extract its text and delete the PDF.

Output: data/raw/wwcb/{YYYY-MM-DD}.txt  (release date)
Run:    python -m src.download_wwcb   (safe to rerun: existing texts are skipped)
"""

import io
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader, apply_configuration

from config import DATA_RAW, END_DATE, START_DATE

SITE = "https://esmis.nal.usda.gov"
LISTING = SITE + "/publication/weekly-weather-and-crop-bulletin?page={page}"
OUT_DIR = DATA_RAW / "wwcb"
MAX_ATTEMPTS = 3
# Some 2009 bulletins hold huge maps: pypdf stops parsing after 2 million
# tokens by default, so we give it a larger budget
PDF_TOKEN_BUDGET = 50_000_000


def list_bulletins() -> pd.DataFrame:
    """Walk the listing pages until bulletins are older than START_DATE."""
    rows, page = [], 0
    while True:
        html = requests.get(LISTING.format(page=page), timeout=60).text
        soup = BeautifulSoup(html, "html.parser")
        # Each release is a table row with a <time> tag and a link to its PDF
        dates = []
        for time_tag in soup.find_all("time"):
            block = time_tag.find_parent(["article", "li", "div", "tr"])
            link = block.find("a", href=lambda h: h and h.endswith(".pdf")) if block else None
            if link:
                date = pd.Timestamp(time_tag["datetime"][:10])
                rows.append({"date": date, "url": SITE + link["href"]})
                dates.append(date)
        # Every page also repeats the latest release in a side box, so stop on
        # the OLDEST date of the page, not on the last one read
        if not dates or min(dates) < pd.Timestamp(START_DATE):
            break
        page += 1
    bulletins = pd.DataFrame(rows).drop_duplicates("date")
    return bulletins[bulletins["date"].between(START_DATE, END_DATE)].sort_values("date")


def fetch_text(date: pd.Timestamp, url: str) -> str:
    out = OUT_DIR / f"{date.date()}.txt"
    if out.exists():
        return f"skip {out.name}"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            pdf = requests.get(url, timeout=300).content
            # The configuration is per thread, so it is set inside the worker
            with apply_configuration(stream_parsing_work_maximum_count=PDF_TOKEN_BUDGET):
                text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf)).pages)
            break
        except Exception as error:  # a damaged PDF or a cut connection should not stop the run
            if attempt == MAX_ATTEMPTS:
                return f"FAIL {date.date()}: {error}"
    out.write_text(text, encoding="utf-8")
    return f"ok   {out.name} ({len(text) // 1000}k chars)"


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bulletins = list_bulletins()
    bulletins.to_csv(OUT_DIR / "index.csv", index=False)
    print(f"{len(bulletins)} bulletins from {bulletins['date'].min().date()} to {bulletins['date'].max().date()}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        for status in pool.map(lambda r: fetch_text(r.date, r.url), bulletins.itertuples()):
            print(status, flush=True)
