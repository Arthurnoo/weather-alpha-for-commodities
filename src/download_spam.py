"""
Download MapSPAM 2010 global crop production (IFPRI, Harvard Dataverse).

MapSPAM spreads national and sub-national crop statistics on a 5 arc-minute
grid (about 10 km). We use it to know how much of each crop is grown in each
weather cell, which gives the weights of the weather aggregation.

Output: data/raw/spam/spam2010_production.parquet
        columns: iso3, adm1, lat, lon + one column per crop (metric tons)
Run:    python -m src.download_spam
"""

import zipfile

import pandas as pd
import requests

from config import COMMODITIES, DATA_RAW

# Dataverse file id of "spam2010v2r0_global_prod.csv.zip" (doi:10.7910/DVN/PRFF8V)
SPAM_PRODUCTION_FILE_ID = 3984975
URL = f"https://dataverse.harvard.edu/api/access/datafile/{SPAM_PRODUCTION_FILE_ID}"


ZIP_PATH = DATA_RAW / "spam" / "spam2010_prod_csv.zip"


def download_spam() -> pd.DataFrame:
    crops = sorted({spec["crop"] for spec in COMMODITIES.values() if spec["crop"]})
    if not ZIP_PATH.exists():  # 150 MB: keep the archive to avoid downloading it twice
        response = requests.get(URL, timeout=600)
        response.raise_for_status()
        ZIP_PATH.write_bytes(response.content)

    with zipfile.ZipFile(ZIP_PATH) as archive:
        # One CSV per farming technology; "_TA" = all technologies together
        with archive.open("spam2010V2r0_global_P_TA.csv") as f:
            keep = {"iso3": "iso3", "name_adm1": "adm1", "y": "lat", "x": "lon"}
            # Production columns are named like "maiz_a" (crop code + "all technologies")
            keep.update({f"{crop}_a": crop for crop in crops})
            raw = pd.read_csv(f, usecols=list(keep), encoding="latin-1")

    spam = raw.rename(columns=keep)
    spam["iso3"] = spam["iso3"].str.upper()  # file uses "usa", config uses "USA"
    # Drop cells where none of our crops is grown: the file shrinks a lot
    return spam[spam[crops].sum(axis=1) > 0].reset_index(drop=True)


if __name__ == "__main__":
    out = DATA_RAW / "spam" / "spam2010_production.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    spam = download_spam()
    spam.to_parquet(out)
    print(f"{len(spam)} cells saved to {out}")
