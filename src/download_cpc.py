"""
Download CPC Global Unified daily weather (Tmax, Tmin, precipitation) for each box.

CPC grids are built from rain gauges and weather stations (observed data, not a
model), at 0.5 deg resolution, from 1979. NOAA publishes one global file per
variable and year (60 to 90 MB). We download it once, cut the five boxes out of
it, and delete the global file, so only the small box files stay on disk.

Output: data/raw/cpc/{box}/{variable}_{year}.nc
Run:    python -m src.download_cpc   (safe to rerun: finished years are skipped)
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import pandas as pd
import requests
import xarray as xr

from config import BOXES, CPC_VARIABLES, DATA_RAW, END_DATE, WEATHER_START

URL = "https://downloads.psl.noaa.gov/Datasets/{path}.{year}.nc"
MAX_ATTEMPTS = 4
# The netCDF library is not thread-safe: downloads run in parallel, file cutting one at a time
NETCDF_LOCK = threading.Lock()


def box_path(box: str, variable: str, year: int):
    return DATA_RAW / "cpc" / box / f"{variable}_{year}.nc"


def cut_box(ds: xr.Dataset, box: str) -> xr.Dataset:
    """Extract one box from a global file (CPC longitudes run from 0 to 360)."""
    lat_min, lat_max, lon_min, lon_max = BOXES[box]
    ds = ds.assign_coords(lon=(ds["lon"] + 180) % 360 - 180).sortby("lon")  # -> [-180, 180]
    lat_slice = slice(lat_max, lat_min) if ds["lat"][0] > ds["lat"][-1] else slice(lat_min, lat_max)
    return ds.sel(lat=lat_slice, lon=slice(lon_min, lon_max))


def download_one(variable: str, year: int) -> str:
    """Download one global (variable, year) file and save every box from it."""
    todo = [box for box in BOXES if not box_path(box, variable, year).exists()]
    if not todo:
        return f"skip {variable}_{year}"

    tmp = DATA_RAW / "cpc" / f"global_{variable}_{year}.nc.part"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    url = URL.format(path=CPC_VARIABLES[variable], year=year)
    # NOAA's server sometimes drops the connection: retry a few times
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as response:
                response.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            break
        except requests.RequestException as error:
            if attempt == MAX_ATTEMPTS:
                tmp.unlink(missing_ok=True)
                return f"FAIL {variable}_{year}: {error}"
            time.sleep(10 * attempt)

    with NETCDF_LOCK, xr.open_dataset(tmp) as ds:
        for box in todo:
            out = box_path(box, variable, year)
            out.parent.mkdir(parents=True, exist_ok=True)
            cut_box(ds[[variable]], box).load().to_netcdf(out)
    tmp.unlink()  # the global file is no longer needed
    return f"ok   {variable}_{year} -> {len(todo)} boxes"


def download_all(workers: int = 3) -> None:
    years = range(pd.Timestamp(WEATHER_START).year, pd.Timestamp(END_DATE).year + 1)
    jobs = list(product(CPC_VARIABLES, years))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status in pool.map(lambda job: download_one(*job), jobs):
            print(status, flush=True)


if __name__ == "__main__":
    download_all()
