"""
From daily CPC grids to one weekly weather table per commodity and zone.

Steps, for each (commodity, zone):
  1. read the daily Tmax, Tmin, precipitation of the zone's cells, year by year
  2. compute the daily stress variables cell by cell (before averaging, so that
     a heat wave in one region is not smoothed away by cool cells elsewhere)
  3. average the cells with the production weights
  4. aggregate days into weeks ending on Friday

Output: data/processed/weather_weekly.parquet
        columns: date, commodity, zone, variable, value  (long format)
Run:    python -m src.weather
"""

import numpy as np
import pandas as pd
import xarray as xr

from config import (COMMODITIES, DATA_PROCESSED, DATA_RAW, DEGREE_DAY_BASE,
                    END_DATE, WEATHER_START, WEEK_RULE)
from src.weights import to_180

HEAVY_RAIN_MM = 20.0   # a day with 20 mm or more counts as a heavy-rain day
HARD_FREEZE_C = -10.0  # gas: very cold nights drive demand spikes and well freeze-offs

# How each daily variable becomes weekly: totals for amounts and counts,
# averages for temperature levels
WEEKLY_AGG = {
    "tmax": "mean", "tmin": "mean", "tavg": "mean",
    "precip": "sum", "et0": "sum", "water_balance": "sum", "heavy_rain_days": "sum",
    "edd": "sum", "frost_dd": "sum",
    "hdd": "sum", "cdd": "sum", "hard_freeze_days": "sum",
}


# ---------------------------------------------------------------------------
# Reading the grids
# ---------------------------------------------------------------------------
def read_cells(box: str, variable: str, lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
    """Daily series of one variable at a list of cells (rows = days, columns = cells)."""
    years = range(pd.Timestamp(WEATHER_START).year, pd.Timestamp(END_DATE).year + 1)
    chunks = []
    for year in years:
        with xr.open_dataset(DATA_RAW / "cpc" / box / f"{variable}_{year}.nc") as ds:
            ds = ds.assign_coords(lon=to_180(ds["lon"].values))
            # Point-wise selection: cell i = (lats[i], lons[i])
            values = ds[variable].sel(lat=xr.DataArray(lats, dims="cell"),
                                      lon=xr.DataArray(lons, dims="cell"))
            chunks.append(values.to_pandas())
    return pd.concat(chunks)


# ---------------------------------------------------------------------------
# Daily variables
# ---------------------------------------------------------------------------
def extraterrestrial_radiation(lat_deg: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """
    Solar radiation at the top of the atmosphere, in mm/day of evaporation
    equivalent (FAO-56, equations 21 to 25). Depends only on latitude and date.
    Returns an array (days x cells).
    """
    phi = np.radians(lat_deg)[None, :]
    j = day_of_year[:, None]
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)          # Earth-Sun distance
    delta = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)     # solar declination
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))  # sunset angle
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    return 0.408 * ra  # MJ/m2/day -> mm/day


def daily_variables(tmax, tmin, precip, lats, spec) -> dict:
    """
    Daily stress variables per cell (each value: DataFrame days x cells).

    edd            extreme degree days: degrees above the crop's heat threshold
    frost_dd       frost degree days: degrees below the crop's frost threshold
    et0            reference evapotranspiration (Hargreaves): water the air takes
    water_balance  precipitation minus et0; negative = soil drying out
    heavy_rain_days, hdd, cdd, hard_freeze_days: see constants above
    """
    tavg = (tmax + tmin) / 2
    ra = extraterrestrial_radiation(lats, tmax.index.dayofyear.values)
    # Hargreaves (FAO-56 eq. 52): needs only temperatures, robust with daily data
    et0 = 0.0023 * ra * (tavg + 17.8) * np.sqrt((tmax - tmin).clip(lower=0))

    out = {"tmax": tmax, "tmin": tmin, "tavg": tavg, "precip": precip,
           "et0": et0, "water_balance": precip - et0,
           "heavy_rain_days": (precip >= HEAVY_RAIN_MM).astype(float).where(precip.notna())}
    if spec["crop"] is None:  # natural gas: demand side
        out["hdd"] = (DEGREE_DAY_BASE - tavg).clip(lower=0)
        out["cdd"] = (tavg - DEGREE_DAY_BASE).clip(lower=0)
        out["hard_freeze_days"] = (tmin <= HARD_FREEZE_C).astype(float).where(tmin.notna())
    else:                     # crops: supply side
        out["edd"] = (tmax - spec["heat_t"]).clip(lower=0)
        out["frost_dd"] = (spec["frost_t"] - tmin).clip(lower=0)
    return out


def weighted_mean(values: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Weighted average across cells, ignoring missing cells on each day."""
    w = pd.DataFrame(np.broadcast_to(weights, values.shape), index=values.index,
                     columns=values.columns).where(values.notna())
    return (values * w).sum(axis=1) / w.sum(axis=1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def zone_weekly(commodity: str, zone: str, cells: pd.DataFrame) -> pd.DataFrame:
    """Weekly weather of one zone (rows = weeks, columns = variables)."""
    spec = COMMODITIES[commodity]
    box = cells["box"].iloc[0]
    lats, lons, weights = cells["lat"].values, cells["lon"].values, cells["weight"].values

    raw = {v: read_cells(box, v, lats, lons) for v in ["tmax", "tmin", "precip"]}
    daily = daily_variables(raw["tmax"], raw["tmin"], raw["precip"], lats, spec)
    zone_daily = pd.DataFrame({name: weighted_mean(df, weights) for name, df in daily.items()})
    zone_daily = zone_daily.loc[WEATHER_START:END_DATE]

    agg = {v: WEEKLY_AGG[v] for v in zone_daily.columns}
    return zone_daily.resample(WEEK_RULE).agg(agg)


def box_is_complete(box: str) -> bool:
    """True when every (variable, year) file of the box has been downloaded."""
    years = range(pd.Timestamp(WEATHER_START).year, pd.Timestamp(END_DATE).year + 1)
    return all((DATA_RAW / "cpc" / box / f"{v}_{y}.nc").exists()
               for v in ["tmax", "tmin", "precip"] for y in years)


def build_weather() -> pd.DataFrame:
    weights = pd.read_parquet(DATA_PROCESSED / "weights.parquet")
    frames = []
    for (commodity, zone), cells in weights.groupby(["commodity", "zone"], sort=False):
        if not box_is_complete(cells["box"].iloc[0]):
            print(f"{commodity:10s} {zone:10s} skipped: weather download not finished")
            continue
        weekly = zone_weekly(commodity, zone, cells)
        long = weekly.rename_axis("date").reset_index().melt(id_vars="date", var_name="variable")
        frames.append(long.assign(commodity=commodity, zone=zone))
        print(f"{commodity:10s} {zone:10s} {len(weekly)} weeks", flush=True)
    return pd.concat(frames, ignore_index=True)[["date", "commodity", "zone", "variable", "value"]]


if __name__ == "__main__":
    build_weather().to_parquet(DATA_PROCESSED / "weather_weekly.parquet")
