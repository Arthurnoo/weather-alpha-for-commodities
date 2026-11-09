"""
Weights of each weather grid cell, per commodity and zone.

Crops: share of the zone's production (MapSPAM 2010) falling in the cell.
Natural gas: share of the population of large US metro areas in the cell.

A cell with no production gets no weight, so zones can be drawn wide without
diluting the signal.

Outputs (data/processed/):
  weights.parquet    commodity, zone, box, lat, lon, mass, weight (sums to 1 per zone)
  regions.parquet    commodity, zone, iso3, adm1, mass, share: production per
                     region, to check the zones against official statistics
  land_mask.parquet  box, lat, lon of every land cell of the weather grid,
                     used as map background in the notebook
Run:    python -m src.weights
"""

import numpy as np
import pandas as pd
import xarray as xr

from config import COMMODITIES, DATA_PROCESSED, DATA_RAW, US_METROS


def to_180(lon):
    """Longitudes from [0, 360] (or beyond) to [-180, 180]."""
    return (np.asarray(lon) + 180) % 360 - 180


def cpc_grid(box: str) -> tuple:
    """Latitude and longitude centres of the CPC grid inside a download box."""
    sample = next((DATA_RAW / "cpc" / box).glob("tmax_*.nc"))
    with xr.open_dataset(sample) as ds:
        return ds["lat"].values, to_180(ds["lon"].values)


def snap(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Replace each coordinate by the nearest grid centre."""
    nearest = np.abs(values[:, None] - grid[None, :]).argmin(axis=1)
    return grid[nearest]


def crop_points(spam: pd.DataFrame, crop: str, zone: dict) -> pd.DataFrame:
    """MapSPAM cells of one zone, with their production of the crop."""
    cells = spam[spam["iso3"].isin(zone["iso3"])]
    if "adm1" in zone:
        cells = cells[cells["adm1"].isin(zone["adm1"])]
    if "bbox" in zone:
        lat_min, lat_max, lon_min, lon_max = zone["bbox"]
        cells = cells[cells["lat"].between(lat_min, lat_max)
                      & cells["lon"].between(lon_min, lon_max)]
    return pd.DataFrame({"lat": cells["lat"], "lon": cells["lon"], "mass": cells[crop]})


def metro_points() -> pd.DataFrame:
    """US metro areas, with their population as mass."""
    return pd.DataFrame(US_METROS, columns=["name", "lat", "lon", "mass"]).drop(columns="name")


def build_weights() -> pd.DataFrame:
    spam = pd.read_parquet(DATA_RAW / "spam" / "spam2010_production.parquet")
    rows = []
    for name, spec in COMMODITIES.items():
        for zone_name, zone in spec["zones"].items():
            if not any((DATA_RAW / "cpc" / zone["box"]).glob("tmax_*.nc")):
                print(f"{name:10s} {zone_name:10s} skipped: no weather file for {zone['box']} yet")
                continue
            points = metro_points() if spec["crop"] is None else crop_points(spam, spec["crop"], zone)
            points = points[points["mass"] > 0]

            # Move every point to its CPC cell, then add up the mass per cell
            lats, lons = cpc_grid(zone["box"])
            points = points.assign(lat=snap(points["lat"].values, lats),
                                   lon=snap(points["lon"].values, lons))
            cells = points.groupby(["lat", "lon"], as_index=False)["mass"].sum()
            cells["weight"] = cells["mass"] / cells["mass"].sum()

            cells["commodity"], cells["zone"], cells["box"] = name, zone_name, zone["box"]
            rows.append(cells)
            print(f"{name:10s} {zone_name:10s} {len(cells):5d} cells")

    cols = ["commodity", "zone", "box", "lat", "lon", "mass", "weight"]
    return pd.concat(rows, ignore_index=True)[cols]


def build_regions() -> pd.DataFrame:
    """Production of each zone split by country and first-level region (state, province)."""
    spam = pd.read_parquet(DATA_RAW / "spam" / "spam2010_production.parquet")
    rows = []
    for name, spec in COMMODITIES.items():
        if spec["crop"] is None:  # natural gas: no crop map
            continue
        for zone_name, zone in spec["zones"].items():
            cells = spam[spam["iso3"].isin(zone["iso3"])]
            if "adm1" in zone:
                cells = cells[cells["adm1"].isin(zone["adm1"])]
            if "bbox" in zone:
                lat_min, lat_max, lon_min, lon_max = zone["bbox"]
                cells = cells[cells["lat"].between(lat_min, lat_max)
                              & cells["lon"].between(lon_min, lon_max)]
            by_region = cells.groupby(["iso3", "adm1"], as_index=False)[spec["crop"]].sum()
            by_region = by_region.rename(columns={spec["crop"]: "mass"})
            by_region["share"] = by_region["mass"] / by_region["mass"].sum()
            rows.append(by_region.assign(commodity=name, zone=zone_name))
    cols = ["commodity", "zone", "iso3", "adm1", "mass", "share"]
    return pd.concat(rows, ignore_index=True)[cols]


def build_land_mask() -> pd.DataFrame:
    """Land cells of each download box (CPC only reports values over land)."""
    rows = []
    for box in sorted(p.name for p in (DATA_RAW / "cpc").iterdir() if p.is_dir()):
        sample = sorted((DATA_RAW / "cpc" / box).glob("tmax_*.nc"))[-1]
        with xr.open_dataset(sample) as ds:
            land = ds["tmax"].notnull().any("time").to_dataframe(name="land").reset_index()
        land = land[land["land"]].assign(box=box, lon=lambda d: to_180(d["lon"]))
        rows.append(land[["box", "lat", "lon"]])
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    build_weights().to_parquet(DATA_PROCESSED / "weights.parquet")
    build_regions().to_parquet(DATA_PROCESSED / "regions.parquet")
    build_land_mask().to_parquet(DATA_PROCESSED / "land_mask.parquet")
