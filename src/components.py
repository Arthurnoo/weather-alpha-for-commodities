"""
Weekly stress components per commodity, oriented so that higher = more stress.

For each zone:
  1. build raw components from the weekly weather
  2. turn them into anomalies (z-scores) against the climatology of the same
     weeks of the year, using PAST years only (no look-ahead)
  3. multiply by the season mask (1 in the zone's growing season, else 0)
Then zones are added up with their share of production.

Output: data/processed/components_weekly.parquet
        index (date, commodity), one column per component
Run:    python -m src.components
"""

import numpy as np
import pandas as pd

from config import COMMODITIES, DATA_PROCESSED

# Components built from the weekly weather table. Each entry: (name, function).
# The sign makes "high" mean "bad for supply" (crops) or "high demand" (gas).
CROP_COMPONENTS = {
    "heat": lambda w: w["edd"],                                 # degree days above heat threshold
    "warm_nights": lambda w: w["tmin"],                         # night heat stops crops recovering
    "dryness": lambda w: -w["water_balance"].rolling(4).sum(),  # 4-week water deficit
    "heavy_rain": lambda w: w["heavy_rain_days"],               # flooding, planting / harvest delays
    "frost": lambda w: w["frost_dd"],                           # degree days below frost threshold
}
GAS_COMPONENTS = {
    "cold": lambda w: w["hdd"],                    # heating demand
    "heat": lambda w: w["cdd"],                    # cooling (power burn) demand
    "hard_freeze": lambda w: w["hard_freeze_days"],  # demand spikes + production freeze-offs
}

CLIMATOLOGY_HALF_WINDOW = 2  # pool +/- 2 weeks of the year to stabilise the climatology
MIN_PAST_YEARS = 5           # no anomaly before 5 years of history exist


def expanding_anomaly(x: pd.Series) -> pd.Series:
    """
    z-score of each week against the same weeks of the year in PREVIOUS years.

    Week w of year Y is compared with weeks w-2..w+2 of all years < Y.
    The anomaly therefore only uses information available at the time.
    """
    year = x.index.year
    woy = np.minimum(x.index.isocalendar().week.values, 52)  # week 53 -> 52
    z = pd.Series(np.nan, index=x.index)

    for y in np.unique(year):
        past_x, past_woy = x[year < y], woy[year < y]
        if len(np.unique(year[year < y])) < MIN_PAST_YEARS:
            continue
        # Climatology table: mean and std for each of the 52 weeks of the year
        mean, std = np.full(53, np.nan), np.full(53, np.nan)
        for w in range(1, 53):
            gap = np.abs(past_woy - w)
            in_window = np.minimum(gap, 52 - gap) <= CLIMATOLOGY_HALF_WINDOW  # circular
            mean[w], std[w] = past_x[in_window].mean(), past_x[in_window].std()

        now = year == y
        m, s = mean[woy[now]], std[woy[now]]
        # std = 0 happens when the event never occurred (e.g. no frost in summer):
        # the anomaly is then 0 instead of a division by zero
        no_spread = np.where(np.isnan(s), np.nan, 0.0)
        z[now] = np.where(s > 0, (x[now].values - m) / np.where(s > 0, s, 1), no_spread)
    return z


def zone_components(weather: pd.DataFrame, is_gas: bool) -> pd.DataFrame:
    """Anomalies of each component for one zone (rows = weeks)."""
    recipes = GAS_COMPONENTS if is_gas else CROP_COMPONENTS
    raw = pd.DataFrame({name: f(weather) for name, f in recipes.items()})
    return raw.apply(expanding_anomaly)


def build_components(use_season: bool = True) -> pd.DataFrame:
    weather = pd.read_parquet(DATA_PROCESSED / "weather_weekly.parquet")
    weights = pd.read_parquet(DATA_PROCESSED / "weights.parquet")
    # Production of each zone, used to add zones up (e.g. soybeans US / BRA / ARG)
    zone_mass = weights.groupby(["commodity", "zone"])["mass"].sum()

    available = set(weather.groupby(["commodity", "zone"]).groups)
    frames = []
    for name, spec in COMMODITIES.items():
        total = None
        shares = zone_mass[name] / zone_mass[name].sum()
        for zone_name, zone in spec["zones"].items():
            if (name, zone_name) not in available:  # weather not downloaded yet
                continue
            w = (weather.query("commodity == @name and zone == @zone_name")
                 .pivot(index="date", columns="variable", values="value"))
            comp = zone_components(w, is_gas=spec["crop"] is None)
            if use_season:
                in_season = comp.index.month.isin(zone["season"]).astype(float)
                comp = comp.mul(in_season, axis=0)
            comp = comp * shares[zone_name]
            total = comp if total is None else total.add(comp, fill_value=0)
        if total is None:
            print(f"{name:10s} skipped: no weather yet")
            continue
        frames.append(total.assign(commodity=name))
        print(f"{name:10s} {list(total.columns)}", flush=True)

    out = pd.concat(frames).rename_axis("date").reset_index()
    return out.set_index(["date", "commodity"]).sort_index()


if __name__ == "__main__":
    build_components().to_parquet(DATA_PROCESSED / "components_weekly.parquet")
