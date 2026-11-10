"""
Weekly stress components per commodity, oriented so that higher = more stress.

For each commodity and region:
  1. Construct physical stress indicators from weekly weather.
  2. Define the regional growing-season mask.
  3. Estimate seasonal climatology using past in-season observations only.
  4. Calculate past-only z-scores.
  5. Set off-season component values to zero.
  6. Aggregate regions using their production shares.

Output: data/processed/components_weekly.parquet
"""

import numpy as np
import pandas as pd

from config import COMMODITIES, DATA_PROCESSED


# ---------------------------------------------------------------------------
# Physical stress indicators
# ---------------------------------------------------------------------------

CROP_COMPONENTS = {
    "heat": lambda w: w["edd"],
    "warm_nights": lambda w: w["tmin"],
    "heavy_rain": lambda w: w["heavy_rain_days"],
    "frost": (lambda w: (w["frost_dd"] > 0).astype(float).where(w["frost_dd"].notna())),
}

GAS_COMPONENTS = {
    "cold": lambda w: w["hdd"],
    "heat": lambda w: w["cdd"],
    "hard_freeze": lambda w: w["hard_freeze_days"],
}

CLIMATOLOGY_HALF_WINDOW = 2
MIN_PAST_YEARS = 5


# ---------------------------------------------------------------------------
# Historical seasonal standardisation
# ---------------------------------------------------------------------------

def expanding_anomaly(
    x: pd.Series,
    in_season: pd.Series,
) -> pd.Series:
    """
    Calculate a past-only seasonal z-score.

    For a given week of year, compare its value with observations from
    +/- CLIMATOLOGY_HALF_WINDOW weeks in previous years.

    Only valid in-season observations are used to estimate the historical
    mean and standard deviation.

    Off-season values are set to zero. Missing in-season values remain NaN.
    """

    years = x.index.year.to_numpy()
    weeks = np.minimum(
        x.index.isocalendar().week.to_numpy(),
        52
    )
    active = in_season.to_numpy(dtype=bool)

    z = pd.Series(np.nan, index=x.index, dtype=float)

    for year in np.unique(years):

        # Only previous years and only in-season observations.
        past_mask = (years < year) & active & x.notna().to_numpy()

        if len(np.unique(years[past_mask])) < MIN_PAST_YEARS:
            continue

        mean = np.full(53, np.nan)
        std = np.full(53, np.nan)

        past_values = x.to_numpy()[past_mask]
        past_weeks = weeks[past_mask]
        past_years = years[past_mask]

        for week in range(1, 53):

            gap = np.abs(past_weeks - week)

            # Circular distance handles the beginning/end of the year.
            in_window = (
                np.minimum(gap, 52 - gap)
                <= CLIMATOLOGY_HALF_WINDOW
            )

            values = past_values[in_window]
            observation_years = past_years[in_window]

            # Require at least MIN_PAST_YEARS distinct historical years.
            if len(np.unique(observation_years)) < MIN_PAST_YEARS:
                continue

            mean[week] = np.mean(values)
            std[week] = np.std(values, ddof=1)

        current_mask = (years == year) & active
        current_weeks = weeks[current_mask]

        m = mean[current_weeks]
        s = std[current_weeks]
        current_values = x.to_numpy()[current_mask]

        # If historical variance is zero, assign zero anomaly.
        valid_mean = ~np.isnan(m)
        result = np.full(len(current_values), np.nan)

        positive_std = valid_mean & (s > 0)
        zero_std = valid_mean & (s == 0)

        result[positive_std] = (
            current_values[positive_std] - m[positive_std]
        ) / s[positive_std]

        result[zero_std] = 0.0

        z.iloc[np.where(current_mask)[0]] = result

    # Off-season is explicitly zero, not missing.
    z.loc[~in_season] = 0.0

    return z


# ---------------------------------------------------------------------------
# Regional components
# ---------------------------------------------------------------------------

def zone_components(
    weather: pd.DataFrame,
    is_gas: bool,
    in_season: pd.Series,
) -> pd.DataFrame:
    """
    Construct and standardise stress components for one commodity-region.
    """

    recipes = GAS_COMPONENTS if is_gas else CROP_COMPONENTS

    raw = pd.DataFrame({
        name: function(weather)
        for name, function in recipes.items()
    })

    if not is_gas:
        # Four consecutive in-season weeks are required.
        raw["dryness"] = -(
            weather["water_balance"]
            .where(in_season)
            .rolling(window=4, min_periods=4)
            .sum()
        )

    # Standardise continuous weather stress indicators.
    components = pd.DataFrame({
        column: expanding_anomaly(raw[column], in_season)
        for column in raw.columns
    })

    if not is_gas:
        # Frost is a binary event indicator, NOT a standardised anomaly.
        components["frost"] = raw["frost"]

        # Preserve missing values during the growing season.
        components.loc[~in_season, "frost"] = 0.0

    return components


# ---------------------------------------------------------------------------
# Production-weighted commodity components
# ---------------------------------------------------------------------------

def build_components(use_season: bool = True) -> pd.DataFrame:

    weather = pd.read_parquet(
        DATA_PROCESSED / "weather_weekly.parquet"
    )
    weights = pd.read_parquet(
        DATA_PROCESSED / "weights.parquet"
    )

    zone_mass = (
        weights.groupby(["commodity", "zone"])["mass"].sum()
    )

    available = set(
        weather.groupby(["commodity", "zone"]).groups
    )

    frames = []

    for name, spec in COMMODITIES.items():

        # Do not silently omit a production region: that would change
        # the intended production exposure weights.
        missing_zones = [
            zone_name
            for zone_name in spec["zones"]
            if (name, zone_name) not in available
        ]

        if missing_zones:
            raise ValueError(
                f"{name}: missing weather data for zones {missing_zones}"
            )

        shares = zone_mass.loc[name]
        shares = shares / shares.sum()

        total = None

        for zone_name, zone in spec["zones"].items():

            w = (
                weather.query(
                    "commodity == @name and zone == @zone_name"
                )
                .pivot(
                    index="date",
                    columns="variable",
                    values="value"
                )
                .sort_index()
            )

            if use_season:
                in_season = pd.Series(
                    w.index.month.isin(zone["season"]),
                    index=w.index,
                    dtype=bool
                )
            else:
                in_season = pd.Series(
                    True,
                    index=w.index,
                    dtype=bool
                )

            comp = zone_components(
                w,
                is_gas=(spec["crop"] is None),
                in_season=in_season
            )

            weighted_comp = comp * shares[zone_name]

            # Do not use fill_value=0: missing in-season observations
            # must not be silently converted into zero stress.
            total = (
                weighted_comp
                if total is None
                else total + weighted_comp
            )

        frames.append(total.assign(commodity=name))

        print(
            f"{name:10s} {list(total.columns)}",
            flush=True
        )

    out = (
        pd.concat(frames)
        .rename_axis("date")
        .reset_index()
    )

    return (
        out.set_index(["date", "commodity"])
        .sort_index()
    )


if __name__ == "__main__":
    build_components().to_parquet(
        DATA_PROCESSED / "components_weekly.parquet"
    )
