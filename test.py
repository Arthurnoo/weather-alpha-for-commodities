
import pandas as pd

from config import COMMODITIES, DATA_PROCESSED
from src.components import zone_components

weather = pd.read_parquet(
    DATA_PROCESSED / "weather_weekly.parquet"
)

for commodity in ["corn", "soybeans"]:

    spec = COMMODITIES[commodity]

    print("\n" + "=" * 65)
    print(commodity.upper())
    print("=" * 65)

    for zone_name, zone in spec["zones"].items():

        print(f"\n--- REGION: {zone_name} ---")

        w = (
            weather.query(
                "commodity == @commodity and zone == @zone_name"
            )
            .pivot(
                index="date",
                columns="variable",
                values="value"
            )
            .sort_index()
        )

        in_season = pd.Series(
            w.index.month.isin(zone["season"]),
            index=w.index,
            dtype=bool
        )

        comp = zone_components(
            w,
            is_gas=False,
            in_season=in_season
        )

        dryness_missing = comp.index[comp["dryness"].isna()]

        print("\nDryness missing observations by month:")
        print(
            pd.Series(dryness_missing.month)
            .value_counts()
            .sort_index()
            .to_string()
        )

        # Missing values in raw weather data
        print("\nRaw weather missing values:")
        for variable in [
            "edd",
            "tmin",
            "heavy_rain_days",
            "frost_dd",
            "water_balance"
        ]:
            if variable in w.columns:
                print(f"{variable:20s}: {w[variable].isna().sum()}")

        # Missing values after component construction
        print("\nComponent missing values:")
        for column in comp.columns:

            missing_dates = comp.index[comp[column].isna()]

            if len(missing_dates):
                print(
                    f"{column:20s}: {len(missing_dates):5d} | "
                    f"{missing_dates.min().date()} "
                    f"to {missing_dates.max().date()}"
                )
            else:
                print(f"{column:20s}: 0")

        # Annual distribution of missing heat observations
        print("\nMissing heat observations by year:")
        print(
            comp["heat"]
            .isna()
            .groupby(comp.index.year)
            .sum()
            .loc[lambda x: x > 0]
            .to_string()
        )
