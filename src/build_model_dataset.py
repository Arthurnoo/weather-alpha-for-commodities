
from pathlib import Path

import numpy as np
import pandas as pd

from config import FIRST_TEST_YEAR


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed")
OUTPUT_PATH = DATA_DIR / "model_dataset.parquet"

WEATHER_LAG = 1

# For the moment only corn and soybeans maybe later we'll extend the research
COMMODITIES = ["corn", "soybeans"]

MARKET_FEATURES = [
    "ret_1w",
    "ret_4w",
    "ret_12w",
    "vol_4w",
    "vol_12w",
    "month_sin",
    "month_cos",
]

WEATHER_FEATURES = [
    "heat",
    "warm_nights",
    "heavy_rain",
    "frost",
    "dryness",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    prices = pd.read_parquet(DATA_DIR / "prices_weekly.parquet")
    components = pd.read_parquet(DATA_DIR / "components_weekly.parquet")
    wsi = pd.read_parquet(DATA_DIR / "wsi_cpri_weekly.parquet")

    # For all commo but for the moment only corn and soybeans
    #for df in [prices, components, wsi]:
    #    df.sort_index(inplace=True)

    prices = prices.sort_index()
    components = components.sort_index()
    wsi = wsi.sort_index()

    return prices, components, wsi


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset():
    prices, components, wsi = load_data()

    # Keep only the five components used in the frozen WSI.
    weather = components[WEATHER_FEATURES].join(
        wsi[["wsi_cpri", "in_season"]],
        how="outer",
    )

    # Lag weather information within each commodity.
    # This avoids using weather observations from the same
    # Wednesday as the return prediction date.
    weather = (
        weather.groupby(level="commodity", sort=False)
        .shift(WEATHER_LAG)
    )

    # Merge market and lagged weather information.
    df = prices.join(weather, how="left")

    # Restrict the analysis to the two selected commodities.
    df = df.loc[
    df.index.get_level_values("commodity").isin(COMMODITIES)
    ].copy()

    # Encode month cyclically.
    month = df["month"]

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Store the lagged weather season indicator as an integer.
    df["in_season"] = df["in_season"].astype("Int64")

    # Remove the original month variable.
    df = df.drop(columns=["month"])

    # Keep the columns needed for modelling and diagnostics.
    selected_columns = (
        ["target_1w"]
        + MARKET_FEATURES
        + ["wsi_cpri"]
        + WEATHER_FEATURES
        + ["in_season"]
    )

    df = df[selected_columns].sort_index()

    return df


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_diagnostics(df):
    print("\n" + "=" * 65)
    print("MODELLING DATASET")
    print("=" * 65)

    print(f"Weather lag: {WEATHER_LAG} week(s)")
    print(f"First test year: {FIRST_TEST_YEAR}")
    print(f"Total rows: {len(df)}")

    for commodity in df.index.get_level_values("commodity").unique():

        data = df.xs(commodity, level="commodity")

        valid_target = data["target_1w"].notna()

        valid_market = (
            data[MARKET_FEATURES].notna().all(axis=1)
        )

        valid_wsi = data["wsi_cpri"].notna()

        valid_components = (
            data[WEATHER_FEATURES].notna().all(axis=1)
        )

        print(f"\n--- {commodity.upper()} ---")
        print("Total observations:", len(data))
        print("Valid targets:", valid_target.sum())
        print(
            "Market + target:",
            (valid_market & valid_target).sum(),
        )
        print(
            "Market + WSI + target:",
            (valid_market & valid_wsi & valid_target).sum(),
        )
        print(
            "Market + components + target:",
            (valid_market & valid_components & valid_target).sum(),
        )

        print("\nMissing values (%):")
        print(
            data[
                ["target_1w"]
                + MARKET_FEATURES
                + ["wsi_cpri"]
                + WEATHER_FEATURES
            ]
            .isna()
            .mean()
            .round(3)
            .to_string()
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    dataset = build_dataset()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT_PATH)

    print_diagnostics(dataset)

    print(f"\nDataset saved to: {OUTPUT_PATH}")
