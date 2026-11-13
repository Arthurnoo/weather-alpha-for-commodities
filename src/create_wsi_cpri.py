
"""Build a CPRI-inspired equal-weight weekly Weather Stress Index.

Reference methodology:
- Guo, Ji and Zhang (2024) construct CPRI using min-max normalisation
  of physical-risk sub-indices, followed by an arithmetic mean.

Project adaptation:
- Uses weekly, production-weighted weather components.
- Applies expanding, past-only min-max normalisation.
- Equally weights the normalised components.
- Processes every commodity defined in config.py independently.
- This is an adaptation, not an exact replication of the paper.

Output:
    data/processed/wsi_cpri_weekly.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import COMMODITIES, DATA_PROCESSED


INPUT_FILE = DATA_PROCESSED / "components_weekly.parquet"
OUTPUT_FILE = DATA_PROCESSED / "wsi_cpri_weekly.parquet"


def get_season_mask(
    commodity: str,
    dates: pd.DatetimeIndex
) -> pd.Series:
    """Return the union of configured production-zone growing seasons."""

    active_months = set()

    for zone in COMMODITIES[commodity]["zones"].values():
        active_months.update(zone["season"])

    return pd.Series(
        dates.month.isin(active_months),
        index=dates,
        name="in_season"
    )


def cpri_style_index(
    components: pd.DataFrame,
    in_season: pd.Series
) -> pd.Series:
    """Build a past-only min-max index with equal component weights."""

    # Only complete in-season observations enter the historical ranges.
    complete = components.notna().all(axis=1)

    historical = components.where(
        in_season & complete
    )

    # Expanding historical minimum and maximum.
    # shift(1) prevents the current observation from defining its own range.
    historical_min = (
        historical
        .expanding(min_periods=1)
        .min()
        .shift(1)
    )

    historical_max = (
        historical
        .expanding(min_periods=1)
        .max()
        .shift(1)
    )

    historical_range = historical_max - historical_min

    # Min-max normalisation.
    denominator = historical_range.where(
        historical_range > 0
    )

    scaled = (
        100.0
        * (components - historical_min)
        / denominator
    )

    # Handle the case where the historical range is zero.
    # Unchanged or lower values receive 0.
    # A new high above the historical constant level receives 100.
    zero_range = (
        historical_range.eq(0)
        & historical_min.notna()
    )

    scaled = scaled.mask(
        zero_range & components.le(historical_min),
        0.0
    )

    scaled = scaled.mask(
        zero_range & components.gt(historical_min),
        100.0
    )

    # Keep the scale bounded between 0 and 100.
    scaled = scaled.clip(
        lower=0.0,
        upper=100.0
    )

    # Require all components to be available.
    valid = (
        in_season
        & complete
        & scaled.notna().all(axis=1)
    )

    # Equal-weighted arithmetic average.
    index = scaled.mean(axis=1).where(valid)

    # Off-season values are not applicable.
    index.loc[~in_season] = 0.0

    return index.rename("wsi_cpri")


def build_index() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}. Run the components pipeline first."
        )

    components = pd.read_parquet(INPUT_FILE)

    if not isinstance(components.index, pd.MultiIndex):
        raise ValueError(
            "Expected components_weekly.parquet indexed by date and commodity."
        )

    if not {"date", "commodity"}.issubset(
        set(components.index.names)
    ):
        raise ValueError(
            "Expected MultiIndex levels named 'date' and 'commodity'."
        )

    output_frames = []

    available = set(
        components.index.get_level_values("commodity").unique()
    )

    configured = set(COMMODITIES)

    unknown = available - configured

    if unknown:
        print(
            "Warning: skipping commodities absent from config.py:",
            sorted(unknown)
        )

    # Process each commodity independently.
    for commodity, raw in components.groupby(
        level="commodity",
        sort=True
    ):

        if commodity not in COMMODITIES:
            continue

        x = (
            raw
            .droplevel("commodity")
            .sort_index()
            .copy()
        )

        x.index = pd.DatetimeIndex(x.index, name="date")

        # Remove components that are entirely missing for this commodity.
        x = x.dropna(axis=1, how="all")

        if x.empty:
            print(f"{commodity}: skipped (no available components)")
            continue

        in_season = get_season_mask(commodity, x.index)

        wsi = cpri_style_index(
            x,
            in_season
        )

        frame = pd.DataFrame(
            {
                "wsi_cpri": wsi,
                "in_season": in_season.astype("int8")
            },
            index=x.index
        )

        frame["commodity"] = commodity

        output_frames.append(frame)

        n_valid = int(
            wsi.loc[in_season].notna().sum()
        )

        print(
            f"{commodity}: "
            f"{n_valid} valid in-season index observations."
        )

    if not output_frames:
        raise ValueError("No commodity index was produced.")

    # Combine all commodities.
    output = (
        pd.concat(output_frames)
        .reset_index()
        .set_index(["date", "commodity"])
        .sort_index()
    )

    return output


if __name__ == "__main__":

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    index = build_index()

    index.to_parquet(OUTPUT_FILE)

    print(f"\nSaved: {OUTPUT_FILE}")

    print("\nWSI summary:")

    print(
        index
        .groupby(level="commodity")["wsi_cpri"]
        .describe()
        .round(3)
    )
