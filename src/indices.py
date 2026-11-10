
"""
Weather Stress Indices built from the weekly weather components.

WSI-CPRI:
    Equal-weight index inspired by the Climate Physical Risk Index
    (Guo, Ji and Zhang, 2024).
    Each component is min-max scaled using its historical range only.
    The scaled components are then averaged.

WSI-PCA:
    First principal component, inspired by Ma, Zhou and Li (2024).
    PCA is re-estimated every January using previous years only.
    Components are standardised using the historical training sample.

Both indices use explicit growing-season definitions from config.py.

Output:
    data/processed/indices_weekly.parquet
    data/processed/pca_loadings.parquet

Run:
    python -m src.indices
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import COMMODITIES, DATA_PROCESSED


MIN_FIT_WEEKS = 104  # Minimum two years of historical in-season observations


def season_mask(commodity: str, dates: pd.DatetimeIndex) -> pd.Series:
    """
    Construct an explicit seasonal mask from config.py.

    A week is considered in-season if at least one production zone
    for the commodity is in its growing season.
    """

    spec = COMMODITIES[commodity]

    active_months = set()

    for zone in spec["zones"].values():
        active_months.update(zone["season"])

    return pd.Series(
        dates.month.isin(active_months),
        index=dates,
        name="in_season"
    )


def cpri_index(
    components: pd.DataFrame,
    in_season: pd.Series
) -> pd.Series:
    """
    Equal-weight index using past-only expanding min-max scaling.

    Historical ranges are estimated using previous in-season observations.
    """

    complete = components.notna().all(axis=1)

    # Only historical, valid, in-season observations enter the ranges.
    active = components.where(in_season & complete)

    low = active.expanding(min_periods=1).min().shift(1)
    high = active.expanding(min_periods=1).max().shift(1)

    historical_range = high - low

    # Standard min-max scaling.
    scaled = (
        100 * (components - low)
        / historical_range.replace(0, np.nan)
    )

    # If the historical range is zero:
    # - unchanged or lower stress -> 0
    # - a new value above the historical level -> 100
    zero_range = historical_range.eq(0) & low.notna()

    scaled = scaled.mask(
        zero_range & components.le(low),
        0.0
    )

    scaled = scaled.mask(
        zero_range & components.gt(low),
        100.0
    )

    scaled = scaled.clip(0, 100)

    # Require all components to be available and scaled.
    valid = (
        complete
        & scaled.notna().all(axis=1)
        & in_season
    )

    index = scaled.mean(axis=1).where(valid)

    # Off-season stress is explicitly defined as zero.
    index.loc[~in_season] = 0.0

    return index.rename("wsi_cpri")


def pca_index(
    components: pd.DataFrame,
    in_season: pd.Series
) -> tuple:
    """
    First principal component, re-estimated every January.

    Only previous in-season observations are used for fitting.
    Standardisation parameters are estimated on the training sample only.
    """

    index = pd.Series(
        np.nan,
        index=components.index,
        name="wsi_pca"
    )

    # Off-season stress is explicitly zero.
    index.loc[~in_season] = 0.0

    loadings = []

    years = sorted(components.index.year.unique())

    for year in years:

        # Historical training observations only.
        train_mask = (
            in_season
            & (components.index.year < year)
        )

        train = components.loc[train_mask].dropna()

        if len(train) < MIN_FIT_WEEKS:
            continue

        # Standardise using the historical training sample only.
        scaler = StandardScaler()

        train_scaled = scaler.fit_transform(train)

        # Fit PCA on the standardised historical observations.
        pca = PCA(n_components=1)

        pca.fit(train_scaled)

        w = pca.components_[0].copy()

        # Orient the component so that higher scores correspond,
        # where possible, to higher values of the average stress signal.
        train_scores = pca.transform(train_scaled)[:, 0]
        reference = train_scaled.mean(axis=1)

        correlation = np.corrcoef(
            train_scores,
            reference
        )[0, 1]

        if np.isfinite(correlation) and correlation < 0:
            w = -w

        elif not np.isfinite(correlation) and w.sum() < 0:
            w = -w

        # Apply the historical scaler and PCA to the current year.
        current_mask = (
            in_season
            & (components.index.year == year)
        )

        current = components.loc[current_mask].dropna()

        if not current.empty:

            current_scaled = scaler.transform(current)

            scores = (
                current_scaled - pca.mean_
            ) @ w

            index.loc[current.index] = scores

        # Save PCA loadings and explained variance.
        loadings.append({
            "year": year,
            "explained_var": pca.explained_variance_ratio_[0],
            **dict(zip(components.columns, w))
        })

    return index, pd.DataFrame(loadings)


def build_indices() -> tuple:

    components = pd.read_parquet(
        DATA_PROCESSED / "components_weekly.parquet"
    )

    indices = []
    all_loadings = []

    for name, comp in components.groupby(level="commodity"):

        comp = (
            comp.droplevel("commodity")
            .sort_index()
            .dropna(axis=1, how="all")
        )

        # Explicit growing-season definition from config.py.
        in_season = season_mask(name, comp.index)

        # Construct the two indices.
        cpri = cpri_index(comp, in_season)

        pca, loadings = pca_index(
            comp,
            in_season
        )

        indices.append(
            pd.DataFrame({
                "wsi_cpri": cpri,
                "wsi_pca": pca,
                "commodity": name
            })
        )

        all_loadings.append(
            loadings.assign(commodity=name)
        )

    out = (
        pd.concat(indices)
        .rename_axis("date")
        .reset_index()
        .set_index(["date", "commodity"])
        .sort_index()
    )

    loadings_out = pd.concat(
        all_loadings,
        ignore_index=True
    )

    return out, loadings_out


if __name__ == "__main__":

    indices, loadings = build_indices()

    indices.to_parquet(
        DATA_PROCESSED / "indices_weekly.parquet"
    )

    loadings.to_parquet(
        DATA_PROCESSED / "pca_loadings.parquet"
    )

    print("\n========== WSI SUMMARY ==========")
    print(indices.groupby("commodity").describe().round(3).T)

    print("\n========== PCA LOADINGS ==========")
    print(loadings.round(3).to_string(index=False))
