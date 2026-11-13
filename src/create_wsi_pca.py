
"""Build a fixed-basis, PCA-based weekly Weather Stress Index (WSI).

Methodology:
- Uses production-weighted weather components from components_weekly.parquet.
- Fits StandardScaler and PCA using only pre-test in-season observations.
- Freezes the scaler and PC1 loadings for the entire sample.
- Orients PC1 so that higher scores correspond to higher overall stress.
- Keeps missing in-season observations as NaN.
- Sets off-season index values to zero and retains an in_season indicator.
- Processes every commodity defined in config.py independently.

Outputs:
    data/processed/wsi_pca_weekly.parquet
    data/processed/wsi_pca_metadata.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import COMMODITIES, DATA_PROCESSED, FIRST_TEST_YEAR


INPUT_FILE = DATA_PROCESSED / "components_weekly.parquet"
OUTPUT_FILE = DATA_PROCESSED / "wsi_pca_weekly.parquet"
METADATA_FILE = DATA_PROCESSED / "wsi_pca_metadata.parquet"

MIN_FIT_WEEKS = 104


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


def orient_component(
    scores: np.ndarray,
    standardized_components: np.ndarray,
    loadings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Orient PC1 so that higher values represent higher overall stress."""

    reference = standardized_components.mean(axis=1)

    correlation = np.corrcoef(scores, reference)[0, 1]

    if np.isfinite(correlation) and correlation < 0:
        return -scores, -loadings

    if not np.isfinite(correlation) and loadings.sum() < 0:
        return -scores, -loadings

    return scores, loadings


def build_index() -> tuple[pd.DataFrame, pd.DataFrame]:

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
    metadata_rows = []

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

        if x.empty or x.shape[1] < 2:
            print(
                f"{commodity}: skipped "
                "(fewer than two available components)"
            )
            continue

        in_season = get_season_mask(commodity, x.index)

        # PCA requires complete observations.
        complete = x.notna().all(axis=1)

        # Initial training period only.
        fit_mask = (
            in_season
            & complete
            & (x.index.year < FIRST_TEST_YEAR)
        )

        fit = x.loc[fit_mask]

        if len(fit) < MIN_FIT_WEEKS:
            raise ValueError(
                f"{commodity}: only {len(fit)} complete in-season "
                f"fitting weeks before {FIRST_TEST_YEAR}; "
                f"need at least {MIN_FIT_WEEKS}."
            )

        # Fit standardisation on the initial training sample.
        scaler = StandardScaler()

        fit_scaled = scaler.fit_transform(fit)

        # Fit the first principal component.
        pca = PCA(n_components=1)

        pca.fit(fit_scaled)

        raw_fit_scores = pca.transform(fit_scaled)[:, 0]

        # Determine the sign of PC1 using the training sample.
        oriented_scores, loadings = orient_component(
            raw_fit_scores.copy(),
            fit_scaled,
            pca.components_[0].copy()
        )

        sign = (
            1.0
            if np.dot(oriented_scores, raw_fit_scores) >= 0
            else -1.0
        )

        # Apply the same scaler and PCA basis to all complete
        # in-season observations, including the test period.
        valid_mask = in_season & complete

        valid = x.loc[valid_mask]

        scores = (
            sign
            * pca.transform(scaler.transform(valid))[:, 0]
        )

        # Initialise the index.
        wsi = pd.Series(
            np.nan,
            index=x.index,
            name="wsi_pca",
            dtype=float
        )

        # Off-season observations are not applicable.
        wsi.loc[~in_season] = 0.0

        # Insert PCA scores.
        wsi.loc[valid.index] = scores

        frame = pd.DataFrame(
            {
                "wsi_pca": wsi,
                "in_season": in_season.astype("int8")
            },
            index=x.index
        )

        frame["commodity"] = commodity

        output_frames.append(frame)

        # Store PCA information for interpretation.
        metadata_rows.append(
            {
                "commodity": commodity,
                "fit_start": fit.index.min(),
                "fit_end": fit.index.max(),
                "fit_weeks": len(fit),
                "n_components": x.shape[1],
                "explained_variance_ratio_pc1": float(
                    pca.explained_variance_ratio_[0]
                ),
                **{
                    f"loading_{col}": float(weight)
                    for col, weight in zip(x.columns, loadings)
                }
            }
        )

        print(
            f"{commodity}: {len(fit)} fitting weeks; "
            f"PC1 explains "
            f"{pca.explained_variance_ratio_[0]:.1%} "
            f"of the variance."
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

    metadata = pd.DataFrame(metadata_rows)

    return output, metadata


if __name__ == "__main__":

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    index, metadata = build_index()

    index.to_parquet(OUTPUT_FILE)

    metadata.to_parquet(
        METADATA_FILE,
        index=False
    )

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Saved: {METADATA_FILE}")

    print("\nWSI summary:")

    print(
        index
        .groupby(level="commodity")["wsi_pca"]
        .describe()
        .round(3)
    )
