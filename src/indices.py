"""
Weather Stress Indices built from the components.

WSI-CPRI  equal-weight index, as in the Climate Physical Risk Index
          (Guo, Ji and Zhang, 2024): each component is min-max scaled to
          [0, 100] with its PAST range, then the scaled components are averaged.
WSI-PCA   first principal component of the components (Ma, Zhou and Li, 2024),
          re-estimated every January on past data only (expanding window).
          The weights (loadings) are kept to show what drives the index.

Output: data/processed/indices_weekly.parquet  index (date, commodity): wsi_cpri, wsi_pca
        data/processed/pca_loadings.parquet    one row per (commodity, year)
Run:    python -m src.indices
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import DATA_PROCESSED

MIN_FIT_WEEKS = 104  # at least two years of in-season weeks before fitting a PCA


def cpri_index(components: pd.DataFrame) -> pd.Series:
    """Equal-weight index of min-max scaled components (past range only)."""
    # Off-season weeks (all components = 0) are excluded from the range and get
    # an index of 0: weather is not supposed to matter then
    in_season = (components != 0).any(axis=1) & components.notna().all(axis=1)
    active = components.where(in_season)
    # shift(1): the range used at week t only contains weeks before t
    low = active.expanding().min().shift(1)
    high = active.expanding().max().shift(1)
    scaled = (100 * (components - low) / (high - low)).clip(0, 100).mean(axis=1)
    return scaled.where(in_season, 0.0).where(components.notna().all(axis=1))


def pca_index(components: pd.DataFrame) -> tuple:
    """
    First principal component, refit each January on all previous weeks.

    Off-season weeks (all components = 0) are left out of the fit: they would
    only teach the PCA that "nothing happens out of season".
    Sign convention: loadings sum to a positive number, so that a higher index
    means more stress (all components are oriented as stress).
    Returns the index and a table of yearly loadings.
    """
    index = pd.Series(np.nan, index=components.index)
    loadings = []
    clean = components.dropna()
    active = clean[(clean != 0).any(axis=1)]

    for year in sorted(clean.index.year.unique()):
        train = active[active.index.year < year]
        if len(train) < MIN_FIT_WEEKS:
            continue
        pca = PCA(n_components=1).fit(train.values)
        w = pca.components_[0]
        w = w if w.sum() >= 0 else -w

        # Project this year's weeks with weights estimated before the year starts.
        # Components are already z-scores, so no extra scaling is needed; the
        # mean is removed so that a normal week sits around 0.
        this_year = clean[clean.index.year == year]
        index[this_year.index] = (this_year.values - pca.mean_) @ w
        index[this_year.index[(this_year == 0).all(axis=1)]] = 0.0  # off season

        loadings.append({"year": year, "explained_var": pca.explained_variance_ratio_[0],
                         **dict(zip(components.columns, w))})
    return index, pd.DataFrame(loadings)


def build_indices() -> tuple:
    components = pd.read_parquet(DATA_PROCESSED / "components_weekly.parquet")
    indices, all_loadings = [], []
    for name, comp in components.groupby(level="commodity"):
        comp = comp.droplevel("commodity").dropna(axis=1, how="all")
        pca, loadings = pca_index(comp)
        indices.append(pd.DataFrame({"wsi_cpri": cpri_index(comp), "wsi_pca": pca,
                                     "commodity": name}))
        all_loadings.append(loadings.assign(commodity=name))
    out = pd.concat(indices).rename_axis("date").reset_index()
    return out.set_index(["date", "commodity"]).sort_index(), pd.concat(all_loadings)


if __name__ == "__main__":
    indices, loadings = build_indices()
    indices.to_parquet(DATA_PROCESSED / "indices_weekly.parquet")
    loadings.to_parquet(DATA_PROCESSED / "pca_loadings.parquet")
    print(indices.groupby("commodity").describe().round(2).T)
