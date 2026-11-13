"""Diagnostics for the crop Weather Stress Index (WSI).

Run from the repository root:
    python -m src.diagnose_wsi

Reads data/processed/components_weekly.parquet and writes CSV diagnostics to
reports/wsi_diagnostics/. It does not modify the production WSI or any inputs.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from config import COMMODITIES, DATA_PROCESSED

INPUT = DATA_PROCESSED / "components_weekly.parquet"
OUT = Path("reports") / "wsi_diagnostics"


def season_mask(commodity: str, dates: pd.DatetimeIndex) -> pd.Series:
    months = set()
    for zone in COMMODITIES[commodity]["zones"].values():
        months.update(zone["season"])
    return pd.Series(dates.month.isin(months), index=dates)


def past_minmax(x: pd.DataFrame, active: pd.Series) -> pd.DataFrame:
    """Past-only expanding min-max scaling, same rule as production WSI."""
    complete = x.notna().all(axis=1)
    hist = x.where(active & complete)
    lo = hist.expanding(min_periods=1).min().shift(1)
    hi = hist.expanding(min_periods=1).max().shift(1)
    rng = hi - lo
    scaled = 100 * (x - lo) / rng.where(rng > 0)
    zero = rng.eq(0) & lo.notna()
    scaled = scaled.mask(zero & x.le(lo), 0.0)
    scaled = scaled.mask(zero & x.gt(lo), 100.0)
    return scaled.clip(0, 100)


def expanding_zscore(x: pd.Series, active: pd.Series) -> pd.Series:
    """Past-only expanding standardisation used to test a frost alternative."""
    hist = x.where(active)
    mean = hist.expanding(min_periods=1).mean().shift(1)
    std = hist.expanding(min_periods=2).std().shift(1)
    z = (x - mean) / std.replace(0, np.nan)
    return z.where(active)


def describe_components(x: pd.DataFrame, active: pd.Series, commodity: str) -> pd.DataFrame:
    rows = []
    for col in x:
        s = x.loc[active, col]
        rows.append({
            "commodity": commodity, "component": col,
            "n_season": int(active.sum()), "n_nonmissing": int(s.notna().sum()),
            "missing_pct": 100 * s.isna().mean(),
            "mean": s.mean(), "std": s.std(), "skew": s.skew(),
            "q50": s.quantile(.50), "q90": s.quantile(.90),
            "q95": s.quantile(.95), "q99": s.quantile(.99), "max": s.max(),
            "zero_pct": 100 * s.eq(0).mean(),
        })
    return pd.DataFrame(rows)


def run() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}. Run the components pipeline first.")
    OUT.mkdir(parents=True, exist_ok=True)
    components = pd.read_parquet(INPUT)
    if not isinstance(components.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex (date, commodity).")

    summary, corr_rows, yearly_rows, scaling_rows, frost_rows = [], [], [], [], []
    for commodity, frame in components.groupby(level="commodity", sort=True):
        if commodity not in {"corn", "soybeans"}:
            continue
        x = frame.droplevel("commodity").sort_index().copy()
        x.index = pd.DatetimeIndex(x.index)
        x = x.dropna(axis=1, how="all")
        active = season_mask(commodity, x.index)

        summary.append(describe_components(x, active, commodity))
        corr = x.loc[active].corr(method="spearman")
        corr.insert(0, "component", corr.index)
        corr.insert(0, "commodity", commodity)
        corr_rows.append(corr.reset_index(drop=True))

        scaled = past_minmax(x, active)
        valid = active & x.notna().all(axis=1) & scaled.notna().all(axis=1)
        wsi = scaled.mean(axis=1).where(valid)
        # Per-component contribution to the equal-weight index.
        contrib = scaled.div(len(scaled.columns))
        for col in scaled:
            s = scaled.loc[valid, col]
            scaling_rows.append({
                "commodity": commodity, "component": col,
                "scaled_mean": s.mean(), "scaled_std": s.std(),
                "scaled_q50": s.quantile(.50), "scaled_q90": s.quantile(.90),
                "scaled_q95": s.quantile(.95), "scaled_q99": s.quantile(.99),
                "share_of_total_contribution": contrib.loc[valid, col].sum() /
                    contrib.loc[valid].sum(axis=1).sum(),
                "corr_with_wsi": scaled.loc[valid, col].corr(wsi.loc[valid]),
            })

        # Annual behaviour: helps reveal structural shifts / season-composition effects.
        for year, idx in wsi.loc[active].groupby(wsi.loc[active].index.year):
            yearly_rows.append({
                "commodity": commodity, "year": year,
                "n_valid": int(idx.notna().sum()),
                "wsi_mean": idx.mean(), "wsi_median": idx.median(),
                "wsi_p90": idx.quantile(.90), "wsi_max": idx.max(),
                "missing_pct": 100 * idx.isna().mean(),
            })

        # Sensitivity: current raw frost degree-days vs a past-only z-score version.
        if "frost" in x:
            frost_z = expanding_zscore(x["frost"], active)
            alt = x.copy()
            alt["frost"] = frost_z
            alt_scaled = past_minmax(alt, active)
            alt_valid = active & alt.notna().all(axis=1) & alt_scaled.notna().all(axis=1)
            alt_wsi = alt_scaled.mean(axis=1).where(alt_valid)
            both = pd.concat([wsi.rename("current"), alt_wsi.rename("frost_zscore")], axis=1).dropna()
            frost_rows.append({
                "commodity": commodity, "n_common": len(both),
                "corr_current_vs_frost_zscore": both.current.corr(both.frost_zscore),
                "mean_current": both.current.mean(), "mean_frost_zscore": both.frost_zscore.mean(),
                "mean_abs_difference": (both.current - both.frost_zscore).abs().mean(),
                "max_abs_difference": (both.current - both.frost_zscore).abs().max(),
            })

    pd.concat(summary, ignore_index=True).to_csv(OUT / "component_distribution.csv", index=False)
    pd.concat(corr_rows, ignore_index=True).to_csv(OUT / "component_spearman_correlations.csv", index=False)
    pd.DataFrame(scaling_rows).to_csv(OUT / "scaled_component_contributions.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(OUT / "annual_wsi_diagnostics.csv", index=False)
    pd.DataFrame(frost_rows).to_csv(OUT / "frost_sensitivity.csv", index=False)

    print(f"Diagnostics saved to: {OUT.resolve()}")
    for name in ["component_distribution.csv", "component_spearman_correlations.csv",
                 "scaled_component_contributions.csv", "annual_wsi_diagnostics.csv",
                 "frost_sensitivity.csv"]:
        print(f" - {name}")


if __name__ == "__main__":
    run()
