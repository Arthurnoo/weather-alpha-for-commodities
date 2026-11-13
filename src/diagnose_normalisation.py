"""Robustness diagnostics for the WSI normalisation method.

Run from the repository root:
    python -m src.diagnose_normalisation

Reads:
    data/processed/components_weekly.parquet

Writes CSV files to:
    reports/wsi_normalisation/

Methods compared on the SAME dates:
1. Expanding min-max: current production method.
2. Rolling 10-year min-max: adapts to changing historical ranges.
3. Expanding 5th-95th percentile scaling: reduces sensitivity to single
   historical extremes; values outside the historical quantiles are clipped.

All reference statistics use past observations only (shifted by one week).
The script does not modify the production WSI or input data.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from config import COMMODITIES, DATA_PROCESSED

INPUT = DATA_PROCESSED / "components_weekly.parquet"
OUT = Path("reports") / "wsi_normalisation"
ROLLING_WINDOW = "3650D"  # approximately 10 years
MIN_ROLLING_OBS = 52      # at least one year of valid weekly observations
MIN_QUANTILE_OBS = 52


def season_mask(commodity: str, dates: pd.DatetimeIndex) -> pd.Series:
    months = set()
    for zone in COMMODITIES[commodity]["zones"].values():
        months.update(zone["season"])
    return pd.Series(dates.month.isin(months), index=dates, dtype=bool)


def historical_sample(x: pd.DataFrame, active: pd.Series) -> pd.DataFrame:
    """Use complete in-season rows, exactly as in the production WSI."""
    complete = x.notna().all(axis=1)
    return x.where(active & complete)


def scale_minmax(
    x: pd.DataFrame,
    lo: pd.DataFrame,
    hi: pd.DataFrame,
) -> pd.DataFrame:
    rng = hi - lo
    scaled = 100.0 * (x - lo) / rng.where(rng > 0)

    # Match the production method's zero-range convention.
    zero_range = rng.eq(0) & lo.notna()
    scaled = scaled.mask(zero_range & x.le(lo), 0.0)
    scaled = scaled.mask(zero_range & x.gt(lo), 100.0)

    return scaled.clip(0.0, 100.0)


def build_method_indices(
    x: pd.DataFrame,
    active: pd.Series,
) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    hist = historical_sample(x, active)

    # 1) Current production method: expanding min-max, past only.
    exp_min = hist.expanding(min_periods=1).min().shift(1)
    exp_max = hist.expanding(min_periods=1).max().shift(1)
    exp_scaled = scale_minmax(x, exp_min, exp_max)

    # 2) Trailing 10-year min-max, past only.
    # The rolling window is time-based, so missing weeks do not change its
    # calendar length. The current observation is excluded via shift(1).
    roll_min = hist.rolling(
        ROLLING_WINDOW, min_periods=MIN_ROLLING_OBS
    ).min().shift(1)
    roll_max = hist.rolling(
        ROLLING_WINDOW, min_periods=MIN_ROLLING_OBS
    ).max().shift(1)
    roll_scaled = scale_minmax(x, roll_min, roll_max)

    # 3) Expanding robust scaling: historical 5th-95th percentiles.
    # Clipping prevents one extreme historical value from setting the scale.
    q05 = hist.expanding(
        min_periods=MIN_QUANTILE_OBS
    ).quantile(0.05).shift(1)
    q95 = hist.expanding(
        min_periods=MIN_QUANTILE_OBS
    ).quantile(0.95).shift(1)
    robust_scaled = scale_minmax(x, q05, q95)

    scaled = {
        "expanding_minmax": exp_scaled,
        "rolling_10y_minmax": roll_scaled,
        "expanding_q05_q95": robust_scaled,
    }

    indices = {}
    for name, values in scaled.items():
        valid = active & x.notna().all(axis=1) & values.notna().all(axis=1)
        indices[name] = values.mean(axis=1).where(valid).rename(name)

    bounds = {
        "expanding_minmax": (exp_min, exp_max),
        "rolling_10y_minmax": (roll_min, roll_max),
        "expanding_q05_q95": (q05, q95),
    }
    return indices, scaled, bounds


def run() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run the components pipeline first."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    components = pd.read_parquet(INPUT)

    if not isinstance(components.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex (date, commodity).")
    if not {"date", "commodity"}.issubset(components.index.names):
        raise ValueError("Expected index levels named 'date' and 'commodity'.")

    comparison_rows = []
    annual_rows = []
    saturation_rows = []
    common_rows = []
    event_rows = []

    for commodity, frame in components.groupby(level="commodity", sort=True):
        if commodity not in {"corn", "soybeans"}:
            continue

        x = frame.droplevel("commodity").sort_index().copy()
        x.index = pd.DatetimeIndex(x.index, name="date")
        x = x.dropna(axis=1, how="all")
        active = season_mask(commodity, x.index)

        indices, scaled, bounds = build_method_indices(x, active)

        index_frame = pd.concat(indices.values(), axis=1)
        common = index_frame.dropna()
        if common.empty:
            print(f"{commodity}: no common observations across methods.")
            continue

        # Compare all methods on exactly the same dates.
        baseline = common["expanding_minmax"]
        for method in ["rolling_10y_minmax", "expanding_q05_q95"]:
            diff = common[method] - baseline
            comparison_rows.append({
                "commodity": commodity,
                "method": method,
                "n_common": len(common),
                "first_common_date": common.index.min().date().isoformat(),
                "last_common_date": common.index.max().date().isoformat(),
                "correlation_with_expanding_minmax": common[method].corr(baseline),
                "mean_baseline": baseline.mean(),
                "mean_alternative": common[method].mean(),
                "mean_abs_difference": diff.abs().mean(),
                "median_abs_difference": diff.abs().median(),
                "p95_abs_difference": diff.abs().quantile(0.95),
                "max_abs_difference": diff.abs().max(),
                "share_abs_difference_gt_5pts": 100 * diff.abs().gt(5).mean(),
                "share_abs_difference_gt_10pts": 100 * diff.abs().gt(10).mean(),
            })

        # Method-specific coverage and annual behaviour.
        for method, idx in indices.items():
            for year, group in idx.loc[active].groupby(idx.loc[active].index.year):
                annual_rows.append({
                    "commodity": commodity,
                    "method": method,
                    "year": year,
                    "n_valid": int(group.notna().sum()),
                    "missing_pct": 100 * group.isna().mean(),
                    "wsi_mean": group.mean(),
                    "wsi_median": group.median(),
                    "wsi_p90": group.quantile(0.90),
                    "wsi_max": group.max(),
                })

        # Compare per-component scaled distributions and clipping.
        for method, values in scaled.items():
            for col in values.columns:
                s = values.loc[active, col].dropna()
                if s.empty:
                    continue
                saturation_rows.append({
                    "commodity": commodity,
                    "method": method,
                    "component": col,
                    "n_scaled": len(s),
                    "mean": s.mean(),
                    "std": s.std(),
                    "q05": s.quantile(0.05),
                    "median": s.median(),
                    "q95": s.quantile(0.95),
                    "pct_at_0": 100 * s.le(1e-12).mean(),
                    "pct_at_100": 100 * s.ge(100 - 1e-12).mean(),
                })

        # Annual comparison on the common sample, to detect time-varying gaps.
        for year, group in common.groupby(common.index.year):
            for method in ["rolling_10y_minmax", "expanding_q05_q95"]:
                diff = group[method] - group["expanding_minmax"]
                common_rows.append({
                    "commodity": commodity,
                    "year": year,
                    "method": method,
                    "n_common": len(group),
                    "mean_abs_difference": diff.abs().mean(),
                    "p95_abs_difference": diff.abs().quantile(0.95),
                    "max_abs_difference": diff.abs().max(),
                    "correlation": group[method].corr(group["expanding_minmax"]),
                })

        # Identify dates on which the alternative methods disagree most.
        for method in ["rolling_10y_minmax", "expanding_q05_q95"]:
            diff = (common[method] - common["expanding_minmax"]).abs()
            top = diff.nlargest(10)
            for date, value in top.items():
                event_rows.append({
                    "commodity": commodity,
                    "method": method,
                    "date": date.date().isoformat(),
                    "expanding_minmax_wsi": common.loc[date, "expanding_minmax"],
                    "alternative_wsi": common.loc[date, method],
                    "absolute_difference": value,
                })

    pd.DataFrame(comparison_rows).to_csv(
        OUT / "normalisation_comparison.csv", index=False
    )
    pd.DataFrame(annual_rows).to_csv(
        OUT / "normalisation_annual.csv", index=False
    )
    pd.DataFrame(saturation_rows).to_csv(
        OUT / "normalisation_component_saturation.csv", index=False
    )
    pd.DataFrame(common_rows).to_csv(
        OUT / "normalisation_annual_comparison.csv", index=False
    )
    pd.DataFrame(event_rows).to_csv(
        OUT / "normalisation_largest_differences.csv", index=False
    )

    print(f"Normalisation diagnostics saved to: {OUT.resolve()}")
    for name in [
        "normalisation_comparison.csv",
        "normalisation_annual.csv",
        "normalisation_component_saturation.csv",
        "normalisation_annual_comparison.csv",
        "normalisation_largest_differences.csv",
    ]:
        print(f" - {name}")

    print("\nComparison summary:")
    if comparison_rows:
        print(pd.DataFrame(comparison_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    run()
