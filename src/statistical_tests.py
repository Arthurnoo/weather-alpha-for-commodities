
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT / "results" / "predictions" / "walk_forward_predictions.parquet"
)

OUTPUT_DIR = ROOT / "results" / "statistical_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HAC_LAGS = 4

COMMODITIES = ["corn", "soybeans"]
ML_MODELS = ["Ridge", "RandomForest", "HistGradientBoosting"]

FEATURE_SETS = [
    "M1_Financial",
    "M2_WSI",
    "M3_Components",
]


# ============================================================
# HAC variance estimator
# ============================================================

def hac_variance_of_mean(values, max_lags=4):
    """
    Newey-West HAC estimator of the variance of the sample mean.

    Uses Bartlett weights.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)

    if n < 2:
        return np.nan

    centered = values - values.mean()

    gamma_0 = np.dot(centered, centered) / n
    long_run_variance = gamma_0

    max_lags = min(max_lags, n - 1)

    for lag in range(1, max_lags + 1):
        weight = 1 - lag / (max_lags + 1)

        gamma_lag = (
            np.dot(centered[lag:], centered[:-lag]) / n
        )

        long_run_variance += 2 * weight * gamma_lag

    return max(long_run_variance, 0) / n


# ============================================================
# Forecast comparison tests
# ============================================================

def test_statistic(loss_differential, alternative="two-sided"):
    """
    Test whether the mean loss differential is zero.

    Positive values favour the first forecast defined in each
    comparison. The alternative controls the p-value calculation.
    """
    d = np.asarray(loss_differential, dtype=float)
    d = d[np.isfinite(d)]

    n = len(d)

    if n < 2:
        return np.nan, np.nan, n, np.nan

    mean_d = d.mean()
    variance_mean = hac_variance_of_mean(d, HAC_LAGS)

    if not np.isfinite(variance_mean) or variance_mean <= 0:
        return mean_d, np.nan, n, np.nan

    standard_error = np.sqrt(variance_mean)
    statistic = mean_d / standard_error

    if alternative == "two-sided":
        p_value = 2 * stats.t.sf(abs(statistic), df=n - 1)
    elif alternative == "greater":
        p_value = stats.t.sf(statistic, df=n - 1)
    else:
        raise ValueError("Unknown alternative.")

    return mean_d, statistic, n, p_value


def diebold_mariano(actual, forecast_a, forecast_b):
    """
    DM test using squared-error loss.

    Positive mean differential means forecast A has lower
    squared forecast error than forecast B.

    H0: equal predictive accuracy.
    H1: different predictive accuracy.
    """
    error_a = actual - forecast_a
    error_b = actual - forecast_b

    differential = error_b**2 - error_a**2

    return test_statistic(
        differential,
        alternative="two-sided"
    )


def clark_west(actual, forecast_restricted, forecast_augmented):
    """
    Clark-West test for nested forecasting models.

    Restricted model: financial-only model.
    Augmented model: financial + weather model.

    Positive adjusted differential favours the augmented model.

    H0: augmented model does not improve forecasting accuracy.
    H1: augmented model improves forecasting accuracy.
    """
    error_restricted = actual - forecast_restricted
    error_augmented = actual - forecast_augmented

    adjustment = (
        forecast_restricted - forecast_augmented
    ) ** 2

    differential = (
        error_restricted**2
        - error_augmented**2
        + adjustment
    )

    return test_statistic(
        differential,
        alternative="greater"
    )


# ============================================================
# Load and validate predictions
# ============================================================

def load_predictions():
    df = pd.read_parquet(PREDICTIONS_PATH)

    required_columns = {
        "date",
        "commodity",
        "model",
        "feature_set",
        "actual",
        "prediction",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in prediction file: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"])

    if df.duplicated(
        subset=["date", "commodity", "model", "feature_set"]
    ).any():
        raise ValueError(
            "Duplicate date/commodity/model/feature_set observations."
        )

    return df


# ============================================================
# Prepare paired forecasts
# ============================================================

def get_paired_forecasts(df, commodity, model_a, feature_a,
                         model_b, feature_b):

    a = df[
        (df["commodity"] == commodity)
        & (df["model"] == model_a)
        & (df["feature_set"] == feature_a)
    ][["date", "actual", "prediction"]].copy()

    b = df[
        (df["commodity"] == commodity)
        & (df["model"] == model_b)
        & (df["feature_set"] == feature_b)
    ][["date", "actual", "prediction"]].copy()

    a = a.rename(columns={
        "actual": "actual_a",
        "prediction": "forecast_a",
    })

    b = b.rename(columns={
        "actual": "actual_b",
        "prediction": "forecast_b",
    })

    paired = a.merge(b, on="date", how="inner")

    if not np.allclose(
        paired["actual_a"],
        paired["actual_b"],
        equal_nan=True
    ):
        raise ValueError(
            f"Actual returns do not match: {commodity}, "
            f"{model_a}/{feature_a}, {model_b}/{feature_b}"
        )

    paired = paired.dropna()

    paired["actual"] = paired["actual_a"]

    return paired[
        ["date", "actual", "forecast_a", "forecast_b"]
    ].sort_values("date")


# ============================================================
# Run all comparisons
# ============================================================

def run_tests(df, output_path=None):
    results = []

    def add_result(commodity, comparison, test_name,
                   mean_diff, statistic, n, p_value,
                   alternative, interpretation):

        results.append({
            "commodity": commodity,
            "comparison": comparison,
            "test": test_name,
            "n_obs": n,
            "mean_adjusted_loss_difference": mean_diff,
            "test_statistic": statistic,
            "p_value": p_value,
            "alternative": alternative,
            "interpretation": interpretation,
        })

    for commodity in COMMODITIES:

        # ----------------------------------------------------
        # A. ML models versus historical mean benchmark
        # ----------------------------------------------------

        for feature_set in FEATURE_SETS:

            for model in ML_MODELS:

                paired = get_paired_forecasts(
                    df,
                    commodity,
                    model,
                    feature_set,
                    "HistoricalMean",
                    "M0_Benchmark",
                )

                mean_d, statistic, n, p_value = diebold_mariano(
                    paired["actual"].values,
                    paired["forecast_a"].values,
                    paired["forecast_b"].values,
                )

                interpretation = (
                    "ML model has lower squared error"
                    if mean_d > 0
                    else "Historical mean has lower squared error"
                )

                add_result(
                    commodity,
                    f"{feature_set}_{model}_vs_HistoricalMean",
                    "Diebold-Mariano",
                    mean_d,
                    statistic,
                    n,
                    p_value,
                    "two-sided",
                    interpretation,
                )

        # ----------------------------------------------------
        # B. Weather versus financial-only
        # ----------------------------------------------------

        for model in ML_MODELS:

            for weather_set in ["M2_WSI", "M3_Components"]:

                paired = get_paired_forecasts(
                    df,
                    commodity,
                    model,
                    "M1_Financial",
                    model,
                    weather_set,
                )

                if model == "Ridge":
                    # Clark-West: nested regression comparison
                    mean_d, statistic, n, p_value = clark_west(
                        paired["actual"].values,
                        paired["forecast_a"].values,
                        paired["forecast_b"].values,
                    )

                    test_name = "Clark-West"
                    alternative = "one-sided: weather improves"

                else:
                    # Supplementary paired DM comparison
                    mean_d, statistic, n, p_value = diebold_mariano(
                        paired["actual"].values,
                        paired["forecast_b"].values,
                        paired["forecast_a"].values,
                    )

                    test_name = "Diebold-Mariano"
                    alternative = "two-sided"

                interpretation = (
                    "Weather model improves forecast accuracy"
                    if mean_d > 0
                    else "Financial-only model has lower adjusted loss"
                )

                add_result(
                    commodity,
                    f"{weather_set}_{model}_vs_M1_Financial",
                    test_name,
                    mean_d,
                    statistic,
                    n,
                    p_value,
                    alternative,
                    interpretation,
                )

        # ----------------------------------------------------
        # C. DM: WSI versus individual components
        # ----------------------------------------------------

        for model in ML_MODELS:

            paired = get_paired_forecasts(
                df,
                commodity,
                model,
                "M2_WSI",
                model,
                "M3_Components",
            )

            mean_d, statistic, n, p_value = diebold_mariano(
                paired["actual"].values,
                paired["forecast_a"].values,
                paired["forecast_b"].values,
            )

            interpretation = (
                "WSI has lower squared error"
                if mean_d > 0
                else "Individual components have lower squared error"
            )

            add_result(
                commodity,
                f"M2_WSI_{model}_vs_M3_Components",
                "Diebold-Mariano",
                mean_d,
                statistic,
                n,
                p_value,
                "two-sided",
                interpretation,
            )

    results_df = pd.DataFrame(results)

    results_df["significant_10pct"] = (
        results_df["p_value"] < 0.10
    )

    results_df["significant_5pct"] = (
        results_df["p_value"] < 0.05
    )

    results_df["significant_1pct"] = (
        results_df["p_value"] < 0.01
    )

    if output_path is None:
        output_path = OUTPUT_DIR / "forecast_comparison_tests.csv"
        results_df.to_csv(output_path, index=False)

        print("\nStatistical tests completed.")
        print(f"Results saved to: {output_path}")
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)

        print("\nStatistical tests completed.")
        print(f"Results saved to: {output_path}")

    print("\nResults by commodity and test:")
    print(
        results_df.groupby(
            ["commodity", "test"]
        ).size().to_string()
    )

    print("\nSignificant results at 5%:")
    significant = results_df[
        results_df["significant_5pct"]
    ]

    if significant.empty:
        print("No statistically significant results at the 5% level.")
    else:
        print(
            significant[
                [
                    "commodity",
                    "comparison",
                    "test",
                    "mean_adjusted_loss_difference",
                    "test_statistic",
                    "p_value",
                ]
            ].to_string(index=False)
        )

    return results_df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    predictions = load_predictions()
    results = run_tests(predictions)
