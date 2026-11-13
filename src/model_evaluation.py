
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTIONS_PATH = Path(
    "results/predictions/walk_forward_predictions.parquet"
)

OUTPUT_DIR = Path("results/metrics")

SUMMARY_PATH = OUTPUT_DIR / "metrics_summary.csv"
ANNUAL_PATH = OUTPUT_DIR / "metrics_annual.csv"

BENCHMARK = "HistoricalMean"
FIRST_TEST_YEAR = 2010


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    predictions = pd.read_parquet(PREDICTIONS_PATH)

    predictions["date"] = pd.to_datetime(predictions["date"])

    return predictions


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(group, benchmark_predictions):

    group = group.set_index("date").sort_index()

    # Align benchmark forecasts with the model's dates.
    benchmark = benchmark_predictions.reindex(group.index)

    actual = group["actual"].to_numpy()
    predicted = group["prediction"].to_numpy()
    benchmark_predicted = benchmark["prediction"].to_numpy()

    if np.isnan(actual).any() or np.isnan(predicted).any():
        raise ValueError("Missing actual values or model predictions.")

    if np.isnan(benchmark_predicted).any():
        raise ValueError("Missing benchmark predictions.")

    errors = actual - predicted
    benchmark_errors = actual - benchmark_predicted

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    benchmark_sse = np.sum(benchmark_errors ** 2)
    model_sse = np.sum(errors ** 2)

    if benchmark_sse == 0:
        oos_r2 = np.nan
    else:
        oos_r2 = 1 - model_sse / benchmark_sse

    directional_accuracy = np.mean(
        np.sign(actual) == np.sign(predicted)
    )

    return {
        "n_obs": len(group),
        "OOS_R2": oos_r2,
        "RMSE": rmse,
        "MAE": mae,
        "Directional_Accuracy": directional_accuracy,
    }


# ============================================================
# EVALUATE MODELS
# ============================================================

def evaluate_all(predictions):

    summary_results = []
    annual_results = []

    for commodity in sorted(predictions["commodity"].unique()):

        commodity_data = predictions.loc[
            predictions["commodity"] == commodity
        ].copy()

        benchmark_data = commodity_data.loc[
            commodity_data["model"] == BENCHMARK
        ].set_index("date").sort_index()

        if benchmark_data.index.has_duplicates:
            raise ValueError(
                f"Duplicate benchmark dates for {commodity}."
            )

        # ----------------------------------------------------
        # Overall out-of-sample performance
        # ----------------------------------------------------

        for (feature_set, model_name), group in (
            commodity_data.loc[
                commodity_data["model"] != BENCHMARK
            ].groupby(["feature_set", "model"])
        ):

            metrics = calculate_metrics(
                group,
                benchmark_data,
            )

            summary_results.append({
                "commodity": commodity,
                "feature_set": feature_set,
                "model": model_name,
                **metrics,
            })

        # ----------------------------------------------------
        # Historical mean benchmark
        # ----------------------------------------------------

        benchmark_metrics = calculate_metrics(
            commodity_data.loc[
                commodity_data["model"] == BENCHMARK
            ],
            benchmark_data,
        )

        summary_results.append({
            "commodity": commodity,
            "feature_set": "M0_Benchmark",
            "model": BENCHMARK,
            **benchmark_metrics,
        })

        # ----------------------------------------------------
        # Annual performance
        # ----------------------------------------------------

        for year in sorted(
            commodity_data["date"].dt.year.unique()
        ):

            if year < FIRST_TEST_YEAR:
                continue

            annual_data = commodity_data.loc[
                commodity_data["date"].dt.year == year
            ]

            annual_benchmark = benchmark_data.loc[
                benchmark_data.index.year == year
            ]

            for (feature_set, model_name), group in (
                annual_data.loc[
                    annual_data["model"] != BENCHMARK
                ].groupby(["feature_set", "model"])
            ):

                metrics = calculate_metrics(
                    group,
                    annual_benchmark,
                )

                annual_results.append({
                    "commodity": commodity,
                    "year": year,
                    "feature_set": feature_set,
                    "model": model_name,
                    **metrics,
                })

            # Annual benchmark results.
            metrics = calculate_metrics(
                annual_data.loc[
                    annual_data["model"] == BENCHMARK
                ],
                annual_benchmark,
            )

            annual_results.append({
                "commodity": commodity,
                "year": year,
                "feature_set": "M0_Benchmark",
                "model": BENCHMARK,
                **metrics,
            })

    return (
        pd.DataFrame(summary_results),
        pd.DataFrame(annual_results),
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    predictions = load_predictions()

    summary, annual = evaluate_all(predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(SUMMARY_PATH, index=False)
    annual.to_csv(ANNUAL_PATH, index=False)

    # Display the overall results.
    print("\n" + "=" * 75)
    print("OUT-OF-SAMPLE PERFORMANCE")
    print("=" * 75)

    for commodity in sorted(summary["commodity"].unique()):

        print(f"\n--- {commodity.upper()} ---")

        results = summary.loc[
            summary["commodity"] == commodity
        ].sort_values(
            ["feature_set", "OOS_R2"],
            ascending=[True, False],
        )

        print(
            results.to_string(
                index=False,
                formatters={
                    "OOS_R2": "{:.4f}".format,
                    "RMSE": "{:.6f}".format,
                    "MAE": "{:.6f}".format,
                    "Directional_Accuracy": "{:.2%}".format,
                },
            )
        )

    print("\n" + "=" * 75)
    print("FILES SAVED")
    print("=" * 75)

    print(SUMMARY_PATH)
    print(ANNUAL_PATH)
