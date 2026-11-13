
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

from src import model_training as mt


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("results/feature_importance")

N_REPEATS = 20
RANDOM_STATE = 42


# ============================================================
# OUT-OF-SAMPLE PERMUTATION IMPORTANCE
# ============================================================

def calculate_importance(data, commodity):

    results = []

    test_years = sorted(
        data.index[data.index.year >= mt.FIRST_TEST_YEAR].year.unique()
    )

    for year in test_years:

        test_data = data.loc[data.index.year == year]

        if test_data.empty:
            continue

        first_test_date = test_data.index.min()

        # Expanding training window, identical to model_training.py.
        train_data = data.loc[data.index < first_test_date]

        if train_data.empty:
            continue

        y_train = train_data[mt.TARGET]
        y_test = test_data[mt.TARGET]

        print(f"\n{commodity.upper()} | {year}")

        for feature_set_name, features in mt.FEATURE_SETS.items():

            X_train = train_data[features]
            X_test = test_data[features]

            for model_name, model in mt.get_models().items():

                # Fit using only observations available before the test year.
                model.fit(X_train, y_train)

                # Baseline out-of-sample prediction error.
                baseline_predictions = model.predict(X_test)

                baseline_mse = mean_squared_error(
                    y_test,
                    baseline_predictions
                )

                # Shuffle one feature at a time and measure the increase
                # in out-of-sample MSE.
                permutation = permutation_importance(
                    estimator=model,
                    X=X_test,
                    y=y_test,
                    scoring="neg_mean_squared_error",
                    n_repeats=N_REPEATS,
                    random_state=RANDOM_STATE,
                    n_jobs=1
                )

                for i, feature in enumerate(features):

                    results.append({
                        "commodity": commodity,
                        "year": year,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "feature": feature,
                        "baseline_mse": baseline_mse,
                        "importance_mean": permutation.importances_mean[i],
                        "importance_std": permutation.importances_std[i],
                    })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = mt.load_dataset()

    all_results = []

    for commodity in mt.COMMODITIES:

        data = mt.prepare_commodity_data(df, commodity)

        print("\n" + "=" * 60)
        print(f"FEATURE IMPORTANCE: {commodity.upper()}")
        print("=" * 60)

        print("Usable observations:", len(data))

        importance_df = calculate_importance(data, commodity)

        all_results.append(importance_df)

    results_df = pd.concat(all_results, ignore_index=True)

    # Aggregate annual importance estimates.
    summary_df = (
        results_df
        .groupby(
            ["commodity", "model", "feature_set", "feature"],
            as_index=False
        )
        .agg(
            mean_importance=("importance_mean", "mean"),
            median_importance=("importance_mean", "median"),
            std_importance=("importance_mean", "std"),
            positive_year_share=(
                "importance_mean",
                lambda x: (x > 0).mean()
            ),
            n_years=("year", "nunique")
        )
        .sort_values(
            ["commodity", "model", "feature_set", "mean_importance"],
            ascending=[True, True, True, False]
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annual_path = OUTPUT_DIR / "permutation_importance_annual.csv"
    summary_path = OUTPUT_DIR / "permutation_importance_summary.csv"

    results_df.to_csv(annual_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE COMPLETED")
    print("=" * 60)

    print("Annual results:", annual_path)
    print("Aggregated results:", summary_path)

    print("\nTop features by commodity and model:")

    for (commodity, model, feature_set), group in summary_df.groupby(
        ["commodity", "model", "feature_set"]
    ):

        print(f"\n{commodity.upper()} | {model} | {feature_set}")

        print(
            group.nlargest(5, "mean_importance")[
                ["feature", "mean_importance", "positive_year_share"]
            ].to_string(index=False)
        )
