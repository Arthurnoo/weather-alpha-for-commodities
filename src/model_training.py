from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/model_dataset.parquet")

OUTPUT_DIR = Path("results/predictions")
OUTPUT_PATH = OUTPUT_DIR / "walk_forward_predictions.parquet"

FIRST_TEST_YEAR = 2010

COMMODITIES = ["corn", "soybeans"]

TARGET = "target_1w"

MARKET_FEATURES = [
    "ret_1w",
    "ret_4w",
    "ret_12w",
    "vol_4w",
    "vol_12w",
    "month_sin",
    "month_cos",
]

WSI_FEATURES = [
    "wsi_cpri",
    "in_season",
]

COMPONENT_FEATURES = [
    "heat",
    "warm_nights",
    "heavy_rain",
    "frost",
    "dryness",
    "in_season",
]

FEATURE_SETS = {
    "M1_Financial": MARKET_FEATURES,
    "M2_WSI": MARKET_FEATURES + WSI_FEATURES,
    "M3_Components": MARKET_FEATURES + COMPONENT_FEATURES,
}


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models():

    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ])

    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=10,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    gradient_boosting = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])

    return {
        "Ridge": ridge,
        "RandomForest": random_forest,
        "HistGradientBoosting": gradient_boosting,
    }


# ============================================================
# PREPARE DATA
# ============================================================

def load_dataset():

    df = pd.read_parquet(DATA_PATH).sort_index()

    # Retain only the selected commodities.
    df = df.loc[
        df.index.get_level_values("commodity").isin(COMMODITIES)
    ].copy()

    # Make the season indicator numerical.
    df["in_season"] = df["in_season"].astype(float)

    return df


def prepare_commodity_data(df, commodity):

    data = df.xs(commodity, level="commodity").copy()
    data = data.sort_index()

    all_features = list(dict.fromkeys(
        MARKET_FEATURES
        + WSI_FEATURES
        + COMPONENT_FEATURES
    ))

    # Common sample across all model specifications.
    # This ensures fair comparisons between feature sets.
    data = data.dropna(
        subset=[TARGET] + all_features
    )

    return data


# ============================================================
# WALK-FORWARD FORECASTING
# ============================================================

def run_walk_forward(data, commodity):

    predictions = []

    models = get_models()

    test_years = sorted(
        data.index[data.index.year >= FIRST_TEST_YEAR].year.unique()
    )

    for year in test_years:

        # First available observation in the test year.
        test_data = data.loc[data.index.year == year]

        if test_data.empty:
            continue

        first_test_date = test_data.index.min()

        # Expanding training window.
        # At the first test date, all previous targets are known.
        train_data = data.loc[
            data.index < first_test_date
        ]

        if train_data.empty:
            continue

        y_train = train_data[TARGET]

        print(
            f"\n{commodity.upper()} | {year} | "
            f"Training observations: {len(train_data)}"
        )

        # ----------------------------------------------------
        # M0: Historical mean benchmark
        # ----------------------------------------------------

        historical_mean = y_train.mean()

        for date, row in test_data.iterrows():

            predictions.append({
                "date": date,
                "commodity": commodity,
                "model": "HistoricalMean",
                "feature_set": "M0_Benchmark",
                "actual": row[TARGET],
                "prediction": historical_mean,
                "refit_year": year,
                "n_train": len(train_data),
            })

        # ----------------------------------------------------
        # M1, M2, M3: Machine learning models
        # ----------------------------------------------------

        for feature_set_name, features in FEATURE_SETS.items():

            X_train = train_data[features]
            X_test = test_data[features]

            for model_name, model in models.items():

                # Fit once per year.
                model.fit(X_train, y_train)

                # Predict every week of the test year.
                forecast = model.predict(X_test)

                for date, actual, predicted in zip(
                    test_data.index,
                    test_data[TARGET],
                    forecast,
                ):

                    predictions.append({
                        "date": date,
                        "commodity": commodity,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "actual": actual,
                        "prediction": predicted,
                        "refit_year": year,
                        "n_train": len(train_data),
                    })

    return predictions


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_dataset()

    all_predictions = []

    for commodity in COMMODITIES:

        data = prepare_commodity_data(df, commodity)

        print("\n" + "=" * 60)
        print(f"COMMODITY: {commodity.upper()}")
        print("=" * 60)

        print("Usable observations:", len(data))
        print(
            "Date range:",
            data.index.min(),
            "to",
            data.index.max(),
        )

        commodity_predictions = run_walk_forward(
            data,
            commodity,
        )

        all_predictions.extend(commodity_predictions)

    predictions_df = pd.DataFrame(all_predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df.to_parquet(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("FORECASTING COMPLETED")
    print("=" * 60)

    print("Total predictions:", len(predictions_df))
    print("\nPredictions by commodity and specification:")

    print(
        predictions_df.groupby(
            ["commodity", "feature_set", "model"]
        ).size().to_string()
    )

    print("\nSaved to:", OUTPUT_PATH)
