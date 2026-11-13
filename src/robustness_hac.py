
from pathlib import Path

import pandas as pd

from src import statistical_tests as st


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "statistical_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HAC_LAG_VALUES = [0, 4, 8]


# ============================================================
# Load predictions once
# ============================================================

predictions = st.load_predictions()

all_results = []


# ============================================================
# Run statistical tests for each HAC specification
# ============================================================

for lag in HAC_LAG_VALUES:

    print("\n" + "=" * 70)
    print(f"RUNNING STATISTICAL TESTS WITH HAC LAGS = {lag}")
    print("=" * 70)

    # Update the HAC lag parameter used by the test functions
    st.HAC_LAGS = lag

    results = st.run_tests(predictions)

    results["hac_lags"] = lag

    all_results.append(results)


# ============================================================
# Combine and save results
# ============================================================

robustness_results = pd.concat(
    all_results,
    ignore_index=True
)

output_path = OUTPUT_DIR / "hac_robustness.csv"

robustness_results.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print("HAC ROBUSTNESS ANALYSIS COMPLETED")
print("=" * 70)

print(f"Results saved to: {output_path}")

# Focus on the central research question:
# Does adding weather information improve forecasting?

weather_results = robustness_results[
    robustness_results["comparison"].str.contains(
        "vs_M1_Financial"
    )
].copy()

print("\nWEATHER VS FINANCIAL-ONLY MODELS")
print(
    weather_results[
        [
            "commodity",
            "comparison",
            "test",
            "hac_lags",
            "mean_adjusted_loss_difference",
            "test_statistic",
            "p_value",
        ]
    ].to_string(index=False)
)

print("\nSIGNIFICANCE AT 5% BY HAC SPECIFICATION")

summary = (
    robustness_results
    .groupby("hac_lags")
    .agg(
        total_tests=("p_value", "count"),
        significant_5pct=("significant_5pct", "sum"),
    )
)

print(summary.to_string())
