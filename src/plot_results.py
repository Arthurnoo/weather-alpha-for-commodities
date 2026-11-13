
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")

SUMMARY_PATH = METRICS_DIR / "metrics_summary.csv"
ANNUAL_PATH = METRICS_DIR / "metrics_annual.csv"

COMMODITIES = ["corn", "soybeans"]

FEATURE_ORDER = [
    "M1_Financial",
    "M2_WSI",
    "M3_Components",
]

FEATURE_LABELS = {
    "M1_Financial": "Financial",
    "M2_WSI": "Financial + WSI",
    "M3_Components": "Financial + Components",
}

MODEL_ORDER = [
    "Ridge",
    "RandomForest",
    "HistGradientBoosting",
]

MODEL_LABELS = {
    "Ridge": "Ridge",
    "RandomForest": "Random Forest",
    "HistGradientBoosting": "Gradient Boosting",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_metrics():

    summary = pd.read_csv(SUMMARY_PATH)
    annual = pd.read_csv(ANNUAL_PATH)

    return summary, annual


# ============================================================
# GENERAL PLOT SETTINGS
# ============================================================

def configure_plot():

    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_figure(fig, filename):

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    path = FIGURES_DIR / filename

    fig.savefig(path)
    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# 1. OVERALL OOS R-SQUARED
# ============================================================

def plot_oos_r2(summary):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 6),
        sharey=True,
    )

    for ax, commodity in zip(axes, COMMODITIES):

        data = summary.loc[
            (summary["commodity"] == commodity)
            & (summary["model"] != "HistoricalMean")
        ].copy()

        x = np.arange(len(MODEL_ORDER))
        width = 0.23

        for i, feature_set in enumerate(FEATURE_ORDER):

            values = []

            for model in MODEL_ORDER:

                row = data.loc[
                    (data["model"] == model)
                    & (data["feature_set"] == feature_set),
                    "OOS_R2",
                ]

                values.append(
                    row.iloc[0] if not row.empty else np.nan
                )

            ax.bar(
                x + (i - 1) * width,
                values,
                width,
                label=FEATURE_LABELS[feature_set],
            )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1.2,
            label="Historical mean benchmark",
        )

        ax.set_title(commodity.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODEL_ORDER],
            rotation=15,
        )
        ax.set_ylabel("Out-of-sample R²")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Out-of-sample R²: model and feature-set comparison",
        fontsize=14,
    )

    fig.tight_layout()

    save_figure(fig, "01_oos_r2.png")


# ============================================================
# 2. RMSE COMPARISON
# ============================================================

def plot_rmse(summary):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 6),
    )

    for ax, commodity in zip(axes, COMMODITIES):

        data = summary.loc[
            (summary["commodity"] == commodity)
            & (summary["model"] != "HistoricalMean")
        ].copy()

        x = np.arange(len(MODEL_ORDER))
        width = 0.23

        for i, feature_set in enumerate(FEATURE_ORDER):

            values = []

            for model in MODEL_ORDER:

                row = data.loc[
                    (data["model"] == model)
                    & (data["feature_set"] == feature_set),
                    "RMSE",
                ]

                values.append(
                    row.iloc[0] if not row.empty else np.nan
                )

            ax.bar(
                x + (i - 1) * width,
                values,
                width,
                label=FEATURE_LABELS[feature_set],
            )

        ax.set_title(commodity.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODEL_ORDER],
            rotation=15,
        )
        ax.set_ylabel("RMSE (weekly log returns)")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Root Mean Squared Error: model comparison",
        fontsize=14,
    )

    fig.tight_layout()

    save_figure(fig, "02_rmse.png")


# ============================================================
# 3. ANNUAL OOS R-SQUARED
# ============================================================

def plot_annual_oos_r2(annual):

    for commodity in COMMODITIES:

        fig, axes = plt.subplots(
            1, 3,
            figsize=(17, 5),
            sharey=True,
        )

        for ax, model in zip(axes, MODEL_ORDER):

            data = annual.loc[
                (annual["commodity"] == commodity)
                & (annual["model"] == model)
            ].copy()

            for feature_set in FEATURE_ORDER:

                subset = data.loc[
                    data["feature_set"] == feature_set
                ].sort_values("year")

                ax.plot(
                    subset["year"],
                    subset["OOS_R2"],
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    label=FEATURE_LABELS[feature_set],
                )

            ax.axhline(
                0,
                linestyle="--",
                linewidth=1,
            )

            ax.set_title(MODEL_LABELS[model])
            ax.set_xlabel("Year")
            ax.set_xticks(
                sorted(data["year"].unique())[::2]
            )
            ax.tick_params(axis="x", rotation=30)

        axes[0].set_ylabel("Annual out-of-sample R²")

        axes[-1].legend(
            fontsize=8,
            loc="best",
        )

        fig.suptitle(
            f"Annual forecasting performance: {commodity.upper()}",
            fontsize=14,
        )

        fig.tight_layout()

        save_figure(
            fig,
            f"03_annual_oos_r2_{commodity}.png",
        )


# ============================================================
# 4. DIRECTIONAL ACCURACY
# ============================================================

def plot_directional_accuracy(summary):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 6),
        sharey=True,
    )

    for ax, commodity in zip(axes, COMMODITIES):

        data = summary.loc[
            (summary["commodity"] == commodity)
            & (summary["model"] != "HistoricalMean")
        ].copy()

        x = np.arange(len(MODEL_ORDER))
        width = 0.23

        for i, feature_set in enumerate(FEATURE_ORDER):

            values = []

            for model in MODEL_ORDER:

                row = data.loc[
                    (data["model"] == model)
                    & (data["feature_set"] == feature_set),
                    "Directional_Accuracy",
                ]

                values.append(
                    row.iloc[0] * 100 if not row.empty else np.nan
                )

            ax.bar(
                x + (i - 1) * width,
                values,
                width,
                label=FEATURE_LABELS[feature_set],
            )

        ax.axhline(
            50,
            linestyle="--",
            linewidth=1.2,
            label="50% reference",
        )

        ax.set_title(commodity.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODEL_ORDER],
            rotation=15,
        )
        ax.set_ylabel("Correct direction (%)")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Directional accuracy: model comparison",
        fontsize=14,
    )

    fig.tight_layout()

    save_figure(fig, "04_directional_accuracy.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    configure_plot()

    summary, annual = load_metrics()

    plot_oos_r2(summary)
    plot_rmse(summary)
    plot_annual_oos_r2(annual)
    plot_directional_accuracy(summary)

    print("\nAll visualisations generated successfully.")
