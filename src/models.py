"""
Walk-forward forecasts for the four feature sets M1 to M4.

  M1  market features only (past returns, volatility, month)
  M2  M1 + individual weather components
  M3  M1 + Weather Stress Indices
  M4  M1 + components + indices

Every January from FIRST_TEST_YEAR on, each model is refit on all previous
weeks and predicts the weeks of the new year. Nothing from the test year is
used for fitting, scaling or tuning.

Targets are either regressions (return, realized volatility) or classifications
(big move, up jump, down jump); see prices.py.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_PROCESSED, FIRST_TEST_YEAR

MARKET = ["ret_1w", "ret_4w", "ret_12w", "ret_52w", "vol_4w", "vol_12w",
          "month_sin", "month_cos"]
INDICES = ["wsi_cpri", "wsi_pca", "d_wsi_cpri", "d_wsi_pca"]
CLASSIFICATION_TARGETS = {"target_big", "target_up", "target_down"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset() -> tuple:
    """Merge prices, components and indices into one panel. Returns (data, component names)."""
    prices = pd.read_parquet(DATA_PROCESSED / "prices_weekly.parquet")
    components = pd.read_parquet(DATA_PROCESSED / "components_weekly.parquet")
    indices = pd.read_parquet(DATA_PROCESSED / "indices_weekly.parquet")

    # A component that does not exist for a commodity (e.g. frost for gas) = no stress
    components = components.fillna(0.0)
    comp_names = list(components.columns)
    # Weekly changes: markets react to news, i.e. to changes, more than to levels
    components = components.join(components.groupby(level="commodity").diff().add_prefix("d_"))
    indices = indices.join(indices.groupby(level="commodity").diff().add_prefix("d_"))

    data = prices.join(components, how="inner").join(indices, how="inner")
    # Month as a point on a circle: December is close to January
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    return data, comp_names


def feature_sets(comp_names: list) -> dict:
    components = comp_names + [f"d_{c}" for c in comp_names]
    return {"M1": MARKET,
            "M2": MARKET + components,
            "M3": MARKET + INDICES,
            "M4": MARKET + components + INDICES}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_model(kind: str, classification: bool):
    """
    ridge     linear benchmark; penalty chosen inside the training sample
    boosting  shallow gradient-boosted trees, can catch thresholds and interactions
              (e.g. heat only matters in July)
    """
    if kind == "ridge":
        if classification:
            # class_weight="balanced": big moves are rare (about 5 % of weeks)
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000))
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    if kind == "boosting":
        params = dict(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
                      l2_regularization=1.0, early_stopping=False)
        if classification:
            return HistGradientBoostingClassifier(class_weight="balanced", **params)
        return HistGradientBoostingRegressor(**params)
    raise ValueError(kind)


def walk_forward(data: pd.DataFrame, features: list, target: str, kind: str) -> pd.DataFrame:
    """Yearly refit, out-of-sample predictions. Works for one commodity or the panel."""
    classification = target in CLASSIFICATION_TARGETS
    df = data.dropna(subset=features + [target])
    years = df.index.get_level_values("date").year
    rows = []
    for year in range(FIRST_TEST_YEAR, years.max() + 1):
        train, test = df[years < year], df[years == year]
        if test.empty:
            continue
        # The targets of the last training weeks fall inside the test year
        # (horizon of 1 or 4 weeks): drop those weeks so nothing leaks into the fit
        horizon = 4 if target == "target_4w" else 1
        cutoff = test.index.get_level_values("date").min() - pd.Timedelta(weeks=horizon)
        train = train[train.index.get_level_values("date") < cutoff]
        model = make_model(kind, classification).fit(train[features], train[target])
        pred = (model.predict_proba(test[features])[:, 1] if classification
                else model.predict(test[features]))
        rows.append(pd.DataFrame({
            "y": test[target], "pred": pred,
            # Naive benchmark: the training average (historical mean / base rate)
            "benchmark": train[target].mean(),
        }, index=test.index))
    return pd.concat(rows)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def oos_r2(y, pred, benchmark) -> float:
    """Out-of-sample R2 against the historical mean (Campbell and Thompson, 2008)."""
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - benchmark) ** 2)


def score(pred: pd.DataFrame, target: str) -> dict:
    if target in CLASSIFICATION_TARGETS:
        brier = np.mean((pred["y"] - pred["pred"]) ** 2)
        return {"auc": roc_auc_score(pred["y"], pred["pred"]),
                "hit_rate_top10": pred.loc[pred["pred"] >= pred["pred"].quantile(0.9), "y"].mean(),
                "base_rate": pred["y"].mean(), "brier": brier}
    return {"r2_oos": oos_r2(pred["y"], pred["pred"], pred["benchmark"]),
            "corr": pred[["y", "pred"]].corr().iloc[0, 1]}


def run_all(target: str, kinds=("ridge", "boosting"), panel: bool = True) -> tuple:
    """
    All feature sets x model kinds, either pooled over commodities (panel=True)
    or one model per commodity. Returns (scores table, predictions).
    """
    data, comp_names = load_dataset()
    sets = feature_sets(comp_names)
    groups = [("panel", data)] if panel else list(data.groupby(level="commodity"))
    scores, preds = [], []
    for group, df in groups:
        for kind in kinds:
            for set_name, features in sets.items():
                p = walk_forward(df, features, target, kind)
                p = p.assign(group=group, model=kind, features=set_name)
                preds.append(p)
                scores.append({"group": group, "model": kind, "features": set_name,
                               **score(p, target)})
    return pd.DataFrame(scores), pd.concat(preds)
