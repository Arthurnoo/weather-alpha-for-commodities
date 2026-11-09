"""
Descriptive statistics and data checks used by the notebook.

Two families of checks:
  prices   moments, normality, tails, autocorrelation, volatility clustering,
           seasonality of volatility, leverage effect, correlations
           (stylized facts of Cont, 2001, and of the commodity literature)
  weather  annual cycle, distributions, persistence, trends, known extreme
           events, ENSO teleconnections, possible breaks in the data

Nothing here is used by the models: it only describes the data.
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_PROCESSED, DATA_RAW

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_prices() -> tuple:
    """(daily closes, clean daily log returns, weekly panel)."""
    closes = pd.read_parquet(DATA_RAW / "prices_daily.parquet")
    returns = pd.read_parquet(DATA_PROCESSED / "returns_daily.parquet")
    weekly = pd.read_parquet(DATA_PROCESSED / "prices_weekly.parquet")
    return closes, returns, weekly


def load_weather() -> pd.DataFrame:
    """Weekly weather, wide: index (commodity, zone, date), one column per variable."""
    long = pd.read_parquet(DATA_PROCESSED / "weather_weekly.parquet")
    return long.pivot_table(index=["commodity", "zone", "date"], columns="variable",
                            values="value").sort_index()


# ---------------------------------------------------------------------------
# Prices: distribution
# ---------------------------------------------------------------------------
def moments(returns: pd.DataFrame, periods_per_year: int) -> pd.DataFrame:
    """
    First four moments of each return series, plus the Jarque-Bera normality test.

    skew       3rd standardized moment: < 0 = long left tail (crashes),
               > 0 = long right tail (spikes). Normal: 0.
    ex_kurt    4th standardized moment minus 3: > 0 = fat tails. Normal: 0.
    jb_pvalue  Jarque-Bera test of skew = 0 and ex_kurt = 0 jointly.
               p < 0.05: normality rejected.
    """
    rows = {}
    for name, x in returns.items():
        x = x.dropna()
        rows[name] = {
            "n_obs": len(x),
            "mean_ann": x.mean() * periods_per_year,
            "vol_ann": x.std() * np.sqrt(periods_per_year),
            "skew": stats.skew(x),
            "ex_kurt": stats.kurtosis(x),  # Fisher definition: normal = 0
            "min": x.min(),
            "max": x.max(),
            "jb_pvalue": stats.jarque_bera(x).pvalue,
        }
    return pd.DataFrame(rows).T


def aggregate(returns: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Non-overlapping returns over a longer horizon (sum of daily log returns)."""
    return returns.resample(rule).sum(min_count=1)


def tail_frequency(returns: pd.DataFrame, sigmas=(2, 3, 4)) -> pd.DataFrame:
    """
    Share of observations beyond k standard deviations, against the normal law.
    Under normality, |z| > 2 happens 4.6 % of the time, |z| > 3 0.27 %, |z| > 4 0.006 %.
    """
    z = (returns - returns.mean()) / returns.std()
    out = pd.DataFrame({f"|z|>{k}": (z.abs() > k).sum() / z.notna().sum() for k in sigmas}).T
    out["normal"] = [2 * stats.norm.sf(k) for k in sigmas]
    return out.T


# ---------------------------------------------------------------------------
# Prices: dependence over time
# ---------------------------------------------------------------------------
def acf(x: pd.Series, lags: int) -> pd.Series:
    """Autocorrelation at lags 1..lags (NaNs dropped pairwise)."""
    return pd.Series({k: x.corr(x.shift(k)) for k in range(1, lags + 1)})


def ljung_box(x: pd.Series, lags: int) -> tuple:
    """
    Ljung-Box test that the first `lags` autocorrelations are all zero.
    Returns (Q statistic, p-value). p < 0.05: significant autocorrelation.
    """
    x = x.dropna()
    n = len(x)
    rho = acf(x, lags).values
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return q, stats.chi2.sf(q, lags)


def autocorrelation_table(returns: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """
    Returns versus absolute returns. Stylized fact: returns have almost no
    autocorrelation, absolute returns have a strong and slowly decaying one
    (volatility clustering).
    """
    rows = {}
    for name, x in returns.items():
        q_r, p_r = ljung_box(x, lags)
        q_a, p_a = ljung_box(x.abs(), lags)
        rows[name] = {"acf1 r": x.autocorr(1), f"LB({lags}) p r": p_r,
                      "acf1 |r|": x.abs().autocorr(1), f"LB({lags}) p |r|": p_a}
    return pd.DataFrame(rows).T


def vol_by_month(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized volatility of daily returns by calendar month (rows = commodities)."""
    by_month = returns.groupby(returns.index.month).std() * np.sqrt(252)
    return by_month.T


def leverage_table(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Does volatility rise more after a fall (equity-style leverage effect) or
    after a rise (inverse leverage, expected for commodities where a price
    spike signals scarcity: Carpantier, 2010)?

    corr       correlation between this week's return and next week's realized vol
    vol after down / up   mean next-week realized vol after the 10 % worst and
                          10 % best weeks
    """
    rows = {}
    for name, g in weekly.groupby(level="commodity"):
        r, rv = g["ret_1w"], g["target_rv"]
        low, high = r.quantile(0.1), r.quantile(0.9)
        rows[name] = {"corr(r, next rv)": r.corr(rv),
                      "next rv after down": rv[r <= low].mean(),
                      "next rv after up": rv[r >= high].mean()}
    return pd.DataFrame(rows).T


def rolling_mean_correlation(weekly_returns: pd.DataFrame, window: int = 104) -> pd.Series:
    """Average of all pairwise correlations over a rolling window of weeks."""
    out = {}
    for end in range(window, len(weekly_returns) + 1):
        c = weekly_returns.iloc[end - window:end].corr().values
        out[weekly_returns.index[end - 1]] = c[np.triu_indices_from(c, k=1)].mean()
    return pd.Series(out)


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def seasonal_value(series: pd.Series, months: list, agg: str = "sum",
                   cross_year: bool = False) -> pd.Series:
    """
    One value per year for a season (e.g. June to August EDD total).

    cross_year=True for seasons spanning New Year (e.g. November to March):
    October to December are counted with the following year, so the 2013-14
    winter is labelled 2014. Incomplete seasons (first and last) are dropped.
    """
    s = series[series.index.month.isin(months)]
    year = s.index.year + ((s.index.month >= 10) if cross_year else 0)
    grouped = s.groupby(year)
    n_weeks = grouped.size()
    # A season has 8 or 9 Fridays depending on the year: only clearly truncated
    # seasons (start and end of the sample) are dropped
    full = n_weeks >= 0.75 * n_weeks.median()
    return grouped.agg(agg)[full]


def weekly_anomaly(series: pd.Series) -> pd.Series:
    """Deviation from the full-sample mean of the same week of the year (descriptive only)."""
    week = np.minimum(series.index.isocalendar().week.values, 52)
    return series - series.groupby(week).transform("mean")


def trend_per_decade(annual: pd.Series) -> float:
    """OLS slope of an annual series, per 10 years."""
    return np.polyfit(annual.index.values.astype(float), annual.values, 1)[0] * 10


# Known events: each one should rank near the top of its column if the data and
# the production weights are right. "highest" / "lowest": which tail to look at.
KNOWN_EVENTS = [
    # event, commodity, zone, variable, months, agg, cross_year, year, tail
    ("US Corn Belt drought and heat 2012", "corn", "US", "edd", [6, 7, 8], "sum", False, 2012, "highest"),
    ("US Corn Belt drought 2012 (rain)", "corn", "US", "precip", [6, 7, 8], "sum", False, 2012, "lowest"),
    ("US Midwest flood 1993", "corn", "US", "precip", [6, 7, 8], "sum", False, 1993, "highest"),
    ("US polar vortex winter 2013-14", "natgas", "US", "hdd", [11, 12, 1, 2, 3], "sum", True, 2014, "highest"),
    ("US warm El Nino winter 2015-16", "natgas", "US", "hdd", [11, 12, 1, 2, 3], "sum", True, 2016, "lowest"),
    ("Russian heat wave 2010", "wheat_srw", "BLACK_SEA", "tmax", [7, 8], "mean", False, 2010, "highest"),
    ("European heat wave 2003", "wheat_srw", "EU", "tmax", [6, 7, 8], "mean", False, 2003, "highest"),
    ("Argentine drought 2022-23", "soybeans", "ARG", "precip", [12, 1, 2, 3], "sum", True, 2023, "lowest"),
    ("Argentine drought 2017-18", "soybeans", "ARG", "precip", [12, 1, 2, 3], "sum", True, 2018, "lowest"),
    ("Brazil coffee drought 2013-14", "coffee", "BRA", "precip", [10, 11, 12, 1, 2, 3], "sum", True, 2014, "lowest"),
    ("Indian monsoon failure 2002", "sugar", "IND", "precip", [6, 7, 8, 9], "sum", False, 2002, "lowest"),
    ("Indian monsoon failure 2009", "sugar", "IND", "precip", [6, 7, 8, 9], "sum", False, 2009, "lowest"),
    ("West Africa heavy rains 2023 (cocoa black pod)", "cocoa", "WAF", "precip", [4, 5, 6, 7], "sum", False, 2023, "highest"),
]


def check_events(weather: pd.DataFrame) -> pd.DataFrame:
    """Rank of each known event among all years of the sample (1 = most extreme)."""
    rows = []
    for event, com, zone, var, months, agg, cross, year, tail in KNOWN_EVENTS:
        annual = seasonal_value(weather.loc[(com, zone), var], months, agg, cross)
        ranks = annual.rank(ascending=(tail == "lowest"))
        rows.append({"event": event, "series": f"{com}/{zone} {var}", "year": year,
                     "value": annual.get(year, np.nan), "sample mean": annual.mean(),
                     "rank": ranks.get(year, np.nan), "n years": len(annual)})
    return pd.DataFrame(rows).set_index("event")


# ENSO teleconnections (Ropelewski and Halpert, 1987; IRI): expected sign of the
# correlation between the ONI and seasonal rainfall
ENSO_TESTS = [
    # label, commodity, zone, months, cross_year, expected sign
    ("India monsoon (Jun-Sep): El Nino -> dry", "sugar", "IND", [6, 7, 8, 9], False, -1),
    ("Thailand (Jun-Sep): El Nino -> dry", "sugar", "THA", [6, 7, 8, 9], False, -1),
    ("Colombia (Dec-Feb): El Nino -> dry", "coffee", "COL", [12, 1, 2], True, -1),
    ("Argentina (Nov-Jan): El Nino -> wet", "soybeans", "ARG", [11, 12, 1], True, 1),
    ("US southern Plains (Dec-Feb): El Nino -> wet", "wheat_hrw", "US", [12, 1, 2], True, 1),
    ("Central Brazil (Dec-Feb): weak link", "soybeans", "BRA", [12, 1, 2], True, 0),
    ("West Africa (Jun-Aug): weak link", "cocoa", "WAF", [6, 7, 8], False, 0),
]


def enso_table(weather: pd.DataFrame, oni: pd.Series) -> pd.DataFrame:
    rows = []
    for label, com, zone, months, cross, sign in ENSO_TESTS:
        rain = seasonal_value(weather.loc[(com, zone), "precip"], months, "sum", cross)
        nino = seasonal_value(oni, months, "mean", cross)
        both = pd.concat([rain, nino], axis=1, keys=["rain", "oni"]).dropna()
        rows.append({"teleconnection": label, "expected sign": sign,
                     "corr": both["rain"].corr(both["oni"]), "n years": len(both)})
    return pd.DataFrame(rows).set_index("teleconnection")


def break_table(weather: pd.DataFrame, variable: str = "precip", year: int = 2006) -> pd.DataFrame:
    """
    Mean of a variable before and after a given year, for every zone.
    CPC precipitation switches in 2006 from its retrospective version (~30,000
    gauges) to the real-time version (~17,000 gauges), and the number of
    reporting stations keeps changing after that: a large shift in the mean
    can be a data artefact rather than a climate change.
    """
    rows = {}
    for (com, zone), g in weather.groupby(level=["commodity", "zone"]):
        s = g[variable].droplevel(["commodity", "zone"])
        before, after = s[s.index.year < year], s[s.index.year >= year]
        rows[f"{com}/{zone}"] = {"before": before.mean(), "after": after.mean(),
                                 "change %": 100 * (after.mean() / before.mean() - 1),
                                 "t-stat": stats.ttest_ind(weekly_anomaly(s)[s.index.year >= year],
                                                           weekly_anomaly(s)[s.index.year < year],
                                                           equal_var=False).statistic}
    return pd.DataFrame(rows).T
