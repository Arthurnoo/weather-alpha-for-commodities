"""
Turn daily futures closes into a clean weekly panel: returns, features, targets.

Main issue: Yahoo's continuous series ("ZC=F") jumps to the next contract when
the front one expires. The price gap on that day is a spread between two
contracts, not a market move. We locate the roll day from each exchange's
expiry rule and drop the return of that day.

Second issue: on its last trading day the expiring contract is thin (most
traders have already rolled) and can print absurd prices, e.g. soybeans on
2008-09-12: 1216 -> 1490 -> 1179. The return of the day before the roll
(the expiry day) is dropped too.

Output: data/processed/prices_weekly.parquet
        index (date, commodity), columns = M1 features + targets
Run:    python -m src.prices
"""

import numpy as np
import pandas as pd

from config import COMMODITIES, DATA_PROCESSED, DATA_RAW, WEEK_RULE

BIG_MOVE_SIGMAS = 2.0  # a "big move" is a weekly return beyond 2 standard deviations


# ---------------------------------------------------------------------------
# Roll days
# ---------------------------------------------------------------------------
def roll_day(trading_days: pd.DatetimeIndex, year: int, month: int, rule: str):
    """
    First trading day on the new contract, for the contract expiring in (year, month).

    Rules follow exchange expiry calendars (approximation of Yahoo's behaviour):
      cbot        grains expire the business day before the 15th
                  -> roll on the first trading day from the 15th
      ice_coffee  expiry 8 business days before the month's last business day
      ice_cocoa   expiry 11 business days before the month's last business day
      ice_sugar   expiry on the last business day of the previous month
                  -> roll on the first trading day of the delivery month
      nymex_ng    expiry 3 business days before the 1st of the delivery month
    """
    in_month = trading_days[(trading_days.year == year) & (trading_days.month == month)]
    if len(in_month) == 0:
        return None
    first = trading_days.get_loc(in_month[0])   # position of 1st trading day of month
    last = trading_days.get_loc(in_month[-1])   # position of last trading day of month

    if rule == "cbot":
        after_15 = in_month[in_month.day >= 15]
        return after_15[0] if len(after_15) else None
    if rule == "ice_coffee":
        return trading_days[last - 7]           # day after expiry (last - 8)
    if rule == "ice_cocoa":
        return trading_days[last - 10]          # day after expiry (last - 11)
    if rule == "ice_sugar":
        return trading_days[first]
    if rule == "nymex_ng":
        return trading_days[first - 2]          # day after expiry (first - 3)
    raise ValueError(f"unknown roll rule: {rule}")


def roll_days(trading_days: pd.DatetimeIndex, rule: str, months: list) -> pd.DatetimeIndex:
    """All roll days of one commodity over the sample."""
    days = [roll_day(trading_days, year, month, rule)
            for year in trading_days.year.unique() for month in months]
    return pd.DatetimeIndex([d for d in days if d is not None])


# ---------------------------------------------------------------------------
# Daily and weekly returns
# ---------------------------------------------------------------------------
def expiry_days(trading_days: pd.DatetimeIndex, rolls: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each expiring contract = the trading day before each roll day."""
    position = trading_days.get_indexer(rolls)
    return trading_days[position[position > 0] - 1]


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns with the roll-day and expiry-day returns set to NaN."""
    returns = {}
    for name, spec in COMMODITIES.items():
        close = prices[name].dropna()
        ret = np.log(close).diff()
        rolls = roll_days(close.index, spec["roll"], spec["months"])
        ret[ret.index.isin(rolls)] = np.nan                            # contract-switch jump
        ret[ret.index.isin(expiry_days(close.index, rolls))] = np.nan  # thin last day
        returns[name] = ret
    return pd.DataFrame(returns)


def weekly_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly features (known at the Friday close of week t) and targets (week t+1).

    Features (model M1):
      ret_1w, ret_4w, ret_12w, ret_52w   past cumulative log returns (momentum / reversal)
      vol_4w, vol_12w                    realized volatility, annualized
      month                              calendar month of the week
    Targets (all about week t+1, except target_4w):
      target_1w     log return of the next week
      target_4w     log return over the next 4 weeks
      target_rv     realized volatility of the next week (annualized)
      target_big    1 if |next week's return| > BIG_MOVE_SIGMAS x current weekly vol
      target_up     1 if the big move is upward  (supply shock: price spike)
      target_down   1 if the big move is downward
    Big moves are measured against the volatility known today, so a "big" week
    in a calm market counts as much as one in a turbulent market.
    """
    # Weekly return = sum of daily log returns of the week (roll day excluded)
    weekly = daily.resample(WEEK_RULE).sum(min_count=1)
    # Weekly realized variance = sum of squared daily returns within the week
    weekly_rv = np.sqrt((daily ** 2).resample(WEEK_RULE).sum(min_count=1) * 52)
    # Daily volatility measured over ~1 and ~3 months, read on Fridays
    vol_4w = daily.rolling(20, min_periods=15).std().resample(WEEK_RULE).last() * np.sqrt(252)
    vol_12w = daily.rolling(60, min_periods=45).std().resample(WEEK_RULE).last() * np.sqrt(252)

    frames = []
    for name in daily.columns:
        r = weekly[name]
        df = pd.DataFrame({
            "ret_1w": r,
            "ret_4w": r.rolling(4).sum(),
            "ret_12w": r.rolling(12).sum(),
            "ret_52w": r.rolling(52).sum(),
            "vol_4w": vol_4w[name],
            "vol_12w": vol_12w[name],
            # shift(-k): value of a future week aligned on today's row
            "target_1w": r.shift(-1),
            "target_4w": r.rolling(4).sum().shift(-4),
            "target_rv": weekly_rv[name].shift(-1),
        })
        # Threshold in return units: annualized daily vol converted to one week
        threshold = BIG_MOVE_SIGMAS * df["vol_12w"] / np.sqrt(52)
        nxt = df["target_1w"]
        known = nxt.notna() & threshold.notna()
        df["target_big"] = (nxt.abs() > threshold).astype(float).where(known)
        df["target_up"] = (nxt > threshold).astype(float).where(known)
        df["target_down"] = (nxt < -threshold).astype(float).where(known)
        df["month"] = df.index.month
        df["commodity"] = name
        frames.append(df)

    panel = pd.concat(frames).reset_index().rename(columns={"index": "date"})
    return panel.set_index(["date", "commodity"]).sort_index()


if __name__ == "__main__":
    prices = pd.read_parquet(DATA_RAW / "prices_daily.parquet")
    daily = daily_returns(prices)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(DATA_PROCESSED / "returns_daily.parquet")
    panel = weekly_panel(daily)
    panel.to_parquet(DATA_PROCESSED / "prices_weekly.parquet")
    print(panel.groupby("commodity")["ret_1w"].describe().round(4))
