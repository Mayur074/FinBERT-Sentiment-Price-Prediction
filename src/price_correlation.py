"""
src/price_correlation.py
------------------------
Downloads historical price data via yfinance and correlates
30-day rolling sentiment scores with forward stock returns.

Key outputs
-----------
- sentiment_price_df : merged DataFrame of daily sentiment + price
- correlation_report : per-ticker Pearson r and p-values
- sector_signals     : aggregated sector-level sentiment trend
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

logger = logging.getLogger(__name__)

# S&P 500 sectors used in the project
SECTOR_MAP = {
    "Technology":   ["AAPL", "MSFT", "NVDA", "META", "GOOGL"],
    "Consumer":     ["AMZN", "TSLA", "HD"],
    "Financials":   ["JPM", "BAC", "GS"],
    "Healthcare":   ["JNJ", "UNH", "PFE"],
    "Energy":       ["XOM", "CVX"],
}


def download_prices(tickers: list[str], days_back: int = 90) -> pd.DataFrame:
    """
    Download daily adjusted close prices for a list of tickers.
    Returns a DataFrame indexed by date with one column per ticker.
    """
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    df = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )["Close"]

    if isinstance(df, pd.Series):      # single ticker
        df = df.to_frame(name=tickers[0])

    df.index = pd.to_datetime(df.index).tz_localize(None)
    logger.info(f"Downloaded prices: {df.shape[0]} days × {df.shape[1]} tickers")
    return df


def build_sentiment_series(
    sentiment_results,           # list[SentimentResult]
    articles,                    # list[NewsArticle]
) -> pd.DataFrame:
    """
    Aggregate daily mean compound sentiment score per ticker.
    Returns a pivoted DataFrame: index=date, columns=ticker, values=compound score.
    """
    rows = []
    for res, art in zip(sentiment_results, articles):
        rows.append({
            "date":     art.published.date(),
            "ticker":   art.ticker,
            "compound": res.compound,
            "label":    res.label,
        })

    if not rows:
        return pd.DataFrame()

    df   = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # Daily mean sentiment per ticker
    daily = (
        df.groupby(["date", "ticker"])["compound"]
        .mean()
        .reset_index()
        .pivot(index="date", columns="ticker", values="compound")
        .sort_index()
    )
    return daily


def compute_forward_returns(prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Compute N-day forward returns from price data.
    forward_return[t] = (price[t+N] - price[t]) / price[t]
    """
    return prices.pct_change(horizon).shift(-horizon)


def correlate(
    sentiment_daily: pd.DataFrame,
    prices:          pd.DataFrame,
    horizon:         int = 5,
    rolling_window:  int = 30,
) -> tuple[pd.DataFrame, dict]:
    """
    Merge rolling sentiment score with forward returns.
    Returns:
      merged_df         : aligned date-indexed DataFrame
      correlation_report: dict of {ticker: {r, p_value, n_obs, significant}}
    """
    forward_returns = compute_forward_returns(prices, horizon)

    # Rolling mean sentiment (smooths noise)
    rolling_sentiment = sentiment_daily.rolling(window=rolling_window, min_periods=3).mean()

    # Align on date
    merged = pd.concat(
        [rolling_sentiment.add_suffix("_sentiment"),
         forward_returns.add_suffix("_return")],
        axis=1
    ).dropna(how="all")

    report = {}
    common_tickers = set(sentiment_daily.columns) & set(prices.columns)
    for ticker in common_tickers:
        s_col = f"{ticker}_sentiment"
        r_col = f"{ticker}_return"
        if s_col not in merged or r_col not in merged:
            continue
        aligned = merged[[s_col, r_col]].dropna()
        if len(aligned) < 5:
            continue
        r, p = stats.pearsonr(aligned[s_col], aligned[r_col])
        report[ticker] = {
            "pearson_r":   round(float(r), 4),
            "p_value":     round(float(p), 4),
            "n_obs":       len(aligned),
            "significant": p < 0.05,
        }

    logger.info(f"Correlation report: {len(report)} tickers")
    return merged, report


def sector_sentiment(
    sentiment_daily: pd.DataFrame,
    sector_map:      dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Average sentiment across tickers in each sector and return
    a DataFrame with columns = sector names.
    """
    if sector_map is None:
        sector_map = SECTOR_MAP

    sector_dfs = {}
    for sector, tickers in sector_map.items():
        cols = [t for t in tickers if t in sentiment_daily.columns]
        if cols:
            sector_dfs[sector] = sentiment_daily[cols].mean(axis=1)

    if not sector_dfs:
        return pd.DataFrame()

    return pd.DataFrame(sector_dfs).sort_index()
