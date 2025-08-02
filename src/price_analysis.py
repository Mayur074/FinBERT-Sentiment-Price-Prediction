"""
src/price_analysis.py
---------------------
Fetches historical OHLCV data for S&P 500 stocks,
computes rolling sentiment-price correlations, and
runs a simple backtesting simulation.
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

logger = logging.getLogger(__name__)

# Representative tickers across 5 sectors
SECTOR_TICKERS = {
    "Technology":  ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "Financials":  ["JPM", "BAC", "GS", "MS", "WFC"],
    "Energy":      ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Healthcare":  ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
    "Consumer":    ["AMZN", "TSLA", "HD", "NKE", "MCD"],
}

ALL_TICKERS = [t for tickers in SECTOR_TICKERS.values() for t in tickers]


def fetch_prices(
    tickers: list[str] | None = None,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Returns a DataFrame of daily close prices.
    Index = dates, columns = ticker symbols.
    """
    tickers = tickers or ALL_TICKERS
    logger.info(f"Fetching prices for {len(tickers)} tickers …")
    raw = yf.download(tickers, period=period, interval=interval,
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
    prices = prices.dropna(how="all")
    logger.info(f"Prices shape: {prices.shape}")
    return prices


def compute_returns(prices: pd.DataFrame, forward_days: int = 1) -> pd.DataFrame:
    """
    Computes forward-looking percentage returns.
    forward_days=1 → next-day return (most common for sentiment lead analysis).
    """
    return prices.pct_change(forward_days).shift(-forward_days)


def correlate_sentiment_with_returns(
    sentiment_series: pd.Series,    # DatetimeIndex → composite_score
    returns_df:       pd.DataFrame, # DatetimeIndex → ticker returns
    rolling_window:   int = 30,
) -> dict[str, dict]:
    """
    For each ticker, compute:
      - Pearson r between sentiment and forward returns
      - p-value
      - 30-day rolling correlation series
      - Signal: is the correlation statistically significant (p < 0.05)?
    """
    results = {}
    for ticker in returns_df.columns:
        ret = returns_df[ticker].dropna()
        sent = sentiment_series.reindex(ret.index).dropna()
        common = ret.index.intersection(sent.index)
        if len(common) < 10:
            continue
        x, y = sent.loc[common].values, ret.loc[common].values
        r, p  = stats.pearsonr(x, y)
        # Rolling correlation
        combined = pd.DataFrame({"sentiment": sent, "return": ret}).dropna()
        rolling_corr = combined["sentiment"].rolling(rolling_window).corr(combined["return"])
        results[ticker] = {
            "pearson_r":     round(float(r), 4),
            "p_value":       round(float(p), 4),
            "significant":   p < 0.05,
            "rolling_corr":  rolling_corr,
            "n_obs":         len(common),
        }
    return results


def run_backtest(
    sentiment_series: pd.Series,
    prices:           pd.DataFrame,
    threshold_buy:    float = 0.15,   # buy when sentiment > threshold
    threshold_sell:   float = -0.10,  # sell when sentiment < threshold
    initial_capital:  float = 10_000,
) -> pd.DataFrame:
    """
    Simple sentiment-driven long-only strategy.
    Signal: buy if composite_score > threshold_buy, sell to cash otherwise.
    Benchmarks against equal-weight buy-and-hold.

    Returns a DataFrame with columns:
      date, strategy_value, buyhold_value, sentiment_score
    """
    tickers = prices.columns.tolist()
    daily_returns = prices.pct_change().dropna()

    records = []
    strategy_cap = initial_capital
    buyhold_cap  = initial_capital
    in_market    = False

    for date, row in daily_returns.iterrows():
        score = sentiment_series.get(date, None)

        if score is not None:
            if score > threshold_buy:
                in_market = True
            elif score < threshold_sell:
                in_market = False

        avg_ret = row.mean()  # equal-weight portfolio return
        buyhold_cap  *= (1 + avg_ret)
        strategy_cap *= (1 + avg_ret) if in_market else 1.0

        records.append({
            "date":             date,
            "strategy_value":   round(strategy_cap, 2),
            "buyhold_value":    round(buyhold_cap, 2),
            "sentiment_score":  score,
            "in_market":        in_market,
        })

    return pd.DataFrame(records).set_index("date")


def summarise_sectors(correlation_results: dict[str, dict]) -> pd.DataFrame:
    """
    Aggregates correlation results by sector and returns a summary DataFrame.
    """
    rows = []
    for sector, tickers in SECTOR_TICKERS.items():
        sector_results = {t: correlation_results[t] for t in tickers if t in correlation_results}
        if not sector_results:
            continue
        avg_r    = np.mean([v["pearson_r"]  for v in sector_results.values()])
        avg_p    = np.mean([v["p_value"]    for v in sector_results.values()])
        sig_pct  = np.mean([v["significant"] for v in sector_results.values()]) * 100
        rows.append({
            "Sector":          sector,
            "Avg Pearson r":   round(avg_r, 4),
            "Avg p-value":     round(avg_p, 4),
            "% Significant":   round(sig_pct, 1),
            "Tickers":         ", ".join(sector_results.keys()),
        })
    return pd.DataFrame(rows).sort_values("Avg Pearson r", ascending=False)
