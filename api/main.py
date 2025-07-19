"""
api/main.py
-----------
FastAPI REST API exposing sentiment scores, price correlations,
and sector trends. Used by the Streamlit dashboard and external consumers.

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations
from datetime import date, datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from src.database import (
    get_conn, load_daily_sentiment, load_correlation_history,
)
from src.scraper    import fetch_news
from src.sentiment  import classify_articles
from src.price_correlation import (
    download_prices, build_sentiment_series, correlate, sector_sentiment
)

app = FastAPI(
    title       = "Financial Sentiment API",
    description = "Real-time FinBERT sentiment scores + price correlation for S&P 500 stocks.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class SentimentPoint(BaseModel):
    date:          str
    mean_compound: float


class CorrelationResult(BaseModel):
    ticker:      str
    pearson_r:   float
    p_value:     float
    n_obs:       int
    significant: bool


class HeadlineSentiment(BaseModel):
    title:    str
    ticker:   str
    label:    str
    score:    float
    compound: float


class SectorTrend(BaseModel):
    date:   str
    sector: str
    score:  float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Financial Sentiment API is running."}


@app.get("/sentiment/{ticker}", response_model=list[SentimentPoint], tags=["Sentiment"])
def get_sentiment_history(
    ticker: str,
    days: int = Query(default=30, ge=1, le=365, description="Number of days of history"),
):
    """Return daily mean compound sentiment for a ticker."""
    try:
        df = load_daily_sentiment(ticker.upper(), days=days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No sentiment data for {ticker}")
    return [
        SentimentPoint(date=str(idx.date()), mean_compound=row["mean_compound"])
        for idx, row in df.iterrows()
    ]


@app.get("/correlation", response_model=list[CorrelationResult], tags=["Correlation"])
def get_correlations(
    min_r: float = Query(default=0.0, description="Minimum absolute Pearson r"),
    significant_only: bool = Query(default=False),
):
    """Return latest correlation results for all tickers."""
    try:
        df = load_correlation_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if df.empty:
        return []

    # Latest run per ticker
    df = df.sort_values("run_date", ascending=False).drop_duplicates("ticker")
    if significant_only:
        df = df[df["significant"] == True]
    df = df[df["pearson_r"].abs() >= min_r]

    return [
        CorrelationResult(
            ticker=row["ticker"], pearson_r=row["pearson_r"],
            p_value=row["p_value"], n_obs=int(row["n_obs"]),
            significant=bool(row["significant"]),
        )
        for _, row in df.iterrows()
    ]


@app.post("/classify", response_model=list[HeadlineSentiment], tags=["Sentiment"])
def classify_headlines(
    headlines: list[str] = Query(..., description="List of news headlines to classify"),
    ticker: str = Query(default="UNKNOWN"),
):
    """
    Classify arbitrary headlines using FinBERT on-the-fly.
    Useful for testing the model without running the full pipeline.
    """
    from src.scraper import NewsArticle
    from datetime import timezone
    dummy_articles = [
        NewsArticle(title=h, source="api", ticker=ticker.upper(),
                    published=datetime.now(timezone.utc))
        for h in headlines
    ]
    results = classify_articles(dummy_articles)
    return [
        HeadlineSentiment(title=r.title, ticker=r.ticker,
                          label=r.label, score=r.score, compound=r.compound)
        for r in results
    ]


@app.get("/sector-trends", response_model=list[SectorTrend], tags=["Sentiment"])
def get_sector_trends(days: int = Query(default=14, ge=1, le=90)):
    """Return rolling sector-level sentiment trends."""
    try:
        conn = get_conn()
        from src.price_correlation import SECTOR_MAP
        all_tickers = [t for tickers in SECTOR_MAP.values() for t in tickers]

        # Load per-ticker daily sentiment from DB
        dfs = {}
        for ticker in all_tickers:
            df = load_daily_sentiment(ticker, days=days, conn=conn)
            if not df.empty:
                dfs[ticker] = df["mean_compound"]

        if not dfs:
            return []

        combined = pd.DataFrame(dfs)
        sector_df = sector_sentiment(combined, SECTOR_MAP)

        results = []
        for idx, row in sector_df.iterrows():
            for sector in sector_df.columns:
                if pd.notna(row[sector]):
                    results.append(SectorTrend(
                        date=str(idx.date() if hasattr(idx, "date") else idx),
                        sector=sector, score=round(float(row[sector]), 4),
                    ))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
