"""
src/pipeline.py
---------------
Orchestrates the full daily ingestion pipeline:
  1. Scrape news from all sources
  2. Classify sentiment with FinBERT
  3. Download price data from yfinance
  4. Compute sentiment × return correlations
  5. Persist everything to PostgreSQL

Run manually:
    python -m src.pipeline

Or schedule via cron / Airflow:
    0 18 * * 1-5  cd /path/to/project && python -m src.pipeline
"""

from __future__ import annotations
import logging
import os
from datetime import datetime

from src.scraper           import fetch_news
from src.sentiment         import classify_articles
from src.price_correlation import (
    download_prices, build_sentiment_series,
    correlate, sector_sentiment, SECTOR_MAP,
)
from src.database import (
    get_conn, init_db, upsert_articles,
    upsert_daily_sentiment, upsert_correlation,
)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "META", "GOOGL",   # Technology
    "AMZN", "TSLA",                              # Consumer
    "JPM",  "BAC",                               # Financials
    "JNJ",  "UNH",                               # Healthcare
    "XOM",                                       # Energy
]
NEWSAPI_KEY  = os.getenv("NEWSAPI_KEY", "")
DAYS_BACK    = int(os.getenv("DAYS_BACK", "30"))
RETURN_HORIZON = 5   # trading days


def run():
    logger.info("=" * 60)
    logger.info(f"Pipeline started — {datetime.utcnow().isoformat()}")
    logger.info("=" * 60)

    conn = get_conn()
    init_db(conn)

    # 1. Scrape ───────────────────────────────────────────────────────────────
    logger.info("Step 1/5 — Scraping news …")
    articles = fetch_news(TICKERS, newsapi_key=NEWSAPI_KEY, days_back=DAYS_BACK)
    if not articles:
        logger.warning("No articles fetched — check network / API keys.")
        return

    # 2. Sentiment ─────────────────────────────────────────────────────────────
    logger.info("Step 2/5 — Classifying sentiment …")
    results = classify_articles(articles)

    # 3. Persist articles + sentiment ─────────────────────────────────────────
    logger.info("Step 3/5 — Persisting to database …")
    upsert_articles(articles, results, conn)

    # 4. Price data + correlation ──────────────────────────────────────────────
    logger.info("Step 4/5 — Downloading prices & computing correlations …")
    prices          = download_prices(TICKERS, days_back=DAYS_BACK + 20)
    sentiment_daily = build_sentiment_series(results, articles)

    if not sentiment_daily.empty:
        upsert_daily_sentiment(sentiment_daily, conn)
        merged, corr_report = correlate(sentiment_daily, prices, horizon=RETURN_HORIZON)
        upsert_correlation(corr_report, RETURN_HORIZON, conn=conn)

        logger.info("Step 5/5 — Correlation summary:")
        for ticker, stats in sorted(corr_report.items(), key=lambda x: abs(x[1]["pearson_r"]), reverse=True):
            sig = "✓ significant" if stats["significant"] else "  n.s."
            logger.info(f"  {ticker:<6} r={stats['pearson_r']:+.3f}  p={stats['p_value']:.3f}  "
                        f"n={stats['n_obs']}  {sig}")
    else:
        logger.warning("No sentiment data to correlate.")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    run()
