"""
scripts/daily_ingest.py
-----------------------
Run this script daily (cron or Task Scheduler) to:
  1. Fetch latest news
  2. Score with FinBERT
  3. Store articles + daily aggregate in PostgreSQL

Usage:
    python scripts/daily_ingest.py
    python scripts/daily_ingest.py --days 7   # backfill last 7 days
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper     import fetch_all
from src.sentiment   import FinBERTClassifier, aggregate_sentiment
from src.storage     import get_conn, upsert_articles, upsert_daily_sentiment

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers= [logging.StreamHandler(), logging.FileHandler("ingest.log")],
)
logger = logging.getLogger(__name__)


def run_ingestion(days_back: int = 1):
    logger.info(f"=== Daily ingestion started (days_back={days_back}) ===")

    # 1. Fetch news
    articles = fetch_all(days_back=days_back)
    if not articles:
        logger.warning("No articles fetched — exiting")
        return

    # 2. Score with FinBERT
    logger.info(f"Scoring {len(articles)} articles with FinBERT …")
    classifier = FinBERTClassifier()
    texts      = [f"{a.title}. {a.description}" for a in articles]
    results    = classifier.classify(texts)

    for article, result in zip(articles, results):
        article.sentiment       = result.label
        article.sentiment_score = result.score

    # 3. Aggregate & persist
    today = datetime.now(timezone.utc).date()
    agg   = aggregate_sentiment(results)

    logger.info(
        f"Aggregate: composite={agg['composite_score']:.4f}  "
        f"pos={agg['label_counts'].get('positive',0)}  "
        f"neg={agg['label_counts'].get('negative',0)}  "
        f"neu={agg['label_counts'].get('neutral',0)}"
    )

    conn = get_conn()
    upsert_articles(conn, articles)
    upsert_daily_sentiment(conn, today, agg)
    logger.info("=== Ingestion complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1, help="Days of news to ingest")
    args = parser.parse_args()
    run_ingestion(args.days)
