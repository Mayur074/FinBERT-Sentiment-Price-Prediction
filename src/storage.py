"""
src/storage.py
--------------
PostgreSQL persistence layer.
Stores articles + sentiment scores for daily ingestion
and backtesting replay.
"""

from __future__ import annotations
import logging
from datetime import datetime

import psycopg2
import pandas as pd
from src.scraper import NewsArticle

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "finsentiment",
    "user":     "postgres",
    "password": "postgres",
}

CREATE_ARTICLES_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    article_id       TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    description      TEXT,
    source           TEXT,
    url              TEXT,
    published        TIMESTAMPTZ,
    tickers          TEXT[],
    sentiment_label  TEXT,
    sentiment_score  FLOAT,
    positive_prob    FLOAT,
    negative_prob    FLOAT,
    neutral_prob     FLOAT,
    inserted_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_DAILY_SENTIMENT_SQL = """
CREATE TABLE IF NOT EXISTS daily_sentiment (
    date             DATE PRIMARY KEY,
    composite_score  FLOAT,
    positive_count   INT,
    negative_count   INT,
    neutral_count    INT,
    total_articles   INT,
    mean_confidence  FLOAT,
    inserted_at      TIMESTAMPTZ DEFAULT NOW()
);
"""


def get_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(CREATE_ARTICLES_SQL)
        cur.execute(CREATE_DAILY_SENTIMENT_SQL)
    return conn


def upsert_articles(conn, articles: list[NewsArticle]):
    sql = """
    INSERT INTO articles
      (article_id, title, description, source, url, published,
       tickers, sentiment_label, sentiment_score,
       positive_prob, negative_prob, neutral_prob)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (article_id) DO UPDATE SET
      sentiment_label = EXCLUDED.sentiment_label,
      sentiment_score = EXCLUDED.sentiment_score,
      positive_prob   = EXCLUDED.positive_prob,
      negative_prob   = EXCLUDED.negative_prob,
      neutral_prob    = EXCLUDED.neutral_prob;
    """
    with conn.cursor() as cur:
        for a in articles:
            cur.execute(sql, (
                a.article_id, a.title, a.description, a.source, a.url,
                a.published, a.tickers or [],
                a.sentiment, a.sentiment_score, None, None, None,
            ))
    logger.info(f"Upserted {len(articles)} articles")


def upsert_daily_sentiment(conn, date: datetime.date, agg: dict):
    sql = """
    INSERT INTO daily_sentiment
      (date, composite_score, positive_count, negative_count,
       neutral_count, total_articles, mean_confidence)
    VALUES (%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (date) DO UPDATE SET
      composite_score = EXCLUDED.composite_score,
      positive_count  = EXCLUDED.positive_count,
      negative_count  = EXCLUDED.negative_count,
      neutral_count   = EXCLUDED.neutral_count,
      total_articles  = EXCLUDED.total_articles,
      mean_confidence = EXCLUDED.mean_confidence;
    """
    counts = agg.get("label_counts", {})
    with conn.cursor() as cur:
        cur.execute(sql, (
            date,
            agg["composite_score"],
            counts.get("positive", 0),
            counts.get("negative", 0),
            counts.get("neutral",  0),
            agg.get("n", 0),
            agg["mean_confidence"],
        ))


def load_daily_sentiment(conn, days: int = 180) -> pd.Series:
    """Returns a DatetimeIndex Series of composite_score for the past N days."""
    sql = f"""
    SELECT date, composite_score FROM daily_sentiment
    ORDER BY date DESC LIMIT {days}
    """
    df = pd.read_sql(sql, conn)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["composite_score"].sort_index()


def load_recent_articles(conn, limit: int = 100) -> pd.DataFrame:
    sql = f"""
    SELECT title, source, published, sentiment_label, sentiment_score
    FROM articles
    ORDER BY published DESC
    LIMIT {limit}
    """
    return pd.read_sql(sql, conn)
