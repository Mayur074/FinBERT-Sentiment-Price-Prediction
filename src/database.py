"""
src/database.py
---------------
Persist news articles, sentiment scores, and correlation results
to PostgreSQL for the daily ingestion pipeline and API queries.
"""

from __future__ import annotations
import logging
from datetime import datetime

import psycopg2
import psycopg2.extras
import pandas as pd

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "finsentiment",
    "user":     "postgres",
    "password": "postgres",
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    id          SERIAL PRIMARY KEY,
    article_id  TEXT UNIQUE NOT NULL,
    ticker      TEXT NOT NULL,
    title       TEXT NOT NULL,
    source      TEXT,
    published   TIMESTAMPTZ,
    url         TEXT,
    inserted_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sentiment_scores (
    id          SERIAL PRIMARY KEY,
    article_id  TEXT REFERENCES articles(article_id) ON DELETE CASCADE,
    ticker      TEXT NOT NULL,
    label       TEXT NOT NULL,
    score       FLOAT,
    compound    FLOAT,
    model       TEXT DEFAULT 'finbert',
    scored_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS daily_sentiment (
    id              SERIAL PRIMARY KEY,
    ticker          TEXT NOT NULL,
    date            DATE NOT NULL,
    mean_compound   FLOAT,
    article_count   INT,
    positive_pct    FLOAT,
    negative_pct    FLOAT,
    neutral_pct     FLOAT,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS correlation_results (
    id          SERIAL PRIMARY KEY,
    ticker      TEXT NOT NULL,
    run_date    DATE NOT NULL,
    horizon     INT,
    pearson_r   FLOAT,
    p_value     FLOAT,
    n_obs       INT,
    significant BOOLEAN,
    UNIQUE(ticker, run_date, horizon)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_ticker ON sentiment_scores(ticker);
CREATE INDEX IF NOT EXISTS idx_daily_ticker_date ON daily_sentiment(ticker, date);
"""


def get_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    return conn


def init_db(conn=None):
    c = conn or get_conn()
    with c.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    logger.info("Database schema initialised.")
    return c


def upsert_articles(articles, sentiment_results, conn=None) -> int:
    """Save articles + their sentiment scores. Returns count inserted."""
    c = conn or get_conn()
    inserted = 0
    with c.cursor() as cur:
        for art, res in zip(articles, sentiment_results):
            try:
                cur.execute("""
                    INSERT INTO articles (article_id, ticker, title, source, published, url)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id) DO NOTHING
                """, (art._id, art.ticker, art.title, art.source, art.published, art.url))

                cur.execute("""
                    INSERT INTO sentiment_scores (article_id, ticker, label, score, compound)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (art._id, art.ticker, res.label, res.score, res.compound))
                inserted += 1
            except Exception as e:
                logger.warning(f"upsert_articles error: {e}")
    logger.info(f"Upserted {inserted} article-sentiment pairs.")
    return inserted


def upsert_daily_sentiment(daily_df: pd.DataFrame, conn=None):
    """Aggregate and store daily mean sentiment per ticker."""
    c = conn or get_conn()
    with c.cursor() as cur:
        for date, row in daily_df.iterrows():
            for ticker in daily_df.columns:
                val = row[ticker]
                if pd.isna(val):
                    continue
                cur.execute("""
                    INSERT INTO daily_sentiment (ticker, date, mean_compound)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (ticker, date) DO UPDATE
                    SET mean_compound = EXCLUDED.mean_compound
                """, (ticker, date.date(), float(val)))


def upsert_correlation(report: dict, horizon: int, run_date=None, conn=None):
    """Store correlation results from a pipeline run."""
    c    = conn or get_conn()
    date = run_date or datetime.utcnow().date()
    with c.cursor() as cur:
        for ticker, stats in report.items():
            cur.execute("""
                INSERT INTO correlation_results
                    (ticker, run_date, horizon, pearson_r, p_value, n_obs, significant)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, run_date, horizon) DO UPDATE
                SET pearson_r=EXCLUDED.pearson_r, p_value=EXCLUDED.p_value,
                    n_obs=EXCLUDED.n_obs, significant=EXCLUDED.significant
            """, (ticker, date, horizon, stats["pearson_r"], stats["p_value"],
                  stats["n_obs"], stats["significant"]))
    logger.info(f"Saved correlation results for {len(report)} tickers.")


def load_daily_sentiment(ticker: str, days: int = 90, conn=None) -> pd.DataFrame:
    c   = conn or get_conn()
    sql = """
        SELECT date, mean_compound FROM daily_sentiment
        WHERE ticker = %s AND date >= NOW() - INTERVAL '%s days'
        ORDER BY date
    """
    return pd.read_sql(sql, c, params=(ticker, days), index_col="date",
                       parse_dates=["date"])


def load_correlation_history(conn=None) -> pd.DataFrame:
    c   = conn or get_conn()
    sql = "SELECT * FROM correlation_results ORDER BY run_date DESC, ticker"
    return pd.read_sql(sql, c)
