"""
src/scraper.py
--------------
Fetches financial news headlines from multiple free sources:
  1. NewsAPI  (requires free API key — newsapi.org)
  2. Yahoo Finance RSS feeds (no key needed)
  3. Finviz news scraper    (no key needed)

Results are deduplicated and returned as a list of NewsArticle objects.
"""

from __future__ import annotations
import hashlib, logging, time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


@dataclass
class NewsArticle:
    title:       str
    source:      str
    ticker:      str
    published:   datetime
    url:         str = ""
    description: str = ""
    _id:         str = field(init=False)

    def __post_init__(self):
        self._id = hashlib.md5(self.title.encode()).hexdigest()


# ── NewsAPI ───────────────────────────────────────────────────────────────────

def fetch_newsapi(ticker: str, api_key: str, days_back: int = 30) -> list[NewsArticle]:
    """Fetch from NewsAPI. Free key at https://newsapi.org"""
    if not api_key:
        return []
    from_dt = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params  = {
        "q": f'"{ticker}" stock OR finance OR earnings',
        "from": from_dt, "sortBy": "publishedAt",
        "language": "en", "pageSize": 50, "apiKey": api_key,
    }
    try:
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        resp.raise_for_status()
        articles = []
        for art in resp.json().get("articles", []):
            try:
                pub = datetime.fromisoformat(art["publishedAt"].replace("Z", "+00:00"))
            except Exception:
                pub = datetime.now(timezone.utc)
            articles.append(NewsArticle(
                title=art.get("title", ""), source=art.get("source", {}).get("name", "NewsAPI"),
                ticker=ticker, published=pub, url=art.get("url", ""),
                description=art.get("description", "") or "",
            ))
        logger.info(f"[NewsAPI] {ticker}: {len(articles)} articles")
        return articles
    except Exception as e:
        logger.warning(f"[NewsAPI] {ticker} failed: {e}")
        return []


# ── Yahoo Finance RSS ─────────────────────────────────────────────────────────

def fetch_yahoo_rss(ticker: str) -> list[NewsArticle]:
    """Parse Yahoo Finance RSS — no API key required."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup, articles = BeautifulSoup(resp.content, "xml"), []
        for item in soup.find_all("item")[:30]:
            title   = item.find("title").get_text(strip=True) if item.find("title") else ""
            link    = item.find("link").get_text(strip=True)  if item.find("link")  else ""
            pub_str = item.find("pubDate").get_text(strip=True) if item.find("pubDate") else ""
            desc    = item.find("description").get_text(strip=True) if item.find("description") else ""
            try:
                pub = datetime.strptime(pub_str, "%a, %d %b %Y %H:%M:%S %z")
            except Exception:
                pub = datetime.now(timezone.utc)
            if title:
                articles.append(NewsArticle(title=title, source="Yahoo Finance",
                    ticker=ticker, published=pub, url=link, description=desc))
        logger.info(f"[Yahoo RSS] {ticker}: {len(articles)} articles")
        return articles
    except Exception as e:
        logger.warning(f"[Yahoo RSS] {ticker} failed: {e}")
        return []


# ── Finviz scraper ────────────────────────────────────────────────────────────

def fetch_finviz(ticker: str) -> list[NewsArticle]:
    """Scrape Finviz news headlines — no API key required."""
    try:
        resp  = requests.get(f"https://finviz.com/quote.ashx?t={ticker}", headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "news-table"})
        if not table:
            return []
        articles, last_date = [], datetime.now(timezone.utc)
        for row in table.find_all("tr")[:40]:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            date_str = cols[0].get_text(strip=True)
            title    = cols[1].get_text(strip=True)
            link_tag = cols[1].find("a")
            link     = link_tag["href"] if link_tag else ""
            source   = cols[1].find("span").get_text(strip=True) if cols[1].find("span") else "Finviz"
            try:
                if "Today" in date_str:
                    t = date_str.replace("Today", "").strip()
                    pub = datetime.now(timezone.utc).replace(
                        hour=int(t.split(":")[0]), minute=int(t.split(":")[1][:2]), second=0, microsecond=0)
                else:
                    parts = date_str.split()
                    pub = datetime.strptime(f"{parts[0]} {parts[1]}", "%b-%d-%y %I:%M%p").replace(tzinfo=timezone.utc) \
                        if len(parts) == 2 else last_date
            except Exception:
                pub = datetime.now(timezone.utc)
            last_date = pub
            if title:
                articles.append(NewsArticle(title=title, source=source,
                    ticker=ticker, published=pub, url=link))
        logger.info(f"[Finviz] {ticker}: {len(articles)} articles")
        return articles
    except Exception as e:
        logger.warning(f"[Finviz] {ticker} failed: {e}")
        return []


# ── Unified fetcher ────────────────────────────────────────────────────────────

def fetch_news(tickers: list[str], newsapi_key: str = "",
               days_back: int = 30, sleep_sec: float = 1.0) -> list[NewsArticle]:
    """Aggregate + deduplicate news from all sources for all tickers."""
    seen: set[str] = set()
    all_articles:  list[NewsArticle] = []
    for ticker in tickers:
        batch = fetch_newsapi(ticker, newsapi_key, days_back) + \
                fetch_yahoo_rss(ticker) + \
                fetch_finviz(ticker)
        time.sleep(sleep_sec)
        for art in batch:
            if art._id not in seen and art.title.strip():
                seen.add(art._id)
                all_articles.append(art)
    logger.info(f"Total unique articles: {len(all_articles)}")
    return all_articles
