"""
src/sentiment.py
----------------
Runs FinBERT (ProsusAI/finbert) from Hugging Face on news headlines.
Returns a SentimentResult per article with label + confidence score.
Falls back to a lightweight VADER-based analyser when transformers
is not installed (useful for quick local testing without a GPU).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

Label = Literal["positive", "negative", "neutral"]


@dataclass
class SentimentResult:
    title:      str
    ticker:     str
    label:      Label
    score:      float          # confidence [0, 1]
    compound:   float          # signed score: +1 = max positive, -1 = max negative


# ── FinBERT pipeline ──────────────────────────────────────────────────────────

_pipeline = None   # lazy-loaded


def _load_finbert():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading FinBERT (ProsusAI/finbert) — first run downloads ~500 MB …")
        _pipeline = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,          # return all label scores
            device=-1,           # CPU; set to 0 for GPU
        )
        logger.info("FinBERT loaded successfully.")
        return _pipeline
    except ImportError:
        logger.warning("transformers not installed — falling back to VADER.")
        return None


def _finbert_classify(texts: list[str]) -> list[SentimentResult]:
    pipe   = _load_finbert()
    results: list[SentimentResult] = []

    # FinBERT has a 512-token limit — truncate long headlines just in case
    truncated = [t[:512] for t in texts]

    raw = pipe(truncated, batch_size=16, truncation=True)

    for text, scores_list in zip(texts, raw):
        # scores_list = [{'label': 'positive', 'score': 0.9}, ...]
        score_map = {s["label"].lower(): s["score"] for s in scores_list}
        pos = score_map.get("positive", 0.0)
        neg = score_map.get("negative", 0.0)
        neu = score_map.get("neutral",  0.0)

        label    = max(score_map, key=score_map.get)   # type: ignore[arg-type]
        confidence = score_map[label]
        compound   = pos - neg                          # signed score in [-1, 1]

        results.append(SentimentResult(
            title=text, ticker="", label=label,  # type: ignore[arg-type]
            score=round(confidence, 4), compound=round(compound, 4),
        ))
    return results


# ── VADER fallback ────────────────────────────────────────────────────────────

def _vader_classify(texts: list[str]) -> list[SentimentResult]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
    except ImportError:
        raise RuntimeError(
            "Install either 'transformers' (for FinBERT) or "
            "'vaderSentiment' (for fallback) before running."
        )
    results = []
    for text in texts:
        scores   = analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label, conf = "positive", scores["pos"]
        elif compound <= -0.05:
            label, conf = "negative", scores["neg"]
        else:
            label, conf = "neutral", scores["neu"]
        results.append(SentimentResult(
            title=text, ticker="", label=label,  # type: ignore[arg-type]
            score=round(conf, 4), compound=round(compound, 4),
        ))
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def classify_articles(articles) -> list[SentimentResult]:
    """
    Classify a list of NewsArticle objects.
    Uses FinBERT if transformers is installed, otherwise VADER.
    """
    if not articles:
        return []

    texts = [art.title + (" " + art.description if art.description else "") for art in articles]

    try:
        raw_results = _finbert_classify(texts)
        model_used  = "FinBERT"
    except Exception as e:
        logger.warning(f"FinBERT failed ({e}), using VADER fallback.")
        raw_results = _vader_classify(texts)
        model_used  = "VADER"

    logger.info(f"Classified {len(raw_results)} articles with {model_used}.")

    for res, art in zip(raw_results, articles):
        res.ticker = art.ticker

    return raw_results
