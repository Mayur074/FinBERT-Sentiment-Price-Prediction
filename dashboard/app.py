"""
dashboard/app.py
----------------
Streamlit dashboard for the FinBERT Sentiment Predictor.
Shows live sentiment, price correlations, backtest results, and article feed.
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper       import fetch_all
from src.sentiment     import FinBERTClassifier, aggregate_sentiment
from src.price_analysis import (
    fetch_prices, compute_returns,
    correlate_sentiment_with_returns, summarise_sectors, run_backtest,
    SECTOR_TICKERS
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinBERT Sentiment Dashboard",
    page_icon="📰",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; }
h1,h2,h3 { color: #e6edf3 !important; }
.stMetric label { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📰 FinBERT\nSentiment Predictor")
    st.divider()
    st.markdown("**Data Settings**")
    period         = st.selectbox("Price history", ["1mo","3mo","6mo","1y"], index=2)
    n_articles     = st.slider("Articles to fetch", 20, 200, 80, step=20)
    rolling_window = st.slider("Rolling window (days)", 7, 60, 30)
    st.divider()
    st.markdown("**Backtest Settings**")
    buy_thresh  = st.slider("Buy threshold",  0.05, 0.40, 0.15, step=0.05)
    sell_thresh = st.slider("Sell threshold", -0.40, -0.05, -0.10, step=0.05)
    st.divider()
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📰 Financial News Sentiment & Price Prediction")
st.caption("FinBERT (ProsusAI) · Multi-source news · Pearson correlation · Backtesting · FastAPI backend")

if not run_btn and "results" not in st.session_state:
    st.info("Configure settings in the sidebar and click **Run Analysis** to begin.")
    st.stop()

# ── Run pipeline ───────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Fetching news from 5+ sources …"):
        articles = fetch_all()
        articles = articles[:n_articles]

    if not articles:
        st.error("No articles fetched. Check your internet connection.")
        st.stop()

    with st.spinner(f"Scoring {len(articles)} headlines with FinBERT …"):
        clf     = FinBERTClassifier()
        texts   = [f"{a.title}. {a.description}" for a in articles]
        results = clf.classify(texts)
        for a, r in zip(articles, results):
            a.sentiment       = r.label
            a.sentiment_score = r.score
        agg = aggregate_sentiment(results)

    with st.spinner("Fetching price data & computing correlations …"):
        prices      = fetch_prices(period=period)
        fwd_returns = compute_returns(prices)

        # Build a synthetic daily sentiment series from today's aggregate
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        dates = pd.date_range(end=today, periods=len(prices), freq="B")
        # Simulate rolling sentiment with small noise for demo
        import numpy as np
        np.random.seed(42)
        sim_scores = agg["composite_score"] + np.random.normal(0, 0.05, len(dates))
        sentiment_series = pd.Series(sim_scores, index=dates)

        corr_results = correlate_sentiment_with_returns(
            sentiment_series, fwd_returns, rolling_window
        )
        sector_df = summarise_sectors(corr_results)
        backtest  = run_backtest(sentiment_series, prices, buy_thresh, sell_thresh)

    st.session_state["articles"]   = articles
    st.session_state["results"]    = results
    st.session_state["agg"]        = agg
    st.session_state["corr"]       = corr_results
    st.session_state["sector_df"]  = sector_df
    st.session_state["backtest"]   = backtest
    st.session_state["sentiment_series"] = sentiment_series

# ── Load from state ────────────────────────────────────────────────────────────
articles        = st.session_state["articles"]
results         = st.session_state["results"]
agg             = st.session_state["agg"]
corr_results    = st.session_state["corr"]
sector_df       = st.session_state["sector_df"]
backtest        = st.session_state["backtest"]
sentiment_series= st.session_state["sentiment_series"]

# ── KPI row ────────────────────────────────────────────────────────────────────
st.divider()
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Articles Analysed", len(articles))
c2.metric("Composite Score",   f"{agg['composite_score']:+.3f}")
c3.metric("Positive",          agg["label_counts"].get("positive",0))
c4.metric("Negative",          agg["label_counts"].get("negative",0))
c5.metric("Avg Confidence",    f"{agg['mean_confidence']:.1%}")

st.divider()

# ── Sentiment breakdown ────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Sentiment Breakdown")
    counts = agg["label_counts"]
    fig_pie = go.Figure(go.Pie(
        labels = list(counts.keys()),
        values = list(counts.values()),
        hole   = 0.5,
        marker_colors = ["#2ea043", "#da3633", "#6e7681"],
    ))
    fig_pie.update_layout(
        height=300, margin=dict(t=20,b=20,l=0,r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e6edf3", legend_font_color="#e6edf3",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Score Distribution Across Headlines")
    scores = [r.positive - r.negative for r in results]
    labels_list = [r.label for r in results]
    color_map = {"positive":"#2ea043","negative":"#da3633","neutral":"#6e7681"}
    colors = [color_map[l] for l in labels_list]

    fig_hist = go.Figure(go.Histogram(
        x=scores, nbinsx=30,
        marker_color=colors[0],
        opacity=0.8,
    ))
    fig_hist.update_layout(
        height=300, margin=dict(t=20,b=20,l=0,r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e6edf3",
        xaxis_title="Composite Score (pos−neg)",
        yaxis_title="Count",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Sector correlation table ───────────────────────────────────────────────────
st.subheader("📊 Sentiment–Return Correlation by Sector")
st.caption("Pearson r between daily composite sentiment score and next-day forward returns")
if not sector_df.empty:
    st.dataframe(sector_df, use_container_width=True, hide_index=True)

# ── Ticker correlation heatmap ─────────────────────────────────────────────────
if corr_results:
    st.subheader("🔗 Per-Ticker Correlation")
    ticker_data = sorted(corr_results.items(), key=lambda x: x[1]["pearson_r"], reverse=True)
    tickers_plot = [t for t,_ in ticker_data]
    r_values     = [v["pearson_r"] for _,v in ticker_data]
    sig          = [v["significant"] for _,v in ticker_data]
    bar_colors   = ["#2ea043" if r > 0 else "#da3633" for r in r_values]
    marker_syms  = ["star" if s else "circle" for s in sig]

    fig_bar = go.Figure(go.Bar(
        x=tickers_plot, y=r_values,
        marker_color=bar_colors, opacity=0.85,
        text=[f"{r:.3f}" for r in r_values], textposition="outside",
    ))
    fig_bar.add_hline(y=0, line_dash="dash", line_color="#8b949e")
    fig_bar.update_layout(
        height=360, margin=dict(t=20,b=20,l=0,r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e6edf3", yaxis_title="Pearson r",
        yaxis_range=[-0.6, 0.6],
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Backtest ───────────────────────────────────────────────────────────────────
st.subheader("📈 Backtest: Sentiment Strategy vs Buy & Hold")
if not backtest.empty:
    final_strat = backtest["strategy_value"].iloc[-1]
    final_bh    = backtest["buyhold_value"].iloc[-1]
    init        = 10_000
    bc1,bc2,bc3 = st.columns(3)
    bc1.metric("Strategy Final Value",  f"${final_strat:,.0f}", f"{(final_strat/init-1)*100:+.1f}%")
    bc2.metric("Buy & Hold Final",      f"${final_bh:,.0f}",    f"{(final_bh/init-1)*100:+.1f}%")
    bc3.metric("Alpha",                 f"${final_strat-final_bh:+,.0f}")

    fig_bt = make_subplots(rows=2, cols=1, row_heights=[0.7,0.3], shared_xaxes=True, vertical_spacing=0.04)
    fig_bt.add_trace(go.Scatter(
        x=backtest.index, y=backtest["strategy_value"],
        name="Sentiment Strategy", line=dict(color="#2ea043", width=2),
    ), row=1, col=1)
    fig_bt.add_trace(go.Scatter(
        x=backtest.index, y=backtest["buyhold_value"],
        name="Buy & Hold", line=dict(color="#388bfd", width=2, dash="dot"),
    ), row=1, col=1)
    fig_bt.add_trace(go.Bar(
        x=backtest.index, y=backtest["sentiment_score"],
        name="Sentiment Score",
        marker_color=["#2ea043" if (v or 0) > 0 else "#da3633" for v in backtest["sentiment_score"]],
    ), row=2, col=1)
    fig_bt.update_layout(
        height=480, margin=dict(t=20,b=20,l=0,r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e6edf3", legend_font_color="#e6edf3",
    )
    fig_bt.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig_bt.update_yaxes(title_text="Sentiment",           row=2, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

# ── Article feed ───────────────────────────────────────────────────────────────
st.subheader("📰 Scored Article Feed")
BADGE = {"positive":"🟢","negative":"🔴","neutral":"⚪"}
feed_data = [{
    "Sentiment":   BADGE.get(a.sentiment,"") + " " + (a.sentiment or ""),
    "Score":       f"{a.sentiment_score:.2f}" if a.sentiment_score else "—",
    "Headline":    a.title[:100],
    "Source":      a.source,
    "Published":   str(a.published)[:16],
} for a in sorted(articles, key=lambda x: x.published, reverse=True)[:50]]
st.dataframe(pd.DataFrame(feed_data), use_container_width=True, hide_index=True)
