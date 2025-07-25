"""
dashboard.py
------------
Streamlit dashboard for the FinBERT Sentiment Predictor.
Pulls data from PostgreSQL directly (not via API) for simplicity.
"""

from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.scraper           import fetch_news
from src.sentiment         import classify_articles
from src.price_correlation import (
    download_prices, build_sentiment_series, correlate, sector_sentiment, SECTOR_MAP
)

st.set_page_config(
    page_title="FinSentiment Dashboard",
    page_icon="📰",
    layout="wide",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0e1a; }
  [data-testid="stSidebar"]          { background: #111827; }
  h1, h2, h3 { color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📰 FinSentiment")
    st.caption("FinBERT-powered news sentiment + price correlation")
    st.divider()
    newsapi_key = st.text_input("NewsAPI Key (optional)", type="password",
                                 placeholder="get free key at newsapi.org")
    selected_tickers = st.multiselect(
        "Tickers to analyse",
        ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "TSLA", "JPM", "BAC", "JNJ", "XOM"],
        default=["AAPL", "MSFT", "TSLA"]
    )
    days_back = st.slider("Days of news to fetch", 7, 60, 30)
    run_btn   = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    st.divider()
    st.caption("Model: ProsusAI/FinBERT\nSources: Yahoo Finance RSS, Finviz, NewsAPI")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📰 Financial News Sentiment & Price Correlation")
st.markdown("Scrape real-time financial news → classify with **FinBERT** → correlate with stock price movements.")

if not run_btn:
    st.info("Select tickers in the sidebar and click **Run Analysis** to begin.")
    st.stop()

if not selected_tickers:
    st.warning("Please select at least one ticker.")
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────
with st.spinner("Fetching news articles …"):
    articles = fetch_news(selected_tickers, newsapi_key=newsapi_key, days_back=days_back)

if not articles:
    st.error("No articles fetched. Check your internet connection or try adding a NewsAPI key.")
    st.stop()

with st.spinner(f"Classifying {len(articles)} headlines with FinBERT …"):
    results = classify_articles(articles)

with st.spinner("Downloading price data …"):
    prices          = download_prices(selected_tickers, days_back=days_back + 20)
    sentiment_daily = build_sentiment_series(results, articles)

# ── KPI row ───────────────────────────────────────────────────────────────────
pos = sum(1 for r in results if r.label == "positive")
neg = sum(1 for r in results if r.label == "negative")
neu = sum(1 for r in results if r.label == "neutral")
mean_compound = sum(r.compound for r in results) / len(results) if results else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Articles", f"{len(articles):,}")
c2.metric("Positive 🟢", f"{pos:,}")
c3.metric("Negative 🔴", f"{neg:,}")
c4.metric("Neutral  ⚪", f"{neu:,}")
c5.metric("Mean Compound", f"{mean_compound:+.3f}")

st.divider()

# ── Sentiment pie chart ───────────────────────────────────────────────────────
col_a, col_b = st.columns([1, 2])
with col_a:
    st.subheader("Sentiment Mix")
    fig_pie = go.Figure(go.Pie(
        labels=["Positive", "Negative", "Neutral"],
        values=[pos, neg, neu],
        marker_colors=["#22c55e", "#ef4444", "#94a3b8"],
        hole=0.45,
    ))
    fig_pie.update_layout(
        height=300, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9", legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    st.subheader("Daily Sentiment Score per Ticker")
    if not sentiment_daily.empty:
        fig_line = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, ticker in enumerate(sentiment_daily.columns):
            fig_line.add_trace(go.Scatter(
                x=sentiment_daily.index, y=sentiment_daily[ticker],
                name=ticker, mode="lines+markers",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        fig_line.add_hline(y=0, line_dash="dash", line_color="#64748b", opacity=0.6)
        fig_line.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9", yaxis_title="Compound Score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig_line.update_yaxes(range=[-1, 1], gridcolor="#1e293b")
        fig_line.update_xaxes(gridcolor="#1e293b")
        st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# ── Correlation results ───────────────────────────────────────────────────────
st.subheader("📊 Sentiment × 5-Day Forward Return Correlation")
if not sentiment_daily.empty and not prices.empty:
    _, corr_report = correlate(sentiment_daily, prices, horizon=5, rolling_window=14)
    if corr_report:
        corr_df = pd.DataFrame([
            {"Ticker": t, "Pearson r": v["pearson_r"], "p-value": v["p_value"],
             "Observations": v["n_obs"], "Significant": "✓" if v["significant"] else "✗"}
            for t, v in sorted(corr_report.items(), key=lambda x: abs(x[1]["pearson_r"]), reverse=True)
        ])
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

        # Correlation bar chart
        fig_bar = go.Figure(go.Bar(
            x=corr_df["Ticker"], y=corr_df["Pearson r"],
            marker_color=["#22c55e" if r >= 0 else "#ef4444" for r in corr_df["Pearson r"]],
            text=corr_df["Pearson r"].round(3), textposition="outside",
        ))
        fig_bar.add_hline(y=0, line_color="#64748b")
        fig_bar.update_layout(
            height=300, yaxis_title="Pearson r", xaxis_title="",
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9", yaxis_range=[-1, 1],
        )
        fig_bar.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Not enough overlapping data to compute correlations yet. Run with more days.")

st.divider()

# ── Recent headlines table ────────────────────────────────────────────────────
st.subheader("📰 Latest Classified Headlines")
headlines_df = pd.DataFrame([{
    "Ticker":   r.ticker,
    "Label":    r.label.capitalize(),
    "Score":    round(r.score, 3),
    "Compound": round(r.compound, 3),
    "Headline": r.title[:100] + ("…" if len(r.title) > 100 else ""),
} for r in results]).sort_values("Compound", ascending=False)

st.dataframe(
    headlines_df.head(50),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Compound": st.column_config.ProgressColumn(
            "Compound", min_value=-1, max_value=1, format="%.3f"
        )
    }
)
