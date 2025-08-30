# InDex — Automated News-Driven Scoring for Index Constituents
# -------------------------------------------------------------
# Streamlit app that lets a user upload index constituent CSV files (e.g., Nasdaq)
# and programmatically searches the web (Google Custom Search) for < 2 days news.
# It then runs a lightweight ML/NLP pipeline to score each constituent based on
# news type & impact and returns a CSV with headers: Ticker, Name, InDex Score.
#
# --- Secrets Setup ---
# Create .streamlit/secrets.toml with:
# [general]
# GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
# GOOGLE_SEARCH_ENGINE_ID = "YOUR_CSE_ID"
#
# --- Run ---
# streamlit run app.py

from __future__ import annotations

import time
import re
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Google CSE
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ML/NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sentiment (VADER) with safe fallback
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    _nltk_ok = True
except Exception:
    _nltk_ok = False

# ----------------------------- Streamlit UI -----------------------------

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

st.set_page_config(page_title="InDex — News-Driven Index Scoring", layout="wide")
st.title("📈 InDex — Automated Index Scoring From Fresh News (< 2 days)")
st.markdown(
    "Upload a CSV of constituents (**Ticker, Name**). I’ll search the last **2 days** of news, "
    "classify & score impact, blend with sentiment, and output an **InDex Score** per company."
)

# --- Secrets for Google CSE ---
API_KEY = st.secrets.get("GOOGLE_API_KEY")
CX = st.secrets.get("GOOGLE_SEARCH_ENGINE_ID")
if not API_KEY or not CX:
    st.warning("Google API credentials missing in secrets. Add GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID.")

# ----------------------------- Sidebar -----------------------------

st.sidebar.header("⚙️ Scoring Settings")
num_results_per_ticker = st.sidebar.slider("Results per Ticker (per CSE call)", 1, 10, 5, 1)

include_press_releases = st.sidebar.checkbox("Include Press Releases (PRNewswire/BusinessWire)", True)

allowed_domains_text = st.sidebar.text_area(
    "Limit to Domains (optional, one per line)",
    value="",
    help="e.g. reuters.com\nwsj.com\nbloomberg.com\nprnewswire.com",
)
blocked_domains_text = st.sidebar.text_area(
    "Exclude Domains (optional, one per line)",
    value=""
)

st.sidebar.subheader("Weights & Impact")
press_release_weight = st.sidebar.slider("Press Release Weight", 0.2, 2.0, 0.8, 0.1)
major_outlet_weight = st.sidebar.slider("Major Outlet Weight", 0.5, 3.0, 1.5, 0.1)
other_outlet_weight = st.sidebar.slider("Other Outlet Weight", 0.1, 2.0, 1.0, 0.1)
st.sidebar.caption("Weights multiply the per-article score. Major outlets detected by domain keywords (e.g., reuters, bloomberg).")

st.sidebar.subheader("Sentiment Blend")
sentiment_blend = st.sidebar.slider("Blend Sentiment into Score (0=no sentiment, 1=full)", 0.0, 1.0, 0.6, 0.05)

st.sidebar.subheader("Impact Multipliers")
st.sidebar.caption("Keywords detected in title/snippet will multiply the score.")
impact_multipliers = {
    "earnings": st.sidebar.slider("Earnings", 0.5, 3.0, 1.3, 0.1),
    "guidance": st.sidebar.slider("Guidance", 0.5, 3.0, 1.6, 0.1),
    "downgrade": st.sidebar.slider("Downgrade", 0.5, 3.0, 2.0, 0.1),
    "upgrade": st.sidebar.slider("Upgrade", 0.5, 3.0, 1.8, 0.1),
    "acquisition": st.sidebar.slider("Acquisition/Merger", 0.5, 3.0, 2.4, 0.1),
    "merger": 2.4,
    "lawsuit": st.sidebar.slider("Lawsuit/Investigation", 0.5, 3.0, 2.2, 0.1),
    "investigation": 2.2,
    "sec": st.sidebar.slider("SEC/Regulatory", 0.5, 3.0, 2.0, 0.1),
    "fda": st.sidebar.slider("FDA/Approval", 0.5, 3.0, 2.3, 0.1),
    "layoffs": st.sidebar.slider("Layoffs/Restructuring", 0.5, 3.0, 1.7, 0.1),
    "buyback": st.sidebar.slider("Buyback/Dividend", 0.5, 3.0, 1.4, 0.1),
    "dividend": 1.4,
    "contract": st.sidebar.slider("New Contract/Deal", 0.5, 3.0, 1.6, 0.1),
    "partnership": st.sidebar.slider("Partnership", 0.5, 3.0, 1.5, 0.1),
    "product": st.sidebar.slider("Product/Launch", 0.5, 3.0, 1.4, 0.1),
}

st.sidebar.subheader("Quality Controls")
use_content_scrape = st.sidebar.checkbox("Fetch & analyze article content (vs. title/snippet only)", True)
dedupe_similar = st.sidebar.checkbox("De-duplicate similar results", True)
sleep_between_calls = st.sidebar.slider("Rate limit (seconds between tickers)", 0.0, 2.0, 0.2, 0.1)

# ----------------------------- Upload -----------------------------

st.subheader("1) Upload Index Constituents CSV")
st.caption("CSV must include at least columns: 'Ticker' and 'Name' (case-insensitive). Extra columns are ignored.")
constituents_file = st.file_uploader("Upload CSV", type=["csv"])

# ----------------------------- Helpers -----------------------------

MAJOR_OUTLETS = [
    "reuters", "bloomberg", "wsj", "financialtimes", "ft.com", "nytimes", "forbes",
    "cnbc", "marketwatch", "theverge", "washingtonpost", "seekingalpha", "yahoo",
    "barrons", "techcrunch", "apnews", "theguardian", "semianalysis"
]

PRESS_RELEASE_HOSTS = ["prnewswire", "businesswire", "globenewswire", "newsfilecorp", "accesswire"]

STOP_WORDS = set(
    [
        'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves',
        'he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs',
        'themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being',
        'have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while',
        'of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to',
        'from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why',
        'how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than',
        'too','very','s','t','can','will','just','don','should','now'
    ]
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

@st.cache_data(show_spinner=False)
def read_constituents(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    cols = {c.lower(): c for c in df.columns}
    ticker_col = cols.get("ticker") or cols.get("symbol") or cols.get("ric")
    name_col = cols.get("name") or cols.get("company") or cols.get("company name")
    if not ticker_col or not name_col:
        raise ValueError("CSV needs columns for Ticker and Name.")
    out = df[[ticker_col, name_col]].copy()
    out.columns = ["Ticker", "Name"]
    out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
    out["Name"] = out["Name"].astype(str).str.strip()
    return out.dropna()

def preprocess_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    return " ".join(tokens)

@st.cache_resource(show_spinner=False)
def get_sia():
    if not _nltk_ok:
        return None
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        try:
            nltk.download('vader_lexicon')
        except Exception:
            return None
    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        return None

SIA = get_sia()

def build_query(company: str, ticker: str) -> str:
    # Prioritize company brand + ticker. Exclude finance quote pages implicitly by using 'news' keyword.
    q = f"{company} ({ticker}) news OR press release"
    return q

def domain_weight(url: str) -> float:
    host = re.sub(r"^https?://", "", url).split('/')[0].lower()
    if any(h in host for h in PRESS_RELEASE_HOSTS):
        return press_release_weight if include_press_releases else 0.0
    if any(h in host for h in MAJOR_OUTLETS):
        return major_outlet_weight
    return other_outlet_weight

def impact_multiplier_from_text(title: str, snippet: str) -> float:
    text = f"{title} {snippet}".lower()
    mult = 1.0
    for k, v in impact_multipliers.items():
        if k in text:
            mult *= float(v)
    return mult

def get_sentiment_score(text: str) -> float:
    if SIA is None:
        # Simple fallback lexicon
        pos_words = {"beat","beats","record","surge","rally","profit","growth","upgrade","outperform","buy","strong","win","wins","approved","approval","acquire","merger"}
        neg_words = {"miss","misses","plunge","drop","loss","downgrade","underperform","sell","weak","lawsuit","probe","investigation","delay","recall","reject","rejects"}
        t = preprocess_text(text)
        pos = sum(w in t for w in pos_words)
        neg = sum(w in t for w in neg_words)
        if pos == neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)
    try:
        return float(SIA.polarity_scores(text).get("compound", 0.0))
    except Exception:
        return 0.0

# ----------------------------- Search & Fetch -----------------------------

@st.cache_resource(show_spinner=False)
def get_cse_service():
    # Construct once per session
    return build("customsearch", "v1", developerKey=API_KEY)

def _domain_ok(u: str, allowed: List[str] | None, blocked: List[str] | None) -> bool:
    host = re.sub(r"^https?://", "", u).split('/')[0].lower()
    if allowed:
        if not any(d.strip().lower() in host for d in allowed if d.strip()):
            return False
    if blocked:
        if any(d.strip().lower() in host for d in blocked if d.strip()):
            return False
    return True

@st.cache_data(show_spinner=False)
def google_search_recent(query: str, num: int = 5, allowed: List[str] | None = None, blocked: List[str] | None = None) -> List[Dict[str, Any]]:
    # Uses CSE 'dateRestrict=d2' to get last 2 days of results.
    try:
        service = get_cse_service()
        params = {
            "q": query,
            "cx": CX,
            "num": num,
            "dateRestrict": "d2",
            # Keep safe default fields; CSE may return 'items' with title/snippet/link
        }
        res = service.cse().list(**params).execute()
        items = res.get("items", [])
        items = [i for i in items if _domain_ok(i.get("link", ""), allowed, blocked)]
        return items
    except HttpError as e:
        # Often quota or temporary errors
        st.warning(f"Google CSE error for query '{query[:60]}...': {e}")
        return []
    except Exception:
        return []

def fetch_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": USER_AGENT})
        if r.status_code != 200 or not r.text:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove non-content
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        texts = [el.get_text(" ", strip=True) for el in soup.find_all(["h1", "h2", "h3", "p", "li", "span"])]
        text = " ".join(t for t in texts if t)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""

# ----------------------------- Scoring -----------------------------

def score_article(url: str, title: str, snippet: str, content: str) -> float:
    if not content:
        content = f"{title} {snippet}"

    # Base sentiment [-1..1]
    sent = get_sentiment_score(content)
    # Normalize to [0..1] for scoring blend
    sent_norm = (sent + 1) / 2.0  # 0 to 1

    # Relevance: cosine similarity (title+snippet) vs (content)
    vectorizer = TfidfVectorizer(max_features=5000)
    try:
        X = vectorizer.fit_transform([
            preprocess_text(f"{title} {snippet}"),
            preprocess_text(content)
        ])
        rel = float(cosine_similarity(X[0:1], X[1:2])[0][0])
    except Exception:
        rel = 0.5

    # Impact
    impact = impact_multiplier_from_text(title, snippet)

    # Source weight
    src_w = domain_weight(url)

    # Final article score (bounded)
    base = (1 - sentiment_blend) * rel + sentiment_blend * sent_norm
    score = base * impact * src_w
    return float(np.clip(score, 0.0, 10.0))

def aggregate_scores(article_scores: List[float]) -> float:
    if not article_scores:
        return 0.0
    # Robust combine: average of top 3
    topk = sorted(article_scores, reverse=True)[:3]
    agg = float(np.mean(topk))
    return round(agg, 4)

# ----------------------------- Utilities -----------------------------

def normalize_title(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "").strip().lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return t

def titles_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    # Fast fuzzy by SequenceMatcher (no extra dependency)
    from difflib import SequenceMatcher
    a_n = normalize_title(a)
    b_n = normalize_title(b)
    if not a_n or not b_n:
        return False
    sim = SequenceMatcher(None, a_n, b_n).ratio()
    return sim >= threshold

def dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return items
    kept: List[Dict[str, Any]] = []
    for it in items:
        t = it.get("title", "")
        link = it.get("link", "")
        if not t or not link:
            continue
        is_dup = False
        for k in kept:
            if titles_similar(t, k.get("title", "")) or link == k.get("link", ""):
                is_dup = True
                break
        if not is_dup:
            kept.append(it)
    return kept

# ----------------------------- Main Flow -----------------------------

if constituents_file is not None and API_KEY and CX:
    try:
        df_const = read_constituents(constituents_file)
        st.success(f"Loaded {len(df_const)} constituents.")

        allowed_domains = [d.strip() for d in allowed_domains_text.splitlines() if d.strip()] if allowed_domains_text else None
        blocked_domains = [d.strip() for d in blocked_domains_text.splitlines() if d.strip()] if blocked_domains_text else None

        if st.button("🚀 Run InDex Scoring (last 2 days)"):
            progress = st.progress(0.0)
            per_ticker_details: Dict[str, List[Dict[str, Any]]] = {}
            scores: List[Dict[str, Any]] = []

            total = len(df_const)

            for idx, row in df_const.iterrows():
                ticker = row["Ticker"]
                name = row["Name"]
                query = build_query(name, ticker)

                try:
                    results = google_search_recent(
                        query,
                        num=num_results_per_ticker,
                        allowed=allowed_domains,
                        blocked=blocked_domains
                    )
                except Exception:
                    results = []

                if dedupe_similar:
                    results = dedupe_items(results)

                article_details: List[Dict[str, Any]] = []
                article_scores: List[float] = []

                for item in results:
                    url = item.get("link", "")
                    title = item.get("title", "") or ""
                    snippet = item.get("snippet", "") or ""

                    if not url:
                        continue

                    content = ""
                    if use_content_scrape:
                        content = fetch_text(url)

                    s = score_article(url, title, snippet, content)
                    if s <= 0.0:
                        continue

                    article_scores.append(s)
                    article_details.append({
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                        "score": round(float(s), 4)
                    })

                final_score = aggregate_scores(article_scores)
                scores.append({"Ticker": ticker, "Name": name, "InDex Score": final_score})
                per_ticker_details[ticker] = article_details

                # Update progress
                progress.progress(min((idx + 1) / total, 1.0))

                # Gentle rate limiting to avoid CSE errors
                if sleep_between_calls > 0:
                    time.sleep(sleep_between_calls)

            result_df = pd.DataFrame(scores)

            st.subheader("📊 InDex Scores")
            st.dataframe(result_df, use_container_width=True)

            # Download CSV (exact header order)
            csv_bytes = result_df[["Ticker", "Name", "InDex Score"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download InDex Scores (CSV)",
                data=csv_bytes,
                file_name="InDex_scores.csv",
                mime="text/csv"
            )

            # Explainability
            st.markdown("---")
            st.subheader("🔎 Per-Ticker Article Breakdown")
            for _, r in result_df.sort_values("InDex Score", ascending=False).iterrows():
                t = r["Ticker"]
                n = r["Name"]
                with st.expander(f"{t} — {n} (Score: {r['InDex Score']})"):
                    arts = per_ticker_details.get(t, [])
                    if not arts:
                        st.info("No qualifying recent articles found (or all filtered).")
                    else:
                        for a in arts:
                            st.markdown(
                                f"**{a['title']}**\n\n"
                                f"{a['snippet']}\n\n"
                                f"[Open]({a['url']}) | Score: **{a['score']}**"
                            )

    except Exception as e:
        st.error(f"Failed to process constituents CSV: {e}")

else:
    st.info("Upload constituents CSV and ensure Google API secrets are set to enable scoring.")
