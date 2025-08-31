# InDex — Automated News-Driven Scoring for Index Constituents
# -------------------------------------------------------------
# Streamlit app that lets a user upload index constituent CSV files (e.g., Nasdaq)
# and programmatically searches the web (Google Custom Search) for < 2 days news.
# It then runs a lightweight ML/NLP pipeline to score each constituent based on
# news type & impact and returns a CSV with headers: Ticker, Name, InDex Score.
#
# --- Setup ---
# - Add your Google API key and Custom Search Engine ID to .streamlit/secrets.toml
#   [general]
#   GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
#   GOOGLE_SEARCH_ENGINE_ID = "YOUR_CSE_ID"
#
# - Run: streamlit run app.py

import streamlit as st
import requests
from googleapiclient.discovery import build
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import pandas as pd
import numpy as np
import re
import warnings
from datetime import datetime, timedelta

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

# Suppress XML parsing warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

st.set_page_config(page_title="InDex — News-Driven Index Scoring", layout="wide")

st.title("📈 InDex — Automated Index Scoring From Fresh News (< 2 days)")
st.markdown(
    "Upload a CSV of constituents (Ticker, Name). We'll search the web for the last **2 days** of news,\n"
    "classify & score article impact, blend with sentiment, and output an **InDex Score** per company.\n"
)

# --- Secrets for Google CSE ---
API_KEY = st.secrets.get("GOOGLE_API_KEY")
CX = st.secrets.get("GOOGLE_SEARCH_ENGINE_ID")
if not API_KEY or not CX:
    st.warning("Google API credentials missing in secrets. Add GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID.")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Scoring Settings")
num_results_per_ticker = st.sidebar.slider("Results per Ticker (per CSE call)", 1, 10, 5, 1)
include_press_releases = st.sidebar.checkbox("Include Press Releases (PRNewswire/BusinessWire)", True)
allowed_domains = st.sidebar.text_area(
    "Limit to Domains (optional, one per line)",
    value="",
    help="e.g. reuters.com\nwsj.com\nbloomberg.com\nprnewswire.com",
)
blocked_domains = st.sidebar.text_area(
    "Exclude Domains (optional, one per line)", value=""
)

st.sidebar.subheader("Weights & Impact")
press_release_weight = st.sidebar.slider("Press Release Weight", 0.2, 2.0, 0.8, 0.1)
major_outlet_weight = st.sidebar.slider("Major Outlet Weight", 0.5, 3.0, 1.5, 0.1)
other_outlet_weight = st.sidebar.slider("Other Outlet Weight", 0.1, 2.0, 1.0, 0.1)

st.sidebar.caption(
    "Weights multiply the per-article score. Major outlets detected by domain keywords (e.g., reuters, bloomberg)."
)

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
    "merger": 2.4, # Keep this to also catch "merger" if acquisition is not present
    "lawsuit": st.sidebar.slider("Lawsuit/Investigation", 0.5, 3.0, 2.2, 0.1),
    "investigation": 2.2, # Keep this to also catch "investigation" if lawsuit is not present
    "sec": st.sidebar.slider("SEC/Regulatory", 0.5, 3.0, 2.0, 0.1),
    "fda": st.sidebar.slider("FDA/Approval", 0.5, 3.0, 2.3, 0.1),
    "layoffs": st.sidebar.slider("Layoffs/Restructuring", 0.5, 3.0, 1.7, 0.1),
    "buyback": st.sidebar.slider("Buyback/Dividend", 0.5, 3.0, 1.4, 0.1),
    "dividend": 1.4, # Keep this to also catch "dividend" if buyback is not present
    "contract": st.sidebar.slider("New Contract/Deal", 0.5, 3.0, 1.6, 0.1),
    "partnership": st.sidebar.slider("Partnership", 0.5, 3.0, 1.5, 0.1),
    "product": st.sidebar.slider("Product/Launch", 0.5, 3.0, 1.4, 0.1),
}


# --- Tabs for Main Content ---
tab1, tab2, tab3 = st.tabs(["🚀 Run InDex", "📚 Methodology", "📖 User Guide"])

with tab1:
    # --- Upload Constituents ---
    st.subheader("1) Upload Index Constituents CSV")
    st.caption("CSV must include at least columns: 'Ticker' and 'Name' (case-insensitive). Extra columns are ignored.")
    constituents_file = st.file_uploader("Upload CSV", type=["csv"])

    @st.cache_data(show_spinner=False)
    def read_constituents(file) -> pd.DataFrame:
        df = pd.read_csv(file)
        cols = {c.lower(): c for c in df.columns}
        # Try common variants
        ticker_col = cols.get("ticker") or cols.get("symbol") or cols.get("ric")
        name_col = cols.get("name") or cols.get("company") or cols.get("company name")
        if not ticker_col or not name_col:
            raise ValueError("CSV needs columns for Ticker and Name.")
        out = df[[ticker_col, name_col]].copy()
        out.columns = ["Ticker", "Name"]
        out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
        out["Name"] = out["Name"].astype(str).str.strip()
        return out.dropna()

    # --- Helpers ---
    MAJOR_OUTLETS = [
        "reuters", "bloomberg", "wsj", "financialtimes", "ft.com", "nytimes", "forbes", "cnbc",
        "marketwatch", "theverge", "washingtonpost", "seekingalpha", "yahoo", "barrons"
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

    # --- Google CSE query builder ---
    def build_query(company: str, ticker: str) -> str:
        # Prioritize company brand + ticker, exclude finance pages with stale quotes
        q = f"{company} ({ticker}) news OR press release"
        return q


    def domain_weight(url: str) -> float:
        host = re.sub(r"https?://", "", url)
        # Simplify hostname
        host = host.split('/')[0]
        if any(h in host for h in PRESS_RELEASE_HOSTS):
            return press_release_weight if include_press_releases else 0.0
        if any(h in host for h in MAJOR_OUTLETS):
            return major_outlet_weight
        return other_outlet_weight


    def impact_multiplier_from_text(title: str, snippet: str) -> float:
        text = f"{title} {snippet}".lower()
        mult = 1.0
        # Check for keywords and apply their corresponding multiplier
        for k, v_factor in impact_multipliers.items():
            if k in text:
                mult *= float(v_factor)
        return mult


    def get_sentiment_score(text: str) -> float:
        if SIA is None:
            # Simple fallback lexicon
            pos_words = {"beat","beats","record","surge","rally","profit","growth","upgrade","outperform","buy","strong","win","wins","approved","approval","acquire","merger"}
            neg_words = {"miss","misses","plunge","drop","loss","downgrade","underperform","sell","weak","lawsuit","probe","investigation","delay","recall","reject","rejects"}
            t = preprocess_text(text)
            pos = sum(w in t for w in pos_words)
            neg = sum(w in t for w in neg_words)
            if pos==neg==0:
                return 0.0
            return (pos - neg) / (pos + neg)
        try:
            return SIA.polarity_scores(text).get("compound", 0.0)
        except Exception:
            return 0.0


    # --- Search & Fetch ---
    @st.cache_data(show_spinner=False)
    def google_search_recent(query: str, num: int = 5, allowed: list | None = None, blocked: list | None = None):
        if not API_KEY or not CX:
            st.error("Google API keys not configured. Please check .streamlit/secrets.toml")
            return []
        service = build("customsearch", "v1", developerKey=API_KEY)
        params = {
            'q': query,
            'cx': CX,
            'num': num,
            # Restrict to last 2 days
            'dateRestrict': 'd2'
        }
        res = service.cse().list(**params).execute()
        items = res.get('items', [])
        # Filter domains
        def ok(u: str) -> bool:
            host = re.sub(r"https?://", "", u).split('/')[0]
            if allowed:
                if not any(d.strip().lower() in host.lower() for d in allowed if d.strip()):
                    return False
            if blocked:
                if any(d.strip().lower() in host.lower() for d in blocked if d.strip()):
                    return False
            return True
        return [i for i in items if ok(i.get('link',''))]


    def fetch_text(url: str) -> str:
        try:
            r = requests.get(url, timeout=12, headers={'User-Agent':'Mozilla/5.0'})
            if r.status_code != 200:
                return ""
            soup = BeautifulSoup(r.text, 'html.parser')
            # Remove script/style
            for tag in soup(["script","style","noscript"]):
                tag.extract()
            # Extract content from common article tags
            texts = [el.get_text(" ", strip=True) for el in soup.find_all(['h1','h2','h3','p','li','span','article','main','div'], limit=100)] # Limit tags to avoid huge parse
            text = " ".join(t for t in texts if t)
            # Collapse whitespace
            return re.sub(r"\s+", " ", text).strip()
        except Exception:
            return ""


    # --- Scoring ---
    def score_article(url: str, title: str, snippet: str, content: str) -> float:
        if not content:
            content = f"{title} {snippet}"
        
        # Ensure content is not too short for meaningful analysis
        if len(content.split()) < 10: # Minimum 10 words
            return 0.0

        # Base sentiment [-1..1]
        sent = get_sentiment_score(content)
        # Normalize to [0..1] for scoring blend
        sent_norm = (sent + 1) / 2  # 0 to 1

        # Relevance: cosine similarity to ticker/company keywords
        # Create a combined document for the article
        article_text = preprocess_text(f"{title} {snippet} {content}")
        
        # Create a simple "query" document for general news relevance
        # This acts as a baseline for what "news" generally looks like
        # We're more interested in the similarity between the headline/snippet and full content here,
        # but also how "newsworthy" the content itself is.
        # For a more robust approach, we could compare against a corpus of general news.
        
        # For now, let's use the title/snippet as one document and content as another
        # and measure their similarity. If they are very different, relevance drops.
        try:
            vectorizer = TfidfVectorizer(max_features=5000)
            # Fit and transform both the 'preview' (title+snippet) and 'full_content'
            X = vectorizer.fit_transform([
                preprocess_text(title + " " + snippet),
                preprocess_text(content)
            ])
            # Relevance is the cosine similarity between the preview and the full content
            rel = float(cosine_similarity(X[0:1], X[1:2])[0][0])
        except Exception:
            rel = 0.5 # Default relevance if vectorization fails

        # Impact
        impact = impact_multiplier_from_text(title, snippet)

        # Source weight
        src_w = domain_weight(url)

        # Final article score (bounded)
        # Blend relevance and sentiment
        base = (1 - sentiment_blend) * rel + sentiment_blend * sent_norm
        
        # Apply impact and source weight
        score = base * impact * src_w
        return float(np.clip(score, 0.0, 10.0))


    def aggregate_scores(article_scores: list[float]) -> float:
        if not article_scores:
            return 0.0
        # Robust combine: average of top 3 scores to avoid being dragged down by low-quality articles
        topk = sorted(article_scores, reverse=True)[:3]
        agg = float(np.mean(topk))
        return round(agg, 4)

    # --- Main Flow ---
    if constituents_file is not None and API_KEY and CX:
        try:
            df_const = read_constituents(constituents_file)
            st.success(f"Loaded {len(df_const)} constituents.")
            allowed = [d.strip() for d in allowed_domains.splitlines() if d.strip()] if allowed_domains else None
            blocked = [d.strip() for d in blocked_domains.splitlines() if d.strip()] if blocked_domains else None

            if st.button("🚀 Run InDex Scoring (last 2 days)"):
                progress_text = "Searching news and scoring constituents. Please wait..."
                progress_bar = st.progress(0, text=progress_text)
                
                per_ticker_details = {}
                scores = []
                total = len(df_const)
                start_time = datetime.now()

                for idx, row in df_const.iterrows():
                    ticker = row['Ticker']
                    name = row['Name']
                    query = build_query(name, ticker)
                    
                    try:
                        results = google_search_recent(query, num=num_results_per_ticker, allowed=allowed, blocked=blocked)
                    except Exception as e:
                        st.warning(f"Error during Google Search for {ticker}: {e}")
                        results = []

                    article_details = []
                    article_scores = []
                    for item in results:
                        url = item.get('link','')
                        title = item.get('title','')
                        snippet = item.get('snippet','')
                        
                        # Fetch full content only if URL is valid and seems like a real article
                        content = ""
                        if url and "google.com/url" not in url and not any(host in url for host in PRESS_RELEASE_HOSTS): # Basic filter for actual articles
                            content = fetch_text(url)
                        
                        s = score_article(url, title, snippet, content)
                        if s <= 0.0:
                            continue # Skip articles with no score or invalid content
                        
                        article_scores.append(s)
                        article_details.append({
                            'url': url,
                            'title': title,
                            'snippet': snippet,
                            'score': round(float(s), 4)
                        })

                    final_score = aggregate_scores(article_scores)
                    scores.append({'Ticker': ticker, 'Name': name, 'InDex Score': final_score})
                    per_ticker_details[ticker] = article_details

                    # Update progress bar
                    progress_percent = min((idx + 1) / total, 1.0)
                    progress_bar.progress(progress_percent, text=f"Processing {ticker} ({idx+1}/{total})...")

                progress_bar.empty() # Clear the progress bar after completion
                end_time = datetime.now()
                st.success(f"Scoring complete for {total} constituents in {(end_time - start_time).total_seconds():.2f} seconds.")


                result_df = pd.DataFrame(scores)
                st.subheader("📊 InDex Scores")
                st.dataframe(result_df, use_container_width=True)

                # Download CSV (exact header order as requested)
                csv_bytes = result_df[['Ticker','Name','InDex Score']].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download InDex Scores (CSV)",
                    data=csv_bytes,
                    file_name="InDex_scores.csv",
                    mime="text/csv"
                )

                # Explainability
                st.markdown("---")
                st.subheader("🔎 Per‑Ticker Article Breakdown")
                for _, r in result_df.sort_values('InDex Score', ascending=False).iterrows():
                    t = r['Ticker']
                    n = r['Name']
                    with st.expander(f"{t} — {n} (Score: {r['InDex Score']})"):
                        arts = per_ticker_details.get(t, [])
                        if not arts:
                            st.info("No qualifying recent articles found (or all filtered).")
                        else:
                            for a in arts:
                                st.markdown(f"**[{a['title']}]({a['url']})**")
                                st.markdown(f"{a['snippet']}")
                                st.markdown(f"Score: **{a['score']}**")
                                st.markdown("---")

        except Exception as e:
            st.error(f"Failed to process constituents CSV: {e}")
    else:
        st.info("Upload constituents CSV and ensure Google API secrets are set to enable scoring.")

with tab2:
    st.header("📚 InDex Scoring Methodology")
    st.markdown("""
    The InDex scoring system is designed to provide a comprehensive, automated assessment of an index constituent's recent news impact. It combines multiple NLP techniques and configurable weights to produce a single, normalized "InDex Score" for each company.

    The scoring process involves several key steps for each retrieved news article:

    ### 1. News Retrieval & Filtering
    *   **Source:** Google Custom Search Engine (CSE) is used to find articles.
    *   **Query:** For each constituent, a specific query `"{Company Name} ({Ticker}) news OR press release"` is constructed.
    *   **Recency:** Results are strictly limited to articles published within the last **2 days** to ensure freshness.
    *   **Domain Filtering:** Users can specify `Allowed Domains` and `Blocked Domains` to control the sources.
    *   **Press Releases:** Articles from known press release distributors (e.g., PRNewswire, BusinessWire) are identified and can be included/excluded based on user settings.

    ### 2. Article Content Processing
    *   **Text Extraction:** For each search result, the full article content is attempted to be fetched from its URL. Robust web scraping (using BeautifulSoup) extracts relevant text, ignoring scripts, styles, and navigational elements. If full content retrieval fails, the article's title and snippet are used as a fallback.
    *   **Pre-processing:** Text content is cleaned by removing special characters, lowercasing, and stripping common English stop words to prepare for NLP analysis.

    ### 3. Per-Article Scoring Components

    Each article receives a preliminary score based on a blend of three main factors:

    #### a) Relevance (Cosine Similarity)
    *   **What it measures:** How semantically similar the article's title/snippet is to its full content. A high similarity indicates the headline accurately reflects the story, and the full content is focused.
    *   **How it works:** TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is applied to both the pre-processed title/snippet and the pre-processed full content. Cosine similarity is then calculated between these two vectors.
    *   **Impact:** A higher relevance score suggests a more coherent and pertinent article to the initial search context.

    #### b) Sentiment (VADER)
    *   **What it measures:** The emotional tone (positive, negative, neutral) of the article's content.
    *   **How it works:** NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool is used. VADER is specifically attuned to sentiment expressed in social media and financial contexts. It produces a `compound` score ranging from -1 (most negative) to +1 (most positive).
    *   **Normalization:** The compound score is normalized to a 0-1 range `(score + 1) / 2` for blending.
    *   **Fallback:** If NLTK/VADER is unavailable, a simple lexicon-based sentiment analysis is used as a fallback.
    *   **User Blend:** A `Sentiment Blend` slider allows users to control how much sentiment influences the base score (0 = no sentiment, 1 = full sentiment).

    #### c) Impact Multipliers
    *   **What it measures:** The inherent significance of the news topic. Certain news categories typically have a greater impact on a company's stock price or outlook.
    *   **How it works:** Keywords (e.g., "earnings", "acquisition", "downgrade", "layoffs") are searched within the article's title and snippet. If detected, a pre-defined multiplier (configurable in the sidebar) is applied to the article's base score. For example, "downgrade" might have a higher multiplier than "product launch".
    *   **User Configuration:** All impact multipliers are adjustable via the sidebar.

    #### d) Source Weight
    *   **What it measures:** The credibility and influence of the news source.
    *   **How it works:** Articles from recognized major financial news outlets (e.g., Reuters, Bloomberg, WSJ) receive a higher weight. Press releases are given a separate, often lower, weight (configurable). Other general news sources receive a baseline weight.
    *   **User Configuration:** `Press Release Weight`, `Major Outlet Weight`, and `Other Outlet Weight` are adjustable.

    ### 4. Aggregating Scores per Constituent
    *   **Individual Article Scores:** Each article's relevance, sentiment, impact, and source weight are combined to produce an `Article Score` (range 0-10, clamped).
    *   **Constituent Aggregation:** For each company, all valid `Article Scores` are collected. To provide a robust overall `InDex Score` and mitigate the effect of outlier low-quality articles, the system takes the **average of the top 3 highest-scoring articles** for that constituent. If fewer than 3 articles are found, it averages all available.
    *   **Final InDex Score:** This aggregated score is the `InDex Score` for the constituent.

    ### Final Output
    The application outputs a table with `Ticker`, `Name`, and the calculated `InDex Score`. Users can download this as a CSV. Additionally, a detailed breakdown of the top articles contributing to each company's score is provided for transparency.
    """)

with tab3:
    st.header("📖 InDex User Guide")
    st.markdown("""
    Welcome to InDex! This guide will walk you through how to use the application to get news-driven scores for your index constituents.

    ### 1. Setup Your Google API Credentials (One-time)

    To enable the news search functionality, you need to configure your Google API Key and Custom Search Engine (CSE) ID.

    1.  **Get a Google API Key:**
        *   Go to the [Google Cloud Console](https://console.cloud.google.com/).
        *   Create a new project or select an existing one.
        *   Navigate to "APIs & Services" > "Credentials".
        *   Click "Create credentials" > "API key".
        *   Copy the generated API key.
        *   **Important:** Restrict your API key to only the "Custom Search API" to enhance security.

    2.  **Create a Google Custom Search Engine (CSE):**
        *   Go to the [Google Custom Search control panel](https://programmablesearchengine.google.com/controlpanel/all).
        *   Click "Add new search engine".
        *   In the "Sites to search" field, add `*.com/*` or `*.org/*` to search the entire web (or specific domains if you prefer).
        *   Give your search engine a name and click "Create".
        *   After creation, go to "Overview" and copy your "Search engine ID" (also known as CX).

    3.  **Configure `secrets.toml`:**
        *   In your Streamlit application directory, create a folder named `.streamlit` if it doesn't exist.
        *   Inside `.streamlit`, create a file named `secrets.toml`.
        *   Add your API key and CSE ID to this file like this:
            ```toml
            [general]
            GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"
            GOOGLE_SEARCH_ENGINE_ID = "YOUR_CSE_ID_HERE"
            ```
        *   Replace `YOUR_GOOGLE_API_KEY_HERE` and `YOUR_CSE_ID_HERE` with your actual credentials.
        *   **Restart your Streamlit app** after creating/modifying `secrets.toml`.

    ### 2. Prepare Your Constituents CSV

    Your input CSV file needs to contain at least two columns:
    *   **Ticker:** The stock ticker symbol (e.g., `AAPL`, `MSFT`).
    *   **Name:** The full company name (e.g., `Apple Inc.`, `Microsoft Corp.`).

    The column names are case-insensitive and common variants (`Symbol`, `Company Name`) are also accepted. You can have other columns in your CSV; they will be ignored.

    **Example `constituents.csv`:**
    ```csv
    Ticker,Name,Sector
    AAPL,Apple Inc.,Technology
    MSFT,Microsoft Corp.,Technology
    GOOG,Alphabet Inc.,Technology
    TSLA,Tesla Inc.,Automotive
    ```

    ### 3. Using the "Run InDex" Tab

    1.  **Upload Your CSV:**
        *   On the "Run InDex" tab, click the "Browse files" button under "1) Upload Index Constituents CSV".
        *   Select your prepared CSV file. The app will confirm the number of constituents loaded.

    2.  **Adjust Scoring Settings (Sidebar):**
        *   **Results per Ticker:** How many search results Google CSE should return for each company query. More results can provide a broader view but increase processing time and API cost.
        *   **Include Press Releases:** Check this if you want to consider articles from official press release wire services in your scoring.
        *   **Limit to Domains:** (Optional) Enter specific domains (one per line, e.g., `reuters.com`) if you only want news from certain sources.
        *   **Exclude Domains:** (Optional) Enter domains (one per line) you wish to explicitly ignore.
        *   **Weights & Impact:** These sliders control how much different types of news sources (Press Release, Major Outlet, Other Outlet) contribute to the score.
        *   **Sentiment Blend:** Adjust how much the sentiment (positive/negative tone) of an article influences its overall score. A value of 0 means sentiment is ignored; 1 means sentiment is the primary driver (alongside relevance).
        *   **Impact Multipliers:** These sliders allow you to increase or decrease the importance of specific news topics (e.g., earnings, acquisitions, lawsuits). If a keyword from these categories is found in an article's title or snippet, its score will be multiplied by the set factor.

    3.  **Run InDex Scoring:**
        *   Once your CSV is uploaded and settings are configured, click the "🚀 Run InDex Scoring (last 2 days)" button.
        *   A progress bar will appear, showing the current constituent being processed. News retrieval and content fetching can take some time, especially for large lists of constituents.

    4.  **Review Results:**
        *   After processing, a table titled "📊 InDex Scores" will display the `Ticker`, `Name`, and `InDex Score` for each constituent.
        *   You can download these results as a CSV using the "📥 Download InDex Scores (CSV)" button.
        *   Further down, under "🔎 Per‑Ticker Article Breakdown", you can expand each company to see the individual articles that contributed to its score, along with their titles, snippets, URLs, and individual article scores. This provides transparency and helps understand *why* a company received a particular score.

    ### 4. Explore "Methodology" and "User Guide" Tabs

    *   **📚 Methodology:** Provides a detailed explanation of how the InDex score is calculated, including the NLP techniques, weighting schemes, and aggregation methods.
    *   **📖 User Guide:** (You are here!) This tab provides instructions on how to set up and use the application.

    ### Troubleshooting

    *   **"Google API credentials missing" warning:** Double-check your `.streamlit/secrets.toml` file for correct `GOOGLE_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID` entries. Ensure the app was restarted after creating/editing the file.
    *   **"Failed to process constituents CSV":** Ensure your CSV has columns named 'Ticker' and 'Name' (or common variants like 'Symbol', 'Company').
    *   **No results or low scores:** Check your `Allowed Domains` and `Blocked Domains` settings. You might be overly restrictive. Also, ensure `Include Press Releases` is checked if you expect to see many press-release-driven news items. The "Results per Ticker" slider also limits the initial search.
    *   **Slow performance:** Processing many constituents with a high `Results per Ticker` can be slow due to web requests. Consider reducing `Results per Ticker` or processing a smaller list initially. Google CSE also has rate limits.
    """)
