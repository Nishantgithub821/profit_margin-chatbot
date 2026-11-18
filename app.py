import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="ProductIQ | AI Product Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM CSS - CONSISTENT DARK THEME + READABLE TEXT
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        color-scheme: dark;
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Main application background and default text */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background:
            radial-gradient(circle at 8% 0%, rgba(99,102,241,.16), transparent 28%),
            radial-gradient(circle at 92% 8%, rgba(20,184,166,.12), transparent 25%),
            #0b1020 !important;
        color: #edf2ff !important;
    }

    /* Force readable text throughout the main page */
    [data-testid="stMain"] p,
    [data-testid="stMain"] li,
    [data-testid="stMain"] label,
    [data-testid="stMain"] span,
    [data-testid="stMain"] .stMarkdown,
    [data-testid="stMain"] [data-testid="stCaptionContainer"] {
        color: #dbe5f5;
    }

    [data-testid="stHeader"] {
        background: rgba(11,16,32,.85) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid rgba(148,163,184,.18);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #9fb0ca !important;
    }

    /* Hero section */
    .hero {
        padding: 30px 32px;
        border: 1px solid rgba(148,163,184,.22);
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(30,41,59,.98), rgba(22,35,62,.94));
        box-shadow: 0 18px 55px rgba(0,0,0,.24);
        margin-bottom: 22px;
    }
    .eyebrow {
        color: #5eead4 !important;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(30px, 4vw, 46px);
        line-height: 1.08;
        margin: 9px 0 10px 0;
        color: #ffffff !important;
    }
    .hero p {
        color: #c0cde0 !important;
        font-size: 15px;
        margin: 0;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(30,41,59,.88);
        border: 1px solid rgba(148,163,184,.20);
        padding: 18px 19px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,.12);
    }
    .metric-label {
        color: #a9bad3 !important;
        font-size: 12px;
        letter-spacing: .5px;
    }
    .metric-value {
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 26px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 16px;
        background: rgba(20,29,49,.88) !important;
        padding: 12px 15px;
        color: #edf2ff !important;
    }
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] span {
        color: #e5edf9 !important;
    }

    /* Bottom bar (jahan chat input rehta hai) - white strip hatao */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background: #0b1020 !important;
    }

    /* Chat input - poora wrapper dark */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"] {
        background-color: #111c31 !important;
    }
    [data-testid="stChatInput"] {
        border: 1px solid #334563 !important;
        border-radius: 15px !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        caret-color: #5eead4 !important;
        background: transparent !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #91a3bd !important;
        -webkit-text-fill-color: #91a3bd !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"] button {
        color: #ffffff !important;
        background: #4f46e5 !important;
        border-radius: 10px !important;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 11px;
        border: 1px solid rgba(129,140,248,.42);
        background: rgba(99,102,241,.16);
        color: #f1f5ff !important;
        transition: .2s ease;
    }
    div.stButton > button:hover {
        border-color: #8b9aff;
        color: #ffffff !important;
        background: rgba(99,102,241,.30);
    }

    /* Expanders, dataframes and other native widgets */
    [data-testid="stExpander"] {
        background: rgba(20,29,49,.75) !important;
        border: 1px solid rgba(148,163,184,.18) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span {
        color: #e5edf9 !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,.2);
        border-radius: 12px;
        overflow: hidden;
        background: #111c31 !important;
    }
    [data-testid="stDataFrame"] > div {
        background: #111c31 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #a9bad3 !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    .small-note {
        color: #91a3bd !important;
        font-size: 12px;
    }

    /* Dividers */
    hr {
        border-color: rgba(148,163,184,.22) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LOAD PROJECT FILES
# Keep these three .pkl files in the same folder as app.py
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_project():
    data_path = BASE_DIR / "product_data.pkl"
    vectorizer_path = BASE_DIR / "tfidf_vectorizer.pkl"
    matrix_path = BASE_DIR / "tfidf_matrix.pkl"

    missing = [
        path.name for path in [data_path, vectorizer_path, matrix_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing project file(s): " + ", ".join(missing))

    data = pd.read_pickle(data_path)
    vectorizer = joblib.load(vectorizer_path)
    matrix = joblib.load(matrix_path)
    return data, vectorizer, matrix


try:
    product_df, product_vectorizer, product_matrix = load_project()
except Exception as exc:
    st.error(f"Project files load nahi ho paayi: {exc}")
    st.info(
        "app.py ke same folder mein product_data.pkl, "
        "tfidf_vectorizer.pkl aur tfidf_matrix.pkl rakho."
    )
    st.stop()


DISPLAY_COLUMNS = [
    "Brand", "Product Name", "Product Type", "Gender",
    "Production Year", "Release Price_num",
    "Profit Made_num", "Profit Margin_num",
]
DISPLAY_COLUMNS = [c for c in DISPLAY_COLUMNS if c in product_df.columns]

products = product_df.drop_duplicates(subset=["Brand", "Product Name"]).copy()
products["Product Name"] = products["Product Name"].fillna("").astype(str)
products["Brand"] = products["Brand"].fillna("").astype(str)
products["Product Type"] = products["Product Type"].fillna("").astype(str)


# ---------------------------------------------------------
# SEARCH + QUESTION UNDERSTANDING
# ---------------------------------------------------------
def clean_query(query: str) -> str:
    query = query.lower().strip()
    remove_phrases = [
        "what is", "tell me", "show me", "give me",
        "please", "can you", "mujhe", "batao", "btao",
        "kitna hai", "kitni hai", "kya hai", "details",
        "complete details", "all details", "information",
        "profit margin", "margin", "profit made", "profit",
        "release price", "price", "product type", "category",
        "production year", "year", "gender",
        "ka", "ki", "ke", "for", "of", "the",
    ]
    for phrase in sorted(remove_phrases, key=len, reverse=True):
        query = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", query)
    query = re.sub(r"[^\w\s'\"-]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query


def requested_field(query: str):
    q = query.lower()
    if "profit margin" in q or "margin" in q:
        return "Profit Margin_num"
    if "profit" in q:
        return "Profit Made_num"
    if "release price" in q or "price" in q:
        return "Release Price_num"
    if "product type" in q or "category" in q:
        return "Product Type"
    if "production year" in q or re.search(r"\byear\b", q):
        return "Production Year"
    if "gender" in q:
        return "Gender"
    if "brand" in q:
        return "Brand"
    return None


def search_products(query: str, top_n: int = 5):
    """Exact/partial name first; fuzzy TF-IDF only when needed."""
    q = query.strip().lower()
    if not q:
        return products.iloc[0:0].copy(), "empty"

    names = products["Product Name"].str.lower().str.strip()

    exact = products[names == q]
    if not exact.empty:
        return exact.head(3), "exact"

    partial = products[names.str.contains(re.escape(q), na=False)]
    if not partial.empty:
        return partial.head(top_n), "partial"

    brand_matches = products[products["Brand"].str.lower().str.strip() == q]
    if not brand_matches.empty:
        return brand_matches.head(0), "brand_only"

    query_vector = product_vectorizer.transform([query.lower()])
    scores = cosine_similarity(query_vector, product_matrix).ravel()

    if len(scores) != len(product_df):
        return products.iloc[0:0].copy(), "not_found"

    raw_scores = pd.Series(scores, index=product_df.index)
    sorted_scores = raw_scores.sort_values(ascending=False)
    ranked = product_df.loc[sorted_scores.index].copy()
    ranked["_score"] = sorted_scores.values
    ranked = ranked[ranked["_score"] >= 0.25]
    ranked = ranked.drop_duplicates(subset=["Brand", "Product Name"])

    if ranked.empty:
        return products.iloc[0:0].copy(), "not_found"
    return ranked.head(top_n), "similar"


def format_value(field, value):
    if pd.isna(value):
        return "Not available"
    if field in ["Release Price_num", "Profit Made_num"]:
        return f"${float(value):,.2f}"
    if field == "Profit Margin_num":
        return f"{float(value):g}%"
    return str(value)


def product_summary(row):
    return (
        f"**{row.get('Brand', 'Unknown')} {row.get('Product Name', '')}**\n\n"
        f"- Product type: {row.get('Product Type', 'Not available')}\n"
        f"- Gender: {row.get('Gender', 'Not available')}\n"
        f"- Production year: {row.get('Production Year', 'Not available')}\n"
        f"- Release price: {format_value('Release Price_num', row.get('Release Price_num'))}\n"
        f"- Profit made: {format_value('Profit Made_num', row.get('Profit Made_num'))}\n"
        f"- Profit margin: {format_value('Profit Margin_num', row.get('Profit Margin_num'))}"
    )


def answer_query(query):
    field = requested_field(query)
    product_query = clean_query(query)

    if not product_query:
        product_query = query.strip()

    results, match_type = search_products(product_query)

    if match_type == "brand_only":
        return (
            "I found that brand, but it has multiple products. "
            "Please enter the exact product name so I can return the correct details."
        ), None, results

    if results.empty:
        return (
            "I couldn't find a confident match. Try the full or a shorter product name."
        ), None, results

    if match_type == "similar" and len(results) > 1:
        return (
            "I found a few similar products. Choose the exact product from the results below."
        ), None, results

    row = results.iloc[0]
    if field:
        field_label = field.replace("_num", "").replace("_", " ").lower()
        answer = (
            f"**{row['Brand']} {row['Product Name']}** — "
            f"{field_label}: **{format_value(field, row.get(field))}**"
        )
    else:
        answer = product_summary(row)

    return answer, row, results


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ✦ ProductIQ")
    st.caption("AI-powered product intelligence")
    st.divider()

    st.markdown("### Dataset overview")
    c1, c2 = st.columns(2)
    c1.metric("Products", f"{len(products):,}")
    c2.metric("Brands", f"{products['Brand'].nunique():,}")

    if "Product Type" in products.columns:
        st.metric("Product categories", f"{products['Product Type'].nunique():,}")

    st.divider()
    st.markdown("### Try asking")
    examples = [
        "Air Max 95 Big Bubble",
        "Kobe 6 Protro profit margin",
        "Nike Club Fleece Hat price",
        "Show complete details of Air Jordan 4",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.pending_query = example

    st.divider()
    st.markdown(
        '<div class="small-note">Answers are based only on the fields available '
        'in the project dataset.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# MAIN HEADER + METRICS
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">PRODUCT INTELLIGENCE · NLP ASSISTANT</div>
      <h1>Ask your products.<br>Get clear answers.</h1>
      <p>Search product information, explore pricing and understand profit metrics — all in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(3)
metric_data = [
    ("PRODUCT RECORDS", f"{len(products):,}"),
    ("UNIQUE BRANDS", f"{products['Brand'].nunique():,}"),
    ("PRODUCT TYPES", f"{products['Product Type'].nunique():,}"),
]
for col, (label, value) in zip(metric_cols, metric_data):
    col.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")


# ---------------------------------------------------------
# CHAT INTERFACE
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm ProductIQ. Enter an exact product name and I'll show "
                "its available details. You can also ask for its price, profit or margin."
            ),
        }
    ]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("table") is not None:
            st.dataframe(message["table"], use_container_width=True, hide_index=True)

pending = st.session_state.pending_query
user_query = st.chat_input("Ask about a product, price, profit or margin...")

if pending:
    user_query = pending
    st.session_state.pending_query = None

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    answer, selected_row, results = answer_query(user_query)

    with st.chat_message("assistant"):
        st.markdown(answer)

        if not results.empty and selected_row is None:
            view = results[DISPLAY_COLUMNS].copy()
            st.dataframe(view, use_container_width=True, hide_index=True)
        elif selected_row is not None:
            with st.expander("View product record", expanded=False):
                st.dataframe(
                    pd.DataFrame([selected_row[DISPLAY_COLUMNS]]),
                    use_container_width=True,
                    hide_index=True,
                )

    table_data = None
    if not results.empty and selected_row is None:
        table_data = results[DISPLAY_COLUMNS].copy()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "table": table_data}
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div class="small-note" style="text-align:center;">'
    'ProductIQ · Built with Python, Streamlit & TF-IDF · Dataset-powered insights'
    '</div>',
    unsafe_allow_html=True,
)