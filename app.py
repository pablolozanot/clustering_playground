import pickle
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from src.retrieval import build_faiss_index, investigate_case

REPO_ID = "pablolozanot/clustering_demo"
TIME_WINDOW_DAYS = 60

EXAMPLE_TEMPLATE = (
    "I am writing to dispute inaccurate information in my credit file. "
    "I recently obtained a copy of my credit report and noticed an account that does not belong to me. "
    "I am a victim of identity theft and have filed a police report and an FTC identity theft affidavit. "
    "Under the Fair Credit Reporting Act I have the right to dispute this information and request its removal. "
    "Please investigate and delete this account immediately as it is causing significant harm to my credit score."
)

EXAMPLE_PERSONAL = (
    "I have been a customer for over 12 years and last month I noticed two unauthorized charges on my account "
    "totaling $847. I called customer service three times and was told someone would call me back. Nobody did. "
    "I visited the branch in person and the manager said they could not help me at the branch level. "
    "I need these charges reversed and a written explanation of what happened to my account."
)

HOW_IT_WORKS = """
**The dataset**

The [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
is a public record of complaints submitted to the Consumer Financial Protection Bureau against financial
companies (banks, credit bureaus, debt collectors, etc.). The raw file contains ~15 million complaints
filed between 2011 and 2026. This demo uses a 2019–2022 slice of the ~207,000 complaints that include
a written narrative — most complaints only have structured fields (company, product, issue) with no
free text.

---

**Under the hood**

1. **Embeddings.** Every complaint narrative is converted into a 384-dimensional vector using
   [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a small but
   capable sentence transformer. Semantically similar sentences end up close together in this space
   regardless of exact wording.

2. **Exact cosine search.** All 207k vectors are stored in a FAISS `IndexFlatIP` index (inner product
   on L2-normalised vectors = cosine similarity). When you submit a complaint, it is embedded on the
   fly and the index returns the *k* nearest neighbours in milliseconds — no approximation, exact
   brute-force search.

3. **Similarity threshold.** A pair of complaints with cosine similarity ≥ 0.85 are considered
   near-duplicates: nearly word-for-word the same, likely derived from the same template. This
   threshold was validated against the full similarity distribution of the corpus.

4. **Verdict.** If 3 or more complaints clear the threshold, the tool flags a **FINDING** — meaning
   the text is not original, it is a known pattern in the database.

---

**The temporal layer**

When you add a filing date, the chart overlays a ±60-day window. Points inside that band that are
*also* above the similarity threshold (highlighted in red) represent complaints filed by other people
using the same template *around the same time* — the signature of a coordinated or viral filing
campaign. Points outside the window (grey) are the same template used at other times, which is normal
background reuse and not a signal of coordination.

The 60-day window and the similarity threshold come from research on this dataset: joint
text-and-time clustering (requiring *both* conditions simultaneously) surfaces candidates that a
purely text-based or purely time-based search would miss.

---

**What the patterns mean**

Through clustering the full corpus, three distinct complaint patterns emerged:

- **Template filers** — legal-aid scripts reused verbatim across thousands of people over years
  (FCRA identity-theft disputes, FDCPA debt-validation letters). High similarity, spread over time.
- **Coordinated or viral campaigns** — the same template concentrated in a short window, often
  targeting multiple companies simultaneously.
- **Single-consumer multi-recipient filings** — one person, one situation (e.g. a debt chased by
  ten different collectors), filing near-identical complaints against every company involved within
  a day or two. Legitimate behaviour, but a distinct and findable pattern.
"""


@st.cache_resource(show_spinner="Loading 207k complaints and building search index (first load ~1 min)...")
def load_everything():
    parquet_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="complaints_500k_narrative.parquet",
    )
    emb_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="complaints_500k_narrative_embeddings.pkl",
    )

    df = pd.read_parquet(parquet_path)
    df["Date received"] = pd.to_datetime(df["Date received"])

    with open(emb_path, "rb") as f:
        embeddings = pickle.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, _ = build_faiss_index(embeddings)

    return df, model, index


def build_chart(matches, threshold, query_date=None):
    plot_df = matches.copy()
    plot_df["snippet"] = plot_df["Consumer complaint narrative"].str[:150] + "..."

    # Determine cluster membership
    above_threshold = plot_df["similarity"] >= threshold
    if query_date is not None:
        qd = pd.Timestamp(query_date)
        in_window = (
            (plot_df["Date received"] >= qd - pd.Timedelta(days=TIME_WINDOW_DAYS)) &
            (plot_df["Date received"] <= qd + pd.Timedelta(days=TIME_WINDOW_DAYS))
        )
        in_cluster = above_threshold & in_window
    else:
        in_cluster = above_threshold

    cluster = plot_df[in_cluster]
    other = plot_df[~in_cluster]

    fig = go.Figure()

    # Gray points — context, not in cluster
    if len(other):
        fig.add_trace(go.Scatter(
            x=other["Date received"],
            y=other["similarity"],
            mode="markers",
            marker=dict(color="#cbd5e1", size=9, line=dict(width=0.5, color="#94a3b8")),
            name="other matches",
            customdata=np.stack([other["Company"].values, other["snippet"].values], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{x|%Y-%m-%d} · sim %{y:.3f}<br>%{customdata[1]}<extra></extra>",
        ))

    # Colored points — the cluster
    if len(cluster):
        fig.add_trace(go.Scatter(
            x=cluster["Date received"],
            y=cluster["similarity"],
            mode="markers",
            marker=dict(color="#ef4444", size=11, line=dict(width=1, color="white")),
            name="cluster match",
            customdata=np.stack([cluster["Company"].values, cluster["snippet"].values], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{x|%Y-%m-%d} · sim %{y:.3f}<br>%{customdata[1]}<extra></extra>",
        ))

    # Threshold line
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#ef4444", line_width=1,
        annotation_text=f"similarity threshold ({threshold})",
        annotation_position="bottom right",
        annotation_font_size=11,
    )

    # Date window overlay
    if query_date is not None:
        qd = pd.Timestamp(query_date)
        fig.add_vrect(
            x0=qd - pd.Timedelta(days=TIME_WINDOW_DAYS),
            x1=qd + pd.Timedelta(days=TIME_WINDOW_DAYS),
            fillcolor="#fbbf24", opacity=0.10, line_width=0,
        )
        fig.add_vline(
            x=qd, line_dash="dot", line_color="#f59e0b", line_width=1.5,
            annotation_text="query date", annotation_position="top",
        )

    fig.update_layout(
        title="Matching complaints over time — red dots are the cluster",
        xaxis_title="Date filed",
        yaxis_title="Similarity score",
        height=400,
        margin=dict(t=50, b=20, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#f1f5f9"),
        xaxis=dict(gridcolor="#f1f5f9"),
    )
    return fig


def main():
    st.set_page_config(page_title="CFPB Complaint Pattern Finder", layout="wide")

    st.title("CFPB Complaint Pattern Finder")
    st.markdown(
        "Paste a consumer complaint narrative and the tool searches **207,000 real CFPB complaints** "
        "for near-identical text — and shows *when* those matches were filed."
    )

    with st.expander("How this works — dataset, methodology, and what the patterns mean"):
        st.markdown(HOW_IT_WORKS)

    df, model, index = load_everything()

    left, right = st.columns([3, 1])

    with right:
        st.markdown("**Try an example**")
        if st.button("Template filer (FCRA dispute)", use_container_width=True):
            st.session_state.input_text = EXAMPLE_TEMPLATE
        if st.button("Personal narrative", use_container_width=True):
            st.session_state.input_text = EXAMPLE_PERSONAL

        st.divider()
        st.markdown("**Date of this complaint** *(optional)*")
        use_date = st.checkbox("Add a filing date", value=False)
        query_date = st.date_input(
            "Filing date", value=date(2021, 6, 1),
            min_value=date(2019, 1, 1), max_value=date(2022, 12, 31),
            disabled=not use_date, label_visibility="collapsed",
        ) if use_date else None

        st.divider()
        st.markdown("**Settings**")
        threshold = st.slider("Similarity threshold", 0.70, 0.99, 0.85, 0.01,
                              help="Minimum cosine similarity to count as a near-duplicate")
        min_matches = st.slider("Min matches for a finding", 1, 10, 3,
                                help="How many near-duplicates before flagging as a pattern")
        k = st.slider("Neighbours to retrieve", 5, 50, 20)

    with left:
        text = st.text_area(
            "Complaint narrative",
            key="input_text",
            height=200,
            placeholder="Paste a CFPB consumer complaint narrative here...",
        )

        if st.button("Investigate", type="primary", disabled=not (text or "").strip()):
            with st.spinner("Searching..."):
                result = investigate_case(
                    text, model, index, df,
                    k=k, threshold=threshold, min_matches=min_matches,
                )

            matches = result["matches"].copy()
            n_q = result["n_qualifying_matches"]

            if query_date is not None:
                qd = pd.Timestamp(query_date)
                in_window = (
                    (matches["Date received"] >= qd - pd.Timedelta(days=TIME_WINDOW_DAYS)) &
                    (matches["Date received"] <= qd + pd.Timedelta(days=TIME_WINDOW_DAYS))
                )
                n_window = int(in_window.sum())
                n_window_qualifying = int(((matches["similarity"] >= threshold) & in_window).sum())

            # Verdict
            if result["verdict"] == "finding":
                st.success(
                    f"FINDING — {n_q} near-identical complaints (≥ {threshold} similarity, "
                    f"needed {min_matches}+)"
                )
            else:
                st.info(
                    f"NO STRONG PATTERN — {n_q} qualifying match{'es' if n_q != 1 else ''} "
                    f"found (needed {min_matches}+ for a finding)"
                )

            # Temporal metrics
            if query_date is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Matches retrieved", len(matches))
                m2.metric(f"Within ±{TIME_WINDOW_DAYS}d of query date", n_window)
                m3.metric(
                    "Qualifying & within window", n_window_qualifying,
                    delta="temporal cluster" if n_window_qualifying >= min_matches else "no temporal cluster",
                    delta_color="normal" if n_window_qualifying >= min_matches else "inverse",
                )

            st.plotly_chart(build_chart(matches, threshold, query_date), use_container_width=True)

            with st.expander("Show all matches"):
                matches["Date received"] = matches["Date received"].dt.date
                matches["similarity"] = matches["similarity"].round(3)
                matches["snippet"] = matches["Consumer complaint narrative"].str[:150] + "..."
                show_cols = [c for c in ["similarity", "Date received", "Company", "Product", "snippet"]
                             if c in matches.columns]
                st.dataframe(matches[show_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        f"Corpus: {len(df):,} CFPB complaints with narratives · 2019–2022 · "
        "Embeddings: all-MiniLM-L6-v2 (384-dim) · "
        "Index: FAISS IndexFlatIP with L2 normalisation (exact cosine search)"
    )


if __name__ == "__main__":
    main()
