import pickle
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
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
    plot_df["Company short"] = plot_df["Company"].str[:30]
    plot_df["snippet"] = plot_df["Consumer complaint narrative"].str[:120] + "..."

    fig = px.scatter(
        plot_df,
        x="Date received",
        y="similarity",
        color="Company short",
        hover_data={"Company short": False, "Company": True, "snippet": True, "similarity": ":.3f"},
        labels={"Date received": "Date filed", "similarity": "Similarity score", "Company short": "Company"},
        title="Where in time do the matching complaints land?",
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="white")))
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#ef4444",
        annotation_text=f"threshold ({threshold})", annotation_position="bottom right",
    )

    if query_date is not None:
        qd = pd.Timestamp(query_date)
        fig.add_vrect(
            x0=qd - pd.Timedelta(days=TIME_WINDOW_DAYS),
            x1=qd + pd.Timedelta(days=TIME_WINDOW_DAYS),
            fillcolor="#fbbf24", opacity=0.12, line_width=0,
        )
        fig.add_vline(x=qd, line_dash="dot", line_color="#f59e0b",
                      annotation_text="query date", annotation_position="top")

    fig.update_layout(height=380, legend_title_text="Company", margin=dict(t=50, b=20))
    return fig


def main():
    st.set_page_config(page_title="CFPB Complaint Pattern Finder", layout="wide")

    st.title("CFPB Complaint Pattern Finder")
    st.markdown(
        "Paste a consumer complaint narrative and the tool searches **207,000 real CFPB complaints** "
        "for near-identical text. It tells you whether the complaint looks like a known filing template "
        "reused by many people, or a unique personal account — and shows *when* those matches were filed."
    )

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

            # Temporal metrics (only when a query date is provided)
            if query_date is not None:
                qd = pd.Timestamp(query_date)
                in_window = (
                    (matches["Date received"] >= qd - pd.Timedelta(days=TIME_WINDOW_DAYS)) &
                    (matches["Date received"] <= qd + pd.Timedelta(days=TIME_WINDOW_DAYS))
                )
                n_window = int(in_window.sum())
                n_window_qualifying = int(
                    ((matches["similarity"] >= threshold) & in_window).sum()
                )

            # Verdict banner
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

            # Temporal summary metrics
            if query_date is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Matches retrieved", len(matches))
                m2.metric(f"Within ±{TIME_WINDOW_DAYS}d of query date", n_window)
                m3.metric(f"Qualifying & within window", n_window_qualifying,
                          delta="temporal cluster" if n_window_qualifying >= min_matches else "no temporal cluster",
                          delta_color="inverse" if n_window_qualifying < min_matches else "normal")

            # Chart
            st.plotly_chart(build_chart(matches, threshold, query_date), use_container_width=True)

            # Detail table
            matches["Date received"] = matches["Date received"].dt.date
            matches["similarity"] = matches["similarity"].round(3)
            matches["snippet"] = matches["Consumer complaint narrative"].str[:150] + "..."
            show_cols = [c for c in ["similarity", "Date received", "Company", "Product", "snippet"]
                         if c in matches.columns]
            with st.expander("Show all matches", expanded=False):
                st.dataframe(matches[show_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        f"Corpus: {len(df):,} CFPB complaints with narratives · 2019–2022 · "
        "Embeddings: all-MiniLM-L6-v2 (384-dim) · "
        "Index: FAISS IndexFlatIP with L2 normalisation (exact cosine search)"
    )


if __name__ == "__main__":
    main()
