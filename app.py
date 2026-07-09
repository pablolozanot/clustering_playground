import pickle

import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from src.retrieval import build_faiss_index, investigate_case

REPO_ID = "pablolozanot/clustering_demo"

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


def main():
    st.set_page_config(page_title="CFPB Complaint Pattern Finder", layout="wide")

    st.title("CFPB Complaint Pattern Finder")
    st.markdown(
        "Paste a consumer complaint narrative and the tool searches **207,000 real CFPB complaints** "
        "for near-identical text. It tells you whether the complaint looks like a known filing template "
        "reused by many people, or a unique personal account."
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
            height=220,
            placeholder="Paste a CFPB consumer complaint narrative here...",
        )

        if st.button("Investigate", type="primary", disabled=not (text or "").strip()):
            with st.spinner("Searching..."):
                result = investigate_case(
                    text, model, index, df,
                    k=k, threshold=threshold, min_matches=min_matches,
                )

            n_q = result["n_qualifying_matches"]
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

            matches = result["matches"].copy()
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
