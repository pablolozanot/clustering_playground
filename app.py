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
DEFAULT_THRESHOLD = 0.85
DEFAULT_K = 20
TEMPORAL_K = 500  # larger pool for temporal analysis

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

# Real cluster from the data: 58 near-identical complaints against 36 companies, all in a 16-day window.
# One consumer disputing fraudulent accounts on their credit report, filing against every creditor involved.
EXAMPLE_TEMPORAL_NARRATIVE = (
    "I've been disputing fraud accounts on my credit report since XX/XX/2020. "
    "I keep sending multiple sets of letters to the bureaus and the creditors so the excuse of "
    "\"we didn't get it\" doesn't happen. Furthermore, each letter is certified mail with tracking "
    "and each letter shows signed and delivered and yet the bureaus are still not taking any actions. "
    "The accounts are not showing in dispute nor are they removed from my report. "
    "The accounts always show up with different name variations and different account number variations "
    "and yet the bureaus seem to ignore the trend and let these fraud accounts affect my credit score."
)
EXAMPLE_TEMPORAL_DATE = date(2021, 7, 7)

HOW_IT_WORKS = """
**The dataset**

The [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
is a public record of complaints submitted to the Consumer Financial Protection Bureau against financial
companies (banks, credit bureaus, debt collectors, etc.). The raw file contains ~15 million complaints
filed between 2011 and 2026. This demo uses a 2019–2022 slice of the ~207,000 complaints that include
a written narrative — most complaints only have structured fields (company, product, issue) with no free text.

---

**Semantic Search — how it works**

Every complaint narrative is converted into a 384-dimensional vector using
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a sentence transformer
that places semantically similar texts close together regardless of exact wording. All 207k vectors are
stored in a FAISS index that returns the nearest neighbours in milliseconds via exact cosine search.

A complaint with cosine similarity ≥ 0.85 to another is considered a near-duplicate — nearly word-for-word,
likely derived from the same filing template. If 3 or more complaints clear that bar, the tool flags a
**FINDING**: the text is not original, it is a known pattern in the database.

When you add a filing date, the chart highlights complaints that are *both* semantically similar *and*
filed within ±60 days — the joint text-and-time signal that distinguishes coordinated campaigns from
ordinary background template reuse.

---

**Temporal Clustering — how it works**

This mode asks a different question: **is there an unusual concentration of complaints around a specific date?**

The corpus is divided into non-overlapping windows of the chosen size. For each window we count how many
complaints land in it, building a baseline distribution. We then count how many complaints fall in the
window around the query date and compute a Z-score against that baseline.

If a narrative is provided, the population is narrowed to semantically similar complaints before the
temporal test — so the question becomes "is *this type* of complaint unusually concentrated around this
date?" rather than "are complaints in general unusually concentrated?"

A Z-score above 2 means the window is in the top ~2.5% of all windows by complaint volume — a meaningful
spike. Above 3 is exceptional.

---

**The three complaint patterns this research found**

Through clustering the full corpus, three distinct filing patterns emerged:

- **Template filers** — legal-aid scripts (FCRA identity-theft disputes, FDCPA debt-validation letters)
  reused verbatim across thousands of people over years. High similarity, spread over time.
- **Coordinated or viral campaigns** — the same template concentrated in a short window, often targeting
  multiple companies simultaneously.
- **Single-consumer multi-recipient filings** — one person, one underlying situation (e.g. a debt being
  chased by ten different collectors), filing near-identical complaints against every company involved
  within a day or two. Legitimate behaviour, but a distinct and findable pattern.
"""


@st.cache_resource(show_spinner="Loading 207k complaints and building search index (first load ~1 min)...")
def load_everything():
    try:
        parquet_path = hf_hub_download(repo_id=REPO_ID, filename="complaints_500k_narrative.parquet")
        emb_path = hf_hub_download(repo_id=REPO_ID, filename="complaints_500k_narrative_embeddings.pkl")
    except Exception as e:
        st.error(f"Could not download data from HuggingFace Hub: {e}")
        st.stop()

    df = pd.read_parquet(parquet_path)
    df["Date received"] = pd.to_datetime(df["Date received"])

    with open(emb_path, "rb") as f:
        embeddings = pickle.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, _ = build_faiss_index(embeddings)

    return df, model, index


# ── helpers ──────────────────────────────────────────────────────────────────

def temporal_stats(dates, query_date, window_days):
    """Z-score of complaint count in the query window vs the baseline distribution."""
    qd = pd.Timestamp(query_date)
    half = pd.Timedelta(days=window_days)
    in_window = int(((dates >= qd - half) & (dates <= qd + half)).sum())

    step = pd.Timedelta(days=window_days * 2)
    t = dates.min()
    counts = []
    while t + step <= dates.max():
        counts.append(int(((dates >= t) & (dates < t + step)).sum()))
        t += step
    counts = np.array(counts, dtype=float)

    mean = counts.mean()
    std = counts.std()
    z = float((in_window - mean) / std) if std > 0 else 0.0
    pct = float((counts < in_window).mean()) * 100

    return {"in_window": in_window, "expected": round(mean, 1), "std": round(std, 1),
            "z_score": round(z, 2), "pct_rank": round(pct, 1), "n_windows": len(counts)}


def volume_chart(dates, query_date, window_days, title):
    weekly = dates.dt.to_period("W").value_counts().sort_index()
    weeks = pd.to_datetime([str(p.start_time) for p in weekly.index])
    counts = weekly.values

    qd = pd.Timestamp(query_date)
    half = pd.Timedelta(days=window_days)
    bar_colors = ["#ef4444" if (qd - half <= w <= qd + half) else "#cbd5e1" for w in weeks]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=weeks, y=counts, marker_color=bar_colors,
                         name="complaints / week", hovertemplate="%{x|%Y-%m-%d}: %{y}<extra></extra>"))
    fig.add_hline(y=counts.mean(), line_dash="dash", line_color="#64748b",
                  annotation_text=f"avg {counts.mean():.0f}/week", annotation_position="top left")
    fig.add_vline(x=qd, line_dash="dot", line_color="#f59e0b", line_width=2,
                  annotation_text="query date", annotation_position="top")
    fig.update_layout(title=title, xaxis_title="Week", yaxis_title="Complaints",
                      height=380, margin=dict(t=50, b=20), plot_bgcolor="white",
                      yaxis=dict(gridcolor="#f1f5f9"), showlegend=False)
    return fig


def semantic_chart(matches, threshold, query_date=None, window_days=60):
    plot_df = matches.copy()
    plot_df["snippet"] = plot_df["Consumer complaint narrative"].str[:150] + "..."

    above = plot_df["similarity"] >= threshold
    if query_date is not None:
        qd = pd.Timestamp(query_date)
        in_win = (
            (plot_df["Date received"] >= qd - pd.Timedelta(days=window_days)) &
            (plot_df["Date received"] <= qd + pd.Timedelta(days=window_days))
        )
        in_cluster = above & in_win
    else:
        in_cluster = above

    other = plot_df[~in_cluster]
    cluster = plot_df[in_cluster]
    fig = go.Figure()

    if len(other):
        fig.add_trace(go.Scatter(
            x=other["Date received"], y=other["similarity"], mode="markers",
            marker=dict(color="#cbd5e1", size=9, line=dict(width=0.5, color="#94a3b8")),
            name="other matches",
            customdata=np.stack([other["Company"].values, other["snippet"].values], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{x|%Y-%m-%d} · sim %{y:.3f}<br>%{customdata[1]}<extra></extra>",
        ))
    if len(cluster):
        fig.add_trace(go.Scatter(
            x=cluster["Date received"], y=cluster["similarity"], mode="markers",
            marker=dict(color="#ef4444", size=11, line=dict(width=1, color="white")),
            name="cluster match",
            customdata=np.stack([cluster["Company"].values, cluster["snippet"].values], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{x|%Y-%m-%d} · sim %{y:.3f}<br>%{customdata[1]}<extra></extra>",
        ))

    fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444", line_width=1,
                  annotation_text=f"threshold ({threshold})", annotation_position="bottom right")

    if query_date is not None:
        qd = pd.Timestamp(query_date)
        fig.add_vrect(x0=qd - pd.Timedelta(days=window_days), x1=qd + pd.Timedelta(days=window_days),
                      fillcolor="#fbbf24", opacity=0.10, line_width=0)
        fig.add_vline(x=qd, line_dash="dot", line_color="#f59e0b", line_width=1.5,
                      annotation_text="query date", annotation_position="top")

    fig.update_layout(
        title="Matching complaints over time — red dots are the cluster",
        xaxis_title="Date filed", yaxis_title="Similarity score", height=400,
        margin=dict(t=50, b=20, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white", yaxis=dict(gridcolor="#f1f5f9"), xaxis=dict(gridcolor="#f1f5f9"),
    )
    return fig


# ── pages ─────────────────────────────────────────────────────────────────────

def tab_semantic(df, model, index):
    left, right = st.columns([3, 1])

    with right:
        st.markdown("**Try an example**")
        if st.button("Template filer (FCRA dispute)", use_container_width=True, key="ex1"):
            st.session_state.sem_text = EXAMPLE_TEMPLATE
        if st.button("Personal narrative", use_container_width=True, key="ex2"):
            st.session_state.sem_text = EXAMPLE_PERSONAL

        st.divider()
        st.markdown("**Date of this complaint** *(optional)*")
        use_date = st.checkbox("Add a filing date", value=False, key="sem_use_date")
        query_date = st.date_input("Filing date", value=date(2021, 6, 1),
                                   min_value=date(2019, 1, 1), max_value=date(2022, 12, 31),
                                   disabled=not use_date, label_visibility="collapsed",
                                   key="sem_date") if use_date else None
        window_days = st.slider("Time window (days)", 7, 90, 60, key="sem_window",
                                disabled=not use_date) if use_date else 60

        st.divider()
        st.markdown("**Settings**")
        threshold = st.slider("Similarity threshold", 0.70, 0.99, DEFAULT_THRESHOLD, 0.01, key="sem_thr")
        min_matches = st.slider("Min matches for a finding", 1, 10, 3, key="sem_min")
        k = st.slider("Neighbours to retrieve", 5, 50, DEFAULT_K, key="sem_k")

    with left:
        text = st.text_area("Complaint narrative", key="sem_text", height=200,
                             placeholder="Paste a CFPB consumer complaint narrative here...")

        if st.button("Investigate", type="primary", disabled=not (text or "").strip(), key="sem_btn"):
            with st.spinner("Searching..."):
                result = investigate_case(text, model, index, df,
                                          k=k, threshold=threshold, min_matches=min_matches)

            matches = result["matches"].copy()
            n_q = result["n_qualifying_matches"]

            if query_date is not None:
                qd = pd.Timestamp(query_date)
                half = pd.Timedelta(days=window_days)
                in_win = ((matches["Date received"] >= qd - half) &
                          (matches["Date received"] <= qd + half))
                n_win_q = int(((matches["similarity"] >= threshold) & in_win).sum())

            if result["verdict"] == "finding":
                st.success(f"FINDING — {n_q} near-identical complaints (≥ {threshold}, needed {min_matches}+)")
            else:
                st.info(f"NO STRONG PATTERN — {n_q} qualifying match{'es' if n_q != 1 else ''} (needed {min_matches}+)")

            if query_date is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Matches retrieved", len(matches))
                m2.metric(f"Within ±{window_days}d of query date", int(in_win.sum()))
                m3.metric("Qualifying & within window", n_win_q,
                          delta="temporal cluster" if n_win_q >= min_matches else "no temporal cluster",
                          delta_color="normal" if n_win_q >= min_matches else "inverse")

            st.plotly_chart(semantic_chart(matches, threshold, query_date, window_days),
                            use_container_width=True)

            with st.expander("Show all matches"):
                matches["Date received"] = matches["Date received"].dt.date
                matches["similarity"] = matches["similarity"].round(3)
                matches["snippet"] = matches["Consumer complaint narrative"].str[:150] + "..."
                show_cols = [c for c in ["similarity", "Date received", "Company", "Product", "snippet"]
                             if c in matches.columns]
                st.dataframe(matches[show_cols], use_container_width=True, hide_index=True)


def tab_temporal(df, model, index):
    left, right = st.columns([3, 1])

    with right:
        st.markdown("**Try an example**")
        if st.button("Coordinated credit dispute", use_container_width=True, key="ex3"):
            st.session_state.tmp_text = EXAMPLE_TEMPORAL_NARRATIVE
            st.session_state.tmp_date = EXAMPLE_TEMPORAL_DATE

        st.divider()
        st.markdown("**Query date** *(required)*")
        query_date = st.date_input("Date to investigate", value=date(2021, 6, 1),
                                   min_value=date(2019, 1, 1), max_value=date(2022, 12, 31),
                                   key="tmp_date")
        window_days = st.slider("Window size (±days)", 7, 90, 30, key="tmp_window")

        st.divider()
        st.markdown("**Semantic filter** *(optional)*")
        st.caption("If provided, the test runs only on complaints similar to this narrative.")
        threshold = st.slider("Similarity threshold", 0.70, 0.99, DEFAULT_THRESHOLD, 0.01,
                              key="tmp_thr")

    with left:
        text = st.text_area("Complaint narrative *(leave blank to test all complaints)*",
                             key="tmp_text", height=160,
                             placeholder="Optional — paste a narrative to narrow the test to similar complaints...")

        if st.button("Analyse", type="primary", key="tmp_btn"):
            with st.spinner("Analysing..."):
                has_narrative = bool((text or "").strip())

                if has_narrative:
                    result = investigate_case(text, model, index, df,
                                              k=TEMPORAL_K, threshold=threshold, min_matches=1)
                    pool = result["matches"]
                    pool = pool[pool["similarity"] >= threshold]
                    dates = pool["Date received"]
                    subtitle = f"Population: {len(pool):,} complaints with similarity ≥ {threshold}"
                else:
                    dates = df["Date received"]
                    subtitle = f"Population: all {len(df):,} complaints in the corpus"

            if len(dates) < 10:
                st.warning(f"Only {len(dates)} similar complaints found — not enough for a reliable test. "
                           "Try lowering the similarity threshold or using a more common narrative.")
                return

            stats = temporal_stats(dates, query_date, window_days)
            z = stats["z_score"]

            if z >= 3:
                st.error(f"STRONG SPIKE — Z-score {z:.2f}: this window is in the top "
                         f"{100 - stats['pct_rank']:.1f}% of all windows by complaint volume")
            elif z >= 2:
                st.warning(f"MODERATE SPIKE — Z-score {z:.2f}: this window is in the top "
                           f"{100 - stats['pct_rank']:.1f}% of all windows by complaint volume")
            else:
                st.info(f"NO UNUSUAL CONCENTRATION — Z-score {z:.2f}: complaint volume "
                        f"around this date is within normal range")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Complaints in window", stats["in_window"])
            m2.metric("Expected (baseline avg)", stats["expected"])
            m3.metric("Z-score", z)
            m4.metric("Percentile rank", f"{stats['pct_rank']:.0f}th",
                      help=f"Based on {stats['n_windows']} non-overlapping windows across the corpus")

            st.caption(subtitle)
            st.plotly_chart(
                volume_chart(dates, query_date, window_days,
                             title="Complaint volume over time — red bars are inside the query window"),
                use_container_width=True,
            )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="CFPB Complaint Pattern Finder", layout="wide")

    st.title("CFPB Complaint Pattern Finder")
    st.markdown(
        "Search **207,000 real CFPB complaints** for semantic patterns, temporal spikes, or both."
    )

    with st.expander("How this works — dataset, methodology, and what the patterns mean"):
        st.markdown(HOW_IT_WORKS)

    df, model, index = load_everything()

    tab1, tab2 = st.tabs(["Semantic Search", "Temporal Clustering"])

    with tab1:
        tab_semantic(df, model, index)

    with tab2:
        tab_temporal(df, model, index)

    st.divider()
    st.caption(
        f"Corpus: {len(df):,} CFPB complaints with narratives · 2019–2022 · "
        "Embeddings: all-MiniLM-L6-v2 (384-dim) · "
        "Index: FAISS IndexFlatIP with L2 normalisation (exact cosine search)"
    )


if __name__ == "__main__":
    main()
