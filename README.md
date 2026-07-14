# clustering_playground

I got curious about what happens when you treat consumer complaints like text data and let the math find the patterns.

The dataset is the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) — about 15 million complaints filed against banks, credit bureaus, and debt collectors, all public. I took a slice of 207,000 complaints from 2019–2022 that have written narratives (most don't) and embedded them with a sentence transformer. Then built a search and clustering layer on top.

The live app has two modes:

**Semantic search** — paste any complaint text, it finds near-identical complaints in the 207k corpus using FAISS cosine similarity. A threshold of 0.85 turns out to be a natural separator between "same template" and "similar topic." When 3+ complaints clear that bar, it's almost always a legal script being reused across many filers, not a coincidence.

**Temporal spike detection** — pick a date and window size; the tool tests whether complaint volume around that date is statistically unusual (Z-score against a rolling baseline of same-size windows). You can layer in a narrative to narrow the test to a specific complaint type, so the question becomes "is *this kind* of complaint spiking right now?" rather than complaints in general.

---

What I found while exploring the data — three distinct filing patterns, and they look quite different from each other:

1. **Template filers** — legal-aid scripts (FCRA identity theft disputes, FDCPA debt validation letters) reused verbatim by thousands of people over years. High similarity, dispersed over time. Background noise, basically.

2. **Temporal spikes** — the same template concentrated into a short window, often targeting many companies at once. The kind of thing that looks coordinated.

3. **Multi-recipient single-consumer filings** — one person with one underlying problem (a debt being chased by ten different collectors, or identity theft behind several fraudulent inquiries) firing off near-identical complaints against every company involved within a day or two. Totally legitimate, but structurally distinct and detectable.

The third one surprised me — nobody told the clustering algorithm to look for it.

A lot of the notebook work (09–10) is a documented negative result: pure temporal shape statistics (KS test, IQR ratios, order statistics, background-rate rescaling) don't work at scale because organic template virality is statistically indistinguishable from coordination once you have enough data. Four different formulations of the same idea hit the same wall. Notebook 11 reframes from "classify everything" to "rank candidates for human review," and that actually works.

---

## Stack

Python · [sentence-transformers](https://www.sbert.net/) (all-MiniLM-L6-v2, 384-dim) · FAISS · Streamlit · Plotly · HuggingFace Hub · pandas / numpy / networkx

## Live app

→ [clusteringplayground-plt.streamlit.app](https://clusteringplayground-plt.streamlit.app)

Data and embeddings load from HuggingFace Hub on first boot — expect about a minute on a cold start.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

No other setup needed. The app downloads everything from HuggingFace Hub automatically.

## Notebooks

`notebooks/` has the full analysis trail. The short version:

- **01–06**: initial exploration, community detection (Louvain), finding that CFPB's own product taxonomy doesn't explain cluster structure — legal citation strategy does.
- **07**: near-duplicate detection, threshold validation, confirming 0.85 is the right cosine similarity cutoff.
- **07b**: FAISS intro on toy data (sanity check before trusting it on real data).
- **08**: the investigator tool — FAISS-based nearest-neighbour search with a confidence gate.
- **09**: temporal fraud gate — works at 6k rows, 0% false positive rate.
- **10**: scales to 207k rows, breaks completely. Negative result, well documented.
- **11**: joint text+time candidate finder — edges require both conditions simultaneously, rank by company diversity / date spread, validate with a rarity check. This one works.
