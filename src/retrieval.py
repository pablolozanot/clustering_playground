import faiss
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def get_neighbors(
    embeddings,
    query_index,
    k=10
):
    similarities = cosine_similarity(
        embeddings[query_index].reshape(1, -1),
        embeddings
    )

    indices = similarities[0].argsort()[-k:][::-1]

    return indices, similarities[0]


def build_faiss_index(embeddings):
    normalized = np.ascontiguousarray(embeddings, dtype="float32").copy()
    faiss.normalize_L2(normalized)
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index, normalized


def investigate_case(
    query_text,
    model,
    index,
    df,
    k=10,
    threshold=0.85,
    min_matches=3,
    self_match_threshold=0.999,
):
    query_vec = model.encode([query_text]).astype("float32")
    faiss.normalize_L2(query_vec)

    sims, idxs = index.search(query_vec, k + 1)
    sims, idxs = sims[0], idxs[0]

    candidates = [
        (int(idx), float(sim))
        for idx, sim in zip(idxs, sims)
        if idx != -1 and sim < self_match_threshold
    ][:k]

    qualifying = [(idx, sim) for idx, sim in candidates if sim >= threshold]
    verdict = "finding" if len(qualifying) >= min_matches else "no strong pattern found"

    matches = df.iloc[[idx for idx, _ in candidates]].copy()
    matches["similarity"] = [sim for _, sim in candidates]

    return {
        "verdict": verdict,
        "n_qualifying_matches": len(qualifying),
        "threshold": threshold,
        "min_matches": min_matches,
        "matches": matches,
    }


def build_joint_clusters(df, embeddings_norm, index, text_threshold=0.85, time_window_days=60, k=50, batch=5000):
    """Connected-component clusters where each edge requires both text sim >= threshold AND date diff <= window."""
    dates_arr = df["Date received"].values.astype("datetime64[D]")
    n = len(df)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for batch_start in range(0, n, batch):
        batch_end = min(batch_start + batch, n)
        sims, idxs = index.search(embeddings_norm[batch_start:batch_end], k)
        batch_dates = dates_arr[batch_start:batch_end]
        for local_i in range(batch_end - batch_start):
            global_i = batch_start + local_i
            di = batch_dates[local_i]
            for sim, j in zip(sims[local_i], idxs[local_i]):
                j = int(j)
                if j == -1 or j == global_i or sim < text_threshold:
                    continue
                if abs((di - dates_arr[j]).astype(int)) <= time_window_days:
                    G.add_edge(global_i, j, weight=float(sim))

    components = [c for c in nx.connected_components(G) if len(c) > 1]
    return sorted(components, key=len, reverse=True)


def score_candidates(components, df, embeddings, index, embeddings_norm, text_threshold=0.85,
                     min_size=8, coherence_threshold=0.85, concentration_k=2000):
    """Score clusters by candidate_score (company diversity / date spread) and concentration_ratio
    (rarity of the cluster's text vs the full corpus). Drops incoherent bridging blobs."""
    rng = np.random.default_rng(42)
    rows = []

    for ci, cl in enumerate(components):
        cl = list(cl)
        if len(cl) < min_size:
            continue

        n_sample = min(len(cl), 50)
        sample_idx = rng.choice(cl, size=n_sample, replace=False) if len(cl) > n_sample else np.array(cl)
        sim_matrix = cosine_similarity(embeddings[sample_idx])
        np.fill_diagonal(sim_matrix, np.nan)
        mean_sim = float(np.nanmean(sim_matrix)) if n_sample > 1 else 1.0
        if mean_sim < coherence_threshold:
            continue

        sub = df.iloc[cl]
        n_companies = int(sub["Company"].nunique())
        date_range = int((sub["Date received"].max() - sub["Date received"].min()).days)

        rep_idx = cl[0]
        sims_global, _ = index.search(embeddings_norm[rep_idx:rep_idx + 1], concentration_k)
        global_pop = int((sims_global[0] >= text_threshold).sum())

        rows.append({
            "cluster_id": ci,
            "size": len(cl),
            "date_range_days": date_range,
            "n_companies": n_companies,
            "candidate_score": n_companies / (date_range + 1),
            "global_population": global_pop,
            "concentration_ratio": len(cl) / global_pop if global_pop > 0 else 0.0,
        })

    return pd.DataFrame(rows).sort_values("candidate_score", ascending=False).reset_index(drop=True)


def report(result, text_col="Consumer complaint narrative", n_show=5):
    print(f"Verdict: {result['verdict'].upper()}")
    print(f"Qualifying matches (>= {result['threshold']}): {result['n_qualifying_matches']} "
          f"(need {result['min_matches']}+ for a finding)")
    print()
    cols = [c for c in ["similarity", "Company", "Date received", "community_label"] if c in result["matches"].columns]
    print(result["matches"][cols].head(n_show).to_string())