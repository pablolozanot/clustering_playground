import faiss
import numpy as np
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


def report(result, text_col="Consumer complaint narrative", n_show=5):
    print(f"Verdict: {result['verdict'].upper()}")
    print(f"Qualifying matches (>= {result['threshold']}): {result['n_qualifying_matches']} "
          f"(need {result['min_matches']}+ for a finding)")
    print()
    cols = [c for c in ["similarity", "Company", "Date received", "community_label"] if c in result["matches"].columns]
    print(result["matches"][cols].head(n_show).to_string())