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