from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_model(model_name=DEFAULT_MODEL):
    return SentenceTransformer(model_name)


def generate_embeddings(texts, model, batch_size=32, show_progress_bar=True):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
