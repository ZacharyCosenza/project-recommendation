from datetime import date
from functools import lru_cache

from .config import EMBED_MODEL_NAME


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_descriptions(descriptions):
    today = date.today().isoformat()
    contexts = [f"{today} {d}" for d in descriptions]
    return _embedder().encode(contexts, normalize_embeddings=True)
