from functools import lru_cache

from app.services.embedding_service import EmbeddingService
from app.services.reranking_service import RerankingService


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """
    Return a single shared embedding service per application process.

    The underlying SentenceTransformer model is loaded only once.
    """
    return EmbeddingService()


@lru_cache(maxsize=1)
def get_reranking_service() -> RerankingService:
    """
    Return a single shared reranking service per application process.

    The underlying CrossEncoder model is loaded only once.
    """
    return RerankingService()