from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingService:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()