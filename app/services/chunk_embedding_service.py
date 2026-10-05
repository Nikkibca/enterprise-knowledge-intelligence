from app.services.chunking_service import TextChunker
from app.services.embedding_service import EmbeddingService


class ChunkEmbeddingService:
    def __init__(
        self,
        chunker: TextChunker | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        self.chunker = chunker or TextChunker()
        self.embedding_service = embedding_service or EmbeddingService()

    def process_text(self, text: str) -> list[dict]:
        chunks = self.chunker.split_text(text)

        results = []

        for index, chunk in enumerate(chunks):
            embedding = self.embedding_service.embed_text(chunk)

            results.append(
                {
                    "chunk_index": index,
                    "content": chunk,
                    "embedding": embedding,
                }
            )

        return results