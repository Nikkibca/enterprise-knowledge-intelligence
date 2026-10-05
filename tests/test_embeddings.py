from app.services.embedding_service import EmbeddingService


def test_embedding_generation():
    service = EmbeddingService()

    text = "Employees receive twenty days of annual leave."

    embedding = service.embed_text(text)

    assert len(embedding) == 384
    assert all(isinstance(value, float) for value in embedding)