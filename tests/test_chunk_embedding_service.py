from app.services.chunk_embedding_service import ChunkEmbeddingService


def test_text_is_chunked_and_embedded():
    service = ChunkEmbeddingService()

    text = (
        "Employees receive twenty days of annual leave. "
        "Employees can request additional leave according to company policy."
    )

    results = service.process_text(text)

    assert len(results) > 0

    for index, result in enumerate(results):
        assert result["chunk_index"] == index
        assert isinstance(result["content"], str)
        assert result["content"]
        assert len(result["embedding"]) == 384
        assert all(isinstance(value, float) for value in result["embedding"])