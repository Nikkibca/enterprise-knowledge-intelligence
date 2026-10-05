from fastapi.testclient import TestClient

from app.db.session import SessionLocal
from app.main import app
from app.repositories.document_repository import DocumentRepository
from app.services.embedding_service import EmbeddingService


client = TestClient(app)


def test_search_returns_metadata():
    db = SessionLocal()

    try:
        repository = DocumentRepository(db)
        embedding_service = EmbeddingService()

        content = (
            "Employees receive twenty days of annual leave."
        )

        document = repository.create_document(
            filename="search_metadata_test.txt",
            content_type="text/plain",
            file_size=len(content.encode("utf-8")),
        )

        embedding = embedding_service.embed_text(content)

        repository.add_chunk(
            document_id=document.id,
            chunk_index=0,
            content=content,
            embedding=embedding,
            metadata={
                "source": "search_metadata_test.txt",
                "chunk_index": 0,
            },
        )

        repository.commit()

    finally:
        db.close()

    response = client.post(
        "/search",
        json={
            "query": "How many days of annual leave do employees receive?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == (
        "How many days of annual leave do employees receive?"
    )

    assert len(data["results"]) > 0

    best_result = data["results"][0]

    assert best_result["filename"] == "search_metadata_test.txt"
    assert best_result["metadata"] is not None
    assert (
        best_result["metadata"]["source"]
        == "search_metadata_test.txt"
    )
    assert (
        best_result["metadata"]["chunk_index"]
        == best_result["chunk_index"]
    )