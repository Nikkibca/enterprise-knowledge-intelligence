from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_ask_returns_answer_with_source_metadata():
    mock_chunk = Mock()

    mock_chunk.id = 1
    mock_chunk.document_id = 1
    mock_chunk.chunk_index = 0
    mock_chunk.content = (
        "Employees receive twenty days of annual leave."
    )

    mock_chunk.metadata_ = {
        "source": "employee_policy.txt",
        "chunk_index": 0,
    }

    mock_chunk.document.filename = "employee_policy.txt"

    repository = Mock()

    repository.similarity_search.return_value = [
        (mock_chunk, 0.25),
    ]

    embedding_service = Mock()

    embedding_service.embed_text.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    reranking_service = Mock()

    reranking_service.rerank.return_value = [
        {
            "chunk": mock_chunk,
            "distance": 0.25,
            "content": mock_chunk.content,
            "document_id": 1,
            "filename": "employee_policy.txt",
            "chunk_id": 1,
            "chunk_index": 0,
            "metadata": mock_chunk.metadata_,
            "rerank_score": 8.0,
        }
    ]

    llm_service = Mock()

    llm_service.generate_answer.return_value = (
        "Employees receive twenty days of annual leave."
    )

    with (
        patch(
            "app.main.DocumentRepository",
            return_value=repository,
        ),
        patch(
            "app.main.get_embedding_service",
            return_value=embedding_service,
        ),
        patch(
            "app.main.get_reranking_service",
            return_value=reranking_service,
        ),
        patch(
            "app.main.LLMService",
            return_value=llm_service,
        ),
    ):
        response = client.post(
            "/ask",
            json={
                "question": "How many annual leave days do employees receive?",
                "top_k": 3,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == (
        "How many annual leave days do employees receive?"
    )

    assert data["answer"] == (
        "Employees receive twenty days of annual leave."
    )

    assert len(data["sources"]) == 1

    source = data["sources"][0]

    assert source["document_id"] == 1
    assert source["filename"] == "employee_policy.txt"
    assert source["chunk_id"] == 1
    assert source["chunk_index"] == 0
    assert source["content"] == (
        "Employees receive twenty days of annual leave."
    )
    assert source["distance"] == 0.25

    assert source["metadata"] == {
        "source": "employee_policy.txt",
        "chunk_index": 0,
    }

    repository.similarity_search.assert_called_once()

    embedding_service.embed_text.assert_called_once_with(
        "How many annual leave days do employees receive?"
    )

    reranking_service.rerank.assert_called_once()

    llm_service.generate_answer.assert_called_once()