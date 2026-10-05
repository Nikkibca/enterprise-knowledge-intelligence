from unittest.mock import Mock

from app.services.rag_service import RAGService


def make_chunk():
    chunk = Mock()

    chunk.id = 1
    chunk.document_id = 1
    chunk.chunk_index = 0

    chunk.content = (
        "Employees receive twenty days of annual leave."
    )

    chunk.metadata_ = {
        "source": "employee_policy.txt",
        "chunk_index": 0,
    }

    chunk.document.filename = "employee_policy.txt"

    return chunk


def test_answer_question_returns_grounded_answer_and_sources():
    repository = Mock()
    embedding_service = Mock()
    llm_service = Mock()

    embedding_service.embed_text.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    chunk = make_chunk()

    repository.similarity_search.return_value = [
        (chunk, 0.25),
    ]

    llm_service.generate_answer.return_value = (
        "Employees receive twenty days of annual leave."
    )

    rag_service = RAGService(
        repository=repository,
        embedding_service=embedding_service,
        llm_service=llm_service,
        similarity_threshold=0.7,
    )

    result = rag_service.answer_question(
        question=(
            "How many days of annual leave "
            "do employees receive?"
        ),
        top_k=5,
    )

    assert result["question"] == (
        "How many days of annual leave "
        "do employees receive?"
    )

    assert result["answer"] == (
        "Employees receive twenty days of annual leave."
    )

    assert len(result["sources"]) == 1

    source = result["sources"][0]

    assert source["document_id"] == 1
    assert source["filename"] == "employee_policy.txt"
    assert source["chunk_id"] == 1
    assert source["chunk_index"] == 0

    assert (
        source["content"]
        == "Employees receive twenty days of annual leave."
    )

    assert source["distance"] == 0.25

    assert source["metadata"] == {
        "source": "employee_policy.txt",
        "chunk_index": 0,
    }

    embedding_service.embed_text.assert_called_once_with(
        "How many days of annual leave do employees receive?"
    )

    repository.similarity_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=10,
    )

    llm_service.generate_answer.assert_called_once()


def test_answer_question_returns_unavailable_when_no_relevant_chunks():
    repository = Mock()
    embedding_service = Mock()
    llm_service = Mock()

    embedding_service.embed_text.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    repository.similarity_search.return_value = [
        (make_chunk(), 0.85),
    ]

    rag_service = RAGService(
        repository=repository,
        embedding_service=embedding_service,
        llm_service=llm_service,
        similarity_threshold=0.7,
    )

    result = rag_service.answer_question(
        question="What is the company's revenue?",
        top_k=5,
    )

    assert result["question"] == (
        "What is the company's revenue?"
    )

    assert result["answer"] == (
        "The information is not available "
        "in the provided documents."
    )

    assert result["sources"] == []

    llm_service.generate_answer.assert_not_called()

    repository.similarity_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=10,
    )


def test_answer_question_accepts_chunk_at_similarity_threshold():
    repository = Mock()
    embedding_service = Mock()
    llm_service = Mock()

    embedding_service.embed_text.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    chunk = make_chunk()

    repository.similarity_search.return_value = [
        (chunk, 0.7),
    ]

    llm_service.generate_answer.return_value = (
        "Employees receive twenty days of annual leave."
    )

    rag_service = RAGService(
        repository=repository,
        embedding_service=embedding_service,
        llm_service=llm_service,
        similarity_threshold=0.7,
    )

    result = rag_service.answer_question(
        question=(
            "How many days of annual leave "
            "do employees receive?"
        ),
        top_k=5,
    )

    assert len(result["sources"]) == 1

    assert result["sources"][0]["distance"] == 0.7

    llm_service.generate_answer.assert_called_once()


def test_answer_question_respects_top_k():
    repository = Mock()
    embedding_service = Mock()
    llm_service = Mock()

    embedding_service.embed_text.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    chunks = []

    for index in range(5):
        chunk = Mock()

        chunk.id = index + 1
        chunk.document_id = 1
        chunk.chunk_index = index

        chunk.content = f"Relevant content {index}"

        chunk.metadata_ = {
            "source": "employee_policy.txt",
            "chunk_index": index,
        }

        chunk.document.filename = "employee_policy.txt"

        chunks.append(chunk)

    repository.similarity_search.return_value = [
        (chunks[0], 0.20),
        (chunks[1], 0.25),
        (chunks[2], 0.30),
        (chunks[3], 0.35),
        (chunks[4], 0.40),
    ]

    llm_service.generate_answer.return_value = (
        "Employees receive twenty days of annual leave."
    )

    rag_service = RAGService(
        repository=repository,
        embedding_service=embedding_service,
        llm_service=llm_service,
        similarity_threshold=0.7,
    )

    result = rag_service.answer_question(
        question="What is the employee policy?",
        top_k=3,
    )

    assert len(result["sources"]) == 3

    assert result["sources"][0]["chunk_index"] == 0
    assert result["sources"][1]["chunk_index"] == 1
    assert result["sources"][2]["chunk_index"] == 2

    # The RAG pipeline retrieves a larger candidate set first.
    repository.similarity_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=10,
    )


def test_rag_retrieves_more_candidates_than_final_top_k():
    repository = Mock()
    embedding_service = Mock()
    llm_service = Mock()

    embedding_service.embed_text.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    chunks = []

    for index in range(5):
        chunk = Mock()

        chunk.id = index + 1
        chunk.document_id = 1
        chunk.chunk_index = index

        chunk.content = f"Relevant content {index}"

        chunk.metadata_ = {
            "source": "policy.txt",
            "chunk_index": index,
        }

        chunk.document.filename = "policy.txt"

        chunks.append(chunk)

    repository.similarity_search.return_value = [
        (chunks[0], 0.20),
        (chunks[1], 0.25),
        (chunks[2], 0.30),
        (chunks[3], 0.35),
        (chunks[4], 0.40),
    ]

    llm_service.generate_answer.return_value = (
        "Employees receive twenty days of annual leave."
    )

    rag_service = RAGService(
        repository=repository,
        embedding_service=embedding_service,
        llm_service=llm_service,
        similarity_threshold=0.7,
        retrieval_top_k=5,
    )

    result = rag_service.answer_question(
        question="What is the policy?",
        top_k=2,
    )

    # Five candidates were retrieved.
    repository.similarity_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5,
    )

    # Only the requested two are returned.
    assert len(result["sources"]) == 2

    assert result["sources"][0]["chunk_index"] == 0
    assert result["sources"][1]["chunk_index"] == 1