from app.db.session import SessionLocal
from app.repositories.document_repository import DocumentRepository
from app.services.embedding_service import EmbeddingService


def test_similarity_search():
    db = SessionLocal()

    try:
        repository = DocumentRepository(db)
        embedding_service = EmbeddingService()

        test_chunks = [
            "Employees receive twenty days of annual leave.",
            "Employees can enroll in the company health insurance plan.",
            "The company provides reimbursement for approved travel expenses.",
        ]

        document = repository.create_document(
            filename="semantic_search_test.txt",
            content_type="text/plain",
            file_size=None,
        )

        for index, content in enumerate(test_chunks):
            embedding = embedding_service.embed_text(content)

            repository.add_chunk(
                document_id=document.id,
                chunk_index=index,
                content=content,
                embedding=embedding,
                metadata={
                    "source": "semantic_search_test.txt",
                    "chunk_index": index,
                },
            )

        query = "How many days of annual leave do employees receive?"
        query_embedding = embedding_service.embed_text(query)

        results = repository.similarity_search(
            query_embedding=query_embedding,
            top_k=3,
        )

        assert len(results) == 3

        best_chunk, best_distance = results[0]

        assert (
            best_chunk.content
            == "Employees receive twenty days of annual leave."
        )

        assert best_distance >= 0

        assert best_chunk.metadata_ is not None
        assert best_chunk.metadata_["source"] == "semantic_search_test.txt"
        assert best_chunk.metadata_["chunk_index"] == best_chunk.chunk_index

    finally:
        db.rollback()
        db.close()