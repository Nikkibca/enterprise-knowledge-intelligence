from app.db.session import SessionLocal
from app.repositories.document_repository import DocumentRepository
from app.services.chunk_embedding_service import ChunkEmbeddingService
from app.services.document_ingestion_service import DocumentIngestionService


def test_ingest_text():
    db = SessionLocal()

    try:
        repository = DocumentRepository(db)
        chunk_embedding_service = ChunkEmbeddingService()

        service = DocumentIngestionService(
            repository=repository,
            chunk_embedding_service=chunk_embedding_service,
        )

        document = service.ingest_text(
            filename="employee_policy.txt",
            text=(
                "Employees receive twenty days of annual leave. "
                "Employees can request additional leave according to company policy."
            ),
        )

        assert document.id is not None
        assert document.filename == "employee_policy.txt"

        db.refresh(document)

        assert len(document.chunks) > 0

        for chunk in document.chunks:
            assert len(chunk.embedding) == 384

            assert chunk.metadata_ is not None
            assert chunk.metadata_["source"] == "employee_policy.txt"
            assert chunk.metadata_["chunk_index"] == chunk.chunk_index

    finally:
        db.rollback()
        db.close()