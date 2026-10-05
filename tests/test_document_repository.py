import pytest

from app.db.session import SessionLocal
from app.repositories.document_repository import DocumentRepository


@pytest.fixture
def db():
    session = SessionLocal()

    try:
        yield session
    finally:
        session.rollback()
        session.close()


def test_create_document_and_chunk(db):
    repository = DocumentRepository(db)

    document = repository.create_document(
        filename="employee_policy.txt",
        content_type="text/plain",
        file_size=1024,
    )

    assert document.id is not None
    assert document.filename == "employee_policy.txt"
    assert document.status == "uploaded"

    chunk = repository.add_chunk(
        document_id=document.id,
        chunk_index=0,
        content="Employees receive twenty days of annual leave.",
        embedding=[0.1] * 384,
        metadata={"source": "employee_policy.txt"},
    )

    assert chunk.id is not None
    assert chunk.document_id == document.id
    assert chunk.chunk_index == 0
    assert chunk.content == "Employees receive twenty days of annual leave."
    assert len(chunk.embedding) == 384

    repository.commit()