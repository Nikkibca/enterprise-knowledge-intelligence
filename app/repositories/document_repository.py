from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.models.document import Document, DocumentChunk


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_document(
        self,
        filename: str,
        content_type: str | None = None,
        file_size: int | None = None,
    ) -> Document:
        document = Document(
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            status="uploaded",
        )

        self.db.add(document)
        self.db.flush()

        return document

    def get_document_by_filename(
        self,
        filename: str,
    ) -> Document | None:
        statement = (
            select(Document)
            .where(Document.filename == filename)
            .order_by(Document.id.desc())
            .limit(1)
        )

        return self.db.execute(statement).scalar_one_or_none()

    def get_document(
        self,
        document_id: int,
    ) -> Document | None:
        statement = (
            select(Document)
            .where(Document.id == document_id)
        )

        return self.db.execute(statement).scalar_one_or_none()

    def list_documents(self) -> list[Document]:
        statement = (
            select(Document)
            .order_by(Document.created_at.desc())
        )

        return list(
            self.db.execute(statement).scalars().all()
        )

    def delete_document(
        self,
        document_id: int,
    ) -> bool:
        document = self.get_document(document_id)

        if document is None:
            return False

        self.db.delete(document)
        self.db.flush()

        return True

    def add_chunk(
        self,
        document_id: int,
        chunk_index: int,
        content: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> DocumentChunk:
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            embedding=embedding,
            metadata_=metadata,
        )

        self.db.add(chunk)
        self.db.flush()

        return chunk

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        distance = DocumentChunk.embedding.cosine_distance(
            query_embedding
        )

        statement = (
            select(
                DocumentChunk,
                distance.label("distance"),
            )
            .options(joinedload(DocumentChunk.document))
            .where(DocumentChunk.embedding.is_not(None))
            .order_by(distance)
            .limit(top_k)
        )

        results = self.db.execute(statement).all()

        return [
            (chunk, float(distance_value))
            for chunk, distance_value in results
        ]

    def commit(self) -> None:
        self.db.commit()

    def rollback(self) -> None:
        self.db.rollback()