from app.repositories.document_repository import DocumentRepository
from app.services.chunk_embedding_service import ChunkEmbeddingService
from app.services.document_extractor import ExtractedContent


class DocumentIngestionService:
    def __init__(
        self,
        repository: DocumentRepository,
        chunk_embedding_service: ChunkEmbeddingService,
    ):
        self.repository = repository
        self.chunk_embedding_service = chunk_embedding_service

    def ingest_text(
        self,
        filename: str,
        text: str,
        content_type: str | None = "text/plain",
        file_size: int | None = None,
    ):
        return self._ingest_blocks(
            filename=filename,
            blocks=[
                ExtractedContent(
                    text=text,
                    metadata={},
                )
            ],
            content_type=content_type,
            file_size=file_size,
        )

    def ingest_structured(
        self,
        filename: str,
        blocks: list[ExtractedContent],
        content_type: str | None = None,
        file_size: int | None = None,
    ):
        return self._ingest_blocks(
            filename=filename,
            blocks=blocks,
            content_type=content_type,
            file_size=file_size,
        )

    def _ingest_blocks(
        self,
        filename: str,
        blocks: list[ExtractedContent],
        content_type: str | None = None,
        file_size: int | None = None,
    ):
        try:
            document = self.repository.create_document(
                filename=filename,
                content_type=content_type,
                file_size=file_size,
            )

            global_chunk_index = 0

            for block in blocks:
                processed_chunks = (
                    self.chunk_embedding_service.process_text(
                        block.text
                    )
                )

                for chunk in processed_chunks:
                    metadata = {
                        "source": filename,
                        "chunk_index": global_chunk_index,
                        **block.metadata,
                    }

                    self.repository.add_chunk(
                        document_id=document.id,
                        chunk_index=global_chunk_index,
                        content=chunk["content"],
                        embedding=chunk["embedding"],
                        metadata=metadata,
                    )

                    global_chunk_index += 1

            self.repository.commit()

            return document

        except Exception:
            self.repository.rollback()
            raise
