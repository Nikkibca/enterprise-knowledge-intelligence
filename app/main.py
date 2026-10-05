import requests

from fastapi import FastAPI, File, HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.dependencies import (
    get_embedding_service,
    get_reranking_service,
)
from app.db.session import SessionLocal, engine
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.schemas.search import (
    AskRequest,
    AskResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.chunk_embedding_service import ChunkEmbeddingService
from app.services.document_extractor import (
    DOCXDocumentExtractor,
    PDFDocumentExtractor,
    TextDocumentExtractor,
)
from app.services.document_ingestion_service import DocumentIngestionService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


app = FastAPI(
    title="Enterprise Knowledge Intelligence Platform",
    description="Production-oriented GenAI and RAG platform",
    version="0.1.0",
)


DOCUMENT_EXTRACTORS = {
    ".txt": TextDocumentExtractor(),
    ".pdf": PDFDocumentExtractor(),
    ".docx": DOCXDocumentExtractor(),
}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "enterprise-knowledge-intelligence",
    }


@app.get("/health/database")
async def database_health_check():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            value = result.scalar()

        return {
            "database": "connected",
            "result": value,
        }

    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Database is unavailable",
        )


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    filename = file.filename

    extension = filename.lower().rsplit(".", 1)[-1]
    extension = f".{extension}"

    extractor = DOCUMENT_EXTRACTORS.get(extension)

    if extractor is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Supported formats: .txt, .pdf, .docx"
            ),
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    try:
        extracted_blocks = extractor.extract_structured(content)

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Text file must be UTF-8 encoded",
        )

    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to extract text from "
                "the uploaded document"
            ),
        )

    if not extracted_blocks or not any(
        block.text.strip()
        for block in extracted_blocks
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "No text could be extracted "
                "from the uploaded document"
            ),
        )

    db = SessionLocal()

    try:
        repository = DocumentRepository(db)

        existing_document = repository.get_document_by_filename(
            filename
        )

        if existing_document is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"A document named '{filename}' "
                    "already exists."
                ),
            )

        embedding_service = get_embedding_service()

        chunk_embedding_service = ChunkEmbeddingService(
            embedding_service=embedding_service
        )

        ingestion_service = DocumentIngestionService(
            repository=repository,
            chunk_embedding_service=chunk_embedding_service,
        )

        document = ingestion_service.ingest_structured(
            filename=filename,
            blocks=extracted_blocks,
            content_type=file.content_type,
            file_size=len(content),
        )

        return {
            "id": document.id,
            "filename": document.filename,
            "content_type": document.content_type,
            "file_size": document.file_size,
            "status": document.status,
            "created_at": document.created_at,
            "updated_at": document.updated_at,
        }

    except HTTPException:
        raise

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed",
        )

    finally:
        db.close()


@app.get(
    "/documents",
    response_model=DocumentListResponse,
)
async def list_documents():
    db = SessionLocal()

    try:
        repository = DocumentRepository(db)
        documents = repository.list_documents()

        return DocumentListResponse(
            documents=[
                DocumentResponse(
                    id=document.id,
                    filename=document.filename,
                    content_type=document.content_type,
                    file_size=document.file_size,
                    status=document.status,
                    created_at=document.created_at,
                    updated_at=document.updated_at,
                )
                for document in documents
            ]
        )

    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Document service is temporarily unavailable",
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to list documents",
        )

    finally:
        db.close()


@app.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
)
async def delete_document(document_id: int):
    db = SessionLocal()

    try:
        repository = DocumentRepository(db)

        document = repository.get_document(document_id)

        if document is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Document with id {document_id} "
                    "was not found."
                ),
            )

        filename = document.filename

        deleted = repository.delete_document(document_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Document with id {document_id} "
                    "was not found."
                ),
            )

        repository.commit()

        return DocumentDeleteResponse(
            id=document_id,
            message=(
                f"Document '{filename}' "
                "was deleted successfully."
            ),
        )

    except HTTPException:
        raise

    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="Document deletion failed",
        )

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Document deletion failed",
        )

    finally:
        db.close()


@app.post(
    "/search",
    response_model=SearchResponse,
    response_model_exclude_none=True,
)
async def search_documents(request: SearchRequest):
    db = SessionLocal()

    try:
        embedding_service = get_embedding_service()

        query_embedding = embedding_service.embed_text(
            request.query
        )

        repository = DocumentRepository(db)

        results = repository.similarity_search(
            query_embedding=query_embedding,
            top_k=request.top_k,
        )

        search_results = [
            SearchResult(
                document_id=chunk.document_id,
                filename=chunk.document.filename,
                chunk_id=chunk.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                distance=distance,
                metadata=chunk.metadata_,
            )
            for chunk, distance in results
        ]

        return SearchResponse(
            query=request.query,
            results=search_results,
        )

    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Search service is temporarily "
                "unavailable"
            ),
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Search failed",
        )

    finally:
        db.close()


@app.post(
    "/ask",
    response_model=AskResponse,
    response_model_exclude_none=True,
)
async def ask_question(request: AskRequest):
    db = SessionLocal()

    try:
        repository = DocumentRepository(db)

        embedding_service = get_embedding_service()
        reranking_service = get_reranking_service()
        llm_service = LLMService()

        rag_service = RAGService(
            repository=repository,
            embedding_service=embedding_service,
            llm_service=llm_service,
            reranking_service=reranking_service,
        )

        result = rag_service.answer_question(
            question=request.question,
            top_k=request.top_k,
        )

        return AskResponse(**result)

    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge retrieval service "
                "is temporarily unavailable"
            ),
        )

    except requests.exceptions.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Language model service is unavailable",
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Question answering failed",
        )

    finally:
        db.close()
