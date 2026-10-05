from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    source: str
    chunk_index: int
    page: int | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    document_id: int
    filename: str
    chunk_id: int
    chunk_index: int
    content: str
    distance: float
    metadata: SourceMetadata | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SearchResult]
