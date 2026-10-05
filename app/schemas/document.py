
from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: int
    filename: str
    content_type: str | None = None
    file_size: int | None = None
    status: str
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DocumentDeleteResponse(BaseModel):
    id: int
    message: str

