"""
Pydantic schemas define the exact shape of API requests and responses.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    session_id: str | None = Field(
        default=None,
        description="If provided, scopes retrieval to only this session's uploaded document.",
    )


class SourceChunk(BaseModel):
    source: str
    page: int | str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    chunks_stored: int