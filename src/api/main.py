"""
FastAPI application: exposes PolicyPilot's RAG chain over HTTP.
"""

import uuid
import tempfile
import os

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient, models as qdrant_models

from src.api.schemas import AskRequest, AskResponse, SourceChunk, UploadResponse
from src.chains.rag_chain import answer_question
from src.ingestion.loader import load_policy_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_and_store
from src.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME

app = FastAPI(
    title="PolicyPilot API",
    description="RAG-based assistant for answering questions about policy documents.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def ensure_qdrant_index():
    """Make sure Qdrant has an index on metadata.session_id before we ever
    try to filter by it. Safe to run every startup — if the index already
    exists, Qdrant just errors on that specific call, which we catch and
    ignore.
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION_NAME,
            field_name="metadata.session_id",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
        print("Created payload index for metadata.session_id")
    except Exception as e:
        print(f"Index setup skipped (likely already exists): {e}")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Accept a PDF upload, ingest it, and return a session_id to query it by."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, file.filename)
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        try:
            documents = load_policy_documents(folder_path=tmp_dir)
            chunks = chunk_documents(documents)
            embed_and_store(chunks, session_id=session_id)
        except Exception as e:
            print(f"Error ingesting upload: {e}")
            raise HTTPException(status_code=500, detail="Failed to process the document.")

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        chunks_stored=len(chunks),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Main endpoint: accepts a question, returns a grounded answer + sources."""
    try:
        result = answer_question(request.question, session_id=request.session_id)
    except Exception as e:
        print(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")

    sources = [
        SourceChunk(source=doc.metadata.get("source", "unknown"), page=doc.metadata.get("page", "?"))
        for doc in result["source_docs"]
    ]

    return AskResponse(answer=result["answer"], sources=sources)