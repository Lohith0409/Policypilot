"""
Turns text chunks into vector embeddings and stores them in Qdrant.

Why this is separate from chunking: embedding is a network call (to
HuggingFace's API) and a storage operation (to Qdrant) — very different
failure modes (network timeouts, auth errors) from pure text splitting.
"""

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from src.config import (
    HUGGINGFACEHUB_API_TOKEN,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL,
)


def get_embeddings_client() -> HuggingFaceEndpointEmbeddings:
    """Create a client that turns text into vectors via HuggingFace's hosted API."""
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )


def embed_and_store(chunks: list[Document], session_id: str | None = None) -> None:
    """Embed all chunks and upsert them into our Qdrant collection.

    session_id: if provided, tags every chunk's metadata with this ID.
    This is the ONLY change needed to support multi-tenancy — a chunk
    tagged with session_id="abc123" will only ever be retrieved when
    someone asks a question scoped to session "abc123". If session_id
    is None (our original static ingest.py script), chunks are stored
    untagged, exactly as before — fully backward compatible.
    """
    if session_id:
        for chunk in chunks:
            chunk.metadata["session_id"] = session_id

    embeddings = get_embeddings_client()

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_NAME,
    )
    print(
        f"Stored {len(chunks)} chunk(s) in Qdrant "
        f"(session_id={session_id or 'none — static corpus'})"
    )