"""
Wraps our Qdrant vector store as a LangChain retriever.
"""

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from qdrant_client import models

from src.config import (
    HUGGINGFACEHUB_API_TOKEN,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL,
)


def get_retriever(top_k: int = 4, session_id: str | None = None) -> VectorStoreRetriever:
    """Return a retriever that finds the top_k most relevant chunks for a query.

    session_id: if provided, ONLY searches chunks tagged with this exact
    session_id — this is what keeps one user's uploaded document from ever
    bleeding into another user's answers. If None, searches the entire
    collection unfiltered (our original static-corpus behavior).
    """
    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_NAME,
    )

    search_kwargs = {"k": top_k}

    if session_id:
        # Scoped to one uploaded document's session only
        search_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.session_id",
                    match=models.MatchValue(value=session_id),
                )
            ]
        )
    else:
        # Default corpus: explicitly exclude ANY chunk that has a
        # session_id tag, so uploaded documents can never leak into
        # the shared default knowledge base, regardless of how many
        # uploads happened in the past.
        search_kwargs["filter"] = models.Filter(
            must=[
                models.IsEmptyCondition(
                    is_empty=models.PayloadField(key="metadata.session_id")
                )
            ]
        )
    return vector_store.as_retriever(search_kwargs=search_kwargs)