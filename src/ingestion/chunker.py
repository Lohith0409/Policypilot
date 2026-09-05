"""
Splits loaded documents into smaller, overlapping chunks.

Why chunk at all: LLMs and embedding models have a limited context window,
and retrieval works better on small, focused pieces of text than on entire
pages — a question about "reimbursement limits" shouldn't retrieve an
entire 10-page PDF just because the answer is in one paragraph of it.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Split documents into chunks, preserving source metadata.

    chunk_size=800 characters (~150-200 tokens) keeps chunks focused enough
    for precise retrieval, but large enough to contain a full policy clause.
    chunk_overlap=120 means the last ~120 characters of one chunk repeat as
    the first ~120 of the next, so a sentence sitting right on a chunk
    boundary still appears whole in at least one chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try splitting on these, in order, before falling back to a hard cut
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} page(s) into {len(chunks)} chunk(s)")
    return chunks