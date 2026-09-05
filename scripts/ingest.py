"""
CLI entrypoint: run this whenever you add or update policy PDFs.

Usage: python scripts/ingest.py
"""

import sys
sys.path.append(".")

from src.ingestion.loader import load_policy_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_and_store


def main():
    documents = load_policy_documents()
    chunks = chunk_documents(documents)
    embed_and_store(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()