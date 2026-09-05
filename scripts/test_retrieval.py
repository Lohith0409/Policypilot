"""
Manual sanity check: does retrieval actually return relevant chunks?

Usage: python scripts/test_retrieval.py "your question here"
"""

import sys
sys.path.append(".")

from src.retrieval.retriever import get_retriever


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "What is covered under this policy?"

    retriever = get_retriever()
    results = retriever.invoke(query)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} chunk(s):\n")

    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"--- Chunk {i} (source: {source}, page: {page}) ---")
        print(doc.page_content[:300])  # first 300 chars, just to eyeball it
        print()


if __name__ == "__main__":
    main()