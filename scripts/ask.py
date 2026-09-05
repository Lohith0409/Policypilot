"""
Manual end-to-end test of the full RAG chain.

Usage: python scripts/ask.py "your question here"
"""

import sys
sys.path.append(".")

from src.chains.rag_chain import answer_question


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "What is this policy about?"

    result = answer_question(question)

    print(f"\nQuestion: {question}\n")
    print(f"Answer:\n{result['answer']}\n")
    print(f"--- Based on {len(result['source_docs'])} source chunk(s) ---")
    for doc in result["source_docs"]:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"  - {source}, page {page}")


if __name__ == "__main__":
    main()