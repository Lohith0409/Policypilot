"""
Loads raw PDF files from disk into LangChain Document objects.

Why this is its own file: if we later want to support .docx or .txt
policy files too, we only touch THIS file. Nothing else in the codebase
needs to know or care where documents came from.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_policy_documents(folder_path: str = "data/policies") -> list[Document]:
    """Load every PDF in the given folder as LangChain Documents.

    Each returned Document already has metadata like {"source": ..., "page": ...}
    attached automatically by PyPDFLoader — we don't need to add that ourselves.
    """
    pdf_paths = list(Path(folder_path).glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in '{folder_path}'. "
            f"Add at least one policy PDF before running ingestion."
        )

    all_documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()  # one Document per page
        all_documents.extend(documents)
        print(f"Loaded {len(documents)} page(s) from {pdf_path.name}")

    return all_documents